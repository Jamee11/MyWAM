#!/usr/bin/env python3
"""Place beside RobotWorkspace's infer.py and convert.py; run on the robot host."""
import io
import json
import os
import signal
import threading
import time
import uuid

import numpy as np
import requests
from scipy.spatial.transform import Rotation

API_URL = os.environ.get('OPENWAM_API_URL', 'http://210.45.70.21:37217')
API_TOKEN = os.environ.get('OPENWAM_API_TOKEN', '')
TASKS = [
    'push the red cube to the target region',
    'pick up the red cube and move it to the visible goal',
    'pick up the red cuboid and move it to the visible goal',
    'pick up the red sphere and move it to the visible goal',
    'stack the red cube on the gray cube',
    'place the red sphere in the bin',
    'pull the red cube to the target region',
    'stand the peg upright on the table',
]
ARM_BASE_FRAME = 'mk1000_fr3_link0'  # Mechanical arm base, NOT mobile base_link.
TCP_FRAME = 'mk1000_fr3_hand_tcp'
T_MODEL_FROM_ARM_BASE = np.eye(4)  # Model robot-base axes identified with FR3 link0.
RESET_EEF = np.array([
    .6119986176490784, -.00874595157802105, .17062480747699738,
    .010432863608002663, .9998855590820312, -.010953102260828018,
    .999905526638031, -.010529833845794201, -.008833232335746288, 1.,
])  # place_sphere train/slot0016/Clean frame0, model-base TCP, metres/column6D.
CHUNK_STEPS = 32
ACTION_DT = .5  # Requested real-robot interval: 0.5s/action; joint limits may slow it further.
MAX_JOINT_SPEED = .5  # rad/s; also used to time the reset.
RESET_SECONDS = 3.
MAX_TRANSLATION = .05  # Per predicted segment, metres; reset uses its own duration.
INITIAL_GRIPPER = 1.  # Command estimate if initial reset is skipped.
SCHEMA = 'openwam.maniskill.absolute_eef10.v1'


def action_to_tcp(action):
    a = np.asarray(action, dtype=float)
    if a.shape != (10,) or not np.isfinite(a).all():
        raise ValueError('Expected finite absolute EEF10')
    x = a[3:6] / np.linalg.norm(a[3:6])
    y = a[6:9] - x * np.dot(x, a[6:9])
    y /= np.linalg.norm(y)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Degenerate rotation columns')
    t = np.eye(4)
    t[:3, :3] = np.stack((x, y, np.cross(x, y)), axis=1)
    t[:3, 3] = a[:3]
    return t


def pose_error(actual, target):
    return (np.linalg.norm(actual[:3, 3] - target[:3, 3]),
            Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).magnitude())


def joint_times(current, targets, minimum_dt=ACTION_DT):
    steps = np.diff(np.vstack((current, targets)), axis=0)
    return np.cumsum(np.maximum(minimum_dt, np.max(np.abs(steps), axis=1) / MAX_JOINT_SPEED))


def build_node():
    from infer import Infer
    from geometry_msgs.msg import PoseStamped
    from moveit_msgs.srv import GetPositionIK
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class OpenWAMInfer(Infer):
        def __init__(self):
            super().__init__()
            self._joint_msg, self._joint_time = None, 0.
            self._joint_ready = threading.Event()
            self.joint_sub = self.create_subscription(JointState, '/mk1000/joint_states', self._joint_cb, 10)
            self.base_frame = ARM_BASE_FRAME  # IK performs any mobile-base transform itself.
            self.current_gripper = INITIAL_GRIPPER
            links = self._fk(np.zeros(7))
            self.link_from_tcp = np.linalg.inv(links[self.ik_link_name]) @ links[TCP_FRAME]
            self.session = requests.Session()
            self.session.trust_env = False
            self.session.headers['Authorization'] = 'Bearer ' + API_TOKEN
            print('EEF/IK frame:', ARM_BASE_FRAME, 'TCP:', TCP_FRAME,
                  'prefix:', CHUNK_STEPS, 'nominal dt:', ACTION_DT)

        def _fk(self, q):
            model = self.fk_arm.robot_model
            poses = model.link_fk(cfg=dict(zip(self.joint_names, q)))
            return {name: poses[model.link_map[name]] for name in
                    (ARM_BASE_FRAME, self.ik_link_name, TCP_FRAME)}

        def arm_link(self, q):
            poses = self._fk(q)
            return np.linalg.inv(poses[ARM_BASE_FRAME]) @ poses[self.ik_link_name]

        def _joint_cb(self, msg):
            self._joint_msg, self._joint_time = msg, time.monotonic()
            self._joint_ready.set()

        def robot_pose(self):
            if not self._joint_ready.wait(5) or time.monotonic() - self._joint_time > 2:
                raise RuntimeError('Missing or stale joint feedback')
            msg = self._joint_msg
            named = dict(zip(msg.name, msg.position))
            q = np.asarray([named[n] for n in self.joint_names], dtype=float)
            if not np.isfinite(q).all():
                raise ValueError('Invalid joint feedback')
            return q, self.arm_link(q)

        def solve_ik(self, target, seed):
            req = GetPositionIK.Request()
            pose = PoseStamped()
            pose.header.frame_id = ARM_BASE_FRAME
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = target[:3, 3]
            quat = Rotation.from_matrix(target[:3, :3]).as_quat()  # ROS xyzw.
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = quat
            req.ik_request.pose_stamped = pose
            req.ik_request.group_name = self.group_name
            req.ik_request.ik_link_name = self.ik_link_name
            req.ik_request.avoid_collisions = True
            req.ik_request.timeout.nanosec = 200000000
            req.ik_request.robot_state.joint_state.name = self.joint_names
            req.ik_request.robot_state.joint_state.position = np.asarray(seed).tolist()
            future = self.client.call_async(req)
            ready = threading.Event()
            future.add_done_callback(lambda _: ready.set())
            if not ready.wait(3):
                future.cancel()
                raise RuntimeError('IK timed out')
            result = future.result()
            if result is None or result.error_code.val != result.error_code.SUCCESS:
                raise RuntimeError('IK failed; this prefix was not sent')
            named = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
            q = np.asarray([named[n] for n in self.joint_names], dtype=float)
            if not np.isfinite(q).all():
                raise ValueError('Invalid IK solution')
            position, angle = pose_error(self.arm_link(q), target)
            if position > .005 or angle > np.deg2rad(3):
                raise RuntimeError(f'IK/FK mismatch: {position*1000:.1f}mm, {np.rad2deg(angle):.1f}deg')
            return q

        def gripper(self, value):
            value = float(value >= .5)
            if value != self.current_gripper:
                msg = Float32()
                msg.data = value
                self.gripper_pub.publish(msg)
                self.current_gripper = value

        def send_motion(self, current, targets, times, grippers=None):
            trajectory = JointTrajectory()
            trajectory.joint_names = self.joint_names
            for q, seconds in zip(np.vstack((current, targets)), np.r_[0., times]):
                point = JointTrajectoryPoint()
                point.positions = np.asarray(q).tolist()
                nanoseconds = round(float(seconds) * 1e9)
                point.time_from_start.sec, point.time_from_start.nanosec = divmod(nanoseconds, 1000000000)
                trajectory.points.append(point)
            start = time.monotonic()
            self.pub.publish(trajectory)
            if grippers is not None:
                for offset, grip in zip(np.r_[0., times[:-1]], grippers):
                    time.sleep(max(0., start + offset - time.monotonic()))
                    self.gripper(grip)
            time.sleep(max(0., start + times[-1] - time.monotonic()) + .15)

        def check_reached(self, target):
            _, link = self.robot_pose()
            position, angle = pose_error(link @ self.link_from_tcp, target)
            print(f'TCP feedback error: {position*1000:.1f}mm / {np.rad2deg(angle):.1f}deg')
            if position > .01 or angle > np.deg2rad(5):
                raise RuntimeError('Robot did not reach the requested TCP pose')

        def reset(self):
            q, _ = self.robot_pose()
            tcp = np.linalg.inv(T_MODEL_FROM_ARM_BASE) @ action_to_tcp(RESET_EEF)
            target = self.solve_ik(tcp @ np.linalg.inv(self.link_from_tcp), q)
            duration = max(RESET_SECONDS, float(joint_times(q, [target])[-1]))
            print('Reset TCP in', ARM_BASE_FRAME, ':', tcp[:3, 3], 'seconds:', round(duration, 2))
            # Always send open on reset, even when the previous command estimate was open.
            self.current_gripper = -1.
            self.gripper(1.)
            self.send_motion(q, [target], np.array([duration]))
            self.check_reached(tcp)

        def infer(self, task):
            front, wrist = self._get_latest_pair(timeout=2)
            _, link = self.robot_pose()
            tcp = T_MODEL_FROM_ARM_BASE @ link @ self.link_from_tcp
            state = np.r_[tcp[:3, 3], tcp[:3, 0], tcp[:3, 1], self.current_gripper]
            nonce = uuid.uuid4().hex
            query = dict(schema=SCHEMA, task_description=task, state=state.tolist(),
                         exp_id='real-openwam', request_id=nonce, gripper_state_source='command_estimate')
            response = self.session.post(API_URL.rstrip('/') + '/api/inference', files={
                'main_images': ('front.jpg', io.BytesIO(bytes(front.data)), 'image/jpeg'),
                'wrist_images': ('wrist.jpg', io.BytesIO(bytes(wrist.data)), 'image/jpeg'),
                'json': ('data.json', json.dumps(query).encode(), 'application/json'),
            }, timeout=(3, 120))
            response.raise_for_status()
            result = response.json()
            if result['request_id'] != nonce or result['policy'] != 'sidecar' or result['contract']['schema'] != SCHEMA:
                raise RuntimeError('Wrong model response')
            actions = np.asarray(result['actions'], dtype=float)
            if actions.shape != (32, 10) or not np.isfinite(actions).all():
                raise ValueError('Expected finite 32x10 actions')
            print('EEF xyz:', state[:3], 'first target:', actions[0, :3])
            return actions

        def execute_chunk(self, actions):
            current, link = self.robot_pose()
            seed, previous, targets, tcp = current, link @ self.link_from_tcp, [], None
            for action in actions[:CHUNK_STEPS]:
                tcp = np.linalg.inv(T_MODEL_FROM_ARM_BASE) @ action_to_tcp(action)
                if np.linalg.norm(tcp[:3, 3] - previous[:3, 3]) > MAX_TRANSLATION:
                    raise ValueError('Predicted TCP segment exceeds 50mm; prefix not sent')
                target = self.solve_ik(tcp @ np.linalg.inv(self.link_from_tcp), seed)
                if np.max(np.abs(target - seed)) > 1.:
                    raise ValueError('IK joint jump exceeds 1rad; prefix not sent')
                targets.append(target)
                seed, previous = target, tcp
            times = joint_times(current, targets)
            print(f'Execute {len(targets)} points in {times[-1]:.2f}s')
            self.send_motion(current, targets, times, actions[:CHUNK_STEPS, 9])
            self.check_reached(tcp)

        def hold(self):
            if self._joint_msg is not None:
                q, _ = self.robot_pose()
                self.send_motion(q, [q], np.array([.2]))

    return OpenWAMInfer()


def main():
    if not API_TOKEN:
        raise ValueError('Set OPENWAM_API_TOKEN')
    if not 1 <= CHUNK_STEPS <= 32 or ACTION_DT <= 0 or MAX_JOINT_SPEED <= 0:
        raise ValueError('Invalid motion settings')
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    def interrupted(*_):
        raise KeyboardInterrupt

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    node = build_node()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        while True:
            reset = input('Reset? 1=yes, 0=no: ').strip() == '1'
            rounds = int(input('How many times? (0=exit): '))
            if rounds <= 0:
                break
            for index, task in enumerate(TASKS, 1):
                print(f'{index}. {task}')
            index = int(input('Task? (1-8): '))
            if not 1 <= index <= len(TASKS):
                raise ValueError('Task must be 1-8')
            if reset:
                node.reset()
            for iteration in range(rounds):
                print(f'Round {iteration+1}/{rounds}: {TASKS[index-1]}')
                node.execute_chunk(node.infer(TASKS[index-1]))
    except KeyboardInterrupt:
        print('Interrupted; sending current-joint hold')
    finally:
        try:
            node.hold()
        finally:
            node.session.close()
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
