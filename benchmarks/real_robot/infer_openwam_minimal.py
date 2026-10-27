#!/usr/bin/env python3
"""OpenWAM absolute-EEF client built on the original RobotWorkspace infer.py."""

import io
import json
import os
import threading
import time
import uuid

import numpy as np
import requests

API_URL = 'http://210.45.70.21:29928'
API_TOKEN = os.environ.get('OPENWAM_API_TOKEN', '')
TASK = 'push the red cube to the target region'
CHUNK_STEPS = 1
STEP_SECONDS = 1.0
MAX_TRANSLATION = .05  # metres, checked against the actual TCP before each step
T_MODEL_FROM_REAL_BASE = np.eye(4)
T_LINK_FROM_TCP = np.array([
    [ .70710678,  .70710678, 0, 0],
    [-.70710678,  .70710678, 0, 0],
    [0, 0, 1, .1054],
    [0, 0, 0, 1],
])  # Real URDF: link8 -> hand -> hand_tcp.
SCHEMA = 'openwam.maniskill.absolute_eef10.v1'


def action_to_tcp(action):
    action = np.asarray(action, dtype=float)
    if action.shape != (10,) or not np.isfinite(action).all():
        raise ValueError('Expected finite absolute EEF10 action')
    x = action[3:6] / np.linalg.norm(action[3:6])
    y = action[6:9] - x * np.dot(x, action[6:9])
    y /= np.linalg.norm(y)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid predicted rotation')
    tcp = np.eye(4)
    tcp[:3, :3] = np.stack((x, y, np.cross(x, y)), axis=1)
    tcp[:3, 3] = action[:3]
    return tcp


def build_node():
    from infer import Infer
    from std_msgs.msg import Float32
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class OpenWAMInfer(Infer):
        def robot_pose(self):
            msg = self.get_current_state(timeout=5)
            if msg is None:
                raise RuntimeError('No joint feedback')
            named = dict(zip(msg.name, msg.position))
            q = np.asarray([named[name] for name in self.joint_names], dtype=float)
            if not np.isfinite(q).all():
                raise ValueError('Invalid joint feedback')
            return q, self.fk_arm(q)

        def infer(self, task):
            front, wrist = self._get_latest_pair(timeout=2)
            q, link = self.robot_pose()
            tcp = T_MODEL_FROM_REAL_BASE @ link @ T_LINK_FROM_TCP
            state = np.r_[tcp[:3, 3], tcp[:3, 0], tcp[:3, 1], self.current_gripper]
            nonce = uuid.uuid4().hex
            payload = dict(schema=SCHEMA, task_description=task, state=state.tolist(),
                           exp_id='real-openwam', request_id=nonce,
                           gripper_state_source='command_estimate')
            response = requests.post(API_URL.rstrip('/') + '/api/inference',
                headers={'Authorization': 'Bearer ' + API_TOKEN}, files={
                    'main_images': ('front.jpg', io.BytesIO(bytes(front.data)), 'image/jpeg'),
                    'wrist_images': ('wrist.jpg', io.BytesIO(bytes(wrist.data)), 'image/jpeg'),
                    'json': ('data.json', io.BytesIO(json.dumps(payload).encode()), 'application/json'),
                }, timeout=(3, 120))
            response.raise_for_status()
            result = response.json()
            if result['request_id'] != nonce or result['policy'] != 'sidecar' or result['contract']['schema'] != SCHEMA:
                raise RuntimeError('Wrong model response')
            actions = np.asarray(result['actions'], dtype=float)
            if actions.shape != (32, 10) or not np.isfinite(actions).all():
                raise ValueError('Expected finite 32x10 actions')
            print('当前EEF:', state, '第一步目标:', actions[0])
            return actions

        def execute_chunk(self, actions):
            for i, action in enumerate(actions[:CHUNK_STEPS], 1):
                q, link = self.robot_pose()
                current_tcp = link @ T_LINK_FROM_TCP
                target_tcp = np.linalg.inv(T_MODEL_FROM_REAL_BASE) @ action_to_tcp(action)
                displacement = np.linalg.norm(target_tcp[:3, 3] - current_tcp[:3, 3])
                if displacement > MAX_TRANSLATION:
                    raise ValueError(f'Step {i}: EEF displacement {displacement:.3f} m exceeds {MAX_TRANSLATION:.3f} m')
                target_link = target_tcp @ np.linalg.inv(T_LINK_FROM_TCP)
                q_target = self.ik_fun(target_link, q.tolist())
                if q_target is None:
                    raise RuntimeError(f'Step {i}: IK failed')
                q_target = np.asarray(q_target, dtype=float)
                if q_target.shape != (7,) or not np.isfinite(q_target).all():
                    raise ValueError('Invalid IK result')
                if np.max(np.abs(q_target - q)) > 1.0:
                    raise ValueError(f'Step {i}: IK joint jump exceeds the original infer.py limit')
                if np.allclose(q_target, q, atol=1e-6) and displacement > .005:
                    raise RuntimeError(f'Step {i}: IK fell back to the current joints')
                print(f'第{i}步: 位移 {displacement*1000:.1f} mm, 目标TCP {target_tcp[:3, 3]}')
                trajectory = JointTrajectory()
                trajectory.joint_names = self.joint_names
                point = JointTrajectoryPoint()
                point.positions = q_target.tolist()
                point.time_from_start.sec = int(STEP_SECONDS)
                point.time_from_start.nanosec = int((STEP_SECONDS % 1) * 1e9)
                trajectory.points = [point]
                self.pub.publish(trajectory)
                time.sleep(STEP_SECONDS + .2)
                _, reached_link = self.robot_pose()  # Joint feedback only; no camera read during execution.
                error = np.linalg.norm((reached_link @ T_LINK_FROM_TCP)[:3, 3] - target_tcp[:3, 3])
                print(f'第{i}步反馈位置误差: {error*1000:.1f} mm')
                if error > .01:
                    raise RuntimeError('Robot did not reach the commanded TCP; stop this chunk')
                gripper = float(action[9] >= .5)
                if gripper != self.current_gripper:
                    command = Float32()
                    command.data = gripper
                    self.gripper_pub.publish(command)
                    self.current_gripper = gripper

    return OpenWAMInfer()


def main():
    if not API_TOKEN:
        raise ValueError('Set OPENWAM_API_TOKEN or use your existing private token line')
    if not 1 <= CHUNK_STEPS <= 32 or STEP_SECONDS <= 0 or MAX_TRANSLATION <= 0:
        raise ValueError('Invalid step settings')
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init()
    node = build_node()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        task = input(f'任务描述，回车使用 {TASK}: ').strip() or TASK
        grip = input('当前夹爪：1=开，0=关: ').strip()
        if grip not in ('0', '1'):
            raise ValueError('Enter 0 or 1 for the current gripper')
        node.current_gripper = float(grip)
        rounds = int(input('推理轮数，回车默认1: ').strip() or '1')
        if rounds < 1:
            raise ValueError('Rounds must be positive')
        if input('准备好后输入 EXECUTE: ').strip() != 'EXECUTE':
            return
        for i in range(rounds):
            if i:
                input('回车采集新图并继续下一轮: ')
            node.execute_chunk(node.infer(task))
            if input('复位到原 infer.py 的默认关节角？输入1复位，回车跳过: ').strip() == '1':
                node.reset()  # Inherited: the user's seven angles (degrees), duration=2.0 s.
                node.current_gripper = 1.0
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
