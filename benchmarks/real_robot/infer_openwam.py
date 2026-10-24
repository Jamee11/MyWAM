#!/usr/bin/env python3
"""Put beside the EXISTING infer.py/convert.py; run python3 infer_openwam.py.

Reuses the original Infer's ROS topics, FK, IK client and publishers. Changes
the HTTP/state/action adapter; never uses its camera-delta apply() or reset().
"""

import io
import json
import os
import threading
import time
import uuid

import numpy as np
import requests

# All user settings are here; no extra config, geometry or token file needed.
API_URL = "http://210.45.70.21:29928"
API_TOKEN = os.environ.get("OPENWAM_API_TOKEN", "")
TASK = "push the red cube to the target region"
EXECUTE = False  # First inspect predictions; no automatic reset.
CALIBRATION_CONFIRMED = False  # Do not set True just to bypass the check.
T_MODEL_FROM_REAL_BASE = np.eye(4)  # Placeholder: model base <- real base, metres.
T_LINK_FROM_TCP = np.eye(4)  # Placeholder: real link8 <- policy TCP.
WORKSPACE_MIN = None  # Measured real-base TCP xyz bounds, metres.
WORKSPACE_MAX = None
EXECUTE_STEPS = 1
STEP_SECONDS = 0.2
MAX_TRANSLATION = 0.02
MAX_ROTATION = 0.15
MAX_JOINT_STEP = 0.06
MAX_JOINT_SPEED = 0.3
MAX_OBSERVATION_AGE = 5.0
SCHEMA = "openwam.maniskill.absolute_eef10.v1"


def check_transform(t):
    t = np.asarray(t, dtype=float)
    if (
        t.shape != (4, 4)
        or not np.isfinite(t).all()
        or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("Invalid rigid transform")
    return t


def action_to_tcp(action):
    a = np.asarray(action, dtype=float)
    if a.shape != (10,) or not np.isfinite(a).all() or not -0.25 <= a[9] <= 1.25:
        raise ValueError("Expected finite physical EEF10, not7D delta")
    x = a[3:6]
    if np.linalg.norm(x) < 1e-6:
        raise ValueError("Degenerate rotation")
    x = x / np.linalg.norm(x)
    y = a[6:9] - x * np.dot(x, a[6:9])
    if np.linalg.norm(y) < 1e-6:
        raise ValueError("Degenerate rotation")
    y /= np.linalg.norm(y)
    t = np.eye(4)
    t[:3, :3] = np.stack((x, y, np.cross(x, y)), axis=1)
    t[:3, 3] = a[:3]
    return t


def current_state(real_link, gripper):
    if not np.isfinite(gripper) or not 0 <= gripper <= 1:
        raise ValueError("Invalid gripper state")
    tcp = check_transform(T_MODEL_FROM_REAL_BASE) @ check_transform(real_link) @ check_transform(T_LINK_FROM_TCP)
    return np.r_[tcp[:3, 3], tcp[:3, 0], tcp[:3, 1], gripper]


def target_link(action):
    # This replaces delta integration in the original infer.py.
    tcp = np.linalg.inv(check_transform(T_MODEL_FROM_REAL_BASE)) @ action_to_tcp(action)
    return tcp @ np.linalg.inv(check_transform(T_LINK_FROM_TCP)), tcp


def check_pose(previous, target):
    lo, hi = np.asarray(WORKSPACE_MIN, dtype=float), np.asarray(WORKSPACE_MAX, dtype=float)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
        raise ValueError("Fill measured WORKSPACE_MIN/MAX first")
    if np.any(target[:3, 3] < lo) or np.any(target[:3, 3] > hi):
        raise ValueError("Outside real TCP workspace")
    angle = np.arccos(np.clip((np.trace(previous[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1))
    if np.linalg.norm(target[:3, 3] - previous[:3, 3]) > MAX_TRANSLATION or angle > MAX_ROTATION:
        raise ValueError("Absolute target too far from current TCP; do not execute")


def build_node():
    from infer import Infer  # Your existing RobotWorkspace infer.py.
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    class OpenWAMInfer(Infer):
        def __init__(self):
            super().__init__()
            self._joint = None
            self._joint_time = self._main_time = self._wrist_time = 0.0
            self.joint_sub = self.create_subscription(
                JointState, "/mk1000/joint_states", self._joint_cb, qos_profile_sensor_data
            )
            if self.fk_arm.robot_model.base_link.name != self.base_frame:
                raise ValueError("FK root must match IK base frame")

        def _main_cb(self, msg):
            with self._lock:
                self._main_msg, self._main_time = msg, time.monotonic()

        def _wrist_cb(self, msg):
            with self._lock:
                self._wrist_msg, self._wrist_time = msg, time.monotonic()

        def _joint_cb(self, msg):
            with self._lock:
                self._joint, self._joint_time = msg, time.monotonic()

        def observe(self):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with self._lock:
                    joint, front, wrist = self._joint, self._main_msg, self._wrist_msg
                    times = (self._joint_time, self._main_time, self._wrist_time)
                if all(x is not None for x in (joint, front, wrist)) and max(time.monotonic() - t for t in times) < 0.5:
                    break
                time.sleep(0.02)
            else:
                raise RuntimeError("Missing/stale joint or camera feedback")
            stamps = [m.header.stamp.sec + m.header.stamp.nanosec / 1e9 for m in (front, wrist)]
            now = self.get_clock().now().nanoseconds / 1e9
            if (
                min(stamps) <= 0
                or max(now - s for s in stamps) > 0.5
                or min(now - s for s in stamps) < -0.05
                or abs(stamps[0] - stamps[1]) > 0.1
            ):
                raise RuntimeError("Stale/unsynchronized camera timestamps")
            names = dict(zip(joint.name, joint.position))
            q = np.asarray([names[n] for n in self.joint_names], dtype=float)  # ROS radians.
            if not np.isfinite(q).all():
                raise ValueError("Invalid joint feedback")
            link = check_transform(self.fk_arm(q))
            return front, wrist, q, link

        def infer(self, task):
            self.observed_at = time.monotonic()
            front, wrist, self.observed_q, self.observed_link = self.observe()
            state = current_state(self.observed_link, self.current_gripper)
            nonce = uuid.uuid4().hex
            query = dict(
                schema=SCHEMA,
                task_description=task,
                state=state.tolist(),
                exp_id="real-openwam",
                request_id=nonce,
                gripper_state_source="command_estimate",
            )
            response = requests.post(
                API_URL.rstrip("/") + "/api/inference",
                headers={"Authorization": "Bearer " + API_TOKEN},
                files={
                    "main_images": ("front.jpg", io.BytesIO(bytes(front.data)), "image/jpeg"),
                    "wrist_images": ("wrist.jpg", io.BytesIO(bytes(wrist.data)), "image/jpeg"),
                    "json": ("data.json", io.BytesIO(json.dumps(query).encode()), "application/json"),
                },
                timeout=(3, 120),
            )
            response.raise_for_status()
            result = response.json()
            if (
                result["request_id"] != nonce
                or result["exp_id"] != "real-openwam"
                or result["policy"] != "sidecar"
                or result["checkpoint_sha256"] != self.checkpoint_sha
                or result["contract"] != self.contract
            ):
                raise RuntimeError("Wrong/stale model response")
            actions = np.asarray(result["actions"], dtype=float)
            if actions.shape != (32, 10):
                raise ValueError("Expected32x10 absolute EEF")
            for action in actions:
                action_to_tcp(action)
            print("第一步绝对EEF:", actions[0], "推理耗时(ms):", result["latency_ms"])
            return actions

        def ik_fun(self, target, q_seed):
            import convert
            from moveit_msgs.srv import GetPositionIK

            req = GetPositionIK.Request()
            ik = req.ik_request
            ik.group_name = self.group_name
            ik.ik_link_name = self.ik_link_name
            ik.avoid_collisions = True
            ik.timeout.sec = 1
            ik.pose_stamped.header.frame_id = self.base_frame
            ik.pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose = convert.rotmat2posequat(target)
            ik.pose_stamped.pose.position.x, ik.pose_stamped.pose.position.y, ik.pose_stamped.pose.position.z = map(
                float, pose[:3]
            )
            (
                ik.pose_stamped.pose.orientation.x,
                ik.pose_stamped.pose.orientation.y,
                ik.pose_stamped.pose.orientation.z,
                ik.pose_stamped.pose.orientation.w,
            ) = map(float, pose[3:])
            ik.robot_state.joint_state.name = self.joint_names
            ik.robot_state.joint_state.position = list(map(float, q_seed))
            event = threading.Event()
            future = self.client.call_async(req)
            future.add_done_callback(lambda _: event.set())
            if not event.wait(2):
                future.cancel()
                raise RuntimeError("IK timeout; no fallback motion")
            result = future.result()
            if result is None or result.error_code.val != 1:
                raise RuntimeError("IK failed; no previous-solution fallback")
            named = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
            q = np.asarray([named[n] for n in self.joint_names])
            self.check_joints(q_seed, q)
            return q

        def check_joints(self, previous, q):
            if q.shape != (7,) or not np.isfinite(q).all():
                raise ValueError("Invalid seven-joint IK result")
            for i, n in enumerate(self.joint_names):
                limit = self.fk_arm.robot_model.joint_map[n].limit
                if not limit.lower <= q[i] <= limit.upper:
                    raise ValueError("URDF joint limit")
            delta = np.abs(q - previous)
            if np.any(delta > MAX_JOINT_STEP) or np.any(delta / STEP_SECONDS > MAX_JOINT_SPEED):
                raise ValueError("Joint jump/speed rejected")

        def execute_absolute(self, actions):
            from std_msgs.msg import Float32
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

            previous_q = self.observed_q
            previous_tcp = self.observed_link @ T_LINK_FROM_TCP
            plan = []
            for a in actions[:EXECUTE_STEPS]:
                link, tcp = target_link(a)
                check_pose(previous_tcp, tcp)
                q = self.ik_fun(link, previous_q)
                plan.append((q, link, tcp, a[9]))
                previous_q, previous_tcp = q, tcp
            # Every selected IK must pass before the first command is sent.
            for q, link, tcp, grip in plan:
                if time.monotonic() - self.observed_at > MAX_OBSERVATION_AGE:
                    raise RuntimeError("Prediction is stale")
                _, _, actual_q, actual_link = self.observe()
                self.check_joints(actual_q, q)
                check_pose(actual_link @ T_LINK_FROM_TCP, tcp)
                msg = JointTrajectory()
                msg.joint_names = self.joint_names
                point = JointTrajectoryPoint()
                point.positions = list(map(float, q))
                point.time_from_start.sec = int(STEP_SECONDS)
                point.time_from_start.nanosec = int((STEP_SECONDS - int(STEP_SECONDS)) * 1e9)
                msg.points = [point]
                self.pub.publish(msg)
                started = time.monotonic()
                while True:
                    time.sleep(0.02)
                    _, _, measured_q, _ = self.observe()
                    if time.monotonic() - started >= STEP_SECONDS and np.max(np.abs(measured_q - q)) <= 0.02:
                        break
                    if time.monotonic() - started > max(3, 2 * STEP_SECONDS):
                        raise RuntimeError("Tracking timeout; stop new commands")
                command = Float32()
                command.data = 1.0 if grip >= 0.5 else 0.0
                self.gripper_pub.publish(command)
                self.current_gripper = command.data

    return OpenWAMInfer()


def main():
    if not API_TOKEN:
        raise ValueError("Use the private ready-to-copy file, or set OPENWAM_API_TOKEN")
    if not 1 <= EXECUTE_STEPS <= 8 or not np.isfinite(STEP_SECONDS) or STEP_SECONDS <= 0:
        raise ValueError("Invalid step settings")
    if any(
        not np.isfinite(x) or x <= 0
        for x in (MAX_TRANSLATION, MAX_ROTATION, MAX_JOINT_STEP, MAX_JOINT_SPEED, MAX_OBSERVATION_AGE)
    ):
        raise ValueError("Invalid safety thresholds")
    check_transform(T_MODEL_FROM_REAL_BASE)
    check_transform(T_LINK_FROM_TCP)
    if EXECUTE:
        if CALIBRATION_CONFIRMED is not True:
            raise ValueError("Confirm real base/TCP calibration before motion")
        lo, hi = np.asarray(WORKSPACE_MIN, dtype=float), np.asarray(WORKSPACE_MAX, dtype=float)
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
            raise ValueError("Fill measured workspace bounds")
    health = requests.get(API_URL.rstrip("/") + "/health", headers={"Authorization": "Bearer " + API_TOKEN}, timeout=10)
    health.raise_for_status()
    health = health.json()
    contract = health["contract"]
    if (
        contract["schema"] != SCHEMA
        or contract["frame"] != "model_robot_base"
        or contract["rotation"] != "first_two_columns"
        or contract["position_unit"] != "metre"
        or contract["gripper"] != "0_closed_1_open"
        or contract["action_dim"] != 10
        or contract["horizon"] != 32
        or health["policy"] != "sidecar"
    ):
        raise ValueError("Wrong model/action contract")
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init()
    node = None
    executor = None
    try:
        node = build_node()
        node.checkpoint_sha = health["checkpoint_sha256"]
        node.contract = contract
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        task = input(f"任务描述，回车使用 {TASK}: ").strip() or TASK
        grip = input("确认当前夹爪：1=开，0=关（此值是估计，不是传感器反馈）: ").strip()
        if grip not in ("0", "1"):
            raise ValueError("Confirm actual initial gripper state with0/1")
        node.current_gripper = float(grip)
        count = int(input("推理几轮，回车默认1: ").strip() or "1")
        if count < 1:
            raise ValueError("Positive round count required")
        print("执行模式" if EXECUTE else "只推理，不下发运动")
        if EXECUTE and input("硬件急停准备好后输入EXECUTE确认: ") != "EXECUTE":
            raise RuntimeError("Not confirmed")
        for i in range(count):
            if i:
                input("回车采集新观测并继续，Ctrl+C退出: ")
            actions = node.infer(task)
            if EXECUTE:
                node.execute_absolute(actions)
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
