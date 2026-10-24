#!/usr/bin/env python3
"""Copy beside openwam_real_contract.py on RobotWorkspace; dry-run unless --execute."""

import argparse
import io
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import requests

try:
    from .openwam_real_contract import CONTRACT, action_chunk, joint_guard, model_state, pose_guard, real_target, rigid
except ImportError:
    from openwam_real_contract import CONTRACT, action_chunk, joint_guard, model_state, pose_guard, real_target, rigid


def validate_config(cfg, execute):
    rigid(cfg["T_model_from_real_base"])
    rigid(cfg["T_link_from_tcp"])
    keys = (
        "max_translation_m",
        "max_rotation_rad",
        "max_joint_step_rad",
        "max_joint_speed_rad_s",
        "step_seconds",
        "max_sensor_age_s",
        "max_pair_skew_s",
        "max_request_age_s",
        "max_state_drift_m",
        "max_state_drift_rad",
        "tracking_tolerance_rad",
    )
    if any(not np.isfinite(cfg[k]) or cfg[k] <= 0 for k in keys):
        raise ValueError("Safety thresholds must be finite and positive")
    if len(cfg["joint_names"]) != 7 or len(set(cfg["joint_names"])) != 7:
        raise ValueError("Need seven unique named joints")
    if cfg["joint_state_unit"] not in ("rad", "deg"):
        raise ValueError("Declare joint_state_unit as rad or deg")
    if execute and cfg.get("calibration_confirmed") is not True:
        raise ValueError("Refusing motion: validate model-base and link-to-TCP calibration first")
    if execute:
        # Check workspace configuration even before ROS starts.
        bounds = np.asarray(cfg["workspace_xyz"], dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
            raise ValueError("Set measured real-base TCP workspace bounds before execution")


def infer_http(session, url, token, front, wrist, state, task, exp_id, source, checkpoint_sha, policy):
    request_id = uuid.uuid4().hex
    query = dict(
        task_description=task,
        state=np.asarray(state).tolist(),
        exp_id=exp_id,
        request_id=request_id,
        schema=CONTRACT["schema"],
        gripper_state_source=source,
    )
    response = session.post(
        url.rstrip("/") + "/api/inference",
        headers={"Authorization": "Bearer " + token},
        files={
            "json": ("data.json", io.BytesIO(json.dumps(query).encode()), "application/json"),
            "main_images": ("front.jpg", io.BytesIO(bytes(front)), "image/jpeg"),
            "wrist_images": ("wrist.jpg", io.BytesIO(bytes(wrist)), "image/jpeg"),
        },
        timeout=(3, 120),
    )
    response.raise_for_status()  # No automatic POST retry or reuse of old action chunks.
    result = response.json()
    if (
        result["contract"] != CONTRACT
        or result["request_id"] != request_id
        or result["exp_id"] != exp_id
        or result["checkpoint_sha256"] != checkpoint_sha
        or result["policy"] != policy
    ):
        raise ValueError("Wrong/stale server, checkpoint, response or action contract")
    return action_chunk(result["actions"]), result


class RobotClient:
    def __init__(self, cfg, args):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, JointState
        from std_msgs.msg import Float32

        self.cfg, self.args = cfg, args
        self.node = rclpy.create_node("openwam_absolute_eef_client")
        self.lock = threading.Lock()
        self.data = {}
        self.estimated_gripper = float(cfg["initial_gripper_open_fraction"])
        if not 0 <= self.estimated_gripper <= 1:
            raise ValueError("Initial estimated gripper must be in[0,1]")
        sys.path.insert(0, str(args.robot_scripts.resolve(strict=True)))
        from convert import FkArm

        self.fk = FkArm(os.path.expandvars(cfg["urdf_path"]))
        if self.fk.robot_model.base_link.name != cfg["base_frame"] or cfg["ik_link"] != "mk1000_fr3_link8":
            raise ValueError("FK root/link must agree with IK; supplied FkArm computes link8")
        self.subscriptions = []
        for key, msgtype, topic in [
            ("front", CompressedImage, cfg["front_topic"]),
            ("wrist", CompressedImage, cfg["wrist_topic"]),
            ("joints", JointState, cfg["joint_topic"]),
        ]:
            self.subscriptions.append(
                self.node.create_subscription(
                    msgtype, topic, lambda msg, key=key: self.receive(key, msg), qos_profile_sensor_data
                )
            )
        if cfg["gripper_feedback_topic"]:
            self.subscriptions.append(
                self.node.create_subscription(
                    Float32,
                    cfg["gripper_feedback_topic"],
                    lambda msg: self.receive("gripper", msg),
                    qos_profile_sensor_data,
                )
            )
        elif cfg.get("allow_estimated_gripper") is not True:
            raise ValueError("Configure measured gripper feedback, or explicitly allow_estimated_gripper")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        if args.execute:
            from moveit_msgs.srv import GetPositionIK
            from trajectory_msgs.msg import JointTrajectory

            self.arm_pub = self.node.create_publisher(JointTrajectory, cfg["trajectory_topic"], 10)
            self.gripper_pub = self.node.create_publisher(Float32, cfg["gripper_command_topic"], 10)
            self.ik = self.node.create_client(GetPositionIK, cfg["ik_service"])
            if not self.ik.wait_for_service(timeout_sec=10):
                raise RuntimeError("IK service unavailable; no motion sent")
            self.limits = []
            for name in cfg["joint_names"]:
                limit = self.fk.robot_model.joint_map[name].limit
                self.limits.append([limit.lower, limit.upper])

    def receive(self, key, msg):
        with self.lock:
            self.data[key] = (msg, time.monotonic())

    def snapshot(self):
        with self.lock:
            data = dict(self.data)
        required = ["front", "wrist", "joints"] + (["gripper"] if self.cfg["gripper_feedback_topic"] else [])
        now = time.monotonic()
        if any(k not in data or now - data[k][1] > self.cfg["max_sensor_age_s"] for k in required):
            raise RuntimeError("Missing/stale sensor data; no inference or motion")
        stamps = []
        for key in ("front", "wrist"):
            stamp = data[key][0].header.stamp
            seconds = stamp.sec + stamp.nanosec / 1e9
            age = self.node.get_clock().now().nanoseconds / 1e9 - seconds
            if seconds <= 0 or age < -0.05 or age > self.cfg["max_sensor_age_s"]:
                raise RuntimeError("Camera timestamp invalid/stale; check ROS clocks")
            stamps.append(seconds)
        if abs(stamps[0] - stamps[1]) > self.cfg["max_pair_skew_s"]:
            raise RuntimeError("Front/wrist images are not sufficiently synchronized")
        msg = data["joints"][0]
        named = dict(zip(msg.name, msg.position))
        q = np.array([named[n] for n in self.cfg["joint_names"]], dtype=float)
        if self.cfg["joint_state_unit"] == "deg":
            q = np.deg2rad(q)
        if not np.isfinite(q).all():
            raise ValueError("Nonfinite joint feedback")
        link = rigid(self.fk(q))
        if "gripper" in data:
            lo, hi = self.cfg["gripper_feedback_closed"], self.cfg["gripper_feedback_open"]
            if not np.isfinite([lo, hi]).all() or lo == hi:
                raise ValueError("Invalid measured gripper calibration")
            grip = (float(data["gripper"][0].data) - lo) / (hi - lo)
            if not np.isfinite(grip) or not -0.01 <= grip <= 1.01:
                raise ValueError("Gripper feedback outside calibrated range")
            grip, source = float(np.clip(grip, 0, 1)), "measured"
        else:
            grip, source = self.estimated_gripper, "command_estimate"
        state = model_state(link, self.cfg["T_model_from_real_base"], self.cfg["T_link_from_tcp"], grip)
        return data["front"][0].data, data["wrist"][0].data, q, link, state, source

    def solve(self, target, seed):
        from geometry_msgs.msg import PoseStamped
        from moveit_msgs.srv import GetPositionIK
        from scipy.spatial.transform import Rotation

        pose = PoseStamped()
        pose.header.frame_id = self.cfg["base_frame"]
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, target[:3, 3])
        quat = Rotation.from_matrix(target[:3, :3]).as_quat()
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = map(
            float, quat
        )
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.cfg["group_name"]
        req.ik_request.ik_link_name = self.cfg["ik_link"]
        req.ik_request.pose_stamped = pose
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec = 1
        req.ik_request.robot_state.joint_state.name = self.cfg["joint_names"]
        req.ik_request.robot_state.joint_state.position = list(map(float, seed))
        event = threading.Event()
        future = self.ik.call_async(req)
        future.add_done_callback(lambda _: event.set())
        if not event.wait(2):
            future.cancel()
            raise RuntimeError("IK timeout; no fallback motion")
        result = future.result()
        if result is None or result.error_code.val != 1:
            raise RuntimeError("IK failed/collision; no previous-solution fallback")
        values = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
        q = np.array([values[n] for n in self.cfg["joint_names"]])
        joint_guard(seed, q, self.limits, self.cfg)
        return q

    def execute(self, actions, current_q, current_link, observed_at):
        # Validate the entire selected prefix and all IK solutions before any publication.
        planned, previous_q = [], current_q
        previous_tcp = current_link @ rigid(self.cfg["T_link_from_tcp"])
        for action in actions[: self.args.execute_steps]:
            target, tcp = real_target(action, self.cfg["T_model_from_real_base"], self.cfg["T_link_from_tcp"])
            pose_guard(previous_tcp, tcp, self.cfg)
            q = self.solve(target, previous_q)
            planned.append((q, float(np.clip(action[9], 0, 1))))
            previous_q, previous_tcp = q, tcp
        if time.monotonic() - observed_at > self.cfg["max_request_age_s"]:
            raise RuntimeError("Observation/inference/IK too old; do not execute")
        # Reject a stale plan if the real robot moved while inference/IK was running.
        _, _, now_q, now_link, _, _ = self.snapshot()
        drift_cfg = dict(
            self.cfg, max_translation_m=self.cfg["max_state_drift_m"], max_rotation_rad=self.cfg["max_state_drift_rad"]
        )
        pose_guard(
            current_link @ rigid(self.cfg["T_link_from_tcp"]), now_link @ rigid(self.cfg["T_link_from_tcp"]), drift_cfg
        )
        joint_guard(now_q, planned[0][0], self.limits, self.cfg)
        from std_msgs.msg import Float32
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        for q, grip in planned:
            # Refresh/check feedback before EVERY command; never replay an old prefix.
            if time.monotonic() - observed_at > self.cfg["max_request_age_s"]:
                raise RuntimeError("Remaining prefix is stale; stop issuing commands")
            _, _, actual_q, actual_link, _, _ = self.snapshot()
            joint_guard(actual_q, q, self.limits, self.cfg)
            pose_guard(
                actual_link @ rigid(self.cfg["T_link_from_tcp"]),
                self.fk(q) @ rigid(self.cfg["T_link_from_tcp"]),
                self.cfg,
            )
            trajectory = JointTrajectory()
            trajectory.joint_names = self.cfg["joint_names"]
            point = JointTrajectoryPoint()
            point.positions = list(map(float, q))
            dt = self.cfg["step_seconds"]
            point.time_from_start.sec = int(dt)
            point.time_from_start.nanosec = int((dt - int(dt)) * 1e9)
            trajectory.points = [point]
            self.arm_pub.publish(trajectory)
            issued_at = time.monotonic()
            deadline = issued_at + max(3, 2 * dt)
            while True:
                time.sleep(0.02)
                _, _, measured_q, _, _, _ = self.snapshot()
                if (
                    time.monotonic() - issued_at >= dt
                    and np.max(np.abs(measured_q - q)) <= self.cfg["tracking_tolerance_rad"]
                ):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Joint tracking timeout; stop new commands and check robot")
            command = Float32()
            command.data = 1.0 if grip >= 0.5 else 0.0
            self.gripper_pub.publish(command)
            self.estimated_gripper = command.data

    def close(self):
        self.executor.shutdown()
        self.thread.join(2)
        self.node.destroy_node()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:5901")
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--robot-scripts",
        type=Path,
        required=True,
        help="Existing RobotWorkspace scripts directory containing convert.py",
    )
    p.add_argument("--task", required=True)
    p.add_argument("--expected-policy", choices=("sidecar", "baseline"), default="sidecar")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--execute-steps", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("openwam_predictions.jsonl"))
    args = p.parse_args()
    if args.rounds < 1 or not 1 <= args.execute_steps <= 8:
        p.error("Positive rounds; execute-steps1..8 only")
    cfg = json.loads(args.config.read_text())
    validate_config(cfg, args.execute)
    token = args.token_file.read_text().strip()
    session = requests.Session()
    health = session.get(args.url.rstrip("/") + "/health", headers={"Authorization": "Bearer " + token}, timeout=10)
    health.raise_for_status()
    meta = health.json()
    if meta["contract"] != CONTRACT or meta["policy"] != args.expected_policy:
        raise ValueError("Server is not the expected absolute EEF10 policy")
    import rclpy

    rclpy.init()
    robot = None
    try:
        robot = RobotClient(cfg, args)
        input("Sensors starting. Position the robot safely; press Enter when ready. No automatic reset. ")
        if args.execute and input("Motion enabled. Hardware E-stop ready? Type EXECUTE: ") != "EXECUTE":
            raise RuntimeError("Motion not confirmed")
        exp_id = uuid.uuid4().hex
        for index in range(args.rounds):
            if args.execute and index:
                input("Press Enter BEFORE capturing the next observation, or Ctrl+C to stop: ")
            observed_at = time.monotonic()
            front, wrist, q, link, state, source = robot.snapshot()
            actions, result = infer_http(
                session,
                args.url,
                token,
                front,
                wrist,
                state,
                args.task,
                exp_id,
                source,
                meta["checkpoint_sha256"],
                args.expected_policy,
            )
            print(
                json.dumps(
                    dict(
                        round=index,
                        mode="EXECUTE" if args.execute else "DRY_RUN_NO_PUBLISH",
                        first_action=actions[0].tolist(),
                        latency_ms=result["latency_ms"],
                        gripper_state_source=source,
                    )
                )
            )
            with args.output.open("a") as handle:
                handle.write(json.dumps(dict(round=index, state=state.tolist(), response=result)) + "\n")
            if args.execute:
                robot.execute(actions, q, link, observed_at)
    finally:
        if robot is not None:
            robot.close()
        session.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
