import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from benchmarks.real_robot.openwam_real_contract import (
    CONTRACT,
    action_chunk,
    eef_to_matrix,
    joint_guard,
    matrix_to_eef,
    model_state,
    pose_guard,
    real_target,
)
from benchmarks.real_robot.serve_maniskill_flask import create_app


def chunk():
    return np.tile(matrix_to_eef(np.eye(4), 1), (32, 1))


def request_data(**changes):
    q = dict(
        task_description="pick up the red cube",
        state=chunk()[0].tolist(),
        exp_id="test",
        request_id="one",
        schema=CONTRACT["schema"],
        gripper_state_source="measured",
    )
    q.update(changes)
    image = io.BytesIO()
    Image.new("RGB", (16, 16)).save(image, format="PNG")
    return dict(
        json=(io.BytesIO(json.dumps(q).encode()), "data.json"),
        main_images=(io.BytesIO(image.getvalue()), "front.png"),
        wrist_images=(io.BytesIO(image.getvalue()), "wrist.png"),
    )


def test_absolute_transform_roundtrip_not_delta():
    base = np.eye(4)
    base[:3, 3] = [0.2, -0.1, 0.3]
    base[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    link = np.eye(4)
    link[:3, 3] = [0.4, 0.1, 0.2]
    link[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    tool = np.eye(4)
    tool[2, 3] = 0.1034
    tool[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    state = model_state(link, base, tool, 0.5)
    target, tcp = real_target(state, base, tool)
    np.testing.assert_allclose(target, link, atol=1e-7)
    np.testing.assert_allclose(tcp, link @ tool, atol=1e-7)
    np.testing.assert_allclose(eef_to_matrix(state), base @ link @ tool, atol=1e-7)
    bad = chunk()
    bad[0, 6:9] = bad[0, 3:6]
    with pytest.raises(ValueError):
        action_chunk(bad)


def test_pose_and_all_seven_joint_guards():
    cfg = dict(
        workspace_xyz=[[-1] * 3, [1] * 3],
        max_translation_m=0.02,
        max_rotation_rad=0.15,
        max_joint_step_rad=0.08,
        max_joint_speed_rad_s=0.3,
        step_seconds=0.2,
    )
    target = np.eye(4)
    target[2, 3] = 0.03
    with pytest.raises(ValueError):
        pose_guard(np.eye(4), target, cfg)
    limits = np.tile([-2, 2], (7, 1))
    q = np.zeros(7)
    q[6] = 0.1
    with pytest.raises(ValueError):
        joint_guard(np.zeros(7), q, limits, cfg)
    q[6] = 0.07
    with pytest.raises(ValueError):
        joint_guard(np.zeros(7), q, limits, cfg)
    joint_guard(np.zeros(7), np.full(7, 0.01), limits, cfg)


def test_http_contract_and_raw_physical_state():
    seen = []

    def infer(front, wrist, state, task):
        seen.append((state, task, front.size, wrist.size))
        return chunk()

    client = create_app(infer, dict(policy="sidecar", checkpoint_sha256="fixture"), "a" * 32).test_client()
    assert client.get("/health").status_code == 401
    auth = {"Authorization": "Bearer " + "a" * 32}
    assert client.get("/health", headers=auth).json["contract"] == CONTRACT
    result = client.post("/api/inference", data=request_data(), headers=auth)
    assert result.status_code == 200 and np.asarray(result.json["actions"]).shape == (32, 10)
    assert result.json["request_id"] == "one"
    assert seen[0][1] == "pick up the red cube" and seen[0][0].shape == (10,)
    assert client.post("/api/inference", data=request_data(state=[0] * 7), headers=auth).status_code == 400
    assert client.post("/api/inference", data=request_data(ea_targets=[1]), headers=auth).status_code == 400
    assert client.post("/api/inference", data=request_data(schema="delta_ee"), headers=auth).status_code == 400


def test_server_records_exact_request_images_and_actions(tmp_path):
    client = create_app(lambda *args: chunk(), dict(policy="sidecar"), "a" * 32, record_dir=tmp_path).test_client()
    data = request_data()
    raw_query = data["json"][0].getvalue()
    raw_front = data["main_images"][0].getvalue()
    raw_wrist = data["wrist_images"][0].getvalue()
    result = client.post("/api/inference", data=data, headers={"Authorization": "Bearer " + "a" * 32})
    assert result.status_code == 200
    (case_dir,) = list(tmp_path.iterdir())
    assert (case_dir / "request.json").read_bytes() == raw_query
    assert (case_dir / "front.png").read_bytes() == raw_front
    assert (case_dir / "wrist.png").read_bytes() == raw_wrist
    assert json.loads((case_dir / "response.json").read_bytes()) == result.json
    assert not any(b"a" * 32 in path.read_bytes() for path in case_dir.iterdir())


def test_no_action_on_model_failure_and_no_queue():
    import threading

    started, release = threading.Event(), threading.Event()

    def infer(*args):
        started.set()
        assert release.wait(5)
        raise RuntimeError("fixture failure")

    app = create_app(infer, {}, "b" * 32)
    auth = {"Authorization": "Bearer " + "b" * 32}
    results = []
    worker = threading.Thread(
        target=lambda: results.append(app.test_client().post("/api/inference", data=request_data(), headers=auth))
    )
    worker.start()
    assert started.wait(5)
    try:
        assert app.test_client().post("/api/inference", data=request_data(), headers=auth).status_code == 409
    finally:
        release.set()
        worker.join(5)
    assert results[0].status_code == 500 and "actions" not in results[0].json


def test_robot_config_is_fail_closed_and_client_rejects_wrong_response():
    from pathlib import Path

    from benchmarks.real_robot.openwam_robot_client import infer_http, validate_config

    cfg = json.loads((Path(__file__).resolve().parents[2] / "benchmarks/real_robot/robot.example.json").read_text())
    validate_config(cfg, False)
    with pytest.raises(ValueError, match="calibration"):
        validate_config(cfg, True)
    cfg["calibration_confirmed"] = True
    with pytest.raises(ValueError, match="workspace"):
        validate_config(cfg, True)
    cfg["workspace_xyz"] = [[-1] * 3, [1] * 3]
    validate_config(cfg, True)
    cfg["max_translation_m"] = float("nan")
    with pytest.raises(ValueError):
        validate_config(cfg, True)

    class Session:
        def post(self, *args, **kwargs):
            q = json.loads(kwargs["files"]["json"][1].getvalue())
            result = dict(
                contract=CONTRACT,
                actions=chunk().tolist(),
                exp_id=q["exp_id"],
                request_id="old-response",
                policy="sidecar",
                checkpoint_sha256="a",
            )
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: result)

    with pytest.raises(ValueError, match="Wrong/stale"):
        infer_http(
            Session(),
            "http://fixture",
            "token",
            b"png",
            b"png",
            chunk()[0],
            "task",
            "episode",
            "measured",
            "a",
            "sidecar",
        )


def test_entire_prefix_must_pass_ik_before_any_publication():
    from pathlib import Path

    from benchmarks.real_robot.openwam_robot_client import RobotClient

    cfg = json.loads((Path(__file__).resolve().parents[2] / "benchmarks/real_robot/robot.example.json").read_text())
    cfg["workspace_xyz"] = [[-1] * 3, [1] * 3]
    robot = RobotClient.__new__(RobotClient)
    robot.cfg, robot.args = cfg, SimpleNamespace(execute_steps=2)
    publications, calls = [], []
    robot.arm_pub = SimpleNamespace(publish=lambda x: publications.append(x))
    robot.gripper_pub = robot.arm_pub

    def solve(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("IK failed")
        return np.zeros(7)

    robot.solve = solve
    with pytest.raises(RuntimeError, match="IK failed"):
        robot.execute(chunk(), np.zeros(7), np.eye(4), 0)
    assert len(calls) == 2 and not publications
