"""Read-only numeric audit of the actual Aloha Sidecar EEF pipeline; CPU only."""
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from benchmarks.robotwin.openwam2robotwin_interface import _extract_eef_proprio
from benchmarks.utils.action_conversion import eef20d_to_ee16d
from openwam.dataloader.ea_rsig_lite import LiteRoboTwinSource
from openwam.dataloader.transforms.normalize import Normalizer, load_mode_stats
from openwam.ea_lite_contract import tcp_context_from_eef20

torch.set_num_threads(1)
OUT = Path(__file__).parent
manifest_path = ROOT / "outputs/sidecar/aloha_4350/manifest.ready.json"
manifest = json.loads(manifest_path.read_text())
run = ROOT / "outputs/sidecar/aloha50_C_joint_10epochs_20261016T131709Z/runs/2026-10-16_13-50-52"
evaluation = ROOT / "outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019"
stats_paths = [Path(manifest["normalization_stats_path"]), run / "normalization_stats.npy", evaluation / "checkpoint_step_32000/normalization_stats.npy"]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in stats_paths}
assert len(set(hashes.values())) == 1
norm = Normalizer(mode="min_max", stats=load_mode_stats(str(stats_paths[0]), "eef"))
reader = object.__new__(LiteRoboTwinSource)
counts = dict(tasks=0, episodes=0, frames=0, sampled_eval_frames=0)
maxima = {}
bf_pos, bf_ang = [], []
fp_pos, fp_ang = [], []
def record(k, value):
    maxima[k] = max(maxima.get(k, 0.), float(np.max(np.abs(value))))

def rotation_error(original, recovered):
    a = np.asarray(recovered, np.float64).reshape(-1, 2, 10)
    b1, b2 = a[..., 3:6].copy(), a[..., 6:9].copy()
    b1 /= np.linalg.norm(b1, axis=-1, keepdims=True)
    b2 -= np.sum(b1 * b2, axis=-1, keepdims=True) * b1
    b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
    matrices = np.stack([b1, b2, np.cross(b1, b2)], axis=-1)
    ref = Rotation.from_quat(original[..., [4, 5, 6, 3]].reshape(-1, 4))
    predicted = Rotation.from_matrix(matrices.reshape(-1, 3, 3))
    return np.rad2deg((ref.inv() * predicted).magnitude())

for task in manifest["tasks"]:
    for episode in task["episodes"]:
        with h5py.File(episode["path"], "r") as f:
            n = episode["frames"]
            raw = reader._read_eef_actions(f, 0, n)
            poses = np.stack([f["endpose/left_endpose"][:], f["endpose/right_endpose"][:]], axis=1)
            oracle = f["ea_rsig_oracle_v3/eef_pose_world_wxyz"][:]
            assert raw.shape == (n, 20) and np.isfinite(raw).all()
            assert all(len(f[f"observation/{camera}/rgb"]) == n for camera in ("head_camera", "left_camera", "right_camera"))
            tcp = tcp_context_from_eef20(raw)
            record("raw_float64_to_float32_position_m", raw.reshape(n, 2, 10)[..., :3] - poses[..., :3])
            record("tcp_vs_oracle_position_m", tcp[..., :3] - oracle[..., :3])
            labels = np.load(episode["labels"])
            record("saved_effectors_vs_recomputed", labels["effectors"] - tcp)
            normalized = norm.normalize(raw)
            recovered = norm.unnormalize(normalized)
            bf_recovered = norm.unnormalize(torch.from_numpy(normalized).to(torch.bfloat16).float().numpy())
            for reconstructed, pos_list, ang_list in ((recovered, fp_pos, fp_ang), (bf_recovered, bf_pos, bf_ang)):
                pos_list.append(np.linalg.norm(reconstructed.reshape(n, 2, 10)[..., :3] - poses[..., :3], axis=-1).reshape(-1))
                ang_list.append(rotation_error(poses, reconstructed))
            record("float32_normalization_roundtrip_all_dims", recovered - raw)
            low, high = norm.stats["min"], norm.stats["max"]
            assert np.all(raw >= low - 2e-7) and np.all(raw <= high + 2e-7)
            for frame in sorted(set((0, n // 2, n - 1))):
                obs = {"endpose": {k: f[f"endpose/{k}"][frame] for k in ("left_endpose", "right_endpose", "left_gripper", "right_gripper")}}
                state = _extract_eef_proprio(obs, "wxyz")
                record("eval_state_vs_training_eef20", state - raw[frame])
                record("eval_vs_training_normalized20", norm.normalize(state) - normalized[frame])
                decoded = eef20d_to_ee16d(recovered[frame]).reshape(2, 8)
                record("decoded_action_position_m", decoded[:, :3] - poses[frame, :, :3])
                # Same xyzw -> wxyz permutation as the actual client EE executor.
                wxyz = decoded[:, [6, 3, 4, 5]]
                decoded_rotation = Rotation.from_quat(wxyz[:, [1, 2, 3, 0]])
                original_rotation = Rotation.from_quat(poses[frame, :, [4, 5, 6, 3]].T)
                record("actual_decoder_angle_deg", np.rad2deg((original_rotation.inv() * decoded_rotation).magnitude()))
                counts["sampled_eval_frames"] += 1
        counts["episodes"] += 1
        counts["frames"] += n
    counts["tasks"] += 1
    print(json.dumps(dict(counts, maxima=maxima)), flush=True)

def summarize(parts):
    a = np.concatenate(parts)
    return dict(max=float(a.max()), p99=float(np.quantile(a, .99)), mean=float(a.mean()))

old = Path("/remote-home/jinminghao/DiT4DiT/.external/RoboTwin")
new = ROOT / ".external/robotwin-official-20261018/RoboTwin"
source_comparisons = {}
for name in ("_trans_endpose", "_trans_from_gripper_to_endlink", "get_left_ee_pose", "get_right_ee_pose", "get_left_tcp_pose", "get_right_tcp_pose"):
    dumps = []
    for source in (old, new):
        tree = ast.parse((source / "envs/robot/robot.py").read_text())
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
        dumps.append(ast.dump(fn, include_attributes=False))
    source_comparisons[name] = dumps[0] == dumps[1]
configs = [yaml.safe_load((source / "assets/embodiments/aloha-agilex/config.yml").read_text()) for source in (old, new)]
geometry_keys = ("gripper_bias", "delta_matrix", "global_trans_matrix", "robot_pose", "ee_joints", "move_group", "urdf_path")
source_comparisons["aloha_geometry"] = {k: configs[0][k] == configs[1][k] for k in geometry_keys}
policy_configs = list(evaluation.rglob("policy_config.yml"))
policy_values = {}
for path in policy_configs:
    cfg = yaml.safe_load(path.read_text())
    key = json.dumps({k: cfg.get(k) for k in ("action_type", "send_state", "state_dim", "source_quaternion_order", "ea_context", "ea_fk")}, sort_keys=True)
    policy_values[key] = policy_values.get(key, 0) + 1
report = dict(status="completed", counts=counts, maxima=maxima, normalization_sha256=hashes,
              float32_position_euclidean_m=summarize(fp_pos), float32_rotation_deg=summarize(fp_ang),
              bf16_normalized_rounding_position_euclidean_m=summarize(bf_pos), bf16_normalized_rounding_rotation_deg=summarize(bf_ang),
              collection_vs_official_source=source_comparisons, actual_eval_policy_configs=policy_values,
              scope="All 2500 Aloha numeric trajectories; 3 eval-state/decoder samples per episode; no physics replay or model inference. BF16 numbers measure representation rounding, not neural prediction error.")
(OUT / "numeric_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
