"""Read existing schema-5 EA datasets without rewriting data or privileged labels."""
from bisect import bisect_right
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.utils.video_io import decode_video_frames
from openwam.dataloader.transforms.multiview import format_prompt_for_inference
from openwam.model.modules.ea_rsig_v3 import validate_ea_rsig_v3_video_plan_alignment


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class EARSIGV3Dataset(BaseDataset):
    """Native [L6,Lgrip,R6,Rgrip] policy; separate deployable and oracle channels."""
    action_dim = 14
    gripper_indices = (6, 13)

    def __init__(self, dataset_dirs, *, action_space="qpos", action_horizon=40,
                 video_anchors=(0, 10, 20, 30, 40), target_dim=130,
                 height=384, width=320, stats_path=None):
        self.roots = [Path(p).resolve() for p in dataset_dirs]
        if not self.roots or len(set(self.roots)) != len(self.roots):
            raise ValueError("Need nonempty, unique dataset directories")
        if action_space not in ("qpos", "robot_base_delta_ee"):
            raise ValueError("Unsupported action space; chunk-relative data needs its own decoder/contract")
        if height % 96 or width % 64:
            raise ValueError("Canvas must preserve the 2:1 RoboTwin layout on Wan patches")
        self.height, self.width = height, width
        self.action_space, self.horizon, self.target_dim = action_space, action_horizon, target_dim
        self.anchors = tuple(video_anchors)
        validate_ea_rsig_v3_video_plan_alignment(self.horizon, self.anchors)
        self.episodes, self.ends, self.infos, self.tasks = [], [], [], []
        total = 0
        aggregate = {k: {"min": [], "max": [], "mean": [], "std": []} for k in ("action", "observation.state")}
        counts = []
        for ridx, root in enumerate(self.roots):
            summary = json.loads((root / "meta/ea_rsig_v3_conversion_summary.json").read_text())
            expected = {"action_horizon": action_horizon, "video_supervision_indices": list(self.anchors),
                        "oracle_target_dim": target_dim, "policy_oracle_inputs": False}
            if summary.get("conversion_schema_version", 0) < 5:
                raise ValueError(f"Need schema >=5: {root}")
            for key, val in expected.items():
                if summary.get(key) != val:
                    raise ValueError(f"{root}: {key} contract mismatch")
            modality = json.loads((root / "meta/modality.json").read_text())
            required = {"qpos": {"left_joints", "right_joints"}, "robot_base_delta_ee": {
                "left_delta_position", "right_delta_position", "left_delta_rotation", "right_delta_rotation"}}
            if not required[action_space] <= modality["action"].keys():
                raise ValueError(f"{root}: action semantics mismatch")
            info = json.loads((root / "meta/info.json").read_text())
            for field, shape in (("observation.state", [14]), ("action", [14]),
                                 ("observation.ea_rsig_v3.effector_state", [20]),
                                 ("observation.ea_rsig_v3.arm_present", [2]),
                                 ("observation.ea_rsig_v3.targets", [target_dim])):
                if info["features"][field]["shape"] != shape:
                    raise ValueError(f"{root}: wrong {field} width")
            self.infos.append(info)
            self.tasks.append({x["task_index"]: x["task"] for x in load_jsonl(root / "meta/tasks.jsonl")})
            episodes = load_jsonl(root / "meta/episodes.jsonl")
            if len(episodes) != info["total_episodes"]:
                raise ValueError("Incomplete episode metadata")
            for ep in episodes:
                length = ep["length"]
                if length < 1:
                    raise ValueError("Empty episode")
                self.episodes.append((ridx, ep["episode_index"], length))
                total += length
                self.ends.append(total)
            counts.append(sum(e["length"] for e in episodes))
            stats = json.loads((root / "meta/stats_gr00t.json").read_text())
            for field in aggregate:
                for k in aggregate[field]:
                    aggregate[field][k].append(stats[field][k])
        self.stats = {}
        for field, values in aggregate.items():
            low = np.min(values["min"], axis=0).astype(np.float32)
            high = np.max(values["max"], axis=0).astype(np.float32)
            if low.shape != (14,) or not np.isfinite([low, high]).all() or np.any(high < low):
                raise ValueError("Invalid normalization statistics")
            # Original DiT4DiT qpos/delta policy uses binary 0/1 grippers;
            # identity min/max keeps those policy coordinates unchanged on deploy.
            low[list(self.gripper_indices)] = -1
            high[list(self.gripper_indices)] = 1
            mean = np.average(values["mean"], axis=0, weights=counts)
            moment2 = np.average(np.square(values["std"]) + np.square(values["mean"]), axis=0, weights=counts)
            std = np.sqrt(np.maximum(moment2 - np.square(mean), 0))
            self.stats[field] = {"min": low, "max": high, "mean": mean.astype(np.float32), "std": std.astype(np.float32)}
        # Never write stats or any artifacts into the source dataset.
        if stats_path is not None:
            p = Path(stats_path)
        else:
            key = hashlib.sha256(json.dumps([str(r) for r in self.roots] + [action_space, "dit4dit_binary_01"]).encode()).hexdigest()[:16]
            p = Path(__file__).resolve().parents[2] / "outputs/ea_stats" / f"{key}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "joint" if action_space == "qpos" else "eef"
        payload = {mode: self.stats["action"], mode + "_state": self.stats["observation.state"]}
        if p.exists():
            saved = np.load(p, allow_pickle=True).item()
            # Bind each stats file to the exact values rather than trusting its name.
            existing = saved[next(iter(payload))]
            for k in ("min", "max"):
                if not np.array_equal(existing[k], self.stats["action"][k]) or not np.array_equal(saved[mode + "_state"][k], self.stats["observation.state"][k]):
                    raise ValueError("Existing stats mismatch; use a new stats output path")
        else:
            tmp = p.with_name(p.name + f".{os.getpid()}.npy")
            np.save(tmp, payload)
            tmp.replace(p)
        self.normalization_stats_path = str(p.resolve())
        self.normalization_stats = payload

    @classmethod
    def from_config(cls, cfg, split="train"):
        if split != "train":
            raise ValueError("Pass an explicit held-out EA dataset; no implicit training-set validation")
        if cfg.get("normalize_mode", "min-max") != "min-max" or cfg.get("unify_action", False):
            raise ValueError("EA native-policy reader requires min-max normalization and unify_action=false")
        mode = "joint" if cfg.get("action_space", "qpos") == "qpos" else "eef"
        if cfg.get("action_mode", mode) != mode:
            raise ValueError("EA action_mode disagrees with its physical action contract")
        keys = ("dataset_dirs", "action_space", "action_horizon", "video_anchors", "target_dim", "height", "width", "stats_path")
        return cls(**{k: cfg[k] for k in keys if k in cfg and cfg[k] is not None})

    def __len__(self):
        return self.ends[-1]

    @lru_cache(maxsize=16)
    def _table(self, ridx, ep):
        info = self.infos[ridx]
        rel = info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
        return pq.read_table(self.roots[ridx] / rel).to_pydict()

    def normalize(self, values, field):
        s = self.stats[field]
        result = np.clip(2 * (values - s["min"]) / np.maximum(s["max"] - s["min"], 1e-6) - 1, -1, 1).astype(np.float32)
        result[..., list(self.gripper_indices)] = values[..., list(self.gripper_indices)] > 0.5
        return result

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        eidx = bisect_right(self.ends, idx)
        ridx, ep, length = self.episodes[eidx]
        row = idx - (self.ends[eidx - 1] if eidx else 0)
        table = self._table(ridx, ep)
        if len(table["action"]) != length:
            raise ValueError("Parquet/episode length mismatch")
        steps = np.arange(row, row + self.horizon)
        action = np.asarray(table["action"], dtype=np.float32)[np.minimum(steps, length - 1)].copy()
        if self.action_space == "robot_base_delta_ee":
            action[steps >= length, :6] = 0
            action[steps >= length, 7:13] = 0
        video_steps = [min(row + t, length - 1) for t in self.anchors]
        views = []
        info = self.infos[ridx]
        for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            key = "observation.images." + cam
            rel = info["video_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep, video_key=key)
            views.append(decode_video_frames(str(self.roots[ridx] / rel), video_steps, 240, 320))
        video = []
        for frame in range(len(self.anchors)):
            canvas = Image.new("RGB", (self.width, self.height))
            hh = 2 * self.height // 3
            canvas.paste(views[0][frame].resize((self.width, hh), Image.Resampling.BILINEAR), (0, 0))
            for v in (1, 2):
                canvas.paste(views[v][frame].resize((self.width // 2, self.height - hh), Image.Resampling.BILINEAR), ((v - 1) * self.width // 2, hh))
            video.append(canvas)
        effectors = np.asarray(table["observation.ea_rsig_v3.effector_state"][row], dtype=np.float32).reshape(2, 10)
        arms = np.asarray(table["observation.ea_rsig_v3.arm_present"][row], dtype=np.float32)
        targets = np.asarray(table["observation.ea_rsig_v3.targets"][row], dtype=np.float32)[None]
        state = np.asarray(table["observation.state"][row], dtype=np.float32)
        if not all(np.isfinite(x).all() for x in (effectors, arms, targets, state, action)):
            raise ValueError("Nonfinite policy/EA fields")
        if not np.all((arms == 0) | (arms == 1)) or not arms.any():
            raise ValueError("Invalid deployable arm mask")
        return dict(video=video, first_frame_image=[video[0]],
                    prompt=format_prompt_for_inference(self.tasks[ridx][table["task_index"][row]]),
                    action=torch.from_numpy(self.normalize(action, "action")),
                    action_mask=torch.from_numpy(np.broadcast_to((steps < length)[:, None], action.shape).copy()),
                    proprio=torch.from_numpy(self.normalize(state, "observation.state"))[None],
                    proprio_mask=torch.ones(1, 14, dtype=torch.bool),
                    video_mask=torch.tensor([row + t < length for t in self.anchors]),
                    ea_effectors=torch.from_numpy(effectors), ea_arm_present=torch.from_numpy(arms),
                    ea_targets=torch.from_numpy(targets), ea_action_space=self.action_space,
                    ea_video_anchors=list(self.anchors))
