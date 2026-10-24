"""Read-only 32 B=1 loss-gradient diagnostic on pinned expert trajectories."""

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from benchmarks.robotwin.sidecar_gradient_probe import measure


@contextmanager
def fixed_times(video_id, action_id, video_bounds, action_count):
    """Intercept only the two audited FM draws, preserving their RNG consumption."""
    original = torch.randint
    expected = [(*video_bounds, video_id), (0, action_count, action_id)]
    calls = []

    def draw(low, high, size, **kwargs):
        assert len(calls) < 2, "Unexpected randint inside diagnostic FM path"
        lo, hi, value = expected[len(calls)]
        assert (low, high, tuple(size)) == (lo, hi, (1,))
        assert lo <= value < hi
        result = original(low, high, size, **kwargs)
        calls.append(value)
        return result.fill_(value)

    with patch("torch.randint", draw):
        yield
        assert calls == [video_id, action_id], calls


def stats(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    assert np.isfinite(a).all()
    return (
        dict(
            n=len(a),
            mean=float(a.mean()),
            std=float(a.std()),
            p50=float(np.median(a)),
            min=float(a.min()),
            max=float(a.max()),
        )
        if len(a)
        else dict(n=0)
    )


def signed_stats(values):
    values = [v for v in values if v is not None]
    result = stats(values)
    if values:
        a = np.asarray(values)
        result.update(
            positive_fraction=float((a > 0).mean()),
            negative_fraction=float((a < 0).mean()),
            zero_fraction=float((a == 0).mean()),
            cancellation_ratio=float(abs(a.sum()) / abs(a).sum()) if abs(a).sum() else None,
        )
    return result


def summarize(reports):
    result = dict(
        completed=len(reports), parameters_unchanged=all(r["parameters_unchanged"] for r in reports), subsets={}
    )
    subsets = {"all": reports}
    for key in ("domain", "task", "stage", "action_id"):
        for value in sorted({str(r["sample"][key]) for r in reports}):
            subsets[f"{key}={value}"] = [r for r in reports if str(r["sample"][key]) == value]
    for domain in sorted({r["sample"]["domain"] for r in reports}):
        for task in sorted({r["sample"]["task"] for r in reports}):
            subsets[f"{domain}/{task}"] = [
                r for r in reports if r["sample"]["domain"] == domain and r["sample"]["task"] == task
            ]
    for task in sorted({r["sample"]["task"] for r in reports}):
        for action_id in sorted({r["sample"]["action_id"] for r in reports}):
            subsets[f"{task}/action_id={action_id}"] = [
                r for r in reports if r["sample"]["task"] == task and r["sample"]["action_id"] == action_id
            ]
    for label, rows in subsets.items():
        if not rows:
            continue
        groups = {}
        for group in rows[0]["groups"]:
            groups[group] = dict(
                norms={
                    loss: stats([r["groups"][group]["norms"][loss] for r in rows]) for loss in ("action", "ea", "video")
                },
                cosines={
                    pair: signed_stats([r["groups"][group]["cosines"][pair]["cosine"] for r in rows])
                    for pair in ("action_ea", "action_video", "video_ea")
                },
                ea_to_action_ratio=stats(
                    [
                        r["groups"][group]["norms"]["ea"] / r["groups"][group]["norms"]["action"]
                        for r in rows
                        if r["groups"][group]["norms"]["action"] > 0
                    ]
                ),
            )
        gates = {}
        for gate in rows[0]["gates"]:
            gradients = {
                loss: signed_stats([r["gates"][gate]["gradients"][loss] for r in rows])
                for loss in ("action", "video", "ea")
            }
            gradients["action_plus_video"] = signed_stats(
                [sum(r["gates"][gate]["gradients"][loss] or 0 for loss in ("action", "video")) for r in rows]
            )
            gates[gate] = dict(
                beta=rows[0]["gates"][gate]["beta"],
                gradients=gradients,
                d_loss_d_gate_scale={
                    loss: signed_stats(
                        [
                            r["gates"][gate]["beta"] * r["gates"][gate]["gradients"][loss]
                            for r in rows
                            if r["gates"][gate]["gradients"][loss] is not None
                        ]
                    )
                    for loss in ("action", "video", "ea")
                },
            )
        result["subsets"][label] = dict(
            n=len(rows),
            groups=groups,
            gates=gates,
            shared_cosines={
                pair: signed_stats(
                    [
                        r["shared_parameter_cosines"][pair]["cosine"] if r["shared_parameter_cosines"][pair] else None
                        for r in rows
                    ]
                )
                for pair in ("action_ea", "action_video", "video_ea")
            },
            d_loss_d_global_write_scale={
                loss: signed_stats([r["d_loss_d_global_write_scale"][loss] for r in rows])
                for loss in ("action", "video", "ea")
            },
        )
    return result


def prepare(out, gtroot):
    """Seal task/stage/time coverage before seeing any gradients."""
    import h5py

    out, gtroot = Path(out), Path(gtroot)
    oldrows = json.loads((gtroot / "heldout_eval/inputs_ready.json").read_text())["samples"]
    episodes = json.loads((gtroot / "expert_collection.json").read_text())["completed"]
    rows = []
    for domain in ("demo_clean", "demo_randomized"):
        for task_index, task in enumerate(("adjust_bottle", "place_shoe", "beat_block_hammer")):
            ep = next(e for e in episodes if e["domain"] == domain and e["task"] == task)
            with h5py.File(ep["path"], "r") as f:
                contact = (f["ea_rsig_oracle_v3/track_contact"][:] * f["ea_rsig_oracle_v3/track_valid"][:]).max(1) > 0.5
            onset = int(np.flatnonzero(contact)[0])
            last = ep["frames"] - 33
            anchors = (
                [0, onset - 8, onset - 1, onset, onset + 8, last]
                if task_index == 0
                else [onset - 16, onset - 1, onset, onset + 8, last]
            )
            assert len(set(anchors)) == len(anchors) and min(anchors) >= 0 and max(anchors) <= last
            reference = next(r for r in oldrows if r["source"] == ep["path"])
            digest = hashlib.sha256(Path(ep["path"]).read_bytes()).hexdigest()
            for j, anchor in enumerate(anchors):
                rows.append(
                    dict(
                        index=len(rows),
                        task=task,
                        domain=domain,
                        source=ep["path"],
                        seed=ep["seed"],
                        frames=ep["frames"],
                        anchor=anchor,
                        source_sha256=digest,
                        payload=reference["payload"],
                        contact_onset=onset,
                        contact_at_anchor=bool(contact[anchor]),
                        stage="pre_contact"
                        if anchor < onset - 4
                        else ("contact_transition" if anchor <= onset + 4 else "post_grasp_contact"),
                        action_id=(100, 500, 900)[(j + task_index) % 3],
                        video_id=500,
                        noise_seed=4296000 + len(rows),
                    )
                )
    assert len(rows) == 32 and sum(r["domain"] == "demo_clean" for r in rows) == 16
    plan = dict(
        status="sealed_before_gradients",
        batches=32,
        batch_size=1,
        independent_episodes=6,
        checkpoint_sha256="501addbed687e6d7587af60ab4480b9c79eaa2c8867d723b74d54e0acc9deb55",
        scope="Three tasks; one expert episode per task/domain, overlapping windows. Randomized diagnostic only. No optimizer, no augmentation, no network/training changes; video ID500 controlled.",
        task_batches_per_domain={"adjust_bottle": 6, "place_shoe": 5, "beat_block_hammer": 5},
        limits="Oracle contact is gripper-object, not hammer-target impact. No release phase present in these six episodes; no independent-noise replicate per anchor. No Adam state/update claim.",
    )
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    (out / "inputs_ready.json").write_text(json.dumps(dict(samples=rows), indent=2))
    return rows


def sample_for(row, cache):
    import h5py

    from openwam.dataloader.ea_rsig_lite import CAMERAS, LiteRoboTwinSource, targets_from_oracle

    if row["source"] not in cache:
        path = Path(row["source"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["source_sha256"]
        manifest = json.loads(Path("outputs/sidecar/aloha_4350/manifest.ready.json").read_text())
        assert (
            hashlib.sha256(Path(manifest["normalization_stats_path"]).read_bytes()).hexdigest()
            == manifest["normalization_stats_sha256"]
        )
        metadata = dict(
            episode_files=[str(path)],
            episode_lengths=[row["frames"]],
            obs_image_size=[240, 320],
            scene_info={},
            instructions={},
        )
        reader = LiteRoboTwinSource(
            str(path.parent),
            num_frames=33,
            video_stride=4,
            height=384,
            width=320,
            split="train",
            task_name=row["task"],
            normalization_stats_path=manifest["normalization_stats_path"],
            normalize_mode="min-max",
            multiview=True,
            camera_layout=list(CAMERAS),
            action_mode="eef",
            unify_action=True,
            unify_action_map=["0-9", "34-43"],
            prepared_metadata=metadata,
        )
        # split only selects the reader's prompt convention; Randomized is never registered as training data.
        with h5py.File(path, "r") as f:
            labels = targets_from_oracle(f["ea_rsig_oracle_v3"], [(320, 240)] * 3, row["task"])
        boxvalid = labels[:, 118:130].reshape(-1, 4, 3)
        boxvalid[:, 1] *= labels[:, 14:15]
        boxvalid[:, 3] *= labels[:, 15:16]
        cache[row["source"]] = (reader, labels)
    reader, labels = cache[row["source"]]
    sample = reader._build_sample(0, row["anchor"])
    sample["prompt"] = json.loads(Path(row["payload"]).read_text())["prompt"]
    sample.update(
        ea_targets=torch.from_numpy(labels[row["anchor"] : row["anchor"] + 1].copy()),
        ea_plan_steps=[8, 16, 32],
        ea_video_anchors=list(range(0, 33, 4)),
    )
    assert sample["action"].shape == (32, 80) and sample["action_mask"].sum() == 640
    assert torch.isfinite(sample["action"]).all() and len(sample["video"]) == 9
    return sample


def run(server, out, rows):
    from openwam.model.video_backbone.wan import encode

    out = Path(out)
    arch = server.engine.architecture
    original_encode = encode.encode_text

    def cached(prompts, **kwargs):
        contexts = [server.engine._prompt_embed_cache[p] for p in prompts]
        return tuple(torch.cat([pair[i].clone() for pair in contexts], dim=0) for i in (0, 1))

    reports = []
    cache = {}
    start = time.monotonic()
    try:
        encode.encode_text = cached
        arch.init_training_schedulers()
        arch.set_training_runtime(
            use_gradient_checkpointing=True, min_timestep_boundary=0.5, max_timestep_boundary=0.501
        )
        assert len(rows) == 32
        cfg = json.loads((out / "plan.json").read_text())
        from omegaconf import OmegaConf

        checkpoint_dir = Path(cfg.get("checkpoint_dir", out / "checkpoint_step_42960"))
        checkpoint_cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
        assert checkpoint_cfg.training.lambda_action == checkpoint_cfg.training.lambda_video == 1.0
        manifest = json.loads(Path("outputs/sidecar/aloha_4350/manifest.ready.json").read_text())
        assert (
            hashlib.sha256((checkpoint_dir / "normalization_stats.npy").read_bytes()).hexdigest()
            == manifest["normalization_stats_sha256"]
        )
        cfg.update(
            aux_loss_scale=arch.aux_loss_scale,
            loss_weights=dict(arch.loss_weights),
            deep_supervision=arch.deep_supervision,
            lambda_action=1.0,
            lambda_video=1.0,
            network_mode="eval, checkpointed backward, backbone parameter gradients disabled, input gradients retained",
        )
        (out / "runtime_contract.json").write_text(json.dumps(cfg, indent=2))
        for row in rows:
            batchstart = time.monotonic()
            torch.manual_seed(row["noise_seed"])
            torch.cuda.reset_peak_memory_stats()
            sample = sample_for(row, cache)
            prepared = arch.prepare_inputs([sample])
            prepared = {k: v.to(arch.device) if isinstance(v, torch.Tensor) else v for k, v in prepared.items()}
            with fixed_times(
                row["video_id"], row["action_id"], (500, 501), len(arch.action_backbone.scheduler.timesteps)
            ):
                report = measure(arch, prepared)
            assert report["timestep_ids"]["_compute_action_loss"] == [row["action_id"]]
            assert report["timestep_ids"]["_compute_video_loss"] == [row["video_id"]]
            report.update(
                sample=row,
                seconds=time.monotonic() - batchstart,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                action_sigma=float(arch.action_backbone.scheduler.sigmas[row["action_id"]]),
                video_sigma=float(arch.video_backbone.scheduler.sigmas[row["video_id"]]),
            )
            (out / f"batch_{row['index']:02d}.json").write_text(json.dumps(report, indent=2, allow_nan=False))
            reports.append(report)
            progress = dict(
                status="running",
                completed=len(reports),
                total=32,
                elapsed_seconds=time.monotonic() - start,
                last_sample=row,
            )
            (out / "progress.json").write_text(json.dumps(progress, indent=2))
            print("GRADIENT_BATCH_COMPLETE", len(reports), round(report["seconds"], 2), flush=True)
            del prepared, sample
        summary = summarize(reports)
        summary["scope"] = cfg
        (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        (out / "progress.json").write_text(
            json.dumps(
                dict(status="complete", completed=32, total=32, elapsed_seconds=time.monotonic() - start), indent=2
            )
        )
    except Exception:
        import traceback

        (out / "gradient32_failed.txt").write_text(traceback.format_exc())
        raise
    finally:
        encode.encode_text = original_encode
