"""Instance-local write-only intervention, paired action metrics and isolated GPU diagnostics."""

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("Expected finite, nonempty diagnostic measurements")
    return dict(
        n=int(x.size),
        mean=float(x.mean()),
        p50=float(np.quantile(x, 0.5)),
        p90=float(np.quantile(x, 0.9)),
        p99=float(np.quantile(x, 0.99)),
        max=float(x.max()),
    )


class WriteProbe:
    """Leave reads active; zero beta only while computing an off write, then restore it."""

    def __init__(self, model):
        if model.training or not model.sidecar_enabled or not model.sidecar.action_conditioning:
            raise ValueError("Use an eval-mode complete Sidecar model")
        self.model, self.mode, self.collect = model, "full", False
        self.rows, self.replays = [], []
        self.time, self.current_projection = None, {}
        self.record_inputs = False
        self.scale = 1.0
        self.original_write = model.sidecar.write_action
        self.original_forward = getattr(model, "_evaluation_eager_forward", model.forward)
        for index, stage in enumerate(model.sidecar.stages):
            stage.write.projection.register_forward_hook(
                lambda module, args, output, index=index: self.current_projection.__setitem__(index, output.detach())
            )
        model.sidecar.write_action = self.write
        model.forward = self.forward
        if hasattr(model, "_evaluation_eager_write_action"):
            model._evaluation_eager_write_action = self.write
            model._evaluation_eager_forward = self.forward

    @contextmanager
    def use(self, mode, collect=False, record_inputs=False, scale=1.0):
        if mode not in ("full", "write_off"):
            raise ValueError(mode)
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("Write scale must be finite and nonnegative")
        previous = self.mode, self.collect, self.record_inputs, self.scale
        self.mode, self.collect, self.record_inputs = mode, collect, record_inputs
        self.scale = 0.0 if mode == "write_off" else float(scale)
        try:
            yield
        finally:
            self.mode, self.collect, self.record_inputs, self.scale = previous

    def forward(self, actions, action_time, **kwargs):
        self.time = (float(action_time.flatten()[0]), float(kwargs["timestep"].flatten()[0]))
        if self.record_inputs:
            self.replays.append((actions.detach().clone(), action_time.detach().clone(), dict(kwargs)))
        return self.original_forward(actions, action_time, **kwargs)

    def write(self, index, actions, tokens, valid):
        beta = self.model.sidecar.stages[index].write.beta
        saved = beta.detach().clone()
        try:
            if self.scale != 1.0:
                with torch.no_grad():
                    beta.copy_(saved * self.scale)
            applied_beta = beta.detach().clone()
            output = self.original_write(index, actions, tokens, valid)
        finally:
            if self.scale != 1.0:
                with torch.no_grad():
                    beta.copy_(saved)
        if self.collect:
            before, after = actions.float(), output.float()
            denominator = before.flatten(1).norm(dim=1) + 1e-8
            actual = (after - before).flatten(1).norm(dim=1) / denominator
            intended = self.current_projection[index].float() * saved.float() * self.scale
            intended_norm = intended.flatten(1).norm(dim=1)
            intended = intended_norm / denominator
            native = (applied_beta.to(actions.dtype) * self.current_projection[index]).float()
            native_norm = native.flatten(1).norm(dim=1)
            delta_norm = (after - before).flatten(1).norm(dim=1)
            elements = before[0].numel()
            magnitudes = dict(
                hidden_l2=before.flatten(1).norm(dim=1),
                projection_l2=self.current_projection[index].float().flatten(1).norm(dim=1),
                gated_fp32_l2=intended_norm,
                gated_native_l2=native_norm,
                effective_delta_l2=delta_norm,
                hidden_rms=before.square().flatten(1).mean(dim=1).sqrt(),
                gated_native_rms=native_norm / (elements**0.5),
                effective_delta_rms=delta_norm / (elements**0.5),
                native_pre_addition_ratio=native_norm / denominator,
                changed_fraction=(output != actions).float().flatten(1).mean(dim=1),
            )
            magnitudes = {key: value.tolist() for key, value in magnitudes.items()}
            for row_index, (measured, unrounded) in enumerate(zip(actual.tolist(), intended.tolist())):
                self.rows.append(
                    dict(
                        tap=self.model.tap_blocks[index],
                        action_t=self.time[0],
                        video_t=self.time[1],
                        ratio=measured,
                        pre_addition_ratio=unrounded,
                        **{key: value[row_index] for key, value in magnitudes.items()},
                    )
                )
        return output


def action_metrics(full, off, normalizer):
    from scipy.spatial.transform import Rotation

    from benchmarks.utils.action_conversion import rot6d_to_quat_xyzw

    full, off = np.asarray(full), np.asarray(off)
    if full.shape != (32, 20) or off.shape != full.shape or not np.isfinite([full, off]).all():
        raise ValueError("Expected finite 32x20 physical EEF predictions")
    normalized_full, normalized_off = normalizer.normalize(full), normalizer.normalize(off)
    positions = np.concatenate([np.linalg.norm(full[:, j : j + 3] - off[:, j : j + 3], axis=-1) for j in (0, 10)])
    rotations = []
    for j in (3, 13):
        a = Rotation.from_quat(np.stack([rot6d_to_quat_xyzw(v) for v in full[:, j : j + 6]]))
        b = Rotation.from_quat(np.stack([rot6d_to_quat_xyzw(v) for v in off[:, j : j + 6]]))
        rotations.extend((a.inv() * b).magnitude().tolist())
    return dict(
        delta_action_relative_normalized=float(
            np.linalg.norm(normalized_full - normalized_off) / (np.linalg.norm(normalized_off) + 1e-8)
        ),
        delta_action_relative_physical_eef20=float(np.linalg.norm(full - off) / (np.linalg.norm(off) + 1e-8)),
        position_difference_mm=distribution(positions * 1000),
        rotation_difference_degrees=distribution(np.rad2deg(rotations)),
        gripper_max_abs_difference=float(np.max(np.abs(full[:, [9, 19]] - off[:, [9, 19]]))),
    )


def run_loaded_server(server, out):
    try:
        return _run_loaded_server(server, Path(out))
    except BaseException as exc:
        (Path(out) / "model_failed.json").write_text(json.dumps(dict(error=repr(exc))))
        raise


def _run_loaded_server(server, out):
    """Use CUDA text embeddings from GPU4, then release T5 and run the WAM on GPU7."""
    from omegaconf import OmegaConf

    from openwam.deploy.optimizations import DiTVelocityCache
    from openwam.deploy.server import PolicyServer
    from openwam.model.video_backbone.wan.encode import encode_text

    out = Path(out)
    inputs = json.loads((out / "inputs_ready.json").read_text())["samples"]
    plan = json.loads((out / "plan.json").read_text())
    expected_samples = len(plan["tasks"]) * len(plan["modes"]) * plan["episodes_per_condition"]
    arch, engine = server.engine.architecture, server.engine
    vb = arch.video_backbone
    prompts = sorted({json.loads(Path(row["payload"]).read_text())["prompt"] for row in inputs})
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "outputs/robotwin_eval/rebuild_official_20261018/action_parity/inputs.json"
    )
    prompts = sorted(set(prompts) | {payload["prompt"] for payload in json.loads(fixture_path.read_text())})

    def wait_memory(physical, minimum):
        import subprocess

        announced = False
        while True:
            raw = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
            )
            free = {int(a): int(b) for a, b in (line.split(",") for line in raw.splitlines())}[physical]
            if free >= minimum:
                return
            if not announced:
                print(f"WAIT_CAPACITY GPU{physical} free={free} need={minimum}", flush=True)
                announced = True
            time.sleep(5)

    # A first completed task can be diagnosed while other scene collectors still use GPU4.
    encoding_device = 1 if len(inputs) < expected_samples else 0
    wait_memory(7 if encoding_device else 4, 17000 if encoding_device else 14000)
    torch.cuda.set_device(encoding_device)
    vb.text_encoder.to(device=f"cuda:{encoding_device}", dtype=arch.dtype)
    encoded = {}
    with torch.inference_mode():
        for prompt in prompts:
            encoded[prompt] = tuple(
                v.to("cpu")
                for v in encode_text(
                    [prompt], tokenizer=vb._tokenizer, text_encoder=vb.text_encoder, device=f"cuda:{encoding_device}"
                )
            )
    vb.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    wait_memory(7, 17000)
    text_encoder = vb.text_encoder
    vb.text_encoder = None
    try:
        torch.cuda.set_device(1)
        arch.set_dtype_device(arch.dtype, torch.device("cuda:1"))
    finally:
        vb.text_encoder = text_encoder
    arch.eval()
    engine._prompt_embed_cache = {key: tuple(value.to("cuda:1") for value in values) for key, values in encoded.items()}
    # Fail before a missed cache lookup could mix CPU text encoding into this diagnostic.
    original_encode = vb.text_encoder.forward

    def forbid_uncached(*args, **kwargs):
        raise RuntimeError("Unexpected uncached prompt; all diagnostic prompts must be precomputed on GPU4")

    vb.text_encoder.forward = forbid_uncached
    server._init_policy()
    from benchmarks.robotwin.check_sidecar_fast_runtime import validate_server

    validation = validate_server(server, out / "fast_validation", strategies=("ffn",))
    if not validation["passed"]:
        raise RuntimeError("FFN numerical validation failed on real fixtures")
    probe = WriteProbe(arch)
    compile_cfg = OmegaConf.create(dict(enabled=True, sidecar_strategy="ffn"))
    arch.apply_compile_optimizations(compile_cfg)
    (out / "gpu_model_ready.json").write_text(
        json.dumps(
            dict(
                device="GPU7",
                initial_text_embeddings_device=f"GPU{7 if encoding_device else 4}",
                t5_resident_device="cpu",
                prompt_count=len(prompts),
                allocated_bytes=torch.cuda.memory_allocated(),
            )
        )
    )
    rows, ratios = [], []
    for cached in (False, True):
        engine._dit_cache = DiTVelocityCache(cosine_threshold=0.99, max_consecutive_skips=3) if cached else None
        for sample in inputs:
            payload = json.loads(Path(sample["payload"]).read_text())
            observation = server._obs_preprocessor.preprocess(payload)
            conditions = server._policy._build_conditions(observation)
            conditions["seed"] = 42
            predictions = {}
            probe.rows, probe.replays = [], []
            with torch.inference_mode():
                for mode in ("full", "write_off"):
                    server.reset()
                    with probe.use(mode, collect=mode == "full", record_inputs=mode == "full" and not cached):
                        predictions[mode] = engine.generate(conditions)["actions"]
                if sample == inputs[0]:
                    server.reset()
                    with probe.use("full"):
                        repeated = engine.generate(conditions)["actions"]
                    if not np.array_equal(repeated, predictions["full"]):
                        raise RuntimeError("Full/off intervention or RNG reset changed the full-repeat reference")
            measured = [
                dict(sample={k: v for k, v in sample.items() if k != "payload"}, cached=cached, **value)
                for value in probe.rows
            ]
            ratios.extend(measured)
            velocity = []
            with torch.inference_mode():
                for actions, action_time, kwargs in probe.replays:
                    with probe.use("full"):
                        full_velocity = arch.forward(actions, action_time, **kwargs)[1]
                    with probe.use("write_off"):
                        off_velocity = arch.forward(actions, action_time, **kwargs)[1]
                    # Unified padding is excluded; only the 20 observed/action dimensions count.
                    indices = list(range(10)) + list(range(34, 44))
                    difference = full_velocity[..., indices].float() - off_velocity[..., indices].float()
                    velocity.append(
                        dict(
                            action_t=float(action_time.flatten()[0]),
                            relative_difference=float(
                                difference.norm() / (off_velocity[..., indices].float().norm() + 1e-8)
                            ),
                        )
                    )
            probe.replays = []
            row = dict(
                sample={k: v for k, v in sample.items() if k != "payload"},
                cached=cached,
                **action_metrics(predictions["full"], predictions["write_off"], arch.normalizer),
                same_input_velocity_comparisons=velocity,
                full_actions=np.asarray(predictions["full"]).tolist(),
                write_off_actions=np.asarray(predictions["write_off"]).tolist(),
            )
            rows.append(row)
            (out / "offline_progress.json").write_text(
                json.dumps(dict(completed=len(rows), total=2 * len(inputs), results=rows))
            )
            (out / "write_ratios_progress.json").write_text(json.dumps(dict(measurements=ratios)))
            print(
                "OFFLINE",
                sample["task"],
                sample["mode"],
                sample["episode"],
                cached,
                row["delta_action_relative_normalized"],
                flush=True,
            )
            if not cached and sample == inputs[-1] and len(inputs) < expected_samples:
                print("INITIAL_TASK_DIAGNOSTICS_COMPLETE; waiting for remaining fixed scenes", flush=True)
                while True:
                    if json.loads((out / "controller_state.json").read_text())["status"] == "failed":
                        raise RuntimeError("Scene collection failed")
                    try:
                        all_inputs = json.loads((out / "inputs_ready.json").read_text())["samples"]
                    except json.JSONDecodeError:
                        time.sleep(2)
                        continue
                    if len(all_inputs) == expected_samples:
                        break
                    time.sleep(2)
                pending = [row for row in all_inputs if row not in inputs]
                extra_prompts = {json.loads(Path(row["payload"]).read_text())["prompt"] for row in pending}
                wait_memory(4, 14000)
                torch.cuda.set_device(0)
                vb.text_encoder.forward = original_encode
                try:
                    vb.text_encoder.to("cuda:0")
                    with torch.inference_mode():
                        for prompt in sorted(extra_prompts - set(engine._prompt_embed_cache)):
                            values = encode_text(
                                [prompt], tokenizer=vb._tokenizer, text_encoder=vb.text_encoder, device="cuda:0"
                            )
                            engine._prompt_embed_cache[prompt] = tuple(value.to("cpu").to("cuda:1") for value in values)
                finally:
                    vb.text_encoder.to("cpu")
                    vb.text_encoder.forward = forbid_uncached
                    torch.cuda.empty_cache()
                    torch.cuda.set_device(1)
                inputs.extend(pending)
    summary = []
    for cached in (False, True):
        for domain in ("demo_clean", "demo_randomized"):
            for tap in (7, 15, 23):
                selected = [
                    r for r in ratios if r["cached"] == cached and r["sample"]["mode"] == domain and r["tap"] == tap
                ]
                summary.append(
                    dict(
                        cached=cached,
                        domain=domain,
                        tap=tap,
                        actual_ratio=distribution([r["ratio"] for r in selected]),
                        pre_addition_ratio=distribution([r["pre_addition_ratio"] for r in selected]),
                    )
                )
    per_time = []
    clean_times = {r["action_t"] for r in ratios if not r["cached"] and r["sample"]["mode"] == "demo_clean"}
    random_times = {r["action_t"] for r in ratios if not r["cached"] and r["sample"]["mode"] == "demo_randomized"}
    if clean_times != random_times or len(clean_times) != 10:
        raise RuntimeError("Uncached Clean/Randomized diagnostics must use identical ten action timesteps")
    for domain in ("demo_clean", "demo_randomized"):
        for tap in (7, 15, 23):
            for action_t in sorted(clean_times):
                selected = [
                    r["ratio"]
                    for r in ratios
                    if not r["cached"]
                    and r["sample"]["mode"] == domain
                    and r["tap"] == tap
                    and r["action_t"] == action_t
                ]
                per_time.append(dict(domain=domain, tap=tap, action_t=action_t, actual_ratio=distribution(selected)))
    (out / "write_ratios.json").write_text(
        json.dumps(dict(summary=summary, per_time=per_time, measurements=ratios), indent=2)
    )
    (out / "offline_complete.json").write_text(
        json.dumps(
            dict(
                results=rows,
                scope="Identical initial inputs/seed42/10 steps; per-forward noise inputs replayed exactly for velocity comparisons",
            ),
            indent=2,
        )
    )
    engine._dit_cache = DiTVelocityCache(cosine_threshold=0.99, max_consecutive_skips=3)
    lock = threading.Lock()
    failed = threading.Event()

    class Endpoint(PolicyServer):
        def reset(self):
            self.save_initial_observation = True
            return super().reset()

        def predict(self, observation):
            with lock, probe.use(self.intervention):
                if getattr(self, "save_initial_observation", True):
                    import hashlib

                    first = out / "online_initial_observations" / self.intervention
                    first.mkdir(parents=True, exist_ok=True)
                    index = len(list(first.glob("*.json")))
                    row = dict(
                        prompt=observation.get("prompt"),
                        state=observation.get("state"),
                        input_sha256=hashlib.sha256(json.dumps(observation, sort_keys=True).encode()).hexdigest(),
                    )
                    (first / f"{index:03d}.json").write_text(json.dumps(row))
                    self.save_initial_observation = False
                return super().predict(observation)

    def serve(mode, port):
        try:
            endpoint = Endpoint(engine, server.cfg, readiness_token=f"42960-write-diagnostic-{mode}")
            endpoint.intervention = mode
            endpoint.run(host="127.0.0.1", port=port)
        except BaseException:
            failed.set()
            raise

    for mode, port in (("full", plan["port_full"]), ("write_off", plan["port_write_off"])):
        threading.Thread(target=serve, args=(mode, port), daemon=True).start()
    while not (out / "STOP_SERVER").exists() and not failed.wait(2):
        pass
    if failed.is_set():
        raise RuntimeError("Diagnostic endpoint failed")
