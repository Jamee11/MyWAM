"""Capture initial observations from sealed RoboTwin scenes without taking policy actions."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def install_capture(module):
    from benchmarks.robotwin import eval_policy_wrapper as wrapper
    from benchmarks.robotwin.openwam2robotwin_interface import _extract_eef_proprio
    from benchmarks.robotwin.prompt_template import format_prompt_for_inference
    from benchmarks.utils.client import build_payload, encode_numpy_b64
    from benchmarks.utils.rng_domain import GlobalRngDomain

    manifest = json.loads(Path(os.environ["ROBOTWIN_EPISODE_MANIFEST"]).read_text())
    original = module.eval_function_decorator
    module.eval_function_decorator = (
        lambda policy, name: (lambda args: None) if name == "get_model" else original(policy, name)
    )

    def capture(task_name, task_env, args, model, st_seed, test_num=100, video_size=None, instruction_type=None):
        rows = []
        args["eval_mode"] = True
        output = Path(os.environ["DIAGNOSTIC_OBSERVATIONS"])
        output.mkdir(parents=True, exist_ok=False)
        for entry in manifest["entries"]:
            seed = entry["seed"]
            domain = GlobalRngDomain.from_seed(seed)
            with domain.activate():
                task_env.setup_demo(now_ep_num=entry["episode"], seed=seed, is_test=True, **args)
                wrapper._initialize_eval_task_state(task_name, task_env)
                task_env.set_instruction(instruction=entry["instructions"]["unseen"])
                wrapper._canonicalize_torch_cuda(seed)
            try:
                with domain.activate():
                    obs = task_env.get_obs()
                images = obs["observation"]
                payload = build_payload(
                    head=encode_numpy_b64(images["head_camera"]["rgb"]),
                    left_wrist=encode_numpy_b64(images["left_camera"]["rgb"]),
                    right_wrist=encode_numpy_b64(images["right_camera"]["rgb"]),
                    state=_extract_eef_proprio(obs, quaternion_order="wxyz").tolist(),
                    prompt=format_prompt_for_inference(entry["instructions"]["unseen"]),
                )
                path = output / f"episode{entry['episode']}.json"
                path.write_text(json.dumps(payload))
                rows.append(
                    dict(
                        task=task_name,
                        mode=args["task_config"],
                        episode=entry["episode"],
                        seed=seed,
                        instruction=entry["instructions"]["unseen"],
                        payload=str(path),
                    )
                )
            finally:
                with domain.activate():
                    task_env.close_env()
        wrapper._write_labtasker_result(
            dict(operation="capture_initial_observations", rows=rows, manifest_hash=manifest["manifest_hash"])
        )
        return manifest["entries"][-1]["seed"] + 1, 0

    module.eval_policy = capture


if __name__ == "__main__":
    from benchmarks.robotwin import eval_policy_wrapper as wrapper

    wrapper._install_eval = install_capture
    raise SystemExit(wrapper.main())
