"""Process-local camera-only omission while a verified client action deque is nonempty."""
import atexit
import json
from pathlib import Path

MARKER = "_openwam_capture_omitted_v1"


def install_observation_gate(base_task, should_skip, stats):
    original = base_task.get_obs
    def get_obs(env, *args, **kwargs):
        # Every upstream get_obs/_update_render/proprio call remains in place.
        stats["get_obs"] += 1
        if not should_skip(env):
            stats["full_observations"] += 1
            return original(env, *args, **kwargs)
        assert env.eval_video_path is None
        unsupported = [key for key in ("third_view", "depth", "pointcloud", "mesh_segmentation", "actor_segmentation")
                       if env.data_type.get(key, False)]
        if unsupported:
            # Preserve behavior for any future sensor configuration.
            stats["full_observations"] += 1
            return original(env, *args, **kwargs)
        cameras = env.cameras
        prior = {name: cameras.__dict__.get(name) for name in ("update_picture", "get_rgb")}
        owned = {name: name in cameras.__dict__ for name in prior}
        cameras.update_picture = lambda: None
        # No stale image is returned. Buffered step must never inspect these values.
        cameras.get_rgb = lambda: {name: {"rgb": None} for name in cameras.get_config()}
        try:
            observation = original(env, *args, **kwargs)
        finally:
            for name in prior:
                if owned[name]: setattr(cameras, name, prior[name])
                else: delattr(cameras, name)
        observation[MARKER] = True
        stats["skipped_observations"] += 1
        return observation
    base_task.get_obs = get_obs


def install_production(evidence_directory):
    import openwam2robotwin_interface as interface
    from benchmarks.robotwin import eval_policy_wrapper as wrapper
    context = {"model": None}
    stats = dict(get_obs=0, full_observations=0, skipped_observations=0, buffered_actions=0)
    original_init = interface.ModelClient.__init__
    def initialize(model, *args, **kwargs):
        original_init(model, *args, **kwargs)
        assert hasattr(model, "_local_actions"), "Camera gate requires validated chunk transport"
        context["model"] = model
    interface.ModelClient.__init__ = initialize
    def eligible(env):
        model = context["model"]
        return (model is not None and bool(model._local_actions)
                and not env.eval_success and env.eval_video_path is None
                and str(env.get_instruction()) == model._task_description)
    original_eval = interface.eval
    def evaluate(env, model, observation):
        if observation.get(MARKER):
            assert context["model"] is model and eligible(env), "Omitted image reached a replan/reset boundary"
            before = len(model._local_actions)
            result = original_eval(env, model, observation)
            assert len(model._local_actions) == before - 1, "Buffered action contract changed"
            stats["buffered_actions"] += 1
            if stats["buffered_actions"] == 1:
                print("CAMERA_CAPTURE_FIRST_BUFFERED_ACTION", flush=True)
            if stats["buffered_actions"] % 256 == 0:
                save()
            return result
        return original_eval(env, model, observation)
    interface.eval = evaluate
    bootstrap = wrapper.bootstrap_robotwin_module
    def install(*args, **kwargs):
        module = bootstrap(*args, **kwargs)
        from envs._base_task import Base_Task
        install_observation_gate(Base_Task, eligible, stats)
        return module
    wrapper.bootstrap_robotwin_module = install
    directory = Path(evidence_directory)
    record = dict(mode="skip_camera_capture_inside_verified_chunk32",
                  preserves="get_obs, scene/wrist sync, lighting RNG, actual proprio, physics, planning, success, budgets",
                  omitted="camera take_picture and Color readback only on buffered steps")
    (directory / "camera_gate_selection.json").write_text(json.dumps(record, indent=2) + "\n")
    def save():
        (directory / "camera_gate_counts.json").write_text(json.dumps(stats, indent=2) + "\n")
    atexit.register(save)
    print("CAMERA_CAPTURE_GATE_ENABLED", json.dumps(record), flush=True)
