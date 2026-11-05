import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import types

import numpy as np
import pytest

from benchmarks.robotwin.native_episode_resume import install, read_prefix, verify_provenance

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / '.external/robotwin-official-20261018/RoboTwin/script/eval_policy.py'


class StopSimulation(BaseException):
    pass


class Env:
    def __init__(self, stop=None):
        self.stop = stop
        self.trace = []
        self.cache_events = []

    def setup_demo(self, now_ep_num, seed, **kwargs):
        if self.stop is not None and self.test_num == self.stop:
            raise StopSimulation()
        np.random.seed(seed)
        random.random()  # Represents Python RNG consumption during scene setup.
        self.seed, self.episode = seed, now_ep_num
        self.plan_success = seed % 7 != 0
        self.eval_success = False
        self.take_action_cnt = 0
        self.step_lim = 1
        self.eval_video_path = None
        self.render_freq = 0

    def play_once(self):
        return {'info': {'seed': self.seed}}

    def check_success(self):
        return self.plan_success

    def close_env(self, clear_cache=False):
        self.cache_events.append(clear_cache)

    def set_instruction(self, instruction):
        self.instruction = instruction

    def get_instruction(self):
        return self.instruction

    def get_obs(self):
        self.trace.append((self.seed, self.instruction, self.episode))
        return {}


def native_module():
    tree = ast.parse(NATIVE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_policy')
    module = types.ModuleType('native_resume_test')
    module.__file__ = str(NATIVE)
    module.np = np
    module.UnStableError = type('Unstable', (Exception,), {})
    module.pools = []
    def descriptions(task, info, count):
        module.pools.append(count)
        options = list(range(count))
        random.shuffle(options)
        return [{'unseen': [f'{info[0]["seed"]}:{x}' for x in options]}]
    module.generate_episode_descriptions = descriptions
    def evaluate(env, model, observation):
        env.take_action_cnt += 1
        env.eval_success = env.seed % 3 == 0
    module.eval_function_decorator = lambda policy, name: evaluate if name == 'eval' else lambda model: None
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NATIVE), 'exec'), module.__dict__)
    return module


def run(module, env, total=70):
    return module.eval_policy('test', env, dict(task_name='test', policy_name='test',
        task_config='demo_randomized', ckpt_setting='test', clear_cache_freq=5,
        render_freq=0), None, 100000, test_num=total, instruction_type='unseen')


def empty_prefix():
    return dict(rows=[], python_rng_state=None, legacy_rng_boundary=False,
                start_seed=100000, total=70)


@pytest.mark.parametrize("total, cut", [(70,68),(100,98)])
def test_actual_native_68_plus_2_matches_uninterrupted_seeds_instructions_and_counters(tmp_path,total,cut):
    random.seed(123)
    baseline, full = native_module(), Env()
    with redirect_stdout(io.StringIO()):
        expected = run(baseline, full,total)
    first = tmp_path/'first';first.mkdir()
    interrupted, env = native_module(), Env(stop=cut)
    initial=empty_prefix();initial["total"]=total
    install(interrupted, first, initial)
    random.seed(123)
    log = io.StringIO()
    with redirect_stdout(log), pytest.raises(StopSimulation):
        run(interrupted, env,total)
    (first/'client.log').write_text(log.getvalue()+'OIDN Error: out of memory\n')
    prefix = read_prefix(first, 100000,total)
    prefix.update(start_seed=100000, total=total)
    assert len(prefix['rows']) == cut and not prefix['legacy_rng_boundary']
    resumed = tmp_path/'resumed';resumed.mkdir()
    module, tail = native_module(), Env()
    install(module, resumed, prefix)
    random.seed(999)  # Must restore the saved state, not inherit a new process RNG.
    log = io.StringIO()
    with redirect_stdout(log):
        result = run(module, tail,total)
    (resumed/'client.log').write_text(log.getvalue())
    assert result == expected
    assert tail.trace == full.trace[cut:]
    assert len(tail.trace) == 2
    assert module.pools == [total, total]
    assert tail.cache_events == full.cache_events[-len(tail.cache_events):]
    assert len(read_prefix(resumed, 100000,total)['rows']) == total
    verify_provenance(resumed)


def test_only_completed_records_before_first_error_are_retained(tmp_path):
    (tmp_path/'client.log').write_text('Success rate: 0/1 => x, current seed: 100000\n'
        'Success rate: 1/2 => x, current seed: 100002\n'
        'OIDN Error: out of memory\nSuccess rate: 2/3 => x, current seed: 100003\n')
    prefix = read_prefix(tmp_path, 100000)
    assert [r['seed'] for r in prefix['rows']] == [100000, 100002]
    assert [r['successes'] for r in prefix['rows']] == [0, 1]
    assert prefix['legacy_rng_boundary'] is True


def test_changed_source_prefix_is_rejected(tmp_path):
    source = tmp_path/'source';source.mkdir()
    (source/'client.log').write_text('Success rate: 0/1 => x, current seed: 100000\n')
    prefix = read_prefix(source,100000);prefix.update(start_seed=100000,total=70)
    target=tmp_path/'target';target.mkdir()
    (target/'resume_prefix.json').write_text(json.dumps(prefix))
    (target/'client.log').write_bytes((source/'client.log').read_bytes())
    verify_provenance(target)
    (source/'client.log').write_text('Success rate: 1/1 => x, current seed: 100000\n')
    with pytest.raises(ValueError,match='changed'):
        verify_provenance(target)
