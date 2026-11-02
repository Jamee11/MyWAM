"""Opt-in experiment adapter. Does not change any default production entrypoint."""
from collections import deque
import time

KEY = 'executable_action_chunk_v1'


def chunk_predict(server, observation, horizon=32):
    """Drain exactly one existing synchronous executable buffer through WAMPolicy."""
    from openwam.deploy.executors.sync_executor import SyncInferenceExecutor
    server._init_policy()
    policy = server._policy
    executor = policy._executor
    if type(executor) is not SyncInferenceExecutor:
        raise ValueError('Chunk transport requires the synchronous executor')
    if executor._action_buffer:
        raise ValueError('Cannot mix per-action and chunk transport without reset')
    begin = time.monotonic()
    # Transport metadata must never become an engine observation/condition.
    processed = server._obs_preprocessor.preprocess({k:v for k,v in observation.items() if k != KEY})
    actions = [policy.predict_action(processed).tolist()]
    if len(executor._action_buffer) != horizon - 1:
        raise ValueError('Unexpected executable horizon; restart/reset required')
    # The executor ignores these conditions while its action buffer is nonempty.
    # Use the existing policy projection on every action, including gripper dims.
    for _ in range(horizon - 1):
        actions.append(policy.predict_action(processed).tolist())
    assert not executor._action_buffer
    elapsed = (time.monotonic()-begin)*1000
    start = server._request_count+1
    server._request_count += len(actions)
    server._total_latency += elapsed
    return dict(action=actions[0], step=start, latency_ms=elapsed,
                **{KEY: dict(actions=actions, start_step=start, count=len(actions))})


def install_server():
    from openwam.deploy.server import PolicyServer
    original = PolicyServer.predict
    original_pong = PolicyServer._pong_payload
    def predict(self, observation):
        request = observation.get(KEY)
        if request is None:
            return original(self, observation)
        if request != dict(horizon=32):
            raise ValueError('Unsupported chunk transport contract')
        return chunk_predict(self, observation)
    def pong(self):
        return {**original_pong(self), KEY: dict(horizon=32)}
    PolicyServer.predict = predict
    PolicyServer._pong_payload = pong


def install_client():
    """Keep original input/EEF/action conversion at generation boundaries."""
    import numpy as np
    import openwam2robotwin_interface as interface
    original_init = interface.ModelClient.__init__
    original_step = interface.ModelClient.step
    original_reset = interface.ModelClient.reset

    def initialize(self, *args, **kwargs):
        self._local_actions = deque()
        self._wire_consumed = 0
        original_init(self, *args, **kwargs)
        if self._debug:
            raise ValueError('Chunk experiment requires debug images disabled')
        if self._client.ping().get(KEY) != dict(horizon=32):
            raise ValueError('Server has not enabled chunk transport')
        def request(payload):
            result = self._client.predict_once({**payload, KEY: dict(horizon=32)})
            chunk = result.get(KEY)
            if not isinstance(chunk, dict) or chunk.get('count') != 32:
                raise ValueError('Missing/invalid action chunk')
            actions = np.asarray(chunk['actions'], dtype=np.float32)
            if actions.ndim != 2 or len(actions) != 32 or not np.isfinite(actions).all():
                raise ValueError('Invalid action chunk values')
            if not np.array_equal(actions[0], np.asarray(result['action'], dtype=np.float32)):
                raise ValueError('Chunk first action differs from response action')
            if chunk['start_step'] != self._wire_consumed+1 or result['step'] != chunk['start_step']:
                raise ValueError('Chunk sequence mismatch; reset required')
            self._local_actions.extend(actions[1:])
            return result
        self._client.predict = request

    def reset(self, *args, **kwargs):
        self._local_actions.clear()
        self._wire_consumed = 0
        return original_reset(self, *args, **kwargs)

    def step(self, example):
        instruction = str(example.get('lang', self._task_description))
        if instruction and instruction != self._task_description:
            self.reset(instruction)
        if not self._local_actions:
            action = original_step(self, example)
            self._wire_consumed += 1
            return action
        # The native evaluator still obtains each observation and takes each
        # action separately; success checks, physics, cameras and budgets remain.
        action = self._local_actions.popleft()
        self._step += 1
        self._wire_consumed += 1
        if self._action_indices is not None:
            action = action[self._action_indices]
        return action

    interface.ModelClient.__init__ = initialize
    interface.ModelClient.reset = reset
    interface.ModelClient.step = step
