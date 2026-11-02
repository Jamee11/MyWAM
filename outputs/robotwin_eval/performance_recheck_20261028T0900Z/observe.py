"""Process-local, buffered timings. Forward each original call exactly once."""
import atexit
import functools
import json
import time
from collections import defaultdict
from pathlib import Path


def install(directory):
    import openwam2robotwin_interface as interface
    from benchmarks.utils.transport import WSPolicyClient
    from envs._base_task import Base_Task

    path = Path(directory) / 'performance_observer.jsonl'
    events = []
    sums = defaultdict(lambda: [0, 0.0, 0.0])
    started = time.monotonic()
    last_flush = started

    def flush():
        nonlocal last_flush
        if not events:
            return
        record = dict(wall=time.time(), elapsed=time.monotonic()-started,
                      pid=__import__('os').getpid(), totals=dict(sums), events=list(events))
        with path.open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        events.clear()
        last_flush = time.monotonic()

    def timed(owner, name, event):
        original = getattr(owner, name)
        @functools.wraps(original)
        def call(*args, **kwargs):
            begin = time.perf_counter()
            result = original(*args, **kwargs)
            seconds = time.perf_counter()-begin
            row = [event, seconds]
            if event == 'rpc':
                row.append(result.get('latency_ms'))
            events.append(row)
            total = sums[event]
            total[0] += 1
            total[1] += seconds
            total[2] = max(total[2], seconds)
            if event == 'action' and time.monotonic()-last_flush >= 30:
                flush()
            return result
        setattr(owner, name, call)

    timed(Base_Task, 'get_obs', 'observation')
    timed(Base_Task, 'take_action', 'action')
    timed(Base_Task, '_update_render', 'render_update_inclusive')
    timed(WSPolicyClient, 'predict', 'rpc')
    timed(WSPolicyClient, 'predict_once', 'rpc_once')
    timed(interface.client, 'encode_numpy_b64', 'png_encode')
    timed(interface.ModelClient, 'step', 'client_step_inclusive')
    atexit.register(flush)
    print('PERFORMANCE_OBSERVER', str(path), flush=True)
