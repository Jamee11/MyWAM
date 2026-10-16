"""Exercise the exact shared endpoint methods with real policy/executor buffers on CPU."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np

from openwam.deploy.policy import WAMPolicy
from openwam.deploy.server import PolicyServer


def endpoints():
    source = Path(__file__).resolve().parents[2] / "benchmarks/robotwin/serve_parallel.py"
    tree = ast.parse(source.read_text())
    definition = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Endpoint")
    namespace = {"PolicyServer": PolicyServer, "lock": threading.Lock(), "args": SimpleNamespace(slots=2)}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    endpoint_type = namespace["Endpoint"]

    class Engine:
        architecture = SimpleNamespace(binary_command_dims=(), repr_contract={"representation": "eef"})
        active = 0
        peak = 0
        calls = 0

        def generate(self, conditions):
            self.active += 1
            self.peak = max(self.peak, self.active)
            time.sleep(.005)
            tag = conditions["proprio"][0]
            self.calls += 1
            self.active -= 1
            return {"actions": np.array([[tag, step] for step in (1, 2, 3)], dtype=np.float32)}

    engine = Engine()
    clients = []
    for index in range(2):
        server = endpoint_type(engine, None, readiness_token="test")
        server.slot = index
        server._policy = WAMPolicy(engine, None)
        server._obs_preprocessor = SimpleNamespace(preprocess=lambda observation: observation)
        clients.append(server)
    return engine, clients


def test_reset_and_buffer_are_private_to_each_port():
    engine, (left, right) = endpoints()
    assert left.predict({"state": [10]})["action"] == [10, 1]
    assert right.predict({"state": [20]})["action"] == [20, 1]
    left.reset()
    assert right.predict({"state": [20]})["action"] == [20, 2]
    assert left.predict({"state": [10]})["action"] == [10, 1]
    assert engine.calls == 3
    assert left._pong_payload()["pool_slot"] == 0
    assert right._pong_payload()["pool_slot"] == 1


def test_parallel_clients_serialize_shared_generation():
    engine, clients = endpoints()

    def run(index):
        return [clients[index].predict({"state": [index + 10]})["action"] for _ in range(12)]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, range(2)))
    assert engine.peak == 1
    assert all(action[0] == 10 for action in results[0])
    assert all(action[0] == 11 for action in results[1])
    assert engine.calls == 8
