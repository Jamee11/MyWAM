"""Activate only subsequent native processes after numerical and native pilot gates."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT=Path('/remote-home/jinminghao/OpenWAM');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from benchmarks.utils.transport import WSPolicyClient

assert json.loads((HERE/'chunk_cpu_acceptance.json').read_text())['passed']
acceptance=json.loads((HERE/'acceptance_summary.json').read_text())
assert acceptance['compat_ready_for_new_tasks']
for name in ['legacy','chunk','compat']:
    assert json.loads((HERE/f'carry_{name}.json').read_text())['passed']
assert all(row['passed'] for row in acceptance['wire'])
registry_path=HERE/'formal_transport_registry.json'
registry=json.loads(registry_path.read_text());assert not registry['enabled']

def check(item):
    port,selection=item;session=json.loads(Path(selection['session']).read_text())
    with WSPolicyClient(f"ws://127.0.0.1:{selection['gateway_port']}",timeout=30) as client:
        health=client.ping()
    assert health.get('type')=='pong' and health.get('readiness_token')==session['readiness_token']
    assert health.get('executable_action_chunk_v1')==dict(horizon=32)
    assert session['checkpoint_sha256']==registry['checkpoint_sha256']
    return dict(backend_port=int(port),gateway_port=selection['gateway_port'],identity_passed=True)

with ThreadPoolExecutor(max_workers=6) as pool:checks=list(pool.map(check,registry['ports'].items()))
main=ROOT/'outputs/robotwin_eval/fullbackward32000_supplement70_4090_20261028T035728Z'
before=json.loads((main/'elastic_v3/state.json').read_text())
record=dict(wall=time.time(),scope='Only newly launched native processes; current processes keep old transport',
            checks=checks,workers_before=[{k:w[k] for k in ['pid','gpu','task','mode'] if k in w}
                for w in before['workers'].values() if w.get('status')=='running'],
            native_source_sha256=hashlib.sha256((main/'run_native.py').read_bytes()).hexdigest(),
            protocol='existing30+70, same seed continuation/prompt pool; inference horizon32; no video, native RT/success/action budget unchanged')
registry.update(enabled=True,activated_at=time.time(),strategy='chunk32_compat_bounded2ms_stream_replies')
tmp=registry_path.with_suffix('.tmp');tmp.write_text(json.dumps(registry,indent=2)+'\n');tmp.replace(registry_path)
(HERE/'formal_bridge_activation.json').write_text(json.dumps(record,indent=2)+'\n')
print('Activated',len(checks),'gateway ports for future native tasks; no model/simulator restarted')
