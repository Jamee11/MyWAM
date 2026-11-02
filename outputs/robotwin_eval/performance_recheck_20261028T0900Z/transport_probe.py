"""Isolated wire benchmark: never connects to production model endpoints."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path('/remote-home/jinminghao/OpenWAM')
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser()
p.add_argument('role', choices=['server', 'client'])
p.add_argument('--port', type=int, default=34990)
p.add_argument('--output', type=Path)
a = p.parse_args()
assert a.port == 34990, 'Diagnostic endpoint only'
if a.role == 'server':
    from websockets.sync.server import serve
    from PIL import Image
    def handler(ws):
        for message in ws:
            begin = time.perf_counter()
            data = json.loads(message)
            if data.get('type') == 'ping':
                ws.send(json.dumps(dict(type='pong')))
                continue
            for encoded in data.get('images', {}).values():
                if isinstance(encoded, str):
                    raw = encoded.split(',', 1)[-1]
                    Image.open(io.BytesIO(base64.b64decode(raw))).load()
            ws.send(json.dumps(dict(type='action', action=[0.0]*20,
                                    latency_ms=(time.perf_counter()-begin)*1000)))
    with serve(handler, '127.0.0.1', a.port, compression=None, max_size=None) as server:
        server.serve_forever()
else:
    from benchmarks.utils.transport import WSPolicyClient
    dep = ROOT/'outputs/robotwin_eval/fullbackward32000_complete100_deploy_20261028T035728Z'
    meta = json.loads((dep/'deployment.json').read_text())
    payload = json.loads(Path(meta['reference_fixtures']).read_text())[0]
    rows = []
    def lane(n):
        with WSPolicyClient(f'ws://127.0.0.1:{a.port}', timeout=30) as client:
            client.ping()
            times=[];server=[]
            for _ in range(n):
                t=time.perf_counter();response=client.predict_once(payload)
                times.append(time.perf_counter()-t);server.append(response['latency_ms']/1000)
            return dict(seconds=times, server_seconds=server)
    for concurrent in (1, 11):
        begin=time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrent) as pool:
            results=list(pool.map(lane, [32]*concurrent))
        times=[x for result in results for x in result['seconds']]
        rows.append(dict(concurrency=concurrent, requests=len(times),
                         wall_seconds=time.perf_counter()-begin,
                         mean_seconds=statistics.mean(times),
                         p95_seconds=sorted(times)[int(.95*(len(times)-1))],
                         results=results))
    a.output.write_text(json.dumps(dict(utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        scope='Synthetic transport-only, separate SSH connection; no physics/model/production queue',
        payload_bytes=len(json.dumps(payload).encode()), rows=rows), indent=2)+'\n')
    print(json.dumps({k:v for k,v in json.loads(a.output.read_text()).items() if k!='rows'}))
    for row in rows:print({k:v for k,v in row.items() if k!='results'})
