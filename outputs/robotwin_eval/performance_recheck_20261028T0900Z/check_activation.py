"""Read-only full client constructor check: never reset/predict a live session."""
from pathlib import Path
import json
import sys
import yaml
ROOT=Path('/remote-home/jinminghao/OpenWAM');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks/robotwin'),str(HERE)]
from activate_transport import install_if_selected
policy=ROOT/'outputs/robotwin_eval/fullbackward32000_supplement70_4090_20261028T035728Z/jobs/demo_clean/blocks_ranking_rgb/attempt1/policy.yml'
cfg=yaml.safe_load(policy.read_text());evidence=HERE/'activation_constructor_smoke';evidence.mkdir(exist_ok=True)
install_if_selected(policy,evidence)
import openwam2robotwin_interface as interface
client=interface.get_model(cfg)
try:
    assert client._ws_url=='ws://127.0.0.1:'+str(cfg['port']+1000)
    assert client._client.ping().get('executable_action_chunk_v1')==dict(horizon=32)
finally:client._client.close()
(evidence/'result.json').write_text(json.dumps(dict(passed=True,
    scope='Read-only full native activation/ModelClient constructor; no reset or prediction on a live session',
    port=cfg['port'],gateway_port=cfg['port']+1000))+'\n')
print('Registry/client constructor identity and remapping smoke passed; no live session reset')
