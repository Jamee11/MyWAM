"""Opt-in selection read once by each newly launched formal native process."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
KEY='executable_action_chunk_v1'


def install_if_selected(policy_path,evidence_directory):
    registry=HERE/'formal_transport_registry.json'
    if not registry.exists():return
    settings=json.loads(registry.read_text())
    if not settings.get('enabled',False):return
    import yaml
    cfg=yaml.safe_load(Path(policy_path).read_text())
    backend=int(cfg['port']);selection=settings['ports'].get(str(backend))
    if selection is None:return
    assert cfg['host']=='127.0.0.1'
    assert settings['checkpoint_sha256']=='2a74859c3c36aa8d9e21af3d98e034230ca367ada0b61403c33f14f86a3adeb9'
    session=json.loads(Path(selection['session']).read_text())
    assert session['checkpoint_sha256']==settings['checkpoint_sha256']
    assert session['model_port']==backend
    from benchmarks.utils.transport import WSPolicyClient
    with WSPolicyClient(f"ws://127.0.0.1:{selection['gateway_port']}",timeout=30) as client:
        health=client.ping()
    if health.get('readiness_token')!=session['readiness_token']:
        raise ValueError('Gateway backend identity mismatch')
    assert health.get(KEY)==dict(horizon=32)
    from chunk_transport import install_client
    import openwam2robotwin_interface as interface
    install_client()
    original=interface.ModelClient.__init__
    def initialize(self,*a,**kw):
        if int(kw.get('port',8848))!=backend:raise ValueError('Unexpected policy session')
        kw['port']=selection['gateway_port']
        return original(self,*a,**kw)
    interface.ModelClient.__init__=initialize
    record=dict(backend_port=backend,gateway_port=selection['gateway_port'],
                checkpoint_sha256=settings['checkpoint_sha256'],
                registry_sha256=hashlib.sha256(registry.read_bytes()).hexdigest(),
                transport='chunk32_bounded_batch_compat_backend',
                semantics='same model horizon32; per-step native observation/physics/success/budget unchanged')
    Path(evidence_directory,'transport_bridge_selection.json').write_text(json.dumps(record,indent=2)+'\n')
    print('TRANSPORT_BRIDGE_SELECTED',json.dumps(record),flush=True)
