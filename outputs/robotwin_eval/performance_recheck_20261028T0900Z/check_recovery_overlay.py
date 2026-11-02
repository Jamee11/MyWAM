"""Verify recovery reporting cannot replace completed cases or cross checkpoint seeds."""
import ast
import copy
import json
from pathlib import Path
import tempfile
ROOT=Path('/remote-home/jinminghao/OpenWAM')
source=ROOT/'outputs/robotwin_eval/fullbackward32000_supplement70_4090_20261028T035728Z/watch_complete100.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='with_recoveries')
with tempfile.TemporaryDirectory() as temporary:
    out=Path(temporary);scope=dict(OUT=out,json=json);exec(compile(ast.Module(body=[node],type_ignores=[]),'report_overlay','exec'),scope)
    merge=scope['with_recoveries']
    state=dict(status='finished_with_errors',checkpoint_sha256='a',conditions=[dict(task='t',mode='m',status='failed',continuation_seed=100036,attempts=[dict(old=True)])])
    recovery=dict(task='t',mode='m',checkpoint_sha256='a',continuation_seed=100036,status='running',attempts=[dict(new=True)])
    def write():out.joinpath('recovery_overlay.json').write_text(json.dumps(dict(conditions=[recovery])))
    write();original=copy.deepcopy(state);result=merge(state)
    assert state==original and result['status']=='running_recovery' and result['recovery_active_simulators']==1
    recovery.update(status='complete',output='audited_new_output',result=dict(successes=50));write();result=merge(state)
    assert result['status']=='complete' and result['conditions'][0]['output']=='audited_new_output'
    recovery['checkpoint_sha256']='wrong';write()
    try:merge(state)
    except AssertionError:pass
    else:raise AssertionError('Cross-checkpoint recovery accepted')
    state['conditions'][0]['status']='complete';result=merge(state)
    assert result['conditions'][0]['attempts']==state['conditions'][0]['attempts']
report=dict(passed=True,original_queue_unchanged=True,complete_case_not_overridden=True,cross_checkpoint_rejected=True,recovery_completion_counted_once=True)
Path(__file__).with_name('recovery_overlay_cpu_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
