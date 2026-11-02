"""Independent render-error audit; no success/result mutation."""
from pathlib import Path
from datetime import datetime,timezone
import json
O=Path(__file__).resolve().parent;ROOTS=[O.parent/'sidecar21480_expanded_ablations_20261102',O.parent/'sidecar21480_all_off_20261102']
MARKERS=(b'OIDN Error:',b'ErrorDeviceLost')
def condition_key(j):return '|'.join((j['task'],j['domain'],j['kind']))
def failed_renderer(job):
 # Any exhausted native attempt lacks a valid result; rerun full10, not a policy failure.
 # A normal completed rollout with success=False has status=complete and is unaffected.
 return job['status']=='failed' and bool(job.get('attempts'))
def scan_completed():
 bad=[];checked=0
 for root in ROOTS:
  state=json.load(open(root/'state.json'))
  for j in state['jobs']:
   if j['kind']=='manifest' or (j['status']!='complete' and not failed_renderer(j)):continue
   if j.get('imported_reference'):log=Path(j['result']['source_result']).parent/'client.log'
   else:log=Path(j['attempts'][-1]['output'])/'client.log'
   text=log.read_bytes();checked+=j['status']=='complete';counts={m.decode():text.count(m) for m in MARKERS+(b'cannot create buffer',b'out of memory') if m in text}
   if counts or j['status']=='failed':bad.append(dict(key=condition_key(j),task=j['task'],domain=j['domain'],kind=j['kind'],source_root=str(root),source_status=j['status'],log=str(log),markers=counts,disposition='Quarantine original result; rerun whole sealed10 after phase1, preserving original artifact'))
 report=dict(utc=datetime.now(timezone.utc).isoformat(),checked_completed_conditions=checked,quarantined_conditions=len(bad),quarantined_completed_conditions=sum(r['source_status']=='complete' for r in bad),failed_conditions_to_repair=sum(r['source_status']=='failed' for r in bad),findings=bad,policy='Any explicit denoiser OIDN Error or Vulkan ErrorDeviceLost quarantines completed attempts; exit0 is insufficient. Warning-only driver messages are not matched.')
 f=O/'quarantined_conditions.tmp';f.write_text(json.dumps(report,indent=2)+'\n');f.replace(O/'quarantined_conditions.json');return report
