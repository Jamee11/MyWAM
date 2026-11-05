# Warmup2K → full 10-epoch checkpoint 21480: live RoboTwin results

Updated 2026-10-30T20:41:48.321742+00:00; status `complete`; complete conditions 100/100.

|Scope|Conditions|Successes / episodes|Current|Previous full-backward 24k|Full-backward 32k original 30|Official same subset|
|---|---:|---:|---:|---:|---:|---:|
|All|100|2245/3000|74.83%|66.13%|71.10%|69.04%|
|Clean|50|1330/1500|88.67%|85.80%|89.00%|89.36%|
|Randomized|50|915/1500|61.00%|46.47%|53.20%|48.72%|

Only complete, natively validated 30-episode conditions enter the success rate. Active partials are excluded. The checkpoint uses its archived eager/cache-off eval settings; native RoboTwin task, expert seed filtering, success criteria, RT and action budget are unchanged.
