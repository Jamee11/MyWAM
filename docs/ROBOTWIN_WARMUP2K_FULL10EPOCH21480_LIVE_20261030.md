# Warmup2K → full 10-epoch checkpoint 21480: live RoboTwin results

Updated 2026-10-30T13:26:09.809144+00:00; status `running`; complete conditions 53/100.

|Scope|Conditions|Successes / episodes|Current|Previous full-backward 24k|Full-backward 32k original 30|Official same subset|
|---|---:|---:|---:|---:|---:|---:|
|All|53|1220/1590|76.73%|68.43%|72.96%|71.25%|
|Clean|34|912/1020|89.41%|85.69%|90.00%|89.94%|
|Randomized|19|308/570|54.04%|37.54%|42.46%|37.79%|

Only complete, natively validated 30-episode conditions enter the success rate. Active partials are excluded. The checkpoint uses its archived eager/cache-off eval settings; native RoboTwin task, expert seed filtering, success criteria, RT and action budget are unchanged.
