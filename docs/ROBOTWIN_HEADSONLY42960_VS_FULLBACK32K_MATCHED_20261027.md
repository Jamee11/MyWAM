# heads-only42960与full-backward32k同条件比较

快照：2026-10-27T17:31:39.577414+00:00。双方只统计同一任务/域、均跑满30条的条件。heads原10条件在L20，其余新增在4090；32k全部4090，混合硬件及30条小样本边界保留。

|范围|条件数|32k|heads42960|差距百分点|
|---|---:|---:|---:|---:|
|All|16|73.33%|72.92%|-0.42|
|Clean|9|87.04%|88.89%|+1.85|
|Randomized|7|55.71%|52.38%|-3.33|

|Task|Mode|32k|heads42960|差距百分点|
|---|---|---:|---:|---:|
|adjust_bottle|demo_clean|100.00%|96.67%|-3.33|
|adjust_bottle|demo_randomized|53.33%|56.67%|+3.33|
|beat_block_hammer|demo_randomized|46.67%|33.33%|-13.33|
|beat_block_hammer|demo_clean|100.00%|96.67%|-3.33|
|click_alarmclock|demo_clean|100.00%|100.00%|+0.00|
|click_alarmclock|demo_randomized|80.00%|86.67%|+6.67|
|place_shoe|demo_randomized|63.33%|63.33%|+0.00|
|place_shoe|demo_clean|96.67%|96.67%|+0.00|
|put_bottles_dustbin|demo_clean|80.00%|86.67%|+6.67|
|open_microwave|demo_clean|60.00%|76.67%|+16.67|
|blocks_ranking_rgb|demo_clean|100.00%|90.00%|-10.00|
|blocks_ranking_rgb|demo_randomized|53.33%|53.33%|+0.00|
|blocks_ranking_size|demo_randomized|20.00%|10.00%|-10.00|
|blocks_ranking_size|demo_clean|53.33%|56.67%|+3.33|
|stack_blocks_three|demo_clean|93.33%|100.00%|+6.67|
|stack_blocks_three|demo_randomized|73.33%|63.33%|-10.00|
