# 当前full-backward32k与isolation warmup24k：固定39条件对照

2026-10-25T15:26:59.519481+00:00。固定上次进度答复的39个完整条件（21Clean/18Random），双方各30条；不可用此前isolation全部100条件68.20%与当前39条件74.36%混比。

|范围|当前full-backward32k|此前isolation24k|差值pp|同run full-backward24k|
|---|---:|---:|---:|---:|
|Clean|87.94%|85.71%|+2.22|84.76%|
|Randomized|58.52%|54.26%|+4.26|52.22%|
|综合|74.36%|71.20%|+3.16|69.74%|

|Task|Mode|当前32k|isolation24k|差值pp|
|---|---|---:|---:|---:|
|adjust_bottle|demo_clean|100.00%|90.00%|+10.00|
|adjust_bottle|demo_randomized|53.33%|50.00%|+3.33|
|beat_block_hammer|demo_clean|100.00%|93.33%|+6.67|
|beat_block_hammer|demo_randomized|46.67%|26.67%|+20.00|
|blocks_ranking_rgb|demo_clean|100.00%|93.33%|+6.67|
|blocks_ranking_rgb|demo_randomized|53.33%|43.33%|+10.00|
|blocks_ranking_size|demo_clean|53.33%|60.00%|-6.67|
|blocks_ranking_size|demo_randomized|20.00%|13.33%|+6.67|
|click_alarmclock|demo_clean|100.00%|100.00%|+0.00|
|click_alarmclock|demo_randomized|80.00%|76.67%|+3.33|
|handover_block|demo_clean|93.33%|73.33%|+20.00|
|handover_block|demo_randomized|26.67%|60.00%|-33.33|
|hanging_mug|demo_clean|30.00%|36.67%|-6.67|
|hanging_mug|demo_randomized|16.67%|3.33%|+13.33|
|open_laptop|demo_clean|96.67%|100.00%|-3.33|
|open_laptop|demo_randomized|73.33%|46.67%|+26.67|
|open_microwave|demo_clean|60.00%|23.33%|+36.67|
|open_microwave|demo_randomized|6.67%|0.00%|+6.67|
|place_bread_basket|demo_clean|93.33%|86.67%|+6.67|
|place_bread_basket|demo_randomized|76.67%|70.00%|+6.67|
|place_can_basket|demo_clean|86.67%|86.67%|+0.00|
|place_cans_plasticbox|demo_clean|100.00%|100.00%|+0.00|
|place_cans_plasticbox|demo_randomized|73.33%|90.00%|-16.67|
|place_object_basket|demo_clean|96.67%|96.67%|+0.00|
|place_shoe|demo_clean|96.67%|100.00%|-3.33|
|place_shoe|demo_randomized|63.33%|70.00%|-6.67|
|put_bottles_dustbin|demo_clean|80.00%|90.00%|-10.00|
|put_bottles_dustbin|demo_randomized|46.67%|40.00%|+6.67|
|shake_bottle|demo_clean|100.00%|100.00%|+0.00|
|shake_bottle|demo_randomized|93.33%|93.33%|+0.00|
|shake_bottle_horizontally|demo_clean|100.00%|100.00%|+0.00|
|stack_blocks_three|demo_clean|93.33%|90.00%|+3.33|
|stack_blocks_three|demo_randomized|73.33%|63.33%|+10.00|
|stack_blocks_two|demo_clean|100.00%|100.00%|+0.00|
|stack_blocks_two|demo_randomized|86.67%|86.67%|+0.00|
|stack_bowls_three|demo_clean|73.33%|86.67%|-13.33|
|stack_bowls_three|demo_randomized|73.33%|66.67%|+6.67|
|stack_bowls_two|demo_clean|93.33%|93.33%|+0.00|
|stack_bowls_two|demo_randomized|90.00%|76.67%|+13.33|

阶段综合+3.16pp，Random+4.26pp，Clean+2.22pp；仍为39条件子集，训练步数32k vs24k、机制、seed/指令/RNG等未配对，不能外推全100或作为warmup机制纯因果证明。
