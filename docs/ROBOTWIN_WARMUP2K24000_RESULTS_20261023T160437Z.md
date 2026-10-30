# 24k checkpoint：30条趋势评估阶段结果

快照：2026-10-23T16:04:37.548820+00:00。完整27/100条件、810条有效；当前覆盖944条；物理1141含重测；10路在跑，历史异常13与上次相同，无最终failed条件。

|范围|条件数|成功/总数|本次|官方同条件|差值pp|
|---|---:|---:|---:|---:|---:|---:|
|all|27|549/810|67.78%|71.85%|-4.07|
|demo_clean|15|369/450|82.00%|86.80%|-4.80|
|demo_randomized|12|180/360|50.00%|53.17%|-3.17|

|Task|Mode|成功/30|本次|官方|差值pp|
|---|---|---:|---:|---:|---:|
|adjust_bottle|demo_clean|27/30|90.00%|90.00%|+0.00|
|adjust_bottle|demo_randomized|15/30|50.00%|51.00%|-1.00|
|beat_block_hammer|demo_randomized|8/30|26.67%|55.00%|-28.33|
|beat_block_hammer|demo_clean|28/30|93.33%|99.00%|-5.67|
|click_alarmclock|demo_clean|30/30|100.00%|99.00%|+1.00|
|click_alarmclock|demo_randomized|23/30|76.67%|77.00%|-0.33|
|place_shoe|demo_randomized|21/30|70.00%|51.00%|+19.00|
|place_shoe|demo_clean|30/30|100.00%|98.00%|+2.00|
|put_bottles_dustbin|demo_clean|27/30|90.00%|83.00%|+7.00|
|put_bottles_dustbin|demo_randomized|12/30|40.00%|52.00%|-12.00|
|open_microwave|demo_randomized|0/30|0.00%|7.00%|-7.00|
|open_microwave|demo_clean|7/30|23.33%|82.00%|-58.67|
|blocks_ranking_rgb|demo_clean|28/30|93.33%|96.00%|-2.67|
|blocks_ranking_rgb|demo_randomized|13/30|43.33%|47.00%|-3.67|
|blocks_ranking_size|demo_clean|18/30|60.00%|55.00%|+5.00|
|stack_blocks_three|demo_clean|27/30|90.00%|95.00%|-5.00|
|stack_blocks_three|demo_randomized|19/30|63.33%|66.00%|-2.67|
|stack_bowls_three|demo_clean|26/30|86.67%|83.00%|+3.67|
|hanging_mug|demo_clean|11/30|36.67%|49.00%|-12.33|
|hanging_mug|demo_randomized|1/30|3.33%|10.00%|-6.67|
|stack_bowls_two|demo_randomized|23/30|76.67%|74.00%|+2.67|
|stack_bowls_two|demo_clean|28/30|93.33%|96.00%|-2.67|
|handover_block|demo_clean|22/30|73.33%|84.00%|-10.67|
|handover_block|demo_randomized|18/30|60.00%|52.00%|+8.00|
|place_cans_plasticbox|demo_randomized|27/30|90.00%|96.00%|-6.00|
|stack_blocks_two|demo_clean|30/30|100.00%|100.00%|+0.00|
|open_laptop|demo_clean|30/30|100.00%|93.00%|+7.00|

差距集中在open_microwave Clean23.33% vs82%和hammer Random26.67% vs55%。仅作事后定位，排除这两项后其他25项71.20% vs72.12%，低0.92pp；不能以排除后结果代替全27项67.78% vs71.85%。

全部完整条件已再次核对真实exit0、30个连续记录、递增seed、原生scalar分母匹配。官方[Issue46](https://github.com/OpenWAM-Official/OpenWAM/issues/46#issuecomment-6118279946)每条件100条，本批30条；覆盖仍不完整，不能判定结构因果或全50任务达标。
