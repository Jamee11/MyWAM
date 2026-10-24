# 24k：调度修复后评估与官方对比

快照：2026-10-23T14:50:53.017923+00:00。每项30条；只汇总已完整条件，不将中止/partial当0。

|范围|条件数|成功/条数|本次|官方同条件|差值pp|
|---|---:|---:|---:|---:|---:|
|all|17|337/510|66.08%|68.82%|-2.75|
|demo_clean|9|226/270|83.70%|84.89%|-1.19|
|demo_randomized|8|111/240|46.25%|50.75%|-4.50|

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
|blocks_ranking_rgb|demo_clean|28/30|93.33%|96.00%|-2.67|
|blocks_ranking_rgb|demo_randomized|13/30|43.33%|47.00%|-3.67|
|blocks_ranking_size|demo_clean|18/30|60.00%|55.00%|+5.00|
|stack_blocks_three|demo_clean|27/30|90.00%|95.00%|-5.00|
|stack_blocks_three|demo_randomized|19/30|63.33%|66.00%|-2.67|
|hanging_mug|demo_clean|11/30|36.67%|49.00%|-12.33|

官方：[Issue46](https://github.com/OpenWAM-Official/OpenWAM/issues/46#issuecomment-6118279946)。官方每项100条，本次30条，不是全50任务最终平均。

累计实际执行（含中止前终态和重测）912条，当前最新attempt覆盖736条。完整17条件已逐项重验原生exit0、30个连续episode、递增seed和native scalar分母一致。

纠正：ANSI误停已修复；新增6次cannot create buffer集中GPU1/4/6，作为基础设施错误保留并重试，不算模型失败。elastic_v3读取可热更新capacity_policy，每卡上限0/1/2/3/4/5/6/7为1/2/1/1/2/1/1/0，现有13个native保留继续，新派发等其自然降到安全容量。旧控制器冻结保留真实wait状态、collector验exit0后并入；没有重启模型或本轮在跑native。渲染失败重试要求至少8192MiB并允许最多3attempt，等待后续观测确认资源异常不再出现，不宣称所有基础设施问题彻底解决。
