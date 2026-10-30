# RoboTwin三组实验组会汇总

冻结快照时间：2026-10-22T03:47:56.942925+00:00（UTC）。数据读取原生run.json与队列，所有正式成功率均只统计exit0且完整100条的条件。

实验：Baseline10740为自训练原版OpenWAM，Sidecar32000为中期checkpoint，Sidecar42960为另一台机器上的最终checkpoint。未与旧环境官方权重复测或其他checkpoint拼接。

## 1. 各实验当前完整条件汇总

每行官方均值只取该行实验已完成的同一批task/setting。三个实验覆盖范围不同，本表的不同实验均值不能直接排名。综合=已完成条件的总成功条数/总评测条数；两域完成数量不等时，不是Clean和Randomized均值的简单平均。

|实验|范围|完整条件|成功/总条数|成功率|官方相同条件|差值pp|
|---|---|---:|---:|---:|---:|---:|
|Baseline10740|Clean|10|891/1000|89.10%|87.90%|+1.20|
|Baseline10740|Randomized|9|460/900|51.11%|56.78%|-5.67|
|Baseline10740|Overall|19|1351/1900|71.11%|73.16%|-2.05|
|Sidecar32000|Clean|50|4383/5000|87.66%|89.36%|-1.70|
|Sidecar32000|Randomized|50|2265/5000|45.30%|48.72%|-3.42|
|Sidecar32000|Overall|100|6648/10000|66.48%|69.04%|-2.56|
|Sidecar42960|Clean|10|957/1000|95.70%|94.60%|+1.10|
|Sidecar42960|Randomized|7|401/700|57.29%|60.29%|-3.00|
|Sidecar42960|Overall|17|1358/1700|79.88%|80.47%|-0.59|

官方全50任务基准：Clean89.36%，Randomized48.72%，综合69.04%。仅Sidecar32000当前可以直接与这个全量均值比较。

## 2. 三者共同完成条件：横向比较

共同8个条件，全部每条件100条。任务集合：adjust_bottle, beat_block_hammer, click_alarmclock, handover_block。这些任务属于当前交集，不代表全部50任务。

|范围|条件数|官方|Baseline10740|Sidecar32000|Sidecar42960|
|---|---:|---:|---:|---:|---:|
|Clean|4|93.00%|97.75%|96.50%|98.50%|
|Randomized|4|58.75%|45.50%|43.75%|53.00%|
|Overall|8|75.88%|71.63%|70.13%|75.75%|

|Task|Setting|官方|Baseline10740（Δpp）|Sidecar32000（Δpp）|Sidecar42960（Δpp）|
|---|---|---:|---:|---:|---:|
|adjust_bottle|Clean|90%|96% (+6)|97% (+7)|99% (+9)|
|adjust_bottle|Randomized|51%|27% (-24)|37% (-14)|50% (-1)|
|beat_block_hammer|Clean|99%|100% (+1)|98% (-1)|98% (-1)|
|beat_block_hammer|Randomized|55%|51% (-4)|33% (-22)|44% (-11)|
|click_alarmclock|Clean|99%|100% (+1)|98% (-1)|99% (+0)|
|click_alarmclock|Randomized|77%|67% (-10)|65% (-12)|66% (-11)|
|handover_block|Clean|84%|95% (+11)|93% (+9)|98% (+14)|
|handover_block|Randomized|52%|37% (-15)|40% (-12)|52% (+0)|

## 3. 50任务完整逐项表

单元格为成功率（相对官方差值，单位pp）。所有数值均来自完整100条；“—”表示尚未完成或尚未评测，不表示0%。真实0%会明确写0%。

### Clean

|Task|官方|Baseline10740（Δpp）|Sidecar32000（Δpp）|Sidecar42960（Δpp）|
|---|---:|---:|---:|---:|
|adjust_bottle|90%|96% (+6)|97% (+7)|99% (+9)|
|beat_block_hammer|99%|100% (+1)|98% (-1)|98% (-1)|
|blocks_ranking_rgb|96%|—|95% (-1)|95% (-1)|
|blocks_ranking_size|55%|—|62% (+7)|—|
|click_alarmclock|99%|100% (+1)|98% (-1)|99% (+0)|
|click_bell|100%|—|100% (+0)|100% (+0)|
|dump_bin_bigbin|99%|—|91% (-8)|—|
|grab_roller|100%|—|100% (+0)|—|
|handover_block|84%|95% (+11)|93% (+9)|98% (+14)|
|handover_mic|90%|—|92% (+2)|—|
|hanging_mug|49%|49% (+0)|33% (-16)|—|
|lift_pot|100%|—|100% (+0)|—|
|move_can_pot|100%|—|100% (+0)|—|
|move_pillbottle_pad|96%|—|96% (+0)|—|
|move_playingcard_away|97%|—|99% (+2)|—|
|move_stapler_pad|76%|—|72% (-4)|—|
|open_laptop|93%|—|92% (-1)|—|
|open_microwave|82%|72% (-10)|61% (-21)|—|
|pick_diverse_bottles|77%|—|78% (+1)|—|
|pick_dual_bottles|85%|—|92% (+7)|—|
|place_a2b_left|87%|—|94% (+7)|—|
|place_a2b_right|89%|—|97% (+8)|—|
|place_bread_basket|92%|—|90% (-2)|—|
|place_bread_skillet|92%|95% (+3)|94% (+2)|—|
|place_burger_fries|97%|97% (+0)|99% (+2)|—|
|place_can_basket|87%|—|75% (-12)|—|
|place_cans_plasticbox|98%|99% (+1)|99% (+1)|—|
|place_container_plate|98%|—|97% (-1)|96% (-2)|
|place_dual_shoes|89%|—|88% (-1)|87% (-2)|
|place_empty_cup|100%|—|100% (+0)|—|
|place_fan|93%|—|89% (-4)|—|
|place_mouse_pad|90%|—|83% (-7)|—|
|place_object_basket|95%|—|88% (-7)|—|
|place_object_scale|94%|—|94% (+0)|—|
|place_object_stand|96%|—|99% (+3)|—|
|place_phone_stand|85%|—|83% (-2)|—|
|place_shoe|98%|—|92% (-6)|94% (-4)|
|press_stapler|90%|—|94% (+4)|—|
|put_bottles_dustbin|83%|—|80% (-3)|—|
|put_object_cabinet|48%|—|45% (-3)|—|
|rotate_qrcode|90%|—|85% (-5)|—|
|scan_object|93%|—|87% (-6)|91% (-2)|
|shake_bottle|100%|—|100% (+0)|—|
|shake_bottle_horizontally|100%|—|100% (+0)|—|
|stack_blocks_three|95%|—|93% (-2)|—|
|stack_blocks_two|100%|—|100% (+0)|—|
|stack_bowls_three|83%|—|76% (-7)|—|
|stack_bowls_two|96%|—|93% (-3)|—|
|stamp_seal|89%|88% (-1)|73% (-16)|—|
|turn_switch|54%|—|47% (-7)|—|

### Randomized

|Task|官方|Baseline10740（Δpp）|Sidecar32000（Δpp）|Sidecar42960（Δpp）|
|---|---:|---:|---:|---:|
|adjust_bottle|51%|27% (-24)|37% (-14)|50% (-1)|
|beat_block_hammer|55%|51% (-4)|33% (-22)|44% (-11)|
|blocks_ranking_rgb|47%|—|46% (-1)|—|
|blocks_ranking_size|14%|—|15% (+1)|—|
|click_alarmclock|77%|67% (-10)|65% (-12)|66% (-11)|
|click_bell|75%|—|71% (-4)|—|
|dump_bin_bigbin|69%|—|64% (-5)|—|
|grab_roller|38%|—|49% (+11)|—|
|handover_block|52%|37% (-15)|40% (-12)|52% (+0)|
|handover_mic|22%|—|15% (-7)|—|
|hanging_mug|10%|10% (+0)|8% (-2)|—|
|lift_pot|6%|—|16% (+10)|—|
|move_can_pot|29%|—|38% (+9)|—|
|move_pillbottle_pad|31%|—|31% (+0)|—|
|move_playingcard_away|49%|—|53% (+4)|—|
|move_stapler_pad|16%|—|16% (+0)|—|
|open_laptop|57%|—|57% (+0)|—|
|open_microwave|7%|—|3% (-4)|—|
|pick_diverse_bottles|62%|—|61% (-1)|—|
|pick_dual_bottles|73%|—|72% (-1)|—|
|place_a2b_left|30%|—|21% (-9)|—|
|place_a2b_right|30%|—|23% (-7)|—|
|place_bread_basket|74%|—|76% (+2)|—|
|place_bread_skillet|61%|48% (-13)|41% (-20)|—|
|place_burger_fries|80%|83% (+3)|70% (-10)|—|
|place_can_basket|45%|—|38% (-7)|—|
|place_cans_plasticbox|96%|99% (+3)|86% (-10)|—|
|place_container_plate|76%|—|79% (+3)|79% (+3)|
|place_dual_shoes|19%|—|10% (-9)|—|
|place_empty_cup|67%|—|77% (+10)|—|
|place_fan|40%|—|42% (+2)|—|
|place_mouse_pad|21%|—|19% (-2)|—|
|place_object_basket|38%|—|30% (-8)|—|
|place_object_scale|38%|—|37% (-1)|—|
|place_object_stand|65%|—|63% (-2)|—|
|place_phone_stand|22%|—|14% (-8)|—|
|place_shoe|51%|—|66% (+15)|52% (+1)|
|press_stapler|53%|—|46% (-7)|—|
|put_bottles_dustbin|52%|—|37% (-15)|—|
|put_object_cabinet|17%|—|13% (-4)|—|
|rotate_qrcode|7%|—|12% (+5)|—|
|scan_object|60%|—|52% (-8)|58% (-2)|
|shake_bottle|91%|—|91% (+0)|—|
|shake_bottle_horizontally|93%|—|90% (-3)|—|
|stack_blocks_three|66%|—|49% (-17)|—|
|stack_blocks_two|89%|—|82% (-7)|—|
|stack_bowls_three|73%|—|73% (+0)|—|
|stack_bowls_two|74%|—|76% (+2)|—|
|stamp_seal|29%|38% (+9)|28% (-1)|—|
|turn_switch|39%|—|34% (-5)|—|

## 4. 未完成任务当前进度

只列正在运行的条件；部分分母不能用于正式成功率平均。其余未启动条件均留空，不视为失败。

|实验|Task|Setting|成功/观察完成|
|---|---|---|---:|
|Baseline10740|open_microwave|Randomized|2/84|
|Sidecar42960|place_dual_shoes|Randomized|6/86|
|Sidecar42960|blocks_ranking_rgb|Randomized|27/44|
|Sidecar42960|click_bell|Randomized|63/83|
|Sidecar42960|grab_roller|Clean|66/66|
|Sidecar42960|grab_roller|Randomized|3/4|
|Sidecar42960|blocks_ranking_size|Clean|17/30|
|Sidecar42960|blocks_ranking_size|Randomized|1/13|
|Sidecar42960|dump_bin_bigbin|Clean|2/2|

## 5. 组会解释边界与来源

- Baseline仅选择32k相对官方低≥10pp的十任务，存在任务选择偏差；当前还缺一个Randomized完整条件。Baseline10740与Sidecar的训练预算和自有归一化统计不同，不能把差值当作纯Sidecar结构因果效应。
- Sidecar两checkpoint采用完整Sidecar、10步去噪、32动作chunk、FFN局部compile、cache .99/max3；Baseline采用原版完整MoT compile与同cache。官方RoboTwin环境、RT、上游专家筛seed、动作预算和成功判据沿用。
- 42960在另一台机器运行，硬件/底层图形栈与并发顺序不同；即使同任务同setting，accepted seeds、指令、policy RNG也非全部逐例严格配对。
- Baseline后续接续任务关闭视频，早先任务保留；32k与42960的视频设置沿原批次。这里报告成功率，不据其wall time作纯模型速度结论。
- 官方逐任务值已与本地封存issue46 scores全部100项核对相同，并在2026-10-22重新浏览官方原帖。
- [官方逐任务发布数据](https://github.com/OpenWAM-Official/OpenWAM/issues/46#issuecomment-6118279946)；[42960实时文档](ROBOTWIN_SIDECAR42960_LIVE_20261020.md)；[32k完整报告](ROBOTWIN_SIDECAR32K_FINAL_RESULTS_20261021.md)。

- 原始来源路径和SHA256、逐条件完整验收证据见 `outputs/robotwin_eval/group_meeting_three_ckpts_20261022/summary.json`。Excel与UTF-8 CSV同目录。
