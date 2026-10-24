> 最新状态：2026-10-22T16:05:41.917766+00:00 已按用户要求停止42960评测。完整成绩和未完成进度见 [停止记录](ROBOTWIN_SIDECAR42960_STOPPED_20261022.md)。下文保留停止前快照。

# RoboTwin Sidecar checkpoint评估最新核查（2026-10-22）

快照：2026-10-22T14:09:38.263864+00:00。本次重新读取原始记录与GitHub官方issue46评论，未启动/停止/修改任何训练或评测。
机器可读快照：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z/comparison_snapshot_20261022T140938Z.json`。

## 1. 42960最终checkpoint：进度与官方同条件比较

完整验收20/100条件（2000/10000条），含进行中条件已观察完成2340条；8路running、72条件queued。11个Clean条件和9个Randomized条件完成，相当于9个任务双域完整、另2任务仅Clean完整。
8个active PID现场可见，client.log均持续产生实际step/episode日志，不仅是队列status在刷新。进行中分母不进入正式成功率。

|范围|完整条件|成功/总数|42960|官方同条件|差值pp|
|---|---:|---:|---:|---:|---:|
|Clean|11|1057/1100|96.09%|95.09%|+1.00|
|Randomized|9|481/900|53.44%|57.33%|-3.89|
|综合|20|1538/2000|76.90%|78.10%|-1.20|

综合按2000条episode加权，两域完成条件数不同，不直接平均两个域的均值。不能把本综合76.90%直接对比官方全50任务69.04%。

### 全部完整条件逐项

|Task|域|42960|官方|差值pp|
|---|---|---:|---:|---:|
|adjust_bottle|Clean|99%|90%|+9|
|adjust_bottle|Randomized|50%|51%|-1|
|beat_block_hammer|Clean|98%|99%|-1|
|beat_block_hammer|Randomized|44%|55%|-11|
|blocks_ranking_rgb|Clean|95%|96%|-1|
|click_alarmclock|Clean|99%|99%|+0|
|click_alarmclock|Randomized|66%|77%|-11|
|click_bell|Clean|100%|100%|+0|
|click_bell|Randomized|73%|75%|-2|
|grab_roller|Clean|100%|100%|+0|
|handover_block|Clean|98%|84%|+14|
|handover_block|Randomized|52%|52%|+0|
|place_container_plate|Clean|96%|98%|-2|
|place_container_plate|Randomized|79%|76%|+3|
|place_dual_shoes|Clean|87%|89%|-2|
|place_dual_shoes|Randomized|7%|19%|-12|
|place_shoe|Clean|94%|98%|-4|
|place_shoe|Randomized|52%|51%|+1|
|scan_object|Clean|91%|93%|-2|
|scan_object|Randomized|58%|60%|-2|

### 正在运行的条件（仅进度，不是最终SR）

|Task|域|成功/当前完成|
|---|---|---:|
|blocks_ranking_rgb|demo_randomized|38/62|
|grab_roller|demo_randomized|39/78|
|hanging_mug|demo_clean|4/21|
|hanging_mug|demo_randomized|4/13|
|move_can_pot|demo_clean|23/25|
|blocks_ranking_size|demo_clean|23/46|
|blocks_ranking_size|demo_randomized|3/26|
|dump_bin_bigbin|demo_clean|64/69|

## 2. 32k已经完成的全50任务结果

|范围|成功/总数|32k|官方全50任务|差值pp|
|---|---:|---:|---:|---:|
|Clean|4383/5000|87.66%|89.36%|-1.70|
|Randomized|2265/5000|45.30%|48.72%|-3.42|
|综合|6648/10000|66.48%|69.04%|-2.56|

## 3. 两个Sidecar checkpoint在相同20条件上的比较

|范围|32k|42960|42960减32k pp|
|---|---:|---:|---:|
|Clean|95.00%|96.09%|+1.09|
|Randomized|50.33%|53.44%|+3.11|
|综合|74.90%|76.90%|+2.00|

当前交集上继续训练有改善，尤其Randomized；不外推为全50任务的最终42960成绩。

## 4. 自训练原版baseline10740

十任务双域2000条已全部完成；最后open_microwave Randomized为3/100。该十任务选自32k表现不足任务，存在选择偏差。

|范围|Baseline10740|官方相同十任务|差值pp|
|---|---:|---:|---:|
|Clean|89.10%|87.90%|+1.20|
|Randomized|46.30%|51.80%|-5.50|
|综合|67.70%|69.85%|-2.15|

三者共同完成的4任务为adjust_bottle、beat_block_hammer、click_alarmclock、handover_block，双域共800条/模型。

|模型|共同Clean|共同Randomized|共同综合|
|---|---:|---:|---:|
|官方|93.00%|58.75%|75.88%|
|Baseline10740|97.75%|45.50%|71.62%|
|Sidecar32k|96.50%|43.75%|70.12%|
|Sidecar42960|98.50%|53.00%|75.75%|

## 5. 证据与解释边界

- 本次核对140份原始run.json（32k100、4296020、baseline20）。42960和baseline同时核对队列、run.results及case result.json，要求exit0/complete/100条/无errors/100个不重复accepted seeds，成功数一致、native_result文件存在。32k重读全部100份原始run的complete与成功计数；其完整性细项沿用最终验收记录。
- GitHub connector本次重新读取issue46的官方逐任务表，全部50任务100项数值与2026-10-21快照一致；原始正文及解析表保存在official_issue46_snapshot_20261022.json。
- 官方发布的是Clean训练的OpenWAM-Alpha-Sim-RoboTwin-Clean2Random：每任务100条、seed0、10步去噪、默认cache、RoboTwin上游动作预算。我们的评测复用官方环境/成功判据/预算，完整Sidecar+cache与局部FFN编译；不是逐例相同accepted seeds/指令/RNG的配对复现。
- checkpoint间训练预算、normalization stats及额外模块不同，baseline也不是同更新预算完整匹配实验。交集上42960更高不能单独证明Sidecar的增量价值；runtime WriteOff和matched Vanilla仍是互补的因果检查。
- 当前主要差距仍在Randomized，42960双鞋7% vs19%、hammer44% vs55%、alarmclock66% vs77%；不能据此断言全部Randomized都差或指定某个架构部件为原因。
- 下一步继续原42960队列；完整100条才更新正式条件均值。当前没有新的重训有效性或warmup效果结论。

官方来源：[OpenWAM官方逐任务成绩](https://github.com/OpenWAM-Official/OpenWAM/issues/46#issuecomment-6118279946)。
