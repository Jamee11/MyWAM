# 42960标准RoboTwin评估：已按用户要求停止

停止并核验时间：2026-10-22T16:05:41.917766+00:00。checkpoint为原joint训练42960，不是新warmup版本。
完整验收20/100个任务域条件（9任务双域完成、另2任务Clean完成），共2000条。含8个中止条件累计2401条；未完成条件不纳入正式成功率，未启动条件72个。

|范围|完整条件|成功/总数|42960|官方同条件|差值pp|
|---|---:|---:|---:|---:|---:|
|Clean|11|1057/1100|96.09%|95.09%|+1.00|
|Randomized|9|481/900|53.44%|57.33%|-3.89|
|综合|20|1538/2000|76.90%|78.10%|-1.20|

## 完整条件逐项

|任务|Clean（官方）|Randomized（官方）|
|---|---:|---:|
|adjust_bottle|99%（90%）|50%（51%）|
|beat_block_hammer|98%（99%）|44%（55%）|
|blocks_ranking_rgb|95%（96%）|未完整完成|
|click_alarmclock|99%（99%）|66%（77%）|
|click_bell|100%（100%）|73%（75%）|
|grab_roller|100%（100%）|未完整完成|
|handover_block|98%（84%）|52%（52%）|
|place_container_plate|96%（98%）|79%（76%）|
|place_dual_shoes|87%（89%）|7%（19%）|
|place_shoe|94%（98%）|52%（51%）|
|scan_object|91%（93%）|58%（60%）|

## 中止条件：只保留进度

|任务|域|成功/已完成|
|---|---|---:|
|blocks_ranking_rgb|demo_randomized|40/66|
|grab_roller|demo_randomized|45/91|
|hanging_mug|demo_clean|7/26|
|hanging_mug|demo_randomized|4/16|
|move_can_pot|demo_clean|38/40|
|blocks_ranking_size|demo_clean|28/52|
|blocks_ranking_size|demo_randomized|3/28|
|dump_bin_bigbin|demo_clean|76/82|

## 进程与资源核验

- 冻结两个dispatcher与monitor后封存队列、完整成绩、进行中日志末段和53个进程的PID/start_ticks/PPID/PGID/SID/命令。只终止该实验的22个已核对进程组，未跨组清理未知作业。先TERM并给25秒退出宽限，残留的本实验已验证进程再KILL；详细信号记录保留在termination.json。
- 三模型服务与八路sim及其子进程均退出，原PID无活跃残留，八个218xx端口均关闭。GPU5/6各剩1MiB；GPU4/7其余约40GiB占用属于其他作业，未清理。
- 0–3卡新warmup训练保持运行；核验时microstep=1172、optimizer update=586，loss有限。
- 原checkpoint、每case原始结果、完整日志、统计和tokenizer均保留；未完成case不伪造成功率、不自动拼入后续评估。停止前完整queue/progress备份在快照内，canonical queue/progress明确标记stopped_by_user/interrupted_by_user。

## 证据与边界

- 记录目录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z/user_stop_20261022T160226Z`。
- 最终机器可读记录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z/user_stop_20261022T160226Z/final_record.json`。
- 官方对照使用本日已在线核对的issue46逐任务表，只比较相同已完成条件。完整成绩为76.90% vs78.10%，不是50任务全量结果。
- 相同20条件32k为74.90%，42960提高2pp；这仍不替代匹配训练预算的Sidecar/Vanilla归因对照。下一步评测若重启需独立新目录及固定协议，本次不自动续跑。
