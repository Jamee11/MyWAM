# PatchWAM ramp100 step24000：用户要求停测时的结果

停止时间：2026-10-30 07:20:20 UTC。因用户要求释放GPU给新完整训练checkpoint，提前停止；这是**部分评测**，不是100条件终测。权威快照：`outputs/robotwin_eval/patchwam_ramp100_24000_20261030/stopped_summary.json`、`comparison_before_user_stop.json`、`state_before_user_stop.json`。

## 已完成数据

按相同30条协议，完整完成56/100个条件，共成功1177/1680，阶段性成功率 **70.06%**。Clean 779/930（83.76%）；Randomized 398/750（53.07%）。停止时另有9个条件在跑、35个排队；观察到的1796条中有116条属于未完成条件，已保留原始日志与计数，但**不进入成功率**。停止后未续跑24k。

|相同56条件|PatchWAM24k|先前isolation warmup24k|先前full-backward warmup24k|
|---|---:|---:|---:|
|综合|70.06% (1177/1680)|65.83% (1106/1680)|65.06% (1093/1680)|
|Clean|83.76% (779/930)|86.77% (807/930)|85.48% (795/930)|
|Randomized|53.07% (398/750)|39.87% (299/750)|39.73% (298/750)|

同条件差值：相对isolation warmup24k综合+4.23个百分点、Clean−3.01、Randomized+13.20；相对full-backward warmup24k综合+5.00、Clean−1.72、Randomized+13.33。对PatchWAM20k同条件为+2.38个百分点；对full-backward32k原始30条同条件为+0.36。官方同条件均值66.98%，当前子集高3.08个百分点。各对照的来源/逐条件结果保留在`comparison_before_user_stop.json`及旧checkpoint的最终结果文件中。

## 解释边界

当前优势集中在Randomized，Clean略低；任务集合只覆盖已完成的31个Clean和25个Randomized条件，不能外推全部100条件。不同checkpoint的专家候选池/指令随机流未逐episode冻结，因此是同条件集合比较，不是逐条输入完全配对，也不构成统计显著性结论。停止时的活动rollout没有拼入完成条件SR。

原评测使用RoboTwin固定环境、专家筛seed、动作/RT/成功判据和每条件30条；无视频、eager+DIT cache，native chunk与生产camera gate。模型服务和仿真均按PID身份定向停止，已有条件日志/结果均保留；DiT4DiT、训练和其他GPU进程未改动。新21480评测独立使用checkpoint归档自带的eager/cache-off eval覆盖，不复用本次partial成绩。
