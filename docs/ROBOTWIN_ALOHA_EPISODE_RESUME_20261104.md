# Aloha补70：按完成episode恢复

记录时间：2026-11-04T09:01:19.408176+00:00。用户明确要求保留已完成记录，只补基础设施故障后的未完成条数。

## 修改与不变项

原调度只接受完整native70，失败后从原起点重跑70；这不是官方要求。新增benchmarks/robotwin/native_episode_resume.py，仅接入当前两组实验私有run_native.py/report.py；官方checkout、默认Aloha入口、模型/训练/数据均不改，旧源码保留episode_resume_backup_20261104T084343Z。

按首次明确renderer错误前完整的累计success/count/seed记录恢复，正常模型失败(0成功增量)照常保留；不保留中断rollout，也不按成功率挑结果。多个历史attempt按已完成条数选择最远前缀，同长度取较早记录。源日志字节前缀SHA、顺序计数/seed与递归来源在最终report复核。

在官方eval_policy初始化处恢复now_id、succ_seed、TASK_ENV.suc/test_num及next seed，保留test_num=70和指令池70；专家筛seed、RT、成功判据、动作预算、完整Sidecar/cache/eager/chunk均不改。新增每episode原始指令和Python RNG journal；新日志的随机状态可原样恢复。旧日志没有保存该状态，明确标记legacy_rng_boundary=true，剩余仍用官方unseen生成与原池70，不能宣称与旧不中断运行逐字一致。

## 实际部署与验证

CPU3项通过，直接调用真实官方eval_policy函数，68+2与连续70的seed/指令/计数/清缓存一致（带新RNG journal），另覆盖错误前截断、正常失败保留、源日志变动拒绝。两份report已重启增加来源校验；原调度器、模型服务及其他正常仿真不重启。

08:49 UTC仅接管四个已证明重复工作的attempt：

|实验/条件|保留原记录|实际需补|接管前重复attempt进度|
|---|---:|---:|---:|
|patchwam/put_object_cabinet/demo_clean|58|12|28|
|patchwam/stack_bowls_three/demo_randomized|42|28|24|
|full21480/move_pillbottle_pad/demo_randomized|57|13|48|
|full21480/place_a2b_left/demo_randomized|68|2|24|

相对继续当前重复attempt，减少约101条额外重复执行；不是直接改变成功率。已完成条件的结果保持原样。

## 第一组真实验收已完成

Original21480/place_a2b_left/Randomized：保留68条，只新执行episode69、70，accepted seeds100129、100131，前面的候选仍由官方专家检查决定是否接受。两条新rollout均成功，native70为36/70，原生_result.txt/exit0/完整计数/源字节SHA通过，前30+后70共100唯一seed通过；没有重跑前68条。journal确实只有2条新记录。证据live_recovery_acceptance.json与该任务attempt3。其他三组继续补齐，不能把已恢复前缀当新吞吐。

## 当前快照

|实验|已各100条条件|均值(已完成集合)|尚待实际执行rollout|
|---|---:|---:|---:|
|patchwam_ramp100_final42960_supp70_20261103|96/100|76.75%|63|
|full21480_remaining32_supp70_20261104|89/100|78.93%|289|

新机制已上线并经一组真实68+2验收，不是仅提出计划。当前ETA按剩余条数及单任务速度估计约2–3小时，新增错误可能延后；出现错误时只恢复错误前可核验完整记录。证据目录outputs/analysis/aloha_episode_resume_20261104。


2026-11-04T09:29:03.997573+00:00：当前待实际执行从17:01快照的352条降至144条（约27.7分钟）；place_a2b_left保留68补2、move_pillbottle_pad保留57补13、put_object_cabinet/Clean保留58补12均已完整完成。其余条数仍需仿真，不能复用其他seed/条件的成功率数字。按当前尾轮速度估计约1–1.5小时，新增故障可能延后。详见latest_progress.json。
