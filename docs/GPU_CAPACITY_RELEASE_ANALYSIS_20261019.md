# L20评测资源与释放整卡分析

核对时间：2026-10-19T12:49:15.548797+00:00，只读分析，未停止/迁移进程。以下来自12:47 UTC实时NVML、queue状态和原版native run。单位GiB，显存随场景波动。

|GPU|整卡已用|空闲|当前主要用途|
|---|---:|---:|---|
|0|35.0|9.6|32k模型+2路Sidecar sim|
|1|35.4|9.2|32k模型+1路Sidecar sim，另有约6.2GiB非本次占用|
|2|34.6|9.9|原版baseline模型约24.1GiB+2路Sidecar sim|
|3|37.1|7.5|5路Sidecar sim+1路原版native sim|
|4|35.8|8.7|32k模型+2路Sidecar sim|
|5|35.0|9.6|32k模型+2路Sidecar sim|
|6|36.0|8.5|32k模型+2路Sidecar sim|
|7|37.6|6.9|32k模型+2路Sidecar sim|

GPU1本次评测约29.1GiB；其他卡当前进程数/显存与已知模型及sim布局吻合，但容器与NVML PID命名空间不同，不能无条件把未来所有新增占用当本次或终止未知进程。

优先释放GPU2：它没有Sidecar模型，结束原版额外传统native复核可释放约24GiB（同时释放GPU3该原版sim约6GiB），保留6份Sidecar模型。预选5任务双域100条的轻量官方复测已全部完成，原始成绩保留；额外native整队当前仍在place_shoe/Randomized，已有双鞋Clean92、双鞋Randomized12、单鞋Clean96，因此不能称native整队已全部完成。停止将影响这部分额外复核的后续结果，而不是已完成的manifest复测。

GPU2上还有handover_mic/Clean约78/84、press_stapler/Clean约38/39（成功/完成），不能只停baseline就称整卡已空。应等当前task/mode到100条再在任务边界迁移/停止这些worker，不能将未完成episode拼接。若减少这两路，Sidecar并发18→16，名义并发减少11.1%；真实吞吐变化需实测，不能保证速度只差11%。安全迁移到其他卡可能进一步降低影响，但现有卡余量多为7–10GiB，不能承诺两路都能无风险容纳，需释放原版sim后重新测峰值。

不优先释放GPU3：没有模型但承载5路Sidecar仿真，直接清空会影响较多任务。其他六张卡均有一份Sidecar模型，直接释放会损失该服务的3个endpoint及相关sim，重新分配成本更大。GPU1还存在其他namespace占用，停止自己的作业也不能承诺整卡完全空闲。

本次结论是资源规划建议，未执行停卡、停任务或改调度。若后续释放整卡，需核验PID归属、在任务边界调整调度、保留原始日志，并更新expanded_queue56与recovery_click_bell的GPU候选，防止恢复作业再次占用被预留的卡。


### 2026-10-19T12:56:44.550636+00:00 原版 baseline 冗余队列收尾

用户指出预定原版异常任务复测已完成。确认 baseline_official_manifest_deadline_20261018/run.json 为 complete；另一条传统 native 串行复核队列仍在运行，属于调度收尾遗漏。本次停止该冗余控制器、当前 place_shoe/Randomized 客户端及其 ffmpeg、19400 原版服务及编译子进程，共 10 个进程；所有已有结果和日志保留，native run.json 明确标为 stopped_redundant，不将未完成结果视为完整评测。详细进程与显存证据：outputs/robotwin_eval/rebuild_official_20261018/official_diagnostic100/redundant_queue_stop_20261019.json。

停止前后 GPU2 用量 38144→13459MiB，释放 24685MiB（约24.1GiB）；GPU3 用量 41341→35567MiB，释放5774MiB（约5.6GiB）。核验32k的6个模型服务、18个评测worker均存活，未停止它们；GPU2仍有Sidecar仿真，不称整卡已空闲。后续仅推进32k既定队列及指定失败恢复，额外原版复核不再继续。
