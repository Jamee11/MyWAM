# PatchWAM ramp100 step20000 最终快速评测

完成时间：2026-10-30T00:40:44Z。100条件（50任务×双模式）每条件30条，共3000条，2093成功，69.7667%。逐条件重新读取原生client.log、runtime结果、run.json退出码与checkpoint SHA，100/100全部通过，证据final_independent_audit.json。历史GPU4一次free1437<1536保护中断0条，已完整重跑通过，历史保留、不混partial。

|范围|成功/条数|新20000|32k同条件原30条|差值pp|官方|较官方pp|
|---|---:|---:|---:|---:|---:|---:|
|综合|2093/3000|69.77%|71.10%|-1.33|69.04%|+0.73|
|Clean|1263/1500|84.20%|89.00%|-4.80|89.36%|-5.16|
|Randomized|830/1500|55.33%|53.20%|+2.13|48.72%|+6.61|

观察：相对于32k相同任务各30条的结果，Randomized提升2.13pp，Clean下降4.8pp，整体低1.33pp。不能据此归因单一增强/模块：训练步数20k vs32k、未固定逐episode语言/输入manifest；均保留原生30指令池。32k完整100条为70.19%，不同采样数量参考，主对照使用32k原30条71.10%。

部署两完整BF16 L20模型/GPU1和4、每份5独立session；4090最多9路稳定运行，GPU4第二路被降容后不重开；其他低余量卡仅Clean。eager+cache/完整Sidecar/denoise10/chunk32/camera gate、原专家seed筛选/RT/实际EEF/250Hz/规划/成功/动作预算、无视频。精确checkpoint SHA与部署门槛见LIVE文档和实验根deployment.json。无训练/模型/数据修改。

说明：进度文件最后一次循环在2999条时保存、调度随后完成，导致旧progress.json陈旧；本轮全部原生结果重验通过后已用审计终值校正到3000、100条件、0活跃sim。旧快照保存在progress_before_final_reconciliation.json，不影响结果原始文件。

下一步：按相同条件分析Clean回撤/Randomized收益的任务分布，后续若比较训练策略需匹配训练预算；本次没有自动启动其他checkpoint。
