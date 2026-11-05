# PatchWAM42960 完全关闭Sidecar：50任务速测

用户2026-11-05授权：在同一个综合76.02%的最终10epoch权重上测试全部关闭Sidecar，并尽量并行启动4090仿真。

checkpoint：outputs/sidecar/aloha50_patchwam_c2r_ramp100_10epochs_20261027T144939Z/F_patchwam_ramp100/runs/2026-10-27_14-57-15/checkpoint_step_42960.safetensors。

实验：50任务×Clean/Randomized×30条=3000条；独立目录outputs/robotwin_eval/patchwam42960_sidecaroff_all50_30ep_20261105。保留在跑Refine-Off及Piper；只释放原Full21480补100条已独立审计完成的闲置模型PID4113133，结果日志均保留。

干预复用既有19组诊断的真正Sidecar-Off实现：加载同一联合训练权重后arch.sidecar_enabled=False，直接走父类主干forward，全部Sidecar Init/Read/Refine/Heads/Write不执行。不是token reset、不是仅关闭Write，也不是重新加载原版baseline。

推理/评测协议保持：Eager、DiT cache .99/max3、denoise10、sync32、384×320三视图、原生专家筛seed/成功判据/RT/动作预算、chunk通信与中间采集优化、无视频。旧near/cache近似/历史语言RNG未冻结边界继续保留，不声称三组逐episode严格配对。

L20主体GPU2、T5 GPU4，32独立会话43800–43831；4090动态6144MiB空余起步、32调度线程，并按容量继续派发。32会话为容量，不是声称32个sim实际同时推进。各会话action buffer隔离、模型推理串行锁沿用。

部署复用同权重RAM缓存和原Full canonical动作：两个fixture先验证Full canonical完全一致，再验证Off重复一致、Sidecar调用数0、恢复Full动作一致、参数版本未变，并记录Off与Write-Off动作差。部署通过后自动启动隧道与原native30调度。已有旧脚本直接复用，不改共享模型/训练/官方任务/Aloha默认配置。

状态：launch_status.json；实际进程/完整条件：state.json；成绩及相同已完成任务与Full差异：comparison.json；固定输入证据replica0/eager_validation/result.json。启动成功及最终结果以后续实际验收为准。

下一步：同任务对比Full、Refine-Off、Sidecar-Off的Clean/Randomized及逐任务成功率；不把小样本差异直接当因果确定结论。


2026-11-05T02:18:22.412621+00:00：实际部署验收通过并启动4090调度。当前17路新仿真进程，32独立session容量；Full两fixturecanonical动作一致，Off重复一致且所有Sidecar调用数0，Off与Write-Off固定输入动作maxabs0、恢复Full一致、参数版本未变。模型PID63734（GPU2/T5 GPU4）、remote controller3724605，尚无完整30条条件成绩。先前加载未验事项在本服务固定输入范围解决；完整闭环/并发峰值及最终SR仍待。Piper与Refine-Off收尾继续。
