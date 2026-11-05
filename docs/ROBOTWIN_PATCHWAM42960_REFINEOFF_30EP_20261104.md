# PatchWAM final42960 Refine-Off 完整50任务速测

2026-11-04 用户明确要求立即启动，不等待Piper完成；Piper运行及结果保持。

权重：outputs/sidecar/aloha50_patchwam_c2r_ramp100_10epochs_20261027T144939Z/F_patchwam_ramp100/runs/2026-10-27_14-57-15/checkpoint_step_42960.safetensors。
该权重Full已完整30条速测：Clean87.8667%、Randomized64.5333%、综合76.20%；扩展100条：88.02%、64.02%、76.02%。

本次50任务×Clean/Randomized×30=3000。Refine-Off复用此前诊断实现：各stage.refine注意力返回zeros_like(query), None，保留输入token残差、Read、FFN、Write，不是关闭Sidecar。独立输出 outputs/robotwin_eval/patchwam42960_refineoff_all50_30ep_20261104。

复用已验证split/诊断server、native30调度/通信/跳过chunk中间相机采集实现；不改共享模型、训练、官方任务、现有Piper或Aloha默认流程。Eager、DiT cache .99/max3、denoise10、sync32、384×320、官方专家筛seed/RT/动作预算/成功判据、无视频保持一致。

首份服务主体L20 GPU0，T5 GPU4，32独立session端口43700–43731。重用已验证RAM权重，不重复下载大文件；Full两fixture对原同权重canonical动作逐元素验收，另外Refine-Off重复输入有限/一致性检查通过后才启动仿真。SSH反向隧道沿用26199端口。

4090动态按实际空余显存6144MiB调度，32线程/会话为容量，非宣称32路同时运行；Piper活动worker不停止，已有GPU余量立即供新任务使用。错误attempt按原规则不计SR并重排。历史同权重Full与本次Refine-Off是同官方协议、相同任务条件比较；历史语言RNG未冻结且专家接受集未强制固定，不宣称逐episode严格配对。

结果看comparison.json、state.json；comparison额外汇总与原同权重Full在已完成相同条件上的差值。部署状态看launch_status.json、replica0/eager_validation/result.json、replica0/refine_off_validation.json和server.log。启动成功与最终成绩需以实际文件确认，不能将计划当结果。

下一步：确认模型和真实仿真启动；完成后按Clean/Randomized及逐任务对比Full，判断旧21480上的Refine-Off现象是否能在PatchWAM最终权重复现。


2026-11-04T18:01:15.396910+00:00：实际部署与自动调度已启动，模型PID30192主体GPU0/T5 GPU4；Full两fixture与原同权重canonical逐元素一致、Refine-Off重复一致且Full差值maxabs均0.0078125。4090 controller3096247，当前16路新仿真进程，Piper保持运行。启动审查范围通过，完整rollout/SR及并发峰值尚待；32为独立会话容量，不当作实际并发数。证据本实验replica0/eager_validation/result.json、refine_off_validation.json、launch_status.json、state.json。


2026-11-05 02:49 UTC：Refine-Off全部100条件3000条完成，2264成功=75.4667%；Clean1330/1500=88.6667%，Randomized934/1500=62.2667%。同权重Full速测76.20%（C87.8667/R64.5333），差综合-0.7333pp、Clean+0.80pp、Randomized-2.2667pp，净少22条成功，不能称关Refine整体改善。来源state.json及comparison.json；尚需最终独立原生日志复核，不将调度汇总等同独立审计。Sidecar-Off仅2完整Clean条件55/60、另在跑213条，不据早期任务子集比较全套。后续完成Sidecar-Off并按同任务三组比较。
