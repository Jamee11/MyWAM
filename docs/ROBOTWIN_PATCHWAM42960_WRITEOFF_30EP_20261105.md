# PatchWAM42960 Write-Off：完整50任务速测

2026-11-05 用户明确要求在Sidecar-Off之外追加Write-Off，每任务每域30条，尽量并行。

checkpoint：outputs/sidecar/aloha50_patchwam_c2r_ramp100_10epochs_20261027T144939Z/F_patchwam_ramp100/runs/2026-10-27_14-57-15/checkpoint_step_42960.safetensors。
完整50任务×Clean/Randomized×30=3000。独立目录outputs/robotwin_eval/patchwam42960_writeoff_all50_30ep_20261105，不改当前Sidecar-Off。

真正Write-Off实现：保留arch.sidecar_enabled=True，arch.sidecar.write_action = lambda index, actions, tokens, valid: actions。即Sidecar初始化、Read、Refine、FFN及原生头部路径保持，所有tap的Write直接返回写入前的action hidden特征。不是将机器人动作置零，不是将token置零，也不是只把MHA某个V置零。

复用已验证的私有split服务和native30调度；加载同一完整权重、不修改权重。启动验收复用Full canonical、Off/Write-Off固定输入对照，再额外在最终Write-Off服务状态执行两fixture重复，确认输出与Write-Off参考逐元素一致，并计数证明Read/Refine仍执行。旧Sidecar-Off与Write-Off仅两个固定输入已验证动作一致，不能不经闭环就宣称所有状态等价。结果也不是原始baseline，主干仍为联合训练后的权重。

参数对齐：Eager、DiT cache .99/max3、10denoise、sync32、384×320、CFG1、官方专家筛seed/RT/成功/动作预算、原chunk通信及camera gate、无视频。cache是近似算法且可能与干预交互；历史语言RNG未冻结，不宣称Full/Refine-Off/Sidecar-Off/Write-Off逐episode严格同指令配对。

已完成Refine-Off的闲置专用模型PID30192核对身份后退休，成绩日志保持。新模型主体L20 GPU0、T5 GPU6；43900–43931共32独立会话。4090动态6144MiB空余启动，32调度线程/会话为容量，不宣称32sim实际全部容纳；当前Sidecar-Off收尾不停止，尽量利用其余显存。其他训练和真机服务不动。

状态launch_status.json；结果state.json和comparison.json（同完成条件四方案对比）；部署证据replica0/eager_validation/result.json及write_off_serving_validation.json。当前启动加载中，未声称正式rollout成功。

下一步：确认实际部署与仿真，汇总Clean/Randomized及逐任务四组结果；若Off/Write-Off闭环不同，优先核对场景/语言/动作，不将不配对的样本差异当模块因果。


2026-11-05T08:33:10.507705+00:00：模型PID155216主体GPU0/T5 GPU6已加载及验证通过；两个fixture最终Write-Off状态重复一致、匹配Write-Off参考、sidecar_enabled=True，Read/Refine确有调用（计数文件write_off_serving_validation.json）。Full canonical/Sidecar-Off对照恢复及参数未变检查亦通过。4090新controller56631，当前19路仿真进程已启动，32会话容量；完整rollout/SR尚待，不将进程数当已验收成绩。旧Sidecar-Off仍继续。
