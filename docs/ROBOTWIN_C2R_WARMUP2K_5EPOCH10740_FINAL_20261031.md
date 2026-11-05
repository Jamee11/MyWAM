# C2R + Sidecar warmup2K + joint5epoch：最终RoboTwin结果

结束：2026-10-31T15:06:38Z；独立原生审计2026-10-31T15:07:13Z通过。100/100条件，每条件30条，共3000条、2254成功；所有中断partial和初始化失败均排除，3次容量中断条件已恢复完整通过。

权重：`outputs/1031-warmup2k-fromc2r-5epoch/checkpoint_step_10740.safetensors`；SHA256 `b27cbb15180e35ccdc43748914f5f9f9adcd5d1c4eaa468eaf19f25a47a3a7fb`。

|范围|本次成功/总数|本次|21480联合10epoch|Full-backward32k原30|官方|
|---|---:|---:|---:|---:|---:|
|Clean|1337/1500|89.13%|88.67%|89.00%|89.36%|
|Randomized|917/1500|61.13%|61.00%|53.20%|48.72%|
|综合|2254/3000|75.13%|74.83%|71.10%|69.04%|

设置：官方C2R微调权重初始化新Sidecar，2K更新warmup后重新加载、新优化器、低LR联合5epoch；本次新增训练约0.93+5=5.93epoch，不含官方既有训练。评估沿用21480的eager/compile-off、cache-on(.99/max3)、10-step denoising、horizon32、无视频、官方专家seed/RT/动作预算/成功规则及已验证的chunk内部采图优化。L20推理，4090仿真。

相对21480总共多9次成功：Clean多7次、Randomized多2次；综合+0.30pp。相对官方综合+6.09pp、Randomized+12.41pp、Clean-0.23pp。当前证据支持与21480效果接近，不能凭这9次差别证明明确优势；两者起点、LR与训练量不同，不能单独归因5epoch或C2R初始化。每条件30条和未固定语言RNG限制仍在。

后续若需要确认0.30pp差距，应优先同一episode/语言清单的更多rollout；本报告不自动启动新实验。

证据：`outputs/robotwin_eval/warmup2k_fromc2r_5epoch_10740_4090_all100_30ep_20261031/final_independent_audit.json`、`final_results.json`及每条件原生`run.json`/日志。
