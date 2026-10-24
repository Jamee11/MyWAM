# 原版baseline异常任务：新官方环境复测分析

> 状态更正（2026-10-20）：预选5任务×双域的原版baseline复测已于2026-10-19 03:04 UTC全部完成。额外传统native串行复核于2026-10-19停止，状态stopped_redundant，原文“整队仍在继续”仅为早期历史状态。目前进行的五项“恢复”是Sidecar32k的仿真故障重跑，不是本报告的原版baseline低分复测。


核对时间：2026-10-19T07:37:38.139533+00:00。

预选5任务双域共10个条件均已完成100条。并行manifest组9项900条于2026-10-19T03:04:09.527003+00:00全部验收，errors=[]；另有传统native双鞋Clean92/100。90个分片的episode_start覆盖0..90、每片10条均已核对。下表官方列来自此前存档的OpenWAM issue #46逐项数据。

| task | condition | old environment | new environment | official #46 | new − old (pp) |
|---|---|---:|---:|---:|---:|
| click_bell | demo_clean | 100% | 100% | 100% | +0 |
| click_bell | demo_randomized | 78% | 77% | 75% | -1 |
| handover_block | demo_clean | 86% | 85% | 84% | -1 |
| handover_block | demo_randomized | 17% | 46% | 52% | +29 |
| place_dual_shoes | demo_clean | 13% | 92% | 89% | +79 |
| place_dual_shoes | demo_randomized | 0% | 16% | 19% | +16 |
| place_shoe | demo_clean | 50% | 97% | 98% | +47 |
| place_shoe | demo_randomized | 19% | 49% | 51% | +30 |
| scan_object | demo_clean | 66% | 89% | 93% | +23 |
| scan_object | demo_randomized | 14% | 56% | 60% | +42 |

共同10项平均：旧44.30%→新70.70%，官方同集合72.10%；提升26.40个百分点，距官方1.40个百分点。Clean同5项：63.00%→92.60%，官方92.80%；Randomized同5项：25.60%→48.80%，官方51.40%。

主要结果：双鞋Clean +79pp、单鞋Clean +47pp、扫码Randomized +42pp、单鞋Randomized +30pp、递积木Randomized +29pp。对照项click_bell Clean100→100，Randomized78→77；handover_block Clean86→85，原本接近官方的项目基本稳定。复测支持之前极低分中存在显著环境/执行栈影响，不能据旧环境成绩直接评价权重能力。

同一发布checkpoint与原生stats、默认10步/32动作/cache/compile；新官方提交与隔离Conda、RT、动作预算、成功判据不改。两真实输入上独立模型服务与共享池动作bitwise相同（仅此fixture范围），9368物体文件/11000背景及相关机器人资产已hash核验；证据降低这些因素的嫌疑，但未对每个闭环输入证明等价。cuRobo源码实际同为d64c4b0/v0.7.8，不得称算法版本升级已定位根因；其Torch绑定二进制和SAPIEN/Torch/Warp运行栈确有变化。尚未做单依赖受控消融，不能确定某一个包是唯一根因。

协议限制：双鞋Clean92为传统native；其余9项为上游提供的manifest/run_eval，固定accepted seed/指令/RNG并分片执行，与原传统native不是逐seed比特配对。传统native随后也完成双鞋Randomized12/100（官方19），与manifest16/100分列保留，不挑高分替换、不拼样本。原传统native整队仍在继续，不代表完整五任务传统native全部完成。

全量旧baseline完整100项的假设替换：Clean80.66%→83.62%，Randomized38.96%→41.28%，综合59.81%→62.45%（+2.64pp）。这只替换预选10项，其余沿用旧值，并非新环境50任务全量成绩；不能直接当作官方全量69.04%的精确复现或证明剩余差距来自模型。

后续：保留原始数据；以本次轻量验证确认异常分数显著恢复。当前GPU优先32k Sidecar；若要完整复现原版官方全量结果，需要单独完成新环境50任务双域，若要精确定位差异，需要固定输入/seed/指令做仿真依赖消融。

证据：`outputs/robotwin_eval/baseline_official_manifest_deadline_20261018/run.json`；`outputs/robotwin_eval/rebuild_official_20261018/official_diagnostic100/run.json`；`outputs/robotwin_eval/audit_official_20261018/per_task_comparison.csv`。
