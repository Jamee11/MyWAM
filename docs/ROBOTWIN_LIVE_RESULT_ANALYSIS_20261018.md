> 2026-10-19：本机20k已按用户要求停止，以下为停止前历史快照，不代表仍在运行。32k见独立L20输出目录。

# RoboTwin 新环境实时结果分析

更新时间：2026-10-19T03:06:20.743706+00:00

原版传统官方任务评测继续；并行组复用官方固定manifest协议（seed/指令/CUDA RNG固定），两者分别记录。
每项100条才视为任务完整结果；进行中的百分比不作为全任务宏平均。

## 原版异常任务并行复测

| 任务 | 条件 | 专家清单 | 策略成功/完成 | 旧环境百条 |
|---|---|---:|---:|---:|
| place_dual_shoes | demo_randomized | 100/100 | 16/100（验收100条） | 0/100 |
| handover_block | demo_randomized | 100/100 | 46/100（验收100条） | 17/100 |
| scan_object | demo_randomized | 100/100 | 56/100（验收100条） | 14/100 |
| place_shoe | demo_randomized | 100/100 | 49/100（验收100条） | 19/100 |
| click_bell | demo_randomized | 100/100 | 77/100（验收100条） | 78/100 |
| handover_block | demo_clean | 100/100 | 85/100（验收100条） | 86/100 |
| scan_object | demo_clean | 100/100 | 89/100（验收100条） | 66/100 |
| place_shoe | demo_clean | 100/100 | 97/100（验收100条） | 50/100 |
| click_bell | demo_clean | 100/100 | 100/100（验收100条） | 100/100 |

目标截止：2026-10-19T01:43:00+00:00；并行组策略完成900/900。
近30分钟吞吐外推结束时间：2026-10-19T03:06:20.743706+00:00（早期/任务差异会影响估计，不能保证）。

## 原模型替换情景

只替换已完整项：传统双鞋Clean92/100，加并行官方manifest预选任务的有效百条结果；未完成沿用旧值。
这是混合环境/协议的假设汇总，不是新环境50任务全量复现，不按成绩择优。

- demo_clean：80.66% → 83.62%（+2.96个百分点）。
- demo_randomized：38.96% → 41.28%（+2.32个百分点）。
- combined：59.81% → 62.45%（+2.64个百分点）。

## Sidecar 20k 新环境

| 任务 | 条件 | 状态 | 成功/完成 |
|---|---|---|---:|
| adjust_bottle | demo_clean | complete | 98/100 |
| adjust_bottle | demo_randomized | complete | 34/100 |
| place_shoe | demo_clean | complete | 94/100 |
| place_shoe | demo_randomized | complete | 49/100 |
| place_container_plate | demo_clean | complete | 98/100 |
| place_container_plate | demo_randomized | complete | 69/100 |
| blocks_ranking_size | demo_clean | failed | 37/66 |
| blocks_ranking_size | demo_randomized | failed | 1/19 |
| dump_bin_bigbin | demo_clean | failed | 7/7 |
| place_dual_shoes | demo_clean | complete | 85/100 |
| place_dual_shoes | demo_randomized | complete | 10/100 |
| scan_object | demo_clean | complete | 83/100 |
| scan_object | demo_randomized | complete | 53/100 |
| place_cans_plasticbox | demo_clean | failed | 46/46 |
| place_cans_plasticbox | demo_randomized | failed | 59/66 |
| click_alarmclock | demo_clean | complete | 100/100 |
| click_alarmclock | demo_randomized | failed | 19/29 |
| beat_block_hammer | demo_clean | complete | 99/100 |
| beat_block_hammer | demo_randomized | complete | 40/100 |
| handover_block | demo_clean | complete | 80/100 |
| handover_block | demo_randomized | failed | 24/77 |
| blocks_ranking_rgb | demo_clean | failed | 23/23 |

完整任务条件14/100，策略episode 1733/10000；覆盖不足时不报全量宏平均。

加速方法/失败尝试/数值与性能证据：SIDECAR_INFERENCE_ACCELERATION_20261018.md。

## 当前训练方案的完整任务对照

仅纳入双方均完成100条的共同任务。仿真栈相同，但原版并行manifest固定指令/RNG，Sidecar为传统native；训练数据/表示及编译范围也不同，不能视为逐seed配对或纯结构消融。

| 任务 | 条件 | Sidecar20k | 新环境原版 | 差值（百分点） |
|---|---|---:|---:|---:|
| handover_block | demo_clean | 80% | 85% | -5 |
| place_dual_shoes | demo_clean | 85% | 92% | -7 |
| place_dual_shoes | demo_randomized | 10% | 16% | -6 |
| place_shoe | demo_clean | 94% | 97% | -3 |
| place_shoe | demo_randomized | 49% | 49% | +0 |
| scan_object | demo_clean | 83% | 89% | -6 |
| scan_object | demo_randomized | 53% | 56% | -3 |
共同完整子集共7项，描述性宏平均Sidecar 64.86%、原版69.14%（-4.29个百分点）；这不是全50任务综合平均，也不是结构因果结论。
