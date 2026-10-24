# Sidecar32k 与官方 baseline 异常任务正常复测：成功率与成功完成步数

核查时间：2026-10-20T14:20:53.594629+00:00。使用已完成的5任务×双域×100条；两模型各1000条，共2000条。

基准采用此前已固定的正常复测结果：9个条件来自官方manifest/run_eval共90个10条分片，place_dual_shoes/Clean使用传统native92/100。不是沿用异常旧环境低分，也不在传统native Randomized12%和manifest16%之间重新择优。Sidecar为本机新官方环境32k正式结果，未混20k或外机42960。

| Task | Mode | 旧环境baseline SR | 正常复测baseline SR | Sidecar32k SR | ΔSR(pp) | baseline成功均步 | Sidecar成功均步 | Δ步数 | Δ步数% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| click_bell | Clean | 100% | 100% | 100% | +0 | 67.46 | 62.67 | -4.79 | -7.10% |
| click_bell | Randomized | 78% | 77% | 71% | -6 | 89.75 | 110.28 | +20.53 | +22.87% |
| handover_block | Clean | 86% | 85% | 93% | +8 | 267.44 | 266.15 | -1.28 | -0.48% |
| handover_block | Randomized | 17% | 46% | 40% | -6 | 292.41 | 288.45 | -3.96 | -1.36% |
| place_dual_shoes | Clean | 13% | 92% | 88% | -4 | 223.63 | 221.48 | -2.15 | -0.96% |
| place_dual_shoes | Randomized | 0% | 16% | 10% | -6 | 228.69 | 240.80 | +12.11 | +5.30% |
| place_shoe | Clean | 50% | 97% | 92% | -5 | 148.44 | 148.37 | -0.07 | -0.05% |
| place_shoe | Randomized | 19% | 49% | 66% | +17 | 191.33 | 191.05 | -0.28 | -0.15% |
| scan_object | Clean | 66% | 89% | 87% | -2 | 160.51 | 163.77 | +3.26 | +2.03% |
| scan_object | Randomized | 14% | 56% | 52% | -4 | 169.80 | 165.83 | -3.98 | -2.34% |

## 汇总与分析

10项条件等权成功率：baseline70.70% → Sidecar69.90%，-0.80pp；Clean92.60% →92.00%，-0.60pp；Randomized48.80% →47.80%，-1.00pp。

先对每个条件计算成功episode平均步数，再10项等权：183.95→185.88，增加1.94步（+1.05%）；Clean173.49→172.49（约-0.58%），Randomized194.40→199.28（约+2.51%）。另一种统计口径：合并所有成功episode，baseline707次成功均172.05步，Sidecar699次成功均173.70步；样本权重不同，不能混为同一均值。

7/10项成功步数差在约±2.4%以内；主要增加为click_bell/Randomized +20.53步（+22.87%，SR77→71），place_dual_shoes/Randomized +12.11步（+5.30%，SR16→10，成功样本少）；click_bell/Clean减少4.79步（-7.10%，均100%成功）。place_shoe/Randomized的SR49→66（+17pp），成功均步191.33→191.05，基本一致。

这个共同子集不支持“Sidecar普遍多花很多成功动作步”的说法。某些Randomized可靠性下降、失败耗尽预算会增加总耗时；也存在单鞋Randomized明显改善，不能笼统判定全部任务劣化。这10项不是全50任务结论。

## 口径与验收

成功均步仅统计该模型成功的episode，成败从native累计Success rate差分或manifest [RESULT]读取；步数取该episode终态前最后的take_action_cnt。不是去噪步、模型请求次数、物理scene.step次数或wall time。视频额外终态帧不计作动作；4个独立视频抽查均满足frames=日志动作步数+1，源是官方_base_task.py在check_success成立后额外写成功终态帧。

全部90个baseline分片逐episode的index/seed/success与result.json相等；baseline每条件episode0…99无遗漏，成功数与已发布本地复测汇总一致。Sidecar100条的成功数、accepted seed顺序、complete及errors均与正式run结果核对。原始逐episode和来源路径见episodes.json。

两模型的成功样本集合不同；9个baseline条件使用sealed manifest而Sidecar使用native seed scan，accepted seed交集为79…100条，同seed也不自动表示指令/场景/RNG完全配对。均步差是描述性比较，不可直接当纯模型结构引起的效率差或秒级吞吐提升。辅助同seed双方成功的子集结果列在下表，仍不视为严格受控配对。

| Task | Mode | accepted seed交集 | 双方成功条数 | baseline均步 | Sidecar均步 | Δ步数 |
|---|---|---:|---:|---:|---:|---:|
| click_bell | Clean | 100 | 100 | 67.46 | 62.67 | -4.79 |
| click_bell | Randomized | 100 | 62 | 84.85 | 102.82 | +17.97 |
| handover_block | Clean | 96 | 77 | 266.19 | 264.75 | -1.44 |
| handover_block | Randomized | 95 | 27 | 278.19 | 279.63 | +1.44 |
| place_dual_shoes | Clean | 83 | 69 | 221.72 | 219.90 | -1.83 |
| place_dual_shoes | Randomized | 79 | 6 | 222.33 | 253.33 | +31.00 |
| place_shoe | Clean | 96 | 87 | 148.62 | 148.16 | -0.46 |
| place_shoe | Randomized | 94 | 36 | 189.58 | 165.53 | -24.06 |
| scan_object | Clean | 92 | 75 | 160.31 | 163.05 | +2.75 |
| scan_object | Randomized | 89 | 40 | 167.05 | 165.07 | -1.98 |

证据目录：docs/evidence/robotwin_baseline32k_success_steps_20261020；包含可复现只读analyze.py、comparison.json/csv、episodes.json、video_frame_crosscheck.json。未改模型/训练/仿真/任务或在跑进程。

后续：完成32k其余条件并按共同完整集合更新；若要确认动作效率的因果差异，应另行固定相同manifest、场景、指令与RNG做两权重配对评测，同时记录失败预算和完整wall time，而非只看成功子集。
