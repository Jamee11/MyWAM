# 原版baseline对照官方：全部50任务逐项差值

核对时间：2026-10-19T08:05:14.396207+00:00。官方数据重新从[issue #46作者回复](https://github.com/OpenWAM-Official/OpenWAM/issues/46#issuecomment-6118279946)获取，50任务双域共100个条件；均为每条件100条。

结论：不能认为旧环境全部差距已在可控范围。原始旧环境综合59.81%，官方69.04%，低9.23pp；替换预选10项后综合62.45%，仍低6.59pp。其余45任务90条件尚未在新环境复测，其中26条件低至少10pp、12条件低至少20pp。±5/10/20pp仅为描述性分档，不是统计显著性检验或预设验收标准。

| 条件 | 官方 | 原始旧环境 | 原始差值(pp) | 替换10项后混合成绩 | 混合差值(pp) |
|---|---:|---:|---:|---:|---:|
| demo_clean | 89.36% | 80.66% | -8.70 | 83.62% | -5.74 |
| demo_randomized | 48.72% | 38.96% | -9.76 | 41.28% | -7.44 |
| combined | 69.04% | 59.81% | -9.23 | 62.45% | -6.59 |

## 全部50任务：原始旧环境逐项比较

每格为“旧成绩 / 官方成绩（旧−官方，百分点）”；正值表示旧成绩更高。已复测列仅标记预先选择的5任务，不把新分数混入原始旧列。

| task | Clean：旧/官方（差） | Randomized：旧/官方（差） | 双域已新环境复测 |
|---|---:|---:|---|
| adjust_bottle | 95/90（+5） | 38/51（-13） | 否 |
| beat_block_hammer | 77/99（-22） | 42/55（-13） | 否 |
| blocks_ranking_rgb | 93/96（-3） | 27/47（-20） | 否 |
| blocks_ranking_size | 53/55（-2） | 10/14（-4） | 否 |
| click_alarmclock | 99/99（+0） | 79/77（+2） | 否 |
| click_bell | 100/100（+0） | 78/75（+3） | 是 |
| dump_bin_bigbin | 97/99（-2） | 51/69（-18） | 否 |
| grab_roller | 100/100（+0） | 37/38（-1） | 否 |
| handover_block | 86/84（+2） | 17/52（-35） | 是 |
| handover_mic | 70/90（-20） | 2/22（-20） | 否 |
| hanging_mug | 47/49（-2） | 9/10（-1） | 否 |
| lift_pot | 100/100（+0） | 10/6（+4） | 否 |
| move_can_pot | 97/100（-3） | 26/29（-3） | 否 |
| move_pillbottle_pad | 96/96（+0） | 21/31（-10） | 否 |
| move_playingcard_away | 98/97（+1） | 45/49（-4） | 否 |
| move_stapler_pad | 59/76（-17） | 9/16（-7） | 否 |
| open_laptop | 91/93（-2） | 53/57（-4） | 否 |
| open_microwave | 53/82（-29） | 5/7（-2） | 否 |
| pick_diverse_bottles | 77/77（+0） | 64/62（+2） | 否 |
| pick_dual_bottles | 77/85（-8） | 67/73（-6） | 否 |
| place_a2b_left | 78/87（-9） | 25/30（-5） | 否 |
| place_a2b_right | 81/89（-8） | 15/30（-15） | 否 |
| place_bread_basket | 88/92（-4） | 73/74（-1） | 否 |
| place_bread_skillet | 85/92（-7） | 25/61（-36） | 否 |
| place_burger_fries | 98/97（+1） | 77/80（-3） | 否 |
| place_can_basket | 80/87（-7） | 30/45（-15） | 否 |
| place_cans_plasticbox | 99/98（+1） | 92/96（-4） | 否 |
| place_container_plate | 99/98（+1） | 86/76（+10） | 否 |
| place_dual_shoes | 13/89（-76） | 0/19（-19） | 是 |
| place_empty_cup | 100/100（+0） | 72/67（+5） | 否 |
| place_fan | 90/93（-3） | 37/40（-3） | 否 |
| place_mouse_pad | 64/90（-26） | 10/21（-11） | 否 |
| place_object_basket | 67/95（-28） | 19/38（-19） | 否 |
| place_object_scale | 88/94（-6） | 30/38（-8） | 否 |
| place_object_stand | 96/96（+0） | 59/65（-6） | 否 |
| place_phone_stand | 79/85（-6） | 14/22（-8） | 否 |
| place_shoe | 50/98（-48） | 19/51（-32） | 是 |
| press_stapler | 62/90（-28） | 34/53（-19） | 否 |
| put_bottles_dustbin | 68/83（-15） | 28/52（-24） | 否 |
| put_object_cabinet | 36/48（-12） | 17/17（+0） | 否 |
| rotate_qrcode | 86/90（-4） | 8/7（+1） | 否 |
| scan_object | 66/93（-27） | 14/60（-46） | 是 |
| shake_bottle | 100/100（+0） | 88/91（-3） | 否 |
| shake_bottle_horizontally | 100/100（+0） | 90/93（-3） | 否 |
| stack_blocks_three | 92/95（-3） | 32/66（-34） | 否 |
| stack_blocks_two | 97/100（-3） | 67/89（-22） | 否 |
| stack_bowls_three | 85/83（+2） | 68/73（-5） | 否 |
| stack_bowls_two | 95/96（-1） | 72/74（-2） | 否 |
| stamp_seal | 83/89（-6） | 28/29（-1） | 否 |
| turn_switch | 43/54（-11） | 29/39（-10） | 否 |

## 尚未复测且低至少10个百分点的全部条件

| task | condition | old | official | difference(pp) |
|---|---|---:|---:|---:|
| place_bread_skillet | demo_randomized | 25% | 61% | -36 |
| stack_blocks_three | demo_randomized | 32% | 66% | -34 |
| open_microwave | demo_clean | 53% | 82% | -29 |
| place_object_basket | demo_clean | 67% | 95% | -28 |
| press_stapler | demo_clean | 62% | 90% | -28 |
| place_mouse_pad | demo_clean | 64% | 90% | -26 |
| put_bottles_dustbin | demo_randomized | 28% | 52% | -24 |
| beat_block_hammer | demo_clean | 77% | 99% | -22 |
| stack_blocks_two | demo_randomized | 67% | 89% | -22 |
| blocks_ranking_rgb | demo_randomized | 27% | 47% | -20 |
| handover_mic | demo_clean | 70% | 90% | -20 |
| handover_mic | demo_randomized | 2% | 22% | -20 |
| place_object_basket | demo_randomized | 19% | 38% | -19 |
| press_stapler | demo_randomized | 34% | 53% | -19 |
| dump_bin_bigbin | demo_randomized | 51% | 69% | -18 |
| move_stapler_pad | demo_clean | 59% | 76% | -17 |
| place_a2b_right | demo_randomized | 15% | 30% | -15 |
| place_can_basket | demo_randomized | 30% | 45% | -15 |
| put_bottles_dustbin | demo_clean | 68% | 83% | -15 |
| adjust_bottle | demo_randomized | 38% | 51% | -13 |
| beat_block_hammer | demo_randomized | 42% | 55% | -13 |
| put_object_cabinet | demo_clean | 36% | 48% | -12 |
| place_mouse_pad | demo_randomized | 10% | 21% | -11 |
| turn_switch | demo_clean | 43% | 54% | -11 |
| move_pillbottle_pad | demo_randomized | 21% | 31% | -10 |
| turn_switch | demo_randomized | 29% | 39% | -10 |

## 如何解释

预选5任务双域新复测平均70.70%，官方同集合72.10%，这一个子集已接近；但不能外推其他45任务。比如place_bread_skillet/Randomized低36pp、stack_blocks_three/Randomized低34pp，仍属需要新环境验证的明显偏差，不能仅归为百条采样误差。旧环境成绩仍可作为“旧环境内”的记录，不能直接称官方复现成功，也不能用它与新环境Sidecar做跨环境模型收益归因。

原始旧环境99项完整加stack_bowls_three/Randomized指定完整重跑68/100，合计100项；旧audit的97项CSV缺的3项本报告已从最终结果补齐。新复测5任务10条件采用此前固定替换规则，无论升降均替换。鞋Clean为传统native92/100，其余为官方manifest；传统native鞋Randomized12/100另存，不择优混算。未完成的新任务、32k Sidecar成绩均不纳入原版baseline对照。

后续优先验证上表明显偏低的尚未复测任务，仍保留当前32k主队列；本次分析不自动停止/改动任何评测进程，也不凭旧环境差距直接定位到某个仿真依赖。

CSV：`outputs/robotwin_eval/baseline_official_comparison_20261019/per_condition_comparison.csv`；统计JSON：`outputs/robotwin_eval/baseline_official_comparison_20261019/summary.json`；本次官方评论原始快照亦在同目录。
