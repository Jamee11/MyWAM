# Sidecar32k与官方逐任务阶段对照

更新时间：2026-10-19T08:11:00.759095+00:00。仅本机L20独立运行，6模型18路；官方列为[issue #46 Clean2Random发布权重](https://github.com/OpenWAM-Official/OpenWAM/issues/46#issuecomment-6118279946)。

当前完整12个条件：Clean 11项、Randomized 1项。共同完整Clean子集：Sidecar 95.73%，官方同任务95.09%，差+0.64pp；不是全50任务Clean平均。

| task | condition | Sidecar32k | official | difference(pp) |
|---|---|---:|---:|---:|
| adjust_bottle | demo_clean | 97% | 90% | +7 |
| beat_block_hammer | demo_clean | 98% | 99% | -1 |
| click_alarmclock | demo_clean | 98% | 99% | -1 |
| click_bell | demo_clean | 100% | 100% | +0 |
| grab_roller | demo_clean | 100% | 100% | +0 |
| handover_block | demo_clean | 93% | 84% | +9 |
| move_pillbottle_pad | demo_clean | 96% | 96% | +0 |
| place_cans_plasticbox | demo_clean | 99% | 98% | +1 |
| place_container_plate | demo_clean | 97% | 98% | -1 |
| place_dual_shoes | demo_clean | 88% | 89% | -1 |
| scan_object | demo_clean | 87% | 93% | -6 |
| click_alarmclock | demo_randomized | 65% | 77% | -12 |

## 尚未完成：仅列暂时比例，不纳入完整平均

| task | condition | success/completed | temporary rate | official full100 |
|---|---|---:|---:|---:|
| adjust_bottle | demo_randomized | 32/88 | 36.4% | 51% |
| handover_block | demo_randomized | 2/3 | 66.7% | 52% |
| place_dual_shoes | demo_randomized | 4/28 | 14.3% | 19% |
| place_container_plate | demo_randomized | 46/61 | 75.4% | 76% |
| beat_block_hammer | demo_randomized | 30/92 | 32.6% | 55% |
| place_cans_plasticbox | demo_randomized | 36/44 | 81.8% | 96% |
| dump_bin_bigbin | demo_clean | 5/5 | 100.0% | 99% |
| scan_object | demo_randomized | 44/82 | 53.7% | 60% |
| blocks_ranking_rgb | demo_clean | 63/67 | 94.0% | 96% |
| blocks_ranking_rgb | demo_randomized | 3/5 | 60.0% | 47% |
| place_shoe | demo_clean | 82/90 | 91.1% | 98% |
| place_shoe | demo_randomized | 39/54 | 72.2% | 51% |
| grab_roller | demo_randomized | 47/93 | 50.5% | 38% |
| move_pillbottle_pad | demo_randomized | 6/11 | 54.5% | 31% |
| pick_diverse_bottles | demo_clean | 14/22 | 63.6% | 77% |
| pick_diverse_bottles | demo_randomized | 3/5 | 60.0% | 62% |

分析：当前已完成Clean子集整体与官方接近，adjust_bottle、handover_block高于官方，scan_object低6pp；差值不能直接解释为纯Sidecar结构收益，训练条件/表示与官方权重不同。唯一完整Randomized为click_alarmclock65/100，对官方77/100低12pp，说明随机域不能称全面持平。进行中的place_shoe、grab_roller暂时较高，而beat_block_hammer、adjust_bottle偏低；均须等待100条。

完整条件明显偏Clean，不能拿此处95.73%与官方全50任务89.36%直接比较，也不能将未完成任务当前比例与完整任务混算全量综合。原baseline官方manifest复测与Sidecar native流程差异、accepted seed集合差异仍需保留，不能宣称逐seed配对或纯架构因果结论。全部任务保持原RT/预算/专家筛seed/成功判据及相同推理配置，不因当前分数调整协议。

CSV快照：`outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/comparison_official_snapshot.csv`；实时进度见ROBOTWIN_SIDECAR32K_L20_LIVE_20261019.md。下一步等待Randomized完整结果扩展覆盖，再形成同任务双域对照。


## 2026-10-19T08:31:55.776245+00:00：与旧环境原版和旧环境20k对照

新32k已完整12项，均可与原版旧环境比较；其中4项能与Sidecar旧环境20k比较，全部为Clean。只比较各组均完成100条、无errors的正式结果；被用户停止的不完整项不补零、不拼接。旧baseline是我们在旧环境运行发布权重的实测值，不是issue46官方数值。

| task | condition | baseline旧环境 | Sidecar20k旧环境 | Sidecar20k新环境 | Sidecar32k新环境 |
|---|---|---:|---:|---:|---:|
| adjust_bottle | demo_clean | 95% | 95% | 98% | 97% |
| beat_block_hammer | demo_clean | 77% | 92% | 99% | 98% |
| click_alarmclock | demo_clean | 99% | — | 100% | 98% |
| click_alarmclock | demo_randomized | 79% | — | — | 65% |
| click_bell | demo_clean | 100% | — | — | 100% |
| grab_roller | demo_clean | 100% | — | — | 100% |
| handover_block | demo_clean | 86% | — | 80% | 93% |
| move_pillbottle_pad | demo_clean | 96% | — | — | 96% |
| place_cans_plasticbox | demo_clean | 99% | 100% | — | 99% |
| place_container_plate | demo_clean | 99% | 99% | 98% | 97% |
| place_dual_shoes | demo_clean | 13% | — | 85% | 88% |
| scan_object | demo_clean | 66% | — | 83% | 87% |

共同4项平均：旧原版92.50%，旧Sidecar20k 96.50%，新Sidecar32k 97.75%；32k比旧20k高+1.25pp，比旧原版高+5.25pp。同旧环境比较中Sidecar20k较旧原版高+4.00pp，但原版部署cache/compile开启、旧Sidecar关闭，训练数据/表示不同，仍不是纯结构消融。

桥接3项（adjust_bottle、beat_block_hammer、place_container_plate Clean）：新环境20k平均98.33%，新32k 97.33%，32k每项均低1pp。因此跨环境相对旧20k的改善，不能直接归因于多训练到32k。扩展至全部新环境共同完整7项，新20k 91.86%，新32k 94.00%，差+2.14pp；子集不同，不混算这几种平均。

旧20k另外两个完整项为beat_block_hammer/Randomized35/100和place_container_plate/Randomized70/100，当前32k均尚未完成百条，不作为完整比较。新32k鞋子/扫码相对旧原版的大幅提升有已证实的新环境原版自身恢复作为混杂因素，不能称全部来自Sidecar。下一步等待Randomized完整结果与更多同环境20k对应项；保留全部旧产物和当前队列。
