# RoboTwin消融与ManiSkill最终结果（2026-11-03）

读取时间：2026-11-03T01:48:40.418369+00:00。两个队列均完成；本次未改配置或启动新实验。

## RoboTwin 21480：19组固定manifest消融

完成时间：2026-11-03T00:14:46.284551+00:00。19×10tasks×2domains×10=3800；最终独立审计通过：seed/语言/manifest一致，纳入日志无明确渲染错误，原生有效结果验收通过。每组200条，不能视作50tasks全套成功率。

| 模式 | Clean | Randomized | 综合 | 对Full差值(pp) |
|---|---:|---:|---:|---:|
| full | 92.0% | 62.0% | 77.0% | +0.0 |
| write_off | 91.0% | 69.0% | 80.0% | +3.0 |
| grounding_reset | 92.0% | 68.0% | 80.0% | +3.0 |
| relation_reset | 93.0% | 70.0% | 81.5% | +4.5 |
| summary_reset | 90.0% | 71.0% | 80.5% | +3.5 |
| plan_reset | 93.0% | 68.0% | 80.5% | +3.5 |
| all_tokens_reset | 91.0% | 66.0% | 78.5% | +1.5 |
| visual_read_off | 90.0% | 66.0% | 78.0% | +1.0 |
| refine_off | 93.0% | 69.0% | 81.0% | +4.0 |
| sidecar_text_off | 93.0% | 68.0% | 80.5% | +3.5 |
| sidecar_proprio_off | 91.0% | 67.0% | 79.0% | +2.0 |
| write_tap7_off | 91.0% | 66.0% | 78.5% | +1.5 |
| write_tap15_off | 93.0% | 68.0% | 80.5% | +3.5 |
| write_tap23_off | 91.0% | 68.0% | 79.5% | +2.5 |
| role_embedding_off | 88.0% | 66.0% | 77.0% | +0.0 |
| ffn_off | 92.0% | 68.0% | 80.0% | +3.0 |
| position_off | 91.0% | 65.0% | 78.0% | +1.0 |
| stage_context_off | 90.0% | 65.0% | 77.5% | +0.5 |
| sidecar_off | 92.0% | 63.0% | 77.5% | +0.5 |

Relation-Reset最高81.5%，Full77%，All-Reset78.5%，完整Sidecar-Off77.5%。因此局部干预改善Randomized是本样本观察，但完整关闭分支并无大幅收益。Write-Off80%与Sidecar-Off77.5%仍相差5/200，需要同输入/同轨迹对照解释；不能因两fixture相等就宣称所有闭环状态等价。Cache近似与干预的交互、仿真执行差异、小样本及多重比较限制均保留；不将表格当显著因果证明。

## ManiSkill最终20872

8tasks×100Clean+100HardRandom=1600，成功1170，综合73.125%；Clean589/800=73.625%，HardRandom581/800=72.625%。四worker日志未检出OIDN Error/ErrorDeviceLost/VK_ERROR_DEVICE_LOST/Traceback/illegal memory access。无视频。本archive评测协议不直接等价于官方发布的其他任务/分布榜单。

| task | Clean | HardRandom | 综合 |
|---|---:|---:|---:|
| lift_peg_upright | 72% | 72% | 72.0% |
| pick_cube | 93% | 94% | 93.5% |
| pick_cuboid | 80% | 70% | 75.0% |
| pick_sphere | 92% | 81% | 86.5% |
| place_sphere | 36% | 41% | 38.5% |
| pull_cube | 63% | 67% | 65.0% |
| push_cube | 80% | 76% | 78.0% |
| stack_cube | 73% | 80% | 76.5% |

薄弱任务是place_sphere（38.5%），其次pull_cube（65%）；pick_cube最高93.5%。当前没有该1600清单下原版baseline的配对结果，不能将73.125%解释为Sidecar增益。

证据：`outputs/robotwin_eval/sidecar21480_19arms_complete10_20261102/merged10_results.json`、`final_independent_audit.json`；`outputs/maniskill_eval/final20872_l20_gpu3_novideo_20261102T130608Z/run/summary.json`。

后续：优先检查Write-Off/Sidecar-Off差异与cache配对；尚未自动恢复最佳21480剩余32条件补测，恢复前确认空闲容量和现存队列，沿用原manifest。
