# 跨机Sidecar42960评测速度与官方同条件对照

快照：2026-10-23T01:45:01.113036+00:00。本次只读核对评测结果并保存分析，不改训练、模型、仿真配置或进程。

## 1. 端到端评测吞吐

| 批次 | 观察完成episode | 从仿真批次启动起耗时 | 平均吞吐 |
| --- | ---: | ---: | ---: |
| 旧同机42960 | 2401 | 61.48小时 | 39.1条/小时 |
| 当前A100模型+4090仿真 | 1324 | 8.91小时 | 148.7条/小时 |

当前观察吞吐为旧批次的 **3.81倍**。旧批次从首个原生run.json的
created_utc=2026-10-20T02:36:45.366615+00:00，到用户停止快照2026-10-22T16:05:41.917766+00:00；分母2401含partial
的已终结episode。当前从dispatcher started_utc开始，分子同样含partial已终结episode：
原六路963条，增量两路361条。
不能用完整百条条件数衡量吞吐，否则会遗漏大量已实际执行的episode。

两个分母都包含启动后的环境准备、规划筛选、排队和后续扩容耗时；模型最初预部署不在分母。
这是实际工作流的描述性吞吐对照，不是同任务/seed/并发/负载匹配的硬件benchmark。
旧批次最终3模型8sim，当前4模型最多8sim且增量一条lane已完成；并发随时间变化。
任务难度、动作预算、成功早停、Clean/Randomized占比和其他作业均影响速度。
无法据此把3.81倍全部归因于4090，也不能单独量化6路扩到8路的因果收益。

## 2. 当前完整条件与官方比较

只取本次新批次中run.json status=complete、result.complete=true、实际100条的条件。
未混入旧20项或任何不足100条的成绩。官方来源：[OpenWAM issue46发布表](https://github.com/OpenWAM-Official/OpenWAM/issues/46#issuecomment-6118279946)，
2026-10-23已在线重核本节9项，与本地2026-10-22快照一致。

| Task | 条件 | Sidecar42960 | 官方发布模型 | 差值（百分点） |
| --- | --- | ---: | ---: | ---: |
| handover_mic | Randomized | 27% | 22% | +5 |
| move_playingcard_away | Randomized | 54% | 49% | +5 |
| open_laptop | Randomized | 61% | 57% | +4 |
| place_bread_skillet | Randomized | 45% | 61% | -16 |
| place_burger_fries | Randomized | 71% | 80% | -9 |
| place_cans_plasticbox | Randomized | 91% | 96% | -5 |
| press_stapler | Clean | 96% | 90% | +6 |
| press_stapler | Randomized | 43% | 53% | -10 |
| rotate_qrcode | Randomized | 15% | 7% | +8 |
| **相同9条件合计** | **900条** | **55.89%（503/900）** | **57.22%（515/900）** | **-1.33** |

5项高于官方、4项低于官方。最大负差为place_bread_skillet Randomized（-16pp），
其次press_stapler Randomized（-10pp）和place_burger_fries Randomized（-9pp）。
最大正差为rotate_qrcode Randomized（+8pp），但绝对成功率仍只有15%。
press_stapler Clean96%对官方90%，Randomized43%对53%，表明这项任务的随机域仍弱；
不能仅凭此表归因相机、数据或模块的具体机制。

上述均值按相同9个完整条件比较，只有1个Clean与8个Randomized，绝不能与官方全50任务
双域综合均值比较。8个Randomized合计407/800=50.875%，官方同8项425/800=53.125%，
低2.25pp。各条件100条，仅描述本次点估计，不宣称统计显著性。
官方是发布Clean2Random模型，不是本地自训练baseline10740；模型、训练预算及统计不同，
不是同accepted seeds/指令/噪声的逐episode配对实验，不能归因Sidecar结构。

## 3. 产物与后续

原生根：`outputs/robotwin_eval/split42960_a100_gpu56_20261022T160926Z`。
机器可读快照：根下`speed_and_official_snapshot_20261023T014501Z.json`，含每行原生run路径。
旧速度依据：`outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z/user_stop_20261022T160226Z/final_record.json`
以及其completed条目指向的run.json created_utc。

下一步继续原队列，新增完整条件后按相同口径更新新快照；针对上述负差任务优先检查失败轨迹。
此分析未逐条审阅失败视频、未做同场景硬件对照，当前快照不能作为完整28条件最终结论。

