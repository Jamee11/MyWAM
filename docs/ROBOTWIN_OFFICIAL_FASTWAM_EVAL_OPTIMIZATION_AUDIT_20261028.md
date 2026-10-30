# RoboTwin / FastWAM 评测优化核查（2026-10-28）

## 当前实际入口和适用范围

本轮使用固定旧版 RoboTwin commit `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4` 的 `script/eval_policy.py`，由本地 `elastic_v3.py` 调度；不是新 main 的 XPolicyLab 多任务入口。六个原模型各有三个独立 policy session，共18 endpoint；另有独立恢复模型/任务。本轮132秒采样中10/10仿真动作/episode计数变化，证据 `outputs/robotwin_eval/render_skip_audit_20261028/progress_after.json`。因此不能把六份模型等同六个并发 job 上限。该采样证明实际推进，不证明无排队或负载均衡。

视频已关闭，不需要再以打开全量视频作为优化起点。chunk32本地缓冲/兼容桥已经减少跨机器请求，后续启动任务采用新入口；在途旧进程不热改。独立 session 保留 action deque、视觉缓存及 reset 隔离，同一 GPU 引擎仍串行推理。2ms合包尚无额外总吞吐增益，整包等全部完成后回传反而增大等待，现用逐条返回。详见当日 chunk bridge 文档。

## 尚未解决：chunk 内 GPU 相机采集

当前原生循环仍逐 action 执行 get_obs，触发相机 take_picture 和 CPU 读回。此前 lazy RGB 仅延迟 CPU 读回，仍采集/渲染；其负结果不能否定跳过 GPU 采集的价值。OpenWAM 当前执行器只在动作队列耗尽时请求下一段动作，当前三图不是逐步视觉历史追加；EEF20到EE16为预测绝对动作投影，不依赖逐步观测做增量积分。但仍应保留真实本体状态读取及 reset 语义。

特别注意 `_base_task._update_render()` 在 randomized 的 crazy_random_light 分支会推进 NumPy RNG 并改灯光。因此不能直接照搬 FastWAM 跳过整个 get_obs，否则可能改变下一次真实模型输入和随机化序列。候选优化只绕过不需要的相机采集/Color读回，保留场景同步、灯光随机数调用、实际EEF/关节、250Hz物理、逐步规划、碰撞与成功判据；需要新观测时强制完整新图，不返回陈旧图像冒充新观测。

## 已完成隔离 RT 实验：不是正式 rollout 加速结论

静态 primitive 场景，四台320×240相机，RT 32spp/depth8/OIDN/near0.1；65次场景更新全部保留。全采集65帧/相机，对照只采集0/32/64帧。4090 GPU1三个独立进程均正常退出，最低空闲约3.5GiB以上；不停止正式任务。此前GPU7诊断进程因1532MiB低于1536MiB保护线退出，保留原始日志，没有OOM证据，不能当性能样本。

| 模式 | 相机采集次数 | 循环耗时 |
|---|---:|---:|
| full_a | 260 | 4.966s |
| skip | 12 | 0.485s |
| full_b | 260 | 3.432s |

保留的3时刻×4视角共12幅图，full_a/full_b/skip逐像素一致。证据 `outputs/robotwin_eval/render_skip_audit_20261028/gpu1_retry/comparison.json` 及npz/日志。仅证明冻结场景中采集可避免且保留图一致，不覆盖移动物体、相机姿态、动态随机光照、模型动作或闭环成功率；正式评测尚未启用跳采集。

## 收益边界和下一步

此前代表性正式样本：控制热路径中 take_action 65.53%、get_obs 24.49%、客户端含推理/排队9.98%；专家/reset/启动未全部分别计时，不能把残差直接算专家。即使将全部观测开销降为1/32，理想控制热路径加速约1.31倍，按同一次attempt总墙钟约1.25倍；真实仍需状态/同步，收益更低，不能声称整体提升7–10倍或32倍。另一任务 stack_bowls_three randomized 最近累计动作5730次，take_action1688.33s、observation652.02s、client245.50s，也表明动作执行仍占主要部分；这些计时含嵌套项目，禁止全部相加。

下一步在空闲仿真容量出现后，用固定有效seed、官方已选指令、相同checkpoint和记录动作做原生回放A/B，分别验证移动场景/相机/随机光照的边界图像及模型输入，再测真实闭环。统计 capture次数、阶段调用数/总计/P50/P95、完整episode墙钟和吞吐；同动作回放用于性能归因，完整闭环重复用于政策行为检查，不能用不同动作步数时间比声称纯加速。未验收前不改变正式RGB输入。逐步规划并行和线程调优此前无可靠通过结果，均保持关闭。

## seed 与公平性

官方 issue522 描述的是新批量评估中多个worker并发专家检查后争抢episode名额，候选seed集合可能变化。本轮每个条件一个原生进程内部串行专家筛seed、跨条件并行，不能直接套用该竞态。但官方Python random未统一固定导致同环境seed不必然同指令；当前用户已选择保留原30池30+补70池70，非原生100池100，差异尚未量化。继续保留当前协议，不以优化名义修改候选池或关闭专家。未来跨checkpoint严格比较应持久化有效seed、确切指令、环境/资产revision、随机配置和候选池口径；不能仅缓存布尔expert结果或只固定seed。

## 上游核查来源

- 官方调度器：[eval_policy_multitask.py](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/scripts/eval_policy_multitask.py)：新远程入口的job数量取GPU slots与endpoint数较小值，与本轮入口不同。
- 官方[XPolicyLab入口](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/scripts/eval_policy_xpolicylab.py)：chunk内仍可获取并上传观测。
- [RoboTwin issue522](https://github.com/RoboTwin-Platform/RoboTwin/issues/522)：并发专家检查的seed集合风险。
- [FastWAM](https://github.com/yuantianyuan01/FastWAM)、[配置](https://github.com/yuantianyuan01/FastWAM/blob/main/configs/sim_robotwin.yaml)、[评测实现](https://github.com/yuantianyuan01/FastWAM/blob/main/third_party/RoboTwin/script/eval_policy.py)：通过should_request_observation条件跳过get_obs。其replan24及指令设置不直接复制到本轮32步协议。

记录时间：2026-10-28T13:57:01.436821+00:00

## 用户授权后上线（2026-10-28）

用户要求按上述优化启动评测。本次新增独立可回退的 `skip_chunk_capture.py`，只在已验证chunk32客户端的本地动作队列非空、指令未变、无视频、未成功且无额外depth/segmentation/pointcloud传感器时启用。继续调用原get_obs，临时仅覆盖该相机实例的update_picture/get_rgb；get_rgb返回明确None而非陈旧图。调用后立即恢复实例方法。正式eval包装断言本步只消费一个已缓存动作；若遇指令改变/队列清空必须完整采图。保留每步_update_render/灯光RNG/真实EEF与qpos/动作转换/规划/物理/成功/预算。

验收证据：
- CPU提取真实上游get_obs函数体：65次场景同步不变，采集65→3，RNG完整状态与每步实际proprio一致；生产constructor/bootstrap/get_obs/eval调用链、32步缓存边界及中途指令更换强制新图通过。
- 确定性动态primitive+移动相机+随机灯光，full/skip/full三独立进程，边界3时刻×4视角共12图逐像素一致，场景实际移动；`render_skip_audit_20261028/dynamic_gpu4/comparison.json`。未将该场景当原生成功率样本。
- 原生click_alarmclock Clean官方seed/指令、固定62动作回放，四次均成功，原逐步规划/物理算法不改。baseline27.268s、skip14.353s、baseline_repeat19.073s、skip_repeat14.316s。以预热重复作参考，控制回放约1.33倍/用时减少约25%；不包含模型推理、专家/初始化，不能声称整套吞吐已提升33%。get_obs重复4.976→0.526s；相机update_picture63→3次。记录的首边界图/状态一致；32步边界独立闭环存在约1e-4实际EEF差异，baseline自身重复也有同量级漂移，未声称跨独立物理轨迹逐bit一致。全probe exit0、GPU6最低空闲8694MiB，所有诊断GPU进程退出。

`formal_transport_registry.json` 已增加并启用 `skip_chunk_camera_capture`，记录代码SHA和验收目录，activate_transport在新native启动时核验代码SHA再安装。原文件/registry备份保留。只影响后续新任务；既有rollout不中断。各新任务写camera_gate_selection.json，首次实际跳过输出CAMERA_CAPTURE_FIRST_BUFFERED_ACTION，缓冲动作计数每256步及正常退出落盘camera_gate_counts.json。

GPU6此前第二路又触发真实保护线1469<1536，已退cap1，不重复盲目打开。GPU4在原两物理进程下连续五次空闲10897MiB，达到本次10240MiB准入后，主队列cap1→2（加独立恢复共三物理进程），补stack_bowls_two Clean，PID3821059。仅更换自有CPU容量监督，六原模型/原仿真保留；新guard3821054，原配置/状态/source均备份。为避免历史renderer失败在guard重启时重复惩罚新任务，启动时将已记录failure output作为历史，仅新增失败触发新的renderer回退。GPU4最低2048保护，若峰值不足只停止最新自有extra，不触碰恢复/原进程；当前headroom不等于峰值已验收。GPU1cap2/GPU5cap1、其它保护维持。

下一步：核验首个正式任务确实经过新入口并推进action/episode，记录实际计数；监测GPU4额外路稳定性及完成吞吐。模型仍六原副本＋一独立恢复副本，10路仿真（9主＋1恢复）为启动时配置，不承诺线性并发加速。保持原30+70指令池/seed续接/完整70分母及仅完整结果入表。

正式接入复核：`stack_bowls_two/demo_clean` PID3821059已输出CAMERA_CAPTURE_FIRST_BUFFERED_ACTION并推进到224/900动作，无Traceback，主队列9路＋独立恢复1路。GPU4当前空闲5630MiB，额外路仍在监督；不是只验证配置文件存在。接入证据 `outputs/robotwin_eval/performance_recheck_20261028T0900Z/camera_gate_formal_activation.json`。正式完整条件尚未因本优化完成，不提供整套加速倍数或新SR结论。

### 2026-10-28T17:50:09.165605+00:00：实际滚动吞吐

{"utc": "2026-10-28T17:50:09.165605+00:00", "active_simulators": 9, "windows": [{"window_minutes": 30, "start": "2026-10-28T17:20:05.112507+00:00", "end": "2026-10-28T17:50:09.165605+00:00", "counter_delta": 96, "counter_rate_per_hour": 191.5686408510711, "conservative_rate_after_subtracting_all_13_interrupted": 165.62705406915524}, {"window_minutes": 60, "start": "2026-10-28T16:50:03.284070+00:00", "end": "2026-10-28T17:50:09.165605+00:00", "counter_delta": 167, "counter_rate_per_hour": 166.72760714839214, "conservative_rate_after_subtracting_all_13_interrupted": 153.74881138234963}, {"window_minutes": 180, "start": "2026-10-28T14:50:00.886843+00:00", "end": "2026-10-28T17:50:09.165605+00:00", "counter_delta": 469, "counter_rate_per_hour": 156.21358748815422, "conservative_rate_after_subtracting_all_13_interrupted": 151.88357333603054}], "limitation": "Aggregate history lacks per-condition counts. Subtracting all13 interrupted episodes from each window delta is a conservative bound, not exact window attribution. Different task composition and concurrency prevent causal speedup claims."}
最近30/60min计数净增长约194/167条每小时；中断handover_block Rand的13条当前仍留在汇总计数中，全部从窗口增量扣除后保守下界约168/154条每小时。历史未记录逐条件每时刻计数，不能声称准确分摊中断13条到各窗口。当前9路，任务组合不同；该吞吐不是纯相机优化因果收益。原始证据 outputs/robotwin_eval/performance_recheck_20261028T0900Z/throughput_snapshot_20261028T1750Z.json。保持完整结果验收，后续继续自动调度。
