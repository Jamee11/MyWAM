# 最新状态：42960已恢复评测

核验时间：2026-10-27T12:50:54.827815+00:00。此前自动接续在旧模型退出的身份检查上失败；退出竞态已修复并经过21项真实进程检查，已恢复运行。

36k已完整完成3000条：2108成功，**70.27%**；Clean89.33%，Randomized51.20%，官方69.04%，同协议32k71.10%。完整结果见[36k最终报告](ROBOTWIN_FULLBACKWARD36000_30EP_FINAL_RESULTS_20261027.md)。

heads-only42960：L20 GPU1/2/4/5/6/7六份模型全部ready，18端口在线真RGB/state→20D动作验收通过；4090实际已启动7路native仿真，当前可用显存不支持此前9–12路，按峰值预算限制、不停止其他作业。原10个完整L20条件保持，剩90在4090完整30条；当前进程尚初始化，不把旧300条误报为新的rollout或新SR。source头部e4c8c2d…权重RAM和config/stats/contract重新哈希通过。完整Sidecar/cache开compile关、官方标准、不存视频保持。

运行目录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/headsonly42960_4090_resume_all100_30ep_20261027T100232Z`，`state.json`、`live_progress.json`、`auto_transition_state.json`、`resume_start_acceptance.json`。后续真实动作进展由只读observer写`first_rollout_acceptance.json`。原100条件汇总将混合10条件L20＋90条件4090，比较时仍需标注硬件。

以下为历史准备/恢复记录，以上文为准。

# 最新调度：heads-only停止，优先36k五模型评测

更新时间：2026-10-27T02:07:15.223668+00:00。用户明确要求暂停heads-only，将释放资源用于36k加速。

heads-only全部L20仿真及两份模型已按身份核验停止。结果完整归档：[停止结果说明](ROBOTWIN_HEADSONLY42960_PAUSED_RESULTS_20261027.md)：完整10条件246/300=82.00%，官方同条件81.10%，父模型82.67%；含partial共444条、320成功，不把partial混入正式子集SR。所有原日志/attempt/模型配置和权重保留。

36k模型GPU0/2原部署保持，GPU1/4/6新增三份，共5份，全部动作一致性与真实端口请求验收通过。GPU1释放后26965MiB空闲满足26000MiB门槛，才部署第三份；没有停止其他作业。总24独立端口（原12＋新增12）；新验收只访问新增端口，避免reset正在评测的原会话。原reverse SSH不动，新增独立12端口隧道。

4090当前12路native仿真，原9路＋新增3路；原9路PID逐一核验仍相同，27个已完整条件逐项结果不变、100条件唯一。原调度仅冻结新分配，作为真实父进程继续记录旧9路wait()/退出码；V4弹性队列采用这些原生run.json，原任务不重启、不拼partial。新分配只取其余queued条件，新模型先承担增量任务。旧9路全部完成且原调度active为空后，V4才停止旧调度并释放其端口锁，原GPU0/2会话随后正常复用。切换期间新模型负载均衡排除暂被旧调度持锁的GPU0/2，避免空闲旧端口阻塞增量调度。

容量策略按7680MiB峰值＋1024MiB余量入场，GPU4最多3路、GPU6最多2路，受实时余量约束；额外进程有低于2048MiB或渲染buffer失败回退，只处理本评测最新额外进程，不终止原9路。实际新增并发已由9提升12，尚未宣称吞吐等比例增加；观察持续吞吐与任务切换。官方原生expert筛seed/unseen/RT/成功/预算、每条件30条、不存视频、完整Sidecar/cache开compile关均保持。

运行状态：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward36000_4090_all100_30ep_20261026T170539Z/state.json` → elastic_v4/state.json。证据：`expansion_handover_acceptance.json`、`handover_v4_snapshot.json`、`dispatcher_v4.log`、`remote4090_expansion_acceptance.json`（部署目录）。此前准备及两组同时运行的历史记录如下，以上文为准。

# 最新状态：两组各两份已实际开测

核验时间：2026-10-26T18:23:09.782771+00:00。显存分配等待已解除，按用户最后指示继续两组各两份，不切换为全部36k。

|实验|L20模型GPU|独立服务端口|当前仿真|状态|
|---|---|---:|---|---|
|full-backward36000|0、2|12（4＋8 slots）|4090 9路|原生动作及首批成功rollout已核验|
|heads-only42960|4、6|12（6＋6 slots）|L20 7路|原生动作及首批成功rollout已核验|

两组各100条件×30=3000，保持官方expert筛seed/unseen/RT/预算/成功判据、不存视频、完整Sidecar/eager+cache。各模型重复生成与同权重跨副本动作逐位一致；36k4090及heads本地L20全部24端口真RGB/state→20D动作验收通过。模型共享引擎在每副本内部串行，独立端口保持会话状态；端口数量不能当GPU并行生成数量。

初始观察的去噪中位数约0.67–0.69秒，p95约0.70–0.74秒，尚无基础设施错误；该时间来自现有tqdm估计，排除预处理、端口锁等待和网络，不冒充端到端请求耗时或GPU利用率。观察窗含场景初始化，不能以此宣称已验证长期满负荷吞吐，也不据最先完成的短任务外推最终SR。整卡GPU利用率有外部作业，不能归因本模型。后续只读observer每15秒记录各模型生成、各sim动作步/rollout、显存，进度报告每30秒更新。

证据：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward36000_4090_all100_30ep_20261026T170539Z/dual_start_and_throughput_acceptance.json`、各实验`first_rollout_acceptance.json`、`performance_latest.json`；联合历史`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward36000_4090_all100_30ep_20261026T170539Z/dual_performance_history.jsonl`。本轮未修改核心源代码、训练/数据或运行中native入口，未插入占用真实eval会话的合成请求。所有旧结果保留。

以下为准备及启动历史记录（其中“等待/未启动”状态已被上文更新）：

# 36k与heads-only42960双实验启动记录

更新时间：2026-10-26T17:11:32.954706+00:00。当前为准备/资源分配阶段，新评测均尚未开始rollout。

| 实验 | 模型机器与目标副本数 | 仿真机器 | 评测规模 | 当前状态 |
|---|---|---|---|---|
| full-backward36000 | L20，4份 | 4090 | 100条件×30=3000 | GPU0一份已完成重复生成、真实动作及4端口验收；其余待显存分配 |
| heads-only42960 | L20，2–3份 | L20 | 100条件×30=3000 | 权重及运行脚本就绪，尚未加载模型 |

两组均沿用官方原生expert筛seed、unseen、成功判据及动作预算、RT；不保存视频。30条用于趋势判断，非官方100条完整统计。完整Sidecar保持；cache .99/max3开启，compile关闭；各用自身config/statistics/contract，跨副本逐位一致性只在同权重内比较。

## 权重与结果隔离

36k：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_36000.safetensors`。SHA256 `7c67c7f7f622f5446a804509a756945ed3e467e0742c65f0c10072c37161b0a2`。

heads-only：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_heads_only_from40000_to10epochs_20261025T171338Z/runs/from40000/checkpoint_step_42960.safetensors`。SHA256 `e4c8c2d41a7f12902dd5ca5dbd807b2ea8b1ad982909433514d9ddaf916b1f35`。

两份源文件均24862363126字节，已复制至独立RAM目录并独立回读验证SHA256；源inode/size/mtime在复制期间稳定。各实验独立deployment、端口、100条件队列、结果目录，不拼接旧episodes。

36k部署目录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward36000_deploy_20261026T165514Z`；评测目录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward36000_4090_all100_30ep_20261026T170539Z`。

heads-only部署目录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/headsonly42960_deploy_20261026T165514Z`；评测目录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/headsonly42960_l20_all100_30ep_20261026T170539Z`。

原full-backward42960已完成3000条、2100成功、70.00%；仅其已核验身份的旧模型服务停止，全部结果保留，其他作业未停止。

## 当前资源限制

初始计划36k用GPU0–3，heads-only用GPU4/6，GPU5/7主要仿真。但释放旧模型后，外部作业新增占用GPU1/3/5/7；目前仅GPU0（本模型已占用）及GPU2/4/6足够安全部署完整模型。模型实测allocated23.195GiB、reserved23.361GiB，加CUDA上下文/负载变化需余量，部署门槛26000MiB；不冒险在余量不足卡启动，不终止归属不明作业。

已向用户说明并等待资源优先级：各2份同时开始、36k先4份heads等待、或保留4+2并协调释放显存。尚未把任何方案视为已获选择。此阶段没有启动新仿真，不能把准备时间计为评测吞吐。

## 启动前检查和后续计划

部署脚本、通用任意slot数验收、4090远端/本地L20验收、两套独立弹性队列、显存保护与进度/官方比较报告脚本已准备并AST检查；最终GPU分配、endpoint清单、capacity policy仍待资源方案明确后写入。L20 native显式绑定CUDA与渲染PCI，使用已有L20 curobo缓存，禁止按4090设备名筛卡。heads-only进度和对比记录明确标注仿真L20，和旧4090结果比较存在硬件差异；之前4任务A/B不代表全部任务硬件等价。

先按实际可用卡完成选定数量模型验收，再启动36k reverse SSH与4090真动作验收、heads-only本地真动作验收，然后各启独立100条件共享动态队列。显存按进程峰值7680MiB预留，不按瞬时空闲盲目塞满；额外仿真由专属监督器保护，只处理本实验进程，失败partial不混正式分母。所有官方baseline、DiT4DiT、训练与数据保持原状。


## 2026-10-26T18:11:45.899940+00:00：用户确认各两份，已执行启动

36k模型GPU0（4 slots）和GPU2（8 slots）；heads-only42960模型GPU4/GPU6（各6 slots）。四份合计，其中GPU0复用既有已验收服务，新增三份正在加载。每组12独立policy session，完整Sidecar/eager+cache。两组自动启动器均已运行，等待模型动作验收通过后自动开启各自仿真和进度报告；36k的12端口reverse SSH已启动，heads本地独立队列。L20仿真仅GPU1/3/4/5/6/7，显存足够才入场；模型GPU4/6目前通常不足额外仿真。未知外部作业保持。仿真实际启动与数量以dispatcher/state及后续验收为准，不把后台等待视为rollout已启动。


### 2026-10-27T01:45:11.427553+00:00：36k与heads-only42960进度复核

{"fullbackward36000": {"conditions": 26, "active": 9, "errors": 0, "observed_episodes": 897, "comparison": {"all": {"conditions": 26, "episodes": 780, "successes": 560, "rate": 71.7948717948718, "official_same_subset": 71.92307692307692, "gap_pp": -0.1282051282051242, "previous32k_same_subset": 71.28205128205126, "delta_vs_previous32k_pp": 0.5128205128205394, "initial32k_first30_same_subset": 67.30769230769232, "delta_vs_initial32k_first30_pp": 4.487179487179475, "initial32k100_same_subset": 67.53846153846153}, "demo_clean": {"conditions": 14, "episodes": 420, "successes": 357, "rate": 85.0, "official_same_subset": 86.21428571428571, "gap_pp": -1.2142857142857082, "previous32k_same_subset": 83.80952380952382, "delta_vs_previous32k_pp": 1.1904761904761756, "initial32k_first30_same_subset": 84.04761904761905, "delta_vs_initial32k_first30_pp": 0.952380952380949, "initial32k100_same_subset": 83.57142857142857}, "demo_randomized": {"conditions": 12, "episodes": 360, "successes": 203, "rate": 56.388888888888886, "official_same_subset": 55.25, "gap_pp": 1.1388888888888857, "previous32k_same_subset": 56.666666666666664, "delta_vs_previous32k_pp": -0.27777777777777857, "initial32k_first30_same_subset": 47.77777777777778, "delta_vs_initial32k_first30_pp": 8.611111111111107, "initial32k100_same_subset": 48.833333333333336}}}, "headsonly42960": {"conditions": 9, "active": 11, "errors": 0, "observed_episodes": 430, "comparison": {"all": {"conditions": 9, "episodes": 270, "successes": 216, "rate": 80.0, "official_same_subset": 79.55555555555556, "gap_pp": 0.44444444444444287, "parent42960_same_subset": 80.74074074074073, "delta_vs_parent42960_pp": -0.7407407407407334, "initial32k_first30_same_subset": 76.66666666666667, "delta_vs_initial32k_first30_pp": 3.3333333333333286, "initial32k100_same_subset": 75.66666666666667}, "demo_clean": {"conditions": 5, "episodes": 150, "successes": 144, "rate": 96.0, "official_same_subset": 96.4, "gap_pp": -0.4000000000000057, "parent42960_same_subset": 95.33333333333333, "delta_vs_parent42960_pp": 0.6666666666666714, "initial32k_first30_same_subset": 97.33333333333334, "delta_vs_initial32k_first30_pp": -1.3333333333333428, "initial32k100_same_subset": 96.0}, "demo_randomized": {"conditions": 4, "episodes": 120, "successes": 72, "rate": 60.0, "official_same_subset": 58.5, "gap_pp": 1.5, "parent42960_same_subset": 62.5, "delta_vs_parent42960_pp": -2.5, "initial32k_first30_same_subset": 50.833333333333336, "delta_vs_initial32k_first30_pp": 9.166666666666664, "initial32k100_same_subset": 50.25}}}}
两组已运行约7.5h，日志持续动作进展、无基础设施错误。正式阶段统计仅纳入完成30条的条件；当前同已完成条件对官方36k差约−0.13pp、heads约+0.44pp，均为不同任务子集，不据两组当前均值判断强弱，不外推完整100条件结果。heads仿真L20、36k仿真4090且当前任务混合不同，吞吐差异不归因单一硬件或模型。


### 2026-10-27T02:33:22.419903+00:00：36k完成30条件的效果复核

{"all": {"conditions": 30, "episodes": 900, "successes": 630, "rate": 70.0, "official_same_subset": 70.83333333333333, "gap_pp": -0.8333333333333286, "previous32k_same_subset": 69.77777777777777, "delta_vs_previous32k_pp": 0.22222222222222854, "initial32k_first30_same_subset": 66.55555555555557, "delta_vs_initial32k_first30_pp": 3.4444444444444287, "initial32k100_same_subset": 66.53333333333333}, "demo_clean": {"conditions": 16, "episodes": 480, "successes": 415, "rate": 86.45833333333333, "official_same_subset": 87.5, "gap_pp": -1.0416666666666714, "previous32k_same_subset": 85.62500000000001, "delta_vs_previous32k_pp": 0.8333333333333144, "initial32k_first30_same_subset": 85.83333333333334, "delta_vs_initial32k_first30_pp": 0.6249999999999858, "initial32k100_same_subset": 85.125}, "demo_randomized": {"conditions": 14, "episodes": 420, "successes": 215, "rate": 51.19047619047619, "official_same_subset": 51.785714285714285, "gap_pp": -0.5952380952380949, "previous32k_same_subset": 51.666666666666664, "delta_vs_previous32k_pp": -0.4761904761904745, "initial32k_first30_same_subset": 44.523809523809526, "delta_vs_initial32k_first30_pp": 6.666666666666664, "initial32k100_same_subset": 45.285714285714285}}
只统计完成30条的30条件900条，630成功70.00%，官方同条件70.83%（−0.83pp）；Clean86.46 vs87.50，Random51.19 vs51.79。当前整体接近官方同子集，仍有任务差异：hanging_mug Clean23.33 vs49（−25.67pp），handover_block Random33.33 vs52（−18.67pp）；place_shoe Random70 vs51（+19pp）。非全100条件，不把整体接近称每任务达标，也不外推最终全任务SR。当前12仿真、错误0，在途旧任务继续原PID执行。官方为归档issue46的100条结果、本地每条件30，样本差异保留。


### 2026-10-27T02:34:28.138441+00:00：32k同条件对照

30完整条件、双方900条，32k628成功69.78%，36k630成功70.00%，+0.22pp；Clean+0.83、Random−0.48。14提升/10下降/6持平，暂基本持平，不称36k明显更好。逐条件查验32k参考同为30条，完整表docs/ROBOTWIN_FULLBACKWARD36K_VS32K_MATCHED_20261027.md。


### 2026-10-27T04:05:10.265416+00:00：36k已完成41条件与GPU6保护回退复核

{"utc": "2026-10-27T04:05:10.265416+00:00", "progress": {"utc": "2026-10-27T04:04:45.546703+00:00", "status": "running", "scope": "30episode trend; not standard100", "observed_episodes": 1391, "observed_successes": 1009, "executed_rollouts_including_retries": 1395, "executed_successes_including_retries": 1011, "completed_conditions": 41, "planned_episodes": 3000, "active_simulators": 11, "errors": 1, "episodes_per_hour_estimate": 197.58482961840943, "linear_remaining_hours": 8.14333774059183, "note": "linear ETA excludes task-mix correction; current short-task coverage is biased"}, "comparison": {"all": {"conditions": 41, "episodes": 1230, "successes": 918, "rate": 74.63414634146342, "official_same_subset": 75.65853658536585, "gap_pp": -1.024390243902431, "previous32k_same_subset": 75.12195121951218, "delta_vs_previous32k_pp": -0.48780487804876316, "initial32k_first30_same_subset": 71.21951219512195, "delta_vs_initial32k_first30_pp": 3.41463414634147, "initial32k100_same_subset": 71.60975609756098}, "demo_clean": {"conditions": 22, "episodes": 660, "successes": 583, "rate": 88.33333333333333, "official_same_subset": 89.68181818181819, "gap_pp": -1.3484848484848584, "previous32k_same_subset": 88.18181818181819, "delta_vs_previous32k_pp": 0.1515151515151416, "initial32k_first30_same_subset": 87.12121212121212, "delta_vs_initial32k_first30_pp": 1.212121212121204, "initial32k100_same_subset": 86.63636363636364}, "demo_randomized": {"conditions": 19, "episodes": 570, "successes": 335, "rate": 58.771929824561404, "official_same_subset": 59.421052631578945, "gap_pp": -0.649122807017541, "previous32k_same_subset": 59.999999999999986, "delta_vs_previous32k_pp": -1.2280701754385817, "initial32k_first30_same_subset": 52.80701754385965, "delta_vs_initial32k_first30_pp": 5.964912280701753, "initial32k100_same_subset": 54.21052631578947}}, "capacity_policy": {"memory_budget_mib": 7680, "memory_margin_mib": 1024, "max_simulators_per_gpu": {"0": 1, "1": 3, "2": 1, "3": 1, "4": 3, "5": 1, "6": 1, "7": 1}, "reason": "extra simulator trial rolled back on GPU6; memory below2048MiB or renderer buffer failure; no task protocol changes"}, "guard_rollback": {"gpu": "6", "free": 1952, "worker": {"status": "running", "gpu": 6, "endpoint": {"gpu": 6, "slot": 0, "port": 28960, "session": "/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward36000_deploy_20261026T165514Z/gpu6/session_slot0.json"}, "task": "place_object_basket", "mode": "demo_randomized", "pid": 3171031}, "action": "revert_cap_and_stop_only_newest_own_extra_sim", "utc": "2026-10-27T02:34:09.260538+00:00"}}
完整41条件918/1230=74.63%，官方同条件75.66（−1.02pp）、full-backward32k75.12（−0.49pp）；当前不同子集仍不可直接与之前总体均值比较。11路持续执行，12降11源于GPU6空闲1952MiB触发已设2048MiB保护，只停止新增place_object_basket Random进程并恢复GPU6上限1；旧任务未杀。首次3条1成功保留但不混正式SR，attempt2迁GPU1已13条4成功且动作持续，故此项恢复已验、错误记录保留1并不说无错误。GPU6双仿真峰值不足为本次已验证风险，后续不按瞬时显存重新盲目恢复2路。旧调度仍剩1条在途条件29/30，正常动作，不称卡住；排空后自动解锁原端口。官方标准判据/训练/数据/模型均未改。


### 2026-10-27T07:01:54.629604+00:00：36k完成68条件及第二次显存保护恢复

完整68条件1494/2040=73.24%，官方同条件72.87（+0.37pp），full-backward32k74.17（−0.93pp）；Clean89.90 vs官方90.71/32k89.71，Random55.56 vs53.94/57.68。观察覆盖约2130/3000，不把partial混正式SR。GPU4额外仿真于04:06:55空闲1796MiB触发保护，将上限3回退1；place_bread_skillet Random首次8条7成功未混最终分母，attempt2已完整完成。先前GPU6 place_object_basket Random重跑也完整结束，累计2保护事件、无最终failed。当前9路动作/日志持续；原9路已全部完成并于04:07:01按真实exit记录接续，旧父进程释放，原0/2端口可复用。GPU4/6多仿真峰值余量风险保持，不能因任务切换瞬时空闲盲目恢复额外路数。当前接近官方同子集，32k仍略高，非完整100条件结论。


### 2026-10-27T10:08:51.300547+00:00：用户授权36k完成后自动接续heads42960，监督已就绪

用户明确确认43960指此前42960。监督PID3397012每15秒检查36k，当前97/100；只有100原生结果复验通过才释放其5份模型并在L20按实时>=26000MiB尽量部署heads、最多8×3sessions，所有通过动作验收后4090启动剩余90条件。旧10完整L20条件246/300通过重新验收并保留；144旧partial独立历史不拼分母，90在4090完整重测，最终混合硬件须标注。GPU4/6既往多sim峰值风险已纳入新静态峰值预算/限流/监督，不能称已永久解决。训练/数据/全局核心不动、无下载、原生标准/不存视频/eager+cache保持。监督和准备路径/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/headsonly42960_4090_resume_all100_30ep_20261027T100232Z；docs/ROBOTWIN_HEADSONLY42960_RESUME_AFTER36K_20261027.md。此刻仅准备与等待，不宣称新评测已开始；真实触发结果后自动记日志/最终报告。
