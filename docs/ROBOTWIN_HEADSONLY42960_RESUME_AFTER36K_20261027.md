# 最新状态：42960已恢复评测

核验时间：2026-10-27T12:50:54.827815+00:00。此前自动接续在旧模型退出的身份检查上失败；退出竞态已修复并经过21项真实进程检查，已恢复运行。

36k已完整完成3000条：2108成功，**70.27%**；Clean89.33%，Randomized51.20%，官方69.04%，同协议32k71.10%。完整结果见[36k最终报告](ROBOTWIN_FULLBACKWARD36000_30EP_FINAL_RESULTS_20261027.md)。

heads-only42960：L20 GPU1/2/4/5/6/7六份模型全部ready，18端口在线真RGB/state→20D动作验收通过；4090实际已启动7路native仿真，当前可用显存不支持此前9–12路，按峰值预算限制、不停止其他作业。原10个完整L20条件保持，剩90在4090完整30条；当前进程尚初始化，不把旧300条误报为新的rollout或新SR。source头部e4c8c2d…权重RAM和config/stats/contract重新哈希通过。完整Sidecar/cache开compile关、官方标准、不存视频保持。

运行目录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/headsonly42960_4090_resume_all100_30ep_20261027T100232Z`，`state.json`、`live_progress.json`、`auto_transition_state.json`、`resume_start_acceptance.json`。后续真实动作进展由只读observer写`first_rollout_acceptance.json`。原100条件汇总将混合10条件L20＋90条件4090，比较时仍需标注硬件。

以下为历史准备/恢复记录，以上文为准。

# 36k结束后自动接续heads-only42960

记录时间：2026-10-27T10:08:51.300547+00:00。用户已确认本次所说43960为此前中止的42960：
`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_heads_only_from40000_to10epochs_20261025T171338Z/runs/from40000/checkpoint_step_42960.safetensors`。

SHA256 `e4c8c2d41a7f12902dd5ca5dbd807b2ea8b1ad982909433514d9ddaf916b1f35`。

## 当前自动监督

L20后台进程PID 3397012已真实启动并核验存活，每15秒读取36k原生队列终态；当前阶段`waiting_for36k`，36k已完成97/100条件。此状态是等待，尚未宣称新的heads服务/仿真已启动。监督具有flock防重复，不使用阻塞50分钟的交互会话。失败会写failed_needs_inspection和完整traceback，保留证据，不擅自修改模型/加速/数据或终止未知作业。

监督目录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/headsonly42960_4090_resume_all100_30ep_20261027T100232Z`。状态`auto_transition_state.json`，日志`auto_transition.log`，进程`auto_transition.process.json`。准备部署目录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/headsonly42960_4090_resume_deploy_20261027T100232Z`。

## 自动执行顺序

1. 等36k真实status=complete且100/100条件，不按时间猜完成；逐份重读run.json真实exit及native30结果，核对总3000条、成功数、唯一条件。自动写36k最终JSON/CSV、独立审计和`docs/ROBOTWIN_FULLBACKWARD36000_30EP_FINAL_RESULTS_20261027.md`。若失败不切服务。
2. 仅核验并停止已完成36k自有5份模型和自有两条reverse SSH；不停止训练或其他作业。保留所有旧结果/配置/日志。
3. 重新校验heads权重RAM SHA及config/statistics/contract。重新查询每张L20显存，仅>=26000MiB的卡启动，最多8份，全部同一完整Sidecar/eager+cache，每模型3独立session（最多24），匹配现有24线程上限。端口优先30900+10×GPU，slot0–2；冲突时顺序尝试31900/32900/33900基址，选择结果写deployment。动态内存变化或失败副本有独立记录；只使用通过重复动作/跨副本/在线真请求验收的副本，不预先宣称能放满8份。
4. 新reverse SSH只映射本次独立端口；4090对实际全部端口做真RGB/state→20D动作验收，再启动原生队列及报告/只读性能观察。启动成功后才更新heads主current指针。
5. 首次在4090完全空闲于本实验时按7680MiB峰值+1024MiB余量计算容量。GPU4最多2路；GPU6此前双路触发过显存保护，仅当本次静态余量>=2×7680+3072MiB才允许2路，否则1路。GPU1上限3，其他上限2；实际不超过按预算可容纳数。有额外路时启用低于2048MiB/渲染buffer失败回退，仅处理本实验新额外进程。保留先前GPU4/6峰值风险，不按瞬时任务切换空闲盲目恢复3路。

## 接续数据规则与比较边界

先前heads10完整条件246/300=82.00%均已逐份原生验收，本次保持其原结果和路径，标注L20。余90条件全部新在4090完整测试30条；中断的144条有独立历史attempt字段及原目录，不能拼成新的30条。新3000条覆盖由300条已验收+2700条新完整评测组成，不将历史partial用于SR或吞吐分母。

因此最终100条件为**前10条件L20、其余90条件4090**，不得标为全4090结果；与父模型或官方比较必须明确此边界。每个新条件30条只是趋势统计，官方issue46为每条件100条，样本差异保留。训练/数据不改，完整Sidecar能力保持，cache .99/max3开、compile关、原始动作chunk32/10去噪步、官方expert筛seed/unseen/RT/预算/成功判据、不存视频均保持。near/EEF replay与严格配对种子等旧未解项不因自动流程视为已解决。

## 已完成的准备验收

全部脚本AST检查；100唯一条件、10 complete+90 queued，原10结果逐项相等；90 queued新attempt为空、历史中断记录独立保留；4090设备筛选及RT缓存路径使用已验证原版36k运行模板。后台监督启动时已重新验收旧10条件native结果后进入waiting_for36k，尚未消耗新增模型GPU显存。实际新副本加载/新4090 rollout验证在触发后由部署和原生监控完成，不把工程准备称为模型效果。


### 2026-10-27T12:40:43.147903+00:00：自动接续回收竞态修复并真实恢复

36k于10:39:13完成100条件3000条、2108成功70.2667%，Clean1340/1500=89.3333、Random768/1500=51.20；独立native重验通过，对官方+1.2267pp、对32k−0.8333pp。原自动接续10:39:29在停止GPU0 launcher后、再检查其child时遇进程退出过渡cmdline为空，身份断言使脚本退出，heads未按预期启动；不能把这两小时称为评测时间。原错误日志和脚本已独立保留，36k结果与heads原10条件不受影响。

现修正stop_owned：对空cmdline退出过渡重读/幂等处理；对非空不匹配仍拒绝，使用pidfd固定进程身份后发SIGTERM，避免PID复用误停。当前Python未提供os.pidfd_open，按本机x86_64内核头文件434/424使用内核接口，原生21项真实子进程退出/重复停止/错误身份不发信号检查通过。恢复PID3404251已重新核验全部36k结果，身份正确的其余4份旧模型已停止，进入heads权重RAM哈希与重新查显存阶段。只修本实验接续脚本，不改模型/评测成功判据/数据，不停止其他工作负载。每轮后续以auto_transition_state/模型验收/原生rollout为准，不预称已运行。


### 2026-10-28T01:44:35.033449+00:00：heads42960续测62完整条件复核

{"all": {"conditions": 62, "episodes_each": 1860, "rate42960": 74.19354838709677, "rate32k": 74.89247311827957, "delta_pp": -0.6989247311827957}, "demo_clean": {"conditions": 33, "episodes_each": 990, "rate42960": 90.0, "rate32k": 89.29292929292929, "delta_pp": 0.7070707070707071}, "demo_randomized": {"conditions": 29, "episodes_each": 870, "rate42960": 56.206896551724135, "rate32k": 58.50574712643678, "delta_pp": -2.2988505747126435}}
完整62条件1380/1860=74.19%，官方同条件74.26（−0.06pp），修改前42960同条件73.66（+0.54pp），fullback32k74.89（−0.70pp）；Random对原42960+1.72pp，对32k−2.30pp。整体覆盖约1965/3000，7路持续、基础设施错误0。最终仍有10L20/90RTX混合硬件及30条样本边界，未作完整100条件结论。


### 2026-10-28T02:39:15.513060+00:00：按用户提示利用4090 GPU1/4余量，各加一路仿真

新增前GPU1空闲约16.7GiB、GPU4约15.3GiB，各原有1路；近30min最小空闲GPU1为5526MiB（有外部负载波动），GPU4为14434MiB。故只各加1路到2路，不盲目开3路；现场preflight两卡均>=11000MiB，guard接续新增需>=7680+2048+1024=10752MiB。总仿真7→9，新增place_a2b_left Clean在GPU1、Randomized在GPU4；实际native PID/正确渲染亲和性已记录，此时尚场景初始化不宣称新episode已终态。原7任务逐一PID相同或已完整结束，100条件唯一、错误0；L20六模型/18端口与原生30条/不存视频/标准判据保持。

为防两个策略写入器竞争，只替换CPU容量监督，不重启模型/仿真/dispatcher。最小扩展原resume分支，使已active GPU6的2路保护原样保留，并把满足现场峰值余量的新GPU1/4加入监测。原代码/config/status/policy/process均备份到/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/headsonly42960_4090_resume_all100_30ep_20261027T100232Z/gpu14_extra_trial_20261028T023634Z；当前active_caps={6:2,1:2,4:2}。低于2048MiB或渲染buffer失败会恢复该卡上限1并仅停最新本实验额外进程；中断partial不拼正式SR。单路/双路历史风险未永久解决，需继续观察完整episode峰值；尤其GPU1外部作业恢复占用时应自动退回，不以当前瞬时空闲继续叠加。


### 2026-10-28T02:43:20.142818+00:00：4090GPU5初始禁用上限随资源释放恢复一条

用户追问GPU1/4/5/6为何仍有空闲。已确认1/4新增路进入动作执行、6两路常态运行；5原启动自由3453MiB，初始cap0未随外部资源释放自动更新，当前8849MiB且30min稳定。以既有原生单路预算7680+1024=8704MiB准入恢复GPU5一条place_a2b_right Randomized，总9→10。对zero-native卡复用原标准首次进程门槛；已有仿真卡扩增仍保留10752MiB门槛，未改变模型/原生协议或全局预算。

CPU监督单写入器原样保留6/1/4 active，再加入5 cap1（original0），低于2048MiB会退回0并只停本实验新native，首次峰值仍需观察。新增GPU5已真实PID启动，但仍初始化，未称满episode稳定；原9个任务PID不变或自然完整结束，100条件唯一、错误0。记录/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/headsonly42960_4090_resume_all100_30ep_20261027T100232Z/gpu5_extra_trial_20261028T024206Z，现1=2/4=2/5=1/6=2。尚余显存是固定上限与场景峰值留量的结果，不一概称还在加载；GPU1/4第三路待首条完整rollout/峰值检验及外部占用波动评估，不盲目占满。


### 2026-10-28T04:10:29.792364+00:00：用户授权当前heads后32k每条件补70，队列已运行

目标full-backward32k source2a74859c…，原100条件×30=3000条2133成功保留，按每条件最后accepted seed+1接原生70（7000新条）形成10000条100/条件。6模型/当前9仿真安全目标切换时刷新，GPU5新额外进程曾触发2032MiB保护，保持cap0不盲目恢复。队列监督PID3459612，只在heads真实complete及100份结果重验后切换；当前heads继续，不提前分GPU。新进程内seed适配不改官方全局source，100条件CPU和真实分母helper联合检查通过，实际native70和终态100seed互斥检查待运行。

新发现关键口径：eval_policy.py把test_num传给generate_episode_descriptions作为候选指令上限，原30+70与native100指令池可能不同；这也适用于历史30条趋势结果的官方对比限制。已向用户说明并提供严格新100 vs按原补70选项；未收到更改前按用户明确补70排队，报告标分段扩展100条、不称严格官方100逐输入等价。若选严格100需完整重测，不把沉默当额外10000预算批准。详见docs/ROBOTWIN_FULLBACKWARD32K_COMPLETE100_QUEUE_20261028.md，/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward32000_supplement70_4090_20261028T035728Z。
