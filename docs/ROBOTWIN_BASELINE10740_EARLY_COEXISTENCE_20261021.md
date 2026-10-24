> 2026-10-21 协议修正：旧eager pilot已单独保留并停止。正式baseline改用官方完整compile＋cache，当前根为 outputs/robotwin_eval/baseline10740_officialcompile_early_20261021，控制器2457886。详见 ROBOTWIN_INFERENCE_PROTOCOL_AUDIT_20261021.md；下文保留早期执行历史。

# Baseline10740 提前并行评测（2026-10-21）

更新时间：2026-10-21T03:15:36.919292+00:00。用户明确将启动条件改为：有空闲评估资源即可提前测试，不等待全部32k收尾。

## 已执行与当前边界

旧等待PID2060359已停止，原脚本/spec分别保存在supervise_wait_all_before_early_20261021.py和spec_wait_all_before_early_20261021.json。当前监督PID2448303/PPID1，模型0 PID2448335正在加载，物理GPU1，计划22400–22402三个独立同步端口。此时尚未宣称模型数值验收或policy rollout已经完成。

实际核验旧GPU1模型3486828三个端口均无running任务，global worker3/4/5全finished，/proc无活跃standard_eval引用，PID/start_ticks与固定记录一致；仅回收这份闲置32k模型及其编译子树，释放后GPU1约39037MiB空闲。其余7路32k原outer PID继续。未暂停/停止任何正在运行的32k评估或未知进程。

当前32k为93个完整条件、约9827/10000条，7项running、无queued/errors。独立全局调度2037013继续推进。32k监控改为watch_results_coexistence.py/PID2448546，仍读同一global状态，头部准确计算存活Sidecar模型而非继续硬写5；监控切换不影响仿真。

## 提前启动的资源隔离

首份baseline放GPU1，首先启3个worker。共存期间baseline仿真仅允许GPU3/4/7，每卡至多1个sim，free>=16000MiB，并同时统计Sidecar+baseline活动占用；它们在核查时没有本次32k活跃物理仿真。GPU3仍有未知namespace计算负载（util100%，初始free约40930MiB），不将它称为空卡、不操作未知进程；后续按实际free与逐次admission启动，不能保证其他作业不会增长。

后台scale_baseline.grow每30秒检查新的闲置32k模型：必须其全部worker finished、无running端口、无活跃standard_eval引用、PID/start_ticks匹配才回收。只在GPU1/4/7/0/5、free>=33000MiB尝试增加新baseline副本，最多3模型9worker；不使用GPU3作为模型卡，以保留仿真空间。新副本先独立数值门槛，ready后向同一全局待跑队列增加worker。若没有容量则等待，原baseline worker继续；全部32k完成后仿真放置可用所有GPU，每卡仍至多2个本机两组sim。

## 固定十任务与协议

只从当前双域都已完成百条的任务选10个最差官方差距，全部≤−10pp；selection.json已在任何baseline得分前固定。不再等最终全50排名，也不会根据baseline新结果换名单。先派发下表差距最大的10条件，再测其另一域，合计20条件/2000条，目标权重仍为已固定校验的10740。

| Task | 首测环境 | Sidecar32k | 官方 | 差距pp |
|---|---|---:|---:|---:|
| beat_block_hammer | demo_randomized | 33% | 55% | -22 |
| open_microwave | demo_clean | 61% | 82% | -21 |
| place_bread_skillet | demo_randomized | 41% | 61% | -20 |
| hanging_mug | demo_clean | 33% | 49% | -16 |
| stamp_seal | demo_clean | 73% | 89% | -16 |
| adjust_bottle | demo_randomized | 37% | 51% | -14 |
| click_alarmclock | demo_randomized | 65% | 77% | -12 |
| handover_block | demo_randomized | 40% | 52% | -12 |
| place_burger_fries | demo_randomized | 70% | 80% | -10 |
| place_cans_plasticbox | demo_randomized | 86% | 96% | -10 |

wxyz/自有EEF统计、10去噪步、32动作、CFG1、sync、cache .99/max3、原官方RT/预算/seed筛选/成功判据/视频不改。新baseline先以两个真实输入比较匹配cache的compile与eager，原atol=.005/rtol=.01和实际kernel门槛；不通过就cache+eager，不降低数值门槛。后续每项完整exit0/100/无errors才入comparison.json/md，基础设施异常保留attempt并最多两次完整恢复。

资源共存、真实baseline加载/compile/闭环尚须实际验收；显存瞬时满足不保证峰值或未知作业增长安全。Baseline配置5epochs和自有stats，与Sidecar32k训练预算不同；两者结果只能诊断checkpoint差异，不能纯归因Sidecar或断言官方虚报。

CPU验收early_preflight_20261021.json确认当前十任务两域均完整、差距阈值及新增入口语法；还没有用CPU检查代替GPU数值/闭环。日志：supervisor_early_20261021.log、model0/server.log；状态：parallel.json/status.json。


## 2026-10-21 基于实际空余容量扩展 baseline 仿真

用户明确要求利用空闲GPU尽快产出分析结果。现场可用显存（MiB）GPU0/1/2/3/4/5/6/7约20772/14501/37656/39029/14891/20815/31254/15817；已部署3个baseline，但仅3个sim运行，根因旧共存规则只放行GPU3/4/7且每卡1sim。不是已有吞吐A/B证明一对一更优。

复用已验收的rebalance_remaining.py进程接管实现，新增输出内expand_capacity.py；冻结旧父2457886，仅暂停派发，2461744/2461942/2462124原sim与全部3个模型保持存活。新控制器2550008沿用canonical parallel.json，3模型×3独立端口、共9worker、全部GPU按剩余显存>=16000MiB且与Sidecar合计每卡最多2sim入场，串行观察分配，100条完整验收与至多两次基础设施重试不变。必须保存所有adopted子进程真实exitcode后才结束冻结旧父，禁止SIGCONT旧父造成重复派发。固定20条件、compile/cache/权重/stats/官方RT/预算/筛seed/成功判据/视频不变；原结果与原代码保留。baseline10740_current.json已指向新控制器。

此前测速只有serial阶段4条/1296.34s=11.11条/小时，parallel2在第二仿真启动前因容量变化中断，不能报告加速倍数。新增单独测速监督2551130，复用闲置Sidecar32k GPU0服务3486827的独立端口21600/21602；等待正式baseline9sim入场后，选择两张free>=16000MiB的固定sim卡，两组adjust_bottle/Clean×2条，serial再parallel2，两阶段同卡/同服务/同协议。记录启动/首动作/首episode/完整wall/步数/accepted seeds/显存峰值/聚合条数每小时，所有测速结果从正式SR排除。小样本、背景负载/随机指令及CUDA RNG交织仍是限制；该测速对象是Sidecar而非vanilla compile baseline，不以其倍数直接外推baseline。正式队列优先，测速若容量不足/异常明确报告而不抢占已有作业。

阶段启动与输出目录见capacity_expansion_20261021/{source_snapshot.json,controller.log,process.json}及Sidecar根throughput_ab_20261021/state.json。下一步确认9路真实动作与完成条数增长，待完整条件报告官方差距；待测速两阶段均完成才得出一对多的吞吐结论。


2026-10-21 07:22 UTC 容量细化：首轮扩至7路后，GPU2/6虽仍有大量显存，但通用每卡2sim上限再次限制入场。因此恢复已有Sidecar全局调度已使用的GPU2/6最多3sim规则（GPU2 free>=20480MiB、GPU6>=16000MiB，其余卡最多2sim），合并统计两批活动sim。新增expand_capacity_balanced.py，控制器2553273，canonical状态与指针已切换；冻结2550008，保留其4路及2457886的3路，按每个实际父进程的adopted子任务分别收齐真实退出码后再结束父进程，禁止恢复旧调度器。07:21已8路，hanging_mug/stamp_seal已出现真实动作；第9路仍按串行容量观察入场。原3模型无重新加载，所有原仿真PID保持。该调整不构成单/多仿真吞吐提升证据，仍等待独立测速。

2026-10-21 07:23 UTC：baseline正式9路全部已派发（3模型各3端口），6路已有真实动作，其余初始化；errors为空，原3路PID保持。独立测速state=running/serial，首个adjust_bottle/Clean两条job PID2554666、sim GPU2，随后同两张固定sim卡测parallel2；尚无加速倍数。证据capacity_balanced_20261021/admission_acceptance.json。


## 2026-10-21T07:35:39.325448+00:00：baseline后续任务关闭视频保存

用户明确要求设置不保存视频以尝试加速。当前无完整100条条件；日志最接近hammer Randomized43/83、skillet Randomized33/73、microwave Clean31/42。仅在本次baseline10740根policy_config.yml显式添加eval_video_log:false，原配置备份保留。wrapper增加可选布尔配置：只在新任务私有runtime内复制两域task config并改该视频标志，其余配置仍链接官方原件；源RoboTwin/DiT4DiT/其他未设标志实验不变。保存源/私有config hashes到runtime/video_logging_override.json，run.json原有policy snapshot及wrapper hash继续记录。

验证：两域解析结果均只差eval_video_log、源文件字节不变、默认无flag仍保留原task_config链接，临时runtime检查通过。当前9路已加载配置，继续保存视频至各自结束，不重启、不删除已生成视频；随后新条件及异常完整重试使用关闭视频配置。视频标志移除逐帧raw RGB pipe/FFmpeg编码/文件I/O，但模型观测的相机渲染及物理步进仍须保留；未实测墙钟加速比例，不能承诺显著提速或直接据不同任务用时作因果比较。现有单/双sim测速仍按原有保存视频协议，两阶段不混改。下一步检查第一项新启动日志private eval_video_log=False及私有JSON、无ffmpeg进程/新mp4；如需量化收益应同任务同manifest单独video on/off对照。

证据：本次根video_logging_change.json、policy_config_before_no_video.yml、eval_policy_wrapper_before_no_video.py。


## 2026-10-21T09:10:44.626574+00:00：32k尾项、baseline首两项完成及单/双仿真测速

32k仍为99/100完整、9983/10000 episodes；put_object_cabinet/Randomized12/83，当前577/700步，日志即时更新、errors空，非卡死。Clean全50为87.66% vs官方89.36%。

baseline10740已有2/20条件完整、458/2000条、9路运行/9条件排队、errors空。hammer Randomized51/100（官方55，32k33，baseline比32k+18pp）；skillet Randomized48/100（官方61，32k41，比32k+7pp）。其余阶段microwave Clean45/61，hanging_mug Clean8/14，stamp_seal Clean38/43，adjust_bottle Randomized2/17，click_alarmclock Randomized43/58，handover_block Randomized5/11，burger_fries Randomized36/38；后续cans Randomized6/6、hammer Clean10/10。不同训练预算/自有stats、非逐指令/RNG配对，不能纯归因模块。

独立Sidecar32k单/双仿真pilot已完整结束，所有8次episode成功、两阶段相同accepted seeds[100001,100002]、固定simGPU2/3及GPU0服务。serial两组各2条总wall1076.919s、13.371条/小时；parallel2同4条wall419.849s、34.298条/小时，观测总吞吐2.565倍。包含启动、环境初始化、expert筛seed、退出，且后跑parallel有热文件缓存/背景负载差异，不能当稳定长跑加速倍数或纯因果结论。动作步总数serial461、parallel479不同，非逐动作/指令CUDA随机严格配对。首动作到单job退出约serial266.6/262.3s、parallel174.9/186.3s，包含episode reset/退出，不等于逐条纯rollout秒数。只能支持本次pilot一服务带两仿真比串行获得更高观测吞吐，不能直接外推baseline三路或全任务。测速不混入正式SR。证据Sidecar根throughput_ab_20261021/state.json。

后续baseline两个新条件已验证private video_logging_override.json=false且无新mp4；原正在运行条件继续保存。证据：[{"path": "outputs/robotwin_eval/baseline10740_officialcompile_early_20261021/capacity_balanced_20261021/jobs/demo_randomized/place_cans_plasticbox/attempt1/demo_randomized/place_cans_plasticbox/runtime/video_logging_override.json", "enabled": false, "mp4_files": 0}, {"path": "outputs/robotwin_eval/baseline10740_officialcompile_early_20261021/capacity_balanced_20261021/jobs/demo_clean/beat_block_hammer/attempt1/demo_clean/beat_block_hammer/runtime/video_logging_override.json", "enabled": false, "mp4_files": 0}]。下一步待32k真实百条终态及baseline更多完整条件，保持既定协议与队列。


## 2026-10-21T09:39:21.837964+00:00：用户要求继续利用空闲GPU扩容、不保存视频

现场GPU0有约24GiB闲置Sidecar32k服务3486827，独立单/双sim测速已complete且无活动引用。调用既有release_previous_models，经PID/start_ticks、三个对应worker finished、无running端口与/proc客户端引用门槛后仅释放该闲置服务；活跃的最后Sidecar任务仍在GPU6仿真/GPU5服务，不操作。GPU0 free变45491MiB。新增baseline副本model3 PID2640695，GPU0/ports22430–22432，相同10740 pinned权重、自有stats、官方完整MoT compile+cache、同两fixture四组合真实推理验收，不静默fallback。启动时CPU依赖/权重加载中，不声称已ready。

新增activate_model4.py监督PID2641965等待独立validation.json/pool与12个端口真实ping通过，再冻结当前2553273仅派发，快照接管所有正在运行PID、保留真实父进程退出码生命周期，追加第4模型与3worker，共12。新的expand_capacity_model4.py复用已有adoption/完整100条验收/最多两次基础设施重试，并把容量配置独立到capacity_limits.json以便后续无需叠加调度器。GPU2/3/6最多3sim，其余最多2；GPU2 free>=20480MiB、其余>=16000MiB，始终合并Sidecar占用并串行观察分配。GPU3此时两sim/free约33968MiB，可再加一个sim；不在有活动仿真的卡冒险加载额外24GiB模型。

新增3项按原全局队列取open_microwave Randomized、place_bread_skillet Clean、hanging_mug Randomized（以加载结束时实际队列为准），不重排已有结果/不重复计分。baseline policy_config.yml已设eval_video_log:false，所有新任务私有runtime关闭视频，当前已在跑的旧配置不重启。原3模型/9sim在新模型加载期间继续；只在验证通过后接管，失败记录model4_handoff_status.json，冻结后尚未派发新控制器的失败会恢复原派发，不重复SIGCONT任何旧冻结父。未更改模型/训练/数据/RT/预算/成功判据/seed。

下一步核对第4副本官方compile验证通过、12路真实动作与新增私有video_logging_override.json=false；本轮未启动新的保存视频A/B，GPU优先成功率评测。证据本次root/model3、model4_handoff_process.json、model4_handoff_status.json与capacity_model4_20261021。


## 2026-10-21T13:21:24.388375+00:00：Sidecar32k全10000条完成；baseline四模型12路已接入

32k100/100条件、6648/10000=66.48%，官方69.04%，差-2.56pp；Clean4383/5000=87.66% vs89.36%（-1.70pp），Randomized2265/5000=45.30% vs48.72%（-3.42pp）。最后put_object_cabinet Randomized13/100（官方17），完整run.json全部逐项验证，无最终errors。解决此前未完成全50双域平均的事项；固定manifest cache A/B、EEF物理跟踪及训练预算/统计因果限制仍未解决。详见docs/ROBOTWIN_SIDECAR32K_FINAL_RESULTS_20261021.md及final_results JSON/CSV。

baseline第4模型已09:46:15 UTC验证并接入，新控制器2648154，当前4模型/12路真实推进，20条件全部启动。13:20快照8/20完整、1266/2000条，errors空；完整条件SR：hammer Randomized51、microwave Clean72、skillet Randomized48、stamp Clean88、alarm Randomized67、burger_fries Randomized83、hammer Clean100、alarm Clean100。其余12条件日志更新，不以部分分母替代100条终态。新增条件关闭视频保留物理/渲染/评判，旧运行任务未重启。继续完成baseline并对照官方及32k；无新增模型/训练/数据逻辑修改。


### 2026-10-21T17:05:38.591646+00:00 baseline10740收尾进度

17:05 UTC现场日志1825/2000条（91.25%）、14/20条件完整、6运行、0排队、errors空。新增完整：adjust_bottle Randomized27%（官方51，-24pp）、cans Randomized99%（官方96）、skillet Clean95%（官方92）、adjust_bottle Clean96%（官方90）、handover_block Clean95%（官方84）、burger_fries Clean97%（官方97）。剩余：hanging_mug Clean42/82、handover_block Randomized25/74、microwave Randomized0/24、hanging_mug Randomized7/59、stamp Randomized36/96、cans Clean89/90。当前最慢microwave Randomized尚余76条、预算1500步；日志更新不能称卡死，部分结果不作最终SR。32k已完整66.48%（Clean87.66/Randomized45.30），保持既定对照和完整终态验收。


## 2026-10-22T14:17:02.241914+00:00：baseline10740十任务双域全部完成

20/20条件、2000/2000条，全部原生run.json/真实退出码0核验、errors空。最后open_microwave Randomized3/100，官方7。Clean891/1000=89.10% vs同任务官方87.90%（+1.20pp）；Randomized463/1000=46.30% vs51.80%（-5.50pp）；综合1354/2000=67.70% vs69.85%（-2.15pp）。完整逐任务与同范围32k见docs/ROBOTWIN_BASELINE10740_FINAL_RESULTS_20261022.md及输出final_results_20261022.json。组会03:48旧快照保留，不把不同快照/不同覆盖均值混用。此十任务按32k负差预选、训练预算/stats不同，非全50或纯模块消融。下一步如需组会最新三组对照，以本最终baseline和42960新快照另生成版本。
