# PatchWAM ramp100 step20000 全任务快速评测

用户授权：等待32k完整验收后，停止16000，改20000，优先4090多仿真、L20少量足够的模型。2026-10-29T13:36:52Z，32k最终独立验收100条件/10000条通过，综合7019/10000=70.19%（官方69.04%），Clean88.02%（官方89.36%），Randomized52.36%（官方48.72%）；该结果是原30+补70，指令候选池分段口径仍保留。

16000已停止CPU调度与3个自有在途sim，保留4个完整30条件44/120=36.67%及全部partial/infra日志，stopped_summary.json可复核，不将partial当正式SR。六份完成的32k模型与16000模型均已核验PID身份后退休，未触碰训练或未知GPU作业。

## 新权重与推理

指定 `outputs/sidecar/aloha50_patchwam_c2r_ramp100_10epochs_20261027T144939Z/F_patchwam_ramp100/runs/2026-10-27_14-57-15/checkpoint_step_20000.safetensors`。
固定SHA256：`65626b3eecb54f5291a2dfa3bb7bd177e7846701099a28dc88eb711f4caedfdc`。独立RAM副本只含该权重与原config/normalization/contract/tokenizer，复制与复读hash相同，避免训练删除/保存/最新选择变化。无网络下载。

L20先两份完整BF16模型（GPU1/4），各5独立PolicyServer/WAMPolicy/action deque，共10个session；模型内推理串行，各模型并行。保持完整Sidecar、cache .99/max3、compile关闭、denoise10、horizon32、384×320输入、绝对EEF合同。单卡准入35000MiB空闲，原始用户空间服务入口复用，不新增架构。两个真实fixture重复一致与两模型canonical精确一致、每session重复chunk32在线验收通过才开放评测。

端口37910–37914、37920–37924通过独立反向SSH转发，独立token/目录，所有进程身份保存；不复用旧checkpoint会话，不展示token。2副本是否足够看实际队列/吞吐，不能仅凭此前别的任务5%–10%比例承诺10sim无等待。

## 官方任务协议与仿真调度

100条件（50任务×Clean/Randomized）各30条，总3000；精确列表plan.json。优先此前弱项Randomized，但低显存卡可先做Clean。保留官方专家筛seed、seed组0从100000升序、unseen候选指令30、原生成功判据/动作预算/250Hz/RT相机与near0.1；无视频。推理采用已验收native chunk32＋相机采集gate，中间不采图但保留场景同步、随机灯光RNG、实际EEF/qpos、逐步规划/物理/成功。PatchWAM训练增强不在推理重复施加。

4090目标最多10sim，实际动态取决于当前未知GPU作业/峰值。每卡上限0:1、1:2、2:1、3:1、4:2、5:1、6:2、7:0；已知失败的GPU1第三路/GPU6第三路不重开。准入>=10240MiB可跑任意条件；8704–10240仅试Clean，避免重走16000 GPU5的Randomized峰值失败。每卡启动保留7680MiB预算并错开；只在实际开始步进或180s后释放reservation。

每3s记录空闲，低于1536MiB仅最新自有sim退让，并把该卡cap降至其余活动数；若原来单路则cap0，本实验不自动重开，任务到其它卡整30重测，禁止同卡反复耗尽retry。PID新旧按started_wall确定，避免PID回绕选错。30分钟无进展只处理自有进程；每条件本轮最多6次infra尝试，完整结果才入SR。最小余量1.5GiB是受保护试验，不是已证明所有任务的峰值边界；高阶增长/真实OOM仍可能发生，必须保留失败信息。准入/模型负载平衡/降容保持6项CPU实际函数检查通过。

## 运行文件与报告

实验根 `outputs/robotwin_eval/patchwam_ramp100_20000_20261029`。
- activation.log、gpu*/eager_validation/result.json、deployment_acceptance.json：部署/在线验收。
- state.json、progress.json、scheduler.log、admission_policy.json：实际任务/并发/容量退让。
- comparison.json：只统计完整条件，与32k原30相同条件及官方同条件对比。
- [实时结果](ROBOTWIN_PATCHWAM_RAMP100_20000_RESULTS_LIVE_20261029.md)：自动刷新。

候选指令池30与官方完整100不同；同环境seed未冻结Python random/确切语言，故不宣称跨ckpt输入manifest逐条完全相同。近裁面、实际vs目标EEF、颜色语义增强风险继续保留。已有物理/控制漂移边界不因采图CPU/动态RT验收被宣称全解决。

下一步：确认真实10个独立会话可用及每路实际action推进，观察低余量Clean试验是否通过峰值；若推理排队明显再增加第三模型，而非先复制大量模型占满L20。两模型和各GPU实际稳定吞吐需实测，当前准备/启动阶段不提供20000 SR。

## 2026-10-29T13:47:40.469927+00:00 实际接入

两份L20模型PID3618817/GPU1、3618818/GPU4，两个真实fixture重复/跨副本canonical精确一致门槛通过；10个独立端口各重复chunk32在线门槛全部通过。反向SSH已建立。4090调度PID335459，当前实际启动9路，余量不足的会话等待。精确任务/GPU/PID及源码SHA见activation_snapshot.json。此时仍在native初始化/专家筛seed，不虚报rollout或SR。新增RPC计时额外记录服务端latency_ms，可从往返时间与服务端时间观察推理/排队，但不能将所有差额当纯网络。实际renderer/CUDA分配失败也会降该卡cap，禁止忽略真实OOM后重复准入。

## 2026-10-29T14:22:54Z 实际吞吐核查

9sim/2模型，累计209条、4完整条件。14:06:52→14:22:54共16.04min增长101条，377.84条/小时；13:46:29启动以来36.42min为344.32条/小时（含初始化）。不是精确15/30min窗口，本轮未满1小时，不编造滚动1h数据。此前进度未持续存档，现新增独立只读10秒sampler；历史种子仅采用工具此前实际返回的四个精确快照，标source=verified_prior_tool_read，不插值。

2013次真实chunk RPC平均0.5474s，服务端平均0.4699s，差额0.0775s（包含序列化/网络/排队，非纯通信或纯队列）；RPC P95为0.8888s，差额P95为0.4273s。控制热路径动作规划执行86.36%、观测6.41%、客户端含推理通信等待7.23%，不含专家/setup等总墙钟未计部分。当前两模型负载下未见推理成为主要瓶颈，无需仅因想扩sim先加模型。GPU4第二路首次free1437<1536退cap1，后续不重开，当前9路稳定推进；其它低余量Clean峰值风险仍持续保留。证据throughput_check_20261029T1420Z.json；不同任务组合不可把此吞吐相对旧实验差异归为纯加速因果。

## 最终完成

2026-10-30T00:40:44Z完成100条件/3000条。2093成功=69.77%；Clean84.20%、Randomized55.33%。全部原生结果重新独立审计通过。最终分析见[最终结果](ROBOTWIN_PATCHWAM_RAMP100_20000_FINAL_20261030.md)。
