# 42960 Sidecar 写回诊断：独立配对实验

2026-10-21T09:39:17.976543+00:00。用户授权GPU4/7，对当前最终42960 checkpoint做A写入幅度与B仅关闭write的诊断，并少量配对闭环。输出：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar42960_write_diagnostic_20261021T092505Z`。状态：已启动；CPU权重已就绪，固定场景的专家筛选/初始观测采集仍进行中，不能宣称已有模型效果结果。

## 固定条件

权重SHA256：501addbed687e6d7587af60ab4480b9c79eaa2c8867d723b74d54e0acc9deb55。复用已固定42960权重的独立hardlink与自有config/stats/tokenizer；不改训练或正式42960评估。3任务adjust_bottle/place_shoe/beat_block_hammer，两域Clean/Randomized，各3个固定专家可解场景，共18对/36rollouts。基于同一RoboTwin commit0aeea2d669c0、同一官方A100 SAPIEN/curobo sm80运行时，RT/原动作预算/unseen，使用已有sealed manifest协议冻结seed与指令，不混入原native百条正式成绩。

## A：测量实际残差

在tap7/15/23写回前后，用FP32计算实际BF16 hidden之差与原hidden范数之比；保存mean/p50/p90/p99/max和逐样本逐Action/Video时间步记录，同时保留FP32 beta×projection的加法前参考比值以区分加法舍入。主诊断cache off，10步全部执行，策略噪声seed42、同一轨迹sigma schedule；另做与当前正式部署相同cache .99/max3结果，后者只统计实际执行的forward，不能把跳过步当0写入。当前新增统计是进程内hook，不修改production模型源码。

## B：只关闭write

同一模型串行运行Full与Write Off，Sidecar read/refine/head路径保留，只在每个write调用期间把内存beta置0并finally恢复，不写checkpoint文件。与旧sidecar_ablation.py/off绕过整个Sidecar不同，不复用那个off模式。对每个原始RGB/proprio/prompt固定输入，保持seed42和10步，计算归一化action相对差异、物理EEF20相对差异、双臂目标位置mm及旋转deg/夹爪差异。位置指标是Full与Off预测的物理EEF目标差异，不是相对GT或真实机器人跟踪误差。

cache-off同时保存Full轨迹的每次noisy_actions/video latents/context/timestep，随后Full/Off在同一noise输入上重放单次forward，报告active20维action velocity差异，避免把两条去噪轨迹后续自然分叉误当完全同噪声输入的单步差异。完整10步action输出比较允许这种分叉，它正是干预的最终影响。对首个样本Full→Off→Full重复要求输出逐位一致；FFN在两个既有真实fixture上与相同cache的eager参考通过现有数值门槛才放行。

## 部署和配对闭环

GPU4当前已有正式42960模型，不覆盖/停止。诊断只加载一份CPU权重；在GPU4对全部固定prompt预先计算CUDA文本embedding并转存缓存，再把T5放回CPU，不改变文本计算精度；其余模型在GPU7计算，所有prompt必须命中缓存，否则明确报错。初始观测在实际RoboTwin场景采集，不将训练Clean图像伪造为Randomized。单实例两个串行互斥端口21940 Full/21941 Write Off，每端口独立动作buffer/reset；仿真分别GPU4/7，每新任务检查空闲显存，不终止未知作业。

闭环两组均用10步、完整Sidecar局部FFN compile、cache .99/max3、同manifest seed/指令/原动作预算；为低成本统一关闭视频落盘，保留相机渲染和RT。记录双方首观测prompt/state/hash以审核初态差异。18对只供诊断，报告双方成功/共同成功/full-only/off-only，不能外推50任务成功率或纯训练架构收益。

## 当前验证与未完成

新增WriteProbe读路径保留/实际off残差0/全部tensor恢复、异常时beta恢复、物理位置/旋转单位与分位数共3项CPU测试通过。CPU权重加载完成。真实GPU数值门槛、A/B完整统计和闭环尚在执行，后续以本输出的offline_complete.json、write_ratios.json、paired_closed_loop.json和controller_state.json更新结论。


## 2026-10-21T10:02:32.143188+00:00 首批真实A/B（adjust_bottle六个初始观测，非闭环）

为优先出A/B，先采瓶子Clean/Randomized各3个观测；这7个独特prompt（含既有验证fixture）先在GPU7完成CUDA文本编码并卸载T5，WAM常驻GPU7约13.56GB allocated。其余12个样本待采集齐后在GPU4补齐CUDA文本缓存，没有CPU文本推理或重复模型载入。此为对初始资源计划的阶段性优化，两个干预条件始终共用同一缓存。

FFN与各自相同cache的eager参考：两个真实fixture、四项比较max_abs_error均0，4个真实Inductor kernels，门槛通过。cache on/off自身存在差异，仍分开报告，不能由FFN参考匹配推断cache质量无损。

cache off、固定seed42/10步；每域3个观测×10时间步，每tap30个r值：

|域|tap7 mean r|tap15 mean r|tap23 mean r|
|---|---:|---:|---:|
|Clean|2.64978e-6|5.31967e-6|7.48859e-6|
|Randomized|2.84463e-6|5.04381e-6|9.89897e-6|

加法前FP32 beta×projection参考mean r约1.31e-5/2.03e-5/2.64e-5（Clean），1.35e-5/2.00e-5/3.36e-5（Randomized）；与实际加法后相比，符合小残差受BF16舍入进一步削弱的表现，未逐元素唯一拆解乘法与加法舍入来源。

六条32步动作chunk Full vs WriteOff：归一化active action整体相对差异0.2031%–0.3305%；每chunk双臂目标位置平均差异0.5378–0.7582mm，六条合计最大3.7841mm；目标旋转最大0.7496deg。同一噪声输入的单forward active20维velocity相对差异0.1239%–0.2505%，平均0.1760%。Full→Off→Full重复对首样本逐位相同的检查已通过。

初步结论限于这六个瓶子初始观测：直接hidden写回非常弱，但最终动作不完全相同；尚无“帮助/干扰成功率”的结论。不能把几毫米差异直接说无害，也不能据r小就说Sidecar训练整体无用（辅助loss已影响backbone）。其余任务A/B和配对闭环仍进行中。证据pilot_adjust_bottle.json、offline_progress.json、write_ratios_progress.json。


## 2026-10-21T10:11:21.549967+00:00 完整离线A/B结果（18观测；闭环仍进行中）

三个任务各Clean/Randomized各3个真实RoboTwin初始观测；每域9观测。无cache时每观测10个相同Action FM时间步，每tap每域90次forward测量。这90次不是90条episode。原始逐样本、逐时间步、按时间步分组与加法前参考值均保存在write_ratios.json。

下表所有数值单位为 **10⁻⁶**，表示实际BF16写回后hidden差异的FP32范数比：

|域|Block|mean|p50|p90|p99|max|
|---|---:|---:|---:|---:|---:|---:|
|Clean|7|2.694|2.403|3.683|4.677|4.892|
|Clean|15|4.995|4.671|6.403|7.876|8.244|
|Clean|23|7.504|7.377|8.614|10.085|10.159|
|Randomized|7|2.747|2.504|4.019|4.865|5.073|
|Randomized|15|4.901|4.647|6.055|7.213|7.389|
|Randomized|23|9.413|9.283|11.281|12.283|13.265|

无cache mean r约2.69e-6–9.41e-6，最大1.33e-5。说明该checkpoint在这些初始观测上的直接写回很弱，不能据此直接断言Sidecar辅助训练无效，也不能反推整个训练期间的写回幅度。没有人为把gate放大。

|部署|动作相对差异mean|范围|双臂EEF目标位置mean / p99 / max (mm)|旋转差异max (deg)|
|---|---:|---:|---:|---:|
|无cache|0.2381%|0.1948%–0.3922%|0.6142 / 3.5420 / 7.2077|0.7496|
|当前cache .99/max3|0.2883%|0.2051%–0.5620%|0.7586 / 4.3083 / 8.8249|1.1328|

固定Full轨迹中相同noisy action/video latent/context/time的180次单forward重放：active20维velocity相对差异mean 0.1659%，p99 0.2496%，max 0.2509%。这与完整10步去噪后两条轨迹自然分叉的最终action指标分开。

归一化动作差异仅比较active20维，其余unified padding为0；每条32动作、双臂共64个目标位置差异，18条合计1152个目标。表中mm是Full与Write Off预测的物理EEF目标之差，并非相对GT的误差或机器人已实现的EEF跟踪误差。未采集配对GT控制目标，不能提供该项GT误差结论。

当前cache下每域每tap只真正执行36次forward（9观测×4），跳过步未记作零；缓存路径可随干预分叉，不能将其最终差异归因于单个固定forward。无cache结论与同噪声重放是主要因果诊断。全部原始32×20动作在offline_complete.json，精确聚合在offline_summary.json。

GPU4/7各一路配对闭环已开始；截至本节记录尚未完成18对，不能提前给出SR结论。


## 2026-10-21T13:08:37.750130+00:00 配对闭环完成（18对，诊断样本）

|任务|域|Full|Write Off|
|---|---|---:|---:|
|adjust_bottle|demo_clean|3/3|3/3|
|adjust_bottle|demo_randomized|2/3|2/3|
|place_shoe|demo_clean|3/3|3/3|
|place_shoe|demo_randomized|1/3|1/3|
|beat_block_hammer|demo_clean|3/3|3/3|
|beat_block_hammer|demo_randomized|1/3|2/3|

合计Full 13/18，Write Off 14/18；Full-only 0对，Off-only 1对。
首态审核：prompt一致18/18、20维proprio逐位一致18/18、完整原始输入哈希一致18/18。逐对记录final_summary.json。
相同sealed seed/指令、原生动作预算与10步、两组相同cache+FFN；如首态哈希有不一致，需将其计作额外混杂。18对不是全任务成功率或独立训练baseline对比；runtime Write Off无法移除EA辅助loss已在训练中施加到主干的影响。EEF离线mm仍是预测目标差异，不是GT跟踪误差。
已有控制器发送STOP_SERVER，本次诊断模型正常退出；释放状态另存cleanup_status.json。其他正式评估/训练不受停止指令影响。
下一步：依据write幅度弱和成败配对结果决定独立训练对照，不直接放大这个checkpoint的beta，也不把小样本成绩外推为架构提升。


## 2026-10-21T13:15:35.707555+00:00 最终复核与解释

18对/36rollouts均完成并校验，17对成败相同；唯一分歧为beat_block_hammer/demo_randomized/seed100001，Full失败、Write Off成功，两组原生动作上限均400。所有18对首态完整输入哈希、prompt及20维proprio逐位一致；没有将初始化差异计作写回收益。已确认模型2931094、控制器2931513及首批仿真2946092/2946093均不存在，cleanup_status.json确认诊断模型正常退出；没有停止正式评估或训练。

结论：这个42960 checkpoint的直接write非常弱，但不是数值上完全不起作用；同输入动作确会变化。本批18对没有观察到write带来净成功率增益，唯一分歧反而偏向Off；只有一对分歧，不能据此认定write有统计可信的负作用。不能把runtime Write Off等同于重训的无Sidecar baseline，也不能据此排除EA辅助loss训练主干的贡献。没有证明新隔离warmup已经改善效果。

下一步优先对唯一分歧场景做Full/Off重复并互换仿真GPU，确认闭环数值稳定性，再决定扩展固定场景数或独立训练对照；此轮未自动扩展实验或手动放大gate。实际EEF相对GT跟踪误差未测，本轮mm指标严格为两组预测目标之差。
