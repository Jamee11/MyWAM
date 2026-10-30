# PatchWAM ramp100 step24000 全任务快速评测

用户要求继续速测同一训练run的24000权重，与20000判断是否改善。独立目录 `outputs/robotwin_eval/patchwam_ramp100_24000_20261030`；不会覆盖20000已验收100条件/3000条结果（2093成功=69.77%，Clean84.20%、Randomized55.33%）。已验证旧实验complete和final_independent_audit，再仅退休两份自有20k模型；未触碰训练、未知GPU作业或历史评测文件。

指定checkpoint：`outputs/sidecar/aloha50_patchwam_c2r_ramp100_10epochs_20261027T144939Z/F_patchwam_ramp100/runs/2026-10-27_14-57-15/checkpoint_step_24000.safetensors`。
SHA256：`0852df3650561e6f28ba71e4b1642d7db762d3ac2fc80d5ebbe09947ee90d48b`，独立RAM副本复制/复读hash一致，只保留该步权重，避免latest自动变化。config/normalization/Sidecar contract与20k逐字节相同，tokenizer复制保留，无网络下载。

## 相同评测配置

100条件（50任务×Clean/Randomized）各30条、总3000；plan.json沿用20k原次序，未改变任务集合。原官方专家筛seed、seed组0、unseen候选指令池30、RT/near0.1/原生动作预算/成功判据/250Hz保持；不存视频。绝对EEF20→EE16与三视图384×320合同、完整Sidecar taps7/15/23、denoise10/horizon32/eager+cache .99/max3、native chunk及相机gate保持。无训练增强在推理时重复施加。

复用20k已验证调度：两模型各5独立session，模型内部串行、模型之间并行，空闲显存>=35000MiB选择两张L20；本次GPU2/4（PID3646387/3646388）。端口38910–38914、38920–38924，独立token/SSH反向隧道。两个fixture重复及跨副本canonical精确一致、每session重复32动作在线检查通过前禁止启动sim。当前模型加载中，准备/进程启动不等同闭环结果通过。

4090最多10路，实际按余量；>=10240MiB允许任意条件，8704–10240仅Clean，先错开保留7680MiB启动预算；min1536MiB只退出最新自有sim并降cap，原生CUDA/renderer分配失败也降cap，同实验不自动重开失败级别。保持每卡0:1/1:2/2:1/3:1/4:2/5:1/6:2/7:0，不重开已知失败第三路。六项实际调度CPU检查通过，显存峰值和未知作业变化仍未决。

## 结果与核验

state.json/progress.json/scheduler.log记录实际运行；comparison.json与自动[实时结果](ROBOTWIN_PATCHWAM_RAMP100_24000_RESULTS_LIVE_20261030.md)按完整30条件同时对比20k、32k原30及官方同条件，不混partial。吞吐从本次新建只读sampler记录，不导入20k的历史快照。

沿用原生validate_result逐任务验收；终态progress在state complete前写入，避免20k最后停在2999的陈旧快照。report结束时额外运行独立audit_completed.py重验100份原始log/runtime/run.json退出码与SHA，生成final_independent_audit.json。未通过之前不宣称全量完成。

新旧checkpoint训练步数20k/24k不同；同环境seed未统一冻结Python random/确切指令，不能称逐episode输入完全相同。near-plane/实际vs目标EEF/颜色语义增强等原边界保留。后续先看完整相同条件的新20k差值，等100条件完成再判断总体趋势。

容量复核：本轮4090启动前GPU1空闲24189MiB、GPU4为23210MiB、GPU6为19771MiB，均超过双路2×7680+3072=18432MiB静态预算。20k GPU4第二路曾在更低初始余量下触发保护，本轮新增约9GiB以上头寸才恢复2路试验，不照搬失败后的瞬时回收空闲；运行中仍降容保护。低余量0/2/3/5各约8850MiB，仅Clean，沿用20k整轮已通过的同配置边界，但新动作路径/外部负载仍需实际监测。

## 2026-10-30T02:05:20.046760+00:00 实际部署/接入验收

两份L20模型GPU2/PID3646387、GPU4/PID3646388，两个fixture各重复两次/跨副本canonical精确一致通过；10独立session各重复chunk32在线精确一致通过。4090调度PID723820，实际启动10路，progress确认active10/0完成。完整任务/GPU/PID/sourceSHA见activation_snapshot.json；当前处于仿真初始化/专家筛seed阶段，无24k完整rollout/SR。20k全部结果保留，新report实时同任务比较，不声称24k已经更好。

## 2026-10-30T02:40:28.836061+00:00 用户要求检查4090剩余显存

现场10路均在推进。GPU1已有2路，5次连续样本空闲9872–10222MiB，本轮已观测最低7756MiB；GPU4已有2路，当前8669–8693MiB、运行最低7919MiB。额外完整sim保守预算7680+1536=9216MiB，GPU1/4的运行低谷不足该预算，且历史第三路触发过保护，不依单次空闲盲目重开。GPU3/5先前显示约8GiB是新任务初始化，进入动作后均回到3828MiB；GPU0/2/6余量3432/2784/4988MiB左右，GPU7无本实验sim但其它负载占用，仅3730MiB空闲。此次不新增、不重启任何在途进程，保留10路。证据capacity_audit_20261030T0240Z.json。
