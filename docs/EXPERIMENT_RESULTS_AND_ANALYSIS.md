# 实验结果与分析

## 2026-10-14 UTC：前四卡共享资源下的 bs=1 启动调试（进行中）

用户要求在物理 GPU 0–3 验证训练启动；随后明确允许在其他人的作业仍占卡时尝试，且暂不计算训练时间。不停止、不挂起其他作业，不改变模型、数据或资源门槛实现。

### 设置与当前证据

- 8×A100-SXM4-80GB；选择物理卡 0,1,2,3。启动前四卡各使用 21155MiB、空闲 59998MiB，计算利用率约 98–100%。容器内 `nvidia-smi --query-compute-apps` 无结果，不代表 GPU 空闲。
- 当前 CPU 数据预检实际通过：50 tasks、2500 episodes、557651 source frames、Foundation 80D heads。证据：`outputs/ea_lite/debug_20261014_bs1/preflight.json`。
- A 配置实际解析成功，batch1/rank、world4、accum8（有效 batch32）、BF16、ZeRO2、fused AdamW、checkpointing、CPU 初始化、完整 Foundation 暖启动、max160 micro-steps、每40 micro-steps保存、保存5个更新后计划退出。证据：同目录 `config_A.yaml`。
- 原 launcher 实测 exit1，原因是 profile80 要求每卡空闲至少70000MiB；日志 `launch_A.log`。这次失败发生在资源门槛，未分配模型，不能解释为 bs1 训练 OOM。
- 用户明确要求在共享卡继续尝试后，直接调用同一 `torchrun → debug_ea_lite.py → official trainer` 入口。保留资源检查文件原样，不伪称独占门槛通过。A 为官方原架构基线，复用 Lite reader，auxiliary=false；它的通过不能替代 C Lite joint 验证。
- 共享试跑日志：`outputs/ea_lite/debug_20261014_bs1/shared_A.log`；输出前缀 `outputs/ea_lite/shared_bs1_A_joint`。当前尚无完整优化器更新成功的结论。

### 分析与下一步

共享作业同时减少显存并占用计算，可能影响启动容量、耗时与运行稳定性。若失败，需定位模型加载、前反向或 AdamW 更新阶段，并保留全 rank 终态；不将共享状态下 OOM 外推为独占四卡不能训练。先观察连续更新和保存，再按证据测试 Lite C joint；warmup 冻结主干通过不能代替 joint。训练时间估算按用户最新要求暂缓。

### A 实测完成与显存解释

2026-10-14 UTC：进程 exit0，全部 rank 完成40 micro-steps / 5 optimizer updates，最后 loss_total=0.1550704688，未发生OOM。完整权重 `checkpoint_step_40.safetensors` 已保存，full-state 的 `trainer_state.json` 确认为global_step40/opt_step5/epoch0，模型、四rank optimizer、scheduler及四rank RNG文件存在；未见独立sampler.bin（不能用保存日志替代文件验收）。尚未实际恢复，不宣称恢复通过。

实际run：`outputs/ea_lite/shared_bs1_A_joint/2026-10-14_12-49-52`。

| rank | peak allocated GiB | peak reserved GiB |
|---|---:|---:|
| 0 | 49.797 | 52.752 |
| 1 | 49.801 | 51.814 |
| 2 | 49.885 | 51.932 |
| 3 | 49.793 | 53.670 |

参数日志：total12406.8M，trainable6021.2M。按BF16全权重、四卡ZeRO2的FP32master/Adam两状态/局部分片梯度估算，分别约23.1/16.8/5.6GiB；与49.9GiB峰值之间余约4.4GiB。这只是理论拆账与差额，未对各项做运行时独立计量，不能把差额全部当作激活或按bs线性放大。权重/master/Adam/最终参数梯度大小主要由参数和world决定；bs增大主要增加中间激活、输入编码特征及部分算子workspace。通信缓冲通常仍主要由参数及bucket配置决定。

bs1单点不能可靠确定空闲四卡最大bs，尚无bs2增量实测；模型峰值也不是nvidia-smi总占用，后者包含其他作业、分配器缓存和CUDA/NCCL开销。前四卡外部占用约20.7GiB/卡，完全空闲只意味着释放这部分空间，不能据此保证bs4/8。

### C joint 验证开始

随后启动完整Lite C joint，保留batch1/rank、world4、accum8、BF16/ZeRO2/fused AdamW/真实全50-task reader/Foundation80D。只覆盖40 micro / 5更新，save_steps0避免再次写入大状态，日志 `outputs/ea_lite/debug_20261014_bs1/shared_C.log`，输出前缀 `outputs/ea_lite/shared_bs1_C_joint`。此轮不验收C保存/恢复/部署；须全rank连续更新与终态证明启动可行。

### C joint bs1 完成；bs2 增量测试

C bs1 进程exit0，四rank均40micro/5updates，最后loss_total=0.4253222644；真实非零EA masks、有限辅助loss、adapter输出非零；最终所有被诊断的EA projection/ground/relation/plan/attention/out与video patch/action encoder梯度有限且非零。全模型C joint训练启动通过，不是冻结主干warmup。

run：`outputs/ea_lite/shared_bs1_C_joint/2026-10-14_13-06-32`。

| rank | peak allocated GiB | peak reserved GiB |
|---|---:|---:|
| 0 | 49.868 | 51.896 |
| 1 | 49.875 | 51.896 |
| 2 | 49.957 | 51.975 |
| 3 | 49.863 | 52.814 |

此轮save_steps0，只有配置/tokenizer/stats与诊断，无更新后的C权重；不计保存恢复或部署通过。停止后GPU0–3均回到原外部占用21155MiB。

用户追问bs2显存与空闲四卡容量，追加C joint bs2短测：其余参数保持bs1配置，accum8/global64、40micro/5updates、save_steps0，真实全模型训练；不是与bs1同global batch的效果对照。日志 `outputs/ea_lite/debug_20261014_bs1/shared_C_bs2.log`，输出前缀 `outputs/ea_lite/shared_bs2_C_joint`。单点bs1和理论差额不能证明最大bs，须等待bs2实际峰值与终态。

### 用户追问：方案对应关系与 loss 趋势

配置与实现核对：C实际variant为joint_self_attn_ea_rsig_lite、stage joint、action_conditioning true、adapter_layer0、512D/15tokens，80D heads及32action/9video；当前clean VAE patchify，不额外调用观测DiT，不用video residual；Ground .025/Relation .01/Plan .05分别进入总loss。对应首版Lite核心方案，直接joint只为容量验收，正式warmup尚未执行。生成拒绝privileged targets，训练targets独立仅用于loss。

40条rank0诊断确认：`loss_total=loss_video+loss_action+loss_ea`最大误差8.94e-8，加权三辅助项与loss_ea最大误差3.58e-8。最后A video/action/total=0.1445425/0.0105280/0.1550705；C video/action/weightedEA/total=0.1434777/0.0098157/0.2720289/0.4253223。总loss更大主要是目标新增辅助监督，不代表原FM误差变大。

按每8micro一组、每组平均（对应一次优化器更新的数据）：

| 更新组 | C video | C action | C weightedEA | C Ground raw | C Relation raw | C Plan raw |
|---|---:|---:|---:|---:|---:|---:|
| 1 | .21256 | .03549 | .28815 | 7.76149 | 1.09556 | 1.66310 |
| 2 | .21748 | .04415 | .27402 | 7.54936 | 1.15964 | 1.47379 |
| 3 | .22948 | .04410 | .26959 | 7.72369 | 1.10710 | 1.30863 |
| 4 | .19564 | .03345 | .26827 | 7.71454 | .94970 | 1.31820 |
| 5 | .22247 | .02853 | .27640 | 7.71360 | .93807 | 1.48361 |

数值有限、masks非零、条件/主干梯度连通，未见数值发散；EA辅助项/Relation/Plan有短期下降迹象，Ground基本持平，均非单调。只有5次更新且不同随机窗口/noise，不能认定正式收敛。A max160micro/LR horizon20updates，C max40micro/horizon5updates，学习率曲线不同（且此次不跑warmup），不能将C主loss略低当作同预算效果提升。下一步正式趋势需固定同LR horizon/数据曝光与seed、更多更新以及grounding/plan质量和闭环SR；当前不擅自延长为长期训练或调整loss权重。

### C joint bs2 完成与增量结论

bs2进程exit0，四rank均完成40micro/5updates，全部loss指标有限，最后loss_total0.4822226167，相关video/action/EA被诊断梯度有限且非零。真实bs2进入整批input_latents/actions/proprio/EA context并参与训练（prepare_inputs整批拼接，Lite observe核验context与latent的batch一致，无拆成bs1的额外wrapper）。run：`outputs/ea_lite/shared_bs2_C_joint/2026-10-14_13-19-15`。

| rank | peak allocated GiB | peak reserved GiB |
|---|---:|---:|
| 0 | 49.868 | 52.098 |
| 1 | 49.875 | 52.098 |
| 2 | 49.957 | 52.176 |
| 3 | 49.864 | 52.820 |

以rank0为例，首个更新后下一累积周期的普通前反向运行峰值（micro10–15）bs1=47.6861GiB、bs2=47.7901GiB，增约0.104GiB；更新边界micro16的全程峰值两者都约49.868GiB。因此bs2主要增加的激活/输入及workspace未超过原本optimizer更新临时峰值，总峰值基本不变；不能据此断言没有任何batch增量，更不能将此小区间斜率外推为最大bs。

共享卡实测bs1和bs2均通过五次全参Lite joint更新；完全空闲四卡仍需继续按4/8等档位实测才可确定最大值。此次没有测bs4/8、独占吞吐、长期训练、C checkpoint部署或ZeRO实际恢复。三run四rank摘要位于 `outputs/ea_lite/debug_20261014_bs1/runtime_verification.json`。结束时无本次debug训练进程，GPU0–3均恢复原外部占用21155MiB；未操作其他作业。官方训练时间估算按用户最新要求暂缓。

## 2026-10-14 UTC：快速启动可靠性与更新速度对照

用户要求复核主方案新增第23节，并允许复用既有数据、日常快检、迁移后全检。现有代码已经提供训练入口默认EA_DATA_CHECK=fast、ready清单和startup_mode=prepared；复用这些开关，未新增重叠校验器、未修改模型/数据核心逻辑。

### CPU与真实资产验证

- 对当前ready清单执行fast→full→fast：0.2637s / 44.9688s / 0.1152s，均通过；fast明确不证明全部未访问文件完整。报告 `outputs/ea_lite/startup_comparison_20261014/preflight_times.json`。
- ready去除startup_metadata后与原manifest逐字段相等；本轮额外核对50任务的2500份缓存指令和全部scene_info与现有源文件一致（约5.50s），不将此重新扫描加入日常启动。报告 `cache_semantics_audit.json`。
- startup/preflight CPU检查13 passed，194.34s包含冷依赖导入；覆盖tiny真实HDF5 prepared/scan首中尾图像/动作/指令/mask逐字段一致、启动不glob/打开HDF5、缓存缺失拒绝、source size/mtime和sidecar SHA按需拒绝。日志 `cpu_tests.log`。
- 原先C bs1/bs2其实已经使用prepared；A最初为旧扫描路径。不能用A/C不同模型直接比较数据初始化策略。为此另跑同配置C scan完整对照。

### 四卡同配置C对照

共同设置：Foundation、C joint、bs1/rank、world4、accum8/global32、seed42、相同损失系数、max40micro/LR horizon5updates、save0、BF16/ZeRO2/fused AdamW/checkpointing；prepared复用前一run，scan新run `outputs/ea_lite/startup_scan_C_joint/2026-10-14_13-38-15`。全量预检此前完成，scan显式恢复逐轨迹reader扫描。两者均40micro/5updates、exit0。排除首次8micro；对后32micro每步取最慢rank时间，再统计完整8micro周期。

| 模式 | 平均micro秒 | 平均完整更新秒 | samples/s | peak allocated GiB | peak reserved GiB |
|---|---:|---:|---:|---:|---:|
| prepared | 2.0257 | 16.2058 | 1.9746 | 49.957 | 52.814 |
| scan | 1.9654 | 15.7231 | 2.0352 | 49.957 | 52.814 |

本次scan较prepared快约2.98%；均为共享GPU且每种模式仅一次短测，不能归因为启动模式、更不能称统计等价。机制上计算路径不改，实测未见明显持续更新速度变化。prepared减少启动全量扫描与重复审计，实际RGB解码、VAE/T5编码与主干forward仍在训练中执行。

全部rank/40步有效标签比例一致，首组8micro所有loss一致；后续loss_total最大差0.00028816、video0.00024045、action0.00034820；梯度范数最大差0.1043124，不能宣称全程bitwise parity。CPU样本parity与缓存源内容核验支持数据合同保持，后续BF16分布式更新差异的根因未单独隔离。报告 `training_comparison.json`；scan日志 `scan_C_training.log`，进程终态 `scan_process_result.json`。

额外观察日志中mixed_precision→first finetune区间：prepared约50.86s、scan229.20s，包含分布式/模块初始化和dataset构造，不能当纯reader计时。scan端到端788.08s，不能与无对应启动时钟记录的prepared直接作精确端到端差值。共享盘依赖冷导入和完整权重加载仍是启动耗时，不在fast preflight的0.26s中。

首次尝试使用本机不存在的/usr/bin/time，exit127，未启动训练；失败保留scan_C.log。随后stdlib subprocess计时器启动实际scan_C_training.log，不因观测等待重复启动。

### 日常使用与EA 0.5建议

日常使用ready+prepared，启动器默认fast；迁移/换数据版本后离线执行full一次，正常训练仍可用prepared，无需再扫描HDF5。指令/scene改动需离线更新ready。诊断旧路径才显式startup_mode=scan；不关闭样本合同、统计hash和按需源检查。

当前C bs1 mean(video+action)=0.2526705、mean(weightedEA)=0.2752869，EA占总loss约52.14%。0.5倍辅助项按当前输出算会降到约35.26%，这是数值重算，不是新训练效果证据。Ground在3×14×14分类位置上的均匀交叉熵加两份BCE约ln(588)+2ln2=7.76，与初始raw Ground实测一致，不能以raw7.7判定损失异常。联合梯度日志未分离EA/FM对共享参数的贡献，不能证明EA梯度支配。

建议joint阶段把0.5作为可选试验，保持warmup与现有基线默认不变。复用现有配置将ground/relation/plan分别设0.0125/0.005/0.025，与原加权EA整体乘0.5严格等效。已实际CHECK_ONLY resolve通过，产物 `config_C_half_aux.yaml`；未运行0.5权重训练，不宣称其收敛/闭环更好。本次速度对照两组都使用原系数，不混入损失策略变化。

最后无本次训练残留进程，GPU0–3恢复原外部占用21155MiB。下一步：正式joint选择辅助系数后固定A/B/C相同LR horizon/曝光预算做更多更新与闭环验证；官方全量微调排期按用户要求仍暂缓。

### 缓存生命周期与预算配置核对补充

代码核对：启动fast preflight只读清单/统计/权重记录与header，不将全量训练数据载入缓存。prepared元数据用于索引/指令/scene构造。每worker的source size/mtime首次检查结果用无界LRU缓存；EA sidecar数据仅缓存8条轨迹，淘汰后再次加载会重新计算SHA。RGB/动作仍按batch从HDF5读取并解码，输入形状合同仍逐batch检查。当前DataLoader未开启persistent_workers（PyTorch默认false），新epoch重新创建worker，原worker缓存随之失效；恢复/重新启动进程同样重建缓存。因此不能声称“全部快检一次后，后续10000步所有数据都常驻且不再发生任何校验/I/O”。相关代码：trainer build_dataloader、Lite reader _check_source_on_first_read/_labels及official reader _build_sample。

在耗时估算仍暂缓的前提下，额外仅CHECK_ONLY核对论文Clean2Random global256/10740更新的计数：world4×bs1×accum64=256，max_steps687360micro对应10740更新。A/C配置dump均成功，保存在 `outputs/ea_lite/official_budget_readiness_20261014/`；save_steps32000仅为对齐500更新边界的示例，不是官方保存协议。配置保留当前数据/增强/LR合同，仅匹配batch与更新数量，不等于复现论文完整protocol；accum64尚未GPU实测，未启动任何长期训练或计算总训练时间。

### 官方在线解码路径与离线RGB缓存体积

2026-10-14 UTC，用户询问官方是否预解码以及缓存规模。在线读取固定官方commit4d7f642b841913ea3c176010b692b1de3d5ef180的robotwin.py及wan_backbone.py：RoboTwin样本从HDF5读取JPEG并在线解码，trainer每batch调用prepare_inputs，Wan训练预处理在线执行text encoder和VAE；这条基线路径没有全数据RGB/latent预缓存。DataLoader worker预取和OS文件缓存不等于应用层全数据解码缓存，也不意味着GPU每一步必须串行等待所有读取。部署prompt cache不能当作训练T5缓存证据。

按当前manifest557651个源时间帧与555151个训练窗口，只算无压缩RGB载荷、忽略容器/索引/压缩效果，十进制GB/TB：
- 逐唯一帧缓存384×320拼接canvas、uint8：205.572GB；FP32：822.290GB。
- 若三路原图均240×320，逐原图缓存uint8：385.448GB；FP32：1541.793GB（原图尺寸假设需按各view确认，非已生成缓存的实测体积）。
- 每训练窗口单独缓存9张384×320 uint8 canvas：1841.858GB，约1.84TB；重叠窗口重复存储是额外膨胀来源。
- 若三路每帧各存一个文件，约1672953个文件；按轨迹/分片打包可减少文件数，不改变总无压缩载荷。

解码RGB缓存消除JPEG解码但仍需VAE/T5在线编码；冻结encoder的latent/text缓存是另一种缓存，通常更紧凑，但需核对encoder版本、窗口因果边界/首帧、增强及prompt合同，不能把每帧独立latent直接拼成时序VAE输出。当前未构建RGB或latent缓存，也没有这类缓存的实测速度/存储结论，未修改核心reader。

官方源码来源：
[固定commit RoboTwin reader](https://github.com/OpenWAM-Official/OpenWAM/blob/4d7f642b841913ea3c176010b692b1de3d5ef180/openwam/dataloader/robotwin.py)、
[固定commit Wan训练预处理](https://github.com/OpenWAM-Official/OpenWAM/blob/4d7f642b841913ea3c176010b692b1de3d5ef180/openwam/model/video_backbone/wan_backbone.py)。


## 2026-10-15 UTC：空闲GPU0–3从bs4开始的容量/速度测量（进行中）

用户明确要求从bs4向上测，重复独立数据预检跳过。起始GPU0–3均4MiB占用、81150MiB空闲。现有完整Lite C joint/Foundation80D/真实50task prepared reader/BF16/ZeRO2/gradient checkpointing/fused AdamW、原loss系数；bs4/rank、accum1/global16、20micro=20次更新、save0，无长期训练。日志outputs/ea_lite/capacity_20261015/bs4.log。独立preflight不运行，reader原样保留stats合同和实际样本按需检查；不重建数据、不改模型或EA权重。

后续准备复用已有benchmark扫描，以可选参数从bs8起、每档连续多次真实更新，并用原DataLoader/Accelerate分片和worker测数据读取开销，避免每档重新加载。新路径仅语法检查，须待实际终态；第2步/optimizer状态、最慢rank稳态速度、allocated/reserved与整卡显存均需记录。


## 2026-10-15T02:54:48.732039+00:00：用户授权正式Clean50 EA-RSIG-Lite五epoch训练（已提交）

用户要求立即将GPU0–3用于bs64、acc1/global256、Clean全数据5epochs；checkpoint按用户后续指定每2000更新。旧不启动长期训练约束对此run已被明确授权覆盖。沿用已验证C joint/Foundation80D/32action9video、原EA系数.025/.01/.05（未使用0.5档）、video/action LR1e-5/base1e-4、BF16/ZeRO2/fused AdamW/checkpointing/seed42、color jitter关闭。num_epochs5/max_steps null，实际按reader epoch终止，不用论文10740更新覆盖本数据集的5epoch。

持久后台PID823163，位置outputs/ea_lite/clean50_C_joint_bs64_5epochs_20261015：train.log、metrics/rank*.jsonl、runs下实际带时间戳产物。使用原trainer与标准DataLoader，只保留逐步测量，不启用batch sweep或额外梯度hooks。独立数据预检跳过，样本合同和源/标签按需检查保留。launch_record.json记录启动参数；首轮连续更新及真实run路径仍待终态证据。显存余量极小，若OOM须报告具体阶段，不擅自降低batch；每2k权重/full-state和最后最终权重按原trainer保存，keep_last_k2/keep_final_resume_state true。


### 2026-10-15T02:56:46.269427+00:00：空闲四卡容量测量已完成

bs4独立20次更新、随后同一模型连续扫描8/16/32/64各8次更新，全部四rank真实DataLoader batch长度与optimizer step断言通过、loss/指标有限、exit0，无OOM。未启用有界参数组、未修改模型/EA权重，accum1。bs4排除前三步、其余每档排除前两步，对每一步取四rank较慢时间；后续每档仅6个稳态样本，属于短测，非长期稳定结论。

| bs/rank | 更新数 | 秒/更新 | 四卡samples/s | allocated GiB | reserved GiB | 整卡1Hz采样峰值GiB |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 20 | 1.525 | 10.489 | 49.957 | 56.674 | 58.304 |
| 8 | 8 | 2.309 | 13.857 | 49.957 | 56.637 | 58.267 |
| 16 | 8 | 3.786 | 16.904 | 49.957 | 56.781 | 58.411 |
| 32 | 8 | 6.937 | 18.451 | 56.313 | 59.828 | 61.458 |
| 64 | 8 | 14.542 | 17.604 | 72.564 | 77.617 | 79.247 |

整卡采样按逐步耗时和JSONL完成时间近似对齐phase，非连续精确峰值。bs64最少仅free4MiB，虽通过8更新但显存几乎用满；吞吐17.60低于bs32的18.45，工程建议bs32作为余量更充分的档位，不能宣称64为绝对最大bs。bs4入口额外挂梯度诊断，sweep入口不挂，测速包含实际数据读取/预取与训练。sweep外层无sampler导致seed wiring warning；实际内层使用trainer.build_dataloader(seed42 generator)与Accelerate分片，未独立审计各档完整样本顺序，不用于收敛效果对比。配置、四rank逐步指标、采样与摘要位于outputs/ea_lite/capacity_20261015，结果summary.json；结束时均释放到4MiB，监控进程已退出。

### 2026-10-15T02:56:46.269427+00:00：五epoch正式训练持久启动修正

初次nohup PID823163在工具返回后消失，train.log空、无worker/更新，不计成功启动。随后新增本次输出目录内launch_tmux.sh，用独立tmux会话ea_clean50_bs64_20261015持久运行相同正式配置，PID825005；在第二次独立终端调用确认pane dead=0、torchrun存活，尚处初始化。launch_record.json保留首次失败原因和新PID/会话。用户再次确认checkpoint每2000步，正式命令已设置2000而非先前建议500；5epoch/bs64/acc1/global256不变。


### 2026-10-15T03:06:29.797018+00:00：正式训练已完成真实更新

四rank均已完成至少10次AdamW更新，tmux ea_clean50_bs64_20261015 / PID825005存活，四卡利用率100%，无OOM。核对实际config：bs64/rank、acc1/global256、5epoch、max_steps null、save_steps2000，调度总10845更新。第10步loss总0.565409（video0.228331/action0.047280/加权EA0.289799），约13.46秒/更新；单卡整卡占用79599–79999MiB，空闲1154–1554MiB。原EA权重.025/.01/.05，未使用0.5档。产物目录outputs/ea_lite/clean50_C_joint_bs64_5epochs_20261015/runs/2026-10-15_03-02-52，日志同根train.log，四rank逐步证据metrics/rank*.jsonl。当前仅确认启动和早期连续更新；尚未达到首个2000步保存点，5epoch训练、长期稳定性和效果评估未完成。下一步保持训练，观察loss/显存并在保存点核验checkpoint。


### 2026-10-15T03:15:38.907570+00:00：论文SFT协议与当前run差异核实（未更改训练）

用户要求核实官方协议。已在线读取官方main的assets/openwam_usage_docs/openwam-alpha-finetuning.md、configs/train.yaml与configs/dataloader/robotwin.yaml；本地既有论文提取docs/OpenWAM_paper_extracted.txt第2372–2387行（附录B.2、表10，PDF第34页）明确：SFT除batch/训练量/是否增强外沿用表9；Clean2Random为global256、5epochs、ColorJitter(0.2,0.2,0.2,0)。官方train默认基础LR1e-4且action/video_lr为null；RoboTwin YAML增强默认false，与论文Clean2Random协议存在差异。当前正式run骨干1e-5与关闭增强来自本地Lite文档16.2/16.3及train_ea_lite_4gpu.yaml，并非论文规定。当前run不可标记为除EA外完全对齐论文。Foundation路径及train.log第454–461行确认实际加载checkpoint_step_154000.safetensors成功，未使用下游C2R SFT权重。

建议保留旧run证据，从同一Foundation新开run，将action_lr/video_lr设1e-4并启用color_jitter.enabled=true，其余bs64/acc1/global256/5epochs/save2000及EA权重保持；本次仅核实和解析配置，不停止、重启或更改运行中训练。用户尚未确认此训练策略调整；按AGENTS第2条须先确认。后续A/B/C同样必须使用一致增强和骨干LR。新配置长期稳定、C保存部署和闭环效果仍待验证。来源：https://github.com/OpenWAM-Official/OpenWAM/blob/main/configs/train.yaml ，https://github.com/OpenWAM-Official/OpenWAM/blob/main/configs/dataloader/robotwin.yaml ，https://arxiv.org/pdf/2609.07398#page=34 。


### 2026-10-15T03:20:40.433424+00:00：用户授权纠正正式SFT学习率并开启ColorJitter

用户先授权将本地自行降低的骨干学习率纠正回官方1e-4，随后明确要求开启ColorJitter。修改configs/train_ea_lite_4gpu.yaml action_lr/video_lr为1e-4；保留三个等学习率优化器组，避免历史单组ZeRO2临时显存问题。configs/dataloader/ea_rsig_lite.yaml enabled改true，幅度0.2/0.2/0.2/0，复用官方reader clip一致增强实现，无核心逻辑改动。旧配置备份在新输出目录*.before.yaml，旧run日志/指标保留。新run为outputs/ea_lite/clean50_C_joint_bs64_5epochs_lr1e4_jitter_20261015，从同一Foundation重新初始化，bs64/acc1/global256/5epochs/save2000/原EA权重保持，prepared启动且跳过独立重复预检。已解析配置，真实更新与新增强下容量/速度待验证；旧run未到2000步无权重checkpoint，不能把日志保留称为权重保存。增加增强不改变窗口数或10845调度更新，可能增加CPU开销。下一步结束旧run、tmux持久启动并确认四rank真实更新。


### 2026-10-15T03:30:21.162129+00:00：LR1e-4 + ColorJitter新run已实际更新

旧run于77步停止，无checkpoint（未到2000步）；日志/配置/逐步指标保留。用户“先新加载再停旧”的消息在停止后到达，已如实告知；本次未做到重叠衔接，后续应先评估双进程内存并将CPU准备阶段提前。新tmux ea_clean50_bs64_lr1e4_jitter_20261015/PID845189、workers845451–845454均存活。Foundation读取成功、CPU构建与共享文件系统加载约数分钟后，四rank均完成至少4次真实AdamW更新，指标全部有限，无OOM。实际run配置确认action/video/base峰值学习率均1e-4、ColorJitter enabled true；当前LR仍处5% warmup，不应将日志瞬时低LR误认为覆盖失败。最新确认loss 0.551812，更新耗时13.643s，allocated72.564GiB，整卡79599–79999MiB/free1154–1554MiB/util99–100%。与先前约13s/步接近，仅早期样本，不能据此声称增强无开销或收敛改善。产物outputs/ea_lite/clean50_C_joint_bs64_5epochs_lr1e4_jitter_20261015/runs/2026-10-15_03-27-34，原5epoch/10845步/bs64/acc1/global256/save2000及EA系数不变；训练继续，首次checkpoint/长期稳定/闭环未验证。


### 2026-10-15T03:54:06.257243+00:00：用户授权Baseline跨集群训练包与Slurm流程

目标/home/mingfeihangroup/xiaoming，incoming上传、projects/OpenWAM解压、ckpts存Foundation；分区mhan0035、单节点4×A100，不设置time。新增configs/train_alpha_baseline_clean50.yaml与scripts/cluster下baseline launcher、sbatch、一次性路径准备、复用旧installer的环境入口、独立打包器，无核心模型/数据逻辑改动。Baseline为官方joint_self_attn/80D，数据沿用现有wxyz适配+prepared reader、auxiliary=false，因此不添加EA模块/损失/输入；bs64/acc1/global256/5epochs/LR1e-4/ColorJitter开启/save2000、fused AdamW/BF16/ZeRO2/等LR三组与当前配对预算。源数据/当前训练不改写。

用户后续明确要求保留EA标签，数据包包含2500个Clean HDF5逻辑35,943,837,132 bytes、全部EA labels/stats/ready manifests及指令/scene信息，不包含MP4/PKL回放。代码包排除venv/git/log/历史权重，Foundation单独包。打包器dereference并仅写regular-file，PAX精确纳秒mtime，流式每文件和归档SHA256；已用软链接输入完成GNU tar解压roundtrip，实字节/无link成员/纳秒时间/hash通过。Hydra参数解析与Shell语法检查执行中/结果见交付记录。目标gnode06目前用户报告drng，可能排队，目标CUDA/NCCL/实际训练未验证；bs64按80GB档，40GB需另配。完整说明docs/BASELINE_CLUSTER_TRANSFER_20261015.md。


### 2026-10-15T04:06:05.510662+00:00：用户将集群Baseline数据改为官方原始Clean，EA训练暂不切换

用户指定/remote-home/jinminghao/DiT4DiT/datasets/robotwin2_official_raw_20261001。只读清点其raw/aloha-agilex内50个Clean包、2500HDF5、35,044,845,222 bytes、无直接软链接；不使用Randomized/Piper/archive zip。抽查官方episode0根仅endpose/joint_action/observation/pointcloud，无ea_rsig_oracle_v3，帧140与旧自采同编号146不同，旧EA标签不得直接嫁接官方RGB。已告知并获用户明确选择“先准备官方baseline，现有EA暂时保持运行”，因此此前“重新启动本机EA”的要求被此澄清暂缓，PID845189不停止。

新增只写新outputs目录的prepare_official_clean50.py，复用现有wxyz→xyzw EEF转换、官方stats accumulator/Rot6D identity，生成独立统计与prepared manifest，无EA伪标签。官方源robot.py _trans_endpose使用transforms3d mat2quat（wxyz），保留0.12m EE/TCP偏移合同。交付数据改为官方HDF5+元数据；按用户保留标签要求，旧全部EA labels/stats/原manifest单独放data/legacy_ea_lite、明确不对应官方轨迹，不自动relocate为官方EA。打包器新增显式manifest与复用已校验Foundation归档选项；旧自采包不作为最终交付。

Slurm仍mhan0035、4×A100、无time，内存改mem0覆盖四进程CPU构建峰值；本站不允许时用sbatch mem参数覆盖。新数据统计/实际baseline样本/最终归档尚待完成验证，目标GPU训练未做。文档BASELINE_CLUSTER_TRANSFER_20261015.md已改为官方源，旧版另存generated_data_superseded.md。


### 2026-10-15T04:12:31.327782+00:00：官方Clean50离线元数据验收完成

prepare_official_clean50.py完整处理50任务/2500轨迹，三相机长度、EEF有限/单位wxyz四元数与夹爪[0,1]检查通过；每任务50份指令，独立stats已拟合且Rot6D identity。新manifest位于outputs/baseline_official_clean50/data_20261015/manifest.ready.json：552287源帧、549787窗口、35044845222源HDF5 bytes，global256每epoch2148更新/5epoch10740（不再使用旧10845）。官方原始数据没有EA监督，auxiliary_labels_available=false；用户选择本机旧EA继续。新交付目录将为/remote-home/jinminghao/transfer_packages/OpenWAM_official_baseline_clean50_20261015。Foundation压缩完成19909835361 bytes，原始文件逐项hash与download_verification核对通过；新包复用此普通归档文件而非重新压缩。官方baseline真实样本/路径迁移/最终归档完整校验继续执行，目标Slurm/GPU仍未运行。


### 2026-10-15T06:47:53.858710+00:00：wrist020本机新入口准备完成；用户手动切换受GPU访问权限阻塞

用户明确新路径wrist020正确，旧clean50_v6_raw腕部near0.1裁剪异常；选取新根raw/aloha/EA_RSIG_v3/Clean，按原prepare_ea_lite_data.py完整再生成2500份8/16/32标签和统计，557651源帧/555151窗口。全部相机/动作时间线、有限/旋转、EE→TCP与oracle几何/130D目标检查及full preflight2500源size/mtime+sidecar/stats hash通过。统计重新拟合后hash与旧相同，符合重建物理轨迹不变；bbox/ground可见标签重新生成，绝不借用旧bbox。三任务old/new图像人工查看，近裁剪原隐藏物体在新图中可见；正常贴近/机械臂遮挡仍存在，不称全数据全视角都无遮挡。

新增configs/train_ea_lite_wrist020_clean50.yaml、scripts/train_ea_lite_wrist020_clean50.sh、scripts/stop_training_run.py、scripts/check_cuda_access.py及LOCAL_EA_WRIST020_CLEAN50_20261015.md。纯配置resolve、shell语法、真实fast preflight+fake torchrun命令构造、只读旧PID核对均通过；未运行真实torchrun新训练，未发停止信号。CPU150样本覆盖50任务首/中/末：32x80绝对EEF、1x80 proprio、9视频、130D EA、2x10 effectors+2presence、图像增强和首帧条件一致/有限值通过。一开始临时校验将effectors误写2x11产生断言，已按真实2x10+独立presence合同修正并完整通过，不是数据/模型错误。新入口保持C joint/bs64/acc1/global256/5epochs10845/LR1e-4/ColorJitter/原EA权重/save2000，从Foundation新开；原run保持。

NVML当前Unknown Error：/dev/nvidiactl/nvidia0/nvidia-uvm对新进程open返回EPERM；ctypes cuInit=100/CUDA_ERROR_NO_DEVICE；新PyTorch进程device_count0/No CUDA GPUs available。旧run PID845189、四worker仍推进到893，因此当前容器新GPU进程访问被拒，不能先结束旧run或绕过GPU检查强行切换。已只读检查设备/proc/权限，dmesg被拒，未修改设备/cgroup、未reset GPU、未重启容器/驱动；触发原因需宿主机管理员核对。官方NVIDIA容器故障文档说明这类报错可能由容器失去GPU访问造成，但不据此断言具体触发事件。新脚本/data准备已完成，GPU容量/首更新须权限恢复后由用户启动验证；GPU访问故障与数据质量/打包独立。

集群official baseline三包已全部生成并逐成员hash+归档hash复核通过：代码原309成员，官方HDF52500+指令/scene+legacyEA共7553成员，Foundation10成员，全部regular-file、PAX纳秒mtime无link。只刷新小代码包以携带新本机脚本/说明；已验证的官方数据/权重包保持字节不变。目标路径/home/mingfeihangroup/xiaoming/projects/OpenWAM，Foundation到ckpts，Slurm mhan0035/4A100/no time；目标集群GPU运行仍未做。


### 2026-10-15T07:16:17.437849+00:00：当前训练RGB通道复核，旧腕部临时预览颜色说明纠正

用户询问旧导出视频的RGB/BGR问题是否影响当前训练。只读读取实际rank0进程845451的EA_LITE_MANIFEST单项，确认当前reader绑定outputs/ea_lite/data/manifest.ready.json及旧clean50_v6_raw。当前RoboTwinDataset._decode_jpeg为cv2.imdecode→Image.fromarray，不额外BGR2RGB；此旧RoboTwin JPEG将RGB直接交给cv2.imencode，名义BGR解码得到原RGB，PIL直接读取反而R/B交换。模型当前读HDF5，不读取此前导出的LeRobot视频；ColorJitter及Wan preprocess均不交换通道。

按实际方法源码提取执行并与独立官方decode_image_bit参考逐像素比对：当前旧数据50task/150episode/1350三视角JPEG，new wrist020同样50/150/1350，标记standard JPEG均0、reference mismatch均0。当前旧数据另对照450帧raw preview MP4 RGB，全像素平均MAE1.8446（H264/JPEG压缩差异），交换R/B后6.2153；349张颜色明显帧在彩色像素上MAE5.3273 vs交换63.3414，无一张交换后改善超过2。blocks_ranking_rgb训练解码/原始视频均左红中蓝右绿，PIL直读变左蓝中红右绿。未发现训练RGB通道错误；未扫描每一帧，不能称所有数据绝无其他问题。已通过的动作/时间线/EA几何工程检查也不证明闭环效果或全部视觉质量。

纠正此前临时腕部对照图：/tmp/inspect_openwam_wrist_views.py曾PIL直解legacy JPEG，颜色不代表训练实际输入；几何裁剪比较不受该通道交换影响。已保留原图并在outputs/ea_lite/rgb_audit_20261015生成*_training_rgb_comparison.png和rgb_order_comparison.png，按真正训练decoder重新导出。完整抽查记录audit.json。未修改训练reader、数据、模型、配置或运行进程。未来混入带XPL-RGB1标记的新版标准JPEG时，当前legacy-only decoder不可不加核对直接沿用；本次两批抽样均legacy，无需改核心解码逻辑。


### 2026-10-15T07:57:17.440916+00:00：运行中与wrist020两套sidecar路径核实

只读核对运行中rank0进程845451的EA_LITE_MANIFEST，仍为outputs/ea_lite/data/manifest.ready.json，对应旧clean50_v6_raw及outputs/ea_lite/data/labels。新wrist020准备清单在outputs/ea_lite/wrist020_clean50_20261015/data/manifest.ready.json，标签在同目录labels，新训练未启动。按(task,episode)对齐2500条记录比较文件SHA256：2411不同、89相同；不将hash不同直接当作所有字段都改变。三个任务adjust_bottle/pick_dual_bottles/place_dual_shoes的episode0实际npz逐数组核对：effectors、arm_present、targets前70D非视觉监督逐元素相同；后60D bbox/visibility不同，符合near修复后的图像监督更新。未改任何训练配置、数据或进程。


### 2026-10-15T09:01:37.765669+00:00：目标集群ensurepip缺失的环境安装修复

用户在(base)执行cluster安装器，系统python3.10缺ensurepip，旧python -m venv失败并留下半成品。仅修改scripts/cluster/install_environment.sh：先找uv，缺工具时明确提示在活动环境安装uv；不完整.venv改名备份后用uv venv --no-python-downloads --seed创建，不依赖系统venv包；健康环境复用，单独缺pip时uv补pip，继续原锁定依赖安装主体。无模型/数据/训练配置变更、无sudo或系统包修改；本机正在训练的.venv未改动。真实/usr/bin/python3.10无ensurepip的隔离恢复验证通过，失败文件保留、pip/activate及UV_BIN交接通过；主体Torch安装用stub隔离，目标完整安装仍待用户执行。说明ENVIRONMENT_ENSUREPIP_FIX_20261015.md提供无需重传数据的命令。已交付原43GB大包保持不变，另提供小安装补丁。


### 2026-10-15T12:29:22Z：uv 新旧索引参数优先级导致 cu128 安装失败的修复

目标用户日志：第一阶段从镜像装入 torch2.7.1（无+cu128，NVIDIA依赖12.6），第二阶段锁定torch2.7.1+cu128无候选。此前建议的UV_DEFAULT_INDEX与旧脚本--index-url混用有缺陷。通过本机uv0.13.0、独立localhost HTTP索引及--dry-run复现：环境UV_DEFAULT_INDEX指mirror时，CLI --index-url仍请求mirror；改CLI --default-index请求显式pytorch路径。仅修改scripts/install_ea_environment.sh第一阶段为--default-index与三包显式+cu128，并在进入其余lock安装前核对发行版metadata精确版本。保留通用依赖镜像、已有venv与缓存；不改模型/数据/训练参数，不执行任何真实安装或GPU操作。shell语法通过，本机既有三包metadata核验通过；没有重跑全依赖安装或目标GPU验收。目标用户需同步小脚本修改后重新执行；cu126到cu128替换仍需下载不同构建，不承诺复用所有已下载字节。暂停的4350数据验收仍保持暂停。


### 2026-10-15T12:43:34Z：PyTorch cu128 本地 wheel 交付完成

目标PyTorch官方源及用户测试的交大镜像连接超时。用户已有torch2.7.0+cu128/torchvision0.22.0+cu128，不满足现有2.7.1/0.22.1锁，保持训练环境版本要求。新增installer可选OPENWAM_TORCH_WHEEL_DIR分支：校验三件套精确文件名存在，直接本地wheel安装、普通依赖沿用可达镜像，第二阶段--find-links复用wheel，保留metadata版本门槛和锁文件；默认在线路径不变。未在正在训练的.venv执行安装或卸载。

本机官方索引可达，已下载torch2.7.1+cu128(1039365846 bytes)、torchvision0.22.1+cu128(8730539)、torchaudio2.7.1+cu128(3863565)，均cp310/cp310/manylinux_2_28_x86_64。三文件SHA256与官方index锚点逐项一致、ZIP METADATA名称/版本/Tag核对通过；官方download-r2链接保留于WHEEL_VERIFICATION.json。打包/逐成员SHA256/仅regular-file与归档SHA256通过：/remote-home/jinminghao/transfer_packages/OpenWAM_torch_cu128_cp310_20261015.tar，1051985920 bytes，旁置.tar.sha256。包含修复installer与README_INSTALL.md，原项目/数据/权重大包不变。

验证：bash -n通过，临时stub覆盖在线显式index、本地三wheel+mirror+find-links、缺wheel安装前拒绝三路径；真实uv --dry-run --no-index --no-deps对三wheel解析通过，未实际安装。此包不是全环境离线包：nvidia运行库和其他依赖仍由目标清华镜像获取。目标完整依赖安装、CUDA扩展、驱动/NCCL和训练尚未验收；下一步用户上传小包，校验、复制installer、设置wheel目录后继续安装，再pip check和目标GPU检查。暂停的数据验收未恢复，无训练进程操作。


### 2026-10-15T13:01:28.128719+00:00：用户要求补齐 NVIDIA cu128 大包并本地安装

用户列出慢速nvidia运行库并授权本机下载打包。按现有requirements-ea-lock固定版本获取14个nvidia-*、triton3.3.1、setuptools79.0.1，共16个Linux/CPython3.10兼容wheel、2800356559 bytes；每个文件均按PyPI per-release JSON的大小/SHA256与ZIP METADATA包名版本检查通过，未改训练版本。新增OPENWAM_CUDA_WHEEL_DIR可选安装路径：先--no-index从运行库包的requirements-cuda-runtime.txt完整安装CUDA依赖，再将runtime/Torch目录共同作为两阶段find-links；保留既有Torch本地入口、三包版本门槛和通用依赖镜像。默认路径不变，正在训练的.venv与GPU进程未操作。

交付/逐成员SHA256/regular-file与归档SHA256核验通过：/remote-home/jinminghao/transfer_packages/OpenWAM_cuda_runtime_cu128_cp310_20261015.tar，2800414720 bytes，SHA256 f459a2bb29181b42c20b887e9c5e0476a042f921c4b1552ffe406fc0d54563b1，旁置.tar.sha256；与此前1.05GB Torch三件套包配合使用，不重复携带Torch。包含更新后的install_ea_environment.sh、runtime锁清单、PyPI哈希证据、SHA256SUMS、README_INSTALL.md。

验证：shell语法、临时stub三阶段参数构造通过；新建空Python3.10虚拟环境，以真实uv --dry-run --no-index只用本地wheel成功解析完整16包依赖闭包（OFFLINE_RESOLVE_CHECK.txt）。初次尝试的uv --ignore-installed参数不支持，未安装任何内容；改用新空环境完成独立解析。未在本机实际安装/卸载NVIDIA包，也未启动训练。其余项目依赖仍需清华镜像，完整环境离线与目标GPU驱动/NCCL/训练尚未验收。目标应先结束仍运行的安装命令，校验上传包、复制新installer并设置两个本地目录再继续；保留.venv及缓存。暂停的4350数据验收保持暂停。


### 2026-10-16：RoboTwin标准评估本地化与首轮运行（进行中）

用户目标为先跑通官方Clean2Random checkpoint，再具备OpenWAM自身标准评测流程；已确认评测层修复，并要求复制runtime、不影响旧DiT4DiT。当前项目主线为80D/32步EEF Lite，历史native14/40步V3文档不能用于该合同。

已复用现有OpenWAM服务/adapter/native evaluator，补齐小样本原生结果分母、独立run身份/结果验收工具；默认保持官方专家筛seed、unseen、RT和原动作预算。私有runtime/链接资产复制后为22793文件/16921381661逻辑bytes，无外部软链接，源码/config hash与独立inode/全文件size核验通过；保留所有tracked基线内容，原DiT4DiT源码/环境不安装不覆盖。31项联合CPU测试通过，最新13项纯评测测试通过；官方权重本机SHA256复核一致，完整BF16加载/normalizer/80D→20D/PONG确认，尚无策略episode结果。

首轮失败原因为迁移后的cuRobo.so需GLIBC2.32而当前系统2.31，自动JIT又因PATH缺ninja失败；不改planner或shader绕过。正在用本地12.4 toolkit+torch2.7.1cu128、L20 sm89编译独立缓存，多个模块已成功。环境副本复制仍进行中，期间检查只读旧已安装依赖、关闭pycache写入，cuRobo源/资产/输出均来自OpenWAM私有目录。下一步必须完成MotionGen/专家规划、真实Clean/Random策略rollout及独立环境验证，之后推进标准100条/全任务入口；不能据服务READY或CPU测试声称闭环跑通。

完整设置、证据、当前run路径及后续命令见`docs/ROBOTWIN_STANDARD_EVALUATION_20261016.md`。


2026-10-16T04:22Z 追加：系统ffmpeg PATH隔离修复后，官方Clean2Random真实adjust_bottle闭环Clean1/1、Randomized1/1通过；Clean seed100001、123/400动作成功，原生_result.txt=1.0/验收一致，H264录像124帧可解码、首尾确认横卧瓶到右手竖直拿起。仅冒烟，不是统计成绩。独立仿真环境复制exit0，本地prefix/torch/cuRobo五个.so/CUDA实际导入通过，原.so保留hash备份。

用户要求多GPU尽量并行，已启动parallel_official100_20261016：100个唯一task/split×各100条，17 workers，GPU7已有模型先供1个标准100条客户端，GPU0共享模型新增16个独立端口/PolicyServer/WAMPolicy动作缓存，predict互斥串行执行相同原engine，多GPU仿真并行。新真实policy/executor的跨端口reset与调用互斥加原评测共16项CPU通过；不改模型/采样参数/RT/成功判据/预算。当前首个正式100条run已成功1个episode，GPU0模型仍加载、16个waiting_model，完整100条和17路真rollout/容量仍待验收。动态队列保留失败不自动换seed，完整coverage前无全量分数；源DiT4DiT diff hash复核不变。详见专用评估记录。
