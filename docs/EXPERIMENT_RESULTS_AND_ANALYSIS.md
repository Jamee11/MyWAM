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


### 2026-10-17：旧 Lite C joint 五 epoch 训练完成核对

范围仅为 outputs/ea_lite/clean50_C_joint_bs64_5epochs_lr1e4_jitter_20261015/runs/2026-10-15_03-27-34。用户要求不处理另一组实验，本次不对其他实验采取操作。

该组配置为原自采Clean50数据、Lite C joint、bs64/rank、acc1、global256、5epochs、LR1e-4、ColorJitter开启、原EA权重、每2000步周期保存。train.log确认10845/10845更新完成，训练进度计时40:06:21，末次显示loss0.2409/video0.0840/action约0.0003；这些是末批显示值而非全数据评估。2026-10-16 19:35:25日志确认Saved final，checkpoint_step_10845.safetensors实际存在、大小24842070546 bytes。

现场保留权重8000、10000、10845三份，完整恢复状态为8000、10000（最终10845只保存权重）。周期keep_last_k=2会清理旧周期产物，但keep_final_resume_state=true使finish阶段跳过统一清理，因此最终额外权重使现场权重数为3；不能声称结束后严格只保留两份，也不能把10000恢复状态说成10845状态。未做24.84GB全量hash或部署/闭环评估。

结论：此旧run的计划训练与最终权重保存已完成，不构成模型效果提升结论。该run仍受已知旧数据wrist near0.1视觉问题限制。下一步如需评估，应明确选用最终10845权重并区分旧数据试验与后续匹配数据对照；本次仅只读核对和记录，没有启动/停止/恢复任何实验或改动训练配置。


### 2026-10-18：官方RoboTwin复现审计更新

97项百条完成，Clean全50=80.66%对官方89.36%；Randomized同47=38.7234%对官方同47=47.7660%。详见[官方复现审计](ROBOTWIN_OFFICIAL_REPRODUCTION_AUDIT_20261018.md)及outputs/robotwin_eval/audit_official_20261018/per_task_comparison.csv。未改当前队列、核心代码、依赖或原DiT4DiT。

已澄清“提交不同”的范围：直接按官方验证提交tree blob核对envs/task_config/description/script，719文件相同，仅_base_task/test_render历史默认关闭扩展及_install安装差异；不能把GitHub compare的共同祖先差异列表当两端点差异。主要未解决项仍是SAPIEN3.0.3对3.0.0b1、cuRobo v0.7.7-9-gd64c4b0对v0.7.8、Warp1.12.1对1.13、仿真Torch2.7.1对2.4.1，以及外部资产revision/hash未全量对照。cuRobo既筛seed也执行策略EEF轨迹，需受控消融，不能宣称已证明低分根因。共享pool仅CPU隔离/互斥测试，真实模型独立服务等价仍未做；不因长期无崩溃断言数值/动作等价。

随机数记录修正：实际engine.generate默认seed42，动作和视频初始化用显式新generator/seed，故早期“共享请求交织导致模型全局噪声流变化”不能直接套用；规划器/环境RNG仍需分别验证。当前100个run启动wrapper/adapter/deploy/policy/step-limit各自hash一致，官方主体生成/预处理代码静态审计一致，未启用EA分支。后续保留结果，独立准备官方栈，优先鞋子/scan_object等差距任务，固定输入/seed/指令对照，不改预算/判据逼分。


### 2026-10-18T06:34:34.447744+00:00：用户授权双轨RoboTwin重建与Sidecar20k评测

新官方源码/Conda/资产全部位于OpenWAM内独立路径，GPU0–3用于复现和控制，GPU4–7用于Sidecar现有环境；旧DiT4DiT不变。官方指定依赖已镜像安装完成，cuRobo v0.7.8及PyTorch3D独立编译准备中；镜像资产三zip固定revision并逐项size/SHA验证/解包/与旧资产hash比较中。Sidecar两个服务已实际strict加载并PONG就绪，12个原生100条仿真客户端启动中，尚无完成episode，不能声称效果通过。20k完整权重已复制独立文件防止训练keep_last策略删源；20k文件对应10000次optimizer更新。

Sidecar源码明确拒绝DiTcache、不支持原compile，按其已有部署配置关闭二者；历史baseline配置不同，额外启动原模型同部署控制全量队列。原模型与Sidecar按各自训练的quaternion/normalizer合同处理，不把表示差异伪装为同权重结构消融。新资产Aloha config planner字符串不同，但实际EEF路径仍固定CuroboPlanner，尚不能归因；全部资产hash比较未完成。新增可选配置/任务子集启动器16项回归通过，默认旧行为保留。独立官方服务与共享池的真实双观测32动作交错/reset对照已启动等待服务；结果未出前不作等价断言。详见ROBOTWIN_DUAL_TRACK_EVALUATION_20261018.md。


### 2026-10-18T06:43:58.768972+00:00 cuRobo源码版本表述修正

重新clone官方v0.7.8后发现其tag指向d64c4b005459db10c5dd867d8b30a87d5bda9bdb，旧cuRobo HEAD同为此提交且两者tracked diff均空。旧describe=v0.7.7-9-gd64c4b0是旧checkout缺少v0.7.8 tag造成的版本描述，不代表算法源码不同；不能将前文版本字符串差异当作cuRobo源码升级/已定位的根因。新环境已实际编译安装nvidia_curobo0.7.8（Torch2.4.1），旧扩展绑定Torch2.7.1；SAPIEN/Torch/Warp及二进制运行栈仍有差异。证据outputs/robotwin_eval/rebuild_official_20261018/curobo_source_equivalence.json及git ls-remote tag。当前正式Sidecar队列为balanced_queue7/balanced_queue5共10sim，旧12sim pilot因实测显存余量仅3–4GiB而停止并保留，不混入正式SR。


### 2026-10-18T06:47:38.838199+00:00：真实独立服务/共享池动作对照通过

fresh官方OpenWAM服务19400（GPU2）对原共享pool19000/19001（GPU0），发布同一权重、官方默认cache/compile/10步，使用两个真实HDF5首观测的三视图/EEF/固定指令，各32动作；池两端口交错请求，再reset并交换观测。四组序列均bitwise_equal=true、max_abs_error=0.0。只证明此真实fixture范围，不外推所有输入/长期共享状态或Sidecar模型；本次已明显降低“共享池本身改变原模型action”的嫌疑。证据outputs/robotwin_eval/rebuild_official_20261018/action_parity/result.json，原始请求inputs.json保留。Sidecar原20k与独立副本SHA256均0d67d651577ef53d5a86b0e1e03dd93f6cc7bf652c3c7af08734b7712042ea6e，size24862363126，copy_verification.match=true。新物体包9368文件与旧资产全同；背景纹理还在逐文件核验，官方native诊断仍等新环境完整编译验收。


### 2026-10-18T06:54:16.855785+00:00：旧baseline尾项渲染失败与完整重跑

stack_bowls_three/demo_randomized在55/92、seed100178后，下一episode获取left_camera RGB时报RuntimeError cannot create buffer，原100条作业已标failed/incomplete，不能作为最终SR。发生于Sidecar12sim pilot扩容期间，GPU6此前仅余约4GiB，可能受显存压力影响；不能仅凭异常字符串断定所有底层原因。保留全部旧日志，Sidecar已调为10sim并分散GPU7；在GPU2较大余量下以同一旧环境/权重/cache/预算/seed group0完整重跑100，目录historical_recovery_stack_bowls_three_randomized_20261018，复用原pool19013。禁止将旧92与重跑拼接、禁止选择较高分；最终汇总应以有效完整重跑替换此task/mode，其他原完成项保留。旧parallel.json继续保留失败事实，不手工改成complete。


### 2026-10-18T07:00:38.577430+00:00：官方独立环境与资产完整验收

Torch2.4.1+cu121/SAPIEN3.0.0b1/Warp1.13.0/mplib0.2.1、cuRobo0.7.8五模块、PyTorch3D0.7.8均在新prefix安装/编译完成。runtime_smoke.json确认五个本地.so实际导入、CUDA张量有效及RT32spp/depth8/oidn离屏32×32RGBA有限。完整依赖检查发现Conda默认继承/root/.local用户site；已启用PYTHONNOUSERSITE=1、在新prefix补齐缺失依赖，pip_check_isolated.log为No broken requirements found，isolated_import_paths.json确认九个关键模块均新prefix加载。旧环境不改。第一次native仅作为preflight_usersite保留，正式official_diagnostic100隔离重启，不混结果。

镜像三zip均上游size/SHA通过，物体9368/9368、背景11000/11000与旧资产同hash。Aloha除config.yml planner字段外文件相同；官方update_embodiment_config_path.py实际生成六个本地路径配置，left config替换Root前缀后与旧版逐字相同。旧流程实际引用InternVLA目录URDF/碰撞文件，已只读比对与本地副本hash相同；不能仅由副本无symlink宣称旧流程完全不依赖外部路径。Sidecar按用户要求保留该现有流程，新官方流程全部绑定新目录。资产、源码、runtime与动作对照证据见rebuild_official_20261018。

Sidecar正式10路已完成首批episode，但未满任何任务100条，不给总体效果结论；初始pilot不计。原版同cache/compile关闭控制4路继续运行，原baseline失败尾项完整100重跑继续，保留旧不完整92条及失败事实。


### 2026-10-18T07:04:32.049446+00:00：双轨真实闭环启动验收

独立官方环境正式place_dual_shoes/Clean已经专家筛seed后进入策略take_action，现场最后步数[('102', '600')]，原预算600，源码/独立服务/环境/资产均新路径；native环境PYTHONNOUSERSITE=1实查。Sidecar正式10sim与原版同部署控制4sim均推进真实闭环；当前未满单任务100，不能给总体SR。16项回归通过、六个修改/新增Python入口ruff通过；新Conda isolated pip check通过。原DiT4DiT diff hash仍为63b6f6bd72695fd2d2c573427b49e59ed3721c8d941b48864899390a8ed1f884，与首次复制记录一致。旧尾项92条渲染失败保留并完整重跑，不混入有效百条。后续按完整结果核查accepted seeds及差值，单个episode/当前子集不作效果结论。


### 2026-10-18 09:14 UTC：首个完整Sidecar结果及新官方环境初步改善

新官方环境place_dual_shoes/demo_clean已34/36（94.44%），尚未满100；旧环境该项完整13/100，官方issue46报告89/100。结合前述真实输入action parity完全一致、物体/背景资产hash一致，当前证据支持优先排查仿真运行栈差异，但单任务部分样本不能证明具体SAPIEN/Torch/Warp版本的因果作用，也不能外推其余任务。继续原生100条，不提前停表。

Sidecar正式两分区合计约370/10000条（约3.7%），首个完整task/mode为place_container_plate/Clean=99/100，无result错误，与历史发布模型该项99/100相同；同cache/compile关闭控制此项仍排队，不能称已完成匹配部署比较。其余当前部分样本：adjust_bottle/Clean=43/46、Randomized=6/21；beat_block_hammer/Clean随后55/59、Randomized=22/56；place_cans_plasticbox/Clean26/26、Randomized15/17。仍不能给50任务综合宏平均。

读取各日志相同前N完成episode进行对照：adjust_bottle/Clean前46条Sidecar43/46、匹配部署原版45/46，accepted seed前缀全部相同；Randomized前21条Sidecar6/21、原版5/21，seed前缀全部相同。beat_block_hammer/Clean前59条Sidecar55/59、原版40/59；Randomized前51条Sidecar20/51、原版18/51，但这两项accepted seed前缀不同，不能视为逐seed配对改善。总体看Sidecar部分Clean任务表现积极，Randomized瓶颈尚在，须等待完整覆盖再判断。下一步继续百条及全量队列，按有效完整结果和accepted seeds复核。


### 2026-10-18 16:13 UTC：官方双鞋百条完成、Sidecar六项完整结果

新官方环境place_dual_shoes/Clean完整92/100，result complete=true/errors=[]；旧环境13/100，官方issue46为89/100。当前Randomized3/20，仍部分样本。相同发布模型、已核对资产及真实输入动作parity条件下，新官方栈显著改善该Clean任务，但尚不能归因到某个独立依赖。后续继续原定五任务双域100。

Sidecar正式两个分区各3 complete/5 running/42 queued，共6/100项、1036/10000 episodes（10.36%）。完整结果且errors均空：adjust_bottle/Clean95；beat_block_hammer/Clean92、Randomized35；place_cans_plasticbox/Clean100；place_container_plate/Clean99、Randomized70（每项分母100）。同cache/compile关闭原模型完成4项：adjust_bottle/Clean97、Randomized33；beat_block_hammer/Clean65、Randomized33。锤子Clean完整任务差值+27个百分点，Randomized+2个百分点；专家接受seed集合可能不同，不能视为严格逐seed配对/纯架构消融。Sidecar当前仍旧环境，双鞋Clean3/31，应与新官方环境92/100区分。

历史原版99项完整、1项旧失败；指定stack_bowls_three/Randomized完整重跑68/100已完成（取代旧55/92，不拼接/选高），因此合并有效覆盖100项/10000条，总5981成功，综合59.81%，Clean80.66%、Randomized38.96%。旧parallel.json失败记录保留。

速度：各服务最后500个完整10步Joint denoising循环，读取最终tqdm rate估算循环时长：Sidecar GPU7均1.18064s、中位1.19617s；GPU5均1.53218s、中位1.53139s；原模型同cache/compile关闭均0.94623s、中位0.94029s；旧原模型cache/compile开启均0.39028s、中位0.41425s。Sidecar相对优化开启历史运行约3.03–3.93倍循环耗时，相对均关闭优化原版约1.25–1.62倍。此为现有并发负载下循环观测，非隔离硬件严格架构benchmark，也非同步CUDA事件或端到端请求延迟（不含排队/预处理/仿真）；两Sidecar GPU任务负载不同，不将全部差值归因于模块。数据来自各pool.log，未改变现行评测或插入请求。

下一步维持正式队列推进，等待更多完整task/mode，分别汇总新官方环境诊断与旧环境Sidecar/匹配部署控制，不把当前混合任务部分样本平均当全量指标。


### 2026-10-18 16:17 UTC：按用户要求计算原模型替换汇总

对原模型历史99项有效百条加指定失败项完整恢复，确认100项/10000条，Clean4033/5000=80.66%、Randomized1948/5000=38.96%、综合5981/10000=59.81%。新官方诊断当前仅place_dual_shoes/Clean完成有效百条，13/100→92/100，增加79个成功（单项+79个百分点）。仅替换这一项后：Clean4112/5000=82.24%（+1.58个百分点）；Randomized仍38.96%；综合6060/10000=60.60%（+0.79个百分点）。此为混合新旧环境的假设替换汇总，不能命名新官方环境全量成绩；Randomized及其余诊断项未满100不纳入。后续对预先指定任务所有有效完整重测均替换，无论升降，避免选择较高分。旧原始文件不改。计算快照：outputs/robotwin_eval/rebuild_official_20261018/mixed_environment_replacement_snapshot_20261018T1617Z.json。


### 2026-10-18T16:40:03.953137+00:00：新环境smoke通过；默认融合compile未通过数值门槛

Sidecar新官方仿真瓶子Clean与双鞋Clean各2/2、exit0、complete=true，均只作链路验收，不计正式任务成绩。GPU0同两真实输入、每组3次CUDA同步warm engine.generate：eager0.95865s、cache0.43736s（每10步skip6）、compile0.62616s、compile+cache0.26386s。默认Inductor融合相对各自eager出现max0.01953125/0.0078125动作误差，预设atol0.005/rtol0.01未通过；launcher正确拒绝正式队列，完整失败产物保留fast_validation_default_fusion。不能将失败改称parity通过，也未放宽门槛。

读取安装版Torch2.7.1 torch/_inductor/config.py说明：低精度融合默认省略BF16中间downcast/upcast；新增compile options emulate_precision_casts=True/force_same_precision=True以保留eager舍入边界，未改模型权重/计算拓扑/训练/缓存阈值。原服务停止后，新候选0验证及1/7服务用新选项重新加载，仍须相同门槛复验；若仍不通过则不能按精度等价放行。保留原官方19400单会话诊断。


### 2026-10-18T16:58:50.387754+00:00：Sidecar编译范围隔离与原版9小时并行验证

第二轮仅保留BF16舍入仍未通过原门槛（compiled max0.0234375、cached对应0.0078125）；原始产物保留fast_validation_precision_casts，不放宽容差。第三轮同一已加载模型依次测试conservative全forward（关闭pattern/epilogue融合、fusion size1）、仅Sidecar read/write、仅三层Sidecar FFN，全部保留原主干/三处读写，均按原eager/同缓存eager参考检查。新增serve_sidecar_checked.py在同一模型上完成验证后直接服务，避免重复加载；只有通过者才写最终deploy_selected.yaml/打开服务。四项CPU测试通过130.11s，相关入口ruff通过；实际GPU数值/速度仍待本轮结果。

用户进一步要求原版异常任务至多9小时出结果。原流程实际仅1个native客户端，仍原样保留；新增官方原有labtasker manifest/run_eval调度（源码不改），3个专家清单builder已实际筛seed，8个策略worker等待100条封存清单。任务为双鞋Randomized及单鞋/扫码/递积木/铃铛两域，共9项900条，每项10个无重叠10条分片。新原版模型pool GPU4、19900–19907，原19400/原native继续；部署默认cache/compile。Sidecar调整8sim（model0/1/7；sim0:2,5,0；sim1:6,2,1；sim7:6,7），原版分片8sim（3,3,3,5,5,5,6,6）以保留显存余量，不操作其他namespace/未知作业。目标截止2026-10-19T01:43UTC，须基于实际吞吐检查，不保证硬件未验证的绝对工期。

官方manifest另含固定instruction/CUDA RNG与任务状态初始化，不能声称与传统native逐位相同，也不能把两者样本拼接；分别报告。结构化JSON汇总按完整100及manifest hash/episode覆盖验证，禁止用上游小样本_result.txt固定100分母。保留native双鞋Clean92/100。目录baseline_official_manifest_deadline_20261018；进度/截止与有效结果自动更新docs/ROBOTWIN_LIVE_RESULT_ANALYSIS_20261018.md，并将新完整结果追加EXPERIMENT_RESULTS_AND_ANALYSIS。专门加速经验文档SIDECAR_INFERENCE_ACCELERATION_20261018.md已建立。


### 2026-10-18T17:07:59.134583+00:00：最终采用FFN局部compile + 联合cache，原门槛通过

本轮8变体在GPU0同两真实输入、每组3次CUDA同步warm，eager0.912392s，eager_cached0.377508s，conservative0.647082s/cached0.275161s，Sidecar读写compile0.941083s/cached0.388895s，FFN局部compile0.939386s/cached0.400740s。前两种编译范围至少一个未缓存参考未过原门槛，均保留失败，不放宽容差；FFN策略的两fixture×未缓存/同缓存四组动作对照max_abs_error全部0.0，原allclose门槛通过。FFN确有4个新Inductor kernels，非仅把enabled写true。

选择ffn：只编译3层Sidecar内部FFN，主干与三次read/refine/write全部保持原路径，未旁路/关闭Sidecar。cache开启0.99/max3，每generate reset，10步实际重算4次/复用6次已实测。组合相对完整eager约2.28倍，速度贡献主要来自cache；本轮FFN compile未观察到额外吞吐收益（相对cache-only略慢），不能称全模型compile已解决。相同缓存条件的编译保真不等于缓存与完整计算等价；缓存相对完整eager的两fixture动作最大差0.03125/0.01171875，须闭环百条判断成功率。

最终配置outputs/robotwin_eval/sidecar20000_officialenv_20261018/deploy_selected.yaml（compile.enabled=true/sidecar_strategy=ffn/cache=true），数值/耗时/全部动作证据fast_validation/result.json。checked模型GPU0直接复用为19600–19602，已预热/reset并启动正式queue0三个百条客户端：adjust_bottle两域、place_shoe/Clean；新官方Conda与runtime/PYTHONNOUSERSITE=1，原RT/预算/成功判据不改。模型GPU1/7新副本正在加载，各预热后启动queue1/queue7，合计8sim/50任务两域10000条，不混短测/旧环境。


### 2026-10-18T17:18:05.575263+00:00：正式新环境双队列启动验收

Sidecar三个模型GPU0/1/7已全部READY并以真实fixture预热/reset，三个pool配置SHA一致d1de0dd89a16eae0a9cb08b685c265100b17b766c77c6d0cf27d35dca70ceb6b。queue0/1/7分别3/3/2个百条作业启动，父进程实查全部存活，共8路覆盖100个唯一task/mode、10000条；queue0已完成真实成功episode，新增各路仍有初始化过程，不能把启动当任务完成。当前任务瓶子两域、单鞋Clean、双鞋两域、扫码Clean、锤子两域。

原版传统官方single_eval仍保留；新增官方manifest并行组8个run_eval子进程已实际创建，双鞋Randomized100专家清单封存，递积木Randomized100亦完成并开始下一专家任务，3个清单builder继续。run.json当前无errors。目标截止2026-10-19T01:43UTC；实际耗时须看分片吞吐，不把队列创建称已完成。每个清单全100，逻辑0–99分成10条且验证无重复；分析仅将协调器确认exit0/manifest/逻辑覆盖的完整100条纳入替换，不以progress=100代替验收。

启动验收证据outputs/robotwin_eval/sidecar20000_officialenv_20261018/launch_acceptance_snapshot.json，包含pid/存活/配置hash/当前baseline任务/显存。四项针对性CPU测试通过130.11s，七个新文件ruff全通过。加速经验与策略限制见SIDECAR_INFERENCE_ACCELERATION_20261018.md；实时结果、共同完整任务对比、替换平均及截止预测自动写ROBOTWIN_LIVE_RESULT_ANALYSIS_20261018.md，新完整成绩自动追加EXPERIMENT_RESULTS_AND_ANALYSIS.md。


2026-10-18T18:28:19.713533+00:00 Sidecar20k新环境 beat_block_hammer/demo_clean：99/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。


2026-10-18T19:13:31.799996+00:00 Sidecar20k新环境 place_shoe/demo_clean：94/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。


2026-10-18T20:24:49.972362+00:00 Sidecar20k新环境 place_dual_shoes/demo_clean：85/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。


2026-10-18T20:33:52.705528+00:00 Sidecar20k新环境 adjust_bottle/demo_randomized：34/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。


2026-10-18T21:03:00.590366+00:00 Sidecar20k新环境 adjust_bottle/demo_clean：98/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。


2026-10-18T21:20:05.387215+00:00 官方manifest复测 place_dual_shoes/demo_randomized：16/100，旧环境0/100；固定专家清单/指令、原成功判据/预算/RT。与传统native协议分列。源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/baseline_official_manifest_deadline_20261018/shards/demo_randomized/place_dual_shoes。


2026-10-18T21:55:16.802714+00:00 Sidecar20k新环境 handover_block/demo_clean：80/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。


2026-10-18T22:04:19.443354+00:00 Sidecar20k新环境 place_container_plate/demo_clean：98/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。


2026-10-18T22:28:27.761361+00:00 官方manifest复测 handover_block/demo_randomized：46/100，旧环境17/100；固定专家清单/指令、原成功判据/预算/RT。与传统native协议分列。源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/baseline_official_manifest_deadline_20261018/shards/demo_randomized/handover_block。


2026-10-18T22:42:33.097384+00:00 官方manifest复测 click_bell/demo_randomized：77/100，旧环境78/100；固定专家清单/指令、原成功判据/预算/RT。与传统native协议分列。源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/baseline_official_manifest_deadline_20261018/shards/demo_randomized/click_bell。


2026-10-19T00:06:01.713359+00:00 Sidecar20k新环境 place_shoe/demo_randomized：49/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。


2026-10-19T00:16:05.003965+00:00 Sidecar20k新环境 scan_object/demo_clean：83/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。


2026-10-19T00:20:06.313364+00:00 Sidecar20k新环境 scan_object/demo_randomized：53/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。


2026-10-19T00:31:10.548306+00:00 官方manifest复测 place_shoe/demo_randomized：49/100，旧环境19/100；固定专家清单/指令、原成功判据/预算/RT。与传统native协议分列。源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/baseline_official_manifest_deadline_20261018/shards/demo_randomized/place_shoe。


2026-10-19T00:57:19.201381+00:00 Sidecar20k新环境 place_dual_shoes/demo_randomized：10/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。


2026-10-19T01:03:21.166950+00:00 Sidecar20k新环境 beat_block_hammer/demo_randomized：40/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。


2026-10-19T01:13:24.661312+00:00 官方manifest复测 handover_block/demo_clean：85/100，旧环境86/100；固定专家清单/指令、原成功判据/预算/RT。与传统native协议分列。源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/baseline_official_manifest_deadline_20261018/shards/demo_clean/handover_block。


2026-10-19T01:21:27.885373+00:00 官方manifest复测 place_shoe/demo_clean：97/100，旧环境50/100；固定专家清单/指令、原成功判据/预算/RT。与传统native协议分列。源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/baseline_official_manifest_deadline_20261018/shards/demo_clean/place_shoe。


2026-10-19T01:30:32.658765+00:00 官方manifest复测 click_bell/demo_clean：100/100，旧环境100/100；固定专家清单/指令、原成功判据/预算/RT。与传统native协议分列。源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/baseline_official_manifest_deadline_20261018/shards/demo_clean/click_bell。


2026-10-19T01:59:45.973492+00:00 Sidecar20k新环境 click_alarmclock/demo_clean：100/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。


2026-10-19T02:44:10.988321+00:00 官方manifest复测 scan_object/demo_randomized：56/100，旧环境14/100；固定专家清单/指令、原成功判据/预算/RT。与传统native协议分列。源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/baseline_official_manifest_deadline_20261018/shards/demo_randomized/scan_object。


2026-10-19T02:48:12.853364+00:00 Sidecar20k新环境 place_container_plate/demo_randomized：69/100；实际优化配置见sidecar20000_officialenv_20261018/deploy_selected.yaml；训练数据/表示合同不同，不是纯架构消融。


2026-10-19T03:04:19.744483+00:00 官方manifest复测 scan_object/demo_clean：89/100，旧环境66/100；固定专家清单/指令、原成功判据/预算/RT。与传统native协议分列。源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/baseline_official_manifest_deadline_20261018/shards/demo_clean/scan_object。


2026-10-19T04:29:24.681371+00:00 Sidecar32k L20 click_alarmclock/demo_clean：98/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu5/queue/jobs/demo_clean/click_alarmclock。独立运行，不与A100或20k样本混合。


2026-10-19T05:21:39.061392+00:00 Sidecar32k L20 beat_block_hammer/demo_clean：98/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_clean/beat_block_hammer。独立运行，不与A100或20k样本混合。


2026-10-19T05:41:45.837369+00:00 Sidecar32k L20 place_container_plate/demo_clean：97/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_clean/place_container_plate。独立运行，不与A100或20k样本混合。


2026-10-19T06:59:09.533809+00:00 Sidecar32k L20 grab_roller/demo_clean：100/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_clean/grab_roller。独立运行，不与A100或20k样本混合。


2026-10-19T07:13:13.991510+00:00 Sidecar32k L20 place_dual_shoes/demo_clean：88/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_clean/place_dual_shoes。独立运行，不与A100或20k样本混合。


2026-10-19T07:15:14.749370+00:00 Sidecar32k L20 scan_object/demo_clean：87/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu7/queue/jobs/demo_clean/scan_object。独立运行，不与A100或20k样本混合。


2026-10-19T07:17:15.473359+00:00 Sidecar32k L20 move_pillbottle_pad/demo_clean：96/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_clean/move_pillbottle_pad。独立运行，不与A100或20k样本混合。


2026-10-19T07:36:22.041365+00:00 Sidecar32k L20 adjust_bottle/demo_clean：97/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu0/queue/jobs/demo_clean/adjust_bottle。独立运行，不与A100或20k样本混合。


2026-10-19T07:51:28.417413+00:00 Sidecar32k L20 click_alarmclock/demo_randomized：65/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_randomized/click_alarmclock。独立运行，不与A100或20k样本混合。


2026-10-19T07:52:29.022205+00:00 Sidecar32k L20 place_cans_plasticbox/demo_clean：99/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_clean/place_cans_plasticbox。独立运行，不与A100或20k样本混合。


2026-10-19T07:54:29.893533+00:00 Sidecar32k L20 handover_block/demo_clean：93/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu0/queue/jobs/demo_clean/handover_block。独立运行，不与A100或20k样本混合。


2026-10-19T08:09:35.928244+00:00 Sidecar32k L20 click_bell/demo_clean：100/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_clean/click_bell。独立运行，不与A100或20k样本混合。


2026-10-19T08:34:45.193352+00:00 Sidecar32k L20 grab_roller/demo_randomized：49/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_randomized/grab_roller。独立运行，不与A100或20k样本混合。


2026-10-19T08:37:46.289368+00:00 Sidecar32k L20 beat_block_hammer/demo_randomized：33/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_randomized/beat_block_hammer。独立运行，不与A100或20k样本混合。


2026-10-19T08:43:48.266709+00:00 Sidecar32k L20 place_shoe/demo_clean：92/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu5/queue/jobs/demo_clean/place_shoe。独立运行，不与A100或20k样本混合。


2026-10-19T08:57:52.875189+00:00 Sidecar32k L20 adjust_bottle/demo_randomized：37/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu0/queue/jobs/demo_randomized/adjust_bottle。独立运行，不与A100或20k样本混合。


2026-10-19T09:21:35.887507+00:00 Sidecar32k L20 scan_object/demo_randomized：52/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu7/queue/jobs/demo_randomized/scan_object。独立运行，不与A100或20k样本混合。


2026-10-19T09:32:39.450788+00:00 Sidecar32k L20 place_container_plate/demo_randomized：79/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_randomized/place_container_plate。独立运行，不与A100或20k样本混合。


2026-10-19T09:42:59.790345+00:00 Sidecar训练收敛审计：4rank连续36030micro、全部有限且跨rank指标已global mean；32k→36k total -1.81%、action -18.67%、relation -43.00%、plan -27.28%。ground3.386接近3000帧soft-target熵抽样3.382，不能将非零底部当未收敛。无独立val曲线，不能从train loss判Randomized泛化。报告docs/SIDECAR_TRAINING_LOSS_ANALYSIS_20261019.md，图/CSV/原始统计同evidence目录；未改训练或进程。


2026-10-19T10:22:59.833359+00:00 Sidecar32k L20 pick_diverse_bottles/demo_clean：78/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_clean/pick_diverse_bottles。独立运行，不与A100或20k样本混合。


2026-10-19T10:27:01.445370+00:00 Sidecar32k L20 lift_pot/demo_clean：100/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_clean/lift_pot。独立运行，不与A100或20k样本混合。


2026-10-19T10:56:12.325386+00:00 Sidecar32k L20 blocks_ranking_rgb/demo_clean：95/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu7/queue/jobs/demo_clean/blocks_ranking_rgb。独立运行，不与A100或20k样本混合。


2026-10-19T11:20:22.102131+00:00 Sidecar32k L20 place_cans_plasticbox/demo_randomized：86/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_randomized/place_cans_plasticbox。独立运行，不与A100或20k样本混合。


2026-10-19T11:31:25.955300+00:00 Sidecar32k L20 dump_bin_bigbin/demo_clean：91/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_clean/dump_bin_bigbin。独立运行，不与A100或20k样本混合。


2026-10-19T11:32:26.417972+00:00 Sidecar32k L20 place_bread_skillet/demo_clean：94/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_clean/place_bread_skillet。独立运行，不与A100或20k样本混合。


2026-10-19T12:11:41.891215+00:00 Sidecar32k L20 pick_diverse_bottles/demo_randomized：61/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_randomized/pick_diverse_bottles。独立运行，不与A100或20k样本混合。


2026-10-19T12:50:57.313364+00:00 Sidecar32k L20 place_shoe/demo_randomized：66/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu5/queue/jobs/demo_randomized/place_shoe。独立运行，不与A100或20k样本混合。


### 2026-10-19T12:56:44.550636+00:00 原版 baseline 冗余队列收尾

用户指出预定原版异常任务复测已完成。确认 baseline_official_manifest_deadline_20261018/run.json 为 complete；另一条传统 native 串行复核队列仍在运行，属于调度收尾遗漏。本次停止该冗余控制器、当前 place_shoe/Randomized 客户端及其 ffmpeg、19400 原版服务及编译子进程，共 10 个进程；所有已有结果和日志保留，native run.json 明确标为 stopped_redundant，不将未完成结果视为完整评测。详细进程与显存证据：outputs/robotwin_eval/rebuild_official_20261018/official_diagnostic100/redundant_queue_stop_20261019.json。

停止前后 GPU2 用量 38144→13459MiB，释放 24685MiB（约24.1GiB）；GPU3 用量 41341→35567MiB，释放5774MiB（约5.6GiB）。核验32k的6个模型服务、18个评测worker均存活，未停止它们；GPU2仍有Sidecar仿真，不称整卡已空闲。后续仅推进32k既定队列及指定失败恢复，额外原版复核不再继续。


2026-10-19T13:16:07.473370+00:00 Sidecar32k L20 place_mouse_pad/demo_clean：83/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_clean/place_mouse_pad。独立运行，不与A100或20k样本混合。


### 2026-10-19T13:22:01.297432+00:00 GPU2 第七份32k模型与仿真按容量扩容

用户授权利用GPU2空余显存增加相同32k模型，并调整/增加仿真。复用既有serve_checked_ffn.py及expand_gpu6.py中的原子保存、真实退出码和百条验收函数，新增独立输出目录控制器expand_gpu2.py：接管GPU7原队列20个条件的3个运行子进程（3849503/4021124/4145828），只暂停原调度器3490337，不暂停任何客户端；原3个子进程退出且验证成功后才终止该已暂停调度器。源快照保留，新权威队列expanded_queue27替代gpu7/queue；与gpu0/1/4及expanded_queue56合计恰好100个唯一task/mode，无重复派发。

新增模型启动PID170637（控制器170490），GPU2，端口21620–21622，独立gpu2/inductor缓存，完全复用固定32k checkpoint/deploy.yaml/官方环境/动作预算/RT；加载等待GPU2空余>=32000MiB。仍须本卡FFN+cache数值验收及端点token检查通过，才能启用新增3个worker。GPU2现有handover_mic/Clean完整100条结束后其后续任务才可迁至其他卡；press_stapler/Clean继续。新队列后续任务在GPU0/2/3/5/6/7中按空闲显存选择，GPU2验收前不分配新仿真；每次目标卡需>=12288MiB，并串行等待60秒让renderer分配后再选下一卡。GPU1/4当前余量不足，新增worker不选择它们；不操作未知进程。7模型/21仿真为目标上限，实际并发以running为准，不保证立即增加3路或线性加速。

更新本次输出内watch_results.py并保留变更前快照；监控PID170795，识别expanded_queue27并排除冻结源队列，验收前仍显示6模型18路、验收后显示7/21配置上限并附容量等待说明。click_bell指定恢复调度保持原协议和独占端点规则；其显存选择仍需>=12GiB，不能与新增仿真保证无竞态，已保留容量余量，后续须监视OOM/OIDN。当前仅启动/容量等待，尚未声称GPU2验收或新增闭环任务已开始。后续查看expansion_gpu2_state.json、gpu2/server.log、expanded_queue27/parallel.json。


2026-10-19T13:28:58.205362+00:00 Sidecar32k L20 open_laptop/demo_clean：92/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_clean/open_laptop。独立运行，不与A100或20k样本混合。


2026-10-19T13:37:01.113361+00:00 Sidecar32k L20 handover_mic/demo_clean：92/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu7/queue/jobs/demo_clean/handover_mic。独立运行，不与A100或20k样本混合。


2026-10-19T14:09:12.434207+00:00 Sidecar32k L20 press_stapler/demo_clean：94/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_clean/press_stapler。独立运行，不与A100或20k样本混合。


2026-10-19T14:35:21.806006+00:00 Sidecar32k L20 move_can_pot/demo_clean：100/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_clean/move_can_pot。独立运行，不与A100或20k样本混合。


2026-10-19T14:41:24.073369+00:00 Sidecar32k L20 dump_bin_bigbin/demo_randomized：64/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_randomized/dump_bin_bigbin。独立运行，不与A100或20k样本混合。


2026-10-19T15:00:32.610222+00:00 Sidecar32k L20 shake_bottle/demo_clean：100/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_clean/shake_bottle。独立运行，不与A100或20k样本混合。


2026-10-19T16:10:02.405350+00:00 Sidecar32k L20 lift_pot/demo_randomized：16/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_randomized/lift_pot。独立运行，不与A100或20k样本混合。


2026-10-19T17:26:37.203965+00:00 Sidecar32k L20 place_mouse_pad/demo_randomized：19/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_randomized/place_mouse_pad。独立运行，不与A100或20k样本混合。


2026-10-19T17:32:39.981363+00:00 Sidecar32k L20 move_can_pot/demo_randomized：38/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_randomized/move_can_pot。独立运行，不与A100或20k样本混合。


### 2026-10-19 18:10 UTC：32k与官方同任务阶段比较

完整百条36/100条件，有效4566/10000条；Clean23项94.09%对issue46相同23项94.39%，-0.30pp；Randomized13项51.15%对相同13项53.15%，-2.00pp。这些是按已完成覆盖配对的均值，不是全50任务均值；新增任务组成使均值变化，不能将相对上一时点的数值变化当作同一模型能力随时间提升。Randomized最高增益place_shoe66vs51、grab_roller49vs38、lift_pot16vs6、move_can_pot38vs29；主要缺口beat_block_hammer33vs55、adjust_bottle37vs51、click_alarmclock65vs77、place_cans_plasticbox86vs96。Clean总体接近，Randomized差异集中且有正负，不据单次100条宣称统计显著或结构因果。

当前19路running、3个failed。只读复核三个原始client.log末尾均为camera.take_picture RuntimeError: cannot create buffer：click_bell/Randomized6/8、shake_bottle/Randomized2/2、move_playingcard_away/Clean1/1；失败11条全部排除成绩。后两项为新增基础设施失败，目前未有本次已启动恢复证据，不将其计为策略失败，也不假称均已排队恢复。后续需恢复完整100条并保留原始日志，继续观察容量/OIDN与渲染错误；本轮报告未改调度/模型/参数。官方数据重新读取https://github.com/OpenWAM-Official/OpenWAM/issues/46；实时详情见docs/ROBOTWIN_SIDECAR32K_L20_LIVE_20261019.md。


2026-10-19T18:12:56.097387+00:00 Sidecar32k L20 stamp_seal/demo_clean：73/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_clean/stamp_seal。独立运行，不与A100或20k样本混合。


2026-10-19T18:33:03.721360+00:00 Sidecar32k L20 move_pillbottle_pad/demo_randomized：31/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_randomized/move_pillbottle_pad。独立运行，不与A100或20k样本混合。


2026-10-19T18:41:06.443479+00:00 Sidecar32k L20 place_a2b_right/demo_clean：97/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_clean/place_a2b_right。独立运行，不与A100或20k样本混合。


2026-10-19T18:48:09.110780+00:00 Sidecar32k L20 blocks_ranking_size/demo_clean：62/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu0/queue/jobs/demo_clean/blocks_ranking_size。独立运行，不与A100或20k样本混合。


2026-10-19T19:22:21.372206+00:00 Sidecar32k L20 open_laptop/demo_randomized：57/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_randomized/open_laptop。独立运行，不与A100或20k样本混合。


2026-10-19T20:25:44.214958+00:00 Sidecar32k L20 handover_mic/demo_randomized：15/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu7/queue/jobs/demo_randomized/handover_mic。独立运行，不与A100或20k样本混合。


2026-10-19T20:42:51.581375+00:00 Sidecar32k L20 place_dual_shoes/demo_randomized：10/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_randomized/place_dual_shoes。独立运行，不与A100或20k样本混合。


2026-10-19T20:49:53.629364+00:00 Sidecar32k L20 press_stapler/demo_randomized：46/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_randomized/press_stapler。独立运行，不与A100或20k样本混合。


2026-10-19T21:20:05.237363+00:00 Sidecar32k L20 place_empty_cup/demo_clean：100/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_clean/place_empty_cup。独立运行，不与A100或20k样本混合。


2026-10-19T21:24:06.753364+00:00 Sidecar32k L20 place_bread_skillet/demo_randomized：41/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_randomized/place_bread_skillet。独立运行，不与A100或20k样本混合。


2026-10-19T21:38:11.002970+00:00 Sidecar32k L20 blocks_ranking_rgb/demo_randomized：46/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu7/queue/jobs/demo_randomized/blocks_ranking_rgb。独立运行，不与A100或20k样本混合。


2026-10-19T21:42:12.837267+00:00 Sidecar32k L20 place_bread_basket/demo_clean：90/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_clean/place_bread_basket。独立运行，不与A100或20k样本混合。


2026-10-19T21:54:16.108917+00:00 Sidecar32k L20 stamp_seal/demo_randomized：28/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue56/jobs/demo_randomized/stamp_seal。独立运行，不与A100或20k样本混合。


2026-10-19T23:03:36.788381+00:00 Sidecar32k L20 place_object_stand/demo_clean：99/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_clean/place_object_stand。独立运行，不与A100或20k样本混合。


2026-10-20T00:22:53.379899+00:00 Sidecar32k L20 click_bell/demo_randomized：71/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/recovery_click_bell_randomized_attempt1。独立运行，不与A100或20k样本混合。


2026-10-20T00:25:53.742782+00:00 Sidecar32k L20 handover_block/demo_randomized：40/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu0/queue/jobs/demo_randomized/handover_block。独立运行，不与A100或20k样本混合。


2026-10-20T00:28:54.685376+00:00 Sidecar32k L20 place_a2b_right/demo_randomized：23/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_randomized/place_a2b_right。独立运行，不与A100或20k样本混合。


2026-10-20T01:25:06.561283+00:00 Sidecar32k L20 place_burger_fries/demo_clean：99/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue27/jobs/demo_clean/place_burger_fries。独立运行，不与A100或20k样本混合。


2026-10-20T01:59:15.550206+00:00 Sidecar32k L20 blocks_ranking_size/demo_randomized：15/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu0/queue/jobs/demo_randomized/blocks_ranking_size。独立运行，不与A100或20k样本混合。


### 2026-10-20 02:04 UTC：32k阶段评测效果

55个完整百条条件，6109条有效episode；Clean30项92.80%对官方相同30项92.97%（-0.17pp），Randomized25项43.52%对相同25项47.24%（-3.72pp）。55条件合计70.40%对72.18%（-1.78pp），覆盖不均衡，不能代替50任务双域全量均值。Clean总体接近但stamp_seal73vs89明显偏低；Randomized新增place_bread_skillet41vs61（-20pp）、handover_block40vs52（-12pp）。Randomized跨25任务仍有正负变化，整体暂低于官方，不能由训练loss收敛推出泛化提升。与上次13项51.15%的绝对均值变化来自新增任务覆盖，不代表同一固定集合性能随时间恶化。

click_bell/Randomized指定100条恢复已complete，71/100对官方75；原失败8条仍保留且不拼接。另shake_bottle/Randomized2条及move_playingcard_away/Clean1条失败部分共3条继续排除，不能按策略失败计入。当前监控14个running，部分0/0不代表实际已完成rollout；没有据配置21路宣称21路活跃。后续继续收集完整任务并解决剩余基础设施失败；本轮只读核查和结果记录，不修改运行代码/参数。源docs/ROBOTWIN_SIDECAR32K_L20_LIVE_20261019.md及相应run.json，官方issue46已存逐项快照。


2026-10-20T02:20:23.054211+00:00 Sidecar32k L20 place_object_stand/demo_randomized：63/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_randomized/place_object_stand。独立运行，不与A100或20k样本混合。


2026-10-20T02:27:25.656009+00:00 Sidecar32k L20 place_empty_cup/demo_randomized：77/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_randomized/place_empty_cup。独立运行，不与A100或20k样本混合。


2026-10-20T03:09:42.329350+00:00 Sidecar32k L20 open_microwave/demo_clean：61/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_clean/open_microwave。独立运行，不与A100或20k样本混合。


2026-10-20T03:15:44.701344+00:00 Sidecar32k L20 rotate_qrcode/demo_clean：85/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_clean/rotate_qrcode。独立运行，不与A100或20k样本混合。


2026-10-20T03:28:50.057448+00:00 Sidecar32k L20 place_bread_basket/demo_randomized：76/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_randomized/place_bread_basket。独立运行，不与A100或20k样本混合。


### 2026-10-20 04:06 UTC：评测数据采集进度与无有效产出进程

实时汇总60个完整条件（Clean32、Randomized28），6439/10000有效episode；相对02:04增加5个完整条件、330条有效episode。Clean同任务91.56vs92.53（-0.97pp），Randomized46.57vs49.54（-2.96pp）。原源队列59complete+click_bell指定恢复1complete=60，不得将源failed重复算作额外完整条件。有效完整双域任务28/50。

只读核查发现expanded_queue27三项被父调度器标running且PID存活，但无有效rollout，不应报告14路全部正常推进：move_playingcard_away/Randomized日志停在2026-10-19 14:12:45任务初始化；pick_dual_bottles/Clean日志停在14:16:25，含OIDN out of memory和vk::Queue::submit ErrorDeviceLost；pick_dual_bottles/Randomized截至04:06仍反复setup_demo异常Robot缺right_planner。存活不等于健康，三项0/0不能参与成功率，当前有效并发低于调度器running14。新的错误表明12GiB启动余量和60秒分配观察未避免所有运行期资源/初始化问题，不可宣称扩到7模型必然加速。后续需保存异常现场、定位卡住子进程与显存，按完整百条恢复并加强健康检查；不能靠改变RT/动作预算或拼接部分结果解决。本轮仅核查与记录，尚未停止/恢复三项，不声称已经修复。


2026-10-20T05:11:36.469368+00:00 Sidecar32k L20 place_burger_fries/demo_randomized：70/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue27/jobs/demo_randomized/place_burger_fries。独立运行，不与A100或20k样本混合。


2026-10-20T05:34:46.989222+00:00 Sidecar32k L20 move_stapler_pad/demo_clean：72/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu0/queue/jobs/demo_clean/move_stapler_pad。独立运行，不与A100或20k样本混合。


2026-10-20T05:38:48.689370+00:00 Sidecar32k L20 hanging_mug/demo_clean：33/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu0/queue/jobs/demo_clean/hanging_mug。独立运行，不与A100或20k样本混合。


2026-10-20T06:13:02.569369+00:00 Sidecar32k L20 place_fan/demo_clean：89/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_clean/place_fan。独立运行，不与A100或20k样本混合。


2026-10-20T07:29:40.045361+00:00 Sidecar32k L20 rotate_qrcode/demo_randomized：12/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_randomized/rotate_qrcode。独立运行，不与A100或20k样本混合。


### 2026-10-20T07:40:44.969554+00:00 修复32k无进展任务与启动共享模型吞吐A/B

用户指出不能仅等待异常，明确授权恢复并在GPU2/7试验共享模型多仿真是否更快。进一步读取pick_dual_bottles/Randomized原日志首个异常：right CuroboPlanner warmup中torch.OutOfMemoryError，CUDA free仅28.56MiB，随后Robot.reset反复缺right_planner；不是新出现的模型结构错误。Clean原日志OIDN out of memory及ErrorDeviceLost；原始日志（Randomized约789MB）保留，不删除、不原地截断。

仅SIGSTOP旧expanded_queue27控制器170490，核对进程树并停止三路异常outer/native/ffmpeg与GPU2模型170637及编译子进程，共15个明确目标。两路正常place_object_basket/Clean PID1388716与Randomized PID1624568原PID继续仿真，采用真实退出码+完整100验收后才回收冻结控制器，不打断正常episode。GPU2移除模型后约44GiB空余，避免模型与多sim叠加原问题。详细现场及进程清单recovery_stalled_20261020/cleanup.json。

新增recover_stalled.py（PID1728264），复用既有真实退出码/百条验收，接管expanded_queue27全部20条件并加shake_bottle/Randomized，21个key覆盖原100集合子集；原运行快照保留。五项完整百条重跑优先于六项原queued任务，最多每项新增两次基础设施恢复，不拼接、不按SR择优。正常两路原PID继续；新增3worker复用已验收GPU5 ports21650/21651和GPU6 port21661。新sim启动free>=16000MiB、每卡本恢复队列至多2sim，派发串行观察CUDA占用>=4GiB或120秒；16GiB余量不是运行期资源安全保证，后续需看峰值和外部作业。每10秒检查致命CUDA/OIDN/renderer/planner错误，30分钟client日志无活动则停止该attempt、保留现场并完整重试，避免永远标running。共享默认runner/官方源码/模型/训练/预算/RT均未修改。

watch_results.py本次输出内已更新：以恢复队列按task/mode覆盖原源记录，强制100唯一key；click_bell指定恢复仍单独覆盖源失败；GPU2模型停止后显示6模型。原失败部分不纳入最终SR，历史日志及history保留；failed=0仅说明现在恢复状态，不表示旧失败不存在。新监控PID1728299。新两路move_playingcard_away/Clean与Randomized父进程已启动，07:40前后Clean开始官方仿真初始化；尚未据父PID宣称恢复完整百条。

此前没有严格相同工作集的单仿真vs多仿真端到端吞吐A/B。原cache约2.28倍来自engine.generate计时，不含仿真/通信，不能用于证明共享模型并发加速。用户要求后新增benchmark_sim_concurrency.py/PID1732239，独立throughput_ab_20261020，不纳入正式成绩：同一GPU6已验收32k模型，独占ports21660/21662，同adjust_bottle/Clean、两份每份2条工作，seed group0及原RT/预算；serial依次在simGPU2/7执行，再parallel2同时在相同GPU位置执行，总8条pilot。记录墙钟/启动与首步/有效episode/实际step行数/三GPU峰值/accepted seeds。原生模型CUDA RNG未额外固定、共享机器负载与I/O可能变化，小样本不能声称确定的普遍加速或严格动作配对。

为使GPU6模型测速期间没有正式推理请求插入，临时SIGSTOP本次恢复控制器1728264（尚未派发port21661），已启动的两个恢复客户端和两个正常客户端继续，不SIGSTOP它们。hold记录benchmark_hold.json；测速脚本finally无论成功失败均SIGCONT恢复调度，phase最多1小时。模型未新增，GPU2/7只是仿真；其他节点42960/ManiSkill不操作。后续须验收本次恢复真实rollout及A/B结果，不能以已启动当完成。


2026-10-20T07:50:44.163266+00:00 恢复与测速实况：move_playingcard_away/Clean恢复已6条、Randomized已2条，均进入真实policy动作且未再见原致命错误；其余三项仍在优先恢复队列，不称五项全已完成。测速serial job0已完成首条，仍在第2条，尚无速度比。守护PID1735504监测暂停派发期间恢复客户端的致命错误及一小时整体限时，必要时SIGINT测速以执行finally释放；结果报告PID1740887持续写docs/ROBOTWIN_SIM_CONCURRENCY_BENCHMARK_20261020.md，结束自动入实验总记。后台共享机器负载/首次启动缓存及实际动作步数差异须保留，不能把小样本倍数当全量保证。


# 单模型单仿真与双仿真吞吐实测

状态：failed

32k固定权重、同一GPU6模型，独占两个策略端口；两份adjust_bottle/Clean任务，各2条native评测，仿真参数为GPU2/3（GPU7余量不足，在其子任务启动前改为GPU3，两个阶段一致）。先serial依次执行，再parallel2同时执行。与正式10000条成绩分开。

| phase | status | wall seconds | episodes/hour (pilot) | printed steps | rollout window seconds |
|---|---|---:|---:|---:|---:|
| serial | complete | 1296.3351739123464 | 11.108238278022437 | 489 | 490.58412903919816 |
| parallel2 | running | pending | pending | 0 | pending |

边界：每组仅4条episode；顺序serial→parallel2可能有共享盘/库/资产缓存预热偏差；其他卡后台负载变化；GPU参数控制CUDA选择，未据此声称已经逐进程审计所有Vulkan/OIDN实际物理设备。native模型CUDA RNG未额外固定，不能称动作逐位配对。比较同seed仍须核对实际step差异。墙钟含启动、权重校验、仿真初始化与收尾；rollout window从首次观察step起估计，不是细粒度模型耗时。共享模型predict锁串行，若有加速来自流水交叠，不是模型并行计算。

证据：outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/throughput_ab_20261020/state.json；不将失败/部分pilot纳入正式SR。


### 2026-10-20T08:14:28.669509+00:00 32k恢复调度正式接续、测速资源不足终止

共享模型推理锁串行，之前没有端到端并发加速对照。此次实际完成serial两子任务各2条adjust_bottle/Clean（总4次rollout，seed group0相同两条各重复一次）：job0 wall630.94s、job1 wall665.40s，合计1296.34s（21.61min，排除GPU7无client运行的纯容量等待）；打印policy step共489行，从首次step到各自结束合计490.58s。GPU7 free由约20GiB降至3.7GiB，未启动的第二slot曾改至GPU3并完成匹配serial子任务；之后GPU3 free亦降至约4.9GiB，parallel第二路未能按14GB余量启动。中断本次未完整parallel组、完整保留serial及partial，未获得speedup，不能说双仿真更快或更慢。原KeyboardInterrupt错误字符串为空，已将state状态明确标interrupted_insufficient_capacity及说明。未发现已结束serial运行的私有runtime残留子进程；新增大显存占用归属未确认，不操作未知进程。

不继续以测速占住正式恢复调度。暂停期间四个正常客户端始终继续，move_playingcard_away双域已实际产出；原source两客户端也仍在执行。恢复控制器先升级为四活跃job真实adoption与GPU2最多三sim/free20GiB门槛，发现当前两sim以外无安全16GB余量，仍会等待。进一步确认GPU6模型3549591已无任何standard_eval客户端引用（指定click_bell恢复早已完整结束，expanded_queue56源已无running），只停止该空闲模型及其编译子进程，释放约24GiB；不停止GPU6原有其他仿真。记录idle_gpu6_model_stop.json。GPU2与GPU6累计移除两份冗余/无可用仿真容量的模型，而非一味增加模型。

最新权威recover_stalled_capacity.py/PID1762422，复用同一21项state，保存resume_capacity_snapshot.json，正常四个outer PID全部保留；其native父调度器170490与1728264继续仅暂停以保留真实僵尸退出码，只有各自所有adopted子任务验收结束才kill父调度器。中间无子进程的等待控制器1759421已停止，不重复派发。7worker固定模型端口：(7,21670)/(7,21671)接原basket双域，(5,21650)/(5,21651)接已恢复card双域，新增(7,21672)/(5,21652)/(1,21610)服务dual_bottles双域和shake/Randomized；源GPU1队列21610对应worker已无queued/running，确认空闲后复用。GPU6专做仿真，新增任务free>=16000MiB且本恢复队列最多3sim；GPU2 free20GiB/max3，其他卡free16000MiB/max2；保留分配观察/致命错误与30分钟无日志活动检测/最多两次完整恢复。瞬时容量仍不保证未知作业不增长，异常需排除并保留。

监控PID1763251读取恢复覆盖与100唯一key；实际模型5份，队列最大15路，不能把上限当真实活跃。08:13 dual_bottles/Clean已启动于simGPU6，另两新worker逐次等待admission锁，不称已完成恢复验收。测速4条独立，不混正式SR；AB、原始所有失败日志与权重均保留。后续验收三项真正step/episode及无fatal，按实际完整任务推进；没有因资源问题降低RT、动作预算、缓存参数或改变权重。


2026-10-20T08:18:14.169764+00:00 现场验收：五项异常条件均已重新启动，原Card双域有实际step/episode（Clean26条、Randomized7条）；dual双域及shake/Randomized已启动于simGPU6，仍处初始化，未宣称真实百条验收。中央process.json现指向1762422并明确两暂停父调度器禁止SIGCONT；100唯一key覆盖检查通过。新控制器errors为空，旧正常四个outer PID继续。


2026-10-20T08:21:53.832643+00:00 Sidecar32k L20 place_a2b_left/demo_clean：94/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu0/queue/jobs/demo_clean/place_a2b_left。独立运行，不与A100或20k样本混合。


### 2026-10-20 08:23 UTC：纠正并发测速分母与吞吐外推口径

8.18分钟（490.584s）为两个各2条子任务从首次观察step至进程退出的窗口累计，共4条，不是一条rollout的完整端到端耗时；遗漏各子任务首条之前的启动/专家筛选等，又包含两条之间重置及退出收尾。粗摊122.646秒/条、理想8路约234.82条/小时，仅adjust_bottle/Clean微样本外推。含每2条重新启动/验权/初始化的总1296.335s外推8路88.87条/小时，但正式100条启动成本会被摊薄，这也不是长期单路吞吐。不能据8.18分钟/条推8路64条/小时。

当前08:22:54有效7146条；自07:30:40的6978条增加168条/0.8704h，即193.01条/h，含恢复/调度变化，非稳定并发A/B；自06:50为176.31条/h、自02:04为164.33条/h。任务与域覆盖不同（单路测速仅瓶子Clean、当前混合长Randomized），不能拿235理论对193混合实际判胜负。结论仍是没有有效同条件端到端并发加速比，不能声称8组单模型单仿真明显更慢。


### 2026-10-20 08:54 UTC：当前模型/仿真并发与吞吐解释

只读按/proc实际非僵尸入口核验：常驻32k模型5份，GPU0/1/4/5/7，每份3个standard_eval客户端，共15路。GPU6仿真参数分配5路，GPU3分配4路，GPU2/0/4各2路；快照GPU3/6利用率100%/99%，GPU6余1453MiB，GPU1瞬时0%。单点整卡利用率同时受仿真、模型及未知作业影响，不能直接当模型推理利用率，也不能据此量化哪一环耗时占比。GPU6容量偏紧与仿真分配集中是明确现场事实，需要持续错误监视，不能把15进程等同15路独立GPU算力。

serve_parallel.py对完整predict持同一模型锁，每模型请求串行；各端口独立动作状态。复用常驻模型不等于重新加载模型，任务runner启动的权重校验/环境检查与仿真初始化应和推理区分。并行可交叠不同仿真的物理/渲染/规划与推理，从而填补模型等待间隙；若推理已饱和则新增请求排队，若仿真GPU已满则增加仿真争用，不能保证加进程提速。没有证据表明复用次数本身造成模型越用越慢；尚未建立preprocess/queue/generate/RT阶段计时，不能将现象归因到未验证的重编译或cache增长。

应区分相同5模型下5sim对15sim，和8模型8sim对5模型15sim两种比较：后者少了3份可独立推理引擎、多了7sim，资源分配不同，并不必然较快。当前小样本serial4条与混合任务整体吞吐不可作此因果对照。此轮仅记录部署实况与代码所确定的机制，不声称获得并发加速实测。


2026-10-20T09:02:11.659470+00:00 Sidecar32k L20 place_fan/demo_randomized：42/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_randomized/place_fan。独立运行，不与A100或20k样本混合。


2026-10-20T09:11:15.297001+00:00 Sidecar32k L20 place_phone_stand/demo_clean：83/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_clean/place_phone_stand。独立运行，不与A100或20k样本混合。


2026-10-20T09:30:24.211641+00:00 Sidecar32k L20 place_object_basket/demo_clean：88/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue27/jobs/demo_clean/place_object_basket。独立运行，不与A100或20k样本混合。


2026-10-20T10:00:37.222719+00:00 Sidecar32k L20 move_playingcard_away/demo_clean：99/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/recovery_stalled_20261020/jobs/demo_clean/move_playingcard_away/attempt1。独立运行，不与A100或20k样本混合。


2026-10-20T11:12:08.454560+00:00 Sidecar32k L20 shake_bottle_horizontally/demo_clean：100/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_clean/shake_bottle_horizontally。独立运行，不与A100或20k样本混合。


2026-10-20T12:21:39.455907+00:00 Sidecar32k L20 move_stapler_pad/demo_randomized：16/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu0/queue/jobs/demo_randomized/move_stapler_pad。独立运行，不与A100或20k样本混合。


### 2026-10-20 12:45 UTC：32k阶段进度与恢复验收

有效7973/10000条，72个完整百条条件（Clean40、Randomized32），32个任务双域均完整。相同完整任务Clean89.70vs官方91.08（-1.38pp），Randomized45.12vs47.81（-2.69pp），不是全50最终均值。五项恢复：card/Clean已99/100完整；card/Randomized46/84、dual_bottles/Clean83/91、dual_bottles/Randomized14/23（第2次恢复）、shake/Randomized70/76仍为部分观察，不计完整SR。当前15项running。

pick_dual_bottles/Randomized首次新恢复在2026-10-20T10:28:16.905799+00:00被watchdog检测OIDN Error: out of memory，完整保留attempt1并停止其进程树；按既定有界恢复策略从头开始attempt2，目前23条。不得将前一次部分与当前拼接，也不能把本次低分当策略失败。该事件表明移除空闲模型、16GB余量与分阶段admission仍未完全消除运行期渲染显存风险；后续更改须继续考虑场景峰值、共卡多sim及未归属作业增长，不能声称基础设施问题已彻底解决。最新状态controller errors为空仅表示没有未处理线程错误，不代表历史无错误。本轮只读核查和记录，未变更任务、模型或协议。


2026-10-20T13:03:55.333366+00:00 Sidecar32k L20 pick_dual_bottles/demo_clean：92/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/recovery_stalled_20261020/jobs/demo_clean/pick_dual_bottles/attempt1。独立运行，不与A100或20k样本混合。


2026-10-20T13:42:10.461374+00:00 Sidecar32k L20 move_playingcard_away/demo_randomized：53/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/recovery_stalled_20261020/jobs/demo_randomized/move_playingcard_away/attempt1。独立运行，不与A100或20k样本混合。


2026-10-20T13:56:16.217370+00:00 Sidecar32k L20 shake_bottle/demo_randomized：91/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/recovery_stalled_20261020/jobs/demo_randomized/shake_bottle/attempt1。独立运行，不与A100或20k样本混合。


### 2026-10-20 UTC：Sidecar 训练/评估 EEF 的目标与实际位姿语义核查

用户要求核对 IK/控制目标 EEF 与执行后实际 EEF 是否混用。已沿实际训练 manifest、reader、原采集 getter、当前官方 sim getter 和 EE action 执行链只读核查：训练 proprio(t) 与评估 proprio(t) 均来自实际 link global_pose 转换的 endpose；训练 action 标签为同一实际 endpose 序列 t+1…t+32，不是专家规划目标。当前 client 解码预测后将其作为绝对 EE 控制目标，重新规划并执行。因此未发现状态侧目标/实际语义混用，但专家 achieved-pose 标签被再次作为 command 执行的跟踪误差尚未测量，不能保证可忽略、不能归因为当前 SR 下降的原因。

特别区分 joint_action/get_arm_jointState 的 drive_target 来源与本次 endpose reader；不能因名称包含 State 就认定 qpos 是实测，也不能把旧 FK 分支当本次训练输入。训练和当前 Sidecar client 的 wxyz/column-Rot6D、左右臂20D→80D、自有stats与官方 endpose/TCP 12cm 区别均已核对代码。数值往返检查不替代物理回放。下一步应做独立专家 EEF 标签经原 take_action('ee') 的闭环回放，记录目标 vs 执行后实际 mm/degree、规划失败和接触阶段误差，并核对采样节奏；任何改标签或控制协议须另行确认。详见 docs/ROBOTWIN_EEF_TRAIN_EVAL_ALIGNMENT_20261020.md。本轮未修改训练/数据/模型/正式评估，未运行新的 GPU 物理实验。


### 2026-10-20T14:13:05.983665+00:00：32k剩余任务全局均衡接管已验收

用户明确授权将未启动任务均匀分配。原独立队列gpu0剩7项、gpu4剩2项、恢复队列剩1项，其他队列完成后不能跨队列领取，导致尾部等待。已新增输出内独立global coordinator，接管原100唯一条件：75complete、15running、10queued；15个原outer PID保留、端口唯一、16线程、errors空。新控制器2037013，监控2037485；实时汇总改以global_remaining_20261020/parallel.json覆盖旧源。15端口全忙，首个结束者即领取全局待跑条件，不等其原分区；不宣称新增派发已发生。

源3490333/3490334/3490335/1762422仅暂停派发，仿真继续。不得SIGCONT源调度器；新控制器收齐每个实际父PID所有adopted子任务真实退出码/百条验收后才kill父，保留此前170490等的退出码生命周期。源快照保留，禁止用旧冻结状态重复计分。容量admission现统计全100条件的sim占用，沿用内存门槛/串行分配观察/有界异常恢复，不提高并发、不改模型/训练/RT/预算/seed。

日志步数证据：hanging_mug/Randomized成功4条均323步，失败56条全900步；open_microwave/Randomized成功3条均460.7步，失败89条全1500步；adjust_bottle/Clean成功97条均119.8步。后期长失败轨迹确实增加工作量，但不是已测得模型每次推理变慢或wall-time倍数。均衡后ETA需看实际吞吐，不能保证此前10–15h估计；场景显存峰值/外部占用/原历史异常重试影响仍在。详见ROBOTWIN_SIDECAR32K_GLOBAL_REBALANCE_20261020.md及global目录handoff_acceptance/slow_episode_step_analysis。


### 2026-10-20T14:23:10.730567+00:00：32k与官方baseline正常复测5任务双域成功步数对照

用户要求比较之前旧环境异常任务在新官方环境正常复测的baseline与当前32k。共同5任务10条件均完成100条，两模型各1000条；baseline沿用9项官方manifest分片+dual_shoes/Clean传统native92的固定规则，不择优换protocol。全部90分片逐episode seed/success核对通过；Sidecar各100条成功数/accepted seed顺序及完整终态核对通过。

SR baseline70.7%→32k69.9%（-0.8pp）；Clean92.6→92.0（-0.6pp），Randomized48.8→47.8（-1.0pp）。各条件成功均步再等权183.95→185.88（+1.94步/+1.05%），7/10项差约±2.4%内。click_bell/Randomized89.75→110.28（+20.53步/+22.87%，SR77→71）；place_dual_shoes/Randomized228.69→240.80（+12.11步/+5.30%，SR16→10）；place_shoe/Randomized SR49→66，成功均步191.33→191.05。这个子集不支持成功动作步数普遍数倍增加；不外推全部50任务。

步骤为原生take_action_cnt而非去噪步/推理请求/scene.step或秒数。4个视频独立核查frames=动作步+1，额外1帧为官方成功终态写入，不把它计动作。成功样本集合不相同，9项baseline manifest vs Sidecar native、accepted seed交集79–100，不能当严格同场景/指令/RNG配对或纯结构效率因果。证据和同seed双方成功子集见ROBOTWIN_BASELINE32K_SUCCESS_STEPS_20261020.md及docs/evidence/robotwin_baseline32k_success_steps_20261020。未改模型/训练/协议/当前进程；后续完成其余条件并按共同集合更新，如做因果对照需固定相同manifest。


### 2026-10-20T14:39:47.405043+00:00：复现baseline10740十任务接续评测已后台排队

用户明确授权：当前32k优先收尾，全部完成且有显存后测outputs/1020-baseline-reproduced/checkpoint_step_10740.safetensors，优先Sidecar比官方低≥10pp的约10任务。当前全局15worker继续共享剩余队列；新增独立one-shot supervisor PID2060359/PPID1，status=waiting_for_sidecar32k_completion，每30秒心跳，未启动baseline GPU。只有100唯一条件全百条/无errors/run.json真实退出码验收后触发，检查本机host+boot。

权重只读hardlink固定在outputs/robotwin_eval/baseline10740_after_sidecar32k_20261020/checkpoint_step_10740，24813767464 bytes，SHAbd1ab96a0ae5697dff3b8f2cedf5519e6be66d6e8141fd2cd245a20238ea2187；config/stats/tokenizer独立复制。结构2089 BF16 tensors无Sidecar；自有20D EEF/旋转identity/夹爪stats通过，不套用Sidecar stats。最终按完整32k每task最差官方差距选10个不同任务并固定selection，优先≤−10pp，不足则明示补位；两域各100共2000，先测入选负差条件，再另一域对照，不按baseline结果换任务。

前序完成后仅释放已固定PID/start_ticks且无活跃客户端引用的本次5个32k模型；实时容量最多3模型9sim，全局队列、模型free33GB/sim16GB且每GPU最多2sim/逐次admission，不抢未知进程。两个真实fixture匹配cache进行baseline compile数值门槛，未过则同cache eager；新权重GPU验收和闭环尚未发生，不能声称已通过。官方RT/seed筛选/预算/成功判据/视频保留，异常最多两次完整恢复、不拼结果。

重要未解决因果约束：baseline配置5epochs/bs64acc1，Sidecar计划10epochs/bs32acc2且测中途32k；自有stats不同、baseline完整日志/实际manifest未提供。对照不能单独隔离模块收益，也不能认定官方数字过高；选取负差任务也不代表无偏全50平均。CPU门槛/排序/语法及实际waiting状态已验收。详细候选、设置和风险见ROBOTWIN_REPRODUCED_BASELINE10740_HANDOVER_20261020.md，后续结果自动comparison.json/md。


2026-10-20T14:40:21.509378+00:00 Sidecar32k L20 shake_bottle_horizontally/demo_randomized：90/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_randomized/shake_bottle_horizontally。独立运行，不与A100或20k样本混合。


2026-10-20T14:51:26.669367+00:00 Sidecar32k L20 place_a2b_left/demo_randomized：21/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu0/queue/jobs/demo_randomized/place_a2b_left。独立运行，不与A100或20k样本混合。


2026-10-20T15:25:41.549365+00:00 Sidecar32k L20 place_phone_stand/demo_randomized：14/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_randomized/place_phone_stand。独立运行，不与A100或20k样本混合。


2026-10-20T16:38:12.330510+00:00 Sidecar32k L20 open_microwave/demo_randomized：3/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu4/queue/jobs/demo_randomized/open_microwave。独立运行，不与A100或20k样本混合。


2026-10-20T16:44:14.677361+00:00 Sidecar32k L20 pick_dual_bottles/demo_randomized：72/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/recovery_stalled_20261020/jobs/demo_randomized/pick_dual_bottles/attempt2。独立运行，不与A100或20k样本混合。


2026-10-20T17:06:23.827941+00:00 Sidecar32k L20 stack_bowls_three/demo_clean：76/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_clean/stack_bowls_three。独立运行，不与A100或20k样本混合。


2026-10-20T17:56:45.657364+00:00 Sidecar32k L20 place_object_scale/demo_clean：94/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/global_remaining_20261020/jobs/demo_clean/place_object_scale/attempt1。独立运行，不与A100或20k样本混合。


2026-10-20T18:16:54.413371+00:00 Sidecar32k L20 stack_bowls_three/demo_randomized：73/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu1/queue/jobs/demo_randomized/stack_bowls_three。独立运行，不与A100或20k样本混合。


2026-10-20T18:59:14.926644+00:00 Sidecar32k L20 place_can_basket/demo_clean：75/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu0/queue/jobs/demo_clean/place_can_basket。独立运行，不与A100或20k样本混合。


2026-10-20T19:00:15.447286+00:00 Sidecar32k L20 turn_switch/demo_clean：47/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/recovery_stalled_20261020/jobs/demo_clean/turn_switch/attempt1。独立运行，不与A100或20k样本混合。


2026-10-20T19:14:21.489375+00:00 Sidecar32k L20 stack_blocks_three/demo_clean：93/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/recovery_stalled_20261020/jobs/demo_clean/stack_blocks_three/attempt1。独立运行，不与A100或20k样本混合。


2026-10-20T19:58:42.145375+00:00 Sidecar32k L20 place_object_basket/demo_randomized：30/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expanded_queue27/jobs/demo_randomized/place_object_basket。独立运行，不与A100或20k样本混合。


2026-10-20T20:25:54.185380+00:00 Sidecar32k L20 place_object_scale/demo_randomized：37/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/global_remaining_20261020/jobs/demo_randomized/place_object_scale/attempt1。独立运行，不与A100或20k样本混合。


2026-10-20T20:26:54.661374+00:00 Sidecar32k L20 hanging_mug/demo_randomized：8/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/gpu0/queue/jobs/demo_randomized/hanging_mug。独立运行，不与A100或20k样本混合。


2026-10-20T23:35:07.717369+00:00 Sidecar32k L20 stack_blocks_two/demo_clean：100/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/global_remaining_20261020/jobs/demo_clean/stack_blocks_two/attempt1。独立运行，不与A100或20k样本混合。


2026-10-21T00:07:20.029375+00:00 Sidecar32k L20 turn_switch/demo_randomized：34/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/global_remaining_20261020/jobs/demo_randomized/turn_switch/attempt1。独立运行，不与A100或20k样本混合。


2026-10-21T00:45:33.649358+00:00 Sidecar32k L20 stack_blocks_two/demo_randomized：82/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/global_remaining_20261020/jobs/demo_randomized/stack_blocks_two/attempt1。独立运行，不与A100或20k样本混合。


2026-10-21T02:24:11.041357+00:00 Sidecar32k L20 stack_bowls_two/demo_clean：93/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/global_remaining_20261020/jobs/demo_clean/stack_bowls_two/attempt1。独立运行，不与A100或20k样本混合。


### 2026-10-21T03:15:36.919292+00:00：baseline10740改为提前共存启动，保留32k收尾

用户明确授权有空闲资源即提前baseline，不等待全32k。停止仅等待的2060359，保留原script/spec历史；新控制器2448303，旧GPU1模型3486828经全3worker finished、无running端口、无活跃引用及PID/start_ticks验证后单独回收，释放后约39037MiB空闲。模型0 PID2448335在GPU1加载，计划3端口/3sim；此时尚未通过GPU数值门槛/闭环，禁止称已完成rollout。其余7个32k外层PID保留，93条件完整约9827条，当前global调度2037013不变。

共存阶段baseline sim限GPU3/4/7、每GPU最多1sim/free16GB，并合并Sidecar活动占用；GPU3有未知计算负载，不称空卡、不动未知进程。后台scaler只释放全部worker finished且无活跃引用的固定32k模型，free33GB后最多增至3baseline模型9worker；ready后加入相同全局队列。先测当前完整双域任务中差距≤−10pp的10任务最差域，再另一域，共2000；selection在baseline成绩前固定，替代原全32k结束后排名规则。

两个真实fixture匹配cache的compile门槛及fallback原样保留，权重、统计、官方RT/预算/判据/视频不变。当前仅CPU入口/选取检查通过；后续关注加载/实际数值/首个原生动作及场景峰值/未知显存增长，不能把已启动当验收通过。Sidecar监控2448546采用存活模型数，避免GPU1回收后仍硬写5。详见ROBOTWIN_BASELINE10740_EARLY_COEXISTENCE_20261021.md；原先5epochs/不同stats的因果限制仍有效。


### 2026-10-21T03:37:20.290378+00:00：原版compile差异与Sidecar缓存影响核查、baseline正式协议纠正

baseline可运行原版完整compile（693kernels，cached warm约.26–.27s vs eager .37–.38s）；先前fallback由本地额外atol=.005/rtol=.01门槛触发，非官方要求/非编译失败。两fixture各3/640标量越界，均近零Rot6D；转物理目标最大3.699/3.404mm、旋转.417/.530deg。PyTorch2.7.1源码说明BF16融合省略中间downcast/upcast可改变舍入，机制符合，但未逐层唯一定位；不可说compile改变动作就等于实现错误或量级无害。

32k已保存动作重算：FFN+cache vs同cache eager在五副本同两个fixture均误差0；cache on/off则最大位置2.980/2.707mm、旋转.450/.401deg。cache均跳过6/10完整forward，仍复用含Sidecar作用的速度但未重算这些步的关系读写。官方cache代码同时核对video/action余弦且逐字相同；不能由FFN匹配证明cache质量无损，尚无同manifest闭环cache A/B，SR影响未解决。

源码校对：原版joint_self_attn、compile_options、dit_cache、model_loader、默认deploy逐字一致；Base.generate/normalize_deploy_proprio/inactive-map AST一致。engine/policy只增加相关EA守卫/字段。当前wxyz适配须跟训练合同，不可盲改默认xyzw。并行、accepted seed/指令、训练5epochs/自有stats等仍是限制，不能纯归因模块。

按用户要求原版加速，旧baseline eager pilot仅停止本身3sim与模型/控制器，32k不动；原文件全保留，hammer有1成功episode，其余无终态，无完整百条，不混新成绩。正式新根baseline10740_officialcompile_early_20261021，控制器2457886，十任务名单/权重不变、原完整compile+cache，不再因额外eager阈值静默关闭compile；真实推理异常/非有限值仍拦截。新模型重新加载，启动四组eager/compiled×cache on/off输入诊断。详见ROBOTWIN_INFERENCE_PROTOCOL_AUDIT_20261021.md；未开启新增闭环cache消融，后续需固定manifest验证，现有32k保持原协议收尾。


2026-10-21T04:07:19.070060+00:00 Sidecar32k L20 put_object_cabinet/demo_clean：45/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/global_remaining_20261020/jobs/demo_clean/put_object_cabinet/attempt1。独立运行，不与A100或20k样本混合。


2026-10-21T04:17:23.821381+00:00 Sidecar32k L20 put_bottles_dustbin/demo_clean：80/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/recovery_stalled_20261020/jobs/demo_clean/put_bottles_dustbin/attempt1。独立运行，不与A100或20k样本混合。


2026-10-21T05:28:49.846453+00:00 Sidecar32k L20 put_bottles_dustbin/demo_randomized：37/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/recovery_stalled_20261020/jobs/demo_randomized/put_bottles_dustbin/attempt1。独立运行，不与A100或20k样本混合。


2026-10-21T05:53:58.997370+00:00 Sidecar32k L20 place_can_basket/demo_randomized：38/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/global_remaining_20261020/jobs/demo_randomized/place_can_basket/attempt1。独立运行，不与A100或20k样本混合。


2026-10-21T06:14:07.033377+00:00 Sidecar32k L20 stack_bowls_two/demo_randomized：76/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/global_remaining_20261020/jobs/demo_randomized/stack_bowls_two/attempt1。独立运行，不与A100或20k样本混合。


## 2026-10-21 基于实际空余容量扩展 baseline 仿真

用户明确要求利用空闲GPU尽快产出分析结果。现场可用显存（MiB）GPU0/1/2/3/4/5/6/7约20772/14501/37656/39029/14891/20815/31254/15817；已部署3个baseline，但仅3个sim运行，根因旧共存规则只放行GPU3/4/7且每卡1sim。不是已有吞吐A/B证明一对一更优。

复用已验收的rebalance_remaining.py进程接管实现，新增输出内expand_capacity.py；冻结旧父2457886，仅暂停派发，2461744/2461942/2462124原sim与全部3个模型保持存活。新控制器2550008沿用canonical parallel.json，3模型×3独立端口、共9worker、全部GPU按剩余显存>=16000MiB且与Sidecar合计每卡最多2sim入场，串行观察分配，100条完整验收与至多两次基础设施重试不变。必须保存所有adopted子进程真实exitcode后才结束冻结旧父，禁止SIGCONT旧父造成重复派发。固定20条件、compile/cache/权重/stats/官方RT/预算/筛seed/成功判据/视频不变；原结果与原代码保留。baseline10740_current.json已指向新控制器。

此前测速只有serial阶段4条/1296.34s=11.11条/小时，parallel2在第二仿真启动前因容量变化中断，不能报告加速倍数。新增单独测速监督2551130，复用闲置Sidecar32k GPU0服务3486827的独立端口21600/21602；等待正式baseline9sim入场后，选择两张free>=16000MiB的固定sim卡，两组adjust_bottle/Clean×2条，serial再parallel2，两阶段同卡/同服务/同协议。记录启动/首动作/首episode/完整wall/步数/accepted seeds/显存峰值/聚合条数每小时，所有测速结果从正式SR排除。小样本、背景负载/随机指令及CUDA RNG交织仍是限制；该测速对象是Sidecar而非vanilla compile baseline，不以其倍数直接外推baseline。正式队列优先，测速若容量不足/异常明确报告而不抢占已有作业。

阶段启动与输出目录见capacity_expansion_20261021/{source_snapshot.json,controller.log,process.json}及Sidecar根throughput_ab_20261021/state.json。下一步确认9路真实动作与完成条数增长，待完整条件报告官方差距；待测速两阶段均完成才得出一对多的吞吐结论。


2026-10-21 07:22 UTC 容量细化：首轮扩至7路后，GPU2/6虽仍有大量显存，但通用每卡2sim上限再次限制入场。因此恢复已有Sidecar全局调度已使用的GPU2/6最多3sim规则（GPU2 free>=20480MiB、GPU6>=16000MiB，其余卡最多2sim），合并统计两批活动sim。新增expand_capacity_balanced.py，控制器2553273，canonical状态与指针已切换；冻结2550008，保留其4路及2457886的3路，按每个实际父进程的adopted子任务分别收齐真实退出码后再结束父进程，禁止恢复旧调度器。07:21已8路，hanging_mug/stamp_seal已出现真实动作；第9路仍按串行容量观察入场。原3模型无重新加载，所有原仿真PID保持。该调整不构成单/多仿真吞吐提升证据，仍等待独立测速。


2026-10-21T07:23:32.541384+00:00 Sidecar32k L20 stack_blocks_three/demo_randomized：49/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/recovery_stalled_20261020/jobs/demo_randomized/stack_blocks_three/attempt1。独立运行，不与A100或20k样本混合。

2026-10-21 07:23 UTC：baseline正式9路全部已派发（3模型各3端口），6路已有真实动作，其余初始化；errors为空，原3路PID保持。独立测速state=running/serial，首个adjust_bottle/Clean两条job PID2554666、sim GPU2，随后同两张固定sim卡测parallel2；尚无加速倍数。证据capacity_balanced_20261021/admission_acceptance.json。


## 2026-10-21T07:35:39.325448+00:00：baseline后续任务关闭视频保存

用户明确要求设置不保存视频以尝试加速。当前无完整100条条件；日志最接近hammer Randomized43/83、skillet Randomized33/73、microwave Clean31/42。仅在本次baseline10740根policy_config.yml显式添加eval_video_log:false，原配置备份保留。wrapper增加可选布尔配置：只在新任务私有runtime内复制两域task config并改该视频标志，其余配置仍链接官方原件；源RoboTwin/DiT4DiT/其他未设标志实验不变。保存源/私有config hashes到runtime/video_logging_override.json，run.json原有policy snapshot及wrapper hash继续记录。

验证：两域解析结果均只差eval_video_log、源文件字节不变、默认无flag仍保留原task_config链接，临时runtime检查通过。当前9路已加载配置，继续保存视频至各自结束，不重启、不删除已生成视频；随后新条件及异常完整重试使用关闭视频配置。视频标志移除逐帧raw RGB pipe/FFmpeg编码/文件I/O，但模型观测的相机渲染及物理步进仍须保留；未实测墙钟加速比例，不能承诺显著提速或直接据不同任务用时作因果比较。现有单/双sim测速仍按原有保存视频协议，两阶段不混改。下一步检查第一项新启动日志private eval_video_log=False及私有JSON、无ffmpeg进程/新mp4；如需量化收益应同任务同manifest单独video on/off对照。

证据：本次根video_logging_change.json、policy_config_before_no_video.yml、eval_policy_wrapper_before_no_video.py。


## 2026-10-21T07:53:53.380899+00:00：Sidecar32k接近完整结果，仍余cabinet Randomized

99/100条件完整且各100条验收，0失败/errors；put_object_cabinet/demo_randomized当前11/78，尚余22条。Clean50条件全部完成，4383/5000=87.66%，官方相同50条件89.36%，差-1.70pp。已完成49个Randomized为2252/4900=45.96%，官方相同49条件49.37%，差-3.41pp；不是最终Randomized全50平均。99完整条件等权67.02%，不能直接对比官方全100的69.04%。

按尚余22条全失败/全成功计算，若本次native继续完成而不重启此前样本，最终100条件整体范围66.46%–66.68%，对官方69.04%低2.36–2.58pp；这是条件边界，不是已完成最终成绩。最后一项官方17%，当前阶段11/78=14.10%。继续保留当前32k进程与原视频/协议，不因低成绩更换样本。后续待最后100条真实退出码/完整验收再生成正式全50双域报告。证据outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/near_final_summary_20261021.json。


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


2026-10-21T12:05:24.187248+00:00 Sidecar32k L20 put_object_cabinet/demo_randomized：13/100；同新环境/FFN compile+cache，原RT/预算；源：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/global_remaining_20261020/jobs/demo_randomized/put_object_cabinet/attempt1。独立运行，不与A100或20k样本混合。


## 2026-10-21T13:21:24.388375+00:00：Sidecar32k全10000条完成；baseline四模型12路已接入

32k100/100条件、6648/10000=66.48%，官方69.04%，差-2.56pp；Clean4383/5000=87.66% vs89.36%（-1.70pp），Randomized2265/5000=45.30% vs48.72%（-3.42pp）。最后put_object_cabinet Randomized13/100（官方17），完整run.json全部逐项验证，无最终errors。解决此前未完成全50双域平均的事项；固定manifest cache A/B、EEF物理跟踪及训练预算/统计因果限制仍未解决。详见docs/ROBOTWIN_SIDECAR32K_FINAL_RESULTS_20261021.md及final_results JSON/CSV。

baseline第4模型已09:46:15 UTC验证并接入，新控制器2648154，当前4模型/12路真实推进，20条件全部启动。13:20快照8/20完整、1266/2000条，errors空；完整条件SR：hammer Randomized51、microwave Clean72、skillet Randomized48、stamp Clean88、alarm Randomized67、burger_fries Randomized83、hammer Clean100、alarm Clean100。其余12条件日志更新，不以部分分母替代100条终态。新增条件关闭视频保留物理/渲染/评判，旧运行任务未重启。继续完成baseline并对照官方及32k；无新增模型/训练/数据逻辑修改。


## 2026-10-22T03:47:56.942925+00:00：三组checkpoint组会汇总已冻结并验证

用户要求汇总32k、baseline10740及另一台机器42960，生成docs/ROBOTWIN_THREE_EXPERIMENTS_GROUP_MEETING_20261022.md与group_meeting_three_ckpts_20261022/RoboTwin_Three_Experiments_20261022.xlsx（9工作表，含全50任务双域、官方差值、共同交集、进度、口径来源），以及CSV/summary.json/source SHA256。136份原生run.json逐项核对exit0、complete、100条、无errors及成功数一致。官方100项与封存issue46 scores逐项相同，重新浏览原官方发布表。

当前Baseline19/20完整（最后microwave Randomized2/84不进入SR）、32k100/100、42960 17/100完整。各自完整范围综合：Baseline71.1053% vs官方同19条件73.1579%（-2.0526pp）；32k66.48% vs69.04%（-2.56pp）；42960 79.8824% vs同17条件80.4706%（-0.5882pp）。这些不同覆盖均值不能直接排名，综合按已完成条件episode等权，不在两个域完成数量不同情况下简单平均两域。

共同4任务（adjust_bottle、beat_block_hammer、click_alarmclock、handover_block）双域共8条件：官方75.875%，Baseline71.625%，32k70.125%，42960 75.75%。共同Randomized官方58.75% vs45.50/43.75/53.00%，42960该域仍低5.75pp，综合接近源于Clean更高，不能只凭综合称Randomized达标。42960共同整体高32k5.625pp、高Baseline4.125pp，只限交集；训练预算/stats/任务选取偏差及跨机非逐例seed/指令/RNG配对限制仍在，不能单独归因Sidecar结构。

Excel ZIP/XML结构、所有sheet行数及汇总分子分母运算检查通过，率和差值保留数值类型，展示四舍五入至2位；未完成留空，不填0，不拼接实验。全程只读原任务/原结果，未改变评测进程/模型/训练/数据。下一步继续原队列，待baseline最后100条和42960更多完整项后按同规则更新新快照，保留本次组会快照。


## 2026-10-22T08:59:45.037497+00:00：L20光追与硬件要求只读核查

L20真实Vulkan扩展确认acceleration_structure/ray_tracing_pipeline，RoboTwin原setup_scene使用rt/32spp/depth8/OIDN，源码SHA匹配32k记录、当前git tracked diff空；既有RT smoke图像有限值。官方只列能力/驱动要求，未要求必须4090；L20当前满足此栈RT运行能力。这个问题已解决到能力与配置层，不等于与官方硬件逐像素/SR等价；资产历史snapshot、同manifest跨卡RGB/动作/闭环、并发RNG及Sidecar cache闭环影响仍未解决。记录docs/ROBOTWIN_L20_RENDER_AND_PROTOCOL_AUDIT_20261022.md；不改当前进程/源码/训练/数据。下一步若需要硬件因果应固定官方权重与完整manifest跨卡对照，不能凭L20品牌或均值直接归因。


### 2026-10-22T09:02:25.279097+00:00：资产表述纠正

用户指出当前已重新下载官方代码/资产并对齐环境。复核官方TianxingChen/RoboTwin2.0固定revision3dc3b798668feb99ac61cc9086d84cbcc3d79186，objects.zip、embodiments.zip、background_texture.zip三份已有验收manifest的SHA256/size均与官方LFS清单逐项相同，且有解包逐文件清单。当前没有资产错误证据。上一条把“缺官方发布成绩当次单独版本快照”列成实际资产疑点措辞过宽，应改为历史复现信息边界，不能用作当前低SR解释或要求重新下载；核查文档已更正。此次比对已有验收记录，并非重新全量hash14GB资产；未改资产/环境/评测进程。


## 2026-10-22T14:17:02.241914+00:00：baseline10740十任务双域全部完成

20/20条件、2000/2000条，全部原生run.json/真实退出码0核验、errors空。最后open_microwave Randomized3/100，官方7。Clean891/1000=89.10% vs同任务官方87.90%（+1.20pp）；Randomized463/1000=46.30% vs51.80%（-5.50pp）；综合1354/2000=67.70% vs69.85%（-2.15pp）。完整逐任务与同范围32k见docs/ROBOTWIN_BASELINE10740_FINAL_RESULTS_20261022.md及输出final_results_20261022.json。组会03:48旧快照保留，不把不同快照/不同覆盖均值混用。此十任务按32k负差预选、训练预算/stats不同，非全50或纯模块消融。下一步如需组会最新三组对照，以本最终baseline和42960新快照另生成版本。


### 2026-10-22T14:23:46.087989+00:00：4350数据集原始图像尺寸与OpenWAM训练输入核对

只读抽查用户数据目录raw/aloha的adjust_bottle、blocks_ranking_size、open_microwave和raw/piper的blocks_ranking_size、open_microwave，各episode0三训练相机首尾JPEG帧，另含Aloha front_camera，共36个JPEG头尺寸均H240×W320。没有据抽样声称4350轨迹逐帧重新验收。当前Sidecar manifest指向raw/aloha，LiteRoboTwinSource继承原RoboTwinDataset读原始JPEG，显式height384/width320，与保存训练config一致；三视角分别resize至H256×W320、H128×W160、H128×W160后拼成H384×W320。目录DATASET_SUMMARY/REAL_LOADER_CHECK中旧验收[3,224,672]属于另一个processed数据loader的横拼输出，不是这次OpenWAM训练输入。此次未修改数据/配置/训练，后续以原始分辨率与模型输入尺寸分层表述，不能混为一谈。


### 2026-10-22T14:26:10.584510+00:00：训练图像实际拼接预览

按用户要求直接读取raw/aloha/adjust_bottle/demo_clean/data/episode0.hdf5首帧和第70帧，沿训练cv2.imdecode→Image.fromarray通道约定（不额外BGR/RGB交换），调用实际assemble_multiview_layout生成两幅H384×W320 PNG并目视检查。三原图均H240×W320；主视角双线性直接变为H256×W320（高放大6.67%，宽不变），腕部变为H128×W160，非保持宽高比的裁剪/补边，不因源图更小而报错，也不补出新细节。无随机ColorJitter，明确为训练几何预处理预览，不冒称训练某次随机增强后的精确输入。原文件与训练/评估配置未改。证据outputs/robotwin_eval/training_image_preview_20261022/{evidence.json,initial_training_input_H384_W320.png,middle_training_input_H384_W320.png,raw_resize_and_full_input_comparison.png}。


## 2026-10-22T14:34:04.316060+00:00：图像尺寸链路一致，但发现训练/官方评估腕部near差异

训练raw JPEG三相机H240×W320，评估D435实时RGB同尺寸、无损PNG/base64传输；两端同一assemble_multiview_layout输出H384×W320，未发现分辨率不匹配。但给定数据采集manifest明确wrist_near_m=0.02、head约0.1，而当前官方camera.py:81及双腕相机构造使用near0.1，wrapper无相关修改。这是具体可见内容分布差异，不能再把“符合官方相机配置”外推“与训练采集全部相机参数一致”。影响大小及SR因果未测，保持为必须考虑的未解决项。证据和范围见docs/ROBOTWIN_TRAIN_EVAL_IMAGE_ALIGNMENT_20261022.md与camera_clipping_alignment_evidence.json。只读核查，不改现行near/训练/资产/已完成成绩；下一步需独立同scene近裁剪面对照，不混入原官方协议。


补充验证：使用此前实际HDF5解码并保存的首帧/第70帧三路RGB，调用真实encode_numpy_b64进行PNG/base64编码，再调用真实ObsPreprocessor.preprocess，输出均为H384×W320；与训练assemble_multiview_layout生成的参考PNG逐像素max_abs_difference=0。为避免加载无关模型依赖，通过文件加载实际模块，未改动图像函数。该检查仅证明同RGB输入下传输/解码/几何预处理一致，不消除训练JPEG压缩、场景差异或腕部near0.02/0.1的内容差异。证据training_image_preview_20261022/train_eval_png_preprocess_parity.json。


## 2026-10-22T14:45:46.966092+00:00：用户授权启动32k腕部near配对诊断及保存图片

独立nearclip32k_ab_20261022，固定三Randomized任务各3原专家manifest场景，9对/18rollout，0.1与0.02两组、无视频；新GPU1模型2905958保持原32k缓存/FFN协议、CPU加载/数值验证与free33GB门槛，控制器2906368先采同状态双near PNG再自动接续策略评估。原官方源/正式成绩及异机进程不改。图片用same physics state、相机姿态/内参/动作计数断言、近裁剪面实际readback、同near重复渲染作噪声对照，并恢复实验near。当前仅启动，不能称策略评估已开始或有SR；结果受9对样本/RT采样与新增拍照调用限制。详见docs/ROBOTWIN_WRIST_NEAR_AB_20261022.md。下一步验首图/完成原生18条及配对seed/指令，保留异常不拼成绩。


## 2026-10-22T15:11:04.263265+00:00 实验异常

["RuntimeError('eval/adjust_bottle/near0.1: exit1; inspect /remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/nearclip32k_ab_20261022/eval/near0.10/adjust_bottle/client.log')"]。这是基础设施/运行异常，不是策略失败率；完整诊断见state.json与各client.log。

所有原始图片：outputs/robotwin_eval/nearclip32k_ab_20261022/{capture,eval}；同状态双near、重复渲染噪声指标见各images/*/evidence.json；原标准评估成绩保持原记录。


## 2026-10-22T15:16:14.878209+00:00：RoboTwin 跨机模型/仿真入口已搭建，42960部署待资源与SSH

用户明确授权当前L20模型服务、4090仿真远程调用，并指定Sidecar42960。新增
benchmarks/robotwin/remote_eval.py、serve_remote_sidecar.py、remote_tunnel.sh，
复用原共享池、FFN/cache数值门槛与single_eval/wrapper；不改模型/训练/原评估入口。
官方near0.1/RT/专家筛seed/unseen/动作预算保留，无视频；1条仅smoke，100条才标准结果。
每sim独占slot，SSH只转发loopback；shared filesystem下按hostname隔离sim JIT缓存，
各部署隔离Inductor缓存，记录checkpoint/config/stats/source指纹、实际分母/原生结果。

本机真实已载入32k服务仅ping：直连与临时TCP转发均返回正确slot/token，错误身份拒绝；
没有reset/predict，不代表42960新模型验收、SSH链路或4090闭环通过。语法/bash检查通过；
42960原路径与封存路径的device/inode/size一致（24862363126 bytes），本次不重复全量hash，
沿用已验证SHA501addbed687e6d7587af60ab4480b9c79eaa2c8867d723b74d54e0acc9deb55。
证据outputs/robotwin_eval/split_host_setup_20261022/validation.json；操作文档
docs/ROBOTWIN_SPLIT_HOST_EVAL_20261022.md。当前所有卡不足新模型33GiB余量，
未停旧/未知作业；用户需提供4090 SSH地址/公钥完成及模型GPU分配。
共享Conda异机GLIBC/驱动/cuRobo/Vulkan RT和真实闭环、网络耗时/完整rollout吞吐
仍待验收，不能据ping外推速度/SR。腕部near训练.02与官方评估.1差异继续保留，
本跨机方案不静默改near，不拼接此前正式成绩。下一步连通后预检→42960 ready→
4090单条→100条，并固定任务/seed对照L20仿真耗时。


## 2026-10-22T15:19:45.279526+00:00：4090 SSH与真实反向隧道通过，补齐异机部署参数

用户提供root@210.45.70.21:26199，现场确认8×4090、共享路径与仿真Python/包版本可读取。新增独立remote_reverse_tunnel.sh，使模型主机主动SSH到4090，不再依赖模型主机SSH入口。临时4090:22790→L20:21652通过真实薄client ping/slot/session验证，随后仅停止本次隧道；未reset/predict、未启动42960新模型或4090闭环。证据split_host_setup_20261022/ssh_reverse_validation.json。当前L20 GPU0仅21287MiB空闲且98%利用率，未强行启动。用户选择可自行在另一模型主机部署；具体GPU/SSH端口/模型端口/隧道左右端口/session/Python/路径/任务参数及三步命令已写docs/ROBOTWIN_SPLIT_HOST_EVAL_20261022.md第4节。共享Conda真实CUDA/cuRobo/RT/首rollout与速度仍未验收，近裁剪面/缓存与跨机随机性边界保持。下一步在可用GPU启动42960→反向隧道→4090一条闭环，再100条正式评估。


## 2026-10-22T16:27:20.236948+00:00：后台SSH重复Stopped处理

用户在A100三次nohup反向隧道均Stopped；符合SSH终端交互导致暂停的现象，尚未确认密码/host-key/密钥口令中的具体原因。两个独立隧道脚本补-n、BatchMode=yes与ConnectTimeout=15，不再后台等待交互。bash语法和隔离假ssh参数转发检查通过，证据split_host_setup_20261022/background_ssh_fix_validation.json；没有据此声称A100认证已成功。文档第5节给出仅清理已核实重复隧道、前台确认认证、免密重试及ssh -fNT先认证后后台方案。不结束模型、训练或未知ssh，不影响正式评测/数据。


## 2026-10-22T16:47:59.973822+00:00：42960未完成80条件中的24条件4090续测方案已交付

用户报告A100→4090模型调用与仿真链路已验证通过，要求写六路标准续测脚本、不重复原完整任务。本次读取旧LIVE及停止final_record，按100=50task×2域口径选12个此前两域均未启动的task，共24条件/2400条，与20complete和8partial重叠0；选集旧记录SHA固定。tasks为handover_mic、lift_pot、move_pillbottle_pad、move_playingcard_away、move_stapler_pad、open_laptop、open_microwave、place_bread_skillet、place_burger_fries、place_cans_plasticbox、stamp_seal、turn_switch。

新增eval_remaining24.py，六独立slot动态队列，Randomized优先/官方预算大优先派发但预算不改；每条件继续调用原remote_eval sim，无视频，保留RT/native筛seed/unseen/EEF/cache/FFN/near0.1。检查checkpoint SHA/session/端口/4090初始容量，旧数据不改。launch_remaining24_from_a100.py通过免交互SSH在4090分离启动dispatcher，shell参数shlex编码、独占启动标记防重复；用户可在A100终端执行全部流程。

CPU替身实际控制流验证通过：24项唯一派发/六路均参与、单项失败继续且分母排除、旧输出拒绝覆盖；远端bootstrap本地替身检查含带特殊字符路径与重复提交拒绝。证据split_host_setup_20261022/remaining24_validation.json与remaining24_launch_validation.json，dry-run计划封存。没有调用真实模型/GPU，不能当新SR/吞吐。本助手未启动2400条，避免与用户当前实例重复。完整选集、GPU/SSH/端口/RUN参数、A100模型及隧道启动、A100一条命令远程六路评测、日志/完成口径已追加docs/ROBOTWIN_SIDECAR42960_LIVE_20261020.md，旧停止快照保留。

限制/下一步：用户选择空闲4090卡并确保旧smoke不占slot后执行；8000MiB/路仅初始门槛不保证峰值，CUDA/cuRobo/RT按用户验证范围而非本CPU测试外推。near训练.02/评估.1、跨硬件RT与随机性、cache闭环影响仍保留。完成24/24且2400实际分母后才给本批SR；旧20条件独立保存，不自动混合episodes，后续跨机总体需逐条件核查并保留硬件来源。


## 2026-10-23T07:32:30.099823+00:00：24000模型多GPU部署启动

用户授权指定outputs/1022-sidecar-warmup2k/checkpoint_step_24000.safetensors尽量并行部署并要求GPU6补充。精确权重SHAc8835677cb59324ce3f2bf8c01e80178e36397ac3379d7bc1ca3d0d6ea23de76，独立hardlink及自有config/stats/tokenizer合同固定；实际config为isolation2K，不混同full-backward新实验。仅回收已完成/失败且无连接的本助手旧32k服务PID2905958/3486830，原日志成绩保留；GPU0–6七份服务CPU加载中，GPU6初始28GB余量按用户要求降初始门槛27000MiB但真实峰值待验，不当成ready。每份FFN/cache真实数值门槛及三端口20D线上动作验收后才汇报。专门文档ROBOTWIN_WARMUP2K24000_DEPLOYMENT_20261023.md；输出/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z。尚未启动仿真/无新SR；未解决near/cache/RNG/跨卡容量边界保持。


## 2026-10-23T07:44:26.112558+00:00：24k FFN数值门槛失败，独立部署改为完整Sidecar eager+cache

GPU0–6载入成功且完成真实fixture推理，FFN4kernel已生成但同缓存参考的严格allclose门槛未通过，最大归一化动作差0.0078125；各卡一致，不是GPU6容量失败。未放宽阈值或声称SR受损。两fixture热请求均值：GPU0 eager_cached0.3634s/ffn_cached0.3758s，GPU5 .3865/.4394，GPU6 .3834/.3850；有限样本，不外推吞吐。

按用户持续部署授权及原先不影响Sidecar能力原则，仅本次输出目录新增serve_eager_cached.py/launch_eager_cached.py，三tap保持、compile=false、cache .99/max3与10步保持，自有stats/权重不变；不改生产核心代码/原入口。GPU0–7八份重新启动，两个fixture各重复两次必须与原同checkpoint eager_cached参考逐元素相等后才开放服务，再验各slot真实20D动作。原FFN失败报告与进程记录完整保留compiled_attempts。为减少GPFS mmap启动等待，将精确权重复制到/dev/shm/openwam24000_c8835677cb59324c，流式SHA/size与共享权重一致（local_weight_cache.json），不是下载、无数学变化；重启主机需重建本机RAM缓存。8卡容量与线上返回仍在验收。文档ROBOTWIN_WARMUP2K24000_DEPLOYMENT_20261023.md/本次deployment.json为权威，不把初期FFN计划当最终部署参数。


## 2026-10-23T07:55:55.087507+00:00：24k八卡模型部署完成，4090全24路验收通过

精确权重c8835677cb59324ce3f2bf8c01e80178e36397ac3379d7bc1ca3d0d6ea23de76，GPU0–7八份完整Sidecar+cache、compile=false服务最终ready。32个真实fixture比较（8×2×2）逐元素等于原eager_cached参考、max差0，本机24路和4090经反向隧道24路真实PNG/state→20D有限动作全部通过并reset。GPU6/7亦通过，未终止其未知原作业。最终证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/final_deployment_acceptance.json；映射ready_endpoints.json，跨机remote4090_acceptance.json，专用文档ROBOTWIN_WARMUP2K24000_DEPLOYMENT_20261023.md已更新。

初期FFN门槛失败不放宽，eager初试保存被处理器修改的fixture产生ndarray序列化错误而退出；保留失败脚本/日志，修复独立副本与原JSON存储后v2全部验收。未改变生产核心或训练。权重在本机/dev/shm精确SHA缓存以加快启动，重启需按local_weight_cache.json重建，原共享权重和固定hardlink保留。这里只验部署，不是正式rollout/成功率或吞吐因果；cache近似、两fixture范围、原near差异/RT跨卡/RNG限制保持。下一步按用户指定任务独立部署sim并绑定本次SHA/端口，不直接运行旧42960 SHA限定队列。


## 2026-10-23T08:30:43.631943+00:00：24k全50任务双域30条趋势评估已在4090启动

用户明确授权标准流程先每条件30条、不存视频，100=50task×Clean/Randomized，总3000。GPU0–7 L20八模型完整Sidecar/cache .99 max3、compile=false，4090 24独占端口待命；当前显存安全容纳9仿真，先分布8模型，第9路第二slot。4090 GPU1/4各2路、0/2/3/5/6各1路，GPU7余量不足不放sim。共享100条件动态队列，短锚点先看趋势、其后长原预算优先、域交错；不固定任务子队列，不停未知原作业。原生expert筛seed/unseen/seed0、RT32/OIDN、near.1、成功判据/预算不变，ROBOTWIN_TEST_NUM=30通过原wrapper修分母。

独立输出/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/warmup2k24000_4090_all100_30ep_20261023T081011Z，controller PID1521681/4090 host55fda0705558。process-local Scene显式cuda:0（CUDA_VISIBLE映射后）绑定，逐进程渲染PCI匹配指定GPU断言通过；不改上游源，官方commit0aeea2d669c0，source_evidence.json记录sha/diff。9路真实policy动作推进/终态已出现，无基础设施错误，不把启动/ping当rollout。

用户要求监督GPU1/4/6可否再加：一分钟采样最低free分别2278/5208/5531MiB，启动预热已上涨，暂不足额外稳定仿真。单路整卡差估计4.5–6GiB；当前预算6656MiB/路+1024MiB卡余量，启动信用避免尚未分配显存时超派。gpu146_peak_sampling.json保留30个2s样本。每条件基础设施失败可独立新attempt重跑至多一次，不拼失败partial、完整30才计该条件SR；45min无完整episode看门狗保守，现有episode远小于此。

raw client日志带ANSI颜色，原native最终验证会剥离；独立watch_trend.py也剥离原日志后生成live_progress.json和docs/ROBOTWIN_WARMUP2K24000_30EP_LIVE_20261023.md（每30s），不依赖初期state内partial计数、原日志不改。中途混合partial仅进度，不外推完整总体。完整100×30结束再汇总正式本批趋势，不当作官方100episode SR。缓存近似/相机near训练.02与官方.1、硬件/seed未配对等边界保持。下一步继续队列、按显存观察补路、记录完整30条条件与实际速度；尚未完成3000条。


## 2026-10-23T08:29:45.223323+00:00：24k 30×100趋势评估初期速度估计

9路仿真，启动约15.12分钟已有65条终态；最近187.88秒完成16条，约306.58episode/h；包含初始化的总平均258.01episode/h。剩余2935条按短窗线性约9.57h。早期以短锚点任务为主，后续较长任务/低成功率完整预算可能拉长，因此目前给用户的工作估计为还需10–16小时（9路持续、无新故障）；不是确定ETA。首个完整30条件尚无，click_alarmclock Clean已19/30，仅此类短任务预计较快先完成；混合57/65不当全任务SR。证据speed_snapshot_before.json/after.json；下一步积累更长窗口及长任务吞吐，容量允许扩路后重估。


## 2026-10-23T08:41:26.985121+00:00：24k首个完整30条条件及官方对照

首个click_alarmclock Clean30/30=100%，官方99%，+1pp；完整范围仅1条件，不能作为总体100%。其他partial：adjust_bottle Clean17/17、Random3/6（官方90/51）；hammer Clean16/18、Random3/8（99/55）；alarmclock Random12/15（77）；shoe Clean23/23、Random6/10（98/51）；dustbin Clean3/3（83）。总观察130条、9路、infra errors0。已逐项验证complete原生30分母/exit/错误，在线复核Issue46官方表；对照快照/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/warmup2k24000_4090_all100_30ep_20261023T081011Z/official_comparison_20261023T084247Z.json。专用FIRST_OFFICIAL_COMPARISON文档已写；smallN/短任务覆盖偏差保留，hammer暂低不能判定全Randomized或模块影响，继续收齐各30条。


## 2026-10-23T09:17:30.822608+00:00：GPU4补路、修复错误看门狗、无损接管进行中评测

用户询问GPU5/4空余及总耗时。GPU5完成place_shoe Clean30条后启动open_microwave Randomized，暂时2.2GiB占用时空闲6GiB，实际随规划/rollout上涨到约5.5GiB、余量约2.8GiB，不能再加。GPU4的hammer Clean完成释放后实际补到3仿真，整卡现约23.8GiB/24.2GiB，余量仅约0.4GiB；全局并行已10路，不再强塞该卡。

发现并已告知用户：原dispatch.py的partial解析未剥离ANSI颜色，导致启动信用未释放且last_progress未更新，45min误将6项仍推进的条件TERM（不是模型失败/OOM）。原partial日志保留，6项由原控制器启动新attempt重测，正式汇总不混入旧partial。看门狗错误发生前3项Clean完整30结果有效。

新增elastic_v2.py，ANSI剥离、动作步/episode双进度心跳、正确释放信用。控制切换采用保留native进程和真实exit状态：封存handoff_snapshot，原snapshot写入路径设排空标记以阻止旧线程继续派发；旧控制器暂停，9个在跑native子进程继续、不重启不重置模型；/proc zombie wait status由独立collector读取，仍要求真实exit0+原validate_result完整30条后写run.json并移交V2，非猜测返回码。真实fork子进程exit7与waitpid对照通过。全部旧native结束后才退出旧控制器释放slot锁；新控制器独立管理空闲slot/新任务。旧watchdog已无法再误杀，collector仅45min无日志动作进展才判停滞。

标准仿真/seed筛选/成功判据/预算/视频关闭均未改变；只有orchestration修正。原state备份state.before_elastic_v2.json，canonical state.json链接elastic_v2/state.json；live watcher保持读取，另统计physical含重测rollout与最新attempt覆盖，避免重跑后把总工作量变少或重复计入最终SR。paused_legacy_parent.json、legacy_exit_collection.json、adopted_native_exit.json、native_exit_status_collection_test.json记录身份和真实退出证据。

09:14:03 UTC快照：自08:14:38运行3565s（59m25s），实际完成229条终态（含重测/中止前终态）；当前最新attempt覆盖128条，其中3项完整90条；10路活跃，历史调度误停6次、未归为模型失败。后续继续30×100趋势任务，需待完整30结果再做官方对照；初期10–16h ETA需考虑这次调度误停损耗重估。


## 2026-10-23T14:50:53.017923+00:00：17条件阶段对比与显存保护进一步修正

{"utc": "2026-10-23T14:50:53.017923+00:00", "elapsed_minutes": 396.24542463333336, "physical_including_retries": 912, "current_attempt_episodes": 736, "current_attempt_successes": 474, "summary": {"all": {"conditions": 17, "successes": 337, "episodes": 510, "rate": 66.07843137254902, "official": 68.82352941176471, "gap_pp": -2.7450980392156907}, "demo_clean": {"conditions": 9, "successes": 226, "episodes": 270, "rate": 83.7037037037037, "official": 84.88888888888889, "gap_pp": -1.1851851851851904}, "demo_randomized": {"conditions": 8, "successes": 111, "episodes": 240, "rate": 46.25, "official": 50.75, "gap_pp": -4.5}}, "active": 13, "historical_errors": 12}

新增6次渲染cannot create buffer：GPU1一次、GPU4一次、GPU6四次（发生于扩至多路后），不直接认定模型失败，不能声称全部修复。冻结V2并保留13个在跑native不重启，V3按live perGPU caps1/2/1/1/2/1/1/0保守接续，不因瞬时低占用超派；给buffer失败重试>=8192MiB。当前在跑超限项自然完成后再回落安全并发，不主动终止有效rollout。新lease epoch避开冻结旧coordinator锁，显式检查legacy相同port忙时不派发，防policy串扰。collector读取冻结父进程持有的真实zombie wait code/原30结果，原生验证未放宽。handoff_v3.json、elastic_v3/capacity_policy.json、exit_collection.json保留审计。全部完整条件重新validate_result通过；详细逐项对比docs/ROBOTWIN_WARMUP2K24000_REPAIR_AND_COMPARISON_20261023.md，原始队列/失败/partial不删除不拼样本。下一步继续监督buffer峰值/重试，重点跟踪hammer Random和microwave Clean明显偏低，不能单独归因Sidecar结构。


## 2026-10-23T15:39:53.073755+00:00：用户质疑吞吐后的实际工作量核对

{"utc": "2026-10-23T15:39:53.073755+00:00", "complete_conditions": 25, "validated_episodes": 750, "current_coverage": 869, "physical_including_retries": 1066, "discarded_previous_attempts": 197, "last_hour_effective_episodes_per_hour": 169.417150369322, "active": 10, "errors": 13}

整项验收、最新attempt覆盖、累计物理终态三个数必须分别报告，不能用含重测的物理数冒充3000条目标进度。调度ANSI误停和多路渲染buffer错误造成重跑浪费；初期按短任务306/h估10–16h依据偏乐观，当前近小时有效吞吐约169.4条/h。V3已经改正心跳/端口隔离与每卡上限、提高buffer重试门槛，现有native不中断；后续需继续记录稳定窗口和任务预算构成再估ETA。确认官方close_env原已按clear_cache_freq周期清理SAPIEN缓存，未擅自修改该频率或渲染质量换速度。


## 2026-10-23T16:04:37.548820+00:00：24k阶段27条件对照

{"all": {"conditions": 27, "successes": 549, "episodes": 810, "rate": 67.77777777777779, "official": 71.85185185185185, "gap_pp": -4.074074074074062}, "demo_clean": {"conditions": 15, "successes": 369, "episodes": 450, "rate": 82.0, "official": 86.8, "gap_pp": -4.799999999999997}, "demo_randomized": {"conditions": 12, "successes": 180, "episodes": 360, "rate": 50.0, "official": 53.166666666666664, "gap_pp": -3.1666666666666643}}
当前10路、944最新attempt覆盖、1141物理含重测、历史异常13未新增、无最终failed项。27条件重新原生exit0/30分母校验通过。最大缺口microwave Clean7/30 vs82%，hammer Random8/30 vs55%；鞋Random21/30 vs51%为正向。仅事后定位：其他25项71.2 vs72.12（-0.92pp），不能替换总体27项67.78 vs71.85（-4.07pp）。文档/remote-home/jinminghao/OpenWAM/docs/ROBOTWIN_WARMUP2K24000_RESULTS_20261023T160437Z.md与comparison_20261023T160437Z.json保留快照。下一步继续全部100项，重点诊断两个大缺口；30对100统计/near差异/cache/跨硬件及训练预算因果边界继续保留，未改模型或训练。


## 2026-10-24T02:09:11.285245+00:00：24k 30条趋势队列终态

{"utc": "2026-10-24T02:09:11.285245+00:00", "status": "complete", "scope": "30episode trend; not standard100", "observed_episodes": 3000, "observed_successes": 2046, "executed_rollouts_including_retries": 3205, "executed_successes_including_retries": 2195, "completed_conditions": 100, "planned_episodes": 3000, "active_simulators": 0, "errors": 14, "episodes_per_hour_estimate": 32.43550696512386, "linear_remaining_hours": 0.0, "note": "linear ETA excludes task-mix correction; current short-task coverage is biased"}
证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/warmup2k24000_4090_all100_30ep_20261023T081011Z；完整率按30实际分母，不与旧实验拼episodes；详见专用LIVE。


## 2026-10-24T02:08:51.994191+00:00：24k全50双域×30条趋势评估完成并重新验收

{"all": {"conditions": 100, "successes": 2046, "episodes": 3000, "rate": 68.2, "official": 69.04, "gap_pp": -0.8400000000000034}, "demo_clean": {"conditions": 50, "successes": 1307, "episodes": 1500, "rate": 87.13333333333333, "official": 89.36, "gap_pp": -2.2266666666666737}, "demo_randomized": {"conditions": 50, "successes": 739, "episodes": 1500, "rate": 49.266666666666666, "official": 48.72, "gap_pp": 0.5466666666666669}}
实际墙钟17.9038h，100/100完整、3000条，100份native exit0/实际30分母/seed递增独立复验通过。没有最终失败/待测项。历史异常{"old_watchdog_false_stop": 6, "renderer_buffer_failure": 8}，中止partial不计正式分母；保留所有尝试。最终整体68.20 vs官方69.04（-0.84pp），Clean87.13 vs89.36（-2.23pp），Random49.27 vs48.72（+0.55pp）。这修正早期选集-4.07pp印象，不能说完整Randomized整体更差，也不能小幅+0.55就认定Sidecar收益。每项30 vs官方100、小样本/训练预算与stats/near/缓存与硬件非配对限制保留。结果/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/warmup2k24000_4090_all100_30ep_20261023T081011Z/final_results_20261024.json及CSV，专用FINAL_RESULTS文档含50task逐项；下一步按用户决定异常任务深入诊断/独立baseline同配置同seed配对，不自行扩到100条或修改训练。


## 2026-10-24T05:21:56.122895+00:00：新full-backward warmup2k20k独立30×100评估准备

用户指定权重：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_20000.safetensors`。SHA256：`0295ed0bebb304929bf0e6baaa94421242440979b1796ccf502c687b01a4127c`；精确20k独立固定目录，自己的config/stats/Sidecar contract。与24k归一化统计SHA相同。

50个task × demo_clean/demo_randomized ×30条 =3000条。完整Sidecar三tap7/15/23；compile=false，cache=true（.99/max3），10步去噪、32动作chunk；与24k最终有效配置一致。不存视频。官方expert筛seed、unseen、seed0、原成功判据/动作预算、RT32spp/OIDN、near.1保持。

本轮模型在L20 GPU0–7，仿真在4090 host55fda0705558。23900/23901/23902至23970/23971/23972，独立token与结果目录。八份模型必须通过本checkpoint fixture重复/跨副本精确一致、本机24路与远端24路真实PNG/state→20D动作验收；验收不等于仿真rollout。

动态共享100条件队列，从八份不同模型开始分配。初始显存预算7680MiB/路+1024MiB整卡余量，perGPU caps1/1/1/1/2/1/1/0；实际并行还受空闲显存约束，不终止未知作业。使用24k最终V3的ANSI剥离、episode/action进度看门狗（45min无进展）、启动信用、端口独占保护；全新队列无旧attempt接管。基础设施异常至多3attempt，失败partial不合并，完整30条才计SR。

输出：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward20000_4090_all100_30ep_20261024T052041Z`。模型部署：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward_warmup2k20000_deploy_20261024T051552Z`。source_evidence.json保留与24k运行脚本的差异/上游commit。live_progress.json记录进度，comparison.json仅完整条件按相同任务子集与官方/24k比较，结束复验后写final_results.json/CSV。动态任务列表见[实时表](ROBOTWIN_FULL_BACKWARD20000_30EP_LIVE_20261024.md)。

24k已完成：Clean87.13%、Randomized49.27%、整体68.20%；官方相应89.36%、48.72%、69.04%。新20k为前2k步完整反向但延迟WAM更新，旧24k为isolation warmup2k；训练步数亦不同，因此差值不能单独归因warmup方案。训练near.02与官方eval.1、cache近似、不同GPU/seed未逐条配对及每项30的小样本边界继续保留。未修改模型、训练、数据或上游官方源。

下一步完成八卡/跨机验收，启动4090队列，确认真实动作推进与显存，再报告初始进度。


## 2026-10-24T05:25:07.667197+00:00：full-backward20k七模型跨机验收通过，30×100队列已启动

精确checkpoint SHA 0295ed0bebb304929bf0e6baaa94421242440979b1796ccf502c687b01a4127c。GPU0–6七模型，本checkpoint两个fixture各重复两次/跨副本逐元素一致；本机及4090全部21路真实PNG/state→20D有限动作通过。GPU7被外部作业显存上涨挤占，加载OOM，失败日志保留在独立deployment_with_gpu7_failed.json；不归为任务失败，不停未知作业。原24k100项完成，已确认无客户端后退出其空闲服务，所有结果与权重保留。

4090 dispatcher PID1887540，开始2026-10-24T05:24:39.708159+00:00，7路当前启动，GPU1自由显存从16173降至7696MiB，7680+1024门槛自动暂缓；容量政策不放松。输出/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward20000_4090_all100_30ep_20261024T052041Z。同24k标准协议/完整Sidecar/cache开启compile关闭/视频关闭；任务各30。watch_trend/watch_comparison后台运行，后者只比较完成30条的同任务子集，终态原生复验。此记录仅部署和队列启动证据，尚待真实policy动作/终态，不当成功率。


## 2026-10-24T05:28:21.325496+00:00：full-backward20k评估真实闭环已跑通

8路全部出现真实policy action推进，首个beat_block_hammer/demo_clean seed100000在111步成功；非ping/fixture冒充rollout。基础设施异常0，尚无完整30条条件，不外推总体成功率。first_rollout_acceptance.json位于/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward20000_4090_all100_30ep_20261024T052041Z，逐路记录原日志/进度；GPU渲染PCI、独占端口、新20kSHA、视频关闭/unseen/EEF/30条配置8路验收通过。持续动态队列与显存门槛，完成同条件再与24k/官方对照；结果独立保留，不修改训练或上游源。


## 2026-10-24T05:32:56.151975+00:00：初始32k与warmup24k全100条件核对

两份封存final_results逐键对齐100/100，官方对照100项完全一致。32k10000条66.48%，warmup24k3000条68.20%，+1.72pp；Clean87.66→87.13（−0.53pp），Random45.30→49.27（+3.97pp）。对官方综合69.04差距−2.56→−0.84pp。不同N、训练步数、编译和仿真硬件，不作warmup单因素因果结论。完整50任务双域表与100条件CSV见docs/ROBOTWIN_INITIAL32K_VS_WARMUP24K_FULL_COMPARISON_20261024.md。下一步继续新full-backward20k30条队列，收齐后同条件比较。


## 2026-10-24T05:39:32.386413+00:00：24k低分任务与官方checkpoint/10740交叉核对

主要证据：open_microwave/Clean10740=72 vs24k=23.33（差−48.67pp），beat_block_hammer/Randomized51 vs26.67（−24.33pp），stamp_seal/Randomized38 vs10（−28pp），stamp_seal/Clean88 vs73.33（−14.67pp）。hanging_mug/Clean10740=49与官方相同、24k36.67；handover_block/Clean官方新环境复测85、10740=95、24k73.33。不能将这些结果统一解释为“本地仿真只能支持到24k这个成功率”。相反place_bread_skillet/Randomized：10740=48、24k=50、官方61，共同下降，仍可能有训练/任务/评估共性因素。

官方同一发布权重5任务双域旧44.3→新70.7%，官方该子集72.1%，证明历史旧环境/执行栈问题确有显著影响。并非全部50任务的官方权重新环境复测；本表11条件仅handover_block/Clean具有该新环境复测，其余不可将旧值当新值。9项manifest+1传统native的协议限制保留。

10740完整20条件：67.70%，官方同范围69.85%；32k同范围62.80%；warmup24k在同20条件61.50%。这十任务按32k低分预选，不能外推50任务。24k全100条件68.20%不应与该20条件67.70%直接当同集合比较。

因果边界：10740和32k主要L20仿真，24k当前4090；样本100 vs30、训练预算/stats、compile配置、accepted seed/指令/RNG未配对。证据倾向checkpoint特定任务能力差异，不能单独认定Sidecar结构或warmup导致，也尚未完全排除当前4090配置交互。下一步若需定因，应在当前4090相同native流程固定场景/指令/RNG/accepted manifest，对open_microwave Clean、beat_block_hammer Randomized、stamp_seal Randomized做官方checkpoint/10740/24k配对对照。此次未额外启动这些权重，保持用户当前20k主队列。


完整11条件表见docs/ROBOTWIN_WARMUP24K_LOW_TASK_BASELINE_DIAGNOSIS_20261024.md；具体JSON/CSV见/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward20000_4090_all100_30ep_20261024T052041Z/low_task_diagnosis_20261024。


## 2026-10-24T05:41:09.762356+00:00：用户授权4090 GPU1/4各补一路，已验证真实动作

GPU1由1→2、GPU4由2→3，总并行8→10。新增put_bottles_dustbin双域各30条、视频关闭、独占23931/23901端口，真实动作推进与渲染绑定已验收。当前free GPU1=5073MiB、GPU4=4651MiB；历史infra errors=0。证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward20000_4090_all100_30ep_20261024T052041Z/gpu14_capacity_trial_20261024T053746Z/action_acceptance.json，2s采样samples.jsonl，sole supervisor PID1896446。使用现有live capacity policy，无重启原评估；低于2048MiB或本卡出现cannot create buffer即恢复原cap，仅停止最新自有额外sim，其partial保留但不计完整SR。GPU4历史三路buffer失败仍为风险，不能因当前成功推进认定永久解决。原supervisor版本留存，v2原状态接续并新增buffer失败回退，不存在两个活跃守护同时改配置。

低分交叉分析：11个warmup24k相对官方≤−10pp条件已核对，10740在microwave Clean72%、hammer Random51%、stamp Random38%明显高于24k23.33/26.67/10%；官方ckpt新环境仅五任务复测，不能用旧环境未复测数值替代。详细docs/ROBOTWIN_WARMUP24K_LOW_TASK_BASELINE_DIAGNOSIS_20261024.md。checkpoint特定差异比统一环境天花板更符合现有证据，但4090尚无同条件官方/10740配对，不能排除硬件/compile/样本协议交互，也不能单独归因模块。继续20k主队列与资源监督，定因需未来同4090固定输入/accepted seed/指令RNG对照。


## 2026-10-24T08:16:14.372580+00:00：full-backward20k首8条件进度与资源回退

{"utc": "2026-10-24T08:16:01.483270+00:00", "status": "running", "scope": "30episode trend; not standard100", "observed_episodes": 402, "observed_successes": 275, "executed_rollouts_including_retries": 402, "executed_successes_including_retries": 275, "completed_conditions": 8, "planned_episodes": 3000, "active_simulators": 9, "errors": 1, "episodes_per_hour_estimate": 73.9326047332202, "linear_remaining_hours": 35.140111854230916, "note": "linear ETA excludes task-mix correction; current short-task coverage is biased"}
完整条件同范围对比：{"all": {"conditions": 8, "episodes": 240, "successes": 189, "rate": 78.75, "official_same_subset": 77.5, "gap_pp": 1.25, "old24k_same_subset": 75.83333333333334, "delta_vs_24k_pp": 2.916666666666657}, "demo_clean": {"conditions": 4, "episodes": 120, "successes": 118, "rate": 98.33333333333333, "official_same_subset": 96.5, "gap_pp": 1.8333333333333286, "old24k_same_subset": 95.83333333333333, "delta_vs_24k_pp": 2.5}, "demo_randomized": {"conditions": 4, "episodes": 120, "successes": 71, "rate": 59.166666666666664, "official_same_subset": 58.5, "gap_pp": 0.6666666666666643, "old24k_same_subset": 55.833333333333336, "delta_vs_24k_pp": 3.3333333333333286}}
完整条件仅4短锚点任务双域8项，不能将78.75%外推全100条件；hammer Random8/30仍明显低于官方55%，新20k未显示该项改善。open_microwave Clean当前9/20=45%只是partial，不能与24k完整7/30直接定论。GPU1额外sim在06:12:58 free1795MiB触发既定2GiB保护，blocks_ranking_size Random首attempt在0episode时TERM、未计SR；已自动在GPU5重跑推进14/30。不是模型失败或未处理卡死。当前GPU1回退1路、GPU4维持3路，总9路，其他活跃日志均有动作推进；无最终failed条件。GPU6 free14GiB为实时资源变化，本次状态核查未改容量/新增进程。后续继续全100条件，留意长预算1700/1500/1200步的吞吐下降及GPU3显存余量，保持原预算/成功判据，不凭早期高SR外推。证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward20000_4090_all100_30ep_20261024T052041Z，具体资源回退见gpu14_capacity_trial_current.json目标rollback_gpu1.json。


## 2026-10-24T08:20:04.804402+00:00：32k四任务双域前30条回溯

结论：32k综合73.25→74.58（+1.33pp），Clean96.25→98.33（+2.08pp），Random50.25→50.83（+0.58pp）；小样本变化不统一，alarmclock Random+11.67pp、hammer Random−6.33pp。新20k同范围78.75%，相对32k完整100差+5.50pp；换同N前30参照差+4.17pp，其中Clean两者都是118/120=98.33，Random为71/120 vs61/120，相差+8.33pp。warmup24k75.83相对32k前30差+1.25pp。hammer Random三份30条均8/30，原32k全部100为33/100，说明该项30条低值亦有样本窗口因素，不能仅据26.67 vs33判warmup退化；对官方55仍有较大差距。

仅8条件早期选集，不外推完整100条件，不能以这一组78.75替代全任务最终成功率。32k前30与新20k的accepted seed不必逐条相同，且指令、RNG、硬件/compile及训练步数不同；同N仅消除样本数量差异，不是受控因果配对。

证据docs/ROBOTWIN_SIDECAR32K_FIRST30_FOUR_TASKS_20261024.md及docs/evidence/sidecar32k_first30_four_tasks_20261024/analysis.json与comparison.csv；全部100条原始成功计数/seed/最终run已校验。


## 2026-10-24T08:36:29.830617+00:00：用户授权L20对照4090四任务各30条

权重：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_20000.safetensors`；SHA256 `0295ed0bebb304929bf0e6baaa94421242440979b1796ccf502c687b01a4127c`。共享盘精确权重已验证RAM缓存，不下载新文件。完整Sidecar三tap、cache .99/max3、compile=false、10步/32chunk，官方同commit和同隔离Conda、标准native seed0/unseen/expert筛seed、RT32/OIDN/near.1、原动作预算与成功判据，视频关闭。

|Task|Mode|L20仿真GPU|独占端口|4090成功/30|4090率|4090整任务秒/条|
|---|---|---:|---:|---:|---:|---:|
|adjust_bottle|demo_randomized|2|24970|16/30|53.33%|162.33|
|beat_block_hammer|demo_randomized|4|24971|8/30|26.67%|154.46|
|click_alarmclock|demo_clean|6|24972|30/30|100.00%|34.54|
|place_shoe|demo_clean|5|24973|28/30|93.33%|91.18|

模型单独部署在本机L20 GPU7，一份engine串行服务4个独立PolicyServer会话，24970–24973，无SSH隧道，避免现有4090调度器随时复用同一端口。初始GPU7 free28989MiB，模型重复/跨GPU动作与同checkpoint规范参考严格逐元素比较通过后，再验四端口真实PNG/state→20D动作。若资源失败保留日志，不终止未知进程。GPU3部署期间空闲下降，place_shoe改GPU5，其他三卡2/4/6，每卡仅新增一个sim。

速度口径：四个历史4090任务原run.json wall_seconds/30（包含CUDA/场景初始化、专家筛seed、全部rollout）与L20相同整任务口径对比。L20另以process-local只读计时包装get_obs、take_action、policy RPC及每条policy rollout边界，原方法每次只调用一次、参数返回值不改、不额外渲染或抽RNG。原4090历史日志没有这些分段计时，不伪造或声称已能纯对比GPU渲染耗时。

限制：新独立模型四会话共用engine、4090历史每模型约1–2路；本机GPU与模型服务共卡负载、背景进程/CPU/网络不同，整任务速度差不能纯归因L20/4090型号。SR采用相同原生seed扫描协议，但不同硬件可能使expert接受seed不同，将报告顺序相同与交集，不自动称逐seed/指令/RNG精确配对。这是用户要求的当前部署方式系统对照；若需单变量硬件因果，需后续固定相同manifest/推理服务负载与软件/场景/指令/RNG再测。

调度：独占token+slot锁，四任务固定卡并行，各任务只有基础设施错误最多3attempt；native完整30/exit0/成功分母验证，policy失败不重试；45min无episode或动作进展才中止。每次启动要本卡至少10GiB空闲，不动未知作业。独立源码source_evidence.json记录上游commit/脚本SHA/只读timing差异/主机负载和驱动。

输出：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward20000_l20_vs4090_4x30_20261024T083037Z`。后台dispatch.py/report.py完成后自动生成final_comparison.json/CSV；[实时表](ROBOTWIN_FULLBACKWARD20K_L20_VS_4090_LIVE_20261024.md)。

当前模型验收/仿真启动进行中，未产生L20正式SR结论。


## 2026-10-24T08:39:45.639214+00:00：L20四路对照已全部进入真实policy动作

GPU2 adjust_bottle/Randomized、GPU4 beat_block_hammer/Randomized、GPU6 click_alarmclock/Clean、GPU5 place_shoe/Clean全部实际推进；四路CUDA/render PCI与NVIDIA L20断言通过，独立24970–24973四会话、GPU7同20k模型通过重复/跨副本逐元素一致及四端口真实动作验收。视频关闭；每任务30，原4090队列继续。timing.jsonl的policy rollout边界/get_obs/take_action/policy_rpc均已验证记录有限非负耗时，原调用一次无额外渲染或RNG。当前infra errors=0，还没有完整30条条件，不能报告L20成功率或GPU速度因果。证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward20000_l20_vs4090_4x30_20261024T083037Z/first_action_acceptance.json。后台dispatch/report持续，结束自动写final_comparison.json/CSV；速度对照保留单新模型四路共享 vs历史4090每模型1–2路、CPU/网络/背景负载不同等限制。


## 2026-10-24T10:23:11.074269+00:00：L20/4090四条件对照结束

{"completed_conditions": 4, "planned_conditions": 4, "observed_episodes": 120, "planned_episodes": 120, "active": 0, "l20_complete_successes": 82, "complete_episodes": 120, "l20_complete_rate": 68.33333333333333, "reference4090_same_subset_rate": 68.33333333333333}
证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward20000_l20_vs4090_4x30_20261024T083037Z；同native协议、30条；速度包含不同模型共享负载/CPU/网络，不能纯归因GPU。完整结果见专用LIVE/final_comparison.json。


## 2026-10-24T14:41:44.244791+00:00：L20/4090同20k四条件全部完成并复验

L20四项120条无infra error，重新validate_result通过。两卡均82/120=68.33%，各项差+2/−1/0/−1次成功；seed交集29/24/30/29，只有alarmclock完全相同accepted seed。整任务等权135.87 vs110.63秒/条，L20多耗时22.82%；鞋L20反而较快21.99%。没有支持硬件导致SR统一降低，亦不能证明硬件完全等价；负载/模型共享/样本未完全配对，不能纯归因GPU速度。hammer Random两卡均23.33/26.67低，不支持仅因4090导致该项异常。详细/remote-home/jinminghao/OpenWAM/docs/ROBOTWIN_FULLBACKWARD20K_L20_VS_4090_FINAL_RESULTS_20261024.md。继续20k主队列，纯因果需固定manifest/指令/RNG与推理负载同仪表配对，不自行扩测或动未知作业。


## 2026-10-24T14:43:37.047849+00:00：主20k评估35条件进度

{"all": {"conditions": 35, "episodes": 1050, "successes": 718, "rate": 68.38095238095238, "official_same_subset": 72.48571428571428, "gap_pp": -4.104761904761901, "old24k_same_subset": 68.57142857142858, "delta_vs_24k_pp": -0.190476190476204}, "demo_clean": {"conditions": 18, "episodes": 540, "successes": 451, "rate": 83.51851851851852, "official_same_subset": 88.44444444444444, "gap_pp": -4.925925925925924, "old24k_same_subset": 84.25925925925927, "delta_vs_24k_pp": -0.7407407407407476}, "demo_randomized": {"conditions": 17, "episodes": 510, "successes": 267, "rate": 52.35294117647059, "official_same_subset": 55.588235294117645, "gap_pp": -3.235294117647058, "old24k_same_subset": 51.96078431372548, "delta_vs_24k_pp": 0.39215686274510375}}
约1191/3000条推进，35完整/8运行/57待测。已完整1050条718成功=68.38，官方同35项72.49（−4.10pp），24k同35项68.57（−0.19pp），覆盖扩展后不再支持早期8项+2.92pp代表总体。Clean83.52 vs官方88.44、24k84.26；Random52.35 vs官方55.59、24k51.96。microwave Clean11/30=36.67比24k23.33改善13.33pp但距官方82仍大；handover_block Random7/30=23.33较24k60下降36.67pp；hammer Random8/30与24k相同；dustbin Random26.67较24k40下降13.33pp。该阶段不能推断warmup方案因果或全100条件最终结论，任务样本/训练预算/RNG等限制保留。两次显存保护触发的重跑条件当前状态[["blocks_ranking_size", "demo_randomized", "complete"], ["stack_blocks_three", "demo_randomized", "complete"]]，没有最终失败；不把中断当policy失败。继续现有标准30×100队列。证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward20000_4090_all100_30ep_20261024T052041Z/progress_snapshot_20261024T1443.json。


## 2026-10-24T14:46:10.105258+00:00：full-backward20k 30条趋势队列终态

{"utc": "2026-10-24T14:46:10.105258+00:00", "status": "interrupted", "scope": "30episode trend; not standard100", "observed_episodes": 1203, "observed_successes": 822, "executed_rollouts_including_retries": 1203, "executed_successes_including_retries": 822, "completed_conditions": 35, "planned_episodes": 3000, "active_simulators": 0, "errors": 2, "episodes_per_hour_estimate": 191.80317488413036, "linear_remaining_hours": 9.368979429488487, "note": "linear ETA excludes task-mix correction; current short-task coverage is biased"}
证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward20000_4090_all100_30ep_20261024T052041Z；完整率按30实际分母，不与旧实验拼episodes；详见专用LIVE。


## 2026-10-24T14:50:55.882374+00:00：用户中止20k，切换full-backward24k全100条件×30

权重：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_24000.safetensors`；SHA256 `d9ed4954c2088f014651d15cee769426c83de6f12674a5f3b32be093242e040d`。精确step24000硬链接单独固定目录，config/stats/contract/tokenizer自本run保留；RAM副本源文件和目标再次流式SHA一致、源inode/size/mtime稳定。不是HF下载，不消耗镜像/VPN流量。

范围：50任务×Clean/Randomized×30条=100条件3000条；不存视频。完整Sidecar三tap7/15/23；cache .99/max3开启、compile=false，10 denoise、32chunk。继续官方原生expert筛seed、seed0/unseen、RT32spp/OIDN/near.1、原动作预算与成功判据。同官方checkout/隔离Conda，不改模型/训练/数据/上游或原DiT4DiT流程。任务顺序沿用20k/前24k，四短锚点双域后长原预算交错共享队列，均完整30才计SR。

L20 GPU0–7已各启动一份新24k模型，25900–25972共24个独占会话；重复两fixture各2次、跨副本逐元素一致、本机24路/4090隧道24路真实PNG/state→20D动作验收通过后才派发。first-ready副本原子建立本checkpoint规范参考，避免固定单卡延迟拖住其他副本，不与20k或其他24k输出强求相等。旧20k八份服务（含已完成L20对照专用GPU7）均核对身份后停止，新端口/token/权重SHA隔离。

4090尽量并行：当前caps1/2/1/1/3/1/1/0，容量允许最多10路；单路启动预算7680MiB+1024MiB卡余量、信用至首条完整policy episode后释放。GPU1/4 live守护2s采样，低于2048MiB或发生render cannot create buffer回退1/2路，仅停最新自有额外sim，原partial保留不混SR，基础设施最多3attempt。GPU7当前free3730MiB不放sim；GPU6当前free11943MiB先1路，其他作业不停止。保留45min无episode或action进展看门狗及ANSI剥离、独占token/slot锁和渲染PCI断言。

旧20k已用户中止：35/100完整，1050条718成功=68.38%，当前子集官方72.49、isolation24k68.57；全部旧原始日志、35完整结果和中断partial保留，不能称20k已完成全量。stop目录由outputs/robotwin_eval/fullbackward20000_stop_current.json指向。旧20k状态interrupted，不并入新24k正式分母。

新输出：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward24000_4090_all100_30ep_20261024T144937Z`；模型部署：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward_warmup2k24000_deploy_20261024T144703Z`。配置/代码差异与官方commit见source_evidence.json；pointer fullbackward24000_30ep_current.json。live_progress.json用于进度，comparison.json只对完整同任务子集对照官方与之前isolation24k，结束原生复验后生成final_results.json/CSV。此刻模型加载/验收进行中，尚无新24k真实rollout结果。

比较边界：本次训练步数都24k，但warmup机制不同，仍须核对其他训练超参/seed；相同30样本数量不保证accepted seed/指令/RNG逐条一致。保持历史near差异、cache近似与硬件/推理负载限制，不将阶段子集外推全量或宣称单因素因果。



## 2026-10-24T14:54:51.196032+00:00：full-backward24k八模型/24端口验收通过，4090九路启动

权重SHA d9ed4954c2088f014651d15cee769426c83de6f12674a5f3b32be093242e040d，8卡全部完整Sidecar/cache开启compile关闭，两个fixture每卡各重复2次并跨卡逐元素一致；本机/4090全部24路真实RGB/state→有限20D动作通过。dispatcher PID2037254，开始2026-10-24T14:53:33.319768+00:00，首批9路，不同模型优先。live caps1/2/1/1/3/1/1/0，GPU4第3路等待启动信用与显存，2GiB/buffer-error资源守护已运行，无占用未知作业。此记录仅队列启动，尚非真实rollout成功率。

training_budget_config_comparison.json核对：new full-backward24k与isolation24k batch32/accum2/lr1e-4/epochs10一致；新backbone_warmup_updates2000/isolation0，旧isolation2000，无架构/数据修改。相同step不自动证明其他训练/场景/RNG完全受控，结果按相同完整条件比较。旧20k已中止，35项1050条718成功保留且不混入本轮。证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward_warmup2k24000_deploy_20261024T144703Z/final_deployment_acceptance.json和/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward24000_4090_all100_30ep_20261024T144937Z。下一步确认真实仿真动作/首条终态后继续后台全部100条件。


## 2026-10-24T14:57:56.349922+00:00：新full-backward24k十路已启动，至少九路真实动作推进

八份L20模型全部ready（GPU0–7），4090现10路，9路均已出现真实policy动作，新增第10路仍在初始化；click_alarmclock双域已有成功终态，非ping/fixture冒充rollout。video关闭、new24k SHA、unseen/EEF和独占端口逐路通过；已出render证据项GPU PCI均通过，infra errors=0。每卡仿真数{"0": 1, "1": 2, "2": 1, "3": 1, "4": 3, "5": 1, "6": 1, "7": 0}，GPU4首条终态释放启动信用后已自动补至3路。资源2GiB/buffer-error保护运行。证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward24000_4090_all100_30ep_20261024T144937Z/first_rollout_acceptance.json；专用LIVE/后台comparison仅完整同条件30条计SR。旧20k已中止/结果保留，新24k独立3000条继续；相同step24000、batch32/accum2配置已核对。
