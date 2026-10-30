# Modification log

## 2026-10-24 — 简化真机client为继承原infer.py的单文件适配

用户认为token/config/几何模块拆分过多，要求按原infer.py直接替换为绝对EE。新增
benchmarks/real_robot/infer_openwam.py继承原Infer，复用现有ROS/FK/IK资源；仅补当前EEF10
输入、绝对TCP变换与HTTP响应适配，保留必要运动门槛，不调用旧相机delta apply或reset。
单文件参数集中顶部，私有交付版本嵌入凭证且权限600，项目源文件不含秘密；不改服务器认证，
不修改原RobotWorkspace或训练核心。3新测试+6旧测试通过，硬件未验收，公网502映射待确认。

## 2026-10-24T08:57Z — ManiSkill绝对EEF Flask server / RobotWorkspace client

用户要求沿用UniVLA Flask+真机client部署方式，但ManiSkill checkpoint输出绝对base-frame
EEF10，不能复用相机系delta累加。新增benchmarks/real_robot服务、共用启动脚本、独立几何模块、
ROS2 dry-run默认client及未标定配置模板；复用现有模型loader/normalizer/L形预处理/engine，
不修改模型、训练、数据或现有server。仅Sidecar33880部署GPU7；baseline21180仅启动选项。
Flask依赖安装独立outputs target，不影响训练环境；认证、busy拒绝排队、32×10契约与七轴/IK
安全门槛测试6项通过。两原生ManiSkill fixture认证HTTP推理及重复一致性通过，未做硬件运动。
标定/工具TCP/实测夹爪/ROS/碰撞场景/急停及sim-to-real有效性仍待真机验证，详见专用部署文档。

## 2026-10-23T14:42Z — checkpoint24000真实写回范数与Action对照

用户要求区分gate参数、gate相乘残差与真实hidden变化。仅扩展WriteProbe的测量输出：
保留旧ratio/pre_addition_ratio，追加原生dtype残差L2/RMS、hidden/projection L2、
真实delta及变化元素比例，不改生产write表达式。探针7项及BF16舍入新增1项测试通过。
独立GPU5/6流程复用旧25GT窗口与32Action FM窗口，固定输入及CPU/CUDA RNG，
不做optimizer更新或修改训练。产物与边界见docs/SIDECAR24000_TRUE_WRITE_RATIO_20261023.md。

## 2026-10-23T08:43Z — isolation2K step24000复用诊断入口

用户要求与旧42960诊断对照。仅诊断工具：sidecar_gt_diagnostics支持plan可选GPU及
scales、summary按实际倍率汇总；sidecar_gradient32支持plan可选checkpoint_dir，
旧默认保持。新增Full/Off汇总回归，联合11项通过。独立快照与bootstrap在
sidecar_isolation2k_step24000_diagnostics_20261023，GPU5/6启动，GT25窗口及梯度32batch
复用旧输入，aux使用新checkpoint真实.5。不改模型/训练/数据核心，不写回权重。
结果与限制见docs/SIDECAR_ISOLATION2K_24000_DIAGNOSTICS_20261023.md；加载不当作效果。

## 2026-10-16T04:20Z — 用户要求扩展多GPU标准评估

双域adjust_bottle smoke真实Clean1/1、Random1/1通过；用户随后要求尽快、多GPU并行。新增serve_parallel.py（16个独立PolicyServer/WAMPolicy缓存、共享原engine、predict互斥，仅sync）和parallel_eval.py（100个原生task/split动态作业、17 workers、每项100条、严格coverage/失败保留），standard_eval增加共享pool端口/PID/config/slot/token校验与并发FD扫描容错。复用原checkpoint/推理参数/原生专家seed筛选/RT/成功判据/400等各任务预算，不改模型、训练、数据；GPU0/7模型，仿真跨0–7，不终止其他GPU作业。

此前已完成ffmpeg系统binary PATH shim（不覆盖旧Conda）、独立仿真环境复制与仅副本pth修正、五个本机构建cuRobo.so安装（旧.so hash备份/构建清单保留）。GPU0共享模型加载与完整10000条队列已启动，但不等于所有worker开始rollout或全量分数完成。详细证据和未完成门槛见docs/ROBOTWIN_STANDARD_EVALUATION_20261016.md。

## 2026-10-16T03:46:00Z — 独立 RoboTwin 标准评估链路

用户已确认仅修改评测层，并进一步要求复制到OpenWAM、不影响旧DiT4DiT评测。

- `eval_policy_wrapper.py`：ROBOTWIN_TEST_NUM除原eval_policy上限外，同步适配上游main唯一的test_num=100赋值，保证原生_result.txt使用实际分母。仅在显式设置时内存AST替换，源文件不写；上游结构不符则提前拒绝，默认不设置时原main保持。
- 新增`standard_eval.py`：复用single_eval/native evaluator，固定seed0/unseen/RT/原步数预算，隔离run目录；记录权重实测SHA256、配置/adapter/runtime hash及diff、环境版本；新服务使用readiness token，复用服务核对PID/端口所有权/checkpoint/GPU；验收完整连续episode计数、seed、原生结果和exit状态，失败episode仍计分，缺失或分母错误不通过。不完整结果不算全量50任务成绩。
- 新增`copy_robotwin_runtime.py`：独立复制现有runtime，资产软链接物化为普通文件，不共享inode，源/config逐文件SHA256、bulk assets逐文件size并记录清单；不复制旧data/eval_result。原RoboTwin源码、环境、资产不修改。另启动仿真Conda环境的普通文件副本`.venv-robotwin`，完成后仅修复副本editable cuRobo路径。
- `.gitignore`增加本地`.external/`和`.venv-robotwin/`；新增评估说明与13项CPU验收（首版12项已通过，追加NaN拒绝后与原18项联合复验中）。没有模型/训练策略/训练数据变化，不新增研究效果结论。
- 本机为L20/GLIBC2.31，cuRobo预编译.so不兼容；本地toolkit12.4、torch2.7.1+cu128的独立缓存编译仍待验收，不开启planner fallback。模型服务完整权重加载、原生normalizer和80D→20D已确认，尚无闭环结果。

## 2026-10-13 UTC — OpenWAM-alpha / EA-RSIGv3 bootstrap

用户要求：独立同级目录搭建 OpenWAM-alpha，迁移现有 EA-RSIGv3 功能与逻辑，复制 DiT4DiT AGENTS.md，HF大文件只走镜像。

- 克隆官方 OpenWAM，upstream commit 4d7f642b841913ea3c176010b692b1de3d5ef180。
- 逐字复制 DiT4DiT/AGENTS.md；保留原隐患快照并建立本地隐患追踪。
- 创建独立 .venv，不修改现有 DiT4DiT 或 4DWAM 环境。
- 核对 checkpoint 架构、原 EA 模块和官方训练/部署接口后作新增扩展；验证与限制记录于专门分析文档。

## 2026-10-13 UTC — 新增 EA joint-self-attention 扩展

理由：Alpha joint hidden可能含future/action信息，现有EA需部署观测-only语义；native14D/40步与官方80D/32步合同不同，不能静默误配。

- 新增注册variant joint_self_attn_ea_rsig_v3，保留官方基线variant；独立no_grad clean观测第18层、原EA core、动作context追加、训练-only future残差、显式aux loss及graph清理。
- 原EA模块仅重定位loss分组helper；复用已有Wan几何bridge，保留许可。
- 新增schema5 reader读取源parquet/三视频，不改源数据；labels与deployable context分离，验收horizon/anchors/columns/action semantics，输出本地自有stats。
- 新增model/data配置与offline训练入口，foundation warmstart仅放行EA新增键和4个native政策头形状适配，EA重载仍strict。
- 部署policy/engine传递当前EEF/arm mask并拒绝oracle标签；未启用编译/cache或CFG分支。
- 新增镜像下载器，用户明确批准mirror返回Xet CDN直连，trust_env=False禁止VPN代理；revision/size/hash验收且可续传。
- 增加real tiny Wan gradient/leakage/checkpoint回归；尚在运行，无完整预训练/闭环成功结论。详见迁移分析与setup日志。

2026-10-13 UTC 读取合同复核：原 DiT4DiT qpos/逐步delta state/action gripper 使用 >0.5 的 binary 0/1，而 EEF context 不做此变换。新reader初稿采用连续min-max，已在验收前修正为旧规则：policy夹爪二值0/1、identity部署stats、proprio二值化及action输出投影，EEF context仍原始读数；改新stats缓存身份，保留旧调试产物。native数组顺序[L6,Lgrip,R6,Rgrip]与DiT内部重排不同但读/训/部署一致且不改物理语义。源Parquet与标签未改。

2026-10-13 独立环境安装完成，锁定torch2.7.1+cu128，pip check通过；49 focused CPU项通过、2官方GPU集成跳过。补充EA runtime dtype/device传播覆盖，否则新EA child未被原backbone-only设备分发覆盖。真实50episode/7134window metadata和3sample图像/normalizer初检通过；binary夹爪修正后专项测试和预检仍在进行。完整权重下载及真实foundation验收未完成。

2026-10-13 增加Aloha qpos评估client的opt-in ea_fk：复用已验证FK实现，复制小型URDF/config/训练合同，核对URDF SHA与kinematics version；由当前native drive qpos产生20D EEF与2D arm mask，不读取object/goal oracle。默认官方client不变；独立benchmark helper避免导入训练模型/DeepSpeed依赖到仿真client。source-row FK与payload回归已启动，尚不构成闭环证据。binary专项首轮仅测试误假设proprio返回2D，已改为接受原API的1D/2D张量，不改runtime接口。

2026-10-13 数据/客户端文本合同复核：新reader的原始instruction在内存中使用官方RoboTwin format_prompt_for_inference统一包装，与既有benchmark client逐字一致；源tasks.jsonl不改。避免初稿raw训练prompt与官方client包装prompt不一致。binary修正后真实source预检已通过（ea_data_binary_preflight.json），EEF与130D targets逐元素保持源当前行；尚非全量视频审计。

2026-10-13 环境完整import check通过；binary预检通过，当前行EEF/target逐元素相等。原依赖/基线49项、binary修正后model7项与source-FK项已通过，client mock缺ping已修正待重跑。添加EA部署薄入口，明确禁用cache/compile默认路径并沿用上游checkpoint loading。下载后安排一次offline真实foundation CPU AdamW短测，16threads/160GiB内存门；下载与排队本身不计验收成功。

2026-10-13 最新证据：完整镜像主权重和8个上游文件hash/size通过；真实foundation加载2085 tensors、4个native政策头重置、新EA73 tensors。完整CPU FP32 AdamW一步总loss3.9434447，video/action/EA梯度有限，视频alpha由0更新到-9.9988e-5。client/FK及旧接口12 passed。完整模型保存/官方部署严格重载/两步无oracle推理正在测试，不称闭环成功。

2026-10-13 补充交付完整性：上游.gitignore忽略AGENTS.md和整个assets目录，显式放行原样AGENTS及assets/ea_robotwin小型URDF/config/合同快照，避免后续代码搬迁遗漏部署FK依赖。Aloha helper已拒绝actual-joint标签冒充drive-target；增加该拒绝分支专项检查。完整往返脚本按官方loader真实签名移除不支持的dtype参数，保存前转BF16匹配部署精度；只终止自有尚未执行该旧调用的测试进程，使用全新v2输出目录重跑。

2026-10-13 新增部署边界回归：使用真实ObsPreprocessor/WAMPolicy条件构造/JointInferenceEngine，检查三相机canvas位置、20D EEF与2D arm mask传递、5视频帧/40动作时间合同，以及oracle/空arm/nonbinary arm/nonfinite EEF/missing EEF拒绝；仅mock模型generate与schedule，模型本身仍由完整foundation往返独立验收。新FK拒绝合同及源行/payload专项3 passed。

2026-10-13 单步官方分布式验收启动：现场nvidia-smi确认5/6空闲，CUDA_VISIBLE_DEVICES=5,6/NPROC_PER_NODE=2，官方torchrun->Accelerate->DeepSpeed，BF16/ZeRO2，CPU初始化、真实adjust_bottle、batch1/rank、seed42、max_steps1、save_steps1、WANDB禁用，输出outputs/setup/official_gpu_smoke，仅短测不启动长期训练。CPU完整往返继续运行；两项完成需终态及实际报告/checkpoint证明。

2026-10-13 完整往返目录审计发现短测脚本遗漏checkpoint finetune路径的tokenizer relay：Wan save_assets仅从可读model_path复制，而foundation配置保留上游占位路径。正式trainer已有copy_ckpt_artifacts，短测复用该函数，不改共享加载逻辑。向正在测试的v2自有输出目录补齐相同tokenizer资产，保留现有真实更新权重；重载终态仍须实际验证。

2026-10-13 v2重载先于tokenizer补齐已经以FileNotFoundError终止，保留失败日志与已更新完整权重，未重跑AdamW。新增独立smoke_ea_deploy.py复用官方严格loader、merge_deploy_cfg和JointInferenceEngine，允许对任一已保存EA run离线重复部署检查；从真实当前源行提供EEF/arm/proprio且不传oracle，检查40x14有限值与binary夹爪。补齐tokenizer后用此独立入口复验。部署adapter回归6 passed，覆盖实际preprocessor/policy/engine边界。

2026-10-13 官方双卡BF16/ZeRO2首轮在AdamW更新OOM：前向/反向已执行，但PyTorch默认foreach在扁平optimizer shard上申请11.27GiB临时sqrt，GPU只余9.27GiB。核对安装版torch/optim/adam.py后，单纯foreach=false的single-tensor路径仍会为该巨大flat shard分配sqrt。新增可选training.adam_foreach/adam_fused，默认null保留官方其他variant行为；EA入口显式foreach=false/fused=true，使用PyTorch原生fused AdamW就地核函数避免分母大临时量，不改模型、AdamW算法、超参或优化器state精度。保留首轮失败日志，在GPU5/6确认释放后重测。

2026-10-13 部署真实开环已生成成功报告ea_full_deploy_v2.json：BF16完整strict reload、官方JointInferenceEngine、当前观测无oracle、两步生成40x14有限动作、夹爪0/1、更新后的alpha精确保留。RoboTwin源码get_left/right_arm_jointState确认为get_drive_target，joint_action.vector拼接该值；新FK client与源训练drive-target输入语义一致。推理成功不构成闭环或速度结论。

2026-10-13 用户询问适配是否保证实验效果：明确不作保证。新增局部数值保真检查，在相同native14头/权重/输入下关闭EA conditioning/residual，要求新增循环的视频/动作预测逐元素等于未改官方joint-forward；仅验证工程循环未引入额外变化，不证明80D->14D、帧/布局或EA融合的效果等价。效果需同数据/预算、多seed、闭环对照，单步与tiny检查不替代实验。

2026-10-13 局部原forward parity 1 passed：同native14头/权重/输入、tiny Wan关闭EA condition/residual，新循环的视频和动作预测逐元素等于原joint-forward。源EA当前全文仅helper import路径不同、原helper函数一致、AGENTS一致；最新比较在source_contract_comparison.json。审计脚本最初误用仓库外层路径和相对import替换字符串，已依据真实package路径和diff修正，未修改EA模块。

2026-10-13 最终工程验证：官方双卡BF16/ZeRO2/完整6048.9M可训练参数/fused AdamW更新一步、最终保存、进程exit0；loss约1.3338。输出official_gpu_smoke_fused/2026-10-13_04-16-38，2162 tensors/24,869,359,726 bytes，alpha=+0.0001001358。此官方产物又在物理GPU5经strict loader+JointInferenceEngine两步无oracle推理通过，40x14有限值、夹爪0/1、alpha精确保持，exit0，测试GPU释放。source/raw数据未修改；README/分析/隐患更新工程完成与性能未知边界。记录official_gpu_train_verification.json及ea_official_gpu_deploy.json，不声称闭环、长期稳定或效果等价。

2026-10-13 训练耗时问答：核对双卡首步9.76秒/global batch2，排除checkpoint保存混入；交叉核对论文附录B的Clean2Random 10,740更新/Full118,655更新，均global batch256。现场汇总现有Clean50任务/2500episode/557651窗口，5epochs在当前microbatch吞吐下线性外推约157.5天，首步不是稳态排期证据。公式、官方与当前样本暴露量区别、micro/optimizer-step计数及测量范围记入docs/2026-10-13_OpenWAM训练吞吐与预算估算.md；未启动新训练或改变核心代码。

2026-10-13 UTC：用户授权 GPU5/6 batch 容量与更新时间实测。新增 scripts/benchmark_ea_batch.py，仅对官方 trainer.log_step 收集逐步时间/显存/损失，复用原始训练循环、BF16/ZeRO2/完整模型/fused AdamW，grad_accum=1、save_steps=0；数据只读 adjust_bottle，输出 outputs/batch_benchmark。无核心架构/优化规则变更；最大 batch 与稳态吞吐须等待真实更新/OOM 边界，不以预估代替。

2026-10-13 UTC：连续容量探测发现原单优化器组在batch1和batch2的第2次更新均ZeRO2 FP32 gradient partition转换OOM（11.27GiB），撤销对一步smoke可持续性的暗示。容量测试使用现成action_lr=video_lr=learning_rate=1e-4配置生成等值三优化器组，缩小临时分片；不改模型/loss/每参数AdamW超参数，待实测。测试脚本新增同进程真实batch sweep；调用安装版DeepSpeed.set_train_micro_batch_size，保持grad_accum1，rank采用同seed全局随机索引的互斥分片，官方训练循环/更新路径不变。输出sweep_groups；仅测试入口，不改生产trainer。

2026-10-13 UTC：三等值组仍在第2步9.31GiB转换申请OOM，reserved未分配约10GiB、allocated约67.4GiB；仅粗分组不足。benchmark入口增加按200M元素有界优化器组，参数顺序/覆盖/唯一性与lr/betas/wd/eps/fused/foreach逐项assert不变；ZeRO最大临时梯度转换约<=0.4GiB，AdamW算法不变，生产核心无改动。新测sweep_bounded启用PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，逐档实际batch；未通过前不宣布持续训练问题解决。

2026-10-14 UTC：按用户指定OpenWAM-alpha_EA-RSIG-Lite_2026-10-14.md重建独立Lite路径；授权范围为实现准备和CPU验证，禁止本轮实际四卡调试。新增ea_rsig_lite架构/模块/reader，保留官方80D policy+state head和33源帧/9视频帧/32 action、联合FM原目标与注意力mask；当前帧VAE patchify一次+轻量投影，不执行观测DiT，不插入主序列token，不调用Video Residual。15x512 EA latent只经cross-attn+zero-init out在mixed block0之后给Action hidden加残差。辅助loss名义权重.025/.01/.05，支持warmup/joint/action_only及A/B/C；warmup冻结原主干，B joint监督可达共享patch embedding，性能/硬件均待验证。

数据契约审计发现原始RoboTwin-v6 endpose是wxyz且相对oracle TCP沿local X少0.12m；官方reader接口按xyzw解释。新增LiteRoboTwinSource仅在读取边界转换四元数，模型内仍官方col-based Rot6D+绝对EEF；TCP context由当前观测endpose+已知0.12m刚性偏移生成，不取oracle EEF作为模型输入。新增部署配置显式wxyz，client在边界与控制器相互转换，原默认xyzw路径不变；尚需真实控制闭环验证。Plan按raw oracle重新生成8/16/32，每view bbox按自身图像尺寸归一化，延用角色ID/valid/无可观测Goal屏蔽，source HDF5只读，sidecars与EEF统计输出outputs/ea_lite/data。

四卡准备：新增Hydra配置/默认CHECK_ONLY启动器、独占四A100资源门槛、诊断和保存后计划退出脚本；max_steps按micro计数，默认acc8/global32/20opt。keep_final_resume_state仅显式开启时保留full resume状态，原默认清理行为不变，支持未来同max_steps计划两次恢复。trainer/reduce_step_metrics额外透传EA loss与mask/门控诊断，Baseline原有四指标运算保持。新增CPU真实tiny Wan防错/梯度/保存恢复测试与真实Foundation CPU载入检查；运行结果未出前不记通过。已终止上一日自己挂起的旧双卡benchmark PID359451/403897/403908，保留所有失败日志，未操作其他作业。

2026-10-14 UTC Lite 准备验收：Foundation CPU实际完整加载2089 tensors、85新EA tensors、80D heads零重置，总参数12420900941、C warmup trainable13687328；50任务2500episode557651源帧的原始Plan8/16/32与Clean EEF统计生成完成，官方窗口555151，所有sidecar/source元数据及stats hash预检通过，实际首/中/末三窗口和相机bbox网格检查通过。最新CPU回归62 passed/1 anyio热导入warning（20.70s），覆盖tiny Wan checkpoint开关/FP32与BF16 identity/因果/当前条件一致/两步梯度/两次CPU full-state save-restore及旧V3兼容。部署YAML已按实际OmegaConf直接加载修正为独立配置，action窗口键num_frames=33；新增未来Lite离线推理检查脚本，未执行GPU推理。A/B/C实际CHECK_ONLY config resolve通过；官方joint架构/backbone与原配置文件无diff，AGENTS与源逐字相同。新增专门准备文档并更新思考与隐患，明确新旧合同与CPU/GPU证据边界。本轮无任何四卡/GPU训练或推理；容量、吞吐、全模型部署与闭环/方法效果保留为待验收。

2026-10-14 UTC 跨服务器交接（用户明确要求追加到原方案并准备脚本）：原方案第1–14节字节保持，末尾追加第15–22节，覆盖实现合同、旧OOM方案与更正、CPU证据、三类搬运资产、新机环境/路径/预检、四卡5→10→20恢复协议、单机多卡预算、部署与未通过门槛。修正install_ea_environment.sh默认Python/uv为可迁移命令，并用锁文件过滤旧editable绝对路径；新增relocate_ea_lite_manifest.py只重定向路径且保留源size/mtime与label/stats hash。原四卡入口默认不变，新增EA_NUM_GPUS/EA_GPU_NAME参数及单机multigpu薄入口，preflight按显式卡数/型号检查，仍默认CHECK_ONLY；不改变模型、loss或训练算法，未扩展为多机。新增package_ea_lite_handoff.py打包工作区、sidecars/统计和小证据，不含权重/raw/venv/.git/Cosmos gitlinks，输出SHA清单。13项纯CPU迁移/启动测试通过（6.18s），原模型62项CPU结果作为历史证据保留；无GPU分配/训练，也未在目标服务器实际安装或通信验证。

2026-10-14 UTC 启动加速：按用户要求将反复数据发现/审计移出日常启动。新增cache_ea_lite_startup.py离线复用已验收manifest/labels/stats，保存指令/scene/图像尺寸为manifest.ready.json（50tasks约2.8MiB），去掉startup_metadata后与原manifest完全相等。官方RoboTwinDataset仅增加optional prepared_metadata（默认None仍原路径），Lite读取预备长度与元数据，跳过启动全量glob/HDF5探测与指令扫描；窗口、动作、图像、normalization/masks/prompt计算不改。prepared数据首次访问每条轨迹检查source size/mtime，sidecar按需hash。preflight新增full/fast，独立审计默认full，训练入口默认EA_DATA_CHECK=fast和manifest.ready.json，快检0.343s；缓存缺失失败而非自动生成。19项CPU启动/迁移测试通过（含冷依赖导入553.40s），官方reader32 passed in3.70s/1模型集成项未重跑。只改数据初始化/校验时机，无GPU测试、无端到端启动提速量化；主文档追加第23节，更新完整交接包，保留旧包和历史证据。


## 2026-10-15 02:26:34 UTC — 空闲四卡Lite大batch容量测量

用户授权GPU0–3从bs4向上短测并跳过重复数据预检。现场确认四张A100-80GB各仅4MiB占用/81150MiB空闲；先启动现有debug_ea_lite.py的C joint真实全参bs4、accum1、20次AdamW更新、save0，复用manifest.ready.json，跳过独立preflight，不修改模型/loss/参数组。

为避免每档反复加载完整模型，复用benchmark_ea_batch.py已有DeepSpeed动态batch扫描，新增可选EA_BENCH_BATCHES和EA_BENCH_STEPS_PER_BATCH控制起点/档位/次数；新增可选EA_BENCH_PRODUCTION_LOADER=1，调用原trainer.build_dataloader与Accelerate分片、真实worker预取，每档真实batch长度断言。原随机同步reader sweep作为未设置新开关时的历史路径保留；不启用有界参数组，不改AdamW超参或主训练循环。语法编译通过；新的扫描路径与容量/速度仍待GPU真实更新验证，不把导入或首步当通过。输出outputs/ea_lite/capacity_20261015，其他卡和其他作业不操作。


## 2026-10-15T02:54:48.732039+00:00 — 用户授权Clean50 Lite C joint五epoch正式训练

用户明确要求GPU0–3、bs64、acc1/global256、Clean全量5epochs，并将checkpoint间隔指定为2000updates。仅通过CLI覆盖training.num_epochs=5/max_steps=null/save_steps=2000，保留原EA权重.025/.01/.05、video/action LR1e-5/base1e-4、BF16/ZeRO2/fused AdamW/checkpointing、seed42及关闭color jitter。独立数据预检跳过；manifest.ready与reader原检查不变。用户指令对本run覆盖旧“仅短测、不长期训练”的范围限制。

使用现有benchmark_ea_batch.py的非sweep路径（SWEEP=0、GROUP_ELEMENTS空），官方trainer原循环+逐步loss/显存JSON，不挂额外梯度hooks，不启用有界optimizer组；nohup后台torchrun PID823163，输出outputs/ea_lite/clean50_C_joint_bs64_5epochs_20261015。每2k保存权重+full-state，保留最近2组；正常结束另存最终权重，保留已有full-state。启动阶段尚不计训练成功；bs64短测显存余量极小，不擅自降batch/改loss或中止用户正式训练。


2026-10-15T02:56:46.269427+00:00 正式训练启动修正：初次nohup PID823163在命令返回后消失、日志空，未计任何更新；改为新输出目录内的launch_tmux.sh+tmux独立会话ea_clean50_bs64_20261015，PID825005，二次独立ps确认存活。保留失败记录；参数为用户指定5epoch、bs64、acc1、save2000，原模型/loss/data不变。


### 2026-10-15T03:20:40.433424+00:00：用户授权纠正正式SFT学习率并开启ColorJitter

用户先授权将本地自行降低的骨干学习率纠正回官方1e-4，随后明确要求开启ColorJitter。修改configs/train_ea_lite_4gpu.yaml action_lr/video_lr为1e-4；保留三个等学习率优化器组，避免历史单组ZeRO2临时显存问题。configs/dataloader/ea_rsig_lite.yaml enabled改true，幅度0.2/0.2/0.2/0，复用官方reader clip一致增强实现，无核心逻辑改动。旧配置备份在新输出目录*.before.yaml，旧run日志/指标保留。新run为outputs/ea_lite/clean50_C_joint_bs64_5epochs_lr1e4_jitter_20261015，从同一Foundation重新初始化，bs64/acc1/global256/5epochs/save2000/原EA权重保持，prepared启动且跳过独立重复预检。已解析配置，真实更新与新增强下容量/速度待验证；旧run未到2000步无权重checkpoint，不能把日志保留称为权重保存。增加增强不改变窗口数或10845调度更新，可能增加CPU开销。下一步结束旧run、tmux持久启动并确认四rank真实更新。


### 2026-10-15T03:54:06.257243+00:00：用户授权Baseline跨集群训练包与Slurm流程

目标/home/mingfeihangroup/xiaoming，incoming上传、projects/OpenWAM解压、ckpts存Foundation；分区mhan0035、单节点4×A100，不设置time。新增configs/train_alpha_baseline_clean50.yaml与scripts/cluster下baseline launcher、sbatch、一次性路径准备、复用旧installer的环境入口、独立打包器，无核心模型/数据逻辑改动。Baseline为官方joint_self_attn/80D，数据沿用现有wxyz适配+prepared reader、auxiliary=false，因此不添加EA模块/损失/输入；bs64/acc1/global256/5epochs/LR1e-4/ColorJitter开启/save2000、fused AdamW/BF16/ZeRO2/等LR三组与当前配对预算。源数据/当前训练不改写。

用户后续明确要求保留EA标签，数据包包含2500个Clean HDF5逻辑35,943,837,132 bytes、全部EA labels/stats/ready manifests及指令/scene信息，不包含MP4/PKL回放。代码包排除venv/git/log/历史权重，Foundation单独包。打包器dereference并仅写regular-file，PAX精确纳秒mtime，流式每文件和归档SHA256；已用软链接输入完成GNU tar解压roundtrip，实字节/无link成员/纳秒时间/hash通过。Hydra参数解析与Shell语法检查执行中/结果见交付记录。目标gnode06目前用户报告drng，可能排队，目标CUDA/NCCL/实际训练未验证；bs64按80GB档，40GB需另配。完整说明docs/BASELINE_CLUSTER_TRANSFER_20261015.md。

2026-10-15T04:04:37.742739+00:00：按用户要求通过hf-mirror下载OpenWAM-Alpha-Sim-RoboTwin-Clean2Random，放在Foundation并列目录。revision 9bb966823547bc959800b45e9bb4005e9c2a3db3，9文件均通过上游大小/哈希验证；主权重24813767464 bytes、SHA256 2cab63b550ff45a52fbde4adc40ee2bdb7d4bafe5fa0dd6879e41e22ca2ff95f。复用下载器新增--repo与1–16并发分段续传；8连接直连镜像返回的获准CDN、requests.trust_env=false，保留旧下载前缀，最终完整SHA校验后清理本次临时分段。下载边界/字节拼接/Range拒绝共3测试通过。证据位于新目录mirror_inventory.json、download_verification.json、asset_verification.json；没有修改训练配置或加载模型。


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


### 2026-10-15T09:01:37.765669+00:00：目标集群ensurepip缺失的环境安装修复

用户在(base)执行cluster安装器，系统python3.10缺ensurepip，旧python -m venv失败并留下半成品。仅修改scripts/cluster/install_environment.sh：先找uv，缺工具时明确提示在活动环境安装uv；不完整.venv改名备份后用uv venv --no-python-downloads --seed创建，不依赖系统venv包；健康环境复用，单独缺pip时uv补pip，继续原锁定依赖安装主体。无模型/数据/训练配置变更、无sudo或系统包修改；本机正在训练的.venv未改动。真实/usr/bin/python3.10无ensurepip的隔离恢复验证通过，失败文件保留、pip/activate及UV_BIN交接通过；主体Torch安装用stub隔离，目标完整安装仍待用户执行。说明ENVIRONMENT_ENSUREPIP_FIX_20261015.md提供无需重传数据的命令。已交付原43GB大包保持不变，另提供小安装补丁。


### 2026-10-15T09:22:49Z：Hanhai CUDA module 与独立 Python 环境衔接

用户确认目标集群 source /etc/profile、module load cuda/12.8、nvcc 检查成功；旧 DiT4DiT Slurm 同样加载 cuda/12.8，但旧 cosmos Conda 不用于 OpenWAM。复用既有 OPENWAM_CLUSTER_SETUP 入口，新增 scripts/cluster/setup_hanhai.sh，在 /etc/profile 与 module 初始化期间关闭 nounset，之后恢复严格模式，并设置 PYTHONNOUSERSITE=1；baseline sbatch 与训练参数不改动，train_baseline.sh 仍选择项目 .venv。迁移文档提交示例显式传入 setup、gnode06 和 gpu:a100:4。PyTorch cu128 依赖的 nvidia-* Python 运行库保留，不因加载系统 CUDA module 手动卸载。Shell 语法检查通过；未在本机加载目标集群 module、未启动或停止训练，目标计算节点的驱动兼容性、CUDA 扩展编译和 NCCL/真实训练仍待提交验证。原迁移大包不改写，目标服务器需另行创建或同步新增 setup 文件。


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


### 2026-10-16T03:17:18.137160+00:00：按用户要求绘制当前 EA-RSIG-Lite 架构图

参考用户提供的docs/image.png与实际Lite bridge、V3 core及forward/loss实现，新增docs/figures/OpenWAM_alpha_EA_RSIG_Lite_architecture_20261016.svg及3600x2600 PNG、图注说明、绘图脚本与imagegen提示词记录。图中明确当前干净首帧VAE+patch输入、EA Ground/Relation/Progress/Plan、15x512 tokens、block0后的Action-only Q/KV交叉注意力及零初始化残差，以及训练专用130D标签/loss隔离。先用内置image_gen生成视觉参考，因连线歧义，最终使用独立SVG脚本固定拓扑；resvg-py仅安装在/tmp独立绘图环境，未修改训练.venv。原始docs/image.png保留。最终SVG XML和PNG尺寸/完整性检查通过，并人工查看连接方向；这只是现有实现的架构说明，不新增模型/数据/训练策略或训练结论，不启动或停止实验。


### 2026-10-16T08:37:31.209655+00:00：用户授权 Multi-Layer Relational Sidecar 独立框架搭建（进行中）

原因：旧 block0/clean patch Lite 保留供已有训练验证，新方案需要在原MoT block7/15/23后读取受保护当前帧，跨层维护15×512关系状态并分层写回Action。用户已明确授权实现和MyWAM备份，不另改旧训练或评估作业。

新增模块 relational_sidecar.py（模块与架构分别独立文件）、独立model/data/train/deploy配置、离线Aloha准备入口及20-update诊断启动器。原主干FM只增加默认不使用的显式 forward_fn 参数，EA输出在一次loss调用内局部返回，避免模型级缓存；旧Lite/module/launcher不覆盖。加入registry导入。标签从用户指定4350集合的raw/aloha重建，保持绝对EEF80D/32动作/9视频/8-16-32和wxyz边界，goal grounding同时mask goal_valid。运行时H×W直接对应L形画布；beta零、projection非零初始化；warmup/action_only/joint显式区分；compile/cache关闭。

改动范围含模型、辅助监督数据构建、配置、启动/部署和测试。实现决策及未完成验收见 docs/RELATIONAL_SIDECAR_IMPLEMENTATION_20261016.md 与原规划第9节。旧版代码本地压缩快照已完成，远端分批上传进行中；CPU结构测试、源数据重建进行中，当前不宣称通过或效果提升。四卡A100/ZeRO2容量吞吐、完整Foundation新模块训练与闭环尚待硬件验证。


### 2026-10-16T09:09:45.620044+00:00：MyWAM备份CI兼容性修复（独立分支）

用户要求检查推送后run failed。确认两次旧版备份CI的真实失败为107项lint债务及Transformers最新版/torch2.7.1的hf_storage收集错误。独立工作树只作等价排版/import清理、测试lambda等价def，并新增CPU-CI约束5.19.0/0.41.0/Ruff0.11.9，feature/fix分支也触发CI；不跳过原测试、不取消lint。源训练工作树和环境保持。Ruff通过，AST排除imports后仅测试lambda存在差异；远端复验待run结果。详见docs/CI_FAILURE_ANALYSIS_20261016.md。


### 2026-10-16 UTC：CI第二轮可迁移性修复
原因与证据：远端38878693936的两个失败。将旧Aloha FK配置从服务器绝对路径改为仓库相对路径，在共享AlohaEEF入口定位；增加异cwd验证。修正旧BF16测试fixture的Tensor类型判断。无策略数学/权重/动作合同改变，原始备份保留。详见docs/CI_FAILURE_ANALYSIS_20261016.md。


### 2026-10-16 UTC：新Sidecar数据和CPU链路验收、双卡短测启动
指定4350集合中Aloha2500轨迹离线准备完成，原始HDF5只读；50任务150真实增强reader窗口、EEF动作/状态往返和A/C数据一致通过。真实Foundation CPU继承2089 tensors/heads不重置。新Sidecar CPU用例通过，组合40pass/1旧fixture错误已修。现场5/6空闲A100两卡启动独立2更新warmup、完整状态保存短测，不改变正式4卡配置，不操作其他作业。设置与未验证范围见独立交接文档。


### 2026-10-16 UTC：Inductor非必要启动池
双卡首次启动现场可见每rank一个torch/_inductor/compile_worker/__main__.py --workers=32子进程，伴随GPFS I/O等待。读取当前torch async_compile.py源码确认TORCH_WARM_POOL=0只跳过import-time AsyncCompile.warm_pool；新Sidecar训练入口设置此变量，模型forward/loss/精度/优化器不改。shell语法及AST核对通过；当前已运行进程不会因此动态改变，不重启已有训练。未宣称消除全部冷启动时间；依赖导入和权重加载仍存在。GPU梯度更新/保存恢复仍待真实验证。


### 2026-10-16 UTC：启动I/O隔离验证
双卡两次真实更新、全量状态marker保存已通过，CPU importtime正在独立测量。原.venv约15GB，位于GPFS，计划复制到独立/dev/shm/openwam-sidecar-runtime-20261016.*目录作本机临时只读运行副本；不安装/升级/改写原环境，不复制原始HDF5。当前tmpfs空闲约482GB、主机可用内存约830GB、cgroup内存无有限配额，副本约15GB，非永久部署产物。新launcher用选定python -m torch.distributed.run代替torchrun shebang，确保OPENWAM_VENV实际生效；其余分布式参数不变。shell语法通过，实际新环境完整启动/恢复仍待验证，不能宣称提速结果。


### 2026-10-16T13:17:09.200360+00:00：用户授权新Sidecar四卡正式10epoch训练

使用共享盘.venv，GPU0–3，指定4350集合内Aloha50任务2500轨迹，prepared manifest不重建。joint全参，ColorJitter .2/.2/.2/0、BF16/ZeRO2/checkpointing/fused AdamW、base/action/video LR1e-4、seed42沿用wrist020正式方案。旧Lite bs64曾近80GB，本次新增三层模块采用bs32/acc2/global256，既有小区间训练不能保证新batch容量；不声称已验证joint最大batch。epochs10/max_steps null，save4000micro=2000updates，最近2组+最终完整状态保留。

新增独立配置、正式入口及launcher；原trainer/旧Lite启动器不改。入口不装grad hooks，只保存轻量per-rank指标。CPU权重与optimizer准备完成后、DeepSpeed大规模GPU迁移前四rank同步；rank0写handover/cpu_ready.json，至少90秒并等待四张卡均free>=70000MiB才放行。初始CUDA/NCCL仍会产生少量显存，不把此阶段称为完整GPU迁移；用户自行结束旧进程，本agent不发kill。门槛逻辑模拟验证了90秒与最后一张卡未释放时持续等待，Ruff/shell检查通过。

启动记录：outputs/sidecar/aloha50_C_joint_10epochs_20261016T131709Z/launch_record.json；tmux=sidecar10e_20261016T131709Z；日志=outputs/sidecar/aloha50_C_joint_10epochs_20261016T131709Z/train.log。当前准备启动，首次joint更新/显存吞吐/保存恢复和闭环仍待真实结果。

### 2026-10-17 UTC：用户授权 ManiSkill 六任务原版 baseline 10 epoch

目的：按用户要求，在指定20260922六任务集合训练原版OpenWAM baseline，每卡bs64/acc1/global256，GPU0–3，10epochs，绝对EEF，Foundation warm-start。先读20261016 Franka pipeline与docs/思考与隐患.md；源LeRobot action7是delta，不能当作绝对EEF，完整集合必须按derived manifest取补采aggregate轨迹。

独立新增openwam/maniskill_eef_contract.py、dataloader/maniskill_absolute.py、scripts/prepare_maniskill_baseline_data.py、train_maniskill_baseline.py、train_maniskill_baseline_4gpu.sh及两份maniskill配置；不改既有模型、trainer、RoboTwin reader或旧实验。仅新入口动态注册reader；log_step包裹只记录有限指标和峰值显存。基于原Panda CPU delta controller重建基坐标panda_hand_tcp绝对command target，xyz+rotation6D+grip占80D中0:10，其余mask false，无RoboTwin工具偏移。action[t]对应观测t到t+1，不把next-achieved pose充当command。构造输出outputs/maniskill_baseline/data_absolute_eef_20261017/，保留原HDF5只读直读RGB；存numeric NPZ/stats/带hash和源size/mtime的ready manifest，无EA输入/辅助loss。

训练沿用默认dual_system joint_self_attn、AdamW1e-4/.9/.95/wd.01、BF16/ZeRO2、gradient checkpointing、33原始帧/stride4/32commands、384x320、seed42、5%warmup/cosine/min.01。CPU初始化、等LR三参数组和fused Adam沿用已验证A100显存实现；不改变学习率。用户要求仓库默认配置，本版ColorJitter遵从仓库默认关闭（不同于先前RoboTwin显式开启实验）。保存2000updates、最近2组及最终完整状态。新增check_maniskill_controller.py、check_maniskill_baseline_data.py、check_maniskill_expert_replay.py进行实际数据/控制合同验证。当前构造和检查尚在进行，GPU更新及完整物理回放结果待后续如实追加。

2026-10-17T05:09Z追加：6000轨迹/542009窗口准备成功，原记录success各任务1000/1000。54真实窗口及时序/单臂mask/RGB源像素/尾部padding、64样本shape检查通过（34.46秒）；实际PandaWristCam控制器18轨迹/1614步复核通过。完整物理replay仍被容器Vulkan/URDF渲染材质阻塞，明确留作闭环评估前门槛，不当成回放成功。05:09:08Z已在持久tmux启动GPU0–3四卡torchrun，配置bs64/acc1/global256/10epoch；启动门槛通过，正加载模型。run=outputs/maniskill_baseline/six_task_bs64_10epochs_20261017T050252Z，数据/配置代码hash在launch_record.json。后续真实更新结果继续记录专门实验文档。

2026-10-17T05:20Z启动验收追加：四rank均完成8次bs64/acc1/global256真实更新，全部loss/grad norm有限且grad norm>0，无OOM；总训练预算21180updates/10epochs。原版baseline6021.2M trainable、Foundation结构0override。除头两步初期均速12.94s/update；NVML最大80963MiB且GPU0free190MiB，保留用户指定batch但不宣称长期/保存稳定。完整证据outputs/maniskill_baseline/startup_verification.json及专门实验文档；run持续运行，未操作其他实验。


## 2026-10-18T06:25:13.393341+00:00 — 双轨RoboTwin复现实验准备

用户明确授权独立官方源码/资产/Conda环境重建及Sidecar20k在现有环境完整评测，GPU按0–3/4–7分组；不修改模型/训练/数据。复用现有standard/parallel/serve入口，新增显式deploy/policy配置、runtime及任务子集参数，默认旧行为保留；复用服务严格比对配置hash。新增镜像资产下载器，固定revision、支持断点、上游size/SHA256与本地逐文件比对。Sidecar现有实现拒绝DiT cache且不支持原compile路径，使用其原有部署配置与wxyz训练合同；历史模型的cache/compile开关差异单独记录，另做同配置控制，不能将历史分数称完全相同部署的公平对照。Sidecar权重只暴露用户指定20000文件，保留原训练目录。当前依赖下载/模型加载中，不称闭环通过。


### 2026-10-18T16:28:03.303620+00:00：按用户授权切换新环境Sidecar与独立加速验收

用户明确要求停止旧环境Sidecar及四路关闭优化原模型对照，保留新官方环境原版五任务轻量诊断，并将大部分GPU转向新环境20k Sidecar，探索cache/compile同时开启。旧三master已SIGTERM并确认无running项，产物保留为incomplete；旧19000/19500闲置模型释放，19400官方模型/原生诊断不动。先复用旧Sidecar已加载模型在新Conda/新RoboTwin做两个2条链路smoke（不并入正式成绩）。

新增仅显式运行时安装的benchmarks/robotwin/sidecar_fast_runtime.py：默认训练/部署/架构源码不改；编译对象是完整Sidecar forward，含7/15/23三处read/write，禁止复用原版foundation-only编译循环。沿用Base.generate原生联合video/action velocity缓存（0.99/max3、每generation reset），其近似跳步会改变输出，须单列加速结果且验收。check_sidecar_fast_runtime.py先以两真实RGB/正确wxyz state固定输入比较eager/缓存/compile/二者，记录CUDA同步耗时、完整动作、缓存统计和实际生成kernel；compiled对各自eager参考数值容差atol0.005/rtol0.01并非严格bitwise。尚未验收通过，不能宣称可用/提速。原文件不覆写，GPU0单独验证中。


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


### 2026-10-19T02:30:34.355454+00:00：32k独立GPU5/6评测

用户明确授权32k cache+Sidecar FFN评测；评测checker仅新增可选strategies（默认不变），本次限定ffn并保留原数值门槛；模型/训练数学不改。固定32k SHA、独立A100 runtime与sm80扩展，过程与限制详见docs/ROBOTWIN_SIDECAR32K_GPU56_20261019.md。未知/空/重复strategy自检查通过，真实GPU检查与闭环待完成。


### 2026-10-19T08:26:30.353305+00:00：32k推理消融准备（未启动GPU实验）

用户要求对adjust_bottle/place_shoe/beat_block_hammer两域做完整、关闭和局部EA推理消融。新增benchmarks/robotwin/sidecar_ablation.py与serve_sidecar_ablation.py，仅独立评测进程配置full/off/first_write/last_write；不改训练/架构/默认部署，不写权重。off复用已有sidecar_enabled=False，局部组只保留block7或23写入，三处关系读取更新保留。已有五项真实tiny CPU检查通过，新增编译重置回归及stage调用检查复验中；未据CPU宣称32k真实数值/成功率验收。

主要实验统一关闭cache，原FFN数值容差不变；off无分支kernel须单独验收，不能误报编译执行。六个条件共享固定专家manifest/指令/seed，四组先同前20条再至100条，总2400rollout。和旧native协议成绩分开，不用旧部分结果充作配对full。完全关分支仍是共同训练后的主干，不是无EA训练baseline；token混合不能当语义能力完全分离。本机5/6各余约11–12GB，无法加载模型，未自动排队抢卡/终止其他作业；待明确可用资源。方案与边界见docs/SIDECAR32K_INFERENCE_ABLATION_20261019.md。


### 2026-10-19T08:28:57.819433+00:00：消融CPU验收完成

`tests/benchmarks/test_sidecar_ablation.py`最终6项通过（150.22秒，包含共享盘导入等待），实际tiny模型覆盖四组输出、write/stage调用、权重不变、配置guard及编译重置不恢复write。两个运行时入口语法检查、git diff --check通过。四组cache关闭配置检查通过。尚未加载真实32k模型做本次消融GPU数值检查、未构建本次固定专家manifest、没有闭环成功率；需可用GPU资源后执行。

### 2026-10-19T09:17Z ManiSkill Sidecar 独立训练适配

原因：用户要求落实六任务ManiSkill数据上的Sidecar训练代码、执行指令与说明文档，进行baseline控制变量对照。此前只有绝对EEF baseline reader，RoboTwin Sidecar reader及旧ManiSkill125D标签不能直接使用。
变更：新增maniskill_sidecar_contract、ManiSkillSidecarDataset、离线准备器、独立dataloader/train配置及训练入口/四卡脚本。复用既有baseline轨迹精确映射、absolute-command EEF80D/10有效维、统计/RGB/文本、原trainer指标记录和block7/15/23 Sidecar数学；只新增loss-only130D监督。用原T+1世界TCP/物体轨迹重建8/16/32局部方向，分割bbox仅front/hand有效，track1及第三视角无效；物理contact阈值1e-6N，near .1m，goal映射.5m。独立入口纠正部署说明JSON中RoboTwin坐标描述，不改模型权重/forward。默认joint、bs32/acc2/global256、10epoch、每4000micro=2000update保存，ColorJitter关闭以对齐当前ManiSkill baseline。原baseline、现有Sidecar架构/训练核心及正在运行进程不改。
验收：实施中；需全量离线数据核对、真实窗口/分割抽查、单臂tiny模型梯度与配置/脚本检查。无GPU训练或闭环效果结论，不启动正式训练。bs32四卡在ManiSkill上的实际容量/吞吐、保存恢复待用户执行短测。

2026-10-19T09:20Z 保存边界更正：检查公共trainer后确认542009窗口/bs32/四rank的轮长4235micro为奇数，acc2的固定4000micro保存可能处于未完成累积。独立Sidecar配置改为save_steps=4235（每轮末，约2118updates）；保留公共trainer不变。说明文档明确轮长/GPU数改变须复核，不能将2000updates间隔描述为精确。本次单臂tiny初测4项通过，测试进一步改用实际action_is_pad掩码后复验。

### 2026-10-19T09:50:30.358939+00:00 ManiSkill Sidecar 交付验收

全量构造与独立reader验收通过：6000轨迹/542009窗口、全部label/derived SHA和baseline身份、54原RGB/动作/状态/mask/文本窗口、原分割bbox与原姿态局部plan、32样本batch。新增文件Python/bash/whitespace检查通过，最终CPU5项通过（188.18s）。配置比较仅batch/acc/save_steps/output不同，其余训练字段和图像处理与ManiSkill baseline相同。
独立入口额外验证实际sharded轮长的保存同步边界，默认4235micro末保存；resume沿用原完整状态加载，但恢复trainer_state中真实opt_step，防止奇数轮末部分累积导致global_micro//acc少计。相关CPU检查已通过，真实GPU恢复尚未验收。导出描述JSON修正为Panda/base/TCP，不更改Sidecar主干/forward或公共trainer源文件。
数据：outputs/maniskill_sidecar/data_absolute_eef_20261019/manifest.ready.json（约138MiB标签+manifest，引用原baseline numeric/stats及源RGB）。说明：docs/MANISKILL_SIDECAR_TRAINING_20261019.md。证据：outputs/maniskill_sidecar/verification_20261019/acceptance.json和data_checks_20261019.json。未启动GPU短测/长期训练，未改现有baseline进程/数据/配置；四卡容量、吞吐、GPU保存恢复及闭环效果待实际执行。


## 2026-10-19T14:10:21.890139+00:00：用户授权RoboTwin结束后自动接续ManiSkill

监督进程PID2193441，脚本scripts/supervise_maniskill_after_robotwin.py，状态/日志/spec位于`outputs/maniskill_sidecar/handover_after_robotwin_20261019T140623Z`。现场状态waiting_for_robotwin_completion，PPID1，独立会话，不依赖当前对话存活。当前RoboTwin仍正常训练，未终止/修改它。

放行条件：固定本机boot与原torchrun/四rank PID start_ticks；原进程全部退出后，四rank最终microstep均42960、最终Saved final日志和对应完整safetensors header/size必须满足；然后等0–3各A100-80GB且free>=70000MiB，确认MASTER_PORT29886可用后只启动一次。若前序异常未到终点，则记录predecessor_incomplete_no_launch并退出；不误把崩溃/临时空卡当完成，不kill任何其他作业。通过持有flock和launched.json防同一接续重复启动。

接续执行用户原命令：CUDA_VISIBLE_DEVICES=0,1,2,3、CHECK_ONLY=0、bash scripts/train_maniskill_sidecar_4gpu.sh；MANISKILL_SIDECAR_OUTPUT_DIR在实际接续时生成outputs/maniskill_sidecar/joint_<UTC>，train.log写入该目录。保持现有joint/10epoch/bs32/acc2/global256/每4235micro保存/ColorJitter关闭，从Foundation新训练，绝非接RoboTwin权重继续训练。默认manifest为已验收6000轨迹/542009窗口的独立ManiSkill Sidecar ready；原baseline数值和原HDF5依赖保持。

已验证并复用现有`/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv`（总runtime约7.5GB），没有新增环境副本；Torch2.7.1+cu128/Accelerate1.15.0/DeepSpeed0.18.9/h5py3.16.0/torchvision0.22.1+cu128均从本地环境实际导入，核心依赖导入6.68秒，CHECK_ONLY配置解析exit0，代码/manifest与现有验收hash一致。若该本地副本消失，回退共享.venv；不安装包、不改原环境。模型权重/数据/输出仍在共享盘，不宣称6.68秒等于正式训练启动时间。

监督逻辑7项CPU测试通过，包含合成子进程完整接续、等待显存、防重复、前序失败不启动、四rank与权重完整性和PID复用检查；未实际提前启动GPU训练。该ManiSkill Sidecar的真实Foundation四卡显存/更新及完整恢复仍待接续后验收，既有未完成风险保留。新训练失败时记录maniskill_failed/returncode及train.log，不自动反复启动。
# 2026-10-19 UTC - OpenWAM ManiSkill baseline 六任务评估入口

- 17:39 UTC运行验收：双模型/12路真实闭环均启动，六任务首批共32条终态通过，首视频H.264检查通过；600条最终SR待完成，结果持续写入gpu56_100正式根。详情见专用评估文档，未以早期32/32外推模型最终效果。

- 最新规格：按用户追加改为每任务100条/共600条，GPU5/6各一模型+6仿真、12独立端口、每任务50+50分片；双模型Inductor缓存隔离。薄client改用已有benchmark prompt模块，修复误引训练包的pandas依赖；模型前做仿真环境依赖预检，不安装任何包。联合8项检查通过，真实模型闭环仍待验收。

- 并行追加：用户指定GPU5/6尽快评估，复用现有shared-engine server为12个独立策略端口；每任务50条分成25+25互斥shards，GPU5/6各6仿真。复用刚验收的六任务专家报告；原专家脚本只增加可选输出路径，不重复实现专家gate。分片与pool联合7项检查通过。正式权重/模型/预算/判据不改，权威结果根新增`parallel12`。

- 原因：当前Franka baseline训练的是绝对base-frame command EEF10，旧DiT4DiT delta7客户端不能直接使用；用户授权迁移必要评估实现。
- 变更：新增`benchmarks/maniskill/{single_eval,run_eval}.py`及README，复用现有官方model server、normalizer、transport、图像与EEF合同；绑定14K checkpoint快照，使用既有六任务test seeds、双控制器专家gate、H.264视频、逐episode/汇总和失败记录。
- 验证：CPU4项通过；独立源专家检查六任务双控制器共12回放成功；新runner模型加载和300条评估已启动，效果尚未形成完整结果。模型、训练、原始数据及DiT4DiT源码不修改。
- 运行合同与未解决门槛：`docs/MANISKILL_BASELINE_EVALUATION_20261019.md`。



### 2026-10-20T02:33:16.661422+00:00：最终42960权重GPU5/6独立标准评测

用户明确授权最终checkpoint_step_42960.safetensors（不是32k、不是消融）。权重固定到outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z/checkpoint_step_42960，24862363126 bytes，SHA256 501addbed687e6d7587af60ab4480b9c79eaa2c8867d723b74d54e0acc9deb55；weight只读hardlink，config/stats/contract/tokenizer独立物化。42960micro=21480参数更新。

本机A100 GPU5/6初始各约81GB空闲，使用两份模型、每模型3独立同步端口/互斥计算锁，六路sim均限制在GPU5/6，不操作0–3上的ManiSkill或其他进程。模型沿用已验证本地RAM venv，仿真用官方Conda及此前独立A100 RoboTwin/sm80扩展，五模块hash复核与已验收一致；process-only现有gpucomp预加载及NVIDIA ICD修复保留，无驱动/依赖/模型源码改动。通用fallback TORCH_EXTENSIONS_DIR仍由旧runner指向共享目录，本次用已安装sm80库，不触发/清理通用JIT。

评测完整Sidecar，10步、32动作、CFG1、sync、wxyz自有stats、384×320、joint velocity cache .99/max3及Sidecar局部FFN compile；每份新权重须两真实fixture、原atol=.005/rtol=.01和真实kernel门槛通过才放行。保留官方native专家筛seed、seed group0、unseen、RT32spp/depth8/oidn、原动作预算、无planner fallback。50任务两域各100条，共10000；任务固定无重叠两分区，优先瓶子/锤子/鞋子/扫码等。每端口独立动作buffer，缓存每generation reset；缓存近似误差仍须闭环评估，不把FFN匹配其cache参考当cache等价完整计算。

当前模型加载，队列等待numerical gate，尚无正式42960成功率；进程/命令、快照与独立监控都在新输出根。实时文档docs/ROBOTWIN_SIDECAR42960_LIVE_20261020.md只读取本次两queue，只验收完整exit0/100条/无errors；异常/部分仅观察进度，不与32k、其他机器合并/择优。未来有渲染异常需保留证据、排除异常未完整项，不降低协议冒充修复。


### 2026-10-20T03:53:18.972628+00:00：按用户授权利用GPU4/7扩容最终42960评测

现场GPU4/7各约40191MiB空闲、总A100-80GB；未知namespace原40GB占用不处理。新增GPU4同一42960模型PID2462558，端口21820/21821，复用现有serve_checked_ffn及完全相同checkpoint/deploy，独立gpu4/inductor；CUDA放置门槛33000MiB，尚须真实FFN数值验收和PONG。GPU7拟新增2路sim，逐任务free>=14000MiB且第二路延迟45秒；不向GPU4叠加sim，目标3模型8路（当前仍2模型6路）。

输出内expand_gpu4.py/PID2463380复用已验收expand_gpu6.py的freeze_source/exit_code/verify/save，并将helper.OUT绑定本次42960输出。只在GPU4验收后暂停本次GPU5原调度器，原三standard_eval/native客户端继续原PID，原生百条不中断；替代队列expanded_queue54接管其50个task/mode，五worker含原GPU5三个slot与GPU4两个slot，GPU6原50个不动，总100唯一条件。源三子进程终态及真实exit status记录后才回收暂停的旧调度器；不要手动SIGCONT旧调度器造成重复派发。

监控已改为读取原子queue_layout.json，未提交接管前仍读原gpu5/queue+gpu6/queue，提交后仅expanded_queue54+gpu6/queue；旧监控源文件保留watch_results_before_expansion.py。模型/训练/数据/RT/预算不改，不操作0–3的ManiSkill。独立输出脚本语法、自带100条件唯一性/重复拒绝，以及目标Python3.10真实exit7僵尸退出码检查通过；系统Python3.8不支持该helper的waitstatus_to_exitcode，控制器明确使用本地3.10，未修改系统或原helper。当前source三job PID核对存在，全部原episode保留。验证产物expansion_preflight.json。

新增卡长期容量、两路真实闭环仍待实际加载/场景创建，free瞬时满足不能保证未知作业不增长。保留日志、失败不计有效百条；无自动降低RT/动作预算或结束外部进程。


### 2026-10-20T03:58:51.913926+00:00：GPU4模型验收通过，最终42960扩至3模型/8个评测父进程

GPU4两真实fixture四个匹配cache参考比较max_abs_error全0.0、真实4个Inductor kernels，原门槛通过；warm cache+FFN约0.3327秒/chunk，不是整episode吞吐。模型仍完整三处Sidecar写入，不改变cache/动作预算/RT。GPU4当前free约15751MiB；GPU7新客户端尚在权重检查/初始化，free约40192MiB，不能据此提前保证两sim完整renderer峰值已验收。

03:57接管提交：旧GPU5调度器2439689仅SIGSTOP，原GPU5三outer PID2442123/2442126/2442125、GPU6三outer PID2442127/2442124/2442122均保留且存活。新增GPU7 shoe/Randomized PID2466152、dual_shoes/Clean PID2466329已启动，两新父进程目前初始化，不提前称已完成新episode。新权威队列expanded_queue54与gpu6/queue恰好100个唯一条件、8个running、errors为空，监控无报错。原source快照保留且不再统计冻结gpu5/queue；原三job真正退出并记录状态后才回收暂停调度器。

实际隔离子进程自检也已通过：暂停仿真stub父进程后，三子进程继续并以0/7/0退出，真实退出码完整读取，不触碰当前评测PID；证据adoption_selfcheck/result.json。扩容验收expansion_gpu4_acceptance.json、实时新状态见原实时文档。后续关注GPU7两sim真实动作、峰值及外部显存变化；尚无本次新增通道的完整百条结果。


### 2026-10-21T04:09:39.851793+00:00：新增 RoboTwin Sidecar 隔离 warmup 训练版本

用户明确授权在联合隔离与冻结 backbone 两种方案中选择最小合理实现。采用 joint 主干 FM 持续训练、Sidecar 前2000 optimizer updates（acc2=4000 microsteps）读取 detach 特征且关闭全部 action 写回，之后恢复完整联合；避免静态冻结参数未注册到 ZeRO2 优化器后再解冻的问题。新增独立配置/薄 launcher，10epochs预算含warmup、bs32acc2/ColorJitter/数据合同不变，Sidecar标称LR5e-5 vs backbone1e-4，整体EA loss含deep统一乘0.5。

触达：RelationalSidecarArchitecture 增加默认关闭的隔离进度与辅助总scale；共享trainer在每microstep向支持的architecture传递已完成opt_step；optimizer_groups增加可选sidecar_lr且保留其余模块原LR；旧正式launcher增加配置名环境覆盖，原默认不变；新增隔离配置、launcher与CPU验证。无新增模型tensor/state_dict键，推理eval走完整分支；不修改现有训练/评估进程及历史权重。

前向/梯度隔离不代表全局梯度裁剪后优化轨迹逐位等同baseline；原cosine绝对floor1e-6意味着LR比例末尾趋向1，Sidecar不超过backbone。GPU ZeRO2阶段切换/完整恢复/闭环与效果提升尚未验证。设计、启动、验证与风险见 docs/ROBOTWIN_SIDECAR_ISOLATED_WARMUP_20261021.md。

验收补充：CPU相关测试两批41/41、33/33通过（重叠不累加）；新入口共享盘环境CHECK_ONLY exit0，配置/验证摘要已存docs/evidence/sidecar_isolated_warmup_20261021/。shell语法、git diff --check通过；GPU训练未启动。


### 2026-10-21T07:10:02.036250+00:00：为既有 Slurm baseline 服务器交付隔离 warmup 代码与精确配对数据

用户授权打包替换目标/home/mingfeihangroup/xiaoming/projects/OpenWAM，并将训练数据放/home/mingfeihangroup/xiaoming/datasets。新增独立cluster/sidecar_isolated_clean50.sbatch、train_sidecar_isolated.sh和仅增加逐rank指标的entry，复用现有Foundation、.venv、配置和trainer，不改变模型/训练数学。保留Slurm CUDA_VISIBLE_DEVICES，不套用源服务器0–3卡及90秒handover；新增package_sidecar_transfer.py复用归档writer，打包当前manifest的50任务2500条HDF5、匹配标签/统计/prepared metadata，保留PAX纳秒mtime、解引用文件并校验标签hash。旧baseline和训练进程均不操作。

打包Python3.10测试发现原writer的w|gz不接受compresslevel；修复同一writer为显式gzip.GzipFile(level1)+tar流，保留原hash/PAX/源变动检测。新增归档mtime/hash/解引用、Slurm分配保持、有限指标和数据合同拒绝测试，与原迁移和累积计数共12项通过；实际CHECK_ONLY解析及shell语法通过。目标服务器/GPU未连接验证，完整数据包生成与目标传输校验仍需交付记录。迁移/备份/解压/预检/短测/正式sbatch见docs/SIDECAR_ISOLATED_CLUSTER_TRANSFER_20261021.md。

交付完成 2026-10-21T07:33:43.341432+00:00：/remote-home/jinminghao/OpenWAM/outputs/transfers/sidecar_isolated_20261021T071002Z；代码1,641,685 bytes、数据23,448,238,134 bytes（50任务2500条配对HDF5/标签）；解压代码51项CPU测试、实际单轨迹重定位/reader、全归档逐文件hash/纳秒mtime/整包SHA与gzipCRC通过。未传输/连接目标/提交训练；目标四卡仍先短测。最终清单与命令见SIDECAR_ISOLATED_CLUSTER_TRANSFER_20261021.md及包外README_TRANSFER.md。


### 2026-10-21T09:39:17.976543+00:00：用户授权42960 checkpoint写回诊断（GPU4/7）

新增benchmarks/robotwin/sidecar_write_diagnostics.py与collect_sidecar_diagnostic_inputs.py、3项CPU测试；复用官方manifest/评测wrapper及FFN数值门槛，实例内write beta0→finally恢复，保留read。新增A实际BF16残差与FP32加法前参考分布、B同输入同seed生成/同噪声单forward重放与EEF物理差异，计划18对闭环。生产架构、训练、现有正式评测不改，不使用旧off整支路绕过。权重固定hardlink同SHA，先CPU载入并以GPU4文本缓存+GPU7模型减显存。输出/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar42960_write_diagnostic_20261021T092505Z；阶段与限制见SIDECAR42960_WRITE_DIAGNOSTIC_20261021.md。目前3项CPU测试通过，场景采集/后续GPU实验未完成。


## 2026-10-21T10:15:38.256967+00:00 — 42960写回诊断离线验证完成及后台闭环收尾

原因：用户要求用GPU4/7判断Sidecar写回的实际幅度、仅关闭write对动作与小规模闭环的影响。已有实例内hook未修改核心训练/模型；18真实初始观测、无cache与当前cache共36组A/B完成，三项CPU测试通过、GPU FFN四项数值对照max误差0。新增输出内finalize.py负责18对闭环完成后的配对/首态一致性校验、结果文档与本次模型退出检查；--check已通过，未停止其他作业。当前闭环未完成，不声明SR收益。完整结果见docs/SIDECAR42960_WRITE_DIAGNOSTIC_20261021.md；原训练交付包与checkpoint文件不变。


## 2026-10-21T13:15:35.707555+00:00 — 42960写回诊断全部完成并复核

原计划18对/36rollouts完成：Full13/18、WriteOff14/18，Full-only0、Off-only1；18对初始完整输入哈希一致。A/B幅度见专门诊断文档，未发现本小样本write净收益，未外推Sidecar训练整体效果。检查本次模型与控制器已正常退出，未停止其他作业。未改训练/核心模型/原checkpoint或交付归档，未额外启动重训。


## 2026-10-21T14:17:54.921886+00:00 — 启动42960无更新GT诊断

用户授权实验一/三实测。仅扩展benchmark WriteProbe的scale参数（默认原行为）、新增sidecar_gt_diagnostics.py、collect_sidecar_expert_gt.py和两项CPU用例；共5项通过。60组Clean训练内窗口推理与36组tap表征统计完成；真Randomized专家GT采集GPU4/7已启动，v6 oracle实例内hook不改外部源码。无optimizer/核心训练逻辑/原权重/训练数据更改。原gate缩放逐调用finally恢复，源标签SHA与数据stat检查。结果、单位、数据范围和实验二方法见SIDECAR42960_GT_DIAGNOSTICS_20261021.md。


## 2026-10-21T14:56:06.449540+00:00 — 完成新场景GT推理诊断与真实梯度短测

实验一/三：6新seed专家轨迹、24窗口+1重复控制，125组五倍率推理完成，GT位置/姿态/时序/seed不重叠验收通过。新增sidecar_gradient_probe.py实例捕获活Action/Video/EA loss，冻结主干参数但保留反传路径，统一prepared mask设备，单batch真实探针通过；新增真实tiny架构梯度测试通过，前5项回归保持通过。无optimizer、核心训练修改、源权重/数据覆盖或新训练。结果和限定范围见SIDECAR42960_GT_DIAGNOSTICS_20261021.md；仅新benchmark与实验输出改变。


### 2026-10-21T17:25:06.653433+00:00：32 Batch 只读梯度诊断
用户明确批准跨任务/双域32批诊断；新增 benchmarks/robotwin/sidecar_gradient32.py，复用真实六条专家HDF5及已有单批活loss分解。只在独立诊断进程固定两个FM抽样，保留随机数消耗并断言调用；按原42960加权Action/Video/EA计算梯度，按Read Projector、Read、Refinement、Plan role_tokens[9:15]、三Write Projection/β记录逐批norm/cos和有符号gate统计。sidecar_gt_diagnostics.py新增独立flag分派，不走倍率扫描；sidecar_gradient_probe.py增加分组统计。新增专门单测。生产网络、训练/2K warmup、数据manifest、原checkpoint与已有训练/评估进程均不改。不建optimizer、不称Adam真实更新。

32批完成验收：全部请求组Action梯度非零、EA→Write结构性零/None、Sidecar权重严格未变，10测试通过，PID3081703正常退出。结论和下一步单一Stage2路由候选已记录docs/SIDECAR42960_GRADIENT32_20261021.md；未改生产训练/网络/当前2K方案。原始JSON与CSV在/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar42960_gradient32_20261021T172128Z。
# 2026-10-22 UTC - ManiSkill完整Sidecar 25410同协议评估

- 用户授权沿baseline逻辑评估Sidecar。复用原six-task客户端/EEF10桥接/seed清单，不改模型、训练和数据核心；两模型统计SHA相同，更新数12708对14000明确记录。
- 评测入口增加显式Sidecar模式/独立仿真卡映射/协议一致性gate/完成后的对照汇总。新增ManiSkill Sidecar服务wrapper，复用现有FFN runtime及检查器；检查器仅新增optional native fixtures，原RoboTwin默认行为保留。
- 回归14项通过；GPU6单模型、GPU5共12sim，启动600条评估，权威根`sidecar25410_six_task_100_replan8_max200_matched_20261022`。真实数值/策略闭环尚待验收，未宣称SR或模块收益。
- 细节与限制：`docs/MANISKILL_SIDECAR25410_EVALUATION_20261022.md`。



### 2026-10-22T09:20:56.846718+00:00：新增正常读写/冻结WAM更新的2K Sidecar warmup
用户明确授权新训练版本。保留旧isolation/joint，新增backbone_warmup_updates=2000配置与入口，两warmup互斥。新模式不改前向/read/write/损失；以WAM叶子gradient mask+原生AdamW step-pre-hook冻结参数/动量/衰减，保持ZeRO分组静态、激活反传可达Sidecar；2K后恢复全部原有可训练参数更新。日志ea_backbone_frozen_warmup，边界复用已恢复opt_step。新profile沿用旧warmup的LR/aux/增强/数据/10epochs，只变阶段策略；未叠加EA-heads-only、Gate初始化/倍率或第三阶段。cluster launcher增加可选profile、默认不变。CPU50测试通过，真实ZeRO2微型测试继续；未启动正式训练/停止已有任务。详细比较与风险见docs/ROBOTWIN_SIDECAR_FROZEN_BACKBONE_WARMUP_20261022.md。

2026-10-22T09:24:45.484493+00:00 最终验收：CPU51pass；两卡tiny BF16/fused AdamW/ZeRO2/acc2、阶段切换及阈值前/边界/后checkpoint恢复后下一更新逐位一致，两rank pass、exit0。新配置resolve与shell语法通过。仅微型工程验证，未启动全量正式训练/未证明效果提升；详见对应validation.json与专门方案文档。


### 2026-10-22T09:52:26.647874+00:00：frozen-backbone2K基于旧isolation交付包的增量迁移
按用户指定GPU-8A100新增独立Slurm脚本，单节点4GPU/32CPU/10天、Slurm分配不覆写、不绑定gnode06。实际对照旧包eccfdbb8...生成15文件增量（6替换9新增、33785bytes），不打包环境/权重/数据/原始根日志。目标根修改/风险记录采用追加，旧方案保留。旧包解压+本增量实际回归52pass、配置resolve、Slurm桩/安全归档/hash验收通过；没有连接目标服务器/传输/提交作业。交付/remote-home/jinminghao/OpenWAM/outputs/transfers/sidecar_frozen_backbone_delta_20261022T094829Z，SHA256 d8b40e4004a02ec3bc8301c6c419b73ed566d08ce527c29c2e0f4909fb59717d。


### 2026-10-22T13:42:33.785223+00:00：按用户澄清保留完整WAM反向，仅跳过前2K参数更新

原因：初版helper在WAM叶子参数上清零梯度，虽然保留Sidecar激活反传，却没有完整保留用户要求的WAM真实参数梯度。用户明确选择先实现正常读写/完整反向的warmup，不叠加其他实验。

修改：移除sidecar_warmup.py中的叶子梯度清零hook；保留原生AdamW step-pre-hook，WAM正常前向、反向、累积、归约和全局裁剪后，仅在optimizer执行前跳过WAM组，参数、动量、step和decay均不更新。Sidecar正常Read/Refine/Head/Write并接收Action/Video/EA梯度，达到2000 optimizer updates后恢复joint。沿用现有独立profile/启动脚本、LR、损失、Gate初始化、ColorJitter和10epoch预算；不加heads-only或第三阶段。永久冻结模块不变。

工程验收：53项CPU测试通过，检查全部真实梯度与joint一致、共同clip后仍一致、Sidecar一次更新与joint逐位一致、WAM权重/Adam不动、1999→2000边界及恢复模型/完整Adam状态一致。GPU5/6真实tiny Wan/ActionDiT、BF16/fused AdamW/ZeRO2/acc2缩阈值2通过：两rank均捕获非零有限WAM反向梯度，warmup仅Sidecar变、joint均变；1/2/3更新处保存重载后下一更新逐位一致。测试进程exit0。Hydra解析及三个入口shell语法通过。证据docs/evidence/sidecar_full_backward_warmup_20261022/validation.json。

边界：WAM重新参与global clip，Sidecar裁剪后更新可能不同于初版；正常反向不节省显存。全局scheduler继续推进、10epoch内WAM少2000次更新、Stage2 EA主导风险保留。GPUfixture不含全量Foundation、真实数据或LR scheduler；未启动正式训练、未停止已有ManiSkill四卡作业、未证明闭环收益。09:48历史传输包未重新打包，仍保留旧叶梯度清零逻辑，迁移文档已标明。


### 2026-10-22T13:50:26.592578+00:00：启动GPU0–3 Aloha50完整反向／WAM延迟更新2K正式实验

用户明确授权替换其GPU0/1/2/3旧ManiSkill实验，并要求新作业先完成加载再结束旧作业。使用已有本地runtime环境、原Foundation与已准备Aloha50/2500 manifest，快速CPU预检及配置解析通过；四rank新进程已启动CPU加载，复用现有cpu_ready/gpu_release门槛。输出/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z，launch.json封存PID启动时间/命令/源SHA与旧进程身份。配置bs32×4、acc2、10epoch、ColorJitter，Sidecar5e-5/WAM1e-4、Action/Video1、EA scale .5。前2K正常读写/完整反向、仅WAM不更新，之后joint。尚未确认真实GPU更新，不将已启动进程当成训练成功。旧进程待CPU-ready后仅按已验证PID身份结束；GPU4–7不操作。进一步交接结果以专门训练记录和输出handover为准。


### 2026-10-22T13:57:58.024694+00:00：GPU0–3新warmup正式训练已接管且完成真实更新
新Aloha50/2500、10epoch、ColorJitter实验已完成CPU加载后按授权结束旧ManiSkill torchrun2430422；旧全部子进程退出，33880完整恢复状态保留。新四rankDeepSpeed运行，micro12/optimizer6，三loss有限、warmup=1、write gates已更新。输出/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z；详见专门训练记录与startup_validation.json。旧0–3作业已由新作业替代，GPU4–7不操作。尚未验证真实run的2K切换/新checkpoint/闭环效果，微型验收不替代后续效果。


### 2026-10-22T16:05:41.917766+00:00：记录42960最新成绩并按用户要求停止评测
20完整条件2000条成绩1057/1100 Clean96.09%、481/900 Random53.44%、1538/2000综合76.90%，官方同条件78.10%。含8个中止条件累计2401条，partial不入正式均值，72条件未启动。全部原始结果/日志/权重保留。先冻结派发并保存带PID/start_ticks/PGID身份的快照，再结束53个本评测进程/22组；TERM宽限后仅清理本作业残留，全部无活跃PID、8服务端口关闭、GPU5/6各1MiB。GPU4/7未知其他作业保留；新0–3四卡训练PID3501138仍存活、micro1172/opt586、loss有限。canonical状态明确stopped_by_user/interrupted_by_user，不把用户中止当模型失败。证据docs/ROBOTWIN_SIDECAR42960_STOPPED_20261022.md、/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z/user_stop_20261022T160226Z/final_record.json及termination.json。未新启动评测、无全50任务最终或新warmup效果结论。


## 2026-10-24T16:38:08.527192+00:00: Multi-simulator Sidecar mid-training — authorized implementation

User requested a current-code backup, canonical dataset preparation under /remote-home/jinminghao/datasets, and an additive Sidecar mid-training pipeline spanning RLBench Franka, ManiSkill Franka, and RoboTwin Aloha/Piper. Backup is complete: local snapshot3737261, remote0bf6aef0212184703803ff7e3aacaa919d6aaf2e, identical tree702163ce79c323a1f232679cf0d471d271dac2b3, branchbackup/pre-multisim-midtrain-20261024T162442Z. Original working branch/index and running jobs preserved.

Scope: immutable accepted-source inventory; ManiSkill old Clean5000+B/C/O/D/M1000 each+hard10000; preserve physical-family splits; absolute EEF20 -> Foundation80 slots, true arm masks, 32 actions/9 video frames, regenerated130D8/16/32 targets, explicit image codecs and geometry; one offline prepared numeric/media dataset with train-only statistics; new isolated loader/config/entry/launcher, tests and documentation. Reuse current full-backward/WAM-update-delayed2K implementation; no Sidecar network changes and no existing baseline removal. Source RGB hardlinks may avoid duplicate payload storage but must be read-only; all runtime paths reside beneath the new dataset root.

Required considerations: RLBench currently has8204 accepted visual episodes rather than20000; distinguish planned quotas from available data. Unknown goals/contact/absent arms must be masked, never false negatives. Source action commands and achieved states are distinct; wxyz vs xyzw, column Rot6D, RoboTwin local-x TCP offset, corrected-vs-legacy JPEG codecs, native time axes and train/test family leakage remain explicit. No full training launch is authorized by this setup task; verify loader/optimizer/save-resume and resource availability before any later formal training.


### 2026-10-24T17:23:38.653984+00:00: Multi-sim implementation/engineering evidence
Added independent multisim_sidecar_contract, prepared mixed reader with per-epoch source coverage and aligned epochs, accepted-source offline converter with source command/time checks and preparation fingerprints, preflight, training profile and launcher. Preserved current Sidecar/full-backward2K optimization math. Current frozen selection52204, initial/final-revision real10-episode smoke passed;27 CPU checks passed including mixed-embodiment loss/warmup update and train-only stats; source parity6 cases passed. Full conversion and one-shot audit remain active, not a ready release or GPU/quality validation. Guide and evidence:docs/MULTISIM_SIDECAR_MIDTRAIN_20261024.md;docs/evidence/multisim_sidecar_midtrain_20261024/verification.json.


### 2026-10-24T17:48:07.194757+00:00：midtrain恢复契约/工程短测与CI边界

补齐新midtrain恢复元数据中的world-size、batch、accumulation与计划micro/epoch预算，拒绝跨world-size或改计划的伪resume；新混合数据回归覆盖此约束。新reader/训练入口/检查器/tests导入排序整理，pytest fixture通过插件显式注册；27项相关回归通过，原模型/原trainer/原单源数据逻辑未改。正在运行的prepare脚本及指纹依赖保持冻结，5项导入排序延后。

GPU5/6各约70GiB空闲后启动独立Foundation工程短测PID3798274，10条真实来源smoke、bs1/acc1/8微步、WAM延迟2更新，检查完整读写反向与保存/重载。启动记录outputs/midtrain_audit_20261024/foundation_smoke/launch.json，尚未验收完成；原0–3训练/其他服务不停止。全量构建3795413继续。

GitHub功能分支初次CI确认为失败而非上传失败：tests 2258 passed/5 failed，失败分别是旧HTTP测试缺Flask及旧远程评估测试依赖未入库的outputs/final_record；不是新增multisim测试失败。全仓ruff的370项在改动前backup commit逐文件复现，另有新prepare脚本5项导入排序待冻结结束清理。不将其报为CI绿灯，也不为本任务批量改旧模块/隐藏失败。详见outputs/midtrain_audit_20261024/ci_initial_*与lint_backup_comparison.json。


### 2026-10-24T17:51:50.656134+00:00：WAM warmup与optimizer offload的必要限制

实查当前Accelerate _prepare_deepspeed在CPU optimizer offload时调用map_pytorch_optim_to_deepspeed，创建新DeepSpeedCPUAdam，不复制原AdamW的step_pre_hook；该hook正是当前“完整反向但WAM不更新”的实现。因此新midtrain入口在prepare前显式拒绝WAM warmup+optimizer offload组合，避免静默变成WAM同步更新；默认offload=none完全不变，原单源入口不改。10行测试覆盖cpu/none及warmup=0；不以CPU offload绕过本轮显存问题。

第一次双卡真实Foundation短测前两次Sidecar更新/完整反向通过，第三次（WAM首次更新）在ZeRO flatten申请9.31GiB时OOM，尚无checkpoint。第二次仅设置PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，PID3800695，其他模型/数据/优化逻辑相同，当前加载中；日志foundation_smoke_expandable/train.log。全仓CPU suite此前2220 passed/18 skipped/21 deselected；新增offload guard随后单独回归，不把旧suite当新增分支覆盖。


### 2026-10-24T17:57:11.267195+00:00：双卡Foundation工程短测容量结论

第二次仅使用expandable_segments的同配置短测亦在WAM首次Adam更新失败，这次exp_avg_sq申请9.31GiB而实际free5.79GiB，缓存仅78.8MiB，确认现有双卡空余容量不足，不能归因纯碎片。两次前2步完整反向/Sidecar更新的全部metrics严格相同；不是数据/形状/nonfinite错误。两个torchrun均已退出，本次显存释放；不再同容量反复重试，不终止其他作业、不改默认四卡训练语义。保存/恢复未验收，须更多可用显存后完成。证据outputs/midtrain_audit_20261024/foundation_smoke_expandable/result.json。


### 2026-10-24T18:02:16.864924+00:00：离线转换并发4→16，保持构造源码指纹

原4worker在共享盘I/O等待为主且CPU/内存余量充足；仅核验并结束本任务原CPU parent/worker和验收watcher，保留冻结selection与逐条numeric/proof，未动GPU或原始数据。原准备脚本SHA d094e6f2、组合指纹3096f6cf保持不变，新16worker PID3802485/验收3802486复核复用旧结果。恢复阶段数十秒内完成旧缓存检查，之后新增转换从9600到11900，测得18.24条/秒；这是当前ManiSkill区段，不外推为全量固定ETA。证据outputs/midtrain_audit_20261024/parallel16_restart.json、parallel16_throughput.json、full_prepare_parallel16.log。正式release仍未发布，继续全量与验收。


### 2026-10-24T20:50:14.604681+00:00：52,204全量数据正式验收及CI修复

全量numeric/media构建和train-only统计完成；manifest SHA6531627b、stats204c49cb，train39816/val6161/test6227、独立物理家族29101。全数numeric hash/media stat/ID与冻结selection/配额/家族split通过，真实全量mixture及启动CHECK_ONLY通过。RLB来源仍4093已验收家族，未漏新增。数据BUILD_SOURCE.json验证精确构造文件等于a9e282c6对应本地75219033快照的组合指纹3096f6cf；CI格式整理发生在builder退出后，旧数据指纹不冒充新文件SHA。原raw、原GPU训练及checkpoint不变，未启动正式midtrain。

用户要求核查GitHub失败并继续修复，远端a9e282c6已存在，失败为CI而非代码上传。根因：3个HTTP测试缺Flask dev依赖、2个队列测试依赖本机outputs历史fixture，以及375项lint（370旧存+5新脚本排序）。补Flask dev依赖、队列测试在tmp_path生成封存历史及SHA、pytest fixture改显式插件注册、格式/导入整理。运行所需环境注册import保留，路径设置后的3个import用局部E402说明；不放宽全局检查/不skip失败测试。37文件变更前备份与计算AST比对位于outputs/midtrain_audit_20261024/ci_repair，除了测试fixture/dependency，计算AST保持一致（导入排序/fixture注册与等价sigmoid命名函数单独归一化比较）。

本地全仓CPU2221 passed/18 skipped/21 deselected，与CI同版ruff0.11.9全仓通过。后续需远端CI实测绿灯；Foundation联合更新/保存恢复GPU短测仍因双卡现有空余容量不足而待四卡资源，不能以数据/CPU检查冒充。


### 2026-10-24T20:57:51.687499+00:00：GitHub CI修复远端验收通过

feature/multisim-sidecar-midtrain提交7a8022ee已回读确认，tree7b75d08a与本地e9d3dd9d一致；Actions40131709993的lint/tests均success，远端2265 passed/16 skipped/21 deselected。Flask dev依赖、独立queue fixture和格式问题已解决，没有skip失败测试/放宽全局lint。原备份分支不变。全量52204数据验收完成，当前仅剩完整Foundation联合更新/保存恢复GPU短测待可用四卡，已向用户询问资源窗口，未结束当前0–3原训练。


### 2026-10-25T05:16:39.134801+00:00：用户确认OpenWAM内独立完整原始/训练归档与默认绝对EEF

按用户新指令，新增datasets/multisim_sidecar_archive_v2_20261025，完整物理复制五个指定原始source tree（含回放/元信息、原ManiSkill未选入训练的原始样本），训练配额仍单独选取，不通过hardlink/symlink替代独立副本。保留MS/RLBench test家族，不删除原来源。新增archive_multisim_sources.py，冻结SQLite逐文件清单、源不变检查、独立inode检查、写后SHA回读与原来源不可用时的全量verify；GPFS用原子mkdir排除重复写入，断点只复用经证明的目标。两个CPU回归通过：独立副本在移除测试源后可验证/能检出损坏，内部符号链接被实体化/循环拒绝。数据目录加入gitignore避免TB数据误提交。

用户最新明确RoboTwin按OpenWAM默认绝对EEF：保持endpose[t] proprio和endpose[t+1:t+33]绝对动作监督，取消上一轮“默认全改FK控制目标”的建议；不是旧DiT4DiT delta动作。命令与实测原始字段均保存。先前命令语义讨论已据此明确，后续必须验证训练/推理同坐标系、TCP、wxyz/Rot6D、夹爪与归一化。独立raw拷贝、回放code/assets依赖闭包、旧EORT test协议归档、训练manifest相对路径及源清理范围仍未完成；不可把任何部分进度称可安全删除原目录。


### 2026-10-25T06:13:50.522706+00:00：RoboTwin 默认绝对 EEF 契约验收与完整归档/训练选择分离

按用户明确要求继续当前 OpenWAM 的绝对实测 EEF：endpose[t] proprio、未来 endpose[t+1:t+33] 动作；没有引入 delta 或默认替换 FK command。12 组 Aloha/Piper×base/clutter×首中末帧通过真实部署反归一化/EEF20→16/quaternion 往返，最大位置误差 <8.5e-8m、旋转 <1.6e-7rad、夹爪一致。此为编码接口证明，非原生物理回放或成功率证明。证据已存 docs/evidence/multisim_archive_v2_20261025/robotwin_absolute_eef_contract.json。

新 multisim reader 增加可选 selected_for_midtrain 过滤，避免把完整归档中未选训练样本隐式扩入训练；val/test 保留完整可读，旧清单默认兼容。archive+selector 15 项回归通过。复制 helper 显式记录杂乱源 execution-view 外部代码/资产链接：代码/资产在 replay 下独立存一份并记录原链接，不能把别名重复数万份，也不跳过原始 episode payload。native A/B、replay 依赖和 training numeric 四任务独立状态库，当前仍 BUILDING；不删除任何来源，不把部分 SHA 证明当可清理许可。详见更新后的 MULTISIM_MIDTRAIN_ACTION_AND_STORAGE_AUDIT_20261025.md。


### 2026-10-25T15:28:13.175236+00:00：用户要求四卡A100/global512的1–2 epoch与500-update完整反向warmup

用户明确改用最新500步方案。只修改独立 configs/train_multisim_sidecar_midtrain.yaml 的 backbone_warmup_updates:2000→500，isolation_warmup_updates保持0，正常Read/Write和Action/Video/EA完整反向，WAM前500次optimizer更新跳过参数/Adam状态更新；之后联合。旧Aloha/ManiSkill与2K实验配置不改。新增薄入口train_multisim_sidecar_4a100.sh：每卡32×4卡×acc4=global512，明确max_steps=null、num_epochs=1/2；Slurm模板显式gpu_8a100 QoS且不固定节点。24项相关回归已通过，目标四卡完整联合更新/保存恢复仍待实际资源，未宣称GPU验收完成。

最终清单检查发现production验收文件外引用的540 RLBench+32 MS-hard视觉样本，补充212完整pilot attempt/9400文件/70,741,015,728字节独立复制且写后SHA回读完成；此前五目录复制完成不意味着accepted source依赖闭包已完成。新增发布流程在全数独立文件/身份/split/统计检查、四组12真实窗口和全量numeric哈希通过后才发布归档manifest.ready.json。原来源不删；原生回放闭包仍未完成。

参考用户80小时/Aloha549775窗口×10epoch，新混合7282176窗口/epoch，global512为14223 updates/epoch；同窗口吞吐1epoch105.97小时、2epoch211.93小时，不能因累积翻倍而再除2。详情docs/MULTISIM_ARCHIVE_V2_4A100_HANDOFF_20261025.md，证据outputs/archive_v2_20261025/training_time_estimate.json。


### 2026-10-25T15:40:07.630969+00:00：独立归档训练清单发布与global512实数据核查

正式manifest.ready.json已发布，SHA a96d7f13dcb243f9429303a667d76d02151098a83c2946d996a31b21bf4a07aa。全部52204条numeric哈希与已完成写后SHA回读的物理复制ledger一致；全部numeric/media当前大小、mtime、独立inode检查通过，保留29101家族原split，12真实窗口通过。未额外等待一次相同全量串行重读作为启动前提；copy-time哈希证明与当前stat/身份核对范围明确记录training_release.json。Python/io/h5py读取拦截旧dataset根后，四来源12窗口仍通过；这是prepared loader独立性证据，不是系统调用沙箱或原生仿真回放。global512实际mixture长度7282176、每epoch14223更新/56892微步。当前用户选择1epoch，新入口及Slurm模板按500optimizer-update WAM延迟更新，旧训练不变。完整Foundation工程短测正在GPU5/6加载，尚不声称联合更新/保存恢复或目标四A100 bs32通过；原生回放闭包仍待验，来源不得删除。


### 2026-10-25T17:24:12.501131+00:00：用户授权40k独立分叉，联合阶段EA heads-only训练至原10epoch终点

原因：用户要求保留原训练逻辑，同时检验共享Sidecar主要服务Action生成。新增可关闭aux_heads_only_after_warmup，默认False；仅training joint且warmup结束后，对三个EA heads调用的tokens/features输入detach，原tokens继续跨tap与Write，Action/Video路径和全部loss权重不改，中间deep supervision同路由。head参数532493，占Sidecar24290323的2.1922%；detach不冻结heads，但不保证关系语义或OOD保持，需后续GT/闭环验证。旧配置默认行为不变。新增ea_aux_heads_only训练指标只在启用时记录。

独立40k fork：outputs/sidecar/aloha50_heads_only_from40000_to10epochs_20261025T171338Z/runs/from40000；原run3501138继续，不终止。完整状态13文件110179810280字节及部署资产/40k权重共21文件135063633621字节均物理复制、写后SHA回读、独立inode验证通过。初始40k权重移到新root/initial_checkpoint免受分支训练自动清理，并记录relocation hash；原路径完全不写。仅复制的config改aux路由与resume/output/finetune记账字段；原2K warmup历史、bs32/acc2/world4、1e-4 WAM/5e-5 Sidecar、cosine与10epoch预算保持。来自40000micro/20000opt/epoch9，继续到42960/21480，共2960微步/1480更新，不重新warmup，不再训练额外10epoch。

41项CPU tiny真实Wan/ActionDiT工程检查通过，包括FP32/BF16+checkpointed前向逐位一致、三tap及deep辅助梯度只到heads、Action/Video至共享和Write保留、500边界、optimizer恢复与输出隔离守卫；不冒充真实40k四卡恢复验收。新增专用resume入口复用原CPU/GPU handover和metrics，强制目标为新run精确40k状态、world4及原预算，并在真实restore后输出resume_verified。当前只有5/6空闲，0–3原run仍训练；新分支将CPU预加载并等待0–3自然释放，不抢卡或杀原实验。真实更新和最终SR尚未验证。


## 2026-10-26T02:28:29.828612+00:00 — 用户授权下一轮完整RoboTwin共享EA梯度退火

用户明确“按刚才方案把下一个实验接上”。保留原2K full-backward/WAM仅跳optimizer更新、Foundation/Clean2500/global256/10epoch/LR/ColorJitter。新增默认关闭的aux_shared_gradient_scale和aux_gradient_anneal_updates；只对三个tap送入EA heads的tokens/features做前向恒等梯度缩放，含deep supervision；原tokens继续Write/后续tap。联合后500 optimizer updates线性rho1→.01，heads、Action/Video监督不缩放；旧硬heads-only默认与行为保留且禁止冲突配置。无新增参数，不改旧数据/实验。新增独立config和快环境4卡入口，复用formal gate，允许等待前序正常完成marker以防失败后误接续。当前heads-only自然完成后才释放新模型到GPU0–3；不终止旧实验。工程测试/启动证据稍后记录专用实验文档，当前不预称SR提升。


### 2026-10-26T02:38:11.028179+00:00：共享EA梯度退火新实验已真实CPU就绪并安排接续

用户已授权，独立root outputs/sidecar/aloha50_ea_grad_anneal_10epochs_20261026T022844Z，实际run outputs/sidecar/aloha50_ea_grad_anneal_10epochs_20261026T022844Z/runs/2026-10-26_02-36-51。监督3948883/torchrun3948895，2026-10-26T02:36:52Z四rank权重/数据/原scheduler已CPU准备完成；实际合同549775 windows/world4/bs32acc2/2K WAM warmup/rho .01/ramp500均验收。与原full-backward实际config差异仅新增EA路由默认/目标与ramp、输出及同manifest解析路径，无额外LR/数据/Video改动。56 CPU工程回归+6 gate/监督检查、GPU5/6双rank BF16/ZeRO2累积及五边界保存恢复检查通过，GPU短测已退出。

0–3原heads-only继续，未杀前序；正常exit0+四rank42960/21480+完整最终weights及显存释放后自动接续，不错误要求最终optimizer marker。当前**新正式run尚无GPU真实更新/正式2K边界或SR结果**，这些继续待验收；supervisor持续核验有限loss/rho，首更新写startup_acceptance。fast runtime复用旧shm环境，全部新产物在共享盘。专用docs/SIDECAR_EA_GRADIENT_ANNEAL_FULL_TRAIN_20261026.md与独立root engineering证据。1%候选、语义保持、clip间接影响、Vanilla/near/EEF回放等限制不撤销。


## 2026-10-26T04:19:57.699625+00:00：Heads-only42960与原warmup42960复用既有离线诊断完成

用户要求参考旧方案比较写回、40k后loss和实际Action精度。完整日志40001–42960（2960微步）同step/LR但增强未逐bit配对：Action+1.128%、Video−.188%、EA−.0189%；末500Action+0.980%，未见训练目标明显改善。最终beta绝对值小约1.064/.532/.505%，Read投影从40k的L2移动更少、Write输出仍约.16–.17%，不据weight norm推效果。

复用原bootstrap/WriteProbe、相同25GT（24 heldout六episode+1 seen校验）、32固定FM样本，GPU5/6各一模型、10step/seed42/eager/no-cache/Full-Off，不做optimizer或改生产代码。两版各50生成和32FM对全部完整，重复逐位相同、off真实delta零、参数/输入hash不变；加载gate与最终safetensors一致。进程退出、GPU5/6已释放，0–3新训练保留。独立证据outputs/robotwin_eval/sidecar_heads_only_vs_warmup42960_20261026T040447Z及docs/SIDECAR_HEADS_ONLY_VS_WARMUP42960_DIAGNOSTICS_20261026.md。

真实write-r三层原.1848/.1645/.2639%、新.1907/.1674/.2627%，没有明显增强；Full/Off归一化Action变化Clean .379→.391%、Random .343→.340%。Full32步专家XYZ Clean7.0419→7.0879mm（略差）、Random14.9150→14.7271mm（略好）；32步旋转两域略好，gripper均略差。配对Action FM Clean+7.269%、Random−3.267%，均只有6/16批更优，非全面改善。Random Write的XYZ误差影响原+ .0262mm→新− .0430mm，但关Write时新模型也好.1186mm，不能将全部.1879mm改善归因Write。

晚期detach未明显损害此批Contact/Moving/方向，但Random presence/track按既有GT mask的accuracy仅54.2%/50%且两版相同，Contact F1=1不能说明全部关系可靠。新增发现限定这批GT可见性/有效性定义，尚未证明物体真实不存在或任务失败原因。窗口重叠、不报SR或显著性、不外推早期soft退火；保留near/EEF物理回放/匹配Vanilla/worker RNG边界。下一步依原计划做最终模型配对闭环，新完整soft实验继续，不叠Gate/网络/LR修改。


## 2026-10-26T13:23:07.940932+00:00 — 用户授权四组2K→3K更新控制实验与0–3连续接续

用户明确要求实现A原配方/B现有1%EA退火/C WAM Ramp/D Write LR×2，依次GPU0–3、每组1000 optimizer updates，下一模型CPU权重及CUDA上下文就绪后才停止明确的当前torchrun。统一S=用户指定accel_state_step_4000（micro4000/opt2000/epoch0），独立物理复制21文件86894380606字节且逐文件SHA写后回读/独立inode通过，冻结于/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_four_arm_1k_20261026T131106Z/common，不覆盖原run。所有新输出独立。

最小opt-in optimizer hook：C WAM lr原cosine×(.1→1/300updates)，post-hook恢复原LR避免递归scheduler污染；D保留所有原ZeRO组和master/moments，使用克隆Write片段与moment/step、相同AdamW functional在2倍LR上算参数更新，普通step只推进一次原moments，post-hook在ZeRO同步前只拷回Write master。不是梯度乘2或模型输出放大。所有参数、数据/增强/loss、bs32acc2/10epoch/21480-update scheduler保持。

专用入口复用正式trainer与gate，恢复共同full-state，验证真实scheduler T_max81624/last_epoch8000；在4500/5000微步存weights，6000存weights+fullstate，随后同步正常退出，不改max_steps重建cosine。记录每rank实际batch/图像增强指纹，对B/C/D逐步核对A；初始Sidecar master/Adam指纹核对，防恢复不一致。正式旧run不被全局代码热改；仅新入口启用。

24 CPU检查加入口/heads-only/gate回归，以及GPU5/6双rank BF16/ZeRO2/acc2四模式实测通过：master更新与独立真实LR参考逐位一致、Adam moments不变动多次、原group布局保留、保存恢复下一步逐位一致。完整真实S四卡恢复/首更新待A实际接卡验证。实现文件sidecar_lr_screen.py、train_sidecar_four_arm.py、supervise_sidecar_four_arm.py；不新建环境、不下载、不改数据或原版架构。


### 2026-10-26T13:43:49.817949+00:00 — 五组1K短训已接卡，新增Main-task Gain×2受控候选

用户授权四组后又要求将Main-task Gradient Gain×2–4设计为第五组1K筛选。本次选择温和E_gain2，与A同EA1/Write LR/WAM LR，仅在2K更新后训练态对Write输入z做前向恒等、反向2倍；其余tokens/head/EA路径保留，不新增参数，默认1和eval行为保持。可能通过Read影响WAM及全局clip，Adam更新不保证翻倍；不外推为必有收益。E单独监督接D，避免重启已运行主队列，不与B/C/D叠加。

四卡A真实恢复4000micro/2000opt、原21480更新scheduler及首次联合更新已验收；13:41Z四rank4074/2037、有限loss、最近20微步中位6.62s。旧3948895已按核验身份停止，公共独立S副本保留。主监督4006566，E跟进4012262，离线诊断4012263。31项Gain/路由/LR/入口检查和3项跟进脚本检查通过（部分重叠），GPU5/6双rankBF16/ZeRO2/acc2 Gain模式保存恢复下一步一致；B–E全模型真实更新尚待队列执行，不预称性能提升。

自动诊断复用旧GT/Full-Off探针、50生成+32固定FM，逐组6000端点和固定文件哈希，GPU5/6有余量才启动，无外部作业终止；合并实际训练batch指纹、loss分段和动作/真实写回/关系指标。已有12k分析器重放和端点固定脚本测试通过，新3k结果仍待生成，未启动配对闭环。详细docs/SIDECAR_FIVE_ARM_1K_SCREEN_20261026.md，独立root outputs/sidecar/aloha50_four_arm_1k_20261026T131106Z。

继续保留near/EEF物理回放/匹配Vanilla、GT小样本及当前原数据依赖等边界；1K只筛选，不按gate或辅助准确率挑赢家。任何队列失败保留证据，不自动改变预算/超参、重建scheduler或广泛杀进程。


### 2026-10-26T16:41:46.643062+00:00 — 按用户要求B–E只保留+1K最终权重，继续五组连续队列

用户明确取消后四组中间权重，保持依次跑完。专用短训入口保存条件仅对A保留4500/5000，对B/C/D/E仅6000；末端完整accel_state_step_6000保留供获选方案续训，原训练数据/LR/loss/1000更新预算不动。不删除已有A或其它checkpoint。按当前每份24862363126字节，避免8份中间文件、共198898905008字节磁盘写入；不是GPU显存节省。

B已在旧代码上CPU就绪，不能靠改磁盘脚本假装热生效。因此核验其尚无GPU授权/训练日志后，仅终止其预加载进程并归档日志/标记，重载B；A训练torchrun4006583持续。新监督4026489附接A并启动B4026490，E4012262/诊断4012263保留。原监督4006566单进程暂停，保留A作为其未回收子进程的真实内核exit status，待新监督读到A退出码后终止原监督；不暂停GPU训练rank，不以完成marker替代exit0。新增附接仅用于本次保存策略交接，身份与启动tick核验。

7项工程检查通过，含真实子进程exit0/exit7附接、原身份守卫和诊断端点固定；AST保存条件核查通过。16:40Z新监督报告A5676微步/2838更新，仍相同PID且有限loss；B正在重载CPU，后续C/D/E自动接续，离线诊断队列无失败。完整B–E端点保存与实际配对batch验收仍待运行，不报告性能结论。证据checkpoint_policy.json、engineering/endpoint_only_handover_tests.log、engineering/endpoint_save_policy_check.json及更新后的五组执行文档。near/EEF回放/匹配Vanilla、小样本、旧数据依赖等未解决事项继续保留。


### 2026-10-27T08:30:44.110910+00:00 — 用户授权E后续A/C各1K，保留各自3K完整优化轨迹

本轮明确3K→4K optimizer updates，即6000→8000微步，顺序E自然结束→A→C；不执行报告中额外的3K→5K或F/去decay/结构修改。专用短训入口以plan可选source/start/target复用原逻辑，默认2K→3K不变，续训模式只在8000保存权重+完整state。A/C各恢复本轮自己的accel_state_step_6000，保留不同Adam历史/RNG，原21480总更新cosine不重建；验证scheduler last_epoch12000/T_max81624，C进度3000时factor=1，不重新ramp或warmup。

独立root /remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_AC_extend1k_20261027T082947Z；各3K权重硬链接固定、config/stats/contract/tokenizer复制，原110GB fullstate只做稳定文件身份清单并引用，不重复复制两份。旧run已结束不发生自动清理，不在源目录训练或覆盖。原/续训config唯一区别为输出路径和权重骨架路径；6k源13文件完整。12项CPU合同/恢复LR/端点/身份检查通过，真实四卡3K恢复及首更新待新A接卡验证。

新监督只在E真实exit0、6000/3000完整状态后释放A到GPU；现在可CPU预加载，A更新到7400时预加载C，A满8000保存并正常退出后才接C。数值batch每微步、增强抽六步与新A比对；A/C初始optimizer本就不同，不再错误要求两者相同hash。后续复用相同25GT窗口50生成+32FM和Loss对齐分析。全程只最终保存，原3K与新4K不混。关键限制：C稳定不等于SR变好，matched Vanilla/near/EEF回放与配对闭环仍待；不按单一loss自动启动10epoch长训。


### 2026-10-27T08:56:27.150503+00:00 — A实际续训已接卡与报告复核/性能候选审计

E于08:43:14Z正常结束且完整3K已保存，原离线诊断继续。A在0–3由自己的3K状态真实恢复，metadata6000/3000、epoch1、skip1704、scheduler12000/Tmax81624、首更新3001有限；08:50Z已3010更新，未中断E。源A/C的13文件完整state身份均未变。C按原3000进度接续时不重新ramp，端点8k微步/4k更新仍待队列验收。三份监督4123250/4125884/4125885分别负责A→C、原协议GT/FM诊断、2K→4K连续Loss图。

完整阅读deep-research-report (1).md后已记录docs/SIDECAR_AC_CONTINUATION_AND_NEXT_OPTIMIZATION_20261027.md。历史原run4K权重在step8000诊断快照中仍在，但本轮共同S后的A4K尚需生成，不能当严格本轮Control。gate真实optimizer组2 wd=.01已审计；A纯decay .04938%、D beta .09873%，直接作用不足解释beta大幅收缩，不调整A/C decay。F静态vector gate可折入可训练Wout，主要优化重参数化；scalar转vector还涉及ZeRO/Adam迁移，前向保留不等于完整恢复保留，本轮不实施F。

用户追问batch等涨点方法已分析：优先A/C、开发seed选checkpoint、有限离线插值候选；条件性后期Video/WAM LR、语义保持增强，global512后置。新发现未来acc2→4 resume会被基础trainer按global_step/acc误算opt_step（6000/4=1500而实际3000），scheduler/样本预算也需处理；当前A/C强制acc2不受影响，未修或启动512，未来改变累积前为必查项。当前执行器确为预测32/消费32，无默认ensemble；若测horizon16/8需独立协议、SO3/gripper处理及所有baseline同配。没有新增训练trick、数据修改或评估协议变更。继续保留near/EEF回放、matched Vanilla、闭环小样本与同起点归因边界。


### 2026-10-27T09:23:38.867547+00:00 — 用户授权C 4K后立即续原10epoch，评估并行；每2000次更新保存

用户在本轮明确要求：A/C各自3K→4K短对照照常完成，C到4K后不等待收益诊断结论，默认继续原10epoch预算，评估并行；若后续发现问题再考虑切换。这替代本文件及此前计划中“需先评估再启动长训”的本轮执行门槛，不代表已确认C提升SR。

新增独立队列root `/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_C_full_from4k_20261027T092219Z`，GPU0–3。复用专用入口的显式full_training模式，仅允许8000微步/4000optimizer updates→42960微步/21480更新。C短训到7400微步时预加载CPU权重骨架；骨架复用已固定C3K权重（硬链接，未复制环境或110GB状态），仅用于提前构建模型。必须等C4K完整状态标记、13文件、配套权重及短训torchrun真实exit0均通过后才授权GPU，随后通过原Accelerate/DeepSpeed恢复完整4K model/master/moments/RNG/scheduler。恢复counter与原cosine last_epoch16000/T_max81624、C factor=1均强制核对，warmup/ramp不重启，未改网络/loss/EA梯度/数据/增强/batch256/decay。A/C短对照的默认入口行为保持不变。

原4K输出与正式续训目录隔离。按用户最新要求，save_steps=4000微步且accumulation=2，即每2000次真实optimizer更新保存权重+完整state；末端42960微步/21480更新额外最终保存。沿用原keep_last_k_ckpts=2滚动保留，仅作用于新正式run，不删改旧3K/4K或评估输出。32K微步权重完成保存后仅硬链接固定一份weights/config/stats/tokenizer供历史最佳时点对比，不另复制optimizer，也不产生额外训练保存。

必要GPU/ZeRO恢复存在不可消除的交接时间；通过提前CPU加载缩短空窗，不声称零耗时。当前真实C4K恢复和首次4001更新仍待队列，工程测试与控制流模拟不等于GPU/SR证据。待评估结论仍保留near-plane、EEF replay、匹配Vanilla、少量GT/闭环样本等边界；不引入新trick或自动超参搜索。

工程验收补记 2026-10-27T09:24:46.822810+00:00：20项CPU合同/真实子进程守卫回归/完整队列模拟通过，Ruff与编译通过。监督PID4137756已启动且状态waiting_for_C_preload_threshold；前序训练/离线诊断/Loss监督均核验存活，未提前加载额外GPU模型或停止现有作业。真实C4K恢复仍待执行。


### 2026-10-27T14:25:10.048560+00:00 — 用户授权保留A/C，先GitHub备份再移植PatchWAM C2R增强

原因：只用Clean训练的外观泛化候选；用户要求按PatchWAM参数做最小接入，不中断当前短训和C正式续训。GitHub原版本备份分支backup/pre-patchwam-c2r-20261027T140234Z、远端66bb8d2056c84c55757f1d3c7a43edc0bced57ad，tree8b99554366c67bf534657bf2f2d33811a09ad1d1与本地完整快照相等。直接git push的VS Code凭据失效，已通过连接GitHub App完成，不导出token、不改旧分支。

具体改动：固定官方PatchWAM dea19d21182a26a8ee16efc2c4125ee7aecf44fb，完整源码1.8MB在.external/PatchWAM；appearance.py按MIT字节原样vendoring，新增PIL三相机clip适配。EARSIGLiteDataset/LiteRoboTwinSource增加默认null的patchwam_augmentation，只在train且启用时接入；新recipe逐项匹配官方p=.8、photometric、AdaIN p=.5和Fourier p=.5及全部细参数，关闭旧jitter并拒绝双开。不改模型/action/EA/mask/归一化/几何/数据文件。缺失相机由HDF5存在性识别、排除统计/增强且保留黑槽，真实黑像素不排除。当前multisim reader的旧缺失黑槽问题仍未解决；本修复保护范围仅新Aloha增强路线。

独立训练配置及scripts/train_sidecar_patchwam_c2r_4gpu.sh沿用当前C的2K full-backward/WAM延迟更新、原300-update ramp、global256、10epochs、4000微步=2000更新保存；不叠报告的100-step ramp/结构/额外3D监督。默认从Foundation新开独立run，不覆盖旧A/C，不声称已实现共同2K state短训分叉。GPU资源尚待用户指定，本轮无新GPU模型/训练启停。

验收：32项CPU回归、Ruff/shell/Hydra通过；12seed×3相机×9帧独立上游oracle逐元素相等；真实三任务窗口所有非图像字段相等，关闭新开关与Git备份旧reader的图像/监督/Python及Torch RNG一致；真实OmegaConf/registry构造50reader549775窗口，2worker3批共6样本输入合同通过。官方tensor类原样；OpenWAM采用既有resize后增强/PIL uint8量化，非PatchWAM整条数据管线逐像素复现。热缓存9样本单线程reader中位旧.095s、新.491s，强制增强约.398s/样本；不是GPU稳态step-time/SR验证，后续需观测loader等待。

证据/remote-home/jinminghao/OpenWAM/outputs/patchwam_c2r_implementation_20261027T140234Z；分析与启动docs/PATCHWAM_C2R_AUGMENTATION_20261027.md。颜色语义任务、强噪声/可见性、CPU吞吐、clip统计耦合仍有风险，不承诺涨点。继续保留near-plane、EEF replay、matched Vanilla、有限GT和真机标定边界；旧C与续训监督均在运行，未给它们打开新增强。


### 2026-10-27T14:56:05.214509+00:00 — 用户改为PatchWAM增强＋100-update WAM ramp，从Foundation完整10epoch，替换C长训接续

用户明确授权：当前C的3K→4K额外1K短训及评估保留，暂停执行原C4K续训10epoch；新实验从原始Foundation开始，Sidecar按原初始化重新构建，前2K正常forward/backward/read/write但WAM不更新，此后100次更新窗口LR倍率.1→1再joint。新训练计数0→21480更新/42960微步，不从C继续，不延长成“再10epoch”；数据/EA/模型/bs32acc2/global256/10epoch及每4000微步=2000更新保存保持。

最小代码：OpenWAMTrainer把training.sidecar_wam_ramp_updates传给现有SidecarLRScreen，缺省仍300，新增强配置显式100；optimizer组、moments和cosine保持，post-hook恢复scheduled LR，不用梯度放大替代LR。正式入口CPU合同增加真实augmentation启用、source/resume/counter/ramp字段；JSON metrics增加已有optimizer hook的scheduled/effective LR轨迹，供2K边界实测。原C在内存的代码/配置未热改，默认300回归通过。

队列root /remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_patchwam_c2r_ramp100_10epochs_20261027T144939Z；scripts/supervise_sidecar_patchwam_after_c.py复用原launcher、CPU-ready GPU gate及C完成状态验收。新torchrun4168821、监督4168818已启动CPU预加载，加载Foundation checkpoint_step_154000.safetensors；仅等当前C真实exit0且8000微步/4000更新完整state+权重后授权0–3，C离线GT/FM诊断仍原GPU5/6队列并行。原C长训监督4137756在无child/无GPU授权时按身份核验停止，旧root不删改模型与数据、只写cancelled并链接新root；没有停止当前C、旧A/C诊断或未知其他GPU作业。

验收51项CPU测试、Ruff、fast preflight、Hydra resolve通过，含100/300真实Adam更新与恢复、训练工厂传参、默认短训launcher不变、fresh源/zero progress/augmentation/R100合同、必须C真实exit0才放行新训练的队列模拟。当前真实新GPU首更新和2K后100步曲线仍待未来执行，不能据配置通过称已训练或收益提高。官方增强CPU开销/颜色语义及near/EEF replay/matched Vanilla等既有边界继续考虑；不新增100步以外的结构/损失trick。完整说明更新docs/PATCHWAM_C2R_AUGMENTATION_20261027.md。

按用户要求已展示并保留三任务对比图，另生成同场景多seed及0/16/32帧对比，保存到outputs/patchwam_c2r_implementation_20261027T140234Z/；保留官方p=.8，seed42未增强也展示，所有非图像字段核对不变。这些是数据图像效果，不是策略SR实验。

验收补记 2026-10-27T15:01:01.391132+00:00：四rank CPU预加载于14:57:16Z完成，真实cpu_contract核验Foundation来源、resume=null、completed_optimizer_updates=0、WAM warmup2000、ramp100、PatchWAM augmentation实际启用、global256全部通过。新监督4168818及torchrun4168821按start-ticks核验存活；原C短训/诊断/Loss监督也均存活，当前约3630更新；原C长训监督已停止且从未生成子模型。GPU release与predecessor_verified尚不存在是按计划等待C短训结束，并非新训练已经更新。证据engineering/final_scheduling_acceptance.json。


## 2026-10-28T13:06:23.538558+00:00 — 用户授权bitdistill迁移与2K权重8卡新finetune

新增独立8卡PatchWAM启动脚本和配置：加载指定2K完整模型weights，新optimizer/scheduler/counter=0、10epoch、batch32×8×acc1/global256；用户明确直接联合训练，无2K延迟更新，无额外WAM ramp；保留标准5%LR warmup/cosine和EA1。每10000真实更新保存，最终另存weights。复用原cluster metrics入口，不改模型/数据/正在运行实验。打包工具复用现有PAX/流式hash归档，write_archive新增默认兼容的compress开关，仅大权重用plain tar；复用manifest SHA完全相同的已完成2500轨迹归档。两份权重独立自包含目录，原2K统计与训练清单绑定；只携带官方Clean2Random，第二版本等用户说明。不复制optimizer或环境。迁移文档docs/BITDISTILL_TRANSFER_AND_8GPU_20261028.md，目标/mnt/100T/lixiangyu/cuigm/bitdistill/OpenWAM。


## 2026-10-29T01:39Z — bitdistill首轮H100八卡启动的部署修复

用户授权立即在空闲整机8卡上启动上传的2K weights-only新10epoch；选择aigc24。未改模型、数据或训练core源码。实际处理：(1)项目内libcuda.so指向现有真实driver，TRITON_LIBCUDA_PATH/LIBRARY_PATH持久写入项目env，真实CudaUtils编译通过；(2)只对该训练设置HDF5_USE_FILE_LOCKING=FALSE，默认只读API的NFS errno37复现及禁锁验证通过；(3)zero-backbone-warmup不生成ea_backbone_group，control诊断hook在首step要求此标记，故用现有CLI覆盖training.sidecar_lr_screen=null。20步CPU parity证明与合法标记control的权重、Adam states及warmup LR逐位一致，不改5%LR warmup/cosine或优化器数学。原包装脚本备份、原配置及失败run保留；未向其它实验启用这些训练覆盖。

当前run outputs/sidecar/patchwam_warmstart8_aigc24_20261029T013934Z，父PID1709151，2K SHA49b6c013…8383835b，global256=8x32acc1，新optimizer/step0、num_epochs10、save10000/20000及最终产物沿原配置。启动验收与历史失败索引见docs/BITDISTILL_WARMSTART8_TRAINING_20261029.md；八rank首个有限loss更新尚待真实日志，不以父进程存在当训练已成功。


2026-10-29T01:47Z验收补记：aigc24当前run八rank共同完成5个真实更新、loss/grad有限且rank归约一致，total=.2973007858，video=.2159739435，action=.01763792895；原8x32acc1/global256、10epoch/21480和1074标准LR warmup保持。真实CUDA allocated峰值47.92GiB。当前仅启动验收，训练与后续checkpoint继续，见专用H100文档及run/startup_acceptance.json。


### 2026-10-30T02:22:18.753244+00:00：官方Clean2Random起点的Sidecar2K + 低LR联合5epoch（用户授权，配置/调度）

用户要求本地保留配置并同步bitdistill aigc24八卡，已有10epoch实验完成后串行执行。新增独立两阶段配置及接续器，不改core model/trainer/optimizer或旧实验。阶段1从官方RoboTwin微调10740权重初始化Sidecar，stage=joint/backbone_warmup_updates=2000/isolation=0，full backward+全局裁剪后仅Sidecar AdamW更新；acc1严格2000步退出。阶段2重新加载该final权重，新optimizer/counter，5epoch；Video/Action/base 2.5e-5、Sidecar1.25e-5，均降至此前1/4；标准5%LR warmup+cosine，原PatchWAM/EA/直接Action Write保持。不叠仅Action结构消融。阶段1按原策略全部2089张量逐位不变验收才接阶段2；原实验21480 final及八卡空闲门槛先通过，不抢占其它作业。

官方与数据stats SHA不同但EEF min/max完全相同，实际min-max映射保持，其他moments不能据此称全文件相同。复用只读HDF5禁锁和项目CUDA链接；lr_screen=null规避既有zero-warmup兼容问题，不宣称全局修复。新实验weights-only保存，减少NFS完整optimizer状态IO，代价为不能精确resume；所有旧权重/输出保留。NFS锁缺失以节点socket互斥，仅限本节点控制器，不保证外部排他；新的GPU容量/真实2K冻结验收/5epoch完成/闭环收益尚待运行。证据与完整配方docs/BITDISTILL_C2R_TWO_STAGE_20261030.md。


### 2026-10-30T13:36:30.219453+00:00：用户授权旧warmup2K→XA联合10epoch（增量实现中）

修改前完整源码已推送GitHub Jamee11/MyWAM backup/pre-alignment-20261030，24328162a20dabb950c6bbf4c1ed44951673cb03；排除下载缓存/环境/权重，505代码/配置/文档文件留存。用户明确使用最初传输的Sidecar-PatchWAM-Warmup2K/checkpoint_step_4000（2000 optimizer updates），排除今天官方C2R→2K。新opt-in cross_appearance配置新增student Action block8→FP32 EMA teacher block20、512 MLP、delta[0,.2)、名义原图/增强图配对；保持旧默认路径无新增训练参数/损失。原有EA/Action/Video与gate×1保持；新增loss按平滑EA标量目标0.8、系数上限0.5、200update渐入，不把标量对齐冒充梯度对齐。读出使用显式返回，避开checkpoint重算副作用。

新增保留增强前clean clip选项（默认false）；XA增强中等亮度/对比/噪声，禁Hue/AdaIN/FFT避免颜色语义变化，不改几何和EA labels。10epoch、原1e-4/Sidecar5e-5/global256；8×16acc2给FP32 EMA留显存，实际吞吐/显存尚待验收，不承诺一天完成。新配置按optimizer5000间隔保存且保留final，未改旧配置计步。独立停止条件仅用于smoke，不截短cosine。weights-only导出不含EMA/optimizer，不支持精确resume，禁止误走resume入口。当前实现测试/远端部署尚未完成。

2026-10-30T14:06:35.472420+00:00增量修正：真实unified EEF padding支持[B,T,80]；teacher FP32 EMA直接引用+逐算子BF16 autocast/cache关闭，避免整模型BF16临时副本导致OOM；按optimizer计步导出附完成marker。用户最新资源约束将中途诊断迁回本地A100，H100仅新增一个8卡训练，不启动额外H100评测。

2026-10-30T18:02:00.409017+00:00用户最终要求串行接C2R：XA launcher改用torchrun standalone+loopback local-addr自动分配端口，消除10步退出后的固定端口EADDRINUSE；新增可选C2R完成依赖（8rank10740、final保存日志和safetensors完整验收，仅作为启动条件，不作XA初始化）。16×acc2若仅OOM则降8×acc4，global256保持；通过10updates后自动正式10epoch。增加CPU-only八rank通信连续两次自检，默认旧版训练launcher未改。
