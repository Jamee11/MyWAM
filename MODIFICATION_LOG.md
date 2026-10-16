# Modification log

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
