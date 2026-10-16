# OpenWAM RoboTwin 标准评估：核查与运行记录

2026-10-16 UTC。本文件持续记录本次实现和真实运行证据；启动命令不等于验收通过。

## 项目文档的逻辑

- `2026-10-13_OpenWAM_alpha_EA_RSIGv3迁移分析.md` 是历史 V3：native14、40步、独立观测 DiT pass，不能当当前 Lite 的合同。
- `OpenWAM-alpha_EA-RSIG-Lite_2026-10-14.md` 定义新的研究主线：保留 Foundation 80D unified action/state、32步绝对 EEF、联合视频动作 FM；当前帧 VAE patch 特征产生 Ground/Relation/Progress/Plan，15×512 条件通过零初始化残差进入动作流。
- `OpenWAM-alpha_EA-RSIG-Lite_实现与四卡调试准备_2026-10-14.md`、`EXPERIMENT_RESULTS_AND_ANALYSIS.md` 和 `思考与隐患.md` 是实施/逐次证据，较新的条目覆盖旧的状态描述，但不追溯改变历史实验。
- 10月15日有学习率/增强、官方 baseline 数据、腕相机 near-plane 和训练 RGB 复核。自采旧数据、wrist020、官方 Clean50 是不同数据版本，标签和统计不可混用。集群 baseline 交付也不构成本机闭环证据。
- `assets/openwam_usage_docs/` 为上游使用文档；`benchmarks/robotwin/README.md` 是现成评测入口。本次先使用独立的官方 Clean2Random checkpoint，不评估历史 EA smoke 权重，不操作其他训练/评测。

## 已存在的评估流程

仓库已有 `scripts/deploy.sh` / `scripts/deploy.py` 模型服务、`single_eval.sh` / `multi_eval.sh` 仿真入口、`eval_policy_wrapper.py`、`openwam2robotwin_interface.py`，以及可选 Labtasker 调度。应复用这些实现，不能再搭一套重复的 seed/成功判据。

服务使用 OpenWAM `.venv`；仿真使用独立 `dit4dit-robotwin2` 环境，通过 WebSocket 通信。仿真 wrapper 动态加载 RoboTwin 的 `script/eval_policy.py`，默认保留官方专家筛 seed 和策略闭环。

标准完整分数要求：50个 Aloha task × Clean/Randomized × 各100个专家验证通过的策略 episode，seed group 0（候选从100000开始）、unseen 指令、官方 RT shader 与 task step limits；策略失败计入分母，专家失败跳过。单条/少量 episode 仅为链路检查，不能外推全量 SR。推理使用 checkpoint 原生32步 chunk，sync、10步去噪、DiT cache 开（0.99/3）、compile 开，见 `configs/deploy.yaml`。若为排错关闭优化，必须另记配置并恢复验证。

## 当前来源与差异

- 官方权重：`assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Clean2Random/checkpoint_step_10740.safetensors`，24,813,767,464 bytes；下载验收记录 revision `9bb966823547bc959800b45e9bb4005e9c2a3db3`、SHA256 `2cab63b550ff45a52fbde4adc40ee2bdb7d4bafe5fa0dd6879e41e22ca2ff95f`。本次启动离线模式，不重新下载。
- 本地仿真：`/remote-home/jinminghao/DiT4DiT/.external/RoboTwin`，HEAD `248266bee01e4bfa9be38be9055a9cd5b4281821`，已有 `envs/_base_task.py` 和 `script/test_render.py` tracked 修改；不由本次改写。
- OpenWAM 文档验过的是 `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4`，本地 git 没有该对象。因此本次可验证本地官方评测入口协议，但不能直接声称精确复现论文依赖栈或分数。
- `.external/RoboTwin_official_pristine_20261007` 是另一布局（`scripts/`、`env_cfg/`），没有目标 `script/eval_policy.py`，不能只因名字包含 official 就替换 runtime。
- 当前硬件为 L20 46GB，并非历史文档的 A100 80GB。本次初始服务用物理 GPU7，仿真 GPU1；不终止其他作业。

## 必须核实的合同和已发现的问题

1. RoboTwin `_trans_endpose` 用 transforms3d 的 `mat2quat`，实际输出 wxyz。但原版 OpenWAM reader 和默认客户端把输入按 xyzw 数值转换。已存在的 `source_quaternion_order: wxyz` 会同时重排输入与执行输出，适用于绑定了真实物理旋转合同的本地 EA/Baseline；不能仅看 runtime 输出就认定适合官方发布权重。官方权重训练是否已预转换，需要以训练来源/真实预测诊断为证，首轮先复用发布的默认适配，不静默更改模型动作合同。
2. `ROBOTWIN_TEST_NUM` 现实现只覆盖 `eval_policy(test_num=...)`，上游 main 写结果仍使用局部 `test_num=100`。单条检查应以 `Success rate: suc/actual` 日志为准；原 `_result.txt` 在小样本时不能当正确 SR。正式100条不受该分母问题影响。修复评测逻辑已单独请求确认。
3. 默认 `step_limits.yml` 全部注释，不改上游预算；`ROBOTWIN_ENABLE_PLANNER_FALLBACK` 不开启。`SAPIEN_DISABLE_RAY_TRACING=0` 保留 RT，不因提速更换 shader。
4. 初次 compile、cuRobo 扩展/专家规划可能慢；必须确认实际进程和日志，不以 READY、端口、CPU单测或模型加载成功冒充 rollout。

## 本次初始命令与证据

输出根：`outputs/robotwin_eval/official_clean2random_20261016/`，服务 `server.log`，单条 Clean `client_clean.log`，仿真结果隔离在 `runtime_clean/eval_result/`。

```bash
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
.venv/bin/python -u scripts/deploy.py \
  --ckpt-dir assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Clean2Random \
  --host 127.0.0.1 --port 18848

ROBOTWIN_PATH=/remote-home/jinminghao/DiT4DiT/.external/RoboTwin \
ROBOTWIN_PYTHON=/remote-home/jinminghao/miniconda3/envs/dit4dit-robotwin2/bin/python \
ROBOTWIN_RUNTIME_ROOT="$PWD/outputs/robotwin_eval/official_clean2random_20261016/runtime_clean" \
ROBOTWIN_TEST_NUM=1 SAPIEN_DISABLE_RAY_TRACING=0 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
bash benchmarks/robotwin/single_eval.sh adjust_bottle demo_clean \
  official_clean2random_smoke 1 18848
```

截至本条记录：新 PyTorch 进程确认 `2.7.1+cu128`、CUDA可用、8卡可见；服务和仿真仍在加载，尚无真实 rollout 终态。下一步：等待实际依赖/权重加载，完成 Clean 与 Randomized 闭环，再验证正式100条入口及全50任务结果验收；有代码修改须先遵守 AGENTS 确认规则。

### 首轮环境失败及第二轮设置

`client_clean.log` 首轮 exit1：cuRobo 预编译 `kinematics_fused_cu` 要求 GLIBC_2.32，当前系统为2.31；触发自动JIT后，PATH找不到ninja，最后任务import无法导入 CuroboPlanner。没有进入专家或策略，不计入策略失败。现有18项客户端proprio/图像传输回归通过（173.76秒），只证明CPU合同测试。

核对本机已有可执行 ninja1.13.2 与 CUDA12.4 nvcc。仿真环境迁移后实际 torch 也是2.7.1+cu128，不能照抄旧经验文档的2.4.1版本。第二轮保留首轮日志，新输出 `client_clean_toolchain.log`、`runtime_clean_toolchain/`，仅补充环境：

```bash
export PATH=/remote-home/jinminghao/miniconda3/envs/dit4dit-robotwin2/bin:/remote-home/jinminghao/miniconda3/envs/nvcu121_cuda-12.4/bin:$PATH
export CUDA_HOME=/remote-home/jinminghao/miniconda3/envs/nvcu121_cuda-12.4
export TORCH_EXTENSIONS_DIR=/tmp/openwam_robotwin_torch_extensions
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9
```

此设置让cuRobo自行编译至独立缓存，未修改原.so、外部RoboTwin源码、仿真依赖或planner；CUDA toolkit12.4与torch12.8的兼容性仍须实际编译/运行验证。服务保持同一存活进程，不重复加载。

## 用户确认后的本地化与评测层实现

用户明确确认评测层修改，并要求“copy过来”“不要影响以前的DiT4DiT的评测流程”。因此已停止初始引用外部runtime的本次自有客户端，只保留OpenWAM模型服务；原DiT4DiT作业不操作。

`scripts/copy_robotwin_runtime.py` 已完成 `.external/RoboTwin` 私有副本：22,775文件、逻辑16,921,222,101 bytes（du的磁盘分配大小更大）；三个外部资产链接全部物化，源/副本inode不同，源码/配置逐项SHA256相等、全部文件size相等、目标无软链接。bulk资产未逐字节完整hash，不将size验证冒充全内容hash。保留源git commit与tracked diff，排除旧data/eval_result/logs，未复制旧评测结果充数。清单：`.external/RoboTwin/OPENWAM_COPY_MANIFEST.json`。只读复核原checkout仍是原来的两个tracked差异。

仿真环境另复制到 `.venv-robotwin`，正在进行；完成后只改副本的editable cuRobo路径为本地 `.external/RoboTwin/envs/curobo/src`，正式入口默认这个新环境。复制期间的预编译/首轮检查显式使用旧环境的**只读已安装依赖**、`PYTHONDONTWRITEBYTECODE=1`，cuRobo源代码已来自OpenWAM副本，JIT仅写 `.cache/robotwin/torch_extensions`；不向原环境安装、不覆盖原.so、不改旧源码。当前已编出本机 `kinematics_fused_cu`、`geom_cu`，其他模块继续；完整MotionGen/规划/闭环尚待验收。

评测层改动：

- `eval_policy_wrapper.py` 的小样本计数同步修改上游main的局部test_num赋值，使用AST仅在内存中替换，不写入任意RoboTwin checkout；上游结构不符明确报错，默认未设置override时main不变。修复实际rollout数与原生_result.txt分母不一致。
- `standard_eval.py` 复用既有`single_eval.sh`，并从既有`multi_eval.sh`取50任务表。默认100条、双域、all任务，固定原生unseen/seed0/RT/原预算，禁planner fallback。每个task/split独立runtime与日志，记录真实checkpoint SHA256、stats/config/runtime/adapter/URDF hash、RoboTwin diff、依赖版本。新服务带readiness token；复用服务核对进程checkpoint/GPU与端口实际归属，且不会由runner终止。
- 结果验收必须同时具备exit0、连续1..N条episode记录、递增有效seed、成功计数合法、原生结果完成标记、unseen header和正确分母；策略失败仍计分。少量episode明确`smoke_only`，子集不标为完整50任务benchmark。中断/缺失/NaN/错误分母不通过。失败即保留证据并停止该队列，不自动改协议或重跑成功seed。
- 原18项客户端/图像测试与新13项共31项通过（`standard_tests_final.log`），最新纯评测13项另通过（`standard_tests_latest.log`）。这些是CPU证据，不是SAPIEN闭环。

当前服务进程的命令、端口归属及PONG已现场核对（`server_identity.jsonl`），PONG representation=eef。默认数值编码/解码对非平凡四元数数组往返误差约7.17e-8；说明相同数组约定可互逆，不独立证明发布权重的训练来源，也不授权替换本地EA的物理旋转合同。

首个新入口run：`outputs/robotwin_eval/official_clean2random_20261016/smoke_local_runtime_shared_dependencies/`，`--tasks adjust_bottle --episodes 1`，分别Clean和Random；名字明确此轮临时只读共享依赖，不把它冒充最终独立环境验收。

## 可复现入口

独立runtime首次复制（拒绝覆盖已有目录，源保持只读）：

```bash
/usr/bin/python3 scripts/copy_robotwin_runtime.py \
  --source /remote-home/jinminghao/DiT4DiT/.external/RoboTwin \
  --destination "$PWD/.external/RoboTwin"
```

正式入口如下；环境副本准备完成后执行。它默认自己启动/回收模型服务，只回收本次创建的进程；已有本机服务可追加`--reuse-server-pid <现场核实的PID>`，权重/端口/GPU不一致会拒绝。不要照抄历史PID。

```bash
.venv/bin/python -u benchmarks/robotwin/standard_eval.py \
  --checkpoint assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Clean2Random \
  --output outputs/robotwin_eval/clean2random_official100_NEW_RUN \
  --server-gpu 7 --sim-gpu 1 \
  --cuda-home /remote-home/jinminghao/miniconda3/envs/nvcu121_cuda-12.4
```

默认为50任务×双域×各100条，不是短测。先检查闭环时增加`--tasks adjust_bottle --episodes 1`并使用新的output目录；单task标准100条则只加`--tasks adjust_bottle`。`run.json`保存身份、完成范围、每项结果与双域macro SR；只有50任务双域全100条通过才会标记`full_50_task_benchmark=true`。当前不承诺论文分数可复现，运行版本差异保留在报告中。

### 复制验收补充与环境复现

初版忽略所有`__pycache__`以及根`data/`时发现上游实际上跟踪了17个pyc和`data/process_stuck.py`。已在OpenWAM副本补齐这些基线文件，改进复制工具保留所有现存tracked文件，并增加fixture验收；未触碰原件。最终manifest为**22,793文件、16,921,381,661 bytes**，私有checkout的tracked diff现与源相同（仅原两处历史修改）。首轮smoke启动时的初始manifest按当时SHA256精确重建并存为该run的`robotwin_copy_manifest_at_start.json`，避免后来补齐文件使启动记录失去可核对来源；补齐内容未改变任何本次运行的策略/仿真源码。之后的新run直接保存manifest快照。

仿真环境复制命令（首次执行，拒绝覆盖；本次已在后台执行，不重复启动）：

```bash
test ! -e .venv-robotwin && \
  cp -a --reflink=auto /remote-home/jinminghao/miniconda3/envs/dit4dit-robotwin2 .venv-robotwin
```

完成后需核对本地`sys.prefix`，并将**副本**`lib/python3.10/site-packages/__editable__.nvidia_curobo-0.7.7.post1.dev9.pth`的旧路径替换为`$PWD/.external/RoboTwin/envs/curobo/src`；原环境文件不改。脚本内其他旧shebang不作为启动入口，显式使用新环境`bin/python`与PATH；cuRobo还由runner的本地PYTHONPATH优先指定。

入口新增`--cuda-arch`，缺省由PyTorch探测当前可见GPU；本机L20可明确`--cuda-arch 8.9`以复用已编译缓存。A100不应照抄8.9。新启动模型服务的Inductor缓存也隔离在OpenWAM `.cache/robotwin/inductor`。

### 首次真实模型请求与ffmpeg修复

cuRobo五个本机扩展已编出并成功导入MotionGen，专家已筛出可解seed并进入策略阶段。服务完成真实10步联合去噪，初次compile约113秒，无OOM；随后客户端收到动作并进入原生`take_action`。但Conda自带ffmpeg依赖GLIBC2.32/2.33/2.34而当前系统2.31，encoder提前退出，`take_action`写视频触发BrokenPipe。首个run明确`incomplete`、**0个完整episode**，不把它当策略失败或闭环通过。日志保留，不覆盖。

系统`/usr/bin/ffmpeg`4.2.7启动与libx264编码器枚举通过。runner新增`select_ffmpeg`和run内`bin/ffmpeg` PATH shim，优先用经验证的系统二进制，允许显式`--ffmpeg`；不替换旧Conda文件，不关闭RT/相机/视频或改变动作预算。新14项CPU评测测试通过。新run为`smoke_local_runtime_ffmpeg_system/`，复用已核对且已warmup的同一模型服务、私有runtime；暂时只读依赖的说明同前，环境完整副本继续复制。

seed0指原生仿真的seed group；launcher不对模型服务逐episode重置噪声RNG，保持上游server默认行为。复用服务也保留已推进的模型RNG状态，因此本记录不声称跨中断逐动作比特复现。

## 已完成的闭环与独立环境验收

`smoke_local_runtime_ffmpeg_system/run.json` 已为complete：adjust_bottle Clean **1/1**、Randomized **1/1**，均为真实原生专家筛seed后的闭环；这只是链路检查，不是任务/50任务统计分数。Clean接受seed100001，123/400动作时成功，原生结果1.0与实际分母一致、exit0。录像ffprobe通过：H264、320×240、124帧/12.4秒；首尾帧人工查看确认瓶子从横卧到被右手竖直拿起，不是冻结视频。两域的原生_result.txt、result.json、完整日志和录像均保留在run内。

`.venv-robotwin` 复制已exit0，sys.prefix与torch.__file__均在OpenWAM目录；torch2.7.1+cu128、SAPIEN3.0.3、warp1.12.1、mplib0.2.1、websockets15.0.1。仅修复副本editable cuRobo路径，原pth内容逐字保持。五个本机已编译扩展原子安装到私有cuRobo包，旧不兼容.so先逐项hash备份到`.external/RoboTwin/.openwam_pre_migration_binaries/`；构建/二进制SHA256及cpp来源记录在`OPENWAM_EXTENSION_INSTALL.json`。新环境实际导入MotionGen、五个本地.so及CUDA张量成功（`independent_sim_import.log`），没有回到DiT4DiT源码。此批.so目标为L20/sm89；迁移A100等GPU需重新编译，不能复用冒充通用二进制。

## 2026-10-16 用户要求多GPU并行：正式队列

用户在双域smoke通过后要求尽快启动、尽可能多GPU并行。已启动`outputs/robotwin_eval/parallel_official100_20261016/`：100个唯一(task, split)作业，每项100个有效策略episode，共10,000条标准范围，无小样本替代。

原PolicyServer的动作缓存是server对象级，不能把多个仿真并发连接同一策略对象。新增`serve_parallel.py`使用**同一模型/engine、不同端口上的独立PolicyServer/WAMPolicy/动作deque**，对完整predict加锁，GPU模型调用串行；SyncExecutor reset仅清本会话deque，JointInferenceEngine每次generate重置DiT cache，prompt缓存按文本key共享。保留原generate/10步/同步/DiT cache/compile，不做批处理、不改模型或训练。只支持sync，拒绝async。这样多GPU的物理仿真/规划并行，同时避免每卡重复加载大权重挤占其他作业显存。

新增`parallel_eval.py`动态分配100个原生整task/split作业，复用单项`standard_eval.py`及严格结果验收；失败保留记录且不自动重试/换seed，其余worker继续领任务，最终有缺口就incomplete。`parallel.json`记录100项coverage、worker/实际PID/端口/GPU和状态，不用进程启动代替完成。每个端口只有一个本队列worker。

- GPU7现有服务18848继续供1个worker，仿真GPU1；无需重新加载即可先跑正式100条。
- GPU0加载共享模型，16个端口19000–19015。16个仿真slot的GPU为`0,1,1,1,2,2,3,3,4,4,5,5,6,6,7,7`，加前述worker共17个。
- 不终止其他GPU作业；已现场核对L20各卡占用，GPU2–6存在高计算负载，因此只分配轻量仿真/规划，不在这些卡新增大模型副本。单模型共享推理锁可限制显存峰值，实际吞吐/各slot容量仍须现场确认。
- 首次完整队列启动时，1个worker已运行独立环境正式100条，16个等待GPU0模型加载。该描述是启动快照，不是17条实际rollout已完成。

新入口与单项PID/端口检查能识别共享pool的manifest与PONG slot/token，验证端口真实归属于模型进程；配置与checkpoint不匹配拒绝。新的CPU测试专门覆盖真实WAMPolicy/SyncExecutor的跨会话reset隔离和共享推理互斥，真实17路闭环/完整100条与全量分数仍在验收中。

本轮模型和队列命令保存在日志/运行记录中。停止时须现场核对这些PID/命令，不照抄历史PID，不使用全局pkill。原DiT4DiT git diff hash再次复核与开始时一致；旧环境未安装/替换依赖。
