# ManiSkill 六任务：OpenWAM Relational Sidecar 训练执行说明

日期：2026-10-19 UTC。本文实现的是独立的训练适配；不替换正在训练的 baseline，不自动停止进程，不自动启动长期训练。当前验收结果见本文末尾。

## 1. 实验与实现

目标：在与当前 ManiSkill baseline 相同的六任务、6000 轨迹、542009 个动作起点上，比较加入 Relational Sidecar 是否改善策略效果。数据来自用户指定入口：

`/remote-home/jinminghao/datasets/maniskill_eort_six_task_ea_rsig_v3_20260922`

实际 RGB 来自 baseline ready manifest 精确指定的原 HDF5：push_cube/pick_cube 在 `maniskill_eort_color_distractors_v4_20260728`，其他四任务在 `maniskill_eort_task_suite_v5_20260730`。使用 manifest 中 aggregate 补采轨迹，不能仅靠轨迹编号猜 raw 路径。

模型复用 `dual_system_relational_sidecar`，结构、参数键和前向数学不变：

1. 保留原 OpenWAM video/action 双流、联合注意力、掩码、RoPE 与 flow-matching loss。
2. block **7/15/23 后**各进行一次 Sidecar read/refine/write；15×512 持续关系 tokens 跨三个阶段保留。
3. 每次只读当前首帧的 video hidden states，以当前 text、归一化 proprio80D 及有效维 mask 初始化关系 tokens；零初始化 write gate 将关系残差写入 action hidden states。
4. 标签只传入 `compute_loss`，不传入模型 `forward/generate`。GT 物体位置、bbox、接触及未来轨迹不是部署输入。
5. 单臂占用 EEF80D 的前10维；track0有效，track1及缺失相机对应监督无效。仍保留双臂模型结构以复用 Foundation，未缩小/重置 action head。

独立入口复用 baseline 的训练器和 per-rank metrics。对导出的 `relational_sidecar_contract.json` 仅修正描述元数据为 Panda/base/TCP，避免原 Sidecar 保存方法写入 RoboTwin frame；不改架构或权重。

## 2. 控制变量与配置

| 项目 | 当前 ManiSkill baseline | 本 Sidecar 默认 |
| --- | --- | --- |
| 样本、文本、原 RGB、绝对动作及统计 | 当前已验收 manifest | 精确复用，逐轨迹核对 |
| 初始化 | Foundation step154000 | 同一 Foundation，新增 Sidecar 随机初始化 |
| 训练阶段 | 原版全参 | joint 全参 |
| epochs / global batch | 10 / 256 | 10 / 256 |
| per-GPU batch / accumulation | 64 / 1 | **32 / 2**，为 Sidecar 留显存余量 |
| LR / AdamW betas / weight decay / clip | 1e-4 / .9,.95 / .01 / 1 | 相同 |
| schedule | cosine，warmup5%，最低比率.01 | 相同 |
| BF16 / ZeRO2 / gradient checkpointing / fused AdamW | 开启 | 相同 |
| Video / action FM权重、shift | 1 / 1；5 / 5 | 相同 |
| 视频与动作窗口 | 33源帧，stride4，9图，32动作 | 相同 |
| 分辨率/布局 | 384×320，front上/hand左下/右下黑 | 相同 |
| ColorJitter | **关闭** | **关闭**，保持本次baseline对照 |
| checkpoint | 每2000 micro = 2000 updates | **每4235 micro（一轮末，约2118 updates）** |
| 保留 | 最近两组及最终、完整恢复状态 | 相同 |
| EA | 无 | Ground .025 / Relation .01 / Plan .05；deep supervision .1 |

辅助损失沿用现有 Sidecar 设置，没有自动乘之前讨论过的0.5。最终 EA 为 `.025*ground + .01*relation + .05*plan + .1*(.025*stage1_ground + .01*stage2_relation)`。总loss比baseline多这些项，不能用两者总loss直接判断优劣。

bs32/acc2是保守启动档，**尚未在本 ManiSkill Sidecar 四卡实测**。当前 baseline 的 bs64 已接近显存上限，不能据它推断新增分支后仍能用64。不同micro batch会影响随机数消费及末批重复/补齐，不能声称逐样本顺序和数值完全一致；有效global batch相同。训练器进度条和checkpoint文件名计microsteps，`metrics_rank*.jsonl`同时记录optimizer_step。预计每轮4235micro、2118updates，共42350micro/21180updates，以实际DataLoader长度和epoch末累积刷新为准。

**保存边界说明：** 当前样本量下每轮4235micro为奇数，acc2在轮末刷新一次不足两micro的累积。原trainer按全局micro数保存；若设4000，部分保存点会处于新一轮的未同步累积中，不能保证该点完整恢复。因此本配置改为每4235micro（一轮末）保存，约2118updates一次，接近2000且保证默认配置的保存点位于轮末。若数据量或GPU数变化，重新核对轮长；不要盲目复用4235。独立入口还会按实际分片后DataLoader长度检查周期保存点与最终保存点，发现落在累积中间就报错，防止换数据后静默沿用错误间隔。轮末可能只有一个micro的样本，global256是常规完整更新的名义batch。此次不修改公共trainer。

## 3. 数据与新标签位置

已存在的 baseline 数值/统计入口：

```text
/remote-home/jinminghao/OpenWAM/outputs/maniskill_baseline/data_absolute_eef_20261017/manifest.ready.json
```

本轮独立准备目录：

```text
/remote-home/jinminghao/OpenWAM/outputs/maniskill_sidecar/data_absolute_eef_20261019/
├── manifest.ready.json
└── labels/<task>/episodeXXXX.npz   # targets: [T,130]
```

新 manifest 直接引用 baseline numeric、stats 和原始 HDF5，不复制 RGB，不修改源数据或 baseline，不重新拟合统计。整理或迁移文件时，这些被引用的目录也必须保留。旧 LeRobot 的125D向量和1/4/8 future labels不使用。

130D按现有Sidecar合同重建：

- 当前 TCP/物体/goal 来自原 HDF5 `obs/extra`，wxyz，世界坐标；与已有 derived 文件及baseline state逐帧核对。动作仍为已验收的**机器人base下绝对controller command target**，不是next-achieved TCP，也不是增量动作；没有RoboTwin的0.12m偏移。
- relation = `R_tcp(t)^T * (p_obj(t)-p_tcp(t))`；near阈值0.10m。
- contact来自`robot_obj_contact_force_norm > 1e-6 N`；缺失时mask为无效，不能标成已知负例。
- goal proximity为`clip(1-distance(object,goal)/0.5,0,1)`。虚拟/被遮挡goal可以有几何监督，但没有伪造的可见bbox。
- plan从真实**观测轨迹**计算8/16/32步TCP与object在各自当前局部旋转下的方向；位移>1mm标moving。使用原T+1 terminal observation，`t+h=T`有效，越界无效，不复制末帧伪造未来。
- bbox复用已配对的derived segmentation xyxy（右下exclusive），除以源图256；只用front/hand，对应模型L画布的上部/左下。right_shoulder虽存在于源文件，但未输入baseline，因此不用它的标签。track1/第三视角恒无效。

准备器全量核对源manifest hash、原HDF5 size/mtime、baseline numeric hash、state映射、derived姿态/接触及label有限性；记录源derived hash。验收脚本另从原分割图重新算bbox，并独立核对局部plan方向。完整ready manifest仅在准备成功后写出；失败残留目录不视为可训练数据，重新准备须用新输出路径。

## 4. 准备和验收指令

本机已准备成功后直接复用ready文件，不必每次训练重新生成。下面是重建指令（目标目录必须不存在；重建时换新的输出目录，并设置MANISKILL_SIDECAR_MANIFEST）：

```bash
cd /remote-home/jinminghao/OpenWAM
.venv/bin/python scripts/prepare_maniskill_sidecar_data.py \
  --baseline-manifest outputs/maniskill_baseline/data_absolute_eef_20261017/manifest.ready.json \
  --output-dir outputs/maniskill_sidecar/data_absolute_eef_20261019
```

离线验收（不占GPU；全量hash + 六任务54实际窗口 + 32样本拼batch）：

```bash
TORCH_WARM_POOL=0 CUDA_VISIBLE_DEVICES='' \
.venv/bin/python scripts/check_maniskill_sidecar_data.py \
  --manifest outputs/maniskill_sidecar/data_absolute_eef_20261019/manifest.ready.json \
  --output outputs/maniskill_sidecar/data_checks_20261019.json
```

只解析配置，不加载权重、不开始训练：

```bash
CHECK_ONLY=1 bash scripts/train_maniskill_sidecar_4gpu.sh
```

训练启动不全量扫描数据或重新构造标签。reader读取ready manifest和stats；原HDF5首次打开时检查size/mtime；numeric及labels按episode读取、hash校验并在每个worker内做64条LRU缓存。LRU淘汰或worker重建后会再读/校验，不是把全部RGB或全部训练数据一次永久缓存到内存。RGB仍按batch从源HDF5加载。离线全量验收不在每步重复。

## 5. 四卡短测：先验证显存和更新

**等GPU0–3真正空闲后执行**。脚本要求选中的四张A100-80GB各free>=70000MiB；不杀作业、不自动等待或抢卡。当前baseline正在这些卡上，不能并跑此正式配置。

```bash
cd /remote-home/jinminghao/OpenWAM
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MANISKILL_SIDECAR_OUTPUT_DIR="$PWD/outputs/maniskill_sidecar/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
CHECK_ONLY=0 bash scripts/train_maniskill_sidecar_4gpu.sh \
  training.num_epochs=null training.max_steps=40 training.save_steps=40
```

这是bs32/acc2的40micro（20次optimizer更新），最后保存权重和完整状态。看四rank metrics的finite loss/grad norm、三个write gate从0更新、最后数步显存/吞吐及exit状态。短测cosine horizon只有20updates，因此**不拿这个checkpoint继续正式10epoch**，正式实验重新从Foundation开始。

## 6. 正式10epoch：具体启动指令

先开持久终端：

```bash
tmux new -s maniskill_sidecar
```

再在其中执行：

```bash
cd /remote-home/jinminghao/OpenWAM
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MANISKILL_SIDECAR_MANIFEST="$PWD/outputs/maniskill_sidecar/data_absolute_eef_20261019/manifest.ready.json"
export MANISKILL_SIDECAR_OUTPUT_DIR="$PWD/outputs/maniskill_sidecar/six_task_joint_10epochs_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$MANISKILL_SIDECAR_OUTPUT_DIR"
set -o pipefail
CHECK_ONLY=0 bash scripts/train_maniskill_sidecar_4gpu.sh \
  2>&1 | tee "$MANISKILL_SIDECAR_OUTPUT_DIR/train.log"
```

按Ctrl-b再按d脱离tmux；`tmux attach -t maniskill_sidecar`重新连接。实际产物在`$MANISKILL_SIDECAR_OUTPUT_DIR/runs/<timestamp>/`：config、stats、tokenizer、修正后的sidecar合同、per-rank metrics、`checkpoint_step_4235/8470/...safetensors`和对应`accel_state_step_*`。文件名计microsteps。默认每轮末保存约2118次optimizer更新；4235/8470分别为第1/2轮结束。只保留最近两组，最终保留完整恢复状态。

配置解析中出现的`/path/to/Wan2.2-TI2V-5B`是继承的默认占位值；正式入口设置了`finetune_ckpt_path`，原训练器从下面自包含Foundation目录重建组件并加载权重，不按占位路径下载模型。

默认Foundation目录：

```text
/remote-home/jinminghao/OpenWAM/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model
```

若不同端口冲突可设置`MASTER_PORT=29887`。如果将来实测bs64能稳定使用，可覆盖`training.batch_size=64 training.gradient_accumulation_steps=1 training.save_steps=2000`；未经短测不直接推断可行。若需要0.5 EA消融，另建实验并覆盖三个权重为`.0125/.005/.025`，不要混入默认对照。

中断后恢复（必须是此Sidecar run的实际`runs/<timestamp>`，且包含完整状态；保持原batch/acc/epochs）：

```bash
export SIDECAR_RESUME_DIR=/absolute/path/to/sidecar/run/runs/TIMESTAMP
CHECK_ONLY=0 bash scripts/train_maniskill_sidecar_4gpu.sh \
  training.finetune_ckpt_path=null \
  "training.resume_ckpt_path=$SIDECAR_RESUME_DIR"
```

模型/优化器/scheduler/RNG恢复由原训练器处理。独立入口另外从`trainer_state.json`保留真实`opt_step`：奇数轮长在epoch末有一次部分累积更新，不能简单用`global_micro//acc`推算更新次数。若原恢复逻辑需要把microstep向前回退，入口拒绝该checkpoint，避免继续使用未同步保存点。新适配尚未实测完整Foundation四卡更新或恢复，不能据CPU测试称这两项已经通过。

## 7. 新增文件与验收范围

- `openwam/maniskill_sidecar_contract.py`：130D单臂监督构造，复用现有Sidecar目标定义。
- `openwam/dataloader/maniskill_sidecar.py`：继承baseline reader，仅增加loss标签。
- `scripts/prepare_maniskill_sidecar_data.py`、`scripts/check_maniskill_sidecar_data.py`：离线准备、全量hash与真实窗口验收。
- `configs/dataloader/maniskill_sidecar.yaml`、`configs/train_maniskill_sidecar_10epochs.yaml`：独立配置。
- `scripts/train_maniskill_sidecar.py`、`scripts/train_maniskill_sidecar_4gpu.sh`：标准trainer入口、空闲卡门槛、坐标描述导出。
- `tests/test_maniskill_sidecar.py`：局部坐标/时间尾端/mask/未知接触/真实tiny单臂loss梯度及oracle隔离。

物理专家replay仍受此前容器Vulkan依赖阻塞，单臂部署控制器和闭环评估尚未落实。本轮交付训练适配，不代表已得到策略提升；最终应以相同任务/seed/执行预算的baseline与Sidecar闭环成功率对照为准。

### 本轮现场验收

验收时间：2026-10-19T09:50:30.358939+00:00。已完成以下现场检查：

- 离线全量准备成功：6任务、6000轨迹、542009动作起点。新目录约138MiB，主要是独立130D标签和manifest；没有复制/转码RGB、没有重新生成baseline动作或统计。
- 第二遍全量reader核对：6000标签SHA256、derived来源SHA256、基线轨迹身份全部通过。
- 54真实窗口（6任务×首/中/末轨迹×首/中/末起点）通过：与baseline逐项比较action/proprio/mask/视频像素/文本；原始分割重算front/hand bbox与标签相等；从原TCP/object世界位姿独立重算8/16/32局部方向与moving/future-valid一致。
- 实际32样本list batch：action[32,32,80]、EA targets[32,1,130]通过。
- `tests/test_maniskill_sidecar.py`最终5项通过（188.18s，包括共享盘依赖导入）；包括单臂真实tiny Sidecar两次loss/backward/AdamW更新、三层read/write梯度、GT输入拒绝、边界mask/接触缺失、保存边界拒绝、恢复真实opt_step及Panda导出frame。
- Hydra配置对比通过：training差异仅batch64→32、acc1→2、保存间隔2000→4235、output路径；其余training字段一致。dataloader除type/manifest外一致，ColorJitter仍关闭。Python编译、bash语法和新增文件whitespace检查通过。

证据：`outputs/maniskill_sidecar/data_checks_20261019.json`；`outputs/maniskill_sidecar/verification_20261019/`下的`prepare.log`、`cpu_tests.log`、`resolved_config.yaml`、`config_comparison.json`、`acceptance.json`（含本轮代码/manifest SHA256）。验收脚本记录的435.57秒不包含前面的依赖导入，属于离线验证耗时，不能当训练dataloader速度。

**尚未进行新Sidecar的真实Foundation四卡更新、显存/吞吐或GPU完整保存恢复；没有启动本轮短测或正式训练，也没有闭环成功率结果。** 当前baseline代码/配置/数据及进程未改。下一步由用户在四卡空闲后执行第5节短测、确认更新与保存，再按第6节从Foundation正式训练。数据本身已准备且上述CPU/真实reader验收通过，无需重复全量构造。


## 2026-10-19T14:10:21.890139+00:00：用户授权RoboTwin结束后自动接续ManiSkill

监督进程PID2193441，脚本scripts/supervise_maniskill_after_robotwin.py，状态/日志/spec位于`outputs/maniskill_sidecar/handover_after_robotwin_20261019T140623Z`。现场状态waiting_for_robotwin_completion，PPID1，独立会话，不依赖当前对话存活。当前RoboTwin仍正常训练，未终止/修改它。

放行条件：固定本机boot与原torchrun/四rank PID start_ticks；原进程全部退出后，四rank最终microstep均42960、最终Saved final日志和对应完整safetensors header/size必须满足；然后等0–3各A100-80GB且free>=70000MiB，确认MASTER_PORT29886可用后只启动一次。若前序异常未到终点，则记录predecessor_incomplete_no_launch并退出；不误把崩溃/临时空卡当完成，不kill任何其他作业。通过持有flock和launched.json防同一接续重复启动。

接续执行用户原命令：CUDA_VISIBLE_DEVICES=0,1,2,3、CHECK_ONLY=0、bash scripts/train_maniskill_sidecar_4gpu.sh；MANISKILL_SIDECAR_OUTPUT_DIR在实际接续时生成outputs/maniskill_sidecar/joint_<UTC>，train.log写入该目录。保持现有joint/10epoch/bs32/acc2/global256/每4235micro保存/ColorJitter关闭，从Foundation新训练，绝非接RoboTwin权重继续训练。默认manifest为已验收6000轨迹/542009窗口的独立ManiSkill Sidecar ready；原baseline数值和原HDF5依赖保持。

已验证并复用现有`/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv`（总runtime约7.5GB），没有新增环境副本；Torch2.7.1+cu128/Accelerate1.15.0/DeepSpeed0.18.9/h5py3.16.0/torchvision0.22.1+cu128均从本地环境实际导入，核心依赖导入6.68秒，CHECK_ONLY配置解析exit0，代码/manifest与现有验收hash一致。若该本地副本消失，回退共享.venv；不安装包、不改原环境。模型权重/数据/输出仍在共享盘，不宣称6.68秒等于正式训练启动时间。

监督逻辑7项CPU测试通过，包含合成子进程完整接续、等待显存、防重复、前序失败不启动、四rank与权重完整性和PID复用检查；未实际提前启动GPU训练。该ManiSkill Sidecar的真实Foundation四卡显存/更新及完整恢复仍待接续后验收，既有未完成风险保留。新训练失败时记录maniskill_failed/returncode及train.log，不自动反复启动。
