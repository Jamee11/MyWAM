# 多仿真、多本体 Sidecar mid-training

日期：2026-10-24 UTC。目标是在已有 OpenWAM Foundation 上，用自采仿真数据继续训练当前三层 Relational Sidecar，再进入目标任务 SFT。它不复现 OpenWAM-alpha 原始大规模 human/robot 预训练，也不自动证明跨本体或真机收益。

## 当前状态

- 原项目代码已完成本地完整归档并备份到 GitHub，远端与本地代码树完全一致。
- 五类输入的正式验收清单已扫描，冻结 **52,204 个已验收外观轨迹版本**。
- 10 条真实来源轨迹的独立 smoke 数据已构建，覆盖 RLBench、原 ManiSkill、hard ManiSkill、RoboTwin Aloha/Piper 与新 clutter；读取和全部 numeric hash/media stat 检查通过。
- 最新指定回归集28项CPU检查通过，含四源单双臂混合 Action/Video/EA loss、完整反向、WAM 暂不更新→联合更新切换、epoch采样覆盖、训练统计排除held-out数据及恢复时world-size/预算身份检查。Hydra 配置解析及 shell 语法通过。
- **全量数值/辅助标签构建正在进行，正式目录尚未发布 `manifest.ready.json`；这不等于全量数据已准备好。**
- 本轮没有启动正式 mid-training，没有停止原 GPU0–3 训练或评估/真机服务；尚无全量 Foundation 混合训练速度、容量或策略成绩。

## 1. 改动前备份

- GitHub：[backup/pre-multisim-midtrain-20261024T162442Z](https://github.com/Jamee11/MyWAM/tree/backup/pre-multisim-midtrain-20261024T162442Z)。
- 远端 commit：`0bf6aef0212184703803ff7e3aacaa919d6aaf2e`。
- 本地快照 commit：`3737261be932cdf23b9d46035576238fc588aa31`。
- 相同 tree：`702163ce79c323a1f232679cf0d471d271dac2b3`。
- 本地归档：`outputs/backups/pre_midtrain_20261024T162442Z/OpenWAM_pre_midtrain_code.tar.gz`，9,667,625 bytes，SHA256 `7dd6f1eb9d4b8623a94b129fa394d6c306996ed263de33403a693bffa03dca7a`。
- 清单/上传证据：同目录 `backup.json`、`source_manifest.json`。

命令行 Git 的旧 VS Code credential socket 已失效，实际使用已连接 GitHub 应用完成对象上传、代码树一致性检查、独立分支创建及回读。没有覆盖 main/旧feature分支，工作区分支与原 index 保持。

## 2. 输入与冻结配额

| 来源 | 本次选入 | 依据 |
| --- | ---: | --- |
| RLBench18 Franka 指定 production | **8,204** | 4,093个已验收物理家族的实际正式产物；目录名20k是目标，不是当前完成数 |
| ManiSkill realwhite 原20k | **10,000** | Clean全部5,000；B/C/O/D/M各1,000，按任务与原split确定性选择 |
| ManiSkill hard-random camera-v2 | **10,000** | 全部正式 `ACCEPTED_RAW.json` 与对应 raw audit |
| RoboTwin wrist020 Aloha/Piper | **20,000** | 20k总验收marker与原始HDF5精确库存 |
| RoboTwin fresh clutter randomized | **4,000** | 每个quota的passed state指向实际成功候选，不扫描失败重试当有效样本 |
| **总计** | **52,204** | 外观轨迹数，不等于独立物理家族数 |

原始路径保持用户指定的五个目录；默认值集中在 `scripts/prepare_multisim_sidecar_data.py`。完整选择记录在新数据根目录 `selection.json`，冻结后不会因后续采集新增文件而隐式改变训练集。

ManiSkill按原train/val/test选取，总计仍为16,000/2,000/2,000；Clean及其B/C/O、D及其M的同源关系保留。RLBench保留原split，当前已采集份额不一定是80/10/10。RoboTwin按本体+任务+完整物理EEF轨迹签名分组，确定性分到train/val/test，配对外观版本不跨split；新clutter若物理轨迹相同也合并同源。

**正式训练只读取train部分，52,204不是train episode数。** 最终各split、domain、group及物理家族数以完成后的manifest为准。不将验证/测试外观版本偷偷并入预训练统计。

RLBench20k若后续采齐，应在新版本目录重新冻结和构建；不能原地扩展已经开始训练的release并称resume。

## 3. 统一格式与动作语义

新数据根：

`/remote-home/jinminghao/datasets/openwam_sidecar_multisim_midtrain_v1_20261024`

```text
selection.json                 # 冻结的已验收来源及配额
media/<source-digest>.h5        # 只读原始RGB载体的硬链接；不重新编码
numeric/<group>/<id>.npz        # 统一state/action/targets/arm_present
numeric/<group>/<id>.json       # 单条完成证据、来源签名、构造代码指纹
progress.json                  # 完成数量，非发布标记
errors/<id>.json                # 如失败，保留来源和traceback，不能静默略过
normalization_stats.npy        # 全量完成后，仅train的有效臂拟合
manifest.ready.json            # 全部选择项成功后才原子发布
```

统一NPZ：`state[T+1,20]`、`action[T,20]`、`targets[T,130]`、`arm_present[2]`。训练reader再映射到Foundation80D：左臂0–9、右臂34–43；不存在的臂与其他槽位为0且mask无效。

每个训练窗口输出32个动作、9帧384×320 L画布，视频锚点0/4/…/32，辅助未来锚点8/16/32。尾部重复最后可用帧/动作，但真实padding通过action/video mask排除；不靠padding制造未来监督。

| 来源 | state | action | 特殊处理 |
| --- | --- | --- | --- |
| RLBench | 实测base-frame EEF10 | native joint target经已审计FK得到的absolute base EEF10 | 复用已独立生成的OpenWAM32/130D缓存，验证其SHA、oracle源码SHA及与raw的数值一致性；不复用125D旧锚点 |
| ManiSkill | 实测base-frame Panda TCP10 | 从实际`pd_ee_delta_pose`命令与当前TCP重建absolute目标 | 核对TARGET_CONTROLLER及backup真实命令/20Hz replay时间轴；不把next-achieved当command，也不混用原joint-source时间轴 |
| RoboTwin | 官方endpose reference的实测EEF20 | t+1到t+32的next-achieved absolute EEF20 | 保留既有Aloha训练语义；从wxyz转矩阵前两列，EA TCP的局部x0.12m偏置不施加到Franka |

各来源均为绝对EEF，但reference/controller语义仍有差别，manifest逐条记录。**统一槽位不代表已把所有机器人变成相同动力学。** RLBench/ManiSkill的20Hz经raw或backup检查；RoboTwin缺可直接验证的原生时间戳时不凭视频fps猜测控制Hz。当前统一的是32个原生控制transition，不伪称跨来源具有完全相同物理时长。

RGB：RLBench/ManiSkill按native uint8 RGB读取；RoboTwin legacy JPEG沿用原reader无BGR交换的解码规则，遇XPL-RGB1则明确走标准RGB-JPEG规则。三视图与两视图统一L布局，两视图右下为空，不伪造第三相机或其GT。

## 4. 标签、统计与防泄漏

- 重用当前 `targets_from_oracle` 的关系、接触、局部未来方向定义；不会将旧2/4/8或10/20/40标签插值成8/16/32。
- ManiSkill bbox从同一时刻真实实例分割生成；LiftPeg无空间goal时关闭相应goal监督。虚拟/不可见goal、未知contact及不存在的第二臂通过valid mask区分，不填零冒充真实负例。
- Ground/Relation/Plan只用于loss，不作为策略推理输入；前向仍为RGB、文本、当前proprio与维度mask。
- state/action保留矩阵前两列Rot6D，归一化固定为identity；gripper约定0闭/1开，min/max固定0/1。
- 全局EEF20统计仅来自train中存在的臂及其state/command，不让单臂填充的右臂零值污染双臂统计。各本体共享一个明确的规范化空间，随checkpoint一起保存。
- 统计拟合、数字标签生成与源库存扫描全部离线。训练启动只读取manifest和准备结果；numeric SHA在worker首次读取相应缓存时检查，media在打开时验证size/mtime，无每步重新生成数据或全盘扫描。
- media硬链接不增加原payload副本，但**禁止原地写入或chmod**，否则可能影响原数据。搬迁时完整复制数据根目录即可，训练reader不访问原树；来源绝对路径只留作provenance。跨文件系统复制会真正占用原始RGB载体的存储空间。

## 5. 训练逻辑与默认预算

保留原网络、三层Read/Refine/Write、Action+Video FM及EA辅助loss。前2000次optimizer更新正常完整前后向/写入，仅跳过WAM参数、Adam状态和weight decay更新；之后joint。没有加入heads-only、Task Gate、更多Transformer或新的辅助目标。

独立配置：`configs/train_multisim_sidecar_midtrain.yaml`。

- 默认Foundation新起点；WAM LR2e-5、Sidecar1e-5，低于当前任务SFT recipe，作为保守的多源继续训练起点，尚无性能最优性证明。
- EA总scale=.5，原Ground .025/Relation .01/Plan .05/Deep .1，Action/Video权重均1；ColorJitter开启。
- 默认4卡×bs32×acc2，global batch256，BF16/ZeRO2/gradient checkpointing，CPU模型初始化，无optimizer offload。
- 默认100,000 microsteps＝50,000optimizer更新，WAM实际48,000次。每4000微步保存完整状态，保留最近3份。
- 相同全局linear+cosine LR schedule，5% LR warmup；不新增独立阶段。该配方与旧单源SFT不构成warmup单变量消融。
- 四组RLBench Franka、ManiSkill Franka、RoboTwin Aloha、RoboTwin Piper默认等权。每epoch重新抽各source的覆盖索引，避免只反复采长source前缀；epoch补齐到global batch整数倍，避免accumulation中间保存。

native mixture接口继续使用，只有新midtrain子类改进其source覆盖与epoch对齐，旧MixtureDataset和单源流程不改。resume校验数据manifest SHA、统计、混合权重、seed及epoch对齐，读取完整trainer计数；换数据应新建finetune阶段，不能伪作同一run恢复。

## 6. 构建、检查、启动

以下命令在项目根执行。`PYTHON_BIN`必须是已有依赖齐全的OpenWAM环境；可选现有本地runtime加速，但其`/dev/shm`路径重启后可能消失。正式midtrain没有自动启动。

```bash
cd /remote-home/jinminghao/OpenWAM
export PYTHON_BIN="$PWD/.venv/bin/python"
export MIDTRAIN_MANIFEST=/remote-home/jinminghao/datasets/openwam_sidecar_multisim_midtrain_v1_20261024/manifest.ready.json

# 离线构建，已有selection和同revision单条完成记录可复用；已发布release拒绝覆盖。
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/prepare_multisim_sidecar_data.py \
  --output-dir /remote-home/jinminghao/datasets/openwam_sidecar_multisim_midtrain_v1_20261024 \
  --workers 4

# 发布后，一次性全量artifact检查；平时启动用默认快速检查即可。
"$PYTHON_BIN" scripts/check_multisim_sidecar_data.py --manifest "$MIDTRAIN_MANIFEST" --full
CHECK_ONLY=1 bash scripts/train_multisim_sidecar_midtrain.sh

# 只在选定GPU空闲、检查通过后执行；不要与当前0–3训练抢同一分配。
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MIDTRAIN_OUTPUT_DIR="$PWD/outputs/midtrain/multisim_$(date -u +%Y%m%dT%H%M%SZ)"
CHECK_ONLY=0 bash scripts/train_multisim_sidecar_midtrain.sh
```

若要用已有本地环境，显式改`PYTHON_BIN=/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv/bin/python`。所有HF访问离线，endpoint默认hf-mirror；没有任何下载或官方fallback。

建议先做独立的小预算真实训练（不冒充有效性实验）：

```bash
CHECK_ONLY=0 bash scripts/train_multisim_sidecar_midtrain.sh \
  training.max_steps=8 training.save_steps=4 training.warmup_ratio=0.5 \
  model.architecture.relational_sidecar.backbone_warmup_updates=2
```

上例仅缩短阶段以检查完整梯度更新、阶段切换、保存与重载；正式配置仍为2000次。小预算检查也使用新输出目录。恢复示例：

```bash
CHECK_ONLY=0 bash scripts/train_multisim_sidecar_midtrain.sh \
  training.finetune_ckpt_path=null \
  training.resume_ckpt_path=/absolute/path/to/the/actual/run
```

恢复必须沿用原manifest、weights、seed、batch/world-size对应的对齐及原计划预算；小测→正式长训不是无条件等价的resume。Foundation或完整Sidecar初始目录通过`BASE_MODEL`指定；旧已训练Sidecar作为新阶段起点应明确是否保留2K WAM延迟更新，不能偷偷复用旧optimizer计数。

如使用8张卡并希望保持global batch256与50,000次更新，新阶段可用`NPROC_PER_NODE=8 GRAD_ACCUM=1`，同时覆盖`training.max_steps=50000`。这是不同world-size的新run；现有ZeRO分片不支持在本入口中随意4→8卡直接resume，元数据会拒绝world/batch/accumulation/计划预算变化。跨world-size请另建finetune阶段，不能把它称为完整optimizer恢复。

## 7. 验证范围与后续实验

工程证据位于`outputs/midtrain_audit_20261024`：源库存、resolved配置、CPU tests、10条真实smoke构建与读取检查、全量日志。全量状态以数据根`progress.json`和最终`manifest.ready.json`为准，某个子进程退出或目录存在不表示完成。

数据构建初期默认fork worker停止产出后改为spawn独立进程，保留既有产物；添加逐源异常证据与源码指纹。后续单条构建与manifest必须同revision，不将旧缓存默认为新逻辑已验证。

尚需：全量生成/检查完成、完整Foundation多源短训和实际状态重载、各来源未见物理家族验证、正式训练吞吐/显存测量。CPU tiny模型、finite loss与数据格式通过都不能称midtrain有效或迁移成功。

科学比较需要相同预算的单源/多源、Vanilla/Sidecar对照，并记录每个来源的暴露量。**本阶段包含Randomized数据，因此后续不能把其成绩直接称为原始Clean-only→Random协议提升。** 已见任务/物体、跨仿真、跨本体和零真实示范应分别定义；目标真机SFT或部署必须使用相应checkpoint的统计、动作reference与显式臂mask，不默认20D双臂normalizer自动代表单臂硬件。

### 2026-10-24：Foundation工程短测进行中

在GPU5/6开始双卡bs1、acc1、8微步的独立短测，使用已验收10条真实来源smoke数据、原Foundation和完整Sidecar；WAM延迟更新缩短为2次，仅验证更新切换/保存/恢复。原0–3训练不操作。日志与精确启动参数见`outputs/midtrain_audit_20261024/foundation_smoke/launch.json`及`train.log`。启动不代表已通过；后续以实际指标、保存和重载结果验收。当前全量数据构建仍独立运行。

新增非转换文件的ruff检查通过；转换脚本保持构建源码冻结，导入排序清理延后至全量构建结束，避免在运行中改变preparation指纹。本地已检查的CPU回归精确命令为`pytest -q tests/test_multisim_sidecar.py tests/test_sidecar_backbone_warmup.py tests/test_optimizer_groups.py`，27 passed，日志`tests_lint_cleanup.log`。

### GitHub CI现状

代码树回读验证成功，但初次功能分支CI失败：tests为2258通过、5失败（旧HTTP测试缺Flask、旧远程评估fixture依赖本机outputs文件）；新增multisim测试未失败。全仓lint的370项在改动前备份代码中亦复现，另有冻结转换脚本5项导入排序待清理。因此代码备份成功不等同全仓CI通过。保留这些边界，不为本次数据框架任务掩盖或批量重写旧模块。

### 双卡容量与offload边界

第一次真实Foundation短测在前两次warmup更新后，于WAM首次optimizer更新出现ZeRO flatten显存不足（申请9.31GiB）；尚无保存/恢复成功结论。第二次仅启用`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，记录于`foundation_smoke_expandable`，不是缩小模型或关闭反向。

当前Accelerate的CPU offload会创建新CPUAdam并丢弃原AdamW warmup hook，新入口因此拒绝`backbone_warmup_updates>0`且`offload_optimizer_device!=none`的组合；默认四卡无offload方案不变。不能将CPU offload当作保留该warmup语义的直接替代。

全仓CPU检查在这次guard之前已通过2220项（18 skipped/21 deselected）；guard及相关混合数据/优化器测试另保留独立回归日志。GitHub旧缺依赖/fixture失败不等同本地全仓失败，二者环境边界分别记录。
