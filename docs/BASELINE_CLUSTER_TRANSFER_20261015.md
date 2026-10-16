# OpenWAM-alpha 官方 Clean50 baseline：集群迁移与 Slurm 提交

2026-10-15用户指定数据源：`/remote-home/jinminghao/DiT4DiT/datasets/robotwin2_official_raw_20261001`，只取其中Aloha Clean，不包含Randomized/Piper/原下载zip。此前自采数据方案已归档，不作为本交付训练源。

目标根目录：`/home/mingfeihangroup/xiaoming`。分区 `mhan0035`，单节点4张A100；不设置 `--time`，使用分区默认值。用户提供的节点 `gnode06` 状态为 `drng`，排空期间作业可能排队；本流程不修改集群节点状态。

## 包内容与训练合同

- `openwam_training_code.tar.gz`：当前训练代码、配置、启动/安装/迁移脚本、小型FK资产、许可和必要说明。没有 `.git`、`.venv`、编译缓存、旧日志、历史实验权重、评测视频或模拟器。
- `openwam_official_clean50_with_legacy_ea_labels.tar.gz`：官方Aloha Clean的50任务2500个真实HDF5轨迹、源指令/scene元数据，以及单独保留的旧自采数据EA标签/统计/manifest。官方HDF5逻辑大小35,044,845,222 bytes；未包含MP4预览/PKL回放。打包时跟随链接读取真实内容，归档成员全部为普通文件，保存PAX纳秒mtime。不要用复制软链接代替本包。
- `openwam_alpha_foundation.tar.gz`：官方Foundation原始权重、config、完整tokenizer与下载校验记录。主权重checkpoint_step_154000.safetensors，revision `52df4e66c82c5c8b480adcc8d01f4db7415dfb56`。
- `SHA256SUMS`：三个压缩包SHA256。相邻 `.inventory.json` 记录每个归档成员的大小/hash，EA标签和Foundation还与已有上游/审计hash核对。

Baseline使用官方 `dual_system / joint_self_attn`、80D action/state，直接加载Foundation，不添加EA模块、EA条件或EA损失。使用现有 `ea_rsig_lite` 数据reader的 `auxiliary=false` 分支，是为了保留本数据wxyz→xyzw转换、同一Clean归一化和窗口、prepared快速启动；它返回普通官方样本，训练不读EA标签。不将原始wxyz轨迹直接交给假定xyzw的reader。

默认bs64/卡×4卡、acc1/global256、5epoch，官方数据552287源帧、549787训练窗口，对应10740更新（每epoch2148更新，5个完整epoch；与论文Clean2Random预算一致）；学习率video/action/base均1e-4、AdamW beta0.9/0.95、weight decay0.01、Cosine/5% warmup/min ratio0.01、clip1、BF16/ZeRO2/gradient checkpointing、fused AdamW、T5/VAE冻结。保留三个等学习率参数组以避免历史ZeRO单大组临时显存峰值。ColorJitter开启0.2/0.2/0.2/0，33源帧/stride4/9视频帧、32actions、384×320、多视角，seed42。每2000更新保存权重和full state，保留最近2组。baseline模型无EA，磁盘保留EA数据供后续实验。

该bs64档针对A100 **80GB**，尚未在目标集群实测。若为40GB不要直接用此档，需另做offload/容量配置；不自动改batch。目标Slurm/CUDA/NCCL/包安装可用性和首次完整更新仍须现场确认。本机正在运行的EA作业按用户最新选择保持原自采数据运行，尚未切换到官方数据。

## 1. 在源服务器用 SFTP 上传

将下方 `SOURCE_BUNDLE_DIR` 替换为交付时给出的完整目录；目标用户名和主机按实际填写。

```bash
sftp <目标登录用户>@<目标集群登录地址>
```

在sftp提示符中执行（若incoming已存在，忽略mkdir的已存在提示）：

```text
lcd SOURCE_BUNDLE_DIR
mkdir /home/mingfeihangroup/xiaoming/incoming
cd /home/mingfeihangroup/xiaoming/incoming
put openwam_training_code.tar.gz
put openwam_official_clean50_with_legacy_ea_labels.tar.gz
put openwam_alpha_foundation.tar.gz
put SHA256SUMS
put README_TRANSFER.md
put TRANSFER_INFO.json
put openwam_training_code.tar.gz.inventory.json
put openwam_official_clean50_with_legacy_ea_labels.tar.gz.inventory.json
put openwam_alpha_foundation.tar.gz.inventory.json
bye
```

大文件中断后可以用 `reput 文件名` 继续上传。

## 2. 目标登录节点校验并解压

```bash
BASE=/home/mingfeihangroup/xiaoming
mkdir -p "$BASE/projects" "$BASE/ckpts"
cd "$BASE/incoming"
sha256sum -c SHA256SUMS
# --keep-old-files 防止覆盖已有同名项目/权重；如有冲突，先选新目录并调整OPENWAM_PROJECT。
tar --keep-old-files -xzf openwam_training_code.tar.gz -C "$BASE/projects"
tar --keep-old-files -xzf openwam_official_clean50_with_legacy_ea_labels.tar.gz -C "$BASE/projects"
tar --keep-old-files -xzf openwam_alpha_foundation.tar.gz -C "$BASE/ckpts"
```

解压后的路径：

```text
/home/mingfeihangroup/xiaoming/projects/OpenWAM/
  scripts/cluster/
  data/clean50/<task>/aloha-agilex_clean_50/aloha-agilex_clean_50/data/episode*.hdf5
  data/legacy_ea_lite/{labels/,manifest.ready.json,normalization_stats.npy,...}
  data/baseline/{manifest.source.json,normalization_stats.npy}
/home/mingfeihangroup/xiaoming/ckpts/OpenWAM-Alpha-Pretrain-Foundation-Model/
```

不要使用会丢失mtime纳秒的工具重新复制已解压HDF5；GNU tar按PAX时间恢复，迁移脚本会校验大小/mtime。三个压缩包SHA256验证负责传输完整性。

## 3. 一次性安装与路径准备

```bash
cd /home/mingfeihangroup/xiaoming/projects/OpenWAM
# 按集群实际环境先module load Python/CUDA；推荐Python3.10。
PYTHON_BIN=python3.10 bash scripts/cluster/install_environment.sh
.venv/bin/python scripts/cluster/prepare_baseline_paths.py
CHECK_ONLY=1 bash scripts/cluster/train_baseline.sh
```

环境安装复用本项目已验证的Torch2.7.1/cu128及锁定依赖，不打包旧venv；登录节点需允许依赖下载。Foundation已完整携带，训练强制HF离线，不下载模型。若必须在计算节点安装，先申请交互作业完成一次环境安装，不在每次正式提交重复安装。

路径准备只运行一次，核对2500个轨迹size/mtime和统计hash，生成baseline `data/baseline/manifest.ready.json`，旧EA标签单独保留在 `data/legacy_ea_lite`，不把其旧轨迹绑定改到官方HDF5；不打开全部HDF5、不构造标签、不改原始数据。baseline训练不再运行独立全量预检，reader保持首读源文件合同检查。`CHECK_ONLY=1` 只解析配置，不占GPU、不训练。

## 4. 提交 baseline

```bash
cd /home/mingfeihangroup/xiaoming/projects/OpenWAM
OPENWAM_CLUSTER_SETUP="$PWD/scripts/cluster/setup_hanhai.sh" \
  sbatch --nodelist=gnode06 --gres=gpu:a100:4 scripts/cluster/baseline_clean50.sbatch
```

需要account时：

```bash
OPENWAM_CLUSTER_SETUP="$PWD/scripts/cluster/setup_hanhai.sh" \
  sbatch --account=<你的account> --nodelist=gnode06 --gres=gpu:a100:4 scripts/cluster/baseline_clean50.sbatch
```

如站点要求显式A100 GRES类型，用 `sbatch --gres=gpu:a100:4 ...`（以本站 `sinfo -o '%P %G'` 为准）。脚本不硬编码物理GPU编号，保留Slurm分配的CUDA_VISIBLE_DEVICES。一个srun任务启动4个torchrun worker。`--mem=0` 申请整节点可分配内存，覆盖四进程CPU模型构建/权重加载峰值；如站点不允许此设置，可按节点容量用 `sbatch --mem=384G ...` 覆盖。没有人工time限制，但仍受分区/QOS实际限额约束。

查看进度：

```bash
squeue -u "$USER"
tail -f slurm-openwam-alpha-baseline-<JOBID>.out
```

默认训练产物：`projects/OpenWAM/outputs/baseline_clean50/<时间戳>/`。可选模块配置写入自有shell文件，再用 `OPENWAM_CLUSTER_SETUP=/绝对路径/setup.sh sbatch ...`。如需不同项目位置，设置 `OPENWAM_PROJECT`；权重路径可用 `OPENWAM_FOUNDATION` 覆盖，输出路径可用 `OPENWAM_OUTPUT` 覆盖。

## 5. 中断后恢复，以及后续EA

若作业中断并已有完整状态，传入已保存run目录；只从最近2000步周期保存点恢复，保留相同训练计划和代码：

```bash
sbatch scripts/cluster/baseline_clean50.sbatch \
  training.finetune_ckpt_path=null \
  training.resume_ckpt_path=/home/mingfeihangroup/xiaoming/projects/OpenWAM/outputs/baseline_clean50/<时间戳>
```

首次新run务必使用Foundation；不要把其他任务SFT权重当作Foundation。目标机尚未验证恢复，首次保存后应核查权重/full-state文件完整。

后续EA的代码、原有标签及其原始manifest均已保留在包中，但 `data/legacy_ea_lite` 的标签来自旧自采轨迹，**不对应本次官方Clean轨迹，不能直接与官方RGB/动作配对训练**。官方原始HDF5只有endpose/joint_action/observation/pointcloud，没有ea_rsig_oracle_v3；缺少物体/目标/接触和bbox等EA监督。目前不生成伪标签，也不自动改成auxiliary=true。正式EA切换需先获得官方轨迹对应、可验收的标签。

本机EA旧run按用户确认继续运行，不受baseline打包影响。两种数据的baseline/EA结果不能解释为仅架构差异；后续对照必须使用同一数据源、归一化、增强和预算。此前自采数据包属于过期候选，不应上传；只上传最终标为official的交付目录。
