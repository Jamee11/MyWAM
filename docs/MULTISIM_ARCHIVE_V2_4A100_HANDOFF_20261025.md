# 多仿真 Sidecar midtrain：独立归档与四卡 A100 启动

日期：2026-10-25。现有 Aloha/ManiSkill 单源训练保持不变。本入口是新的跨仿真训练阶段。

## 路径和验收范围

项目：`/remote-home/jinminghao/OpenWAM`。数据：项目内 `datasets/multisim_sidecar_archive_v2_20261025`。

正式训练清单为数据根下 `manifest.ready.json`；只有最终逐条校验及四组真实数据加载成功后才发布。`manifest.training.candidate.json` 是发布过程的中间文件，启动时不要使用它。

- `native/`：五个指定来源的完整独立物理副本，以及验收文件引用到的历史 pilot 原生样本。不是原文件的软链接或硬链接。
- `training/`：已有 52,204 条选定样本的数值/130D 标签及训练集统计。训练 39,816、val 6,161、test 6,227。原 ManiSkill 未选入 midtrain 的额外原始数据仍完整保存在 native，未擅自扩入训练。
- `evaluation/`、`replay/`、`provenance/`：原六任务评估 test、代码/资产/原始元信息；原生回放的跨环境依赖闭包仍须独立验收，数据复制完成不等于可以清理所有旧目录。
- `_archive_state/*.sqlite`：逐文件原/副本 SHA256 与独立 inode 证据，`training_release.json` 是训练数据发布证明。

原来的目录复制检查另外发现：540 条 RLBench、32 条 ManiSkill-hard 视觉样本的 production 验收文件指向目录外的历史 pilot。已追加复制它们对应的 212 个完整原生 attempt、9,400 个文件、70,741,015,728 字节；保持其原 split 和身份，不重新采集或替换轨迹。

RoboTwin 保持 `endpose[t]` 输入与未来实测绝对 `endpose[t+1:t+33]` 动作，部署按同一参考系的绝对 EE 执行；ManiSkill/RLBench 保持已验证的绝对控制目标。所有来源均不是 delta EE，绝对 commanded 与 achieved 仍分别注明。12 组 RoboTwin 编码往返检查已通过，不冒充仿真实际轨迹回放。

工程验证区分：CPU 数据/优化器路由和入口回归已执行；完整 Foundation 双卡短测曾在前两次 warmup 更新后、首次 WAM Adam 状态分配时 OOM，不能当成完整训练验收。四卡完整联合更新/保存/恢复尚需目标机器实测；本地没有四张空闲 80GB 卡，未结束已有作业。

## 迁移

将代码包覆盖到目标项目，完整数据目录放到目标项目 `datasets/` 下。`manifest` 的 numeric/media/statistics 路径相对数据根，可整体搬迁；原来源绝对路径只作为 provenance，不参与训练取样。训练不需要安装 ManiSkill、RLBench、RoboTwin 仿真器。

已有 `.venv` 与 Foundation 模型沿用；代码包不包含环境、TB 数据或预训练权重。`BASE_MODEL` 必须指向完整 Foundation 目录，包括其配置/组件/tokenizer。保持离线，不自动下载或绕过 hf-mirror。

推荐用 rsync 保留时间戳并断点复制整个数据目录。若只能 SFTP，请先使用保留纳秒时间戳的 pax tar，再传 tar 文件并解压；不要直接 `put -r` 后认为文件 mtime 纳秒仍一致，启动检查会核对这一点。

```bash
# 源机器，可选；需要额外约 3.75 TB 空间容纳未压缩 tar。归档目录正在修改时不要执行。
tar --format=pax -cf /足够空间/multisim_sidecar_archive_v2_20261025.tar \
  -C /remote-home/jinminghao/OpenWAM/datasets multisim_sidecar_archive_v2_20261025

# 目标机器，数据目标目录不能已有其他版本的同名文件。
cd /home/mingfeihangroup/xiaoming/projects/OpenWAM
mkdir -p datasets
tar -xf /已上传路径/multisim_sidecar_archive_v2_20261025.tar -C datasets
```

不要重新运行旧 preparer 覆盖本次归档。原始来源保留，尚无安全清理许可。

## 数据预检与四卡启动

下面在目标项目根执行。`CHECK_ONLY=1` 不加载完整模型、不占用训练卡；`--full` 是迁移后一次性的全量 numeric hash/media stat 检查，常规启动只检查 12 个真实窗口。

```bash
cd /home/mingfeihangroup/xiaoming/projects/OpenWAM
export PYTHON_BIN="$PWD/.venv/bin/python"
export MIDTRAIN_MANIFEST="$PWD/datasets/multisim_sidecar_archive_v2_20261025/manifest.ready.json"
export BASE_MODEL="$PWD/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model"

"$PYTHON_BIN" scripts/check_multisim_sidecar_data.py --manifest "$MIDTRAIN_MANIFEST" --full
CHECK_ONLY=1 bash scripts/train_multisim_sidecar_4a100.sh
```

四张卡应为 A100 **80GB**。默认 4×每卡32×累积4＝global batch512；增加累积不增大每个前后向的 batch，不能据此承诺速度翻倍或显存减半。若每卡32实测余量不足，可用 `BATCH_SIZE=16 GRAD_ACCUM=8` 保持 global512；不要开启 CPU optimizer offload 绕过当前 WAM warmup hook 的保护。

首次目标机器运行，先做 8 微步完整模型工程验收（约几分钟计算，另加环境/模型加载及保存时间）。只为验证两次 Sidecar warmup 更新、两次联合更新、两个 checkpoint 和完整优化器状态；不是正式训练预算：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Slurm 内不要覆盖调度器给的可见卡。
export BATCH_SIZE=1 GRAD_ACCUM=2 NPROC_PER_NODE=4
export MIDTRAIN_OUTPUT_DIR="$PWD/outputs/midtrain/acceptance_$(date -u +%Y%m%dT%H%M%SZ)"
CHECK_ONLY=0 bash scripts/train_multisim_sidecar_midtrain.sh \
  training.max_steps=8 training.num_epochs=null training.save_steps=4 \
  training.warmup_ratio=0.5 training.dataset_num_workers=1 \
  model.architecture.relational_sidecar.backbone_warmup_updates=2
```

验收要看真实有限 loss、micro8/opt4、warmup 后 WAM 权重变化及完整 `accel_state_step_4/8/trainer_state.json`。恢复测试必须使用同 world/batch/accum/manifest/预算；正式训练新建 run，不能把短测改预算伪作 resume。具体恢复时指定实际含 `config.yaml` 的 run 目录；本入口会拒绝不匹配的恢复契约。未完成这项实测前，不宣称四卡验收成功。

正式 1 epoch：

```bash
export BATCH_SIZE=32 GRAD_ACCUM=4
export MIDTRAIN_EPOCHS=1
export MIDTRAIN_OUTPUT_DIR="$PWD/outputs/midtrain/multisim_1ep_b512_$(date -u +%Y%m%dT%H%M%SZ)"
CHECK_ONLY=0 bash scripts/train_multisim_sidecar_4a100.sh
```

正式 2 epoch：将 `MIDTRAIN_EPOCHS=2` 并换新的输出目录。该入口明确设置 `training.max_steps=null`，按 epoch 跑完；不会误沿用旧默认 100,000 microstep 截断。

Slurm（从项目根提交，调度分配任意合格节点，不绑定 gnode06）：

```bash
export MIDTRAIN_EPOCHS=1  # 或 2
sbatch --export=ALL --partition=GPU-8A100 --qos=gpu_8a100 \
  scripts/cluster/multisim_sidecar_4a100.sbatch
```

`gpu_8a100` 是此前用户提供的账户允许 QoS，避免先前 Invalid qos specification。其他集群须使用其实际 partition/QoS；本脚本不会更改账户授权。Slurm 默认 10 天：2 epoch 的线性估计约 8.8 天，I/O/保存/排队后的运行余量偏紧；依赖保存的完整状态续跑，不能更改恢复契约。`MIDTRAIN_OUTPUT_DIR` 不要沿用短测目录。

当前配方（按用户最新要求）：Foundation 初始化；Sidecar 正常读写/完整 Action+Video+EA 反向；前 500 **optimizer updates** 仅不更新 WAM 的参数/Adam 状态，之后联合更新。累积4对应前2,000微步，WAM第501次更新开始参与；不是关闭 WAM 反向或关闭 Sidecar 写回。旧2K方案保留为历史实验，新midtrain配置单独改为500。WAM LR2e-5、Sidecar1e-5、EA总scale0.5、ColorJitter开启、BF16/ZeRO2/gradient checkpointing；每4,000微步保存，最近3份完整状态保留。没有改网络或辅助目标数学。

## 1/2 epoch 时间估算

用户参考：Aloha50 Clean2500，549,775 窗口/epoch，global256，10 epoch 约80小时。新数据每个混合 epoch 7,282,176 窗口；四个来源等权抽样，这是 mixture 的虚拟 epoch，不代表每条原始轨迹恰好访问一次。

同样每样本吞吐下：`80h × (7,282,176 × 新epochs)/(549,775 × 10)`。

| 预算 | global512 更新数 | 累积4的微步数 | 线性参考时间 |
|---|---:|---:|---:|
| 1 epoch | 14,223 | 56,892 | 105.97小时，约4.4天 |
| 2 epoch | 28,446 | 113,784 | 211.93小时，约8.8天 |

global512 通过增加累积获得；在同四卡上，每次更新的前后向量约翻倍，更新次数减半，总计算量近似不变。不能直接给出53/106小时。上述参考相当于约19.09窗口/秒、global512约26.82秒/update，非新数据实测。共享盘随机读、RGB布局解码、增强、跨机环境、检查点保存、LR阶段等会改变结果；冷启动、迁移、排队不计入上述小时数。

目标机正式启动后，等联合更新阶段稳定，统计至少100次 optimizer update 的墙钟吞吐，包含其间实际微步；用 `(剩余窗口/实测窗口每秒)` 重估。不要把 warmup 首步、首次编译或单个 microstep 当作一次 global512 更新。
