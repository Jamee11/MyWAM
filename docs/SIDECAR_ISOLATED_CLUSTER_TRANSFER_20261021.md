# Sidecar 隔离 warmup：Slurm 服务器迁移包

2026-10-21。目标项目 `/home/mingfeihangroup/xiaoming/projects/OpenWAM`，目标数据 `/home/mingfeihangroup/xiaoming/datasets/`。复用目标机已有 `.venv` 和训练 baseline 时使用的 Foundation；不安装环境、不重复携带权重、不提交作业。本包包含最新源文件及未提交的 Sidecar 实现，不仅是 git HEAD 的内容。

## 文件和数据范围

- `openwam_sidecar_isolated_code.tar.gz`：以 `OpenWAM/` 为顶层，包含训练框架、配置、脚本、小型FK资产、测试、AGENTS.md、修改记录和必要文档。覆盖更新项目代码；不含 `.venv`、`.git`、输出权重、baseline数据、日志或模拟器。归档不是删除同步，目标已有 `.venv`、outputs、data、ckpts不会被清理。
- `openwam_sidecar_aloha50_data.tar.gz`：2500条、50任务 Aloha Clean HDF5，**当前这一批轨迹对应的**已构造Sidecar标签、统计、prepared启动元数据，以及源instructions/scene文件。HDF5逻辑大小35,417,456,051 bytes；552275帧、549775训练窗口。只取当前manifest引用的Aloha Clean，不含Piper、Randomized、预览视频或PKL。不是旧baseline包里的legacy EA标签，不可互换。
- `SHA256SUMS` 与各包的 `.inventory.json`：归档SHA256和逐文件大小/hash/mtime；真实内容解引用保存，PAX保留纳秒mtime。标签/统计打包时与现有manifest hash比对。
- `TRANSFER_INFO.json`：数据来源、目标路径、数量和归档摘要。`README_TRANSFER.md`：本说明。

数据解压后：

```text
/home/mingfeihangroup/xiaoming/datasets/
  robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015/
    raw/aloha/<task>/demo_clean/data/episode*.hdf5
    raw/aloha/<task>/demo_clean/{instructions/,scene_info.json}
    openwam_sidecar/aloha50/
      labels/<task>/episode*.npz
      normalization_stats.npy
      manifest.source.json
      manifest.ready.json  # 解压后用现成迁移脚本一次性生成
```

## 1. SFTP 上传

在SFTP中将 `SOURCE_BUNDLE_DIR` 替换成给出的打包目录。目标建议使用全新的incoming子目录，避免与旧SHA256SUMS混淆。

```text
lcd SOURCE_BUNDLE_DIR
mkdir /home/mingfeihangroup/xiaoming/incoming/sidecar_isolated_20261021
cd /home/mingfeihangroup/xiaoming/incoming/sidecar_isolated_20261021
put *.tar.gz
put *.json
put SHA256SUMS
put README_TRANSFER.md
```

如incoming父目录不存在先创建它。大文件中断使用 `reput openwam_sidecar_aloha50_data.tar.gz` 续传。本次不需要发送已有Foundation或环境。

## 2. 校验、备份当前代码、覆盖解压

在目标服务器shell运行：

```bash
BASE=/home/mingfeihangroup/xiaoming
PROJECT="$BASE/projects/OpenWAM"
INCOMING="$BASE/incoming/sidecar_isolated_20261021"
cd "$INCOMING"
sha256sum -c SHA256SUMS

mkdir -p "$BASE/backups" "$BASE/datasets"
# 只备份这次涉及的现有代码目录/小文件；不复制环境、权重或训练输出。
backup_items=()
for item in openwam configs scripts tests assets/ea_robotwin AGENTS.md LICENSE \
    EA_RSIG_DIT4DIT_LICENSE EA_RSIG_README.md README.md pyproject.toml \
    requirements-ea-lock.txt requirements-ci.txt MODIFICATION_LOG.md \
    docs/思考与隐患.md docs/BASELINE_CLUSTER_TRANSFER_20261015.md \
    docs/ROBOTWIN_SIDECAR_ISOLATED_WARMUP_20261021.md \
    docs/SIDECAR_ISOLATED_CLUSTER_TRANSFER_20261021.md; do
  if [[ -e "$PROJECT/$item" ]]; then backup_items+=("$item"); fi
done
tar -czf "$BASE/backups/OpenWAM_before_isolated_$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
    -C "$PROJECT" "${backup_items[@]}"
tar -xzf openwam_sidecar_isolated_code.tar.gz -C "$BASE/projects"
# 数据使用全新目录，防止无意覆盖已有同名数据；GNU tar保留PAX纳秒时间。
tar --keep-old-files -xzf openwam_sidecar_aloha50_data.tar.gz -C "$BASE/datasets"
```

若目标已经有同名数据目录，先停在解压数据这一步，使用一个新目录解压，再设置 `OPENWAM_DATASET=/新目录/robotwin2_...`。不要把本包覆盖混入另一版本数据，也不要覆盖 `.venv`。代码包会更新原有 baseline 共用源码和脚本，已在本机回归检查；目标自己修改过的文件应从上述backup对照恢复定制配置。

## 3. 一次性迁移路径并预检

```bash
cd "$PROJECT"
DATA="$BASE/datasets/robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015"
META="$DATA/openwam_sidecar/aloha50"
FOUNDATION="$BASE/ckpts/OpenWAM-Alpha-Pretrain-Foundation-Model"
# Foundation在其他位置时修改此处，并在sbatch前export OPENWAM_FOUNDATION="$FOUNDATION"。

.venv/bin/python scripts/relocate_ea_lite_manifest.py \
  --manifest "$META/manifest.source.json" \
  --raw-root "$DATA/raw/aloha" --data-dir "$META" \
  --output "$META/manifest.ready.json"

.venv/bin/python scripts/preflight_ea_lite.py \
  --manifest "$META/manifest.ready.json" --foundation "$FOUNDATION" --data-check full

CHECK_ONLY=1 bash scripts/cluster/train_sidecar_isolated.sh
```

迁移脚本只改manifest的路径，核对源size/纳秒mtime，不重新构造数据、不重算归一化或EA标签。源manifest保留作为来源记录，其中 `source_dataset_root` 是原服务器的来源字段，训练实际使用已迁移的 `source_root` 和每条episode路径。输出已存在时拒绝覆盖；重复操作时直接使用已有ready manifest进行预检。`full`预检一次核对全部2500条的元信息和标签hash；每次启动仅fast预检，不对每步训练增加数据校验。

CHECK_ONLY不读取完整模型/不分配GPU。若集群登录节点需要module，先按baseline已有方式加载环境，或在计算节点执行预检；无需重新安装 `.venv`。若Foundation不在默认位置：

```bash
export OPENWAM_FOUNDATION=/实际Foundation目录
```

## 4. Slurm短测和正式提交

分区沿用之前文档的 `mhan0035`，单节点四卡、一个srun任务启动四个torchrun worker；不固定物理GPU索引，不要求卡必须编号0–3，也不使用本机CPU-ready的90秒handover等待。保留调度器的 `CUDA_VISIBLE_DEVICES`。资源不足可以排队，不要手动覆盖调度器分配。

先做一次跨阶段短测：临时把隔离期改为2次更新、总计8个microstep（4次更新），使用独立输出，验证2次隔离+2次联合。保留num_epochs=10/max_steps=8使原调度器短测计算方式不变；短测并非正式训练速度或效果结论。

```bash
cd "$PROJECT"
OPENWAM_CLUSTER_SETUP="$PWD/scripts/cluster/setup_hanhai.sh" \
OPENWAM_OUTPUT="$PWD/outputs/sidecar/isolated_smoke_$(date -u +%Y%m%dT%H%M%SZ)" \
sbatch --gres=gpu:a100:4 scripts/cluster/sidecar_isolated_clean50.sbatch \
  model.architecture.relational_sidecar.isolation_warmup_updates=2 \
  training.max_steps=8 training.save_steps=4
```

短测第1–4个microstep应记录 `ea_isolation_warmup=1`，第5–8个为0，write gate在联合更新后可开始变化；检查loss有限、完整state保存和实际恢复。真实四卡ZeRO2切换尚未在源服务器验证，应在目标先跑此项。默认bs32档沿用四张80GB卡配置；不是40GB卡容量已验证方案。

短测通过后正式提交（不带短测覆盖项）：

```bash
OPENWAM_CLUSTER_SETUP="$PWD/scripts/cluster/setup_hanhai.sh" \
sbatch --gres=gpu:a100:4 scripts/cluster/sidecar_isolated_clean50.sbatch
```

如果需要account或节点，按照现有baseline成功命令额外传 `--account=...`、`--nodelist=...`；本说明不强绑之前可能draining的gnode06。站点GRES名称不同则按实际覆盖，不修改训练配置。setup_hanhai仅复用原CUDA12.8 module加载；已有不同setup可通过OPENWAM_CLUSTER_SETUP替换。

正式训练：50任务2500条、ColorJitter开启、bs32/卡×4、acc2/global256、10epoch；前2000真实更新/4000microstep隔离，后续联合，warmup包含在10epoch内；Sidecar标称LR5e-5、video/action1e-4、EA总loss含deep乘0.5，保留原cosine共同绝对floor。当前549775窗口对应每epoch4296microstep，10epoch42960microstep/21480次更新。默认输出 `outputs/sidecar/aloha50_isolated2k_10epochs/` 下；每rank保存metrics_rank*.jsonl，其中含隔离标志、EA各loss、gate、步数及显存峰值。

```bash
squeue -u "$USER"
tail -f slurm-openwam-sidecar-isolated2k-<JOBID>.out
```

恢复同一个新方案run（不要把baseline或旧Sidecar full state当作新方案恢复源）：

```bash
OPENWAM_CLUSTER_SETUP="$PWD/scripts/cluster/setup_hanhai.sh" \
sbatch --gres=gpu:a100:4 scripts/cluster/sidecar_isolated_clean50.sbatch \
  training.finetune_ckpt_path=null \
  training.resume_ckpt_path=/实际新方案run目录
```

恢复必须使用完整state，才能继承优化器、调度器和更新计数；只加载safetensors做finetune会重置计数。

## 验证边界

源服务器已验证CPU隔离前向/裁剪前梯度、阶段边界、累积/恢复计数和LR分组；本次新增Slurm脚本与归档迁移检查另见同目录验证记录。没有连接目标服务器、没有替用户传输/解压/覆盖文件，也没有提交Slurm。尚无此方案成功率提升证据；共享全局clip的间接影响、真实ZeRO2切换/恢复、缓存部署对成功率的影响仍须现场验证。原baseline数据与当前数据略有差异，不能宣称与旧baseline比较仅改了Sidecar架构。

本次迁移新增检查：`test_sidecar_cluster_transfer.py`、既有`test_ea_lite_handoff.py`和`test_sidecar_isolation_trainer.py`共12项通过；实际新cluster launcher CHECK_ONLY解析到指定目标路径，bash语法检查通过。归档writer Python3.10兼容已修复，纳秒mtime经GNU tar实解压验证。


## 本次实际交付与最终验收（2026-10-21T07:33:43.341432+00:00）

源服务器打包目录：`/remote-home/jinminghao/OpenWAM/outputs/transfers/sidecar_isolated_20261021T071002Z`。两个压缩包均已完成，不是partial；请上传该目录全部文件。

|包|字节数|SHA256|
|---|---:|---|
|openwam_sidecar_isolated_code.tar.gz|1641685|`eccfdbb8f06e729720f2bc3e62d15ef468755011655b395332420b2e3e26c738`|
|openwam_sidecar_aloha50_data.tar.gz|23448238134|`c4dbcc666c712746fd7283b64ddff91f45c90be37ed340132806d1af94e5bf75`|

代码包536个文件，已从压缩包实际解压并通过51项相关CPU测试；数据包7552个文件，包含2500 HDF5与2500标签NPZ。两个归档全部成员hash、PAX纳秒mtime、普通文件/无链接与安全相对路径、整包SHA256及gzip CRC已流式复核通过。另从数据归档取出真实episode与标签验证路径迁移、full预检和带ColorJitter训练样本读取（32×80 action/9 video/1×130 EA targets）。这些均不是目标机器上的GPU训练验收。

目录中的ARCHIVE_VALIDATION.json、VALIDATION.json、DATA_SAMPLE_VALIDATION.json记录证据；TRANSFER_INFO.json记录来源与目标。源代码归档在数据压缩开始时固定，归档内本说明/修改日志不含这段压缩完成后的交付补记，训练源码不受影响；以归档inventory和SHA256为准。目标机仍需按上文步骤校验、代码备份、覆盖、数据解压、一次性重定位和Slurm短测。
