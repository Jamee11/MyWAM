# bitdistill迁移与2K权重8卡新训练

目标根目录 `/mnt/100T/lixiangyu/cuigm/bitdistill`，解压后项目在其 `OpenWAM/` 下。
仅实现用户确认的第一个版本：加载当前PatchWAM run的2K optimizer-update权重，然后**新优化器、新scheduler、step0、新10epoch**。不读取旧optimizer/RNG，不重复2K WAM延迟更新，不启用100/300步解冻ramp。仍保留标准5%学习率warmup＋cosine；这是逐步增大学习率，期间WAM也更新。第二版本待用户说明；官方Clean2Random权重先完整携带，不自动选作训练起点。

## 包内容

- `openwam_code.tar.gz`：当前工作树的训练/模型/配置/脚本/测试、AGENTS.md、许可和必要文档；不只是git HEAD。不含环境、旧outputs、模拟器或多源数据。
- `openwam_aloha50_data.tar.gz`：原已完成的真实数据归档直接复用，源manifest SHA与当前一致；50任务2500 HDF5、labels、统计、prepared元数据、instructions/scene信息。原HDF5保留所有已有字段，非软链接；不含其它仿真数据、Randomized、PKL/视频或模拟器运行资产。当前训练仍用这些源HDF5中的绝对EEF序列，不转换动作/重新标注。
- `openwam_warmup2k.tar`：用户指定的固定step4000权重，附该run的config/tokenizer/normalization_stats/Sidecar contract，放到 `OpenWAM/assets/openwam_ckpt/openwam_alpha/Sidecar-PatchWAM-Warmup2K/`。step4000为2K更新。没有打包旧优化器；这些完整模型权重包含重建所需主干，训练不需另下载Foundation。
- `openwam_clean2random.tar`：官方 `OpenWAM-Alpha-Sim-RoboTwin-Clean2Random`，含step10740、原config/tokenizer/统计和校验记录，保持独立目录，不混用它的统计作为Sidecar训练统计。
- `SHA256SUMS`和逐包inventory：写包时计算的哈希、成员大小、PAX纳秒mtime，供目标验证。数据包沿用原已完成归档的hash，不重新扫描35GB内容。

四包必须全部完成且SHA256SUMS存在后再开始传输；不要上传`.partial`。

## 1. 在源服务器SFTP上传

`BUNDLE_DIR`替换为本次提供的打包目录；主机、用户名、端口按实际填写。

```bash
sftp -P <SSH端口> <目标用户名>@<目标主机>
```

SFTP中：

```text
lcd BUNDLE_DIR
cd /mnt/100T/lixiangyu/cuigm/bitdistill
mkdir incoming_openwam_20261028
cd incoming_openwam_20261028
put openwam_code.tar.gz
put openwam_aloha50_data.tar.gz
put openwam_warmup2k.tar
put openwam_clean2random.tar
put *.inventory.json
put SHA256SUMS
put TRANSFER_INFO.json
put README_TRANSFER.md
bye
```

断线后用`reput 文件名`续传。目标父目录需已存在且当前账号可写。

## 2. 目标服务器校验并解压

```bash
BASE=/mnt/100T/lixiangyu/cuigm/bitdistill
cd "$BASE/incoming_openwam_20261028"
sha256sum -c SHA256SUMS
# 新目录交付：不覆盖已有同名项目。已有OpenWAM时先选另一个BASE。
tar --keep-old-files -xzf openwam_code.tar.gz -C "$BASE"
PROJECT="$BASE/OpenWAM"
mkdir -p "$PROJECT/datasets"
tar --keep-old-files -xzf openwam_aloha50_data.tar.gz -C "$PROJECT/datasets"
tar --keep-old-files -xf openwam_warmup2k.tar -C "$BASE"
tar --keep-old-files -xf openwam_clean2random.tar -C "$BASE"
```

总归档约73GB，解压内容约85GB，建议目标至少预留170GB迁移空间，训练权重/恢复状态另留空间。不复制旧venv或临时盘环境。GNU tar恢复PAX纳秒mtime，勿通过丢失精度的文件复制再搬HDF5。

## 3. 环境与一次性路径迁移

使用目标机兼容的Python3.10训练环境。已有环境通过`OPENWAM_VENV`指定；若未准备环境，可运行项目现有安装脚本（需要uv和兼容CUDA），不要每次训练重复安装：

```bash
cd "$PROJECT"
# 仅在没有兼容环境时：
# PYTHON_BIN=python3.10 bash scripts/cluster/install_environment.sh
export OPENWAM_VENV="$PROJECT/.venv"  # 或已有训练环境的绝对路径
DATA="$PROJECT/datasets/robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015"
META="$DATA/openwam_sidecar/aloha50"
"$OPENWAM_VENV/bin/python" scripts/relocate_ea_lite_manifest.py \
  --manifest "$META/manifest.source.json" \
  --raw-root "$DATA/raw/aloha" --data-dir "$META" \
  --output "$META/manifest.ready.json"
```

这一步只迁移路径、检查文件size/mtime，不重新构造样本和标签；输出存在时不覆盖，直接使用已有ready manifest。原source_dataset_root保留作来源记录，不是训练实际读取路径。数据包和2K模型的统计一致，不使用官方Clean2Random的另一份统计。

## 4. 8卡正式启动

```bash
cd /mnt/100T/lixiangyu/cuigm/bitdistill/OpenWAM
export OPENWAM_VENV="$PWD/.venv"  # 按目标已有环境修改
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 只解析配置，不占GPU；可跳过：
CHECK_ONLY=1 bash scripts/train_sidecar_patchwam_warmstart_8gpu.sh

RUN="$PWD/outputs/sidecar/patchwam_warmstart8_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN"
OPENWAM_OUTPUT="$RUN/runs" nohup bash scripts/train_sidecar_patchwam_warmstart_8gpu.sh \
  > "$RUN/launch.log" 2>&1 &
echo "PID=$! OUTPUT=$RUN"
tail -f "$RUN/launch.log"
```

Slurm分配时保留调度器给出的CUDA_VISIBLE_DEVICES，不覆盖为物理0–7；通过srun调用同一脚本即可，本站partition/QOS未知故不编造sbatch参数。脚本固定torchrun单节点8进程；可用MASTER_PORT改变端口。OPENWAM_INIT_CKPT覆盖模型目录、OPENWAM_MANIFEST覆盖ready manifest、OPENWAM_OUTPUT覆盖新输出，不需要全项目替换绝对路径。

配置 `train_sidecar_patchwam_warmstart_8gpu_10epochs`：每卡batch32×8、acc1/global256；WAM标称LR1e-4、Sidecar5e-5、EA监督和完整共享梯度保持；PatchWAM全部参数沿用当前方案，旧ColorJitter关闭；10epoch约21480更新。第10000、20000更新保存权重和恢复状态，训练结束另存最终权重（预计step21480）；最终不另写非周期fullstate。原2K权重不进入新输出，不与新checkpoint混淆。

这改变了world size和optimizer历史，不能声称与本机4卡acc2逐步数值完全相同。加载2K模型后WAM和Sidecar第一步都参与更新；原本冻结的文本编码器/VAE仍冻结。脚本不会自动下载模型，也不会抢占/结束旧进程。

## 验证范围

按用户要求，只做文件/配置/路径和小型archive逻辑检查，不做GPU短测或全数据重解码。尚未验证目标机8卡显存、NCCL和环境；当前每卡batch32按80GB级GPU方案准备。near/contact参考点、物理EEF回放、强增强颜色语义以及匹配baseline等既有研究边界保留，不因迁移悄悄改标签或训练目标。
