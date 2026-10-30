# Frozen-backbone 2K：旧“不写回2K”版本的增量更新

版本说明（2026-10-22后续修订）：下文09:48归档保留的是初版WAM叶梯度清零逻辑，不包含当前“WAM保留完整真实梯度、只跳过optimizer更新”的修订。使用新语义需同步最新sidecar_warmup.py及对应测试/说明；不能把旧归档的hash或验收当作新版本证据。本轮未重打或覆盖该历史归档。

目标项目 `/home/mingfeihangroup/xiaoming/projects/OpenWAM`。本包只基于此前 `sidecar_isolated_20261021T071002Z/openwam_sidecar_isolated_code.tar.gz` 增量更新；旧版保留。不是完整新项目，不重新携带数据/环境/权重，不覆盖目标全量修改日志和隐患文档。

新增逻辑：前2000次optimizer更新Sidecar正常读写、WAM更新冻结，之后直接joint。仍bs32/卡×4、acc2/global256、10epochs、ColorJitter、Sidecar LR5e-5、WAM1e-4、aux总scale .5。不叠加其他路由修改，不启动第二段WAM-only阶段。

## 1. SFTP上传（进入sftp后执行）

```text
lcd /remote-home/jinminghao/OpenWAM/outputs/transfers/sidecar_frozen_backbone_delta_20261022T094829Z
cd /home/mingfeihangroup/xiaoming
mkdir sidecar_frozen_20261022
cd sidecar_frozen_20261022
put openwam_sidecar_frozen_backbone_delta_20261022.tar.gz
put SHA256SUMS
put README_TRANSFER.md
put UPDATE_MANIFEST.json
put BASE_TO_NEW.patch
put VALIDATION.json
```

`mkdir`目录已存在时跳过即可。本次只有小代码包，不传数据，不改已重定位的manifest。

## 2. 校验、备份被覆盖文件、解压

在目标shell执行。使用旧isolation版本作为基础，旧训练/新训练应使用独立输出。若旧作业仍在运行，建议在它结束后覆盖共用源码，避免跨进程混用版本。

```bash
BASE=/home/mingfeihangroup/xiaoming
PROJECT="$BASE/projects/OpenWAM"
BUNDLE="$BASE/sidecar_frozen_20261022"
cd "$BUNDLE"
sha256sum -c SHA256SUMS

mkdir -p "$BASE/backups"
backup_items=()
while IFS= read -r item; do
  item="${item#OpenWAM/}"
  if [[ -f "$PROJECT/$item" ]]; then backup_items+=("$item"); fi
done < <(tar -tzf openwam_sidecar_frozen_backbone_delta_20261022.tar.gz)
for item in MODIFICATION_LOG.md docs/思考与隐患.md; do
  if [[ -f "$PROJECT/$item" ]]; then backup_items+=("$item"); fi
done
tar -czf "$BASE/backups/OpenWAM_before_frozen2k_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"   -C "$PROJECT" "${backup_items[@]}"
tar -xzf openwam_sidecar_frozen_backbone_delta_20261022.tar.gz -C "$BASE/projects"

# 增量记录追加；不会覆盖目标已有历史记录。
cat "$PROJECT/docs/SIDECAR_FROZEN_BACKBONE_UPDATE_LOG_20261022.md" >> "$PROJECT/MODIFICATION_LOG.md"
cat "$PROJECT/docs/SIDECAR_FROZEN_BACKBONE_UPDATE_LOG_20261022.md" >> "$PROJECT/docs/思考与隐患.md"
```

UPDATE_MANIFEST逐文件记录旧/新hash。目标如果对上述覆盖文件做过本地定制，先对照backup及BASE_TO_NEW.patch保留定制，不要覆盖后丢失路径/环境设置；路径建议沿用OPENWAM_*环境变量。

## 3. 配置预检

复用已有`.venv`、Foundation和数据。默认位置与之前包相同：

- Foundation：`/home/mingfeihangroup/xiaoming/ckpts/OpenWAM-Alpha-Pretrain-Foundation-Model`
- 数据：`/home/mingfeihangroup/xiaoming/datasets/robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015`
- manifest：上述数据目录下`openwam_sidecar/aloha50/manifest.ready.json`

```bash
cd /home/mingfeihangroup/xiaoming/projects/OpenWAM
# 若此前成功启动需要module，继续沿用原setup；无需重新安装环境。
source scripts/cluster/setup_hanhai.sh
OPENWAM_TRAIN_CONFIG=train_relational_sidecar_aloha50_frozen_backbone_10epochs CHECK_ONLY=1 bash scripts/cluster/train_sidecar_isolated.sh
```

检查`isolation_warmup_updates: 0`、`backbone_warmup_updates: 2000`、`color_jitter.enabled: true`。需要不同路径时，使用已有OPENWAM_FOUNDATION/OPENWAM_DATASET/OPENWAM_MANIFEST/OPENWAM_VENV覆盖。首次已做过数据重定位和full校验则复用；启动仅fast预检。

## 4. GPU-8A100：先四卡跨阶段短测

脚本`scripts/cluster/sidecar_frozen_backbone_gpu8a100.sbatch`：GPU-8A100、单节点4GPU、32CPU、最长10天，自动选择有资源的节点，不锁gnode06或用户列出的某一节点。一个srun任务内torchrun4worker，保留Slurm的CUDA_VISIBLE_DEVICES。默认每作业独立输出含job ID。

```bash
cd /home/mingfeihangroup/xiaoming/projects/OpenWAM
OPENWAM_CLUSTER_SETUP="$PWD/scripts/cluster/setup_hanhai.sh" OPENWAM_OUTPUT="$PWD/outputs/sidecar/frozen2k_smoke_$(date -u +%Y%m%dT%H%M%SZ)" sbatch scripts/cluster/sidecar_frozen_backbone_gpu8a100.sbatch   model.architecture.relational_sidecar.backbone_warmup_updates=2   training.max_steps=8 training.save_steps=4
```

短测只缩阶段阈值：前4个microstep应`ea_backbone_frozen_warmup=1`，后4个为0；loss有限，write gate可开始学习，保存权重和完整恢复状态，无NCCL/OOM。短测不是训练效果或2000步适配完成证据。bs32仍是四张80GB卡配置，分区名称本身不能证明GPU显存；作业头部打印nvidia-smi。若站点要求带类型GRES，可在sbatch命令加`--gres=gpu:a100:4`，沿用此前成功的账户设置。

## 5. 正式提交

短测通过后，不带缩阈值/步数覆盖项，另起新的正式训练：

```bash
cd /home/mingfeihangroup/xiaoming/projects/OpenWAM
OPENWAM_CLUSTER_SETUP="$PWD/scripts/cluster/setup_hanhai.sh" sbatch scripts/cluster/sidecar_frozen_backbone_gpu8a100.sbatch

squeue -u "$USER"
```

日志`slurm-openwam-sidecar-frozen2k-JOBID.out/.err`；输出默认`outputs/sidecar/aloha50_frozen_backbone2k_jobJOBID/<日期>/`。前4000microstep冻结WAM，之后联合；正式2000指真实更新，文件checkpoint_step编号仍是microstep。

新策略要从Foundation新建训练，不加载旧isolation的完整optimizer状态继续跑：后者已更新过主干、没有提前训练Write，不能变成新策略的前2K。恢复本新策略自身run时才用training.finetune_ckpt_path=null和training.resume_ckpt_path=新run路径。

切回旧版：保留的`scripts/cluster/sidecar_isolated_clean50.sbatch`仍使用旧不写回模式；新GPU-8A100脚本固定新模式。不要把旧sbatch默认分区mhan0035与本次新脚本混淆。

验证范围：源端两卡tiny BF16/ZeRO2/acc2跨阶段和Adam恢复通过；增量包在旧包解压目录重新执行CPU回归，结果见VALIDATION。未连接目标服务器、未传输或提交Slurm、未验证全量四卡与成功率。
