# Multi-Layer Relational Sidecar 实施与交接（2026-10-16）

实现依据：OpenWAM_alpha_MultiLayer_Relational_Sidecar_Codex_Impl.md。保留旧Lite与原版，新增独立版本；本次是工程搭建，不是效果评估。

## 架构和数据合同

- 原版：`model=dual_system`；旧版：`model=dual_system_ea_rsig_lite`；新版：`model=dual_system_relational_sidecar`。旧版checkpoint继续使用旧模型配置，新版不能宽松加载旧EA权重。
- 新模块注册于architecture.sidecar，15个token在block7/15/23之后持续更新；只读runtime H×W首帧，额外15token不进入MoT，不改mask/RoPE/QKV。
- 从当前归一化80D proprio+80D dimension mask和masked text mean初始化，不使用oracle EEF/arm/context。Action仅作为写入Query，不参与Z更新。输入维度mask从reader或部署normalizer恢复；GT130D独立用于loss。
- Ground4，Relation4，InteractionSummary1，Plan6。Summary没有百分比进度监督。全部stage更新全部有效token；最终三loss，stage1ground/stage2relation有0.1深监督。read alpha=.01；write beta=0，投影保持非零随机。
- 使用请求数据集 `robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015/raw/aloha`；不读取Piper，不以LeRobot 14D qpos充当EEF。原始HDF5不改写；source size/mtime、label/stats hash绑定和prepared startup复用现有实现。
- 新labels/EEF stats从此批HDF5独立构建。32绝对EEF动作、9视频、Plan8/16/32，每角色local frame，wxyz→xyzw/列Rot6D，原EE/TCP参考点分离，local-X .12m合同沿用已审计映射。
- 每视角归一化bbox在loss中显式映射到L形全画布，不再固定14×14。goal_valid=0不可形成可见goal监督；无效arm/track/未来尾部均mask。RGB沿用legacy decoder，准备程序检查全部episode三视角首尾标记并拒绝XPL-RGB1；不声称全JPEG逐帧验收。

## 一次性准备和版本切换

```bash
cd /remote-home/jinminghao/OpenWAM
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/prepare_relational_sidecar_data.py \
  --dataset-root /remote-home/jinminghao/datasets/robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015 \
  --output-dir outputs/sidecar/aloha_4350
# 准备程序拒绝覆盖已有目录；已准备好后直接使用ready manifest，不重复执行。
CHECK_ONLY=1 EA_STAGE=warmup bash scripts/train_relational_sidecar.sh
# 旧Lite入口保持原样：
CHECK_ONLY=1 bash scripts/train_ea_lite_4gpu.sh
```

默认新manifest：`outputs/sidecar/aloha_4350/manifest.ready.json`。换机器用EA_SIDECAR_MANIFEST覆盖，复用 `relocate_ea_lite_manifest.py` 迁移路径并保留纳秒mtime；不要直接放宽校验。完整代码、Foundation、raw HDF5、准备好的labels/stats/ready manifest和环境锁需分别搬迁。新版本独立配置将存入部署checkpoint的model_config，不能仅替换权重不替换config/normalizer。

## 四卡20次真实更新命令（待在空闲A100执行）

```bash
# 160 microsteps / accumulation8 = 20 optimizer updates；每40 microsteps保存完整状态。
CUDA_VISIBLE_DEVICES=0,1,2,3 CHECK_ONLY=0 EA_NUM_GPUS=4 \
  EA_GPU_NAME=A100 EA_GPU_PROFILE=80 EA_STAGE=warmup \
  EA_STOP_AFTER_OPT_STEPS=20 EA_OUTPUT_DIR=outputs/sidecar/smoke_warmup \
  bash scripts/train_relational_sidecar.sh
# 确认warmup稳定、保存恢复通过后，可新开Action适配；并非自动延续上个warmup权重：
CUDA_VISIBLE_DEVICES=0,1,2,3 CHECK_ONLY=0 EA_STAGE=action_only \
  EA_STOP_AFTER_OPT_STEPS=20 EA_OUTPUT_DIR=outputs/sidecar/smoke_action \
  bash scripts/train_relational_sidecar.sh
```

要接续warmup权重或完整状态，分别显式覆盖 `training.finetune_ckpt_path=<保存weights目录>` / `training.resume_ckpt_path=<完整状态目录>`，resume时同时设 `training.finetune_ckpt_path=null`；不要把warmup当普通SFT。`EA_EXPERIMENT=A/B/C` 分别baseline/aux-only/read-refine-write，初始调试默认bs1、acc8、BF16、ZeRO2、checkpointing、LR1e-4和相同ColorJitter；正式对比应统一训练阶段、global batch/总更新/数据/控制器。Video部分解冻、LoRA、action-plan-align不实现默认流程，不宣称硬件容量。

逐rank `diagnostics_rank*.jsonl` 含每项aux loss、三read/write gates、所有新参数梯度范数、三层token梯度范数、microstep时间、显存allocated/reserved峰值；acc8的更新时间应汇总8个microstep，不把单个microstep时间叫optimizer step时间。输出目录中的完整Accelerate状态与部署weights各有不同用途。

## 部署

`configs/deploy_relational_sidecar.yaml`关闭compile与DiTcache、CFG1、sync、32动作/9视频。沿用官方部署入口和相同绝对EEF/controller，不发送ea_targets或旧ea_effectors；当前state由normalizer变成80D proprio，当前维度mask由normalizer散射表恢复。每次MoT forward都从当前观测重建Z，三处stage均执行。调试不能用旧Lite专用请求字段或旧原生14D/40步入口。

## 验收状态

实施与测试进行中；最终数字和证据路径完成后补在本节。CPU tiny检查不能替代全量Foundation BF16、四卡ZeRO保存恢复或闭环验证。当前既有GPU评测作业不操作。
