# RoboTwin Sidecar：隔离 warmup 新训练方案

时间：2026-10-21 UTC。用户授权在“联合训练但 Sidecar 不影响 backbone”与“冻结 backbone”之间选择最小且合理的实现；采用前者。原有 joint 配置、历史权重和运行中任务保留。本轮未启动正式训练。

## 训练行为

|阶段|Backbone|Sidecar|向主干写回|EA 辅助监督回传主干|
|---|---|---|---|---|
|前 2000 次 optimizer update|原有 video/action FM 训练|读取 detach 后的当前观测特征，训练关系/计划等辅助能力|关闭|阻断|
|完成 2000 次 update 后|原有联合训练|完整读/精炼/写分支|开启，沿用原零初始化 write gate|恢复|

这里的 2k 是真实参数更新次数。当前 acc=2，所以第 1–4000 个 microstep 为隔离阶段，从第 4001 个 microstep 起恢复完整联合训练。切换发生在累积更新边界，不在一个累积窗口内改变目标。warmup 包含在原来的 10 epochs 中，不额外增加 2k 更新。训练仍使用同一数据流，不重启模型、不重新加载权重、不重建优化器。

隔离时所有 Sidecar 读取的主干特征均 detach；初始化 token 所用 text/proprio 也 detach。三处 write_action 完全跳过，并非仅把 gate 初始化为零。视频与动作主干继续用原始 FM 目标学习，Sidecar 用已有 EA 标签学习 read/refine/token/head；write adapter 在此阶段没有任务梯度，联合阶段才开始学习动作写入，原有 write gate=0 保证初始残差为零。未新增额外的注入 ramp。

选择原因：现有静态 `stage=warmup` 在优化器创建前把 backbone 参数排除；仅在 2k 后设置 requires_grad=True，不能可靠地补入 DeepSpeed ZeRO2 的参数分组和分片状态。隔离方案从开始就注册完整联合训练参数，只改变计算图的连接，复用原有优化器、调度器、保存与数据加载流程。此处“warmup”与原有学习率的 5% linear warmup 是不同概念。

## 学习率与辅助权重

- action/video backbone 的标称学习率保持 `1e-4`；新增独立 Sidecar 参数组，标称 `sidecar_lr=5e-5`。其他顶层模块继续使用原 base LR，没有一起降速。
- 保留已有 linear warmup + cosine。已有调度器对各组使用共同绝对下限 `1e-6`：Sidecar LR 全程不高于 backbone，标称为一半，接近训练末尾时二者趋向同一下限；不是全程固定 0.5 倍。
- 新增 `aux_loss_scale=0.5`，对整个 EA extra loss（包括 deep supervision）统一减半。原 ground/relation/plan 权重保持 `.025/.01/.05`，等效为 `.0125/.005/.025`；deep supervision 的相对比例仍 `.1`。
- 这不是“EA 必须占总 loss 的某个百分比”。实际各 loss 的数值、梯度尺度会变化；日志保留 `loss_ea`（最终加权值）、各未加权分项、`loss_ea_deep`（原始 deep 值），新增 `ea_isolation_warmup`（1 隔离、0 联合）。

共享的全局 gradient clipping 仍沿用原方案。CPU 验证的是主干前向及**裁剪前梯度**与无 Sidecar 一致；当全局裁剪触发时，Sidecar 梯度会参与共同范数，从而可能间接改变主干的缩放系数。不能将“计算图隔离”表述为整个优化轨迹与 baseline 逐位一致。

## 配置、启动和恢复

新配置：`configs/train_relational_sidecar_aloha50_isolated_10epochs.yaml`。
新入口：`scripts/train_relational_sidecar_aloha50_isolated_10epochs.sh`，复用现有正式训练 launcher 和 CPU-ready/GPU handover 门槛。

继承旧方案的 Aloha50/2500 episodes 数据清单、32 帧 action/9 帧 video/80D 合同、bs32/acc2/四卡、10 epochs、ColorJitter 开启、原数据/归一化与随机 seed。初始化来源仍为同一个 Foundation，不能把旧42960权重当成新的随机 Sidecar 对照起点。默认输出到 `outputs/sidecar/aloha50_isolated2k_10epochs`。

```bash
cd /remote-home/jinminghao/OpenWAM
export EA_HANDOVER_DIR="$PWD/outputs/sidecar/handover_isolated_$(date -u +%Y%m%dT%H%M%SZ)"
CHECK_ONLY=1 bash scripts/train_relational_sidecar_aloha50_isolated_10epochs.sh
# GPU0–3 可用、准备正式训练时，将上面的 CHECK_ONLY 改为 0。
```

启动沿用共享盘 `.venv`；没有复制新环境到 /tmp 或 /dev/shm。CHECK_ONLY 只解析配置，不占训练卡。正式启动的 handover 路径必须全新，原脚本仍要求四张 A100-80GB、CPU 准备后等待至少90秒且各卡空闲显存≥70000MiB。

Trainer 每个 microstep 前将已完成的 `opt_step` 传入 architecture，恢复完整训练状态后也先使用恢复后的计数。计数本身不新增模型 tensor，因此旧 checkpoint 的 state_dict 键不变；原配置 `isolation_warmup_updates=0`、`aux_loss_scale=1` 时保留原行为。推理 `.eval()` 始终走完整 Sidecar，不依赖训练进度。只加载 safetensors 做 finetune 会从 update=0 开始；要继续原阶段，必须使用原有完整 resume state，而非仅加载权重。

当前 RoboTwin 每 epoch 4296 microsteps，与 acc2 对齐，沿用原恢复映射。若改数据/批量使 epoch 长度变成奇数，不可直接用 microstep//acc 替代真实更新数；ManiSkill 的独立恢复修正仍保留，未在本次顺带重构。

## 验证和边界

新增 CPU 小模型测试覆盖：有非零 write gate 时隔离阶段前向仍与无 Sidecar 一致；EA loss 对主干无梯度、Sidecar read/head 有梯度；checkpointing 开关下主干裁剪前 FM 梯度一致；1999→2000 的阶段边界；切换后恢复写入和 EA→backbone 梯度；权重保存/恢复及 eval 不受计数影响；包含 deep supervision 的完整 EA loss 减半；参数分组完整且不重复；调度全程 Sidecar LR≤backbone LR；实际 trainer 循环在 acc2 下恢复1999更新后传入 `[1999,1999,2000,2000,2001,2001]`。

新配置 Hydra resolve、新 launcher 的共享盘环境 CHECK_ONLY（exit0）及新旧 launcher shell 语法检查通过。CPU 小模型与模拟 Accelerator 不等于真实四卡 ZeRO2 验收；当前未运行 GPU 短测、完整训练或新模型闭环评测。ZeRO2 使用现有 `ignore_unused_parameters=True`，隔离阶段 write adapter 无梯度；恢复写入后的真实多卡归约和显存峰值仍需短测。

下一步先用独立输出做小阈值（例如2更新）跨阶段四卡短测，检查有限 loss、read/write 梯度、阶段切换、完整状态恢复和显存，再训练正式2k方案。评估需与原方案固定相同数据/总预算/统计和部署协议，优先比较 adjust_bottle、place_shoe、beat_block_hammer 的 Clean/Randomized；此前不同训练预算、统计和接受 seed 的对比不能唯一归因随机 Sidecar。该方案能减少早期干扰，但没有提升成功率的实验保证；缓存质量影响仍需独立固定 manifest 对照。

验证记录：两批相关 CPU 测试分别41/41、33/33通过（含重复覆盖，不相加）；第二批包括新增调度器与实际 trainer 累积/恢复计数测试。配置和验证摘要保存于 `docs/evidence/sidecar_isolated_warmup_20261021/`。`git diff --check` 通过。
