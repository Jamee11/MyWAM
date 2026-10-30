# OpenWAM-α + EA-RSIG-Lite：4×A100 条件下的实现与实验方案

> 版本：2026-10-14 · 用途：研究方案冻结、工程实现与 Codex 开发依据  
> 方案性质：**拟实施设计**，不是已复现的结果。核心目标是在短时间内建立可运行、可验证、可公平比较的预训练 WAM + 任务关系增强系统；**不承诺必然提升 Randomized 成功率**。

## 0. 执行摘要

**推荐主线：OpenWAM-α Foundation → 官方 RoboTwin Clean-only EEF 微调 → EA-RSIG-Lite（Grounding + Typed Relation + Multi-Horizon Plan）→ Gated Action Adapter → 同预算闭环对照。**

1. **保留 OpenWAM-α 原始预训练能力**：Wan2.2-5B Video DiT、独立 Action DiT、Video–Action 联合去噪机制、80D 统一动作槽位和官方 EEF 动作语义；不要同时重写 Action Space 与视频主干。
2. **EA-RSIG 只读取因果可观测信息**：当前 RGB、指令、当前 EEF proprioception；从尚未融合未来视频／动作的视觉特征中提取空间特征，不复用可能泄漏未来信息的中间 DiT hidden states。
3. **复用 v3 的核心算法**：Object/Goal Grounding、双轨 EEF→Object→Goal Typed Relations、Progress summary、Multi-Horizon Future Motion Plan；将 horizon 从 40 步的 10/20/40 转换为 OpenWAM 32 步的 **8/16/32**。
4. **改变 Action 接入方式**：不向 OpenWAM 的 joint self-attention 主序列强行插入 15 个新 token；将 `[4 Ground + 4 Relation + 1 Progress + 6 Plan]` 经轻量 cross-attention / projection 转为 Action Stream 的残差条件，使用零初始化投影或小门控稳定初始化。
5. **首版不迁移训练期 Video Residual 的第二次 forward**，不立即开发 Future Relation Transition / Action–Relation Consistency；先保留官方视频和动作训练目标，再加上现有辅助 loss。
6. **硬约束为 4×A100**：如为 80GB 版，先测试完整 optimizer step 与吞吐；若为 40GB，需把冻结／offload／更小训练方案设为主路线，不能假定 5B full fine-tuning 可行。
7. **核心证据是配对闭环结果**：同 Foundation、同 Clean 数据、同有效 batch／更新预算、同 EEF 解码与评测；重点比较 50-task Clean / Randomized SR，而非只看离线 RMSE。

## 1. 研究定位与已有证据

### 1.1 要解决的问题

生成式 WAM 的视频与动作主干能够学习强大的运动先验，但对操作任务而言，**当前目标、机器人末端、被操作物体以及未来的目标关系变化**未必获得稳定且可部署的显式表征。EA-RSIG 希望为预训练 WAM 提供一个 task-centric、robot-centric 的中间接口：

`Current Observation → Task Grounding → EEF–Object / Object–Goal Relations → Future Foresight → Action Guidance`。

希望验证：**在强机器人预训练模型上，关系条件是否能独立提升视觉域偏移下的闭环操作能力。** 这不同于单纯展示更低的 auxiliary loss，也不是因为换用了更强的 backbone 就将收益归给 EA-RSIG。

### 1.2 旧 DiT4DiT 实验给予的启示

- 现有 EA-RSIG-v3 在相同离线样本、图像、语言、噪声条件下，Randomized 的末端位置误差从旧 Baseline 的 **62.43 mm 降到 30.05 mm**，姿态误差从 **18.94° 降到 8.91°**；但 checkpoint 的训练步数不同，不能严格归因于新结构。
- Randomized **闭环成功率仅由 3.4% 增至 4.4%**。说明专家轨迹上的动作误差降低并不自动转化为可靠闭环成功，基础动作先验、执行误差和 OOD 状态分布依然是关键瓶颈。
- EA 条件的 token intervention 已确认会影响 Action；但 Grounding/Plan 在 Randomized 中退化，GT bbox attention oracle 也未明显恢复平均离线动作精度。这说明不仅是“有没有关系模块”，还要确保其可靠、可控、能在强基础策略上创造净收益。

因此选择具备大规模机器人 Video–Action 预训练的 OpenWAM-α 进行验证，同时**维持旧 DiT4DiT 为已有实验和迁移失败时的备份**。

## 2. Backbone 与 Action 约定：哪些必须保持

### 2.1 OpenWAM-α 基础结构

- 基础模型：OpenWAM-α Foundation（Wan2.2-TI2V-5B 视频主干 + 独立 Action DiT）。
- 模型机制：World–Action 双流联合训练／注意力交互，**Video Stream 与 Action Stream 并非两个完全隔离的前向网络**。
- Action Contract：固定 **80D** unified slots，RoboTwin 双臂夹爪 EEF 有效部分为 **20D**（每臂 XYZ 3 + Rotation6D 6 + gripper 1）。其余维度由相应 mask 控制。
- 官方 RoboTwin 微调建议使用其已支持的 `eef` 目标；**它是 EEF 绝对目标位姿语义，不等同于 delta EEF**。官方 checkpoint 的其他机器人通道／mask 也不能被解释为全部数据都提供同一套物理参考坐标系下的 EEF 标签。
- 时域：官方 Action Horizon **32**，按已验证的 32-step 训练窗口和视频采样协议对齐；不要在第一轮一并修改为 40 步。

官方入口：
- 代码与配置：[OpenWAM-Official/OpenWAM](https://github.com/OpenWAM-Official/OpenWAM)
- Foundation 模型：[OpenWAM-Alpha-Pretrain-Foundation-Model](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model)
- RoboTwin 数据配置：`configs/dataloader/robotwin.yaml`（以当前 checkout 为准）

### 2.2 为什么第一版不改 delta EEF 作为模型内部 Action Target

OpenWAM 现有的机器人 Action Head 已在固定 EEF 目标定义上训练。此时把输出标签全部改为 chunk-relative delta，会改变动作分布、旋转编码、归一化与控制器契约，可能损失最需要利用的预训练先验。

**推荐分层表示**：

- **Action Head**：保留官方 `T_EEF^base(t+h)` 绝对目标；
- **EA-RSIG**：围绕相对几何／相对运动建模，如 `T_EEF^{-1} T_Object`、`T_Object^{-1} T_Goal`（必要时使用可获得的简化监督）；
- **Robot Adapter**：根据当前末端位姿将预测绝对目标转换为真机执行器要求的 delta／IK／绝对 EEF 指令。

完整 SE(3) 相对运动：`ΔT = (T_EEF,current^base)^(-1) · T_EEF,target^base`。若使用 base-frame 平移差分 `Δp = p_target^base - p_current^base`，它与 body-frame SE(3) delta 不完全等价；旋转须使用合法的相对旋转而非 Rot6D 数组相减。

跨本体收益来自共享的任务关系／运动语义和可复用的控制接口，**不会仅因输出 delta EEF 就自动实现 zero-shot Sim2Real**。Robot-specific frame、IK、工作空间、工具坐标系、gripper 与频率仍需适配。

## 3. 目标整体架构与信息流

```text
                           OPENWAM-α FOUNDATION
          ┌──────────────────────────────────────────────┐
Current RGB + Text ─► Pretrained Video/Action World Model │
          │              原 Video/Action FM Loss           │
          └───────────────────┬──────────────────────────┘
                              │ Original Action Hidden States
Current RGB ─► Causal Observation-only Spatial Features
Text + Current EEF proprioception ───────┤
                              ▼
                       EA-RSIG-Lite
                     Grounding: O/G slots
                  Typed Relations: EO / OG
                   Progress latent summary
               Multi-Horizon Plan: 8/16/32
                              │
                    15×512 EA latent tokens
                              ▼
                Lightweight Conditional Aggregator
                   + Zero-init Gated Adapter
                              │
                              ▼
                  OpenWAM Action Stream (1024D)
                              ▼
                   Original Action Output Head
                              ▼
                   32-step Absolute EEF Targets
                              ▼
                RoboTwin / Real Robot Adapter
```

**强制的因果边界**：EA 特征只依赖当前 observation。不能将已经混入训练时未来视频 latent、未来真实 Action、动作 diffusion noise 的 Video DiT hidden state 输入 EA-RSIG。OpenWAM 的 `first_frame_causal` 不自动保证经历 cross-stream mutual attention 之后仍然因果安全。

## 4. EA-RSIG-Lite 四个保留模块

### 4.1 Observation-only Spatial Features：新增安全输入接口

优先尝试直接复用 OpenWAM **当前帧 VAE／Patch Embedding 在未来／动作混合之前的空间特征**，通过轻量 projector 转换为适配 Grounding 的 2D spatial grid。需要保留多相机空间布局和各相机 bbox 对应关系；**不能把画布平均池化成全局向量**。

- **首选 A**：当前帧 VAE/patch spatial features + 轻量 spatial projector，计算开销较低；风险是低层语义不足。
- **备用 B**：Frozen DINOv3 spatial features，提高视觉语义质量的候选；但需要测额外延迟、显存和相机对齐。
- **暂不做 C**：额外执行完整 Video DiT 的独立 observation-only forward，因 4×A100 的训练吞吐与部署延迟压力较大。

必须做可视化／数值检查：三相机 grid 的像素映射、bbox heatmap、各 role 的有效性、当前帧外观扰动时的关注位置。此处特征提取接口的代码位置须通过实际 repo forward trace 确认，不应假设某个固定 `H18` 层可以直接替代。

### 4.2 Task Grounding：复用 v3

保留四个 Role Slots（双轨 Object/Goal）、presence、track-valid、null slot 与 bbox 分布监督，结合当前 EEF proprioception 建立双臂 EEF slots。每个角色 token 约 **512D**。

`S_obj = Attention(Q_obj, F_obs)`，`S_goal = Attention(Q_goal, F_obs)`。

注意：RoboTwin 中某些 Goal 是虚拟目标／无可见 bbox，不能强制填入不存在的框；角色身份与左右臂选择须来自一致的 task/role registry，而非仅凭空间距离重新猜测。

### 4.3 Typed Relation：复用 v3 的双轨关系

两条轨道：`EEF_i → Object_i → Goal_i`，`i ∈ {L,R}`。

`R_EO = LN(MLP([S_EEF; S_obj; e_EO]))`；
`R_OG = LN(MLP([S_obj; S_goal; e_OG]))`。

输出 `R_EO, R_OG ∈ R[B,2,512]`，共同形成 4 个 512D Relation Tokens。先继续使用已有 contact / near / goal-proximity 标签与有效性 mask；第一轮**不新增完整 Object 3D pose / contact dynamics head**。

### 4.4 Progress + Multi-Horizon Foresight：复用并调整时域

- Progress：现有对关系 slots 的 Attention Summary，**不是有监督的任务完成百分比**。
- Plan：双轨 × 3 个相对 horizon，预测 EEF/Object 的单位运动方向与 moving probability。
- 原 DiT4DiT：`H={10,20,40}`；迁移 OpenWAM：**`H={8,16,32}`**。
- 输出核心张量：`plan_direction: [B,3,2,2,3]`，`moving_logits: [B,3,2,2]`（最后两个维度为 EEF/Object role 与 XYZ／moving）。
- 若 Direction 标签原先定义为 EEF-local frame，应沿用原 converter 的参考系约定；不可直接与 base-frame EEF pose 做逐维比较。

**第一版不实现完整 Future Relation Transition**（未来 contact、相对 XYZ 转移等）。只有在最小模型显示明确闭环收益、数据可用时才增加，避免同时改变太多训练因素。

## 5. 核心新增：Gated Residual Action Adapter

原 v3 的 15 个 condition latents 保留：`Ground 4 + Relation 4 + Progress 1 + Plan 6 = 15`，形状 `Z_EA∈R[B,15,512]`。

OpenWAM 不直接在 Joint Self-Attention 主序列拼接它们，而使用单独的聚合器：

1. 从 Action time positions 构造 `Q_action`（最初可按 32 个动作位置查询；实际应以官方 Action Stream 的真实序列 layout 为准）；
2. `C_EA = CrossAttn(Q_action, K=Z_EA, V=Z_EA)`；
3. `C_EA` 投影到 Action DiT hidden size（约 1024D），形状与相应 Action Hidden States 对齐；
4. **仅在一个确定的 Action Block Boundary** 添加残差：

`H_action' = H_action + γ · W(C_EA)`。

**初始行为等价性**：末端投影 `W` 零初始化，或者 gate `γ=0`，确保未经训练的 EA 分支不会立即破坏 Foundation 模型。实施时需监控零初始化第一批次的梯度路径（gate 可以先更新，投影随后更新）。

主要优点：不增加 joint attention 序列长度；不需要重写 Video/Action RoPE 和 mask；Action Output Head 不变；新增参数较少，适合 4 卡资源；方便 adapter-on/off 诊断。

**关键实现限定**：官方 Backbone hooks 如 `pre_attn_at_layer`、`post_attn_at_layer` 可能提供可扩展点，但不能假设现成 YAML 就有 EA 开关。必须检查源码真实执行顺序、残差在 mixed attention 前或后，以及 Action 预测的梯度能否传回 EA 分支。若接口不满足，新增最小受控调用点，并保证 baseline 默认路径不变。

## 6. 首版训练目标与梯度策略

推荐保留 OpenWAM 官方 World–Action 目标，同时加入 EA auxiliary：

`L_total = L_OpenWAM + λ_EA · L_EA`，

`L_OpenWAM = λ_action L_action + λ_video L_video`，

`L_EA = λ_ground L_ground + λ_rel L_relation + λ_plan L_plan`。

EA losses：
- Ground：bbox heatmap CE + presence BCE + track-valid BCE；
- Relation：contact BCE + near BCE + goal-proximity 误差；
- Plan：masked moving BCE + masked direction cosine loss。

可以先尝试名义上的总损失系数 Ground **0.025**、Relation **0.01**、Plan **0.05** 做 sanity check；这些不是经过验证的最优值，必须按照 loss 尺度、有效样本数与梯度范数调整。

**Video Residual 暂不迁移**：DiT4DiT-v3 第二次 Future Video Forward 的残差机制第一轮移除；OpenWAM 原本已有 World–Action joint attention。Relation Adapter 经 Action Stream 可能参与后续联合注意力，但“可传播”不意味着已证明能改善 Video Loss，须另行验证。

**训练阶段**：
- Warmup：预训练主干暂时冻结，先验证 Grounding/Relation/Plan 与 adapter 的前反向、有效 mask、梯度连通性和初始化无偏移；有限 100–300 optimizer steps 仅做 sanity check。冻结主干参数不保证 activation 显存自动下降，需实测。
- Joint SFT：逐步解冻需要更新的 Action／Video 参数，或按 4 卡资源选择 adapter-first/partial unfreeze；保持低 backbone LR，并监控预训练能力退化。
- 正式对照：Baseline 与 EA 必须共享 **相同 Foundation 初始化**；Warmup 增加的额外步数要记录并提供 matched budget 对照。

## 7. RoboTwin 数据迁移契约

**不能直接把原来 `14D qpos + 22D EEF + 130D oracle targets` 拼入 OpenWAM 80D proprio。** 未来信息只作为 `ea_targets` 的 Training-only 字段，推理时不能读取。

| 数据项 | 第一版处理 | 风险检查 |
|---|---|---|
| 三相机 RGB / Instruction | 复用 Clean 示范，走官方 RoboTwin reader | 图像尺度、L-shaped canvas、三视图同步 |
| 14D qpos | 通过官方／已校验 FK 转为 EEF current state | Base/frame、旋转、左右臂、夹爪 |
| Action | 官方 32×80D，RoboTwin EEF 有效 20D | Absolute target、normalization、mask、control adapter |
| Object/Goal bbox | 复用 role registry 与标签，按新 canvas 重新映射 | 无可见框／虚拟 Goal／相机维度 |
| Current near/contact/goal | 延用原定义与 valid mask | 标签同步、实例身份 |
| Future Plan | 重新按 8/16/32 生成 | Expert future、尾部 valid、局部坐标系 |
| Future Relation Transition | 暂不引入 | 无需本轮重新采集 |

原 v3 的 130D 布局只用于理解已有监督，不强制复用为新模型硬编码输入；可在新数据管线中显式存储 label dict，以降低后续扩展风险。

重点测试：Action 与 current state 是否对应正确的 33-state window；视频帧采样与 32-step action 时域是否严格对齐；BBox 归一化基于**每个 view 自身尺寸**；train/test 的数据轨迹、角色和有效标注是否一致。

## 8. 四卡 A100：工程门槛与降级路线

**4×A100 80GB**：应先验证真实峰值显存和完整 optimizer step；官方推荐 8×80GB、部分 4×H100 80GB 配置有内存通过记录，**不能视为 4×A100 训练速度或适配 EA 后的可行性保证**。

**4×A100 40GB**：不能把 5B 双流全参微调当作既定可行方案；应优先用冻结／Adapter／partial unfreeze、offload，必要时重新决策 backbone。

建议显存节约顺序：
1. micro-batch=1；启用模型确实支持的 activation/gradient checkpointing；
2. ZeRO-2、gradient accumulation，与官方代码的配置项核对；
3. optimizer CPU offload（注意吞吐退化）；
4. 冻结部分 Backbone / 仅训练 Adapter 并测梯度路径；
5. 若仍 OOM 或训练耗时不可接受，申请 8 卡或切换备选。

以下为**配置意图示例，不保证每个 CLI key 在所 checkout 版本中原样存在**；正式命令必须通过 `--help` / Hydra config dump / 一次 dry-run 验证：

```bash
NPROC_PER_NODE=4 bash scripts/train.sh \
  model=dual_system \
  dataloader=robotwin \
  dataloader.variant=clean_50 \
  dataloader.action_mode=eef \
  training.finetune_ckpt_path=/path/to/foundation \
  training.batch_size=1 \
  training.gradient_accumulation_steps=8 \
  training.zero_stage=2 \
  training.use_gradient_checkpointing=true \
  training.max_steps=20
```

**真实验收必须包含**：完整 forward、backward、optimizer.step、至少两次保存／恢复、稳定 step-time、峰值显存、单 chunk 推理延迟与 RoboTwin 解码闭环；不能只检查模型加载与 GPU forward。

## 9. 训练与实验日程（以可用时间为约束）

| 阶段 | 建议目标时间 | 具体动作 | 停止／继续标准 |
|---|---|---|---|
| G0 基础资源验收 | Day 1–2 | 官方 Foundation + RoboTwin EEF full optimizer step、保存恢复、吞吐 | OOM/吞吐无法承受则降级，不开发 EA |
| G1 接口验收 | Day 2–3 | EEF decoder、causal observation feature、zero-init identity | 未通过 causality/identity 不训练 |
| G2 EA Sanity | Day 3–5 | Ground/Relation/Plan + Adapter 小步训练 | loss、mask、梯度与定位均健康 |
| 第一轮正式 SFT | Day 5–12 左右 | Baseline A 与 EA C 同 Foundation 并行或顺序 | 早期 Clean/Random 闭环没有明显恶化 |
| G3 方法评估 | Day 8–14 左右 | 配对关键任务 SR、动作误差、Ground/Plan OOD | 无可重复净收益则不追加复杂模块 |
| 扩展评测与论文 | 后续时间 | 50-task、Aux-only B、关键消融、真机条件检查 | 论文只报告实际复现结果 |

天数是工程目标，不是 OpenWAM 的实测训练时长。**14 天左右设为继续投资的决策截止点**，尽量为最终评测、重复试验与论文写作保留时间。

## 10. 对照实验：三组最小主实验

| 实验 | Foundation 初始化 | EA auxiliary | Relation/Plan 进入 Action | 目标 |
|---|---|---|---|---|
| **A：OpenWAM Baseline** | 相同 | 无 | 否 | 基础预训练 WAM 的 Clean-only OOD 性能 |
| **B：OpenWAM + EA Aux-only** | 相同 | 有 | 否 | 区分监督本身和条件机制的作用 |
| **C：OpenWAM + EA Conditional** | 相同 | 有 | 是（Gated Adapter） | 证明结构化条件是否有独立价值 |

注：B 必须明确 auxiliary 与可训练共享模块是否连通；如果完全隔离、只更新独立 Heads，B 不会提供 backbone 表示监督，实验解释需相应限制。A/B/C 必须控制训练步数、总训练 token/样本曝光、有效 batch、数据增强、Action space、控制器、推理 denoising 与评测预算。**不要用官方已在 RoboTwin-C2R 后训练的 checkpoint 作为 A/B/C 的 Clean-only 初始权重**，它只适用于工程参考／上限比较。

**第一优先指标**：官方完整 50-task Clean / Randomized closed-loop SR、任务级 SR、配对差值及 episode-level 不确定度。**辅助指标**：专家状态 Action error（位置／姿态／gripper）、Grounding OOD、Plan direction/moving、token intervention、推理延迟、各任务首次失败阶段。不要再将离线动作误差改善直接等价为闭环鲁棒性提升。

## 11. 必须通过的三类防错测试

**T1. Adapter identity：** 固定同一 Foundation、输入、timestep、diffusion noise、精度与执行路径，EA disabled 或 zero-init 时应与官方前向数值一致（允许预先定义的数值容差）。

**T2. Observation causality：** 固定当前 RGB、指令、EEF state，独立扰动 future video、future action、diffusion noise、future-step labels；EA Ground/Relation/Plan 条件不应改变。若改变，存在未来泄漏或不正确状态依赖。

**T3. Train/deploy consistency：** 同一个当前 observation 的 EA 条件在确定性训练测试与部署前向下保持一致。训练用 Oracle Target 绝不能作为推理输入。

其他检查：RoboTwin role registry／左右臂；相机 grid／bbox；Goal invalid；32-step time alignment；有效 mask；Offload/checkpointing 与 DDP/ZeRO 在多卡条件下的实际梯度；Action absolute EEF 恢复与 IK／控制器追踪误差。

## 12. 研究强化路径：第一轮出结果后只选一条

**路线 A：Domain-Stable Relational Foresight（优先针对视觉 OOD）**。对于同一物理状态、不改变任务语义的 appearance perturbations，约束 Relation/Plan 预测一致：`L_stable = D(R(o),R(aug(o))) + D(P(o),P(aug(o)))`。保持原 GT 监督，避免常数解；务必设置同 augmentation、但没有 relational consistency 的 Baseline，对比区分图像增强与算法收益。使用官方 Randomized 域渲染训练时须披露训练域暴露，不能继续声称严格未见 Randomized。

**路线 B：Future Relation Transition（优先针对阶段／执行问题）**。在已有稳定关系感知基础上，预测 `R_t → R_{t+h}`，包括 EEF–Object／Object–Goal 相对几何与 future contact/goal 状态，并把 transition latent 作为动作条件。需要额外的 future object/goal/contact GT；**不能假定现有 130D 标签足以重建全部 future transition**。暂不为此额外重写完整 action-conditioned object dynamics。

**当前不建议同时实现 A、B**；应由 G3 实验发现决定。

## 13. 已知风险、边界与 Codex 交付验收

| 风险 | 直接应对 |
|---|---|
| 4×A100 无法完成或训练过慢 | G0 立即量化峰值显存/秒每步，优先降低配置而非长期盲跑 |
| Observation feature 缺乏目标语义 | 先跑 Grounding sanity；必要时启用 Frozen DINOv3 备选 |
| Joint Attention 泄漏训练未来 | T2 causality 必过；只在未来/动作混合之前取当前 observation features |
| EEF 标签与真实控制器不一致 | 官方 action contract、FK/base frame、Rot6D、gripper、mask 端到端单测 |
| EA 破坏预训练能力 | zero-init adapter、adapter-on/off identity、梯度范数、低 LR |
| 新 head 只降低离线误差 | 主指标坚持 Randomized closed-loop SR 与失败阶段，不以 RMSE 替代 |
| 不公平的训练对照 | 同 Foundation/同 Clean 数据/同更新预算；单独报告额外 warmup 与参数预算 |
| 真实跨本体结论不足 | 真机明确记录本体差异、控制适配与真实数据使用；不可把机器人专用微调称作 zero-shot |

**Codex 开发要求**：保留官方 Baseline 路径与配置；新增独立 EA 模块、EA target reader、adapter、配置开关、诊断日志和测试；修改点必须可以关闭并回退。首轮交付先是**能重现官方输出的无侵入 Adapter + 数据对齐 + 因果测试**，再是完整训练入口。不要先修改全局混合注意力、视频扩散目标、Action 80D 语义或官方控制器。

## 14. 决策结论

**采用 OpenWAM-α 的机器人预训练与绝对 EEF Action Prior，EA-RSIG 负责当前任务角色、相对几何和多时域前瞻，并通过一个轻量零初始化 Action Adapter 指导已有 Action Stream。**

本轮成功的最低标准是：

1. 在 **4×A100** 条件下完成可复现训练与控制闭环；
2. 通过 observation causality、adapter identity 和 action/data contract 检查；
3. 同一 Foundation 对照中，EA-RSIG 相对于 OpenWAM Baseline 展现**可重复、具有任务意义的 Randomized 闭环收益**；
4. 保留仿真到真机的相对 EEF 几何接口，但**不为 Delta Action 直接重训预训练 Head**。

最终论文命题应是：*Task-centric relational foresight 是否能在强机器人预训练 WAM 上进一步增强视觉域偏移下的可执行动作泛化*；不是“把 OpenWAM 的官方成绩换算成 EA-RSIG 的方法贡献”。

---

### 参考资源与说明

- 官方代码：https://github.com/OpenWAM-Official/OpenWAM
- 预训练 Foundation：https://huggingface.co/OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model
- RoboTwin 接口：https://github.com/OpenWAM-Official/OpenWAM/tree/main/benchmarks/robotwin
- 预训练与官方约定以实际下载的 repo commit、checkpoint config、fine-tuning 文档为准；**本文中对工程可行性、插入位置和新损失的描述属于建议方案，未经在用户 4×A100 环境端到端实测。**

---

## 15. 跨服务器 Session 交接：先明确当前状态

**2026-10-14 UTC 实现记录。第1–14节是原方案；以下补充实际代码、已测结果、失败经历和下一台服务器的操作流程。原方案的研究成功标准仍未达到。**

当前状态为：**EA-RSIG-Lite 实现和 CPU 准备完成，停在实际四卡调试之前。** 用户要求找到四卡后再实测，本次交接准备没有启动 GPU 训练／推理。不要把 CPU 通过、旧 V3 一步训练通过，或官方文档的硬件说明写成 Lite 四卡已通过。

新 Session 的第一项工作：阅读本项目 `AGENTS.md`、`docs/思考与隐患.md` 和本文第15节以后内容，然后核对实际硬件和迁移资产，先做 A baseline 的资源验收，再做 C warmup/joint。不要先运行旧 V3 batch sweep，也不要从头重复下载已验证的 Foundation。

代码基线为官方 commit `4d7f642b841913ea3c176010b692b1de3d5ef180`；本地 EA 修改包含尚未提交的新文件，**仅重新 git clone 官方仓库不会得到当前实现**。另有逐项实现验收表：[实现与四卡调试准备](OpenWAM-alpha_EA-RSIG-Lite_实现与四卡调试准备_2026-10-14.md)。历史依据在 `MODIFICATION_LOG.md`、`docs/2026-10-13_OpenWAM训练吞吐与预算估算.md`，以较新的更正为准。

### 15.1 新旧实现不要混用

| 项目 | 历史迁移 V3 | 当前 Lite，后续工作的默认路径 |
|---|---|---|
| 模型入口 | `dual_system_ea_rsig_v3` | `dual_system_ea_rsig_lite` |
| Action/state | native14 qpos 路径，重置4个政策头 | Foundation 80D heads 原样继承，RoboTwin 有效20D EEF |
| 时域 | 40 actions / 5 video frames | 32 actions / 9 video frames，33-state window |
| 观测特征 | 独立多层 observation Video DiT pass | 仅当前 clean VAE latent 的 patchify + 轻量投影 |
| EA 注入 | 旧 context／video residual 方案 | 15×512 条件经单点 zero-init Action adapter |
| 最新 GPU 证据 | 一步通过，连续第二步 OOM | 尚无 GPU runtime 证据 |

旧文件、日志和权重保留用于溯源。`scripts/train_ea_rsig_v3.sh` 默认拒绝启动，只有显式 `ALLOW_LEGACY_EA_V3=1` 才能运行。不要为“先跑起来”解除该保护，把旧 native14 结果混入 Lite 结果。

## 16. 当前实现思路及必须守住的合同

### 16.1 模型路径

1. 当前三相机 RGB 走官方预处理、L-shaped canvas、VAE；**EA 只读 clean current-frame latent**，不读 future/noisy video、future actions 或 labels。
2. 使用原 Video patch embedding，将当前空间网格分为 head / left / right 三视图，各保留14×14网格，投影到512D。指令经 masked text pooling 做轻量通道调制；没有对视觉特征先全局平均池化。
3. 复用 `EARSIGV3Module` 的角色感知、typed relation 与 plan：Ground×4 + Relation×4 + Progress×1 + Plan×6，共15 tokens。legacy video 分支冻结且不执行。
4. 实际 Action hidden 生成 query，对 EA tokens 做独立 cross-attention；零初始化输出投影形成 residual，**只在第一个 mixed block 后注入一次**。EA tokens 不追加到主联合序列，不改 joint attention、RoPE、mask，也不重置 Action／proprio head。
5. 原 video/action flow-matching loss 保持，辅助 Ground/Relation/Plan 名义系数分别 .025/.01/.05。没有迁入 Video Residual、第二次 future forward、domain consistency 或 future relation transition。

相关文件为 `openwam/model/modules/ea_rsig_lite.py` 和 `openwam/model/architectures/dual_system/ea_rsig_lite.py`。官方 `dual_system/joint_self_attn.py`、backbones 和 baseline model YAML 本次未修改。新增注册、可选 EA 数据透传和指标不应改变 A 的模型数学路径。

### 16.2 数据、旋转、TCP 和控制边界

原始数据来自 `robotwin2_ea_rsig_v3_clean50_v6_raw_20260921`：50任务、2500 episodes、557651源帧。新 reader 的有效起点为555151（每条轨迹最后一帧不作为 t+1 起点）。原 HDF5 只读；新 sidecars、索引和统计写到 `outputs/ea_lite/data/`，约179MiB，不是把旧40-step标签直接改名。

- **Action**：从现成 raw endpose 读取绝对 EEF t+1…t+32，无需再从 qpos 估计 FK。20D 有效数据映射到80D的0–9、34–43槽，其余用官方 mask。夹爪走官方 min-max／反归一化，不沿用旧 native14 binary patch。
- **旋转**：本次采集器的 endpose 是 **wxyz**；官方转换函数接收 **xyzw**。Lite reader/client 在边界重排，模型保持官方列式 Rot6D。执行前按该版 RoboTwin controller 要求转回 wxyz。四元数模长正确不能证明顺序正确，必须核对旋转矩阵。
- **坐标系**：保留原 RoboTwin endpose 公共参考系中的 absolute targets，不能直接称为“每只手臂自身 base frame”，更不能偷偷改成 delta／chunk-relative。真机需明确 workspace/base 刚性变换。
- **EA 当前 TCP**：raw EE endpose 与 oracle TCP 有局部 X 方向 **0.12m** 刚性偏移。输入由当前观测 `EE + 0.12 * R[:,0]` 推导；全数据准备时核对 oracle 仅用于验收，不能把 oracle pose 当推理输入。公共公式在 `openwam/ea_lite_contract.py`。
- **Plan**：从 raw v6 pose/role ID/valid 重新计算8/16/32 horizon，使用当前角色局部坐标，moving 阈值1mm；尾段、无效角色和身份变化均 mask，不能从旧40步标签插值。
- **Ground/Relation**：每个 view 用自身图像尺寸归一化 bbox；无可见框／虚拟 goal 不伪造框。保留原 near/contact、0.5m proximity 定义及特殊任务 goal-valid 规则。130D 只复用监督布局，始终是独立 `ea_targets`，不进入80D proprio 或 generate。
- **归一化**：只用当前 Clean 训练数据重算统一 EEF stats，Rot6D维度保持 min=-1/max=1 的恒等缩放。A/B/C 共用同一统计、图像和动作合同；当前三组 color jitter 均关闭。

角色映射快照在 `assets/ea_robotwin/role_registry_lite.json`；小型 FK 资产和源说明也保留。后续不要用模块名相同来推断语义一致，要检查 role ID、左右臂、时间索引、绝对位置和旋转的物理含义。

### 16.3 训练阶段与对照解释

| 开关 | 实际行为 | 解释限制 |
|---|---|---|
| `EA_EXPERIMENT=A` | 官方原架构，无 EA auxiliary，自动 joint | 用它先验收 Foundation + EEF 的资源容量 |
| `EA_EXPERIMENT=B` | 有 EA losses，Action condition 关闭，无用 condition 参数冻结 | 只有共享 patch embedding 可训练时，才能说 auxiliary 监督影响 backbone |
| `EA_EXPERIMENT=C` | 有 auxiliary，也有 Action condition | zero-out 初始化不改变输出；训练后可改变 Action |
| `EA_STAGE=warmup` | 原主干冻结，训练 Lite | C约1369万 trainable；总模型仍约124.2亿参数，冻结不等于权重/激活显存消失 |
| `EA_STAGE=joint` | 按官方冻结表保留冻结 T5/VAE，其余 Action/Video + EA 更新 | 需要独立做完整多步容量测试；不能从 warmup 通过外推 |
| `EA_STAGE=action_only` | 冻结 Video，仅 Action 和 EA 更新 | 是部分微调方案，不是等价全参复现 |

joint 的初始配置为 backbone LR=1e-5、EA LR=1e-4。A/B/C 必须控制 Foundation、有效 batch、样本曝光、增强和评测；额外 warmup 需计入预算。不要拿已经 RoboTwin-C2R SFT 的官方权重作为 Clean-only 对照初始值。

## 17. 实测方案、失败原因和已经学到的经验

### 17.1 历史双卡实测（V3，不是 Lite）

当时为2×A100-80GB、BF16、ZeRO2、activation checkpointing、约60.49亿 trainable、无 optimizer offload。请同时看日志与较新的连续复验，不只看最早的“首步通过”。

| 方案 | 实际结果 | 正确结论 |
|---|---|---|
| 默认 AdamW foreach | 更新阶段申请11.27GiB OOM | forward/backward 执行不等于 optimizer 更新完成 |
| 原生 fused AdamW，bs1，一步 smoke | 一次真实更新／保存／strict reload／旧40×14离线生成通过 | 只证明一次工程链路，不证明可持续训练 |
| fused、单大组，bs2 | 首次更新约15.86–15.89s，第二次11.27GiB flat-gradient FP32转换 OOM | 不能把首步时间当稳态 |
| 同配置 bs1 | 首次约14.88–14.98s，第二次仍11.27GiB OOM | 继续反复试 bs1 不能解决固定状态内存 |
| 等 LR 的 action/video/base 三组 | bs1第二次仍申请9.31GiB OOM，allocated约67.4GiB | 粗分组不足，尚未形成稳定 batch |
| 每组上限200M元素 + expandable_segments | 构造及参数覆盖检查完成，后来卡被其他任务占用，未完成更新验收 | **没有稳定结果**，不能报告“已解决”或“最大bs” |

Adam 的 exp_avg/exp_avg_sq 常在首次 step 才建立，所以第二步可能比第一步多出大量常驻状态。旧 OOM 与 batch 从2降到1无关的主要部分是状态分片、flat gradient 转换和临时量；梯度累积／CPU初始化不消除它们。ZeRO2 不把模型权重全部分片到所有卡，不能把2×80GB看成单张160GB。

旧每卡约23.16GiB全模型 BF16权重，加双卡分片 FP32 master/Adam/grad/转换临时量的粗预算已接近79.5GiB，尚未含激活和通信；这是旧参数规模的解释，不是 Lite 精确显存预测。fused AdamW 解决一处 sqrt 大临时量，不会消除 Adam 常驻状态。

旧有界 sweep 曾 SIGSTOP；**2026-10-14已终止自有 launcher/workers，日志保留**。`outputs/batch_benchmark/summary.json` 里 suspended/PID字段是旧快照，禁止照抄旧PID继续／kill，也不能把后来他人显存占用倒推为早期 OOM 原因。当前不能给出稳定 bs、稳态 step-time 或可信训练天数。

### 17.2 当前 Lite 已通过什么

| 证据文件（`outputs/ea_lite/readiness/`） | 已验证范围 |
|---|---|
| `foundation_cpu.json` / `.log` | **实际完整 Foundation CPU 加载**，2089 tensors、85新EA tensors、80D policy heads零重置、无meta；总参数12420900941，C warmup trainable13687328 |
| `preflight.json` | 2500源文件 size/mtime、全部sidecar SHA、stats SHA、Foundation记录及80D header |
| `data_audit.json` | 全50 reader库存，首/中/末3处真实RGB窗口，32×80/9帧、inverse normalization、当前TCP train/client一致、A/C相同输入 |
| `sidecar_audit.json` | 全2500 sidecars有限值、双臂常量、50任务raw v6 collection contract |
| `camera_bbox_grid.png` | 人工查看三相机canvas/grid/GT框映射；不是 trained grounding score |
| `cpu_tests_acceptance.log` | **62 passed**：T1 FP32/BF16、checkpoint开关，T2独立未来扰动，T3当前条件一致/敏感性，两步梯度，joint双流梯度，eval inference-mode，两次真实CPU Accelerate保存恢复，旋转/解码边界 |
| `config_A/B/C.yaml` | CHECK_ONLY实际Hydra resolve通过；旧B/C快照为warmup，正式joint需重新dump |
| `readiness_summary.json` | 上次完成时的文件hash与状态快照；后续本文追加和迁移脚本不在旧hash快照中 |

新增迁移支持另有 **13项通过**：路径重定位、源绑定保护、2/4/8卡及显式GPU型号选择、CHECK_ONLY／torchrun参数路由、40 profile offload、环境锁文件移除旧editable路径。GPU选择和启动测试均用 mock／stub，无实际GPU分配；它们不能验收 NCCL、offload 编译或模型容量。

### 17.3 排错经验：后续必须避免重复消耗

- 先看进程、日志最后阶段、CPU/RAM和磁盘I/O，区分“Python首次导入／24.8GB权重加载／分布式等待／真实更新”。本机共享文件系统冷导入曾持续数分钟，最初CPU回归约522s；观察超时不意味着进程死掉，不要重复启动多个完整加载进程。
- 先做一档 bs1 的**连续完整更新**，跨过 Adam state 初始化，并覆盖保存恢复；通过后才增加 bs。不要为了查一个参数反复重新加载全量模型。
- OOM记录出错操作、申请量、allocated/reserved/free、rank、bs、trainable参数、optimizer group和offload，区分固定状态、激活、临时flat和碎片。expandable_segments只改变分配器，不是显存容量保证。
- 不改 SGD、不换小模型、不重置80D head来冒充 full Foundation AdamW 验收；微调集合或动作合同变化必须单独命名。
- 零初始化时第一步主要是 adapter out 得到动作梯度，上游 query/attention 的动作梯度可以为零；第二步后应连通。aux heads仍可由辅助loss更新。只测第一步会误判断梯度。
- 不要在 `torch.inference_mode()` 下无条件 checkpoint；当前tiny eval检查通过，完整BF16/ZeRO仍需实测。
- 保存权重不等于可精确恢复。必须有 optimizer/scheduler/RNG／Accelerate状态；tokenizer/config/stats也需一并保存。旧测试曾因漏relay tokenizer导致strict重载失败，正式checkpoint链路已经补齐。
- 不操作其他用户GPU进程。清理自有失败进程前核对当前PID/命令/PGID，退出后查孤儿workers；旧日志PID不是当前作业身份。

## 18. 搬运清单与已经放好的脚本

### 18.1 脚本索引

| 文件 | 用途 |
|---|---|
| `scripts/install_ea_environment.sh` | 新机重建 `.venv`；默认python3.10/uv，支持PYTHON_BIN/UV_BIN覆盖，锁定cu128和依赖，移除lock中旧绝对editable路径 |
| `scripts/relocate_ea_lite_manifest.py` | 迁移后只重写manifest路径，保留源size/mtime和标签／stats hash；不改raw/标签内容，不覆盖已有输出 |
| `scripts/preflight_ea_lite.py` | stdlib只读数据／权重门槛；可选 --num-gpus/--gpu-name，预检前后各查显存 |
| `scripts/train_ea_lite_4gpu.sh` | 默认4卡、默认CHECK_ONLY=1；共享A/B/C配置，支持EA_NUM_GPUS供多卡入口复用 |
| `scripts/train_ea_lite_multigpu.sh` | 单机多卡薄入口，必须显式EA_NUM_GPUS，与四卡共用同一逻辑 |
| `scripts/debug_ea_lite.py` | 复用官方trainer，诊断JSONL、保存后计划退出，用于两次恢复验证 |
| `scripts/check_ea_lite_foundation.py` | CPU完整Foundation加载检查，要求CUDA_VISIBLE_DEVICES为空 |
| `scripts/audit_ea_lite_data.py` | 真数据窗口、物理解码与相机bbox预览 |
| `scripts/prepare_ea_lite_data.py` | 无sidecars或源数据发生变化时，从raw v6重新准备；不要用旧14D数据清单 |
| `scripts/smoke_ea_lite_deploy.py` | 后续训练产物strict reload + 当前观测离线32×20 chunk，统计延迟；不发控制命令 |
| `scripts/package_ea_lite_handoff.py` | 在本机Git checkout打包实际工作区、小证据、Lite sidecars/stats，生成逐文件SHA清单 |

配置必须一起搬：`configs/model/dual_system_ea_rsig_lite.yaml`、`configs/dataloader/ea_rsig_lite.yaml`、`configs/train_ea_lite_4gpu.yaml`、`configs/deploy_ea_rsig_lite.yaml`、`benchmarks/robotwin/policy_config_ea_rsig_lite.yml`。整个 `openwam/`、`scripts/`、`tests/`、`assets/ea_robotwin/`、`AGENTS.md`、许可证和修改日志也必须保留；不能只拷贝新adapter文件。

### 18.2 三类资产分开搬

1. **交接包**：`outputs/ea_lite/handoff/OpenWAM-ea-lite-handoff-20261014.tar.gz`，相邻 `.tar.gz.json` 为包SHA256与逐文件SHA。包内顶层为 `OpenWAM/`，包含当前工作区文件、Lite sidecars/stats、CPU证据和旧OOM小日志。**不含 `.git`、`.venv`、Foundation权重、raw HDF5、大训练产物、Cosmos gitlink内容**；本包针对Alpha/Wan，不用于复现其他backbone。
2. **完整Foundation目录**：原本的 `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model/`，包括主safetensors、config、tokenizer、上游inventory和 `download_verification.json`，不能只搬单个权重文件。支持迁移到任意位置后设置EA_FOUNDATION_DIR。
3. **完整raw数据目录**：`robotwin2_ea_rsig_v3_clean50_v6_raw_20260921/`，包含各任务HDF5、指令文件与 `ea_rsig_v3_collection_manifest.json` 等元数据。不能只拷贝video或label sidecars。当前统计只使用Clean，Randomized评测环境需另行准备。

RoboTwin模拟器及其完整资产／依赖不在交接包内；训练先依赖已有HDF5，真实闭环评测需按官方benchmark说明单独准备环境，并核对该版controller的EEF/四元数合同。checkpoint/full-state写入量远大于小标签，目标输出盘也要留足空间；可在启动命令末尾加 `training.output_path=/local_fast_disk/ea_lite/debug_A_joint`，不要因共享盘I/O慢不断重启。

复制raw时用 `rsync -a` 保留mtime；若打tar，使用PAX格式保存纳秒mtime。源清单强校验size/mtime，普通复制改变时间会失败；不要为了过检直接刷新绑定。若原件无法证明一致，应重新审计／生成。交接包本身使用PAX。

重新生成交接包时，在完整Git工作区运行如下命令，输出用新文件名，脚本不覆盖已有包：

```bash
python scripts/package_ea_lite_handoff.py --output outputs/ea_lite/handoff/OpenWAM-ea-lite-handoff-NEW.tar.gz
```

不要用 `git archive HEAD` 代替，它漏掉未提交的 EA 文件；也不要只依赖gitignore同步，`outputs/ea_lite/data` 和Foundation恰好被忽略。

## 19. 新服务器恢复环境与路径：先CPU，后GPU

以下路径为新机示例，请替换为真实挂载位置。只修改环境变量和新manifest，不对整个仓库全局替换旧路径。

### 19.1 解包与资产完整性

先对比源 `.tar.gz.json` 的 `sha256` 与新机 `sha256sum`，再解包到希望存放项目的父目录；避免覆盖已有同名工作区。

```bash
tar -xzf OpenWAM-ea-lite-handoff-20261014.tar.gz -C /data/projects
cd /data/projects/OpenWAM
export EA_FOUNDATION_DIR=/data/models/OpenWAM-Alpha-Pretrain-Foundation-Model
export EA_RAW_ROOT=/data/datasets/robotwin2_ea_rsig_v3_clean50_v6_raw_20260921
```

Foundation revision应为 `52df4e66c82c5c8b480adcc8d01f4db7415dfb56`；主权重 `checkpoint_step_154000.safetensors` 为24813767464 bytes，SHA256为 `180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d`。新机至少核对主文件SHA，并按download_verification记录核对配套资产。preflight查记录／大小／header，不会重算24.8GB主文件SHA，不能把它当传输完整性校验替代品。

```bash
sha256sum "$EA_FOUNDATION_DIR/checkpoint_step_154000.safetensors"
```

### 19.2 重建环境

本机版本快照：Python3.10.20、torch2.7.1+cu128、DeepSpeed0.18.9、Accelerate1.15.0、Transformers5.19.0，其余见 `requirements-ea-lock.txt`。不要复制旧 `.venv` 作为新环境：其中脚本shebang和editable install含旧路径，CUDA扩展也与新机工具链有关。

```bash
export PYTHON_BIN=/absolute/path/to/python3.10
export UV_BIN=/absolute/path/to/uv
bash scripts/install_ea_environment.sh
export PATH="$PWD/.venv/bin:$PATH"
python -m pip check
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
```

安装器先装固定PyTorch cu128三件套，再使用过滤旧editable路径后的锁文件和当前 `-e .`。新机需要这些包的可访问安装源／缓存；这个脚本的路由已测，**没有在目标服务器实际安装验证**。核对driver对cu128的支持，offload还需本机C++/CUDA扩展工具链及CPU RAM，`DS_BUILD_OPS=0`不保证后续CPUAdam无需JIT。不要混入另一项目的cu124 NVIDIA pins，或为消除一个import错误随意升级整个栈。

HF资产本已齐全，训练离线；确需重下时遵循AGENTS：先hf-mirror，镜像返回获准CDN强制直连，禁止代理/VPN和静默官方回退。不要从日志打印token或代理凭据。

### 19.3 重定位manifest并CPU预检

```bash
python scripts/relocate_ea_lite_manifest.py \
  --manifest outputs/ea_lite/data/manifest.json \
  --raw-root "$EA_RAW_ROOT" --data-dir "$PWD/outputs/ea_lite/data" \
  --output outputs/ea_lite/data/manifest.relocated.json
export EA_LITE_MANIFEST="$PWD/outputs/ea_lite/data/manifest.relocated.json"
python scripts/preflight_ea_lite.py \
  --manifest "$EA_LITE_MANIFEST" --foundation "$EA_FOUNDATION_DIR"
```

重定位输出若已存在，先检查是否就是本次迁移清单，确认后复用或换新输出名；脚本不会覆盖源清单。它只改路径、核对复制源size/mtime和文件存在性；真正label/stats内容hash由紧接着的preflight验证。

可在此做新机CPU回归和真实窗口检查（可能首次import较慢，保留一个进程观察日志，不重复启动）：

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_ea_lite_preflight.py tests/test_ea_lite_handoff.py
CUDA_VISIBLE_DEVICES='' python scripts/audit_ea_lite_data.py \
  --manifest-path "$EA_LITE_MANIFEST" --output-dir outputs/ea_lite/new_server_audit
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_ea_rsig_lite.py tests/test_ea_lite_metrics.py tests/test_ea_lite_deploy_config.py \
  tests/test_ea_rsig_v3_integration.py tests/test_robotwin_interface_proprio.py \
  tests/test_finetune_config_override.py tests/test_ea_deploy_contract.py tests/test_ea_robotwin_client.py
```

如果要在新机再次确认完整权重构建，可运行 `CUDA_VISIBLE_DEVICES='' python scripts/check_ea_lite_foundation.py --foundation "$EA_FOUNDATION_DIR" --report outputs/ea_lite/new_server_audit/foundation_cpu.json`；它会真加载整份模型，需要CPU内存和时间，**不是每次启动前都要重跑**。如果没有搬运sidecars，使用 `prepare_ea_lite_data.py --raw-root "$EA_RAW_ROOT" --output-dir outputs/ea_lite/data_new` 从原始数据生成并将EA_LITE_MANIFEST指向新清单，不要覆盖历史已验收资产。

### 19.4 先dump实际配置

```bash
CHECK_ONLY=1 EA_EXPERIMENT=A bash scripts/train_ea_lite_4gpu.sh
CHECK_ONLY=1 EA_EXPERIMENT=B EA_STAGE=joint bash scripts/train_ea_lite_4gpu.sh
CHECK_ONLY=1 EA_EXPERIMENT=C EA_STAGE=warmup bash scripts/train_ea_lite_4gpu.sh
CHECK_ONLY=1 EA_NUM_GPUS=8 EA_EXPERIMENT=C EA_STAGE=joint \
  bash scripts/train_ea_lite_multigpu.sh training.gradient_accumulation_steps=4
```

CHECK_ONLY仅resolve配置，主动隐藏CUDA，不检查实际GPU、不分配模型、不训练。看到yaml输出不等于dry-run已通过。命令行参数放最后覆盖默认值，最终必须检查finetune路径、manifest、action/state80、33/9时域、stage/condition、offload、max/save_steps和LR。

## 20. 四卡实际调试：明确顺序、计数和保存恢复

**本节命令仅在目标服务器找到独占GPU之后执行；当前session没有执行。** 先记录 `nvidia-smi`、`nvidia-smi topo -m`、CPU RAM、磁盘剩余／IO、driver、包版本、选中物理卡号和配置dump。不是每次因等待都重启。

默认四卡A100-80GB profile：每卡总显存至少75000MiB、空闲至少70000MiB；40 profile为39000/35000MiB。门槛前后各查一次，但不预约资源，也不证明装得下。默认要求型号包含A100；`EA_GPU_NAME`可显式选择H100/H200等，profile表示内存门槛等级，不代表已测试该硬件。只接受物理数字GPU ID，不支持UUID/MIG选择；容器或调度器若使用其他映射，应先核对可见设备，不能乱改ID绕过检查。

### 20.1 G0先A：全链路20次更新，含两次恢复

下面0,1,2,3仅为例子，替换成真正分配到的卡。A自动joint，C/B由EA_STAGE选择。不要改动全局默认CUDA_VISIBLE_DEVICES去占用其他作业。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export EA_EXPERIMENT=A EA_GPU_PROFILE=80
EA_STOP_AFTER_OPT_STEPS=5 CHECK_ONLY=0 bash scripts/train_ea_lite_4gpu.sh
```

默认每卡bs1、accum8、world4，因此effective batch=32；`max_steps=160`指**每rank micro-steps**，是20个optimizer updates。`save_steps=40`是每5个updates存一次。计划停止在原trainer的weights/full-state保存完成、所有rank同步后发生，退出码0；不是kill进程模拟保存。

从日志取得真实带时间戳run目录后，以相同参数恢复：

```bash
export EA_RUN_DIR=/absolute/path/to/the/actual/run_dir
EA_STOP_AFTER_OPT_STEPS=10 CHECK_ONLY=0 bash scripts/train_ea_lite_4gpu.sh \
  training.finetune_ckpt_path=null training.resume_ckpt_path="$EA_RUN_DIR"
CHECK_ONLY=0 bash scripts/train_ea_lite_4gpu.sh \
  training.finetune_ckpt_path=null training.resume_ckpt_path="$EA_RUN_DIR"
```

三次均保持max_steps160、同stage、world、batch/accum、数据与LR horizon，分别到5、10、20 updates。不要把max_steps从40改成80冒充一致恢复；不要同时指定finetune和resume。配置启用 `save_full_states_for_resume=true`、`keep_final_resume_state=true`、keep_last_k=2；否则官方正常收尾会清掉resume状态，仅剩权重。观察实际 `accel_state_step_*`、optimizer/scheduler/RNG与下一步进度，不能仅凭“目录存在”验收。

### 20.2 G1/G2再C：warmup通过后独立测joint

A通过后，新run改为 `EA_EXPERIMENT=C EA_STAGE=warmup`，复用上述5→10→20协议。检查Ground/Relation/Plan有效mask和loss、adapter out更新、第二步以后的condition梯度、禁用条件的数值变化。warmup不是正式SFT，不要求20步就学到可执行任务。

再测joint或action_only时，建立新的run，使用warmup权重作finetune初始化，并保持完整checkpoint目录可用；不要恢复一个不同trainable set的旧optimizer。示例：

```bash
EA_EXPERIMENT=C EA_STAGE=joint EA_STOP_AFTER_OPT_STEPS=5 CHECK_ONLY=0 \
  bash scripts/train_ea_lite_4gpu.sh training.finetune_ckpt_path="$EA_RUN_DIR"
```

此处EA_RUN_DIR应为C warmup产物，不能误用前面的A。新的joint还须单独完成连续更新和两次恢复、容量和吞吐验收。warmup与joint所需显存不同，前者通过不意味着后者通过。B的正式aux-only解释以joint共享梯度为准。

### 20.3 容量不够时按证据降级

先保持bs1/checkpoint/ZeRO2，确认不是其他作业占卡、碎片或错误重复进程。80GB可显式 `EA_OFFLOAD=cpu`；40GB用 `EA_GPU_PROFILE=40` 强制CPU optimizer offload。CPUAdam编译、RAM和PCIe代价都要记录，首轮JIT不计稳态速度。当前未在GPU上验证offload，报错时核对实际optimizer/DeepSpeed实现，不宣称“切CPU必定能跑”。

如果joint仍不成立，再选择action_only或EA-only并明确实验名、trainable参数和效果边界，或申请8卡；不要隐式改变研究合同。bs只有在完整状态初始化后连续更新稳定，才按1→2→4逐档增加；OOM需干净退出受损分布式进程后再试，不能直接在报错NCCL上下文中继续。

## 21. 单机更多GPU、预算换算与多机边界

### 21.1 单机2/4/8卡共用入口

`EA_NUM_GPUS`决定单机进程数，`CUDA_VISIBLE_DEVICES`必须给同样数量的不同物理ID。不要只设置NPROC_PER_NODE：这里由EA_NUM_GPUS派生，四卡脚本默认4。多卡入口强制显式卡数并始终 `--nnodes=1`。

保持与四卡debug相同global32的8卡例子：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EA_NUM_GPUS=8 EA_EXPERIMENT=A EA_GPU_PROFILE=80
# 若实际为H200，显式 export EA_GPU_NAME=H200；默认A100。
EA_STOP_AFTER_OPT_STEPS=5 CHECK_ONLY=0 bash scripts/train_ea_lite_multigpu.sh \
  training.gradient_accumulation_steps=4 training.max_steps=80 training.save_steps=20
```

同一run后续恢复仍保持80/20/accum4及8卡；两次恢复STOP分别5、10，最后不设STOP。从四卡改八卡是**新的容量/训练配置**，不要默认旧ZeRO状态支持world-size变化的精确恢复。

计数公式：`effective_batch = world_size × batch_per_gpu × accumulation`；`optimizer_steps ≈ micro_steps / accumulation`（还需考虑epoch尾部边界）。如需global256，4卡bs1为accum64，8卡bs1为accum32；每卡bs2时相应减半。保持global batch不代表速度相同，也不自动保证不同训练预算公平。

论文RoboTwin C2R为global256、5epochs、10740 updates；Full为118655 updates。预训练128×H200、global3072不等于微调必须8×H200；仓库8卡启动示例也不证明所有论文SFT均使用8×H200。当前自有Clean窗口数不同，正式epoch/sample预算应按实际reader、sampler/drop_last计算。20 updates debug不能直接用于论文结论；旧9.76s首步不能线性换算成当前Lite完整实验排期。

### 21.2 多机不是本次已准备验收的范围

新增Lite launcher只负责单机多卡，不能把NNODES=2传进去就当多机支持已通过。官方 `scripts/train.sh` 有NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT入口，但多机还需要共同资产路径、网络/NCCL、rank拓扑和全局预算核验。若转多机，先在目标集群做单机验收，再按官方入口单独设计调试；需要同等接入Lite config与诊断，并实际测跨节点保存恢复。不要拿单机mock或四卡CPU配置证明多机可用。

## 22. 吞吐记录、部署与完成标准

### 22.1 稳态速度应如何报告

每rank的 `diagnostics_rank*.jsonl` 有micro_step、optimizer_step、每卡bs/world/accum、seconds_micro、loss/mask/gate、selected gradient norms、peak allocated/reserved。排除首次导入/模型加载、JIT、首个更新与checkpoint保存边界，至少观察后续多次完整更新；需要时在资源验收后延长到更多updates。

分别报告micro-step秒数、完整累积更新秒数、samples/s、每rank峰值及全体最大值、offload/保存耗时；多卡时间以慢rank与实际wall-time为准。吞吐不能只取某卡最好一步，峰值不能只取清零后某个micro。正式排期用样本曝光量/samples per second，再单列保存、评测和重启，不用论文“更新次数”乘当前global32的更新时间冒充global256预算。

### 22.2 Lite完整checkpoint部署门槛

```bash
# EA_RUN_DIR必须是已训练Lite产物目录，不是Foundation或历史V3。
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_ea_lite_deploy.py \
  "$EA_RUN_DIR" --manifest-path "$EA_LITE_MANIFEST" --device cuda:0 \
  --report outputs/ea_lite/new_server_audit/deploy_runtime.json
CUDA_VISIBLE_DEVICES=0 bash scripts/deploy.sh "$EA_RUN_DIR" \
  --config configs/deploy_ea_rsig_lite.yaml
```

离线脚本只用当前RGB/指令/EEF推导TCP，不读oracle，检查strict reload、32×20有限物理EEF并计时；2 denoise steps只为短测。正式延迟和SR使用固定真实denoise预算。服务端直接 `OmegaConf.load`，所以Lite deploy YAML是独立完整配置，不能只写Hydra defaults；Action窗口键是 `num_frames: 33`，video_num_frames9，horizon32，CFG1、cache/compile关闭。

RoboTwin客户端用 `benchmarks/robotwin/policy_config_ea_rsig_lite.yml`，state20/action_type ee/source_quaternion_order wxyz。离线32×20解码和fake-controller旋转往返不证明真实IK跟踪；下一步需环境内确认绝对EEF、左右臂、夹爪与控制频率，再做Clean/Randomized配对闭环。

### 22.3 下一Session必须留下的结果

至少记录：目标服务器硬件／环境、Foundation revision和SHA、实际资产路径与数据manifest hash、A/B/C与stage、world/bs/accum/有效batch、trainable参数、真实连续更新数、两次恢复的前后步数/LR/loss、稳态时间／峰值显存、完整Lite reload/chunk结果、失败日志与是否有遗留进程。更新本文件末尾的实测记录和 `docs/思考与隐患.md`，不要覆盖本次CPU历史证据。

停止条件是资源验收不能持续、速度预算不可接受或因果/动作合同失败；继续条件是完整优化器多步、保存恢复、数值/物理接口与可承受吞吐均通过。之后才进行匹配预算SFT和方法评估。**当前仍无四卡／offload稳定性、最大bs、完整Lite推理、闭环SR、效果等价或sim2real证据。**

可直接给下一Session的任务说明：

> 先读AGENTS.md、docs/思考与隐患.md和本文第15–22节。核对迁移包/完整Foundation/raw v6数据，重建锁定环境，重定位manifest并CPU预检。取得独占卡后，从A baseline开始做单机四卡或显式EA_NUM_GPUS多卡20-update验收，按5→10→20覆盖两次full-state恢复；随后测C warmup，再独立测joint。保留80D绝对EEF与当前观测因果合同，不运行旧V3 sweep，不重复下载已验收资产，不把首步成功当稳定训练。未有稳态数据前不要给训练排期；所有新结果及更正写回本文件和隐患文档。

## 23. 启动加速更新：复用本地数据，离线准备元数据，避免反复全量校验

2026-10-14 UTC，用户进一步要求缩短启动，并指出已经处理好的数据无需每次重做全量校验。**本节覆盖第18–21节中启动默认清单和校验模式的旧说明。**

### 23.1 原先启动时实际做了什么

原先Lite入口没有重新生成Plan标签，也没有重新拟合normalization stats；这两部分已经离线完成。重复开销来自：启动器全量检查2500个源文件并hash全部sidecars；随后每个rank的reader又glob并打开全部HDF5探测长度、读入指令/scene信息、枚举窗口。这是重新发现和验证已有数据，不是重新生成训练标签。在共享文件系统和多rank并发下仍会明显拖慢启动。

**无需每次完整重验相同、已验收且不变的数据。** 全量审计适合首次接入、迁移后或更换数据版本；之后日常调试使用准备好的清单即可。快速模式有意不证明全部未访问文件完好，坏文件可能在后续首次读取时报错，不能将快速预检记录当作全数据完整性证据。

### 23.2 本次改动

- 新增 `scripts/cache_ea_lite_startup.py`：复用已验收的episode长度、标签和统计，离线缓存每任务的指令、scene_info和原图尺寸，不重写raw/labels/stats，不导入Torch或加载模型。
- 已在本机生成 `outputs/ea_lite/data/manifest.ready.json`（约2.8MiB），包含50任务的启动元数据；原 `manifest.json` 保留。四卡与多卡入口默认改用ready清单，配置为 `dataloader.startup_mode=prepared`。
- 官方 `RoboTwinDataset` 新增可选 `prepared_metadata` 参数，默认None仍走原路径；Lite传入准备好的元数据，启动不glob、不打开HDF5、不逐文件读取指令。样本的RGB/Action/normalization/mask/prompt计算不改，窗口次序仍按官方字典序与t+1规则在内存枚举。该文件在此前版本未修改的历史说明不再代表本次文件状态。
- 启动器默认 `EA_DATA_CHECK=fast`：只读manifest、小型stats并校验其hash、Foundation记录/header及GPU可用性，不stat/hash全部2500个episodes/sidecars。缺失prepared元数据会在加载大模型前明确报错，**不在训练启动时自动补建**。
- prepared dataset构造跳过全量raw扫描；每个worker首次实际访问某轨迹时检查源size/mtime，实际加载sidecar时保留SHA检查。stats检查和输入合同保留。未访问的损坏文件可稍后才被发现，这是有意的检查时机变化。
- 仍提供 `--data-check full` 和 `dataloader.startup_mode=scan` 诊断路径。standalone preflight默认仍是full，避免把手动“审计”静默降级成快速检查。

指令/scene_info是数据版本的一部分：修改它们后应离线更新ready清单。不要在启动时自动混用缓存指令和新raw版本。原始数据仍需搬运，因为训练batch要读取RGB和动作；本轮没有将全部RGB展开成tensor，也没有加入VAE/T5缓存。

### 23.3 现有本地数据怎么启动

本机ready清单已经生成，无需再执行离线缓存命令。直接使用现有启动入口：

```bash
# 如果shell残留了旧manifest路径，先指向新的ready清单。
export EA_LITE_MANIFEST="$PWD/outputs/ea_lite/data/manifest.ready.json"
CHECK_ONLY=1 EA_EXPERIMENT=A bash scripts/train_ea_lite_4gpu.sh
# GPU到位后才加CUDA_VISIBLE_DEVICES和CHECK_ONLY=0；仍按第20节的两次恢复协议。
```

如果更换了指令／清单，需要离线生成新版本（输出已存在会报错，换新文件名即可）：

```bash
python scripts/cache_ea_lite_startup.py \
  --manifest outputs/ea_lite/data/manifest.json \
  --output outputs/ea_lite/data/manifest.ready.NEW.json
export EA_LITE_MANIFEST="$PWD/outputs/ea_lite/data/manifest.ready.NEW.json"
```

想独立做一次全量审计，不需要启动训练：

```bash
python scripts/preflight_ea_lite.py --manifest "$EA_LITE_MANIFEST" \
  --foundation "$EA_FOUNDATION_DIR" --data-check full
```

如需恢复旧扫描路径用于排错，在实际启动参数末尾加 `dataloader.startup_mode=scan`，并设 `EA_DATA_CHECK=full`。普通重复调试无需这样做。

### 23.4 新服务器怎么复用

优先搬运ready清单，重定位时将第19.3节的输入换成 `manifest.ready.json`，不必再次构造标签或缓存指令：

```bash
python scripts/relocate_ea_lite_manifest.py \
  --manifest outputs/ea_lite/data/manifest.ready.json \
  --raw-root "$EA_RAW_ROOT" --data-dir "$PWD/outputs/ea_lite/data" \
  --output outputs/ea_lite/data/manifest.ready.relocated.json
export EA_LITE_MANIFEST="$PWD/outputs/ea_lite/data/manifest.ready.relocated.json"
# 迁移后推荐完整审计一次；之后启动器默认fast，不重复全量扫描。
python scripts/preflight_ea_lite.py --manifest "$EA_LITE_MANIFEST" \
  --foundation "$EA_FOUNDATION_DIR" --data-check full
```

若只搬了旧manifest，也可先重定位旧清单，再对它离线执行cache脚本。不要重新跑标签生成器来解决单纯的路径或启动元数据缺失问题。

本次更新后的完整交接包为 `outputs/ea_lite/handoff/OpenWAM-ea-lite-handoff-20261014-startup-fast.tar.gz`，包含ready清单和新代码；旧97MiB交接包保留为历史版本，默认启动行为以本节及新包为准。

### 23.5 性能证据与边界

本机fast preflight实际约 **0.343秒**，见 `outputs/ea_lite/readiness/startup_fast_preflight.json`，其中显式记录 `full_dataset_integrity_audited=false`。这是CPU预检时间，不包括Python冷导入、模型权重读取、DeepSpeed/NCCL或首次训练batch。

`startup_tests.log`：**19 passed**，包括真实tiny HDF5的prepared/scan逐字段样本对比、首/中/尾窗口、prompt/scene/mask，以及启动不打开HDF5、不扫描目录、损坏文件按需拒绝和启动路由。首次冷导入计入后共553.40秒，不是样本构造耗时。`startup_reader_regression.log`：官方reader回归 **32 passed in 3.70s**，1个模型集成项未重跑；本轮不涉及模型计算修改。ready清单去掉新增startup_metadata后与原manifest完整相等，原labels/stats/源绑定均未修改。

修改已完成，日常启动默认prepared+fast。没有GPU训练或完整启动时间的新结论；模型冷导入、权重加载与分布式初始化仍可能耗时，四卡验收待目标服务器实测。

### 23.6 后续四卡实测与日常开关（2026-10-14 UTC）

第23.5节“无GPU结论”为实现时状态；之后已在GPU0–3（各有21155MiB外部作业占用）完成A bs1、Lite C joint bs1/bs2，各40micro/5次AdamW更新、exit0。C joint约50.0GiB allocated / 52.8GiB reserved；仅A保存了更新权重及full-state，未验收ZeRO恢复或C部署。

快速启动复验：fast预检0.264s（随后热读0.115s），full44.969s；startup/preflight检查13 passed，缓存2500份指令及50task scene_info与当前源一致，ready原始绑定不变。C prepared与C scan同Foundation/bs1/world4/accum8/seed42/5update horizon/原损失系数/save0对照，排除首次更新后平均micro为2.026s和1.965s，完整累积更新16.206s和15.723s，差约3%。共享卡每模式单次短测，不将该差值归因为初始化方式；未观察明显持续更新速度变化。首8micro全部loss相等、全程有效标签比例相等，后续总loss最大差约0.00029，梯度范数非bitwise一致。

日常保持默认prepared+fast。更换设备/数据版本后可离线完整审计一次，训练仍复用prepared缓存；仅排查旧reader行为时启用scan：

```bash
# 日常默认：不重复全量源扫描/sidecar hash，样本按需检查仍保留。
export EA_DATA_CHECK=fast
export EA_LITE_MANIFEST="$PWD/outputs/ea_lite/data/manifest.ready.json"

# 迁移/换数据版本后单独全检；此命令不启动模型或GPU训练。
.venv/bin/python scripts/preflight_ea_lite.py --manifest "$EA_LITE_MANIFEST" \
  --foundation "$EA_FOUNDATION_DIR" --data-check full

# 必须诊断旧路径时：EA_DATA_CHECK=full，并在launcher末尾加
# dataloader.startup_mode=scan；普通训练无需这样做。
```

CPU快检不包含冷依赖导入、NCCL/DeepSpeed与24.8GB权重读取。shared资源下本次直接调用同一torchrun入口，未降低launcher独占显存门槛，未操作其他人的作业。不能用0.26s快检时间声称完整启动也只需0.26s。

EA raw Ground起始约7.76与ln(3×14×14)+2ln2的未训练尺度相符；加权EA均值约0.275、Video+Action约0.253。可选joint试验将原EA整体乘0.5：复用三个现有权重，设置0.0125/0.005/0.025，不改默认架构或损失实现。等效配置已解析通过，尚无0.5实际训练或效果结论：

```bash
CHECK_ONLY=1 EA_EXPERIMENT=C EA_STAGE=joint bash scripts/train_ea_lite_4gpu.sh \
  model.architecture.ea_lite.ground_weight=0.0125 \
  model.architecture.ea_lite.relation_weight=0.005 \
  model.architecture.ea_lite.plan_weight=0.025
```

详细计数、失败记录、时钟边界、损失分解与下一步见`docs/EXPERIMENT_RESULTS_AND_ANALYSIS.md`；证据为`outputs/ea_lite/startup_comparison_20261014/`与`outputs/ea_lite/debug_20261014_bs1/runtime_verification.json`。长期稳定性、最大bs、ZeRO实际恢复、Lite部署/闭环与方法效果仍未通过；训练排期暂缓。
