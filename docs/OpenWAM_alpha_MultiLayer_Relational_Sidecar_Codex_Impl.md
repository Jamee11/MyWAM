# OpenWAM-α + Multi-Layer Relational Sidecar：Codex 实施说明

**目标**：以 `OpenWAM-Official/OpenWAM` 官方 `dual_system + joint_self_attn`、OpenWAM-α Foundation checkpoint 为基础，使用 4×A100（优先 80GB），**最小侵入**加入 3-stage Relational Sidecar。持续从浅/中/深层的**当前第一帧** Video hidden states 读取信息，更新任务关系 tokens，通过**外置 gated cross-attention residual**逐层写回 Action hidden states。保留预训练 Video/Action MoT Attention、Flow Matching heads、80D action contract。第一版**不修改联合 self-attention QKV、mask、RoPE，不插入 EA tokens 到 MoT 序列，不增加第二次 5B Video forward**。

> 注意：这份文档是拟实现的技术设计，不是 OpenWAM 官方已有功能。优先保证因果一致、可以完整反向、关闭模块时复现原版 checkpoint；不得承诺 SR 必然提升。

## 1. 精确的读/写位置与信息流

官方源码：
- `openwam/model/architectures/dual_system/joint_self_attn.py`：Architecture、`forward()`；
- `openwam/model/architectures/dual_system/mot_driver.py`：`DualSystemMoTDriver.run_joint_loop()`、`step()`；
- `openwam/model/video_backbone/wan_backbone.py`：Video block 的 `pre_attn_at_layer` / `post_attn_at_layer`；
- `openwam/model/action_backbone/separate_action_dit.py`：ActionDiT state 与 per-layer block；
- `openwam/model/architectures/utils/mask_modes.py`：注意力可见性；
- `configs/model/dual_system.yaml`：`attention_mask_mode: mutual`、`video_attention_mask_mode: first_frame_causal`。

**在 block index（从 0 开始）7、15、23 的原始 Video/Action 联合块完成后**，执行一次 sidecar：

```text
current RGB / instruction / current EEF state
    │
OpenWAM original 30-block joint Video DiT + Action DiT
    ├── after joint block 7:  READ protected first-frame H_V^7
    │                       → EA stage 1: role grounding / spatial anchor
    │                       → WRITE EA condition into H_A^7 (gated residual)
    ├── after joint block 15: READ protected first-frame H_V^15
    │                        → EA stage 2: relation refinement
    │                        → WRITE EA condition into H_A^15
    ├── after joint block 23: READ protected first-frame H_V^23
    │                        → EA stage 3: multi-horizon foresight
    │                        → WRITE EA condition into H_A^23
    └── joint blocks 24–29 → original action/video heads

EA auxiliary heads: grounding, near/contact/goal relation, future plan
train-only GT: boxes, validity, contact, relation, future directions/moving
```

每个 stage 读取 **Video first-frame**，而不是未来 Video tokens，也不是 noisy Action tokens。保留原有 `mutual` + `first_frame_causal` mask：默认实现里第一帧不能关注后续 Video，也不能关注 Action；未来 Video 与 Action 仍可以互相传递信息。**不要修改该注意力掩码**。需以测试验证当前 checkpoint/具体数据预处理下这一性质成立。

**Action Conditioning 的确切含义**：额外新增一个**Sidecar→Action Cross-Attention**，它产生残差并写回 `astate.payload.x_action`；不是改变原有 MoT shared attention 的 mask、QKV 拼接、RoPE 或增加主干 token 长度。新的 Action hidden state 会经过余下 OpenWAM MoT blocks，因而直接影响 Action head，也可通过后续原始 `mutual` attention 影响 future Video hidden states。Sidecar 不直接往第一帧 Video tokens 写任何东西。

## 2. Sidecar 内部网络结构

- 维护 `Z ∈ [B, 15, 512]`，来自 v3 的 **4 Ground + 4 Relation + 1 Interaction Summary（旧称 Progress）+ 6 Plan**，一次 `forward` 内跨三个 stage **持续更新**，不要每阶段重新初始化。
- `Z_0` = learned role/type tokens + 当前任务文本摘要 + 当前机器人 EEF/Gripper 状态 embedding；禁止读入 simulator privileged labels 作为输入。机器人状态维度应该从**当前 80D proprio 及 mask**解析，而不是从 Action 的 noisy target 取值；双臂与单臂均支持无效轨道 mask。
- 各 stage 从 Video `vstate.hidden_states` **按真实 runtime grid 计算 first-frame token 数**，取 `[:, :N_first, :]`。Wan 当前工程在此处使用 `(B, S_video, D_video)`；不要硬编码 `588`、`14×14` 或实际画布 patch 大小。三相机 bbox 必须从 L-shaped 384×320 canvas 正确映射到实际 grid。
- 对多层特征分别 `LN + Linear(D_video→512)`（每层独立投影）。先用 sidecar Query `Z` CrossAttn 到 first-frame Feature；再由 15-token SelfAttn、FFN 与 Text/EEF context 更新。额外读取采用**残差门控**：

  `U_k = CrossAttn(Q=Z_{k-1}, K=F_k, V=F_k)`  
  `Z_k = Z_{k-1} + α_k·U_k + Refine_k(Z_{k-1}, text, ee)`

  其中 `α_k` 初始化小值（例如 0.01）；深层如果无益应能学会趋近零，但不宣称自动实现 OOD 可靠性估计。
- 三个 stage 均更新**整套 15 tokens**，只是监督重心不同：stage1 ground、stage2 relation、stage3 plan；第一阶段已经形成初步关系/计划 token，因此从 block7 起的写回有任务信息。最终输出有对应 auxiliary heads，stage1/2 可加轻量 deep supervision。
- Ground 采用 task-conditioned spatial attention/role query，不单纯使用全局 pooled feature；用 GT bbox、role visible/valid 掩码监督。Virtual Goal 不可见时不能硬标可见 bbox。
- Relation 使用 EEF→Object、Object→Goal 两轨逻辑；沿用 near/contact/goal、track/label valid 的目标定义。Progress 暂时只称 `interaction_summary_token`：如果没有可审计的真实 progress/stage GT，**不得声称监督了准确任务进度百分比**。
- Plan 使用 `[8,16,32]` 未来步、双轨 EEF/Object direction + moving logits；适配原 EA 监督表的 mask；末尾不足 horizon 的 future-valid 必须 mask。方向约定与 base/local frame 明确固定。

### Action 写入模块（每个 stage 独立）

输入：`A_k ∈ [B, T_action, 1024]`（正常 T_action=32）、`Z_k∈[B,15,512]`。

```python
R_k = CrossAttention(query=LayerNorm(A_k), key=LayerNorm(Z_k), value=LayerNorm(Z_k))
A_next = A_k + beta_k * OutputProjection(R_k)
```

- 建议 `8 heads`，输出 1024D；`beta_k` **初始化为 0**，确保初始模型与原 Foundation 输出等价；sidecar 通过 auxiliary loss 可以立即学习，Action 写回权重通过 Action loss 逐渐学习。
- 不拼接额外 15 tokens 进原 Video/Action MoT token 序列。
- 不让 `Z_k` 读取 `A_k`（尤其 noisy future expert action）用于关系预测；但 `A_k` **作为 CrossAttn Query**读取 `Z_k` 是允许的，因为回传方向是 `Z→A`，不是 `A→Z`。
- 零初始化 write gate 刚启动时，Sidecar 的 action-gradient 可能暂时较小；必须记录 gate、projection gradient 和 EA token gradient，辅助监督保证 Sidecar 不会僵死。

## 3. Codex 应如何修改官方工程（最小修改路径）

1. **新建 Sidecar 模块**：建议 `openwam/model/ea_rsig/relational_sidecar.py`：`RelationalSidecar(nn.Module)`、三个 `SidecarStage`、`ActionWriteAdapter`、`EAHeads`。将其**注册到 Architecture nn.Module**，保证 state_dict、optimizer、保存和加载都包含新参数；不要只放进普通 Python `DualSystemMoTDriver` 对象而漏掉 checkpoint。
2. **增加带 EA 的 Architecture 变体，保留官方原路径**。建议从 `DualSystemSelfAttnArchitecture` 做可选扩展 `dual_system_self_attn_ea`，或在 `architecture.use_relational_sidecar` 下分支；默认 false 原样执行，原 checkpoints strict/非 strict 加载行为与缺失 EA keys 分开处理。
3. **修改/包装 `DualSystemMoTDriver.run_joint_loop()` 的 block loop**：

   ```python
   z = sidecar.init_tokens(text_context, current_proprio, masks)
   for layer_id in range(driver.num_layers):
       vstate, astate = driver.step(layer_id, vstate, astate,
                                    attn_mask=original_mask,
                                    use_gradient_checkpointing=...)
       if layer_id in (7, 15, 23):
           stage = tap_to_stage[layer_id]
           n0 = derive_first_frame_token_count(vstate.grid_height,
                                               vstate.grid_width, original_geometry)
           h_clean = vstate.hidden_states[:, :n0, :]
           z = sidecar.read_refine(stage, z, h_clean, text, current_proprio, masks)
           astate.payload.x_action = sidecar.write_action(stage,
                                                           astate.payload.x_action, z)
   aux = sidecar.predict_aux(z, intermediate_outputs)
   ```

   其中 `n0` 应按仓库实际提供的 `grid_height/grid_width` 或官方 `compute_video_tokens_per_frame` 计算，并进行 shape assert。上述伪代码是调用逻辑，**不是已验证可直接复制的 API**。
4. **必须保护 checkpointing 语义**：官方 `driver.step()` 启用 gradient checkpoint 时在内部对 `vstate`/`astate` 做浅拷贝、只更新 layer-varying tensors。**Sidecar 放在 `driver.step()` 返回之后、下一 layer 之前**；不要在 checkpoint closure 内原地修改 `extras/context`、persistent Z 等共享 Python 对象。`z` 用普通 tensor 变量显式传递；不能通过 forward hook 全局缓存或副作用隐式传值。
5. **不破坏原 `forward()` 2-tuple 合同**：原模型继续支持 `(video_velocity, action_velocity)`。EA 的 `aux_outputs` 在显式、受控的训练路径中传回 loss，建议新增可选 `forward_with_aux()` / `return_aux` 变体或对应 EA architecture 的 `compute_loss()` 扩展。不要将辅助标签混进 `pipeline_inputs` 透传到 `vb.prepare()`，也不要依赖 `self.last_aux` 等全局 mutable cache（checkpoint 重算、编译/并发危险）。
6. **配置**：新建 `configs/model/ea_rsig_sidecar.yaml` 或在新 architecture YAML 下配置 `enabled`, `tap_blocks:[7,15,23]`, `latent_dim:512`, `num_ea_tokens:15`, `heads:8`, `read_gate_init:0.01`, `write_gate_init:0.0`, `aux_loss_weights`, `stage_supervision`；模型外的 batch/dataloader 配置不改动原版文件。
7. **首次训练关闭 `torch.compile`**（EA 的自定义 Python sidecar hook 未包含在官方预编译 MoT loop 内），使用官方 checkpointing 路径、BF16/ZeRO-2；正常训练后再优化 compiled path，不要让 eval 自动退回旧的无 EA compiled loop。

## 4. 数据 / 监督 / Loss（不改变预训练 Action 合同）

- OpenWAM RoboTwin 配置继续使用：`action_mode:eef`、`unify_action:true`、`action_dim=80`、`state_dim=80`、33 raw states→32 actions、video stride4→9 frames。双臂有效 20D 按 `0:10`、`34:44` 映射；单臂保留相应槽位与 mask。
- 使用 OpenWAM 官方 normalization、rot6d、gripper 方向、EEF 刚性参考点与逆转换；**不要把 14D qpos 当 EE action**，也不要首轮改成 delta EE。
- EA GT 为单独 `ea_targets` 字段：role bbox（canvas 坐标）、visibility/track valid、near/contact/goal、未来 8/16/32 的有效 EEF/Object direction 与 moving 等。Simulator GT **只用于 loss**，不能输入 sidecar 或 Action。
- Loss 第一版：`L = L_video + L_action + λg L_ground + λr L_relation + λp L_plan`，主干 loss 权重沿用官方；EA aux loss 必须做有效标签 mask。初始 EA 系数可以沿用当前 v3 量级作为调试起点，但最终要记录每项 loss 和梯度范数，不凭 loss 数值直接决定权重。
- 可选 `L_action_plan_align` **默认关闭**：Action FM 网络预测 velocity，不直接是 clean action；必须基于官方 flow schedule 正确重建 predicted clean action，再转换回物理 EEF Pose 才可算方向一致性。不得直接对 velocity/normalized rot6d 计算几何 loss。先跑通基本方法再启用。
- 每一训练样本的 EA supervision 时间轴必须与 official 32 action steps 对齐；跨 RoboTwin ALOHA/Piper、ManiSkill/Franka 数据要分别记录 `control_dt`、EEF reference frame、TCP offset、achieved-versus-commanded action 定义。

## 5. 训练策略（4×A100 优先）

- **阶段 0 / 工程验证**：Foundation + EA（所有 write gates=0）跑 full forward+backward+optimizer step；相同输入/噪声下 EA-disable 和原 Foundation action/video velocity 数值一致；检查 checkpoint save/load。4×A100 显存/吞吐必须实测，不能据 4×H100 经验保证。
- **阶段 1 / Sidecar warmup**：先冻结 OpenWAM 主干；训练新的 Sidecar 和 Aux Heads（以及 write gates），先让 Ground/Relation/Plan 的有效标签监督工作。若侧路只靠 Action loss、而 write gate 置零，早期梯度可能不足；辅助损失不能省。
- **阶段 2 / Joint Action Adaptation**：Action DiT + Sidecar 可训练；Video DiT 可以先冻结，维持预训练 prior；检查梯度是否通过原 MoT 回传导致超预算。若 4 卡资源允许且确有证据需要，再考虑 Video DiT 部分解冻 / LoRA。
- **阶段 3 / 下游 SFT**：与原 OpenWAM baseline 完全匹配训练数据、Action 格式、学习率/有效 batch、更新步数和 Eval Controller。先实现同一平台 Clean/Random 闭环，再扩至 multi-embodiment privileged midtrain。不要把“额外仿真数据收益”计作 Sidecar 的独立贡献。
- 训练先小规模 smoke（例如 20 optimizer steps）+ 少量任务验证；不按假定 4×A100 速度提前承诺全量 fine-tune 时长。

## 6. 必须通过的验收测试（Codex 先写测试再开大训练）

1. **Identity**：`enabled=False` 或 `beta_7=beta_15=beta_23=0`，加载同一预训练主干，固定输入/timesteps/noise，Action/Video outputs 与官方基线在数值精度容差内一致；不能静默修改官方 mask / RoPE。
2. **Current-observation causality**：固定当前 RGB、Text、Proprio 与干净第一帧；改变 noisy future video、未来 GT action、action diffusion noise/time。三层读取的 first-frame features、Sidecar `Z`、所有辅助预测应保持一致（容差内）。**注意：Action 输出当然可以改变，不能要求它不变**。
3. **Read/Write isolation**：关闭深层 read gate 不应破坏浅层已存储的 `Z`；关闭全部 write gates 时 EA 仍能从 aux supervision 学习但不能影响主干输出。确认 EA `Z` 无 Action→EA 路径。
4. **Gradients**：分别测试 stage1/2/3 read projector、aux heads、三处 write gate 的非零梯度；冻结 Video 参数时其 grad 应为 None；至少完成一个真正 optimizer step。避免自动丢失 EA 参数在 optimizer/state_dict 中。
5. **Mask/roles**：双臂、单臂、不可见 goal、invalid track、静止臂、episode 末尾无未来标签必须数值稳定、损失 mask 正确；bbox 在三相机拼图到 latent grid 上对齐。
6. **Training/deploy parity**：同一当前输入、不同 FM timestep，EA 的 `Z`/aux 不可变化；确保部署推理每次都执行 Sidecar 而不是误入原 compiled MoT loop；测额外 latency。
7. **Checkpointing**：与关闭 checkpointing 的小模型梯度对齐；save→resume 完整恢复 gate、Sidecar、optimizer；不得原地修改被 shallow copy 共享的 extras/context。

## 7. 最小有意义的消融与验收指标

- A：OpenWAM Foundation → 同数据普通 SFT（baseline）。
- B：OpenWAM + Sidecar Aux-only（`beta=0`，相同监督，不写 Action）。
- C：OpenWAM + **3-stage Read–Refine–Write**（主实验）。
- D（资源允许）：C 仅早期 Read 或仅第一处 Write，检查 multi-stage 净收益；注意推理置零只是筛查，严格归因仍需匹配训练。

报告 `Clean/Random closed-loop SR`（主指标）、动作离线 EEF error、按接触/夹爪/静止臂分阶段错误、Ground/Relation/Plan 的 Clean/Random 退化、参数量、每步训练时间、推理 latency。关注你们旧 DiT4DiT 的教训：**Action error 降低不保证 closed-loop SR 提升**。

## 8. 交付给 Codex 的验收产物

1. 配置开关即可加载 **原版 OpenWAM** 或 **OpenWAM + Sidecar**，不覆盖官方 checkpoints 和 YAML。
2. Sidecar 模块、训练损失接线、数据字段/掩码、部署路径均打通；新参数可保存/恢复。
3. `pytest` 覆盖 causality、identity、gradient/checkpoint、单双臂 invalid masks、deploy parity。
4. 提供单卡 CPU/mock 结构测试、4×A100 的 20-step smoke 命令，以及训练日志中的每个 gate、辅助 loss 和 step time。
5. 输出具体更改文件清单、未解决事项与 GPU 显存/吞吐实测；严禁声称本架构已经验证提升 C2R SR。

**一句话核心机制：** 以 OpenWAM 受因果保护的当前第一帧多层表示为知识源；15 个 EA tokens 在三处 sidecar stage 累积、精炼和接受仿真特权监督；每次经新建 gated cross-attention 将关系条件写回现有 Action hidden states；随后由**原始** MoT joint-attention 传播到后续动作和预测世界，训练与部署均不读取 GT 未来状态。


## 9. 2026-10-16 实施中的合同细化

本节补充实现决策，前述机制和验收目标保留。具体执行证据持续记入 `docs/RELATIONAL_SIDECAR_IMPLEMENTATION_20261016.md`。

1. 独立模型配置 `dual_system_relational_sidecar`、架构 variant `joint_self_attn_relational_sidecar`；原 `dual_system_ea_rsig_lite` 及 block0 实现保留。关闭新模块用 `model.architecture.relational_sidecar.enabled=false`；普通 baseline 直接选择 `model=dual_system`。旧 Lite checkpoint 与新 Sidecar checkpoint 不可互相宽松加载；二者分别从同一 Foundation 初始化，防止静默丢失 EA 权重。
2. Z 的机器人输入是 **已按自有 EEF stats 归一化的当前80D proprio，以及同维 mask**，拼为160D后编码。没有额外 oracle EEF/TCP 输入，也不把80D归一化坐标当物理米制坐标计算几何 loss。监督沿用每角色当前 local frame 的物理方向；输入学习映射而非伪装为已解析的物理 TCP。mask 从当前观测/部署 unified normalizer 映射取得，不能从GT track_valid取得。
3. 空间监督原始130D字段仍保存每相机归一化 xyxy，以复用 reader 和标签合同；loss 显式把 head 映射到画布上2/3、左右腕映射到底部1/3的左右半幅。直接在真实 runtime H×W 上计算分布，不将三个视角各重采样到14×14。高斯仅在所属视角区域有质量；不可见或goal_valid=0的目标不强造可见框。
4. 最终 stage 接 Ground/Relation/Plan 三项；stage1 Ground、stage2 Relation 额外乘 deep_supervision=0.1。15个 token 在全部stage更新。深层 alpha=0 的意义是停止新增视觉读取，Refine 仍可变换旧 Z，不能错误要求与浅层 Z 逐元素相同；应测试浅层干预仍能影响最后 Z。
5. 新模型 `forward_with_aux()` 显式返回三层输出，普通 `forward()` 保持2-tuple。基础 FM 只新增可选 `forward_fn` 参数，默认调用原 self(...)；新模型用一次 compute_loss 调用内的局部闭包取得辅助输出，不写 self.last_aux / extras / context，也不依赖 forward hook 传递关系状态。梯度诊断 hook 仅记录梯度范数，不参与前向数据流。
6. beta=0、输出投影保持普通随机初始化，避免双零初始化使 beta 无梯度。warmup 冻结原主干；action_only 仅训练 Action+Sidecar，Video 参数冻结但不把整个 MoT 包进 no_grad；joint 为显式全量 DiT 适配。三模式均不自动启动长训练。全量 joint 不保证在现有硬件容纳。
7. 数据限定新4350集合内 raw/aloha 的50任务/2500轨迹，**不是将处理后14D qpos作为动作**。离线从本批原始 oracle 重新构造8/16/32步和新RGB匹配标签、拟合 EEF stats 并缓存启动元数据；不复用旧腕部 near=.1 的 bbox 或旧10/20/40计划。训练启动读取 ready manifest，保留一次快速预检和首次读取源文件的size/mtime绑定。
8. RGB reader 暂时复用已验证的 RoboTwin legacy JPEG 约定（cv2 decode 后不额外交换通道）；准备程序检查每episode三相机首尾 JPEG 标记，出现 XPL-RGB1 就拒绝，要求先明确对应 codec。该检查不是逐帧 RGB 视觉验收，4350原数据的最终验收记录须与本次OpenWAM EEF重建验收分开。
9. action-plan alignment loss 不实现、不开启；缺少物理 clean-action 重建时不能拿 FM velocity 当运动方向。Interaction Summary 没有百分比 progress 标签，不引入虚假监督。
10. 本机旧评测作业保留；本轮以 CPU tiny真实主干、源数据合同和启动命令验证为主。4×A100/ZeRO2的20次更新、容量/速度、全量 Foundation BF16 数值及闭环 SR 是独立硬件/实验门槛，未执行时必须写“未验证”。
