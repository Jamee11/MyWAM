# Relational Sidecar 实现阅读与合理性评估（2026-10-17）

范围：按用户要求阅读现有实现日志、实施说明、模型代码、配置和已有验收证据。未运行新实验，未修改模型、数据、训练配置或进程。记录中的CPU/GPU验收为既有证据，不冒充本轮重测。

## 实际实现

- 旧Lite保留，新variant为dual_system_relational_sidecar。原30层MoT在零基block7/15/23之后调用三次Sidecar，即第8/16/24层之后；非每层注入。
- Z为[B,15,512]，按Ground4/Relation4/InteractionSummary1/Plan6分配；由learned role tokens、masked text mean投影、当前归一化80D proprio和80D dimension mask（160D编码）初始化。Z在一次forward内跨stage传递，不是跨控制时刻或扩散采样迭代的历史记忆。
- 每stage读取runtime首帧H×W个Video hidden tokens。独立LN+Linear投影到512D，并加入(x,y,三视角one-hot)五维位置编码。Z作为query读取视觉，然后经过15-token self-attention和FFN。全部15个token在全部stage更新，不能把三次更新解释成严格依次只算Ground/Relation/Plan。
- 每stage独立Action adapter：Action hidden作Q、Z作KV，8头cross-attention后投影回Action hidden，乘beta再残差相加。alpha初始化0.01；beta初始化0，投影保持非零初始化。原QKV/mask/RoPE/主序列长度与FM heads不变，不执行额外Video forward。
- 共享EAHeads读取typed slots：Ground与当前网格计算空间logits；Relation预测near/contact/goal；Plan每个token线性输出8维（EEF/Object各3方向+1moving）。Summary无百分比进度监督。typed语义主要通过槽位、预测头和辅助loss约束，不是显式硬编码几何图运算。
- 原FM loss保留。第三stage监督三项，第一stage额外Ground、第二stage额外Relation乘deep_supervision=0.1。系数为.025/.01/.05。130D GT独立仅用于loss；辅助输出通过forward_with_aux与局部闭包返回，不写模型级last_aux缓存。
- 新数据为robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015/raw/aloha，50任务2500轨迹552275帧549775窗口；独立再生成labels/stats，32绝对EEF动作/9视频/8-16-32保持。runtime L画布bbox映射替代固定14×14，goal_valid屏蔽不可见虚拟goal监督。

## 合理性判断

相对旧patch-only/block0一次写入，新版把预训练主干中深层的当前观察表征作为知识来源，跨深度精炼关系表示，再在多个深度条件化动作，直接回应旧浅层分支承担过重理解任务的疑问。可以作为主要实验原型，但结构收益仍待同协议闭环对照。

因果路径核对：first_frame_causal禁止首帧读取未来视频；mutual跨模态掩码单独禁止首帧读取Action；首帧time modulation固定干净条件。Sidecar只读首帧且Z更新不读Action，因此在当前代码合同下不存在通过Action间接回流未来信息的显式路径。tests/test_relational_sidecar.py包含五种干预、恒等、梯度/checkpoint与部署路由测试；已有2174项标准CPU回归记录。tiny检查不等于完整Foundation/BF16部署的所有数值性质已逐项验收。

## 仍需验证的点

1. 三个tap是工程设计点，未证明7/15/23优于其他层或更多层。
2. 当前帧的深层语义不增加真实历史；遮挡/阶段歧义仍可能不可辨识，Plan为监督预测而非已知未来。
3. 预测track/presence由辅助loss学习，但write_action的valid实际来自可观测arm/dimension mask，没有直接用预测track_valid门控。不能将当前mask称为对象置信度或OOD可靠性；这不是GT输入缺失的bug，但无效role写入的影响需评估。
4. 2026-10-17读取的既有3979-update指标：read alpha约.0347/.0708/.0388，write beta约-5.60e-5/1.85e-4/9.73e-5。它们证明门控在学习，不证明动作收益或写入幅度；后者须看delta-hidden/hidden范数以及禁用写入的配对评估，不能仅凭beta小认定无效。
5. 当前10epoch新数据Sidecar与旧5epoch旧腕部数据Lite、集群official_raw_20261001 baseline均有数据/预算差异。结构归因须在新数据上统一A baseline、B aux-only、C完整Sidecar，且匹配Foundation、归一化、增强、LR/global batch/总更新/评估控制器。不能把更换数据和训练加倍的收益都归给Sidecar。
6. 设计文档列出warmup/action_only/joint，但现有正式run配置直接joint；已有两步warmup是工程短测，不代表完成了充分预热再转joint的训练流程。真实长训与CPU检查不能替代GPU恢复及Clean/Random闭环证据。

## 主要依据

docs/RELATIONAL_SIDECAR_IMPLEMENTATION_20261016.md；docs/OpenWAM_alpha_MultiLayer_Relational_Sidecar_Codex_Impl.md；openwam/model/modules/relational_sidecar.py；openwam/model/architectures/dual_system/relational_sidecar.py；openwam/model/architectures/utils/mask_modes.py；openwam/model/video_backbone/wan_backbone.py；docs/evidence/relational_sidecar_20261016；正式run已保存config与metrics_rank0.jsonl。
