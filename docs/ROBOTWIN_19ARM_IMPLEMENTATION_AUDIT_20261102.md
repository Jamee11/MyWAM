# RoboTwin 19组推理干预实现审计（2026-11-02）

本次只读审计实际源码、既有GPU固定输入验收，并新增checkpoint小张量CPU计算；没有修改模型、干预、调度或正在运行的实验。诊断权重：`outputs/1030-warmup2k-to-full10epoch/checkpoint_step_21480.safetensors`，SHA256 `b141060b4ab33b0a2542effad452203b2ee5a5d287e5fd38ef092abe7326de7e`。本文件不是最终成功率报告。

## 1. All-Reset确实保留非零Write残差

实现：`openwam/model/modules/relational_sidecar.py`的`ActionWriteAdapter`。

```
kv = LayerNorm(tokens)
r = MHA(query(actions), kv, kv)
actions_out = actions + beta * projection(r)
```

Write的beta是直接训练标量，不经过sigmoid；初始化0.0，Read alpha初始化0.01。因此**新初始化时beta=0，确实零Write残差**；不能用初始化值代替已训练值。LayerNorm默认有affine bias，MHA默认有in_proj_bias及out_proj.bias，最终projection无bias。

本次从真实21480权重读取：

| tap（0-based） | 训练后beta | LN bias L2 | MHA V bias L2 | MHA输出bias L2 | 全零token时Write隐空间残差L2 | 最大绝对分量 |
|---|---:|---:|---:|---:|---:|---:|
| 7 | -0.0026397705 | 0.316713 | 0.351207 | 0.200389 | 0.078297 | 0.017098 |
| 15 | -0.0025177002 | 0.229941 | 0.253180 | 0.180461 | 0.042425 | 0.007148 |
| 23 | 0.0037231445 | 0.261294 | 0.273942 | 0.190172 | 0.056420 | 0.006894 |

当全部15个Z置零，LN(Z)=b_LN，每个有效token的K/V相同；注意力权重和为1，故单层残差可写成：

`c = beta * W_projection * [W_out * (W_V * b_LN + b_V) + b_out]`。

这不是直接把K或V置零。K bias与query影响注意力分数，但当所有有效V相同，不能改变加权和。固定权重、mask、有至少一个有效token且dropout关闭时，此分支在实数数学中不依赖动作query/图像。CPU FP32真实MHA两组随机query与公式最大误差小于4.7e-9。读取的是BF16权重转FP32，**不是GPU BF16精确复现**。这些数值是隐空间单位，不是米、动作误差或成功率变化；后续非线性网络、cache决定最终action变化，不能由常量分支推导整条轨迹常量偏置。

可复现计算与JSON：`docs/evidence/sidecar_intervention_audit_20261102/measure_reset_write.py`、`reset_write_measurements.json`。

## 2. Reset位置、时序和EA head的细节

运行hook源文件：`outputs/robotwin_eval/sidecar21480_expanded_ablations_20261102/intervention.py`。

架构forward中每个tap顺序为：主MoT block → `sidecar.read_refine(index, z, ...)` → `sidecar.write_action(index, actions, z, valid)`。

hook先执行原始read_refine，再clone返回的tokens并把指定slice赋0，随后返回给架构Write。因此与“read_refine之后、Write之前”一致；每次实际执行的forward中三个tap均重复干预，不是仅初始化reset一次。reset后的z也传到下一tap；下一tap重新Read/Refine后再次reset。token不跨generate持久化。

重要限制：原始read_refine已计算EAHeads并返回`output`，hook不重新计算这个output。故本组是**latent token干预**，不是EA预测头输出置零；返回aux中的预测可能是reset前的值。闭环动作路径使用reset后的z。部分token reset不保证消除其先前向其他未reset token传播的信息；不能视作该语义信息从整个模型彻底删除。

## 3. 19组逐项实际实现

所有off都发生在推理wrapper中，不改权重。前三个tap分别是block7/15/23（零基编号）。

| 模式 | 具体干预位置与操作 | 仍保留什么 |
|---|---|---|
| full | 不干预 | 完整Sidecar |
| write_off | 三个`write_action`直接return原actions，不调用Write | Init、Read、Refine、FFN、EA heads照常计算 |
| grounding_reset | 每tap read_refine后`tokens[:,0:4]=0` | 其他11token、Write |
| relation_reset | 同位置`tokens[:,4:8]=0` | 其他11token、Write |
| summary_reset | 同位置`tokens[:,8:9]=0` | 其他14token、Write |
| plan_reset | 同位置`tokens[:,9:15]=0` | 其他9token、Write |
| all_tokens_reset | 同位置`tokens[:,0:15]=0` | Write LN/MHA/bias/gate、主网络 |
| visual_read_off | 每stage.read(MHA)直接返回`(zeros_like(query),None)` | tokens残差、position/projector仍计算、Refine/FFN/Write |
| refine_off | 每stage.refine(MHA)直接返回`(zeros_like(query),None)` | 输入tokens残差、Read/FFN/Write |
| sidecar_text_off | `sidecar.text`投影完整输出置零（含bias） | 主网络文本输入、Sidecar proprio、role、视觉Read |
| sidecar_proprio_off | `sidecar.proprio`完整MLP输出置零 | 主网络proprio、Sidecar text、role、视觉Read |
| write_tap7_off | 仅stage0的write_action直接return actions | 另外两个Write及全部Read/Refine |
| write_tap15_off | 仅stage1的write_action直接return actions | 另外两个Write及全部Read/Refine |
| write_tap23_off | 仅stage2的write_action直接return actions | 另外两个Write及全部Read/Refine |
| role_embedding_off | Init返回后减去`role_tokens[None]*valid` | context初始化、后续所有stage；低精度减法不保证等于独立重建无role的逐bit结果 |
| ffn_off | 每stage.ffn完整输出置零（先算原输出再zeros_like） | tokens残差、Read/Refine/Write |
| position_off | 每stage.position投影完整输出置零（含bias） | 视觉projector、其他位置编码及主网络 |
| stage_context_off | 每stage.context投影完整输出置零（含bias） | Init仍含text/proprio context，主网络条件仍在 |
| sidecar_off | 专用服务`arch.sidecar_enabled=False`，forward直接走父类 | 同一joint-trained主网络权重与原始RGB/text/proprio输入；全部Sidecar Init/Read/Refine/Heads/Write不运行 |

`zero_output`对于Linear/FFN是先运行原module再把完整输出置零；对于Read/Refine MHA则直接跳过计算返回零。所有reset是确定性赋零，当前19组里没有随机shuffle、随机重采样或“恢复到learned初始token”。头预测单独关闭的旧离线对照不属于这里19组；EA head输出本身不回灌action，所以其单独置零不等于latent token reset。

## 4. Sidecar-Off与Write-Off不是同一代码路径

Write-Off走Sidecar子类forward并计算Sidecar；Sidecar-Off在子类forward起始处调用父类`DualSystemSelfAttnArchitecture.forward`，generate中的proprio-mask注入分支也不同。当前Sidecar唯一显式向主干写入的接口为Write，因此在无随机副作用、相同主干条件/数值执行下，两者预期动作一致，但不能称代码路径相同或无条件对所有未来实现都等价。

已有真实GPU验收：两个固定原生输入、相同cache配置，两组actions逐元素完全一致，maxabs0；Sidecar-Off的Sidecar所有子模块及init调用为0，重复Off一致，恢复Full后也逐元素一致，参数版本未改变。证据：`outputs/robotwin_eval/sidecar21480_all_off_20261102/replica0/eager_validation/result.json`。

该结论仅覆盖两个固定输入，**尚未证明所有闭环状态/整个轨迹等价**。若最终两组小样本SR不同，应先检查相同seed实际初态/渲染/执行轨迹和逐chunk动作，再解释性能差异；不能直接据此声称Read模块有独立的动作收益。Sidecar-Off也不是重新加载未联合训练的原版OpenWAM baseline。

## 5. 随机性、cache与配对比较

- Read/Refine/Write的MHA均`dropout=0`；Sidecar FFN没有dropout。训练role随机初始化已被checkpoint替代，不是eval随机重初始化。
- model_loader调用`architecture.eval()`，服务验收检查`not arch.training`。编译关闭（Eager），DiT cache开启，阈值0.99、最多连续skip3，10次denoise，CFG1，同步32步chunk。
- Engine每次generate重置DiT cache；主生成动作noise使用`torch.Generator(device).manual_seed(seed)`，本路径未传seed时engine固定42。环境seed和模型生成noise seed是两个不同概念。
- 共享服务对整个predict使用同一锁；mode为thread-local并在finally恢复；端口各自policy/action buffer。已有18模式跨两副本动作逐元素一致、各模式后恢复Full逐元素一致的固定输入验收，不是仅ping。
- **DiT cache是近似算法，不是随机dropout。** 其复用判据依赖最近两次video/action velocity余弦，干预可能改变阈值是否触发，从而改变实际forward次数与Sidecar tap调用次数。跳过forward时三个tap都不执行，沿用前次joint velocity。相同cache配置避免明显启动方案不齐，但不能把SR差完全归因于“无cache原始模型的Sidecar作用”，cache与干预可能交互。
- 尚未对19组每个chunk记录cache skip schedule并做同输入无cache/固定schedule对照；当前两个fixture的repeat精确一致不能证明所有状态确定性，更不能消除仿真接触/渲染/规划数值差异。
- 同manifest固定环境seed、语言并不保证分叉后的模型观测仍相同。渲染错误条件已隔离，最终完整同seed重测；错误不能记为策略失败。最终每组10task×2domain×10=200条，19组3800条，当前仍未完成。

后续最小必要核对：代表性固定输入分别比Write-Off/Sidecar-Off（缓存开和关）；对Full/All-Reset记录skip schedule；若需闭环机制结论，再对少量同manifest任务做cache关闭的配对复核。先完成当前队列，不在本次审计中切换在跑配置。没有使用独立重复实验时，10条/条件差异不能当稳定显著收益。

## 6. 代码快照范围

独立Git分支保存本次相关模型/部署代码、实际运行的19组hook与调度脚本、必要通信适配、验收JSON、推理配置和本审计。`outputs/`中的这些源文件原本被gitignore忽略，本次显式归档。排除权重、数据、运行时env、readiness token、SSH密钥和大日志。克隆后的固定绝对路径需要按机器修改；checkpoint、RoboTwin资产和环境单独准备，不能把源码快照当自包含可运行镜像。当前工作分支、index和在跑进程保持原状。
