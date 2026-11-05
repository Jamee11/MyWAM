# Sidecar下一步：完整速测与DiT4DiT v1/v2/v3的证据对照

日期：2026-11-03。本报告是研究与实验决策，不启动新训练、不修改模型/训练逻辑。已授权的Refine-Off收尾与PatchWAM最终42960速测继续。分析快照：`outputs/analysis/sidecar_nextstep_dit4dit_comparison_20261103/analysis.json`；逐条件表`per_condition.csv`。

## 1. 当前判断

优先保留“监督学到任务相关表征，再用于动作”的主线，但不要把“每种token都需要在线不可替代”作为必须达到的目标。现有证据更接近**表征有训练价值，在线条件的增量收益较弱且不稳定**，而不是已证明整个Sidecar有害。

目前没有可靠依据承诺一个几行改动能增加5–10pp。最短路径是完成已经启动的两个完整速测，保留已有Grounding-guided Read主训练，把新增尝试限制为一个有针对性的短续训对照，而不是同时改loss、gate、LR、mask和数据。

## 2. 从19组到全任务：收益为什么收窄

### 2.1 固定manifest诊断已完成

10个选定任务×双域×10条；每组200条，19组3800条，最终seed/语言/渲染审计通过。

| 干预 | Clean | Randomized | 综合 | 对Full差值 |
|---|---:|---:|---:|---:|
| Full | 92% | 62% | 77% | — |
| Relation-Reset | 93% | 70% | 81.5% | +4.5pp |
| Refine-Off | 93% | 69% | 81% | +4pp |
| Write-Off | 91% | 69% | 80% | +3pp |
| All-Reset | 91% | 66% | 78.5% | +1.5pp |
| Sidecar-Off | 92% | 63% | 77.5% | +0.5pp |

每种干预只有200条，19个配置不是19次独立重复训练。Full与Refine-Off有13个失败变成功、5个相反，净+8；未校正配对McNemar p约.0963，不能由最高组的点估计声称已确证收益。Sidecar-Off净+1/200尤其不能支持整体有害。多重比较与任务挑选要计入解释。

### 2.2 Refine-Off全50任务速测：08:31后分析快照

当前98/100条件有效完成（尚缺place_can_basket/Randomized和put_object_cabinet/Randomized）。不是最后3000条全套。与旧Full严格匹配**任务和域子集**，旧native30未冻结语言/全套manifest，不能称逐episode控制所有变量的配对实验。

| 比较范围 | Refine-Off | 同任务域旧Full | 差值 |
|---|---:|---:|---:|
| 已完成98条件，2940条 | 2247/2940=76.43% | 2219/2940=75.48% | +0.95pp |
| Clean，全部50条件/1500条 | 1341/1500=89.40% | 1330/1500=88.67% | +0.73pp |
| Randomized，48条件/1440条 | 906/1440=62.92% | 889/1440=61.74% | +1.18pp |
| 原诊断10任务×双域×30条 | 477/600=79.50% | 471/600=78.50% | +1.00pp |
| 原诊断10任务Randomized×30条 | 203/300=67.67% | 196/300=65.33% | +2.33pp |
| 其余40任务已完成78条件 | 1770/2340=75.64% | 1748/2340=74.70% | +0.94pp |

43条件改善、28下降、27持平。Randomized中24改善、15下降、9持平。例子：move_stapler_pad 7→13/30，place_object_basket16→21；adjust_bottle21→14，open_laptop22→17，stack_bowls_two26→21。因此绝不是每个任务统一受益，也不只是原10任务有效。

对任务平均差做10000次描述性task bootstrap：全域约[-0.30,+2.33]pp，Randomized约[-1.46,+3.89]pp。部分域缺失时task平均和按episode平均不完全相同；该区间不是训练种子置信区间，不包含未冻结语言、物理重复性、所有搜索/多重比较不确定性。它至少说明不宜把当前约1pp宣称稳定显著。

收窄的合理解释包括：原10任务代表性有限；每条件10条的采样波动；从18个候选选择较好组的乐观偏差；新增20条难例；native30的语言/seed集合差别；cache交互与仿真执行差异。现在不能把这些因素分解成各自百分比，也不能选定唯一原因。

原Full完整3000条74.83%（Clean88.67/Randomized61.00）；当前98条件分布更容易，不能拿76.43直接减74.83。最终100条件再算全套差值。

## 3. 离线正、闭环弱，如何理解

训练Clean40个窗口，Full未来EEF XYZ误差4.889mm，Write-Off5.035mm（均值差约0.146mm）；Grounding-Reset5.042mm，Relation-Reset4.890mm，Plan-Reset4.875mm。这仅是训练窗口、单次固定noise生成对专家future achieved EEF的比较。

它说明某些通路改变了拟合，但不说明分布外闭环收益：模型一旦产生不同动作，下一次输入就来自其自身轨迹，接触、遮挡、物体偏移和恢复状态会不断改变。均值亚毫米差也不直接等价于抓取/放置是否过成功门；误差时序、方向、旋转/夹爪和关键阶段更重要。[DAgger论文](https://proceedings.mlr.press/v15/ross11a.html)解释了这种由策略动作改变后续输入分布的问题；它支持这一机制可能性，不证明本项目已定位到该原因，也不意味着现在必须上在线专家重标。

这里的unseen主要是任务内新seed/布局/外观/语言和策略偏移状态，不是50个全新任务。Randomized还含相机、光照、纹理、干扰物等变化，不能全部称“没见过的物体”。不能从SR表辨认具体哪类randomization致因。

## 4. DiT4DiT历史：已经从原始summary重算

主目录和独立`DiT4DiT-EA-RSIGv2`分开检查，避免把“数据v3/v4”误当“模型RSIG-v3”。以下同为六任务×50条、replan8、200步协议，不与当前八任务600步HardRandom的73.13%直接比较。

| DiT4DiT配置 | 成功数/300 | 综合 |
|---|---:|---:|
| Baseline80K | 223 | 74.33% |
| RSIG-v1 Full80K | 234 | 78.00% |
| RSIG-v1 80K推理Plan-only | 286 | 95.33% |
| RSIG-v1 60K推理Plan-only | 291 | 97.00% |
| EA-RSIG-v2 Full80K | 299 | 99.67% |
| EA-RSIG-v2 80K去直接Plan token | 292 | 97.33% |

六任务summary每份episodes=records=50、成功字段与records求和一致，已本次重验，完整来源在analysis.json。V1 80K Plan-only比V1 Full+17.33pp，而V2 Full比V1 Plan-only只多4.33pp。这两种差值必须分开。历史600步协议下baseline60K本身97.3%，所以200步结果包含明显执行效率/预算因素，不是所有任务永久不可达。

### 4.1 v1具体薄弱处

`DiT4DiT/model/modules/rsig.py`：只有EEF/Object两role，没有Goal；role/interaction latent直接投影给Action。Plan数值由interaction预测，再`plan_features.detach()`后投影。故Action loss可训练plan projection，却不能沿这条直接Plan路径优化上游plan predictor；上游仍可通过其他共享latent路径收到Action梯度，不是整个RSIG与Action完全断开。

Runtime Plan-only只把直接role/interaction Action tokens归零，内部角色提取与训练过的辅助表征还在，主干H18也在。这个历史结果证明“更多无约束条件不一定更好”，不证明能把角色学习从训练中删掉。

### 4.2 v2改变的是信息构造与梯度，而非只删模块

`DiT4DiT-EA-RSIGv2/DiT4DiT/model/modules/ea_rsig.py`：

- 显式增加Goal/Reference；四节点EEF0/EEF1/Object/Goal。
- 推理增加22D可部署EEF及arm mask（Panda16+22=38D），不是用sim物体GT。
- 节点attention使用key padding mask屏蔽缺失臂；不能误写成已实现EO/OG硬编码边的typed adjacency。
- graph节点/pooled条件与更多关系/目标监督共同改变表征。
- `_project_plan(plan_features)`不detach，Action可以沿Plan条件训练预测器。
- 新辅助权重、训练批量等也变化，99.67%不是某一层改动的纯消融收益。

No-plan仍97.33%，说明v2性能不主要依赖三个直接Plan tokens的绝对必要性。Goal、EEF条件与图表征可能整体更稳；从总分不能确定各项因果贡献。

### 4.3 v3可借鉴的实现与证据边界

当前`DiT4DiT/model/modules/ea_rsig_v3.py`：Object/Goal soft-region grounding得到的attention直接池化visual values形成slots；presence预测插值null slot；typed EO=EEF/Object、OG=Object/Goal关系再驱动Progress/Plan；每臂每horizon分别保留Plan token，避免v2 active-arm平均把相反方向抵消；非Plan直连使用初值0.1的gate。

这是一条有方向的计算链，不是仅给15个自由token命名并加语义head。v3并非完全无attention，但主要语义依赖更明确。代码当前还包含之后新增的训练专用H18 residual，不能把9月初旧文档“无video injection”的版本描述照搬到当前。

**不能确认“v3已解决了与当前相同的推理干预问题”。** 查到的RoboTwin v3全任务EQpos可达Clean76.42/Random23.26%，EClean64.40/4.30%，但数据/动作/初始策略/LR有混杂，未找到足以证明“同权重Full稳定胜过所有token-off”的对应ManiSkill闭环组。v2的正式结果是真实证据；v3的结构启发与对应同问题解决证据需要分开。

## 5. 当前OpenWAM已经有的，不要重复修

- 已有实际EEF的80D masked proprio、Goal token、Object/Goal bbox监督；不需要照v2再添加一套EEF或GT输入。
- current Sidecar Action Write没有v1的`plan_features.detach()`；正常joint下Action loss可通过Write回传到latent。把“重新打通Plan梯度”当作当前修复是错误的。
- 原版Ground heatmap由EAHeads在Read/Refine之后读出；Relation/Plan最初自由读取全部visual features。heatmap不是原Read的同一个attention，优化ground head不自动保证下游Relation/Plan读取对应位置。
- Refine对15个token做全连接self-attention，只有valid mask，无明确EO/OG依赖边。Read有alpha、Write有beta，但Refine残差没有单独强度gate。这是值得检验的结构自由度，不是已证明bug。
- All-Reset不是Sidecar-Off：LN/MHA biases造成常量Write残差。Sidecar-Off仍用联合训练后的WAM，不能替代matched从头no-Sidecar训练。
- 辅助head的输出不回灌action，其预测做得好只是间接训练信号。当前名字叫“Plan token”不保证Action依赖其可解释Plan数值。

## 6. 建议：一个主线，一个便宜候选

### 6.1 主线：保留已经启动的Grounding-guided Read

已存在`docs/GROUNDED_READ_CLUSTER_RUN_20261103.md`与作业1310504（本轮仅核查本地已有运行证据，未重复登录核验其即时状态）。不要再实现一套重叠Grounded模块。

它把stage7/15预测的Grounding先验接到stage15/23 Relation/Plan Read logits；Object/Goal置信度与arm mask控制先验，混50%uniform，强度0→.5，保持Ground/Summary自由Read和所有原Write/FFN/Refine。无新标签、无GT推理输入、新teacher或额外完整主干forward。

与DiT4DiT v3最有价值的共通点：让监督的定位结果有具体下游计算用途，减少自由读取无关背景的可能。不同点是当前仍保留原15token与自由Refine，不能称已复制v3或保证解决语义混合。

它是高于盲目调gate的机制候选，但**没有闭环收益保证**。新作业在目标机器重新Foundation→2K预热→fresh joint10epoch，并未使用本地原同一warmup权重，故将来对旧21480比较时应标注预热实现/随机起点差异；不能只把所有差值归因Read。额外10epoch只改变一个主训练因素更容易解释，后续不要在同run再叠XA、三stage loss或augmentation修改。

### 6.2 最小新候选：Refine-Off短续训，而不是再开第二个10epoch

这是**提案，未修改core/配置，未启动**。若希望在现有主训练之外快速验证训练逻辑，建议：

1. 从同一最佳原21480复制两个独立run；Full继续训练与Refine-Off继续训练各0.5epoch（约1074 optimizer updates/global256）；若资源只允许一个新run，则先Refine-Off，但必须承认与未续训Full存在额外训练预算混杂。
2. 唯一架构因素是三个Refine自注意力残差分支关闭，训练与评估都关闭；保留原模块权重以兼容checkpoint，但不用“输出全零token”替代。Read/FFN/Write/全部辅助loss不变。
3. 两个run同数据次序、样本曝光、初始化、global batch、噪声schedule；fresh optimizer，小LR建议WAM/Action1e-5、Sidecar5e-6，短程warmup50更新+cosine，具体数值是工程候选不是证实最优。无新2K冻结，无新augmentation；不要同时把aux权重改小。
4. 在独立验证manifest先测10–12任务双域30条，完整相同seed/指令/noise/cache配置。当前公开速测种子已多次开发使用，不再作为无偏选择集。优先代表抓取、目标放置、多阶段、干扰物，再做未用于选方案的最终50任务确认。
5. 满足Clean无系统性退化、Randomized改善且不是单个任务净增后，再决定1epoch或完整匹配训练；0.5epoch失败不证明完整重训无用，但足以拒绝将其作为赶进度的主投入。

为什么有必要短训：现有Refine-Off是在训练完整Refine的checkpoint上临时关闭，存在训练/推理不一致；短训可检验去除后重新适配是否能保持/扩大收益。为什么不保证：训练可能把原本抑制的shortcut重新学进Read/FFN，收益也可能消失。训练期移除Refine会减少该路径梯度，不能声称全部模块因此自动正贡献。

不要同时开Refine缩放0/.1/.25/.5、Relation mask、降aux、多stage监督、teacher等网格。若最终完整Refine-Off优势消失或独立重复不稳，放弃该短训候选，把唯一预算留给Grounding-guided Read。

## 7. 先不做的几类修改

| 修改 | 为什么不作为本轮首选 |
|---|---|
| 直接扩大Write gate | 既有放大未显示总体收益；小beta不等于作用弱，可能放大错误条件 |
| 把All-Reset当可解释baseline | 仍有bias残差；与Off含义不同 |
| 把Ground/Relation/Plan数值头全detach或全关闭 | 原v1的Plan直接detach恰是值得避免的设计之一，当前没有证据需要切断Action适配 |
| 一起改loss权重/LR/augmentation | 无法知道收益来自什么，还增加重新训练预算 |
| 立即上PCGrad或完整多目标优化器 | 尚未测共享参数真实梯度冲突；标量aux loss较大不是冲突证明 |
| 立即把所有条件都变成typed graph | 方向可借鉴，但大于当前最小改动，先让已训练Grounding有计算用途 |
| 立即扩大Random数据与在线DAgger | 或许有效，但涉及数据/专家与新的公平预算，不是几行训练修改 |
| 强迫每个token的runtime-off都掉分 | 不是科学上必要目标，冗余/正则化本就可能存在；容易在测试集上反复选择 |

若短训/主训练仍不稳，下一项诊断只需小样本：同batch测共享Sidecar的Action/Video/EA分项梯度norm与cos；记录Grounding预测在Clean/Randomized的角色定位/置信度；检查动作关键阶段误差。只有实际出现长期梯度冲突，才考虑降低单一aux共享梯度系数或[PCGrad](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)，不从loss数值大小直接推导。

## 8. 赶论文的最短证据链

1. **结果**：正常部署Full在相同budget/数据下达到强水平；同时报告Full、Refine-Off和强官方/复现baseline。原74.83%对官方69.04%是不同训练方案与官方参考，不是matched模块增益。
2. **机制**：一个明确计算链改进（Grounding引导Read），用新Full/Guidance-Off/Write-Off解释，而非再铺19组。
3. **泛化**：主要指标是Randomized，另报Clean保留情况；独立manifest，不反复刷原诊断seed。
4. **归因**：至少一个匹配训练预算的no-Sidecar或EA-only基线，分清“辅助监督改善主干”和“推理时Sidecar必要性”。如果这项没算力完成，就收窄论文因果表述，不能用runtime-off替代。
5. **不确定性**：保留逐任务/逐seed数据，报告任务分布与重复性。少量训练run的点估计会造成不稳定排名，[rliable论文](https://arxiv.org/abs/2108.13264)给出报告区间的依据；一次bootstrap不能补回缺失训练种子。

不需要证明每个新增token都正贡献；需要证明最终完整方案相对强对照有可重复收益，并解释在线模块的实际职责。若结果主要来自EA训练正则，应如实把贡献定位到训练期表征，而不是宣称关系token是唯一推理瓶颈或已学得反事实动力学。

## 9. 评测执行状态

- Refine-Off尚剩2条件；没有排队未启动任务，不为占显存重复跑同条件。
- PatchWAM最终42960已经真实进入4090仿真：一份split服务（主GPU3/T5 GPU4），固定输入重复/完整Sidecar验收通过；16独立端口/16路实际推进，原生30/no-video/cache/eager不变，旧Refine-Off同时收尾。此前两次OOM/CUBLAS失败没有计为模型成绩。
- 42960实际config：Foundation起点、backbone_warmup_updates2000、sidecar_lr_screen=wam_ramp、ramp100，4rank×32×acc2/global256，最终micro42960=optimizer21480=总10epoch；**2K冻结包含在总预算内**。与原1030 warmup后再额外10epoch不是相同联合训练曝光。目录名含c2r不能替代真实finetune_ckpt_path。
- 新42960刚完成的少量Clean高分不用于评价方法排名。日志和状态见两个独立run的comparison/state；新的最终表必须待100条件完整。

## 10. 源码与证据索引

- OpenWAM：`openwam/model/modules/relational_sidecar.py`、`openwam/model/architectures/dual_system/relational_sidecar.py`；19组实际hook及bias/cache审计见`docs/ROBOTWIN_19ARM_IMPLEMENTATION_AUDIT_20261102.md`。
- DiT4DiT v1：`/remote-home/jinminghao/DiT4DiT/DiT4DiT/model/modules/rsig.py`。
- v2：`/remote-home/jinminghao/DiT4DiT-EA-RSIGv2/DiT4DiT/model/modules/ea_rsig.py`。
- v3：`/remote-home/jinminghao/DiT4DiT/DiT4DiT/model/modules/ea_rsig_v3.py`。
- 旧协议/混杂：`/remote-home/jinminghao/DiT4DiT/docs/PROJECT_IMPLEMENTATION_AND_RESULTS_AUDIT_20260905_CN.md`、`docs/项目分析汇总/2026-10-09_EA-RSIGv3与baseline训练日志参数审计.md`。
- 本次重算：`outputs/analysis/sidecar_nextstep_dit4dit_comparison_20261103/analysis.json`（所有旧ManiSkill原始summary路径）、`per_condition.csv`。
- 已有Grounding-guided主训练：`docs/GROUNDED_READ_CLUSTER_RUN_20261103.md`；不重复创建。
