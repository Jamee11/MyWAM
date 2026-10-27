# RoboTwin Sidecar：下一轮最小训练改动的决策

时间：2026-10-25；本轮只读核查实现、已有诊断与正在运行的评估，新增分析记录，未改模型、训练配置或评测进程。

## 1. 建议先做什么

**下一项训练实验只改联合阶段 EA 梯度的归属：辅助 heads 继续学习，EA loss 不再经 heads 输入更新共享的 Read/Refine/Plan 表征及 WAM；正常读写和 Action+Video FM 梯度全部保留。先用已训练 checkpoint 做 1,000 optimizer updates 的短分叉对照，不先重跑完整 10 epoch。**

这是当前证据最支持的训练候选，不是已经验证有效的实现或成功率承诺。它直接检验：共享关系表征是否需要从“持续优化辅助预测”转向“被动作生成有效使用”。主要优势是没有新参数、没有推理结构变化，不改变动作语义、采集数据或部署接口，能做严格的前向等价和梯度可达性检查。

不建议本轮同时加 Transformer、动态 gate、Action–Plan alignment、降低 Video 权重、重设 gate 初始化和改变采样。当前分支约 24.29M 参数，三层均有 Read attention、Refine attention+FFN 和 Write attention；尚无容量不足的直接证据。

## 2. 当前效果：改善存在，但不能把子集和训练时长混在一起

本轮冻结 `outputs/sidecar_next_step_review_20261025/analysis_snapshot.json`，评估快照时间 2026-10-25T16:27:04Z。与调度 state 逐条件核对 complete、exit0、30 条、成功数及30个唯一 accepted seeds 一致；这是本机保存的 comparison/state 交叉核对，未重新登录远端逐份重审原生日志。

| 对照范围 | 新 full-backward | 对照 | 可以支持的结论 |
|---|---:|---:|---|
| 同为24k，完整100条件×30条 | 66.13% | isolation24k 68.20% | 新方案在这个预算并未整体更好，差−2.07pp |
| 用户指定的固定39条件，各30条 | full-backward32k 74.36% | isolation24k 71.20% | +3.16pp，但32k比24k训练更久，且场景/指令/RNG未完全配对 |
| 最新完成45条件，各30条 | full-backward32k 74.89% | 初始直接训练Sidecar32k的同条件前30条70.37% | 同microstep的整套recipe有+4.52pp的阶段改善，尚非全100条件成绩或warmup单因素证明 |
| 上述45条件的Randomized22项 | 60.45% | 初始32k同N 52.58% | +7.88pp，是值得保留当前方案的信号；剩余任务仍可能改变总体 |
| 上述45条件的Clean23项 | 88.70% | 初始32k同N 87.39% | +1.30pp；不能为Random收益牺牲已经较高的Clean能力 |

上述45条件官方同范围为74.31%，与本地74.89%接近，但官方每条件100条、本地30条。不能把这个子集与官方全100条件69.04%相减。官方是公开参考；论文的因果归因需要自己的匹配对照。

用户指定的39条件还有明显的任务异质性：Random handover_block 新26.67%对旧60.00%，place_shoe新63.33%对旧70.00%；open_laptop则73.33%对46.67%，hammer46.67%对26.67%。这不像一个“统一增大写入就都解决”的问题。每条件30条本身分辨率3.33pp，单任务波动不能当机制证明。

依据：[固定39条件](ROBOTWIN_FULLBACKWARD32K_VS_ISOLATION24K_39COND_20261025.md)、[当前32k队列](ROBOTWIN_FULL_BACKWARD32000_30EP_LIVE_20261025.md)、[完整full-backward24k](ROBOTWIN_FULL_BACKWARD24000_30EP_FINAL_RESULTS_20261025.md)。冻结快照保存了原comparison字节及SHA，后续LIVE更新不会改变本报告口径。

## 3. 当前实现中真正需要注意的地方

### 3.1 写回已经能工作，不宜继续沿用旧模型的“写不进去”解释

旧42960实际残差比约1e-6量级；当前full-backward run在12k约0.61%/0.65%/0.90%，20k约0.27%/0.23%/0.39%。20k Full相对WriteOff，GT全32步XYZ在Clean/Random分别改善约0.256/0.144mm；原训练Action FM配对平均改善3.43%。

这说明接口的局部控制作用已经出现。强度下降同时动作可能改善，不能把目标设为“r越大越好”。这些真实残差只测到20k，不能把数值当成32k或40k的实测。

当前快照约40k：三个beta约−0.00284/−0.00285/−0.00302，read alpha约0.0674/0.1191/0.0332。beta不等于实际写回比例，也不是概率，负号不表示有害。

### 3.2 关系表征和写回接口受到不同训练目标支配

实际路径为：首帧Video hidden → Read/Refine/15 tokens → 两个出口：EA heads、Action Write。Write用Action hidden作query，因此它本来就有动作/去噪状态依赖，并非只靠一个常数gate决定全部行为。

| 参数/模块 | 当前Action梯度 | 当前Video梯度 | 当前EA梯度 |
|---|---|---|---|
| Read/Refine/Plan tokens | 经Write可达 | 经Write及原MoT路径可达 | 经heads直接可达 |
| Write projection / beta | 可达 | 可达 | 结构性不可达 |
| EA heads | 不可达 | 不可达 | 可达 |
| WAM | 原FM路径及Sidecar读取路径 | 原FM路径及Sidecar读取路径 | 可沿Read回到首帧视觉表征 |

首帧在当前mask中不能看Action，且受first_frame_causal保护；不是因为mutual就让EA读取到了GT未来Action。EA→Write的结构性零不是断图。beta零、输出投影正常初始化，也不存在双零初始化死锁。

12k的实际加权梯度诊断中，共享Refine/Plan的EA/Action中位比：Clean约112–236倍，Random约1,914–3,284倍。比旧模型几万倍改善，但训练侧重点仍值得怀疑。**这些是12k、六条episode的重叠B=1诊断，不是32k、正式batch32的当前梯度比，更不是Adam真实更新比。** Randomized只用于诊断，没有被偷偷并入Clean训练。

旧32批没有证实普遍强梯度冲突：Action–EA全局cos约+0.031，Action–Video约−0.011。因此不优先关Video或上PCGrad；PCGrad针对有冲突的梯度投影，不能自动解决所有优化侧重问题。[PCGrad原论文](https://arxiv.org/abs/2001.06782)

### 3.3 小辅助loss权重不等于共享表征受它影响小

最新1,000条训练记录为micro39003–40002，Action均值约0.000234，Video0.0768，EA0.0472，grad norm约0.0264，norm>1比例0。EA标量比Action大并不构成证明，因为FM、soft CE和归一化不同；Ground CE还有目标熵下界，不能要求它下降到零。

降低整个Sidecar LR会同时降低FM和EA两路的优化速度，不会选择性解决共享梯度比例。只降低EA总scale也会一起削弱辅助heads学习，而不能清楚分辨“heads监督有用”与“共享表征被辅助任务牵引”。梯度平衡文献支持关注训练目标的梯度分配，但不提供本项目最优比例或成功率保证。[GradNorm原论文](https://proceedings.mlr.press/v80/chen18a.html)

### 3.4 表征目标与控制需求仍有缺口，但不宜赶DDL时一起扩任务

当前没有明确受监督的task progress；Plan主要是8/16/32步的方向与moving，没有完整幅度/旋转监督；高Contact F1、方向cos只在少量专家轨迹上成立，不证明包含主干缺失的增量信息。15 tokens在一次forward内跨tap持续，跨32步真实执行不会保持一个独立的Sidecar记忆。

这些是今后改任务表示的依据，不是现在加入新loss就能提分的证据。新增幅度、方向–Action对齐或progress标签涉及工具参考、局部坐标、角色切换、静止/无效mask与新head兼容，比本轮单一梯度路由风险大。

## 4. 唯一优先训练候选：联合阶段EA只更新heads

拟新增的是一个可关闭的训练选项，目前代码里没有实现；本轮没有替用户打开它。

- warmup期间：沿用已选方案，正常Read/Write和完整Action+Video+EA反向，只跳过WAM optimizer更新。
- warmup之后：Read/Refine/Write仍正常前向和反向；EA heads读取tokens和features的detach副本，正常用原EA权重学习；Write继续用原tokens。
- Action、Video权重仍各1；aux scale仍0.5；不同时改学习率、beta初始化、token数、tap位置、帧数或数据。

拟议计算路径示意，**不是已经存在的API**：

```python
z, features = stage.read_and_refine(...)       # 保留正常计算图
z_for_heads = z.detach() if routed else z
f_for_heads = features.detach() if routed else features
aux = heads(z_for_heads, f_for_heads, ...)
actions = write(actions, z, valid)             # 必须使用原z
```

必须覆盖最终三类EA loss和中间层deep supervision。不能只detach最终tap，不能detach传给下一tap的z，不能detach已经算好的logits或整个EA loss——后者会让heads也学不到。

切换瞬间，同权重同输入的动作、视频、辅助预测值应完全一致；改变的是梯度归属，后续训练才会产生差别。因此没有新增部署算子或权重维度，原checkpoint和cache/FFN部署结构可沿用，但新checkpoint仍需重新做加速数值检查。

预期机制：保留warmup已学到的关系初始化，让共享表示随后主要响应原WAM生成目标，Write不再一直适配一个受辅助目标主导的移动接口。**有理由检验它，没有证据保证它一定涨SR。**

风险和边界：

1. 关系语义可能遗忘；不能把EA头继续有loss当作关系表示仍被直接约束。须保留held-out关系/方向指标作保护栏。
2. Action可能本来不需要这些额外信息；如果短对照无收益，应承认该假设未获支持，不能继续强迫写回变大。
3. 已有Adam一、二阶状态包含EA历史，切路由后不是第一步就“纯FM的Adam更新”；保留相同状态，不同时重置优化器，观察至少数百次更新。
4. heads梯度仍参与原全局clip，所以不能声称EA完全没有任何间接数值影响。当前稳定期clip未触发支持先保持原裁剪，不叠加分组裁剪修改。
5. 梯度旧结果只到12k，实施前用当前起点的少量真实训练batch复核共享模块；若新的正式batch下EA已不占主导且写回有稳定控制收益，应降低该实验优先级，不照搬旧几万倍结论。

## 5. 如何用最少训练预算决定是否值得继续

### 5.1 晚期同起点分叉，比先重跑10epoch更实际

先固定一份仍完整的本run晚期checkpoint及其同step的optimizer/scheduler/RNG状态。复核时40k已经保存、32k的原run full-state已按keep-last清理，36k和40k仍完整。**当前优先选择36k分叉到38k（1,000次更新），不要继续写“从现成32k完整状态恢复”。实现获授权后再固定当时实际可用的一份；不得拿32k权重配36k/40k optimizer。** 也不重新执行一次500或2K warmup；这两臂都是在同一已完成warmup的状态上继续。

若36k状态也已不在，40k之后原cosine接近低端，用它做极短试验可能因为有效学习率太低而缺少辨别力；阴性结果不能直接否定早期Stage2路由。此时应从同一可用权重建立两个明确的新finetune阶段，采用完全相同的新optimizer和预注册短程LR（例如两臂均WAM1e-5、Sidecar5e-6、100更新warmup、总1,000更新），只让EA路由不同。这属于共同新阶段下的受控比较，不是无缝resume；不能与继续原scheduler的A臂混比。两种起点方案实施时选定一种，不混用、不按评测结果追改LR。

| 分支 | 起点/数据/预算 | 唯一差异 |
|---|---|---|
| A | 同一完整状态；续训1,000 optimizer updates | 当前EA正常回到共享表征 |
| B | 同一完整状态；续训1,000 optimizer updates | EA仅更新heads |

正式RoboTwin配方保持global256、ColorJitter、同一数据/采样顺序、LR和增强设置；不要把新midtrain的global512、WAM2e-5/Sidecar1e-5同时带进这个消融。这个短试只检验晚期EA持续牵引是否限制效果，不能直接证明“从warmup结束立即路由”的全程训练效果。

**不要为了只跑1,000更新而直接把max_steps改成2,000并重新创建cosine。** 应保留原训练的总scheduler预算/状态，在约定step保存比较点后停止两臂；或使用专门且不参与scheduler计算的实验停止条件。否则“小实验”实际上也改了学习率。记录前几批sample ID、增强/输入摘要和噪声，确认对照可比；不先承诺worker增强逐bit恢复。

最近约6.65–6.88秒/micro、acc2：每臂1,000更新约3.7–3.8小时计算，两臂共用同四卡串行约7.4–7.7小时，另加加载与完整状态保存。不是重新承担80小时×两次。复核仅5/6两卡显存充足，当前没有四张空闲80GB卡；实施时需实际资源窗口，本报告不自动抢卡。

### 5.2 先检查路由正确，再做一次闭环开发集比较

工程验收必须同时证明：

1. 开关关闭时前向/梯度恢复旧逻辑；开启但未训练时动作、视频及EA预测数值相同。
2. warmup期间仍有EA→共享梯度；进入联合阶段后，EA→共享Read/Refine/Plan/视觉主干的直接梯度为零，EA→heads非零。
3. Action→Read/Refine/Write仍非零；Video路径仍正常。三处Write和deep supervision均覆盖。
4. 显式500-update边界、accumulation和保存/恢复保持一致；WAM不是requires_grad=False，也不是关闭write。

只复用已有固定输入/标签做最小实现检查，不重新拉长一轮32batch全景诊断。离线Action误差/FMFULL-OFF只是排查严重回退，不能替代SR。

闭环开发对照预先固定6任务×双域×30场景/臂，例如adjust_bottle、beat_block_hammer、place_shoe、handover_block、stack_bowls_three、open_laptop，覆盖已有改善/退化、接触/放置/交接类型。两臂复用同accepted seed、指令、policy噪声，并固定场景/RNG游标；不能只用相同整数seed就声称完全配对。共720条，任务/场景清单先冻结，不按结果挑选或替换失败。

预注册工程筛选线可设：Random净胜至少5pp、Clean下降不超过2pp，并检查是否集中于单一任务；它只是预算决策线，不是统计显著性门槛。报告配对净胜/净负与不确定性，任务样本少时不强作显著结论。若关系F1或方向cos明显塌缩、两域控制均恶化，立即放弃该候选，不增加倍率或第二种结构来挽救。

有信号才扩到100条件，随后补强固定协议的完整评估。这个预选开发集不能再冒充未触碰的论文测试集。

## 6. 500步warmup、2K旧run和新midtrain不要混成一个实验

用户最新500方案已用于独立 `configs/train_multisim_sidecar_midtrain.yaml`，它是跨本体global512的新run；当前被评估的RoboTwin32k来自2K full-backward、global256的旧run。两者不是同一训练日志或直接消融。

后续若RoboTwin从头训练采用500 optimizer-update warmup，则A/B两臂都用相同500；global256/acc2下对应1,000microsteps。跨本体global512/acc4下500对应2,000microsteps。不要在一轮对照中把warmup长度、batch、WAM LR和EA路由一起改。

WAM重启更新时原2K run有一次接近峰值LR、Adam首次建状态的短暂冲击；已有证据不证明它决定最终SR。500与全局LR warmup相交后初始LR也不同，因此它不是只改变预热时间。额外WAM LR ramp可作为独立稳定性候选，但不与本次EA路由叠加。

## 7. 其他小改动为什么排后面

| 候选 | 判断 |
|---|---|
| 增大beta/规定write-r下限 | 旧sweep无稳定单调收益，新run已可测写入；不优先 |
| 关Video或大幅增大Action权重 | 没有稳定全面Action–Video冲突证据；会改原预训练目标，不优先 |
| Task/state-conditioned gate | Write query本已随Action状态变化；全局beta冲突未证明是主因，不先加参数 |
| 统一降低Sidecar LR | 不选择性改变共享FM/EA分配，不当作主要解决方案 |
| 单独把aux scale从.5降到.05 | 配置简单，但同时削弱heads与共享监督；可作为后续更保守的权重消融，不与路由一起扫 |
| 新增Plan幅度/旋转/Progress head或Action–Plan loss | 更贴近控制，但新标签/坐标/归一化/无效mask与兼容风险更多，DDL前不作为第一轮 |
| 任务采样平衡 | 新核查确认窗口数最大/最小8.10倍；当前按窗口混合，不是按任务等权。但失败既出现在长任务也出现在中短任务，不能据此宣布根因；如以后测试，单独做温和重平衡，保留原采样对照 |

窗口审计：click_bell3,855、click_alarmclock4,252，put_bottles_dustbin31,231、open_microwave24,333；每任务都50episode，总549,775窗口。完整表保存在analysis_snapshot.json。

## 8. 如优先追求短期部署表现，可独立试一次replan16

这是可选的部署对照，不属于上面的训练改动，不应混入其A/B。当前实际`inference_horizon=32`，也就是生成32步后连续执行32步才重新生成；Sidecar关系只在生成时重新读取。可以保持预测32步不变，只执行前16步后用新观测重规划。现有SyncInferenceExecutor已有此能力，不需要加网络或改动作标签。

动机是接触/交接过程中更频繁地更新关系条件，而不是假定后16步一定不准。预计模型调用数约翻倍，可能更慢，也可能出现重规划不连续；没有当前checkpoint的闭环证据，不承诺提升。receding-horizon本身已有Diffusion Policy先例，不能声称新方法。[原论文](https://arxiv.org/abs/2303.04137)

若做，只选32对16一个对照，固定当前checkpoint和场景；Vanilla也须用同horizon。标准32协议的主表继续保留，16步结果单列为部署优化，不能把改变协议后的涨幅全归Sidecar。cache近似及其对新checkpoint的动作影响也需保持一致；不在同一组同时调cache阈值、去噪步数和horizon。

## 9. DDL下的决策顺序

1. 保留当前2K训练和32k全100条件评估，等待完整结果；不因45条件上涨就提前下结论。
2. 在实现前完成一次小规模当前梯度复核；获用户确认后，仅实现EA heads-only开关及路由/边界/恢复验收。
3. 一天量级完成同起点短续训A/B和预冻结开发闭环；无信号就停止该候选，不叠更多结构。
4. 有信号再把获胜设置用于一组正式训练并扩完整评估。保留匹配Vanilla预算，至少固定Foundation、Clean2500、增强、有效样本数/总更新、归一化及部署参数，并另外披露WAM实际更新次数。现有baseline10740十任务和官方结果不能代替这个对照。
5. RGB尺寸/拼图及绝对EEF编码已有检查，但腕部near训练.02/官方评估.1、专家实测EEF再次作为目标执行的物理跟踪、cache近似仍是独立未解决因素。主成绩不偷偷改near、动作预算、seed筛选或成功判据。

本轮结论是一个**明确、可证伪、成本可控的训练决策**：先检验持续辅助监督是否限制控制融合，让现有三层Sidecar学会为原WAM生成目标服务；没有证据时不靠增大结构或写入强度推进。

## 核对过的源码与证据

- `openwam/model/architectures/dual_system/relational_sidecar.py`：tap调用、warmup状态、最终EA/deep loss组合。
- `openwam/model/modules/relational_sidecar.py`：24M级Read/Refine/Write、heads计算和初始化。
- `openwam/model/architectures/utils/mask_modes.py`：首帧隔离、Action/Video互见路径。
- `openwam/model/architectures/base.py`：Action/Video FM与有效维度mask。
- `openwam/train/utils/sidecar_warmup.py`：完整反向与clip后，仅跳过WAM optimizer step。
- `docs/SIDECAR_FULL_BACKWARD2K_12000_DIAGNOSTICS_20261023.md`、`SIDECAR_FULL_BACKWARD2K_8K_TO_20K_DIAGNOSTICS_20261024.md`：当前run真实写回、Full/Off和梯度的适用范围。
- `outputs/sidecar_next_step_review_20261025/analysis_snapshot.json`、`comparison_snapshot.json`：本轮冻结数值、采样分布及源码SHA。
