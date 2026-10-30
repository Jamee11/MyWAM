# RoboTwin 下一轮完整训练：逐步降低共享表征的EA梯度

日期：2026-10-26。状态：**待用户确认的新方案，未修改生产训练逻辑、未启动新完整训练。** 当前40k heads-only续训、原最终checkpoint评估及跨本体midtrain均保持原设置。

## 结论与问题

建议下一轮从同一Foundation完整训练Aloha50 Clean2500、10 epoch；只增加一项策略：**联合阶段逐步降低EA loss对共享Sidecar/WAM表征的梯度，最后保留1%，辅助heads仍接受原强度监督。**

它补充当前晚期heads-only短试，回答“较早让Sidecar服务生成目标，同时保留弱关系约束，是否比全程强辅助监督更好”。不是加网络，不是把所有EA loss乘0.01，也不是在现有40k分支上再叠修改。

## 1. 新的完整证据

32k full-backward快速评估已完成50任务×双域×30条，2133/3000=71.10%；独立复核100份原生结果通过。

| 模型/范围 | Clean | Randomized | 综合 |
|---|---:|---:|---:|
| 官方，每条件100条 | 89.36% | 48.72% | 69.04% |
| 初始Sidecar32k，相同条件前30条 | 87.80% | 45.53% | 66.67% |
| 当前2K full-backward32k，每条件30条 | 89.00% | 53.20% | 71.10% |

对官方Random+4.48pp、综合+2.06pp；对初始32k同N，Random+7.67pp、综合+4.43pp。样本数、accepted scene/指令/RNG与其他训练因素未完全配对，不能据此宣称warmup或Sidecar单因素因果/显著性。

这些成绩支持保留当前整套recipe，而非推倒架构。下一轮需要保护已有Random收益和接近官方的Clean能力。现有24M级Sidecar已能产生有用写入，继续强迫gate变大不是首选。

证据：[完整32k结果](ROBOTWIN_FULL_BACKWARD32000_30EP_FINAL_RESULTS_20261026.md)；`outputs/robotwin_eval/full_backward32000_4090_all100_30ep_20261025T073709Z/final_results.json`、`final_independent_audit_20261026.json`。

## 2. 为什么当前heads-only短试仍不能代替完整训练

当前分支只在40k之后关闭共享EA梯度，继续约1,480 optimizer updates至42960；此前约20,000次更新已经接受原EA监督。它检验的是晚期调整，不能回答从联合训练早期就重分配梯度的效果。

另外，40k之后cosine已接近1e-6，变化小也可能是有效学习率与剩余预算不足；变化好也不能推出从2K后完全detach一定更好。

辅助heads只有532,493参数，不能凭自身保证变化中的上游特征始终包含关系信息。完全detach有语义遗忘风险。因此完整训练选择“降低但保留”共享EA梯度，而不是在未经验证的情况下从很早就完全切断。

## 3. 精确的新策略

令 \(u\) 为已经完成的optimizer updates，\(\rho\) 仅控制EA回到共享tokens/features的梯度：

\[
\rho(u)=
\begin{cases}
1,&u<2000\\
1-0.99\min((u-2000)/500,1),&u\ge2000.
\end{cases}
\]

| 阶段 | WAM参数更新 | 共享EA梯度 | heads监督 | 正常Read/Write及Action+Video反向 |
|---|---|---|---|---|
| 前2,000次更新 | 跳过 | 原强度 | 原强度 | 保留 |
| 2,000→2,500更新 | 正常 | 线性从100%降到1% | 原强度 | 保留 |
| 2,500更新后 | 正常 | 保留1% | 原强度 | 保留 |

仅在送往heads的支路上使用：

```python
# 数学原型，不是当前已有的生产API。
head_input = x.detach() + rho * (x - x.detach())
```

前向仍等于x，heads参数的局部梯度不因该算子缩小；回到x的EA梯度乘rho。tokens和first-frame features两条输入都要覆盖，三tap和deep supervision同样覆盖。传给后续tap和Write的仍是原tokens，Action/Video对共享模块的梯度不缩放。

完整模型的总loss数值、EA heads监督权重不变；记录的EA loss不必因rho降低而立即降低。需额外记录真实rho，不能把标量loss当作共享优化强度。

现有硬heads-only开关等价于联合阶段rho=0；原逻辑等价于rho=1。新增实现应复用同一输入路由，保留这两个旧行为，禁止同时误开两种互相矛盾的配置。没有新增参数、额外模型前向或额外训练样本。推理时直接走原路径，不引入该训练算子的开销；真实训练显存/速度仍需实测，不能说完全零成本。

## 4. 为什么选1%和500更新，而不是继续扫超参数

本run12k的诊断中，真实加权EA/Action梯度比在Clean共享Refine/Plan约112–236倍。单看这批样本，将共享EA梯度缩小100倍后会落在约1.12–2.36倍的量级，因而0.01是一个有数据依据、值得预注册尝试的起点。

但这不是32k正式batch32的当前比例，也不是Adam更新比。诊断来自六条episode的重叠B1样本，Randomized比例更高且不参与Clean训练；不应声称1%就是最优，或Action/EA必须严格1:1。

500更新的线性过渡用于减少一次性改变优化侧重，保留旧Adam状态的自然过渡；500是固定工程选择，不是已有实验发现的最优值。本轮不做0/.001/.01/.1的大网格，也不一起改学习率。

GradNorm等工作支持关注共享多任务梯度分配，但并未证明本项目的固定1%策略有效；这里是根据本项目证据提出的可证伪候选。[原论文](https://proceedings.mlr.press/v80/chen18a.html)

## 5. 完整训练配方

保持已获提升的RoboTwin配方：

- 相同Foundation初始化、相同Aloha50 Clean2500与549,775窗口；不换为Sim专用权重，不并入Randomized数据。
- 四卡A100 80GB；每卡32、accumulation2、global256。
- 10 epoch，42,960 microsteps / 21,480 optimizer updates；其中WAM实际更新19,480次，Sidecar更新21,480次。
- WAM LR1e-4、Sidecar5e-5、原cosine和5%全局LR warmup；不额外加WAM LR ramp。
- Action=1、Video=1、EA总scale=.5，Ground/Relation/Plan/deep内部权重不变；ColorJitter保持开启。
- 三处Write位置/初始化、15 tokens、32动作/9视频帧、绝对EEF、归一化、mask、ZeRO2和checkpointing均保持。
- 新的独立输出根，例如 `outputs/sidecar/aloha50_ea_grad_anneal_10epochs_<UTC>`，不写入原run或当前heads-only分支。

**此处保留2K而非同时改500 warmup，是为对照已取得成绩的RoboTwin方案，只改变共享EA梯度策略。** 当前global512、500 warmup的跨本体midtrain属于另一实验。若用户决定这里也使用500，需明确它是同时改变预热长度的新recipe，不能将全部收益归于梯度策略；最好再补同500的控制。

沿用每4,000微步保存，保留关键12k/24k/32k/final权重；必要的warmup终点4k完整状态也应固定，以便后续从同一预热起点分叉。无需为每个观察点重复存一套110GB optimizer，常规恢复状态保持现有策略。

按原实验约80小时/10epoch，四卡同吞吐下新完整实验预计约80小时，加载、保存和资源争用另计，不承诺新算子加速。确认后可在当前heads-only分支正常完成、四卡实际释放后接续；当前两个实验和正在进行的最终ckpt评估不停止、不替换。

## 6. 三个实验各自回答什么

| 实验 | 目的 |
|---|---|
| 原full-backward最终42960 | 现有最佳recipe完整训练的效果，与32k比较是否继续改善 |
| 当前40k→42960 heads-only | 已成型表征在晚期去掉共享EA梯度是否有收益；不能替代全程策略 |
| 新完整共享EA梯度退火 | 较早改变训练侧重、又保留弱语义约束能否进一步提高控制 |

新实验的主对照是原full-backward的相同32k和42960预算，不只挑各自最好点相减。新完整soft与当前晚期hard不只差rho，还差生效时机；两者比较不能证明soft优于hard本身。

优先复用完整100条件×30条协议比较32k和最终权重；保持near.1、cache、10去噪步、执行32步、预算及成功判据一致。筛选配置使用开发场景，关键最终对比用独立或预冻结的accepted seed/指令/噪声；不把同seed整数当作严格场景配对。

语义保护栏复用现有Ground峰值命中、Contact/Moving F1、8/16/32方向cos；不新造无GT的progress accuracy。重点看Random是否提高、Clean是否保持、收益是否分布在多任务，以及是否以语义塌缩换取短期拟合。训练loss、gate或真实write-r都不替代闭环SR。

匹配Vanilla仍是论文必要证据：相同Foundation、Clean2500、增强、样本/更新预算、统计与部署参数，并披露WAM实际更新次数。现有十任务baseline与官方结果不能完全替代；不要为多扫一个Sidecar超参数挤掉它的算力。

## 7. 实施前的验收与停止条件

1. 默认rho=1必须保持当前逻辑；已有heads-only分支仍按rho=0工作，不能改变正在运行的进程。
2. FP32/BF16及checkpointed模型中前向一致；heads局部梯度不缩放、共享EA梯度按rho缩放，Action/Video梯度保留。
3. 验证2K→2.5K边界、accumulation的optimizer计数、保存/恢复和日志rho；不得按microstep误计。
4. 实际模型短测有限loss/真实参数更新/保存恢复通过后才提交完整训练。数学toy不代替这些检查。
5. 使用少量真实训练批次复核EA侧重；若已无该问题或共享EA正是已验证收益来源，应降低此候选优先级。
6. 若关系指标明显塌缩且动作/闭环不改善，保留失败结果，停止该方向；不同时加动态gate、Plan新loss或新结构去补救。1%的语义约束并不构成关系可靠性保证。

本轮已完成8项CPU数学检查：float32/BF16下，rho=0/.01/.1/1的前向和heads梯度一致、共享梯度按比例变化；并核对计数示例。证据 `outputs/sidecar_full_train_proposal_20261026/gradient_scaling_math_check.json`。**这仅验证算子原理，尚未实现到WAM、没有新的GPU或成功率结果。**

## 实施授权边界（历史提案；2026-10-26已授权）

用户已授权的是现有40k heads-only短续训。本次是新的软路由训练策略，需要在具体方案确认后再修改核心逻辑与启动新80小时实验；依据项目AGENTS.md第2条。旧方案、数据和结果均保留。

后续状态：用户已明确要求按本方案接续实验。实现、工程测试与独立启动记录见[正式实验记录](SIDECAR_EA_GRADIENT_ANNEAL_FULL_TRAIN_20261026.md)；上述待确认说明保留为提出方案时的历史状态。
