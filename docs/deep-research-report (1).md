# 五组短训之后：ReLAC / Sidecar 下一步最可靠的训练与结构优化路线

## 核心结论

结合你这次 **A/B/C/D/E 五组从同一 2K warmup 起点继续 1K optimizer updates 的短训结果**、此前真实 Write Ratio / Full-vs-Off 诊断，以及近期 WAM 和条件注入方法，我现在的判断比之前更明确：

> **当前不应该继续围绕“让 Sidecar 写得更猛”做优化，而应该优先解决 WAM 解冻时的优化冲击，并保留关系表征持续学习。**

五组里，**C 是目前唯一一个已经发现“明确、可解释、而且真实存在”的优化问题并把它解决掉的方案**：WAM 在 2K 后开始更新时，原方案 A 出现了第二次更新 Action loss 突增、全局梯度范数达到约 12.67、100 次更新中 13 次触发 clipping；C 用短 WAM LR ramp 后，峰值降到约 0.94，前 1K 甚至一次都没有触发 clip。与此同时，C 的后段 Action loss 高于 A，并不能简单解释成 C 更差，因为 C 在这 1K 中的 WAM LR 积分只有 A 的 86.37%，确实存在适配进度落后的混杂因素。

而另外三种修改目前没有出现类似 C 这样清楚的正面机制证据：

| 方案 | 我现在的判断 | 是否值得直接长训 |
|---|---|---|
| **A 原方案** | 已经是强 Control，且完整训练已有正向 SR | 必须保留 |
| **B EA shared grad 1%** | Plan / Relation 明显平台化，Action 后段也没有收益 | **不建议** |
| **C WAM LR ramp** | 明确消除解冻冲击，短期存在适配滞后 | **最高优先级** |
| **D Write LR ×2** | Write/beta 参数移动更多，但没有 Loss 收益，实际 Write 反而可能更弱 | **不建议** |
| **E Write token backward ×2** | 目前接近 A，没有形成持续领先 | 完成诊断即可，不优先长训 |

你们当前的整体方向本身不需要推翻。OpenWAM 的原始研究结论就强调，World–Action 协同需要明确的信息流和独立的动作能力，并且其主要优势之一体现在 OOD 泛化；最近 LiLa-WAM、Magic-W0、AffordanceWAM 也都在往“紧凑、任务相关、由预测与动作共同塑造的表示”发展。citeturn3view5turn7academia1turn7academia0turn7academia2 这和你们用 Grounding / Relation / Future Motion 来塑造 Relational Latents、再写回 Action 的主思路是同向的。你们之前的实现审查也已经指出：当前最缺的不是更多网络容量，而是把关系监督转化成可归因的控制收益。fileciteturn0file0

所以我建议下一步不要再开五六条互相纠缠的路线，而是形成：

\[
\boxed{
\text{C 主线延长}
\quad+\quad
\text{最多一个新的低风险结构候选}
}
\]

其中，我对新的结构候选最推荐的是 **Scalar Write Gate → Per-Channel LayerScale Gate**，而不是再调 Write LR、EA gradient 或直接增大 Gate。

---

## 五组短训实际上告诉了我们什么

### A 已经不是“失败 baseline”，而是需要被小心超过的强方案

这一点非常重要。

你们最早看到旧 checkpoint 的真实写回只有 \(10^{-6}\sim10^{-5}\)，当时我们怀疑 Sidecar 基本没真正进入 Action。后来新的 WAM-delayed 2K warmup 已经把实际 Write Ratio 提高到约：

\[
0.61\%,\quad 0.65\%,\quad 0.90\%
\]

而且 Full 相比 Write-Off：

\[
L_{\text{action}}
\downarrow 5.35\% \text{ Clean},
\qquad
\downarrow 5.96\% \text{ Randomized}
\]

说明现在的训练配方已经跨过了最关键的一道坎：

> **Action 模型开始利用 Sidecar。**

所以现在任何修改都必须尊重这一点。

我们已经不应该把问题定义成：

> “怎样才能强迫 Sidecar 进入 Action？”

而应该定义成：

> **怎样在保持当前已经有效的关系→动作通路的情况下，让 WAM 和 Sidecar 更平滑地共同适配？**

这也是为什么 D 和 E 对我的吸引力明显下降。

### B 基本告诉我们：现在不适合把 EA 共享梯度砍到很低

B 最有价值的并不是 Action loss 高了多少，而是它提供了一个非常清楚的机制信号。

Plan 相对 A 的差距：

\[
0.08\%
\rightarrow
3.48\%
\rightarrow
8.56\%
\rightarrow
12.34\%
\rightarrow
19.17\%
\rightarrow
26.63\%
\]

而最后 Relation 也高约 40%。

与此同时，Heads 本身仍然继续更新。

这说明：

\[
\boxed{
\text{Heads can still learn}
\neq
\text{Shared relational representation is still being shaped well}
}
\]

也就是说，之前“Heads 保持 100% supervision、Shared gradient 降到 1%”这个想法虽然计算图上很漂亮，但实际看来很可能**削弱了我们真正希望保留的 relational latent learning**。

这和近期 WAM 的方向也比较一致。LiLa-WAM 的核心就是让 compact latent 同时受到未来状态预测和 Action Generation 塑造；Magic-W0 也强调 structured world representation 与 action generation 的持续耦合，而不是先学完一个辅助预测器再让 Action 自己接管。citeturn7academia1turn7academia0 AffordanceWAM 同样把任务相关的 interaction representation 与 action 放在联合学习框架中。citeturn7academia2

所以我现在**明确不建议**把 B 的 1% 版本扩成完整 10 Epoch。

以后确实仍可以研究 10%、20% 或自适应梯度平衡，但那属于论文之后的优化空间，而不是现在冲刺阶段最可靠的方案。

### D 说明“写得更快”并不等于“写得更好”

D 的结果其实非常有价值。

Write 参数位移：

\[
1.58\times A
\]

Beta 参数位移：

\[
1.91\times A
\]

但是：

- 后段 Action loss 与 A 基本重合；
- 固定 FM 没改善；
- rotation 还变差；
- beta 更快往 0 方向走；
- 实际 Write 并没有因此更强。

所以我们基本可以排除一个简单解释：

> “Sidecar 效果还不够，只是因为 Write module 学得太慢。”

如果这是主要问题，D 至少应该在某些稳定 Action 指标上开始持续领先。

没有。

这也符合已有条件注入网络的经验：ControlNet 使用零初始化输出连接来让条件逐步进入强预训练模型；Flamingo 用零初始化 gated cross-attention，使新模态逐渐进入 frozen LM；DiT 的 adaLN-Zero 同样把残差分支初始化成接近 identity，并且其效果优于普通 cross-attention 条件化。citeturn3view2turn4view1turn4view2

这些设计背后的共同原则不是“尽快让新分支变大”，而是：

\[
\boxed{
\text{Preserve pretrained function}
+
\text{learn conditional residual progressively}
}
\]

所以当前不应该继续：

- Gate ×5；
- Write LR ×2/×4；
- 强制 Write norm；
- 给 Write 加一个额外必须变大的 loss。

### E 说明单纯加大 Action→Sidecar 梯度也未必是瓶颈

E 的前 700 updates 与 A 非常接近，没有持续领先趋势。

这告诉我们一个重要事实：

> **当前 Action 对 Sidecar 的反向信号可能不是“数量不足”，而是整个 WAM–Sidecar 联合优化轨迹更重要。**

这也解释了为什么 C 的影响反而大。

C 并没有改 Action→Sidecar 梯度，它只是处理了 WAM 从冻结切到更新时的 optimization shock，却立刻改变了训练动力学。

所以目前最值得动的是：

\[
\boxed{\text{Backbone adaptation dynamics}}
\]

而不是：

\[
\text{arbitrarily amplify Sidecar gradients}.
\]

---

## 下一步我建议立刻启动的主实验

### 不重新从 2K 开一条复杂新路线，直接延长 C

这是我最推荐的立即行动。

你们已经有：

\[
2K\rightarrow3K
\]

的 C 完整 optimizer 状态。

现在直接：

\[
\boxed{
C:\quad3K\rightarrow5K
}
\]

再增加 **2,000 optimizer updates**。

不修改任何东西：

- 不改 EA；
- 不改 Write；
- 不改 Gate；
- 不改 optimizer；
- 不重新 ramp；
- 不改数据；
- 不改 scheduler；
- 不把任何其他 Trick 叠进去。

因为 C 的 300-step ramp 已经结束，3K 以后它实际上已经回到了原始 LR schedule 上。

接下来 2K 的真正问题只有一个：

> **早期少走的 WAM 适配进度，在进入正常 LR 后是否会逐步追上 A？**

这个实验的信息量极高，而且成本很低。

### 为什么我不建议现在直接重新完整训练一个 C'

你现在已经拥有一个有效的 C 状态。

它唯一明显的负面是：

\[
\text{Action}_{C}
\approx
\text{Action}_{A}+8.7\%
\]

在最后 250 updates。

但 C 前 1K 的 WAM LR 积分只有 A 的 86.37%。

而且 C 的 Action loss 差距曾经从中间接近 +20% 回落到后段约 +9%。

这还远远没有达到“C 已经被证明长期更差”的程度。

根据你们此前 10 Epoch 训练规模，完整训练远大于这 1K 窗口；因此早期 LR ramp 带来的适配差距在完整训练中的相对占比会迅速降低。这只是优化直觉，不是保证 C 最终追平。

所以：

**先让已经付过训练成本的 C 再跑 2K，比从头再开一个 200-step / 150-step ramp 更合理。**

### 同时不必重新跑 A

A 就是你们原始已经完成 10 Epoch 的主实验。

只要存在匹配的：

- 4K checkpoint；
- 5K checkpoint；

直接拿历史 A 做比较。

如果没有 4K/5K 权重，才需要从 A 的 3K 状态短续训。

不要为了“公平”重新烧一遍完整 A。

### C 延长期间需要自动保存什么

建议至少保存：

| optimizer update | 用途 |
|---:|---|
| 3.5K | 看 Action loss 是否立即继续追赶 |
| 4K | 中间判断 |
| 5K | 第一次正式短筛 |

重点比较的不是单个终点，而是：

\[
\Delta L_{\rm action}(t)
=
\frac{L_C(t)-L_A(t)}{L_A(t)}
\]

是否呈现：

\[
+8.7\%
\rightarrow +5\%
\rightarrow +3\%
\rightarrow 0
\]

这样的趋势。

如果有，就基本证明之前看到的是 **adaptation lag**。

如果一直稳定在 +8%～+10%，那 C 虽然稳定，但可能确实学得更保守。

---

## 我最推荐增加的一个小结构 Trick

如果你们有额外 GPU，我认为可以并行增加**唯一一个新结构分支 F**：

# Per-Channel Write Gate

你们目前每个 Stage 是一个标量：

\[
H_A'
=
H_A
+
\beta_l P_l
\]

其中：

\[
\beta_l\in\mathbb R.
\]

也就是说 Block 7 的整个 1024-d Action Hidden：

\[
[P_1,P_2,\ldots,P_{1024}]
\]

全部只能乘同一个数字。

我认为这是目前 Sidecar 中一个非常值得低成本改善的地方。

### 改成向量 Gate

改为：

\[
\boxed{
H_A'
=
H_A
+
\boldsymbol{\beta}_l
\odot P_l
}
\]

其中：

\[
\boldsymbol{\beta}_l
\in\mathbb R^{1024}
\]

或者如果 Write projection 的 action dimension 不同，就用对应 dimension。

新增参数只有：

\[
3\times1024=3072
\]

几乎可以忽略。

### 为什么这个比 Dynamic Gate 更可靠？

因为它可以做到**严格 function-preserving initialization**。

假设原 2K checkpoint：

\[
\beta_7=-0.02.
\]

初始化：

\[
\boldsymbol{\beta}_7
=
[-0.02,-0.02,\ldots,-0.02].
\]

那么转换后的第一帧前向：

\[
H_A^{new}=H_A^{old}
\]

几乎逐元素完全一致。

也就是说：

> **我们不是重新随机初始化一个 Action Bridge，而是在当前已经有效的函数基础上，给 Gate 增加一点自由度。**

这在赶 deadline 时非常重要。

### 为什么它可能适合我们？

现在一个标量 β 同时控制：

- 位置相关维度；
- 姿态相关维度；
- Gripper 相关维度；
- Flow Matching 不同 latent directions；
- 可能与 world-action joint representation 相关的其他 channel。

但 Relational Latents 提供的信号并不一定应该均匀影响所有 hidden dimensions。

例如：

Contact

可能更应该改变与 Gripper / grasp phase 有关的 hidden subspace。

---

Object–Goal relation

更可能影响 transport / placement 相关 subspace。

---

Future direction

更可能影响 EEF position 的 movement subspace。

所以一个 Scalar Gate 本质上在做：

\[
\text{all channels accept relational information equally}.
\]

Per-channel Gate 可以学：

\[
\text{which hidden subspaces should actually use relation information}.
\]

这个思路在 Transformer 优化中有很好的先例。CaiT 的 LayerScale 就是在 residual branch 后加入**可学习的逐 channel diagonal scaling**，作者发现相比单 scalar residual scaling，它提供了更多优化自由度，并改善了深 Transformer 的训练；其核心形式与你们这里非常接近。citeturn12view0 DiT 的 adaLN-Zero 也使用 dimension-wise residual scaling 参数，并发现这种 identity-preserving conditioning 对训练质量非常重要。citeturn4view2

但我要强调：

**这只是非常合理的候选，不是已有证据保证能提高 RoboTwin SR。**

所以它适合做一个 1K screen，而不适合直接替换 C 去跑完整 10 Epoch。

### F 的具体设计

直接从共同 2K checkpoint：

```python
# current
beta = nn.Parameter(torch.tensor(beta_ckpt))

residual = beta * p
```

改成：

```python
# proposed
beta_vec = nn.Parameter(
    torch.full((action_dim,), beta_ckpt)
)

residual = beta_vec[None, None, :] * p
```

只改这一项。

其他全部使用 A 配方。

或者更保守：

\[
F=C+\text{vector gate}
\]

但我**不推荐第一轮就把 C 和 Vector Gate 叠在一起**。

因为我们现在还不知道 C 是否最终有效。

为了归因，最好：

\[
F=A+\text{Vector Gate}.
\]

跑 1K。

如果 F 明确有收益，最终再考虑是否与 C 合并。

---

## 还有几个值得马上做、但不值得重新开完整训练的工程检查

### Gate 参数是否被 AdamW weight decay

这个我现在建议**马上查代码**。

因为 D 有一个很值得注意的现象：

\[
LR_{\rm Write}\times2
\]

以后，三个负 β 都更快向 0 移动。

如果 β 位于 AdamW weight-decay 参数组中，那么 weight decay 更新项包含：

\[
-\eta\lambda\beta.
\]

LR 翻倍也会把这个 shrinkage 翻倍。

所以 D 中：

> “更高 Write LR → Gate 更快趋近 0”

有可能不仅来自 Action gradient，也可能有 optimizer regularization 的贡献。

对于这种可学习的 residual scale，我个人更倾向于：

\[
\boxed{\text{weight decay}=0}
\]

类似：

- LayerNorm scale/bias；
- bias；
- Gate scalars / LayerScale；
- 某些 learned embeddings。

但在没有确认当前 optimizer param group 之前，不应该改。

因此下一步只做审计：

```text
beta / alpha
↓
属于哪个 AdamW param group？
↓
weight_decay = ?
```

如果确实有 decay，把它记成一个 **F2 1K candidate**，而不是直接悄悄加入 C。

### 保持当前零残差 / 小残差初始化，不改 Gate=1

结合现在的数据，我已经不建议优先测试：

\[
\beta=1,\quad W_{\rm out}=0.
\]

此前这个方案的优化理论是合理的，但你们新的 warmup 已经证明当前 Gate 方案能产生：

\[
0.6\%\sim0.9\%
\]

真实写入和 5%～6% 的 Full-vs-Off Action FM improvement。

因此 Gate 初始化已经不是当前最高优先级瓶颈。

更重要的是，零/小 residual 初始化并不是一个奇怪设计：ControlNet 的 zero convolution、Flamingo 的 zero-initialized gated cross-attention 和 DiT 的 adaLN-Zero 都是在强预训练主干旁加入新条件时保留 identity 起点。citeturn3view2turn4view0turn4view2

所以：

\[
\boxed{\text{现在不要重做 initialization}}
\]

### 可以快速测试 Temporal Ensemble，但把它当部署 Trick

如果 RoboTwin/OpenWAM 的部署方式允许每个控制周期重新预测 32-step action chunk，那么还有一个**完全不需要重训练**的东西值得小测：

\[
\text{overlapping action-chunk temporal ensemble}.
\]

ACT 就通过更频繁地产生重叠 action chunks，然后对同一时间点来自不同预测 chunk 的动作做时间加权平均，以改善动作平滑性和闭环稳定性。citeturn9view0

例如过去三次预测：

\[
A^{t-2},A^{t-1},A^t
\]

对当前目标动作：

\[
a_t=
w_0 A^t_0+
w_1 A^{t-1}_1+
w_2 A^{t-2}_2.
\]

可以用：

\[
w_i\propto e^{-ki}.
\]

它对 Sidecar 本身没有任何新增训练影响，但有机会减少 Flow Matching sampling noise。

不过它有三个前提：

1. 当前 deployment 确实有 overlapping chunks；
2. 动作是同一绝对/相对控制语义；
3. 不会破坏需要快速切换的 contact/gripper event。

因此只能做小规模 A/B。

而且如果它进入论文主评测，必须同时应用于 Vanilla 和 ReLAC，不能只给我们的方法使用。

---

## 我明确不建议现在做的事情

### 不要上 PCGrad

PCGrad 的核心针对的是：

\[
g_i^\top g_j <0
\]

即负 cosine 的任务梯度冲突，通过把冲突梯度投影到另一个任务梯度的法平面上来处理。citeturn3view3

你们目前没有新的证据证明：

\[
g_{\rm Action}
\]

和

\[
g_{\rm EA}
\]

在关键共享参数上存在稳定的负 cosine。

旧诊断甚至整体接近：

\[
+0.031.
\]

所以现在上 PCGrad 属于**没有诊断依据地增加训练复杂度**。

### 不要现在上 GradNorm

GradNorm 的确专门用于根据 task learning rate 和梯度范数动态调整多任务 loss 权重。citeturn3view4

理论上它非常适合：

\[
L=
L_{\rm video}
+
L_{\rm action}
+
L_{\rm EA}.
\]

但是现在有两个问题。

第一，你们的 EA 不是普通并列任务，它是论文里希望保留的物理 inductive bias。

GradNorm 自动把它调小或调大，可能改变论文方法本身。

第二，B 已经显示关系学习对 shared EA gradient 很敏感。

在 deadline 前引入一个动态 loss balancing controller，会制造比解决的问题更多的新变量。

所以我建议放到 supplementary / future work，不作为最终模型抢救方案。

### 不要增加更多 Relation Head

当前已经有：

- Ground；
- Relation；
- Summary；
- Plan；
- Deep supervision。

最近的 structured WAM 工作确实支持 task-relevant geometry / affordance / future representations。citeturn7academia0turn7academia2

但你们现在的主要矛盾并不是“信息不够”。

B 甚至告诉我们现有 Plan / Relation 都还没有完全充分学好。

因此：

\[
\text{new heads}
\]

只会进一步增加多任务优化难度。

### 不要突然增加更多 Write Layers

Flamingo 的实验确实表明，在其 VLM setting 中，更频繁插入 gated cross-attention 一般性能更好，但计算成本也更大。citeturn4view1

但那不能直接推出：

> “我们从三个 Write 增到六个就会提高机器人控制。”

你们目前 Block 7 / 15 / 23 已经有三层 injection，而且真实写回已经有效。

此时加层会同时改变：

- 参数量；
- 梯度路径；
- Write depth；
- 计算成本；
- Gate 数量。

对 CVPR deadline 太冒险。

---

## 下一轮实验我会这样排

如果我是现在负责这套实验，我会把 GPU 任务安排成下面这样。

### 主线继续 C

**立即：**

\[
C_{3K}\rightarrow C_{5K}
\]

不做任何新修改。

同时 E 跑完原定 1K，不中断。

如果有空闲 GPU，再开：

\[
F=A+\text{Per-Channel Write Gate}
\]

从相同 2K checkpoint 跑 1K。

形成：

| Run | 目的 | 训练量 |
|---|---|---:|
| A | 已有 Control | 已完成 |
| C-extension | 判断稳定化收益是否追上 Action adaptation | +2K |
| E | 收尾，不扩大 | 剩余约 280 |
| F-vector-gate | 测试低风险 Action interface flexibility | 1K |

不要再开 B2、D2、E2。

### C 到 4K 时

只做离线自动评估，不停训练：

\[
\text{A@4K vs C@4K}.
\]

至少检查：

- Action loss；
- Video loss；
- Plan；
- Relation；
- actual Write Ratio；
- Full-vs-Off Action FM；
- XYZ；
- Rotation；
- Gripper。

如果 C 的 Action 差距：

\[
8.7\%\rightarrow 4\%\sim5\%
\]

已经明显收窄，这是很好的信号。

继续训练。

如果 C Action 仍约 +10%，但 Full-vs-Off、XYZ 或 Randomized 离线指标更好，也继续到 5K。

**不要单独根据 train Action loss 杀掉 C。**

### C 到 5K 时

做一次真正有决策价值的 paired closed-loop screen。

使用你们已经固定过的三个代表任务：

\[
3\text{ tasks}
\times
2\text{ domains}
\times
10\text{ seeds}
=
60
\]

每个 A / C 使用完全相同 seed。

不要为了筛选跑完整 50 任务。

重点记录：

\[
\text{A success vs C success}
\]

以及：

- A success / C fail；
- A fail / C success；
- both success；
- both fail。

如果 C：

- Randomized 整体不下降；
- paired wins ≥ losses；
- Clean 没有明显损失；
- Action / relation 指标没有异常；

我会**直接把 C 继续跑到完整 10 Epoch**。

即使 5K train Action loss仍稍高于 A，也会继续。

因为论文指标最终是 closed-loop SR，不是 training MSE。

### 什么情况下直接停止 C？

我会用比较严格的 Stop 条件：

\[
\boxed{
\begin{aligned}
&\text{Action gap持续不收敛}\\
+&\text{paired closed-loop明显更差}\\
+&\text{没有 Randomized 收益}
\end{aligned}}
\]

三者同时出现。

而不是因为一个 Action loss 高 5% 就停止。

---

## 对 CVPR 最重要的最终路线

我认为你们现在已经不应该把主要精力放在继续探索十种 Trick 上。

你们已经拥有一个很重要的结果：

> **原 2K WAM-delayed warmup 的完整模型已经能比 baseline 平均提高约 2 个点，并且 Randomized 的改善尤其明显。**

接下来最理想的论文路径其实是：

**Matched Vanilla**

原始 WAM

**ReLAC-A**

现有有效方案

**ReLAC-C**

更平滑的 WAM adaptation

**No-EA Adapter**

同容量但没有 relational supervision

**Write-Off**

证明 relational latent 真正参与 Action

你们的方法之所以具有研究价值，不应该最终建立在：

> “我们把 LR ramp 调得特别好。”

而应该是：

\[
\boxed{
\text{Simulation-supervised Relational Latents}
\rightarrow
\text{Action Conditioning}
\rightarrow
\text{Better OOD Control}
}
\]

OpenWAM 本身认为显式 world-to-action information flow 对 WAM 的协同很重要；近期 LiLa-WAM 强调由 future prediction 和 action generation 共同塑造 compact latent，Magic-W0 强调 structured physical representations 与 action 的 layer-aligned interaction，AffordanceWAM 则进一步把 task-relevant interaction representations 纳入 world-action joint learning。citeturn3view5turn7academia1turn7academia0turn7academia2

这说明你们现在最应该保护的是：

**Relational Latents 确实持续保持物理监督，同时被 Action 真正利用。**

而五组短训已经告诉我们：

\[
\boxed{
\text{B 太早削弱 relation learning}
}
\]

\[
\boxed{
\text{D / E 单纯强化 Write 没解决问题}
}
\]

\[
\boxed{
\text{C 发现了一个真实的 backbone-unfreeze instability}
}
\]

所以我的最终推荐非常明确：

> **现在立刻续 C 到 5K，不改其他任何配置；E 收尾；有富余 GPU 才开唯一一个 F：function-preserving 的 Per-Channel Write Gate 1K screen。**

如果 C 在 5K 的 paired closed-loop 至少保持现有性能并在 Randomized 上表现更好，就让 C 直接跑完整 10 Epoch。

如果 F 的 1K 只是和 A 持平，就立即停止，不继续探索动态 Gate。

如果 F 明确改善 Full-vs-Off Action、XYZ/rotation，而且没有损害 Relation/Plan，再考虑后续是否将 **C + Vector Gate** 作为最后一个组合候选；**在此之前绝对不要叠加。**

我认为这是现在最符合你们现有证据、训练成本和 CVPR 时间约束的一条路线：**先把已经证明存在的优化问题解决干净，再用一个几乎不改变现有函数的轻量 Gate 扩展作为唯一结构候选，而不是继续通过更大的梯度、更低的 EA 权重或更强的 Write 去“硬推”Sidecar。**