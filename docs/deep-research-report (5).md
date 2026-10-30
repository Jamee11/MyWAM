# ReLAC 下一阶段 C2R 实验决策：从“强增强 + WAM-100 Ramp”走向显式 Action Invariance

## 核心判断

基于你现在给出的 **A/B/C/D/E 五组短训、A/C 4K 延长结果，以及正在运行的“PatchWAM-style 强增强 + 100-step WAM LR Ramp”**，再结合我重新检查的 RoboTwin 2.0、PatchWAM、PICA-WAM、PrismWAM、OA-WAM、Perturbot 和 AttenA+ 等近期工作，我的判断已经比较明确：

> **当前正在跑的“强增强 + 100-step Ramp”不要停，至少跑到 4K 做固定输入和小规模闭环。下一条最值得投入 GPU 的新实验，不是继续放大 Write、削弱 EA 梯度或增大 Batch，而是在当前方案上增加“Clean → Augmented 的 Action Representation Consistency / Cross-Appearance Alignment”。**

也就是把现在的：

\[
\text{Strong Augmentation}
+\text{2K Sidecar Warmup}
+\text{100-step WAM Ramp}
\]

升级成：

\[
\boxed{
\text{Strong Augmentation}
+\text{2K Warmup}
+\text{100-step Ramp}
+\text{Action-only Cross-Appearance Alignment}
}
\]

我认为这是目前**证据最强、和你们现有问题最匹配、结构改动又最小**的一条路线。

原因不是单纯因为 PICA-WAM 最近效果好，而是它恰好解释了你们现在出现的现象：

> 强增强让模型看到了更多视觉变化，但 WAM 的 Video Objective 仍要求网络忠实编码这些外观变化；Action Tokens 又不断读取 Video Features，所以“看到增强”不等于“Action 表征学会忽略增强”。

PICA-WAM 正是从这个问题出发：augmentation-only 并不会显式要求同一物理状态的 clean/perturbed 输入产生相同 Action Representation；它因此增加 clean teacher 与 perturbed student 的 Action Hidden 对齐。citeturn17view0turn17view4

这与 ReLAC 的研究逻辑非常吻合：你们现在已经有一套**关系表征 → Action Write** 的通道，下一步不应该一味让 Write 更大，而应该确保 **Write 进去的是对背景、光照、纹理等扰动不敏感、同时保持任务关系和动作相关的信息**。

我的优先级是：

| 优先级 | 方案 | 建议 |
|---|---|---|
| **P0** | 当前 Strong Aug + WAM Ramp100 | **继续跑到 4K，不停** |
| **P0** | Clean↔Aug paired invariance diagnostic | **4K 必做** |
| **P0** | **Action-only Cross-Appearance Alignment** | **下一条主训练分支** |
| P1 | Sidecar Relational Latent Consistency | Action alignment 有效后再考虑 |
| P1 | 颜色语义保护型 augmentation | 根据分任务结果决定 |
| P2 | Distractor-specific consistency | 很有潜力，但工程风险高于 PICA |
| P2 | Action criticality / AttenA+ weighting | 解决尾部动作问题，不是 C2R 第一矛盾 |
| 暂停 | B 的 EA 1% annealing | 已有负面语义证据 |
| 暂停 | D Write LR×2 | 已经证伪“参数动得更多=写得更好” |
| 暂停 | E Write grad×2 | 没形成完整动作优势 |
| 暂停 | Batch 512 | 暂无证据能解决 C2R 主问题 |

## 当前“增强 + Ramp100”应该怎样解释

### 训练 Loss 高于 A 现在并不是主要负面证据

你现在的数据是：

| 区间 | A Action | Aug+Ramp100 | 相对 A |
|---|---:|---:|---:|
| 1–100 | 0.04344 | 0.04433 | +2.1% |
| Warmup 1901–2000 | 0.01591 | 0.01832 | +15.2% |
| 解冻 2001–2100 | 0.01243 | 0.01060 | **−14.8%** |
| 2101–2200 | 0.00487 | 0.00638 | +30.9% |
| 2501–2600 | 0.00303 | 0.00366 | +20.6% |

最近 100 updates：

\[
L_\text{action}:+20.6\%,\qquad
L_\text{video}:+12.3\%,\qquad
L_\text{EA}:+2.5\%.
\]

我不会因为这些数字停掉当前实验。

因为如果你们实现接近 PatchWAM C2R augmentation，那么它对 80% 样本使用明显更强的 brightness/contrast/saturation、Hue、Gamma、Exposure、Gaussian noise/blur，以及 AdaIN 和 FFT 扰动，而且同一视角的当前帧和未来帧共享变换；这会直接改变 Video Prediction 的训练目标难度。PatchWAM 在 C2R 配置中还特意不使用 crop/rotation，以避免改变物体几何位置。citeturn18view3

因此：

\[
L_{\rm video}^{\rm Aug}>L_{\rm video}^{A}
\]

本身并不奇怪。

Action Loss 也不能再作为严格 apples-to-apples 的主排名指标，因为两组训练时面对的是不同难度的视觉输入。

**真正公平的是：在同一个关闭 augmentation 的固定诊断集上测两者。**

也就是说，从现在开始，我建议把：

> training action loss

从“模型优劣指标”降级为：

> optimization health 指标。

只要它在稳定下降、不发散、不出现大规模裁剪，它就已经完成了训练监控的主要任务。

### Ramp100 的表现其实比我预期更合理

A 解冻后前 100 updates：

\[
g_{\max}=12.67,\qquad \#clip=13
\]

当前 Ramp100：

\[
g_{\max}=1.08,\qquad \#clip=4
\]

而 C 的 Ramp300：

\[
g_{\max}=0.94,\qquad \#clip=0.
\]

这三组放在一起看很有意思。

C300 已经告诉我们：

> **把所有解冻冲击完全压掉，并不等于最后动作最好。**

它虽然 1K 内完全没有 clipping，但有明显适配滞后；到 4K 虽然 Random 全 32-step XYZ 略优于 A：

\[
20.301\ {\rm mm}\quad vs.\quad20.918\ {\rm mm},
\]

但 Clean 仍差，而且固定 Action FM 相对 A 的差距扩大到了 +13.16%。

因此，我反而不希望现在 Ramp100 也追求“0 次 clipping”。

**1.08 的 peak、4/100 的 clip，在我看来很可能比 C300 是更好的 stability–adaptation compromise。**

它把 catastrophic jump：

\[
12.67\rightarrow1.08
\]

几乎压掉了，同时又没有像 300-step ramp 那样长期压低 WAM 的有效步长。

所以除非 4K 诊断显示明显退化，**我会保留 100-step Ramp，不回到 300-step。**

### 还有一个非常关键的混杂变量

当前新实验相对 A 一次改变了两件事：

\[
\text{augmentation}+\text{Ramp100}.
\]

所以不能说：

> “强增强让解冻稳定了。”

从 C300 的结果看，更合理的推测是：

> **解冻稳定改善主要来自 Ramp；训练 Loss 更高主要来自 augmentation 的任务难度变化。**

这只是基于现有数据的推断，还不是严格单因素证明。

但赶 CVPR 的情况下，我不建议为了这个问题再浪费一次完整 10 epoch。若有空闲卡，只需要从相同 2K 状态跑一个 **No-Aug + Ramp100 的 200–300 update 小诊断**，即可确认 gradient spike 是否主要由 Ramp 消除，不需要把它训完整。

## 最新 C2R 工作给我们的一个非常重要的共同信号

你之前的直觉是：

> Randomized 主要多了干扰物，Sidecar 有 grounding 和 progress awareness，所以理论上应该更有优势。

方向是对的，但 RoboTwin C2R 的分布变化其实比“多几个干扰物”更广。

官方 leaderboard 规定训练固定为 50 tasks × 50 clean demonstrations，再分别在 clean 和 randomized 环境测试 100 trials/task。citeturn14view4 RoboTwin-Clean2Rand 的公开协议描述中，Clean 是固定白背景、默认灯光、无 distractor、固定桌高；Randomized 会改变背景、光照、clutter 和桌高，部分评测还包含未见语言表述。citeturn14view5turn17view0

所以目前 PatchWAM-style augmentation 主要覆盖的是：

\[
\text{lighting/background/color/texture/noise}
\]

而**没有真正模拟**：

\[
\text{clutter + table geometry + task-dependent distractor competition}.
\]

这恰好意味着 ReLAC 仍有一个很大的发挥空间。

### PatchWAM 告诉我们的事情

PatchWAM 的 C2R 强增强本质上非常“appearance-oriented”：颜色、亮度、Gamma、Exposure、Gaussian noise/blur、AdaIN、FFT；C2R 版本反而不做 crop 和 rotation。citeturn18view3

这可以帮助模型减少对表面 appearance 的依赖，但**不能直接教会模型在多出来的物体之间选对 target**。

所以强增强值得保留，但它和你的 Grounding Sidecar 应该是互补关系，而不是替代关系。

### PICA-WAM 给出的证据更直接

PICA 做了一个非常重要的控制实验。

在相同 WAM 上：

| 方法 | LIBERO-Plus 五种视觉扰动 Avg |
|---|---:|
| Baseline | 69.6 |
| + Augmentation | 79.0 |
| + Aug + Action Alignment | **83.1** |

也就是说，augmentation 本身已经很有效，但**显式让 Action Representation 在 clean/perturbed appearance 下保持一致，又额外提高了 4.1 pp**。其中相对 +Aug，Noise +7.5、Background +5.1、Layout +4.3。Action stream alignment 也优于 Video stream alignment 和两者同时 alignment。citeturn14view2

更关键的是他们做了与你们现在很适合复现的 paired diagnostic：

| 方法 | Seen perturbation 32-step drift | Unseen perturbation drift |
|---|---:|---:|
| Baseline | 16.14 mm | 16.42 mm |
| +Aug | 9.41 mm | 11.24 mm |
| PICA | **5.49 mm** | **6.21 mm** |

也就是说，**augmentation 的确让动作更稳定，但还有大量扰动信息残留在 Action Representation 里；显式 Action Alignment 又进一步消除了这部分差异。** citeturn18view0turn18view1

这与当前 ReLAC 的状态非常像。

### PrismWAM 对我们还有一个特别重要的警告

PrismWAM 的 RoboTwin C2R ablation 很值得我们注意。

完整模型：

\[
87.4\%\ Clean,\quad54.1\%\ Random.
\]

让 Action **直接读取 observed RGB** 后：

\[
89.1\%\ Clean,\quad40.3\%\ Random.
\]

也就是 Clean 反而提高了 1.7 pp，但 Random 下降了 13.8 pp。citeturn14view6

这对 ReLAC 特别重要，因为我们的 Sidecar 本质上也是：

\[
\text{Visual Features}
\rightarrow
\text{Relational Latents}
\rightarrow
\text{Action Write}.
\]

如果 Relational Latents 真正编码的是 geometry、target、contact、task progress，它应该增强 C2R。

但如果它们顺便编码大量 background/color/texture shortcut，那么它实际上可能成为：

\[
\boxed{\text{新的 appearance → action shortcut}}
\]

所以现在的目标不能再是：

> “把 Write Ratio 从 3% 做到 5%。”

而应该变成：

> **“让这 2%–3% 写进去的东西更加 task-relevant、appearance-invariant。”**

这也是为什么 D 的 Write LR×2 已经告诉我们：单纯强化 Write 并不是正确方向。

## 下一条主实验：ReLAC Cross-Appearance Action Alignment

这是我现在最推荐直接实施的实验。

为了便于讨论，可以暂时叫：

**ReLAC-XA**  
*Cross-Appearance Action Alignment*

不是最终论文名称，只是实验名。

### 保持所有已验证部分不变

从**当前 Strong-Aug 实验的同一 2K full-state checkpoint** 分叉，而不是重新从头开始：

| 项目 | Current Aug+Ramp | ReLAC-XA |
|---|---|---|
| 0–2K strong augmentation warmup | 相同 | 相同 checkpoint |
| WAM 0–2K | frozen | frozen，已完成 |
| EA supervision | 100% | **100%** |
| Write | 原配置 | **原配置** |
| β / projection LR | 原配置 | **原配置** |
| WAM unfreeze ramp | 100 updates | **100 updates** |
| Global batch | 256 | **256** |
| Augmentation | 当前版本 | **相同** |
| 新增 | — | **Action representation alignment** |

因此这个实验只回答一个问题：

> 在 augmentation 已经存在的情况下，显式要求 Action Representation 对 appearance invariant，是否能进一步改善 C2R？

这是非常干净的 CVPR 消融。

### 具体结构

每个 training sample 保留两份视觉输入：

\[
x_{\rm clean}
\]

和

\[
x_{\rm aug}=T(x_{\rm clean}).
\]

Student 就是现在正常训练的 ReLAC：

\[
h_s^{(m)}
=
f_{\theta}
(x_{\rm aug},a_\tau,c).
\]

它照常计算：

\[
L_{\rm video}+L_{\rm action}+L_{\rm EA}.
\]

EMA Teacher 接收同一物理状态的 clean input：

\[
h_t^{(n)}
=
f_{\phi}
(x_{\rm clean},a_\tau,c),
\]

其中：

\[
\phi\leftarrow
\mu\phi+(1-\mu)\theta.
\]

然后只对 Action tokens 做：

\[
L_{\rm inv}
=
-\frac1K\sum_k
\cos\left(
g(h_{s,k}^{(m)}),
\operatorname{sg}(h_{t,k}^{(n)})
\right).
\]

最终：

\[
\boxed{
L=
L_{\rm video}
+
L_{\rm action}
+
L_{\rm EA}
+
\lambda L_{\rm inv}
}
\]

PICA-WAM 公布的跨四个 WAM backbone 共用配置是：Action stream，student layer 8、teacher layer 20，\(\lambda=0.05\)，noise gap \(\delta\sim U[0,0.2)\)，两层 Linear–GELU–Linear projector、hidden 512，EMA decay 0.9999；teacher 不反传，推理时 teacher 和 projector 都丢弃，因此没有推理时开销。citeturn14view1turn17view0

### 对 ReLAC 我会做一个小适配

你们的 Sidecar 在多个深度 Read / Write。

如果 OpenWAM block 编号和 PICA 使用的网络完全对应，我首先直接用：

\[
m=8,\quad n=20.
\]

如果层号不严格对应，就不要机械套 8/20，我会选择：

> **Student：第一次 Relational Write 之后的 Action hidden。**

> **Teacher：第二次 Relational Write 之后的更深 Action hidden。**

原因是这样会直接要求：

\[
\text{Augmented Relational Write}
\]

产生的 Action Representation 向：

\[
\text{Clean Relational Write}
\]

的更成熟表示靠近。

这是对 PICA 思想的 ReLAC 适配，是我的推论，而不是 PICA 原文已经验证过的配置。

### EMA Teacher 应该包含 Sidecar

我推荐 Teacher 不只是 EMA WAM，而是：

\[
\boxed{\text{EMA(WAM + ReLAC shared/read/refine/write)}}
\]

EA auxiliary heads 不需要复制，因为 teacher 不做 EA prediction。

这点很重要。

否则 clean teacher 的 Action hidden 没有对应 clean relational conditioning，而 student 有 Sidecar Write，两边目标不完全一致。

### 不要对 Video stream 做 Alignment

这一点我非常明确。

Video stream 本来就需要知道：

> “这张图现在是什么颜色、背景是什么、光照是什么。”

因为它承担 future video prediction。

真正应该 invariant 的是：

> **“面对这些外观变化，机器人该怎么行动。”**

PICA 的 ablation 也显示 Action-only alignment 83.1，高于 Video-only 81.6 和 Both 80.7。citeturn14view2

这也是当前 `Video loss +12.3%` 不应该让我们慌的原因。

**不要强行把 Video Features 也变成 appearance invariant。**

### 第一轮只跑 1K screen

从当前 Strong-Aug 的 **2K checkpoint**：

\[
2K\rightarrow3K
\]

先跑 1K。

不是立刻重训 10 epoch。

因为我们已经有：

- Current Aug+Ramp 的 3K；
- A 3K；
- A/C 4K；
- 很快会有 Current Aug+Ramp 4K。

因此 ReLAC-XA 跑到 3K 就已经可以做相当强的横向比较。

如果它在 paired invariance 和 Random offline 上有清晰优势，再直接续训，不需要重新开始。

## 到 4K 时不要只做原来的 Action Error，要增加一个新的核心诊断

我建议当前实验到 4K 时，除了你们已经计划的 augmentation-off 固定输入诊断，再增加：

# Clean ↔ Aug Paired Action Invariance Probe

这是我认为目前信息价值最高的诊断。

### 构造方式

对同一个 simulator state：

\[
(x,c,s,a)
\]

构造：

\[
x_{\rm clean},\quad x_{\rm aug}=T(x_{\rm clean}).
\]

保持完全相同的：

- instruction；
- proprioception；
- Action FM noise seed；
- Video noise seed；
- timestep；
- sampler；
- Sidecar 状态。

唯一差异：

\[
\text{appearance}.
\]

然后比较四个层次。

| 层次 | 指标 | 我们想看到什么 |
|---|---|---|
| Sidecar | Ground/Relation/Plan token cosine | Aug 后关系表示稳定 |
| Action hidden | clean↔aug normalized feature distance | 越低越好 |
| FM output | velocity-field difference | 越低越好 |
| sampled chunk | 32-step XYZ/rot/gripper drift | **越低越好** |

PICA 的 paired diagnostic 就使用相同 simulator state 和 noise seed，将 feature sensitivity 一路追踪到最终 32-step trajectory drift；这是比单纯比较训练 loss 更直接的鲁棒性机制证据。citeturn18view0turn18view1

### 对 Sidecar 特别增加一个指标

对于 15 个 relational latents：

\[
E_{\rm clean},\qquad E_{\rm aug}
\]

分别计算：

\[
C_{\rm Ground},
C_{\rm Relation},
C_{\rm Summary},
C_{\rm Plan}.
\]

这能回答一个我们之前从来没有真正回答的问题：

> **我们的 Relational Latents 到底是真的关系 latent，还是“带着关系监督的视觉 latent”？**

如果 Strong Aug 后：

\[
C_{\rm relation},C_{\rm plan}\uparrow
\]

且 Action drift ↓，

说明 augmentation 正在塑造我们想要的 representation。

如果 EA metrics 不错，但：

\[
E_{\rm clean}\not\approx E_{\rm aug},
\]

那就解释了为什么 Sidecar 有监督、Write 也有作用，却 C2R 仍然有限。

### Tail error 必须单独报告

你们的 A/B/C/D/E 已经暴露出一个非常重要的模式：

> 均值经常被一两个 32-step 后段 catastrophic divergence 拉坏。

所以之后每个 checkpoint 不再只报：

\[
\operatorname{Mean}(XYZ_{1:32}).
\]

我建议固定报：

| 指标 | 目的 |
|---|---|
| Step 1–8 mean | 第一段动作精度 |
| Step 9–16 mean | 中段 |
| Step 17–24 mean | 后段 |
| Step 25–32 mean | 最后段 |
| Full-32 mean | 总体 |
| Median window | 常态 |
| **P90 / P95 window error** | 尾部风险 |
| **Worst-3 windows** | catastrophic behavior |

你们现在的数据已经证明 mean 不够。

B 的 9/12 Random windows 可以比 A 更好，却被一个约 195 mm 的 window 拉爆；E、C 也有类似现象。

这恰恰是闭环里真正危险的东西。

## 目前还值得做的两个 Sidecar 专项 Trick

Action Alignment 是我最推荐的下一条主线。但如果还有一条短实验预算，我会优先在下面两种里选，而不是 B/D/E。

### Semantic-Preserving Augmentation：非常便宜，但值得检查

PatchWAM C2R 会做最大约 ±36° Hue shift，而且还会改变色温。citeturn18view3

但 RoboTwin 确实存在**颜色本身就是任务语义**的任务。例如官方 `Stack Blocks Three` 的目标明确要求按照红、绿、蓝的身份进行堆叠。citeturn19search0

因此对于 ReLAC 来说存在一个特别值得警惕的问题：

\[
\text{instruction: ``red block''}
\]

但增强之后，原来的红块可能视觉上已经不再“红”。

普通 WAM 可能还能靠空间 shortcut 撑过去，但你们正在显式训练 **Grounding**，所以这种 label–appearance inconsistency 对你们反而可能更敏感。

**现在不要停掉当前实验去改。**

到 4K 时，把带颜色语义的任务单独列出来。

如果出现：

\[
\text{non-color tasks C2R}\uparrow
\]

但：

\[
\text{color-reference tasks}\downarrow,
\]

下一版本非常简单：

> 对含 color-referential language 的任务，关闭或显著减小 Hue / RGB-temperature augmentation；Brightness、Contrast、Gamma、Noise、Blur、AdaIN、FFT 保留。

这样几乎不增加训练复杂度。

我把它看作一个**很便宜的潜在 +X% trick**。

### Distractor-Invariant Relational Conditioning：很适合 ReLAC，但放在第二梯队

你关于 Sidecar 对 distractor 应该有优势的直觉，其实有近期工作直接支持。

OA-WAM 专门用 object-addressable slots，把 target/reference 与 distractors 分开。它训练时会：

- 对非 target/reference 的 distractor slots 进行 permutation；
- 从 batch 里插入额外 distractor slot；
- 要求原始和 perturb 后的 Action 输出与 attention 保持一致。

其 consistency loss 包括 attention KL 和 action L2，并将权重在训练 Stage II 前 30% 从 0 ramp 到 0.1。citeturn14view7

Perturbot 则采用另一种思路：插入**不改变动作标签、也不遮挡目标和接触几何**的 plausible distractor，使模型不能继续依赖“最显眼的东西就是目标”这种 shortcut。citeturn16view0

这与 ReLAC 非常契合。

但我**暂时不建议马上做像素级随机 distractor insertion**，因为你们要赶 deadline，而且如果使用 simulator 重新渲染 randomized scenes，还要额外考虑 RoboTwin leaderboard 固定 2,500 clean demonstrations 的训练数据规范。官方规定训练数据固定为 50 × 50 clean trajectories。citeturn14view4

更安全的 ReLAC 版本是：

\[
\textbf{Distractor-Invariant Sidecar}
\]

利用你们训练时已有的 simulator grounding 信息，确定：

- target ROI；
- reference ROI；
- robot ROI；
- non-target/background region。

仅对非任务相关区域进行有限 feature/patch perturbation，然后要求：

\[
E_{\rm Relation}^{orig}
\approx
E_{\rm Relation}^{dist},
\]

\[
E_{\rm Plan}^{orig}
\approx
E_{\rm Plan}^{dist},
\]

以及：

\[
H_A^{orig}
\approx H_A^{dist}.
\]

这个思路很符合论文 story，但它比 PICA Action Alignment 风险更高，因为需要决定“哪些 visual tokens 可以安全扰动”。

所以我的顺序仍然是：

\[
\boxed{
\text{Appearance invariance first}
\rightarrow
\text{Distractor invariance second}
}
\]

而不是同时叠。

## 哪些东西我现在明确不建议做

### 不建议继续削弱 EA 梯度

B 已经给了非常好的证据：

\[
L_{\rm Plan}+23.35\%,
\qquad
L_{\rm Relation}+40.19\%
\]

Ground Hit：

\[
95/70\%
\rightarrow
80/60\%.
\]

方向 cosine：

\[
0.775/0.817
\rightarrow
0.718/0.767.
\]

而完整 Random 32-step XYZ 还出现严重 tail failure。

这说明至少目前：

> **EA supervision 不是一个明显应该被消灭的 nuisance gradient。**

尤其现在我们想做 C2R，更不应该主动削弱正在提供 grounding/relationship inductive bias 的路径。

所以 B 暂停。

### 不建议再提高 Write 强度

D 是很漂亮的反例。

Write LR ×2：

\[
\|\Delta\theta_{\rm Write}\|
\approx1.58\times A
\]

beta 位移：

\[
1.91\times A,
\]

但实际 Write Ratio 反而降为：

\[
0.696/1.116/1.854\%
\]

而 Full-Off FM improvement 只剩 1.26%。

所以：

\[
\boxed{
\text{more Write optimization}
\neq
\text{more useful Action conditioning}
}
\]

目前 A 已经有：

\[
1.9\%\sim3.2\%
\]

的真实写入和约 12.93% 的自身 Full-vs-Off FM 改善。

**现在不是“Sidecar 写不进去”的阶段了。**

现在的问题是：

> 写进去的表征够不够 robust。

### 不建议现在改 Batch 512

Batch 512 可能让梯度噪声更低，但 Ramp100 已经把最大 gradient spike 从 12.67 压到了 1.08。

因此主要 optimization pathology 已经得到很大程度缓解。

与此同时，PICA、PatchWAM 等相关训练配置本身就在 global batch 256 级别工作；PICA 的核心收益来自 representation invariance，而不是扩大 batch。PatchWAM 的 RoboTwin 配置同样使用 global batch 256。citeturn14view6turn14view3

现在有限 GPU 预算，我更愿意把计算量给：

\[
\text{Clean Teacher Forward}
\]

而不是：

\[
B:256\rightarrow512.
\]

### Tail-aware Action Loss 可以留作最后一个“小 Trick”

你们现在 32-step 尾部异常让我重新看了一下 AttenA+。

它对 flow-matching Action Loss 按 EEF velocity 做 timestep weighting，对低速、精细接触阶段提高权重，同时做 weight clipping 和 mean normalization，避免改变整体学习率尺度。citeturn14view8

在其 Fast-WAM RoboTwin 实验中，报告的平均成功率由 91.80% 提高到 92.46%，约 +0.6 pp；但该实验的训练方式是冻结视觉编码器和 pretrained WAM backbone、只调最终 action head，所以不能直接认为你们也会得到 +0.6 pp。citeturn20view0turn20view1

我会先做一个零训练成本检查：

\[
\text{tail error}
\quad\text{vs.}\quad
\text{EEF velocity/contact phase}.
\]

如果 catastrophic windows 明显集中在：

- approach；
- contact；
- alignment；
- release；

这样的低速关键阶段，再加入 AttenA-style weighting。

否则不动。

## 最终执行方案

我建议你们现在把路线正式收敛成下面这个 pipeline。

### 当前正在训练的模型

**继续跑：**

\[
\boxed{
\text{ReLAC + Strong Aug + 2K Warmup + Ramp100}
}
\]

至少到 4K。

**不要因为当前 Action +20.6%、Video +12.3% 就停。**

4K 必须使用**关闭 augmentation 的固定输入**重新比较 A4K。

同时跑：

\[
\text{Clean}\leftrightarrow\text{Aug}
\]

paired invariance diagnostic。

### 4K 的决策表

我会按下面的优先级判断，而不是按 Training Loss。

| 优先级 | 指标 | 理由 |
|---|---|---|
| **最高** | Randomized paired closed-loop SR | 真正目标 |
| **最高** | Clean↔Aug 32-step action drift | 是否真的学到 invariance |
| **最高** | Random 32-step P90/P95 tail error | 当前最大潜在失败模式 |
| 高 | Ground Hit / Relation / Plan quality | ReLAC 是否保留物理关系能力 |
| 高 | Full vs Off Action FM | Sidecar 是否仍有正向作用 |
| 中 | 固定 unaug Action FM | 本体精度 |
| 中 | Clean SR | 不能明显倒退 |
| 低 | augmented training loss vs A | 输入难度不同，不再公平 |

如果当前 Aug+Ramp：

- Random closed-loop 明显优于 A；
- 或者 Random SR 暂时相近，但 paired drift / P95 tail / grounding 三者明显改善；

我会继续它到完整训练预算。

如果：

- Random SR 下降；
- paired drift 没变好；
- grounding 也变差；
- tail 还更严重；

那就停止，不因为“训练曲线还在下降”继续浪费算力。

### 下一张 GPU 卡应该启动什么

不是 B，不是 C300，不是 D，不是 E，不是 batch512。

而是：

\[
\boxed{
\textbf{ReLAC-XA}
}
\]

具体：

```text
Initialization:
    当前 Strong-Aug run 的相同 2K full-state checkpoint

Student:
    augmented observation
    normal Video + Action + EA losses

Teacher:
    EMA(WAM + ReLAC shared/read/refine/write)
    clean observation
    no gradient

Alignment:
    Action hidden only
    student shallow/post-first-Write
        ->
    teacher deeper/post-second-Write

Loss:
    L = L_video
      + L_action
      + L_EA
      + 0.05 * L_cross_appearance

WAM unfreeze:
    100-update 0.1x -> 1.0x LR ramp

EA shared gradient:
    100%

Write:
    original LR
    original gradient
    original beta

Global batch:
    256
```

首先：

\[
2K\rightarrow3K
\]

短训 1K。

然后与同 3K 的 Current Aug+Ramp 做完全相同的 fixed / paired diagnostics。

**这是一条非常适合你们当前 CVPR 时间压力的路线，因为它不需要改 inference，不需要重构 Sidecar，不需要改 action representation，也不需要改变 leaderboard 推理接口。**

PICA 的 teacher/projector 训练完成后可以直接丢弃；其核心机制也已经在多种 WAM 上验证，而且在 OpenWAM-\(\alpha\) 上的 EBench 实验从 55.3% overall SR 提高到 61.4%，说明这种训练方式至少已经和 OpenWAM 类 backbone 具有实际兼容证据。citeturn14view2

### CVPR 最终最理想的实验故事

如果结果顺利，你们最终会有一个非常清晰的 progression：

\[
\text{OpenWAM}
\]

↓

\[
+\text{ReLAC}
\]

**任务关系 / grounding / progress latent**

↓

\[
+\text{Strong Appearance Augmentation}
\]

**让模型看见视觉变化**

↓

\[
+\text{Cross-Appearance Action Alignment}
\]

**让 Action 真正忽略无关视觉变化**

于是论文的机制故事不再只是：

> “我们外挂了一个 Sidecar，所以 C2R 高了一点。”

而可以变成：

> **Simulation-derived relational supervision encourages task-grounded latent representations, while cross-appearance action consistency prevents visual nuisance factors from leaking through the relational interface into action generation.**

这个 story 与你们现在的数据其实已经非常契合。

A/B/C/D/E 已经帮我们排除了几个错误方向：

\[
\text{弱化关系监督}\quad\color{gray}{\text{不优}}
\]

\[
\text{盲目放大 Write}\quad\color{gray}{\text{不优}}
\]

\[
\text{完全消灭解冻梯度}\quad\color{gray}{\text{不一定更优}}
\]

而现在越来越清楚的正确方向是：

\[
\boxed{
\text{Stable Optimization}
+
\text{Relational Grounding}
+
\text{Appearance-Invariant Action Conditioning}
}
\]

其中我认为**最后这一项是当前最缺、也最可能继续推高 RoboTwin C2R 的部分**。

所以我的最终决策非常明确：

> **现有 Strong-Aug + Ramp100 继续到 4K；现在就准备 ReLAC-XA 代码，从同一个 augmented 2K checkpoint 分叉，下一轮只增加 Action-only Cross-Appearance Alignment。不要再叠 B/D/E，不增 Batch，不放大 Gate。**

而如果 ReLAC-XA 确认能进一步降低 clean↔aug Action drift，同时保持你们已经获得的 Ground/Plan 质量，我会把它作为**下一轮唯一值得投入完整 10-epoch 训练预算的候选**。