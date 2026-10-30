# RoboTwin C2R 前列 WAM 深度调研与 ReLAC 下一步提升路线

## 先给结论：你们现在真正应该追的不是“更强写回”，而是“更稳健的目标相关表征”

我把你截图中排名靠前的 RoboTwin 2.0 方法、目前公开的论文/技术报告和项目实现放在一起看之后，有一个相当清楚的结论：

**RoboTwin Clean-to-Randomized（C2R）做得好的模型，最近已经明显从“把 WAM 做大、让 Action Expert 更强”转向了三件事：**

> **把外观变化与任务真正相关的几何/语义因素解耦；  
> 让模型更明确地知道“哪个物体、哪个关系、当前处于什么操作状态”；  
> 然后保证这些结构化信息真正进入 Action Generation。**

这和你们 ReLAC 当前的方向其实是高度一致的。

RoboTwin 2.0 的 C2R 并不只是简单 ColorJitter。官方 benchmark 的 domain randomization 明确包含 **clutter、lighting、background、tabletop height 和 language**；其对象库还专门用大量 Objaverse 等物体构建 clutter 和 distractors。也就是说，你说的“Randomized 主要多了干扰物，而且定位和任务目标选择更困难”是对的，但还要再加上**几何高度、背景、光照和语言变化**。citeturn22academia5turn22search1

官方 leaderboard 固定用 50 个任务、每任务 50 条 `demo_clean`，总共 2,500 条训练轨迹，然后分别在 clean/randomized 环境每任务测试 100 次。因此 C2R 本质上是在考验：**只见 Clean demonstrations 后，策略能否保持正确的 object/task grounding 和控制。** citeturn20view0

基于你提供的当前结果：

- OpenWAM-α 官方大约是 **89.4 C2C / 48.7 C2R**；这一数值也被 PrismWAM 的对照实验报告。citeturn16search1
- 你们当前 ReLAC 已经做到大约 **89.0 Clean / 53.2 Randomized**。
- 也就是说，Clean 基本没变，而 C2R 已经提升约 **4.5 个百分点**。
- 真实 Write Ratio 已从旧模型近乎不可见，提高到约 **0.6%–0.9%**，且 Full vs Write-Off 的 Action FM loss 已出现约 5%–6% 的一致收益。

这实际上是非常好的信号：

**你们现在已经不是“Sidecar 写不进去”的问题了。**

下一步最可能带来 C2R 增益的，不是继续把 \(\beta\) 做大、Write LR ×2、或强行让 Action gradient 更大，而是：

\[
\boxed{
\text{让 Sidecar 更准确地找到任务相关实体}
\rightarrow
\text{形成外观不敏感的几何/关系表征}
\rightarrow
\text{把这些表征有选择地写给 Action}
}
\]

我的最终优先级是：

| 优先级 | 修改 | 我对它的判断 |
|---|---|---|
| **P0** | **PatchWAM-style C2R 强外观增强** | 现在最应该立即跑，风险最低 |
| **P0** | **Grounding-Guided Relational Read** | 最适合你们 Sidecar 的结构性改进 |
| **P0/P1** | **2.5D / Relative-3D privileged relation supervision** | 很可能补上你们现在缺的几何信息 |
| **P1** | **WAM 解冻短 LR ramp** | 解决你们已实测存在的优化冲击 |
| **P1** | **关键关系事件重采样** | 比降低整个 EA gradient 更合理 |
| P2 | Future visual latent noise | 值得短测，但不是第一轮 |
| P2 | 更短 replan horizon | 几乎零训练成本，值得并行测 |
| 暂不做 | Batch 512 / Write LR×2 / EA→1% | 当前没有证据是主要瓶颈 |

下面解释为什么。

## 前列模型到底靠什么把 C2R 做高

先说明一个数据口径问题：下面“榜单分数”优先采用你 **2026 年 10 月 27 日截图中的 leaderboard 状态**。一些论文和仓库已经更新了 checkpoint，因此当前 paper/repo 数值会略有不同。例如 UniWAM 最新论文报告 75.14/68.32/71.73，而你截图中的 leaderboard 是 75.12/67.56/71.34；ME-Dex 的当前仓库也已有比截图更新的 checkpoint。citeturn20view3turn18search1

| 方法 | 截图 C2C | 截图 C2R | 我认为真正重要的 C2R 机制 |
|---|---:|---:|---|
| GO-3 Nano Preview | 88.54 | **81.48** | 尚缺可核验技术细节 |
| PrismWAM* | 91.84 | **75.50** | RGB/Depth 解耦 + staged geometry learning + scaling |
| CloudRobo-WM | 88.90 | **75.00** | DINO semantic + RGB-estimated depth geometry |
| MWA-Phoenix | 86.54 | **72.42** | structural + semantic representation |
| ME-Dex-1.0 | 89.58 | **68.12** | train-time privileged tactile/physical supervision |
| ViGAR | 82.00 | **67.02** | explicit visual subgoal / stage decomposition |
| PatchWAM | 91.56 | **66.72** | unified action/video coupling + **强 C2R augmentation** |
| UniWAM | 75.12 | **67.56** | physical semantics + robust world/action coupling |
| PrismWAM | 87.98 | 53.16 | 同样结构，小模型/较少 pretraining |

### PrismWAM：对你们最有启发的是“物理表征先学好”

PrismWAM 把 RGB 和 depth 明确拆开：

- RGB stream 更偏向语义和未来目标；
- Depth stream 负责更密集的物理/几何 rollout；
- Action tower 与这些视觉分支通过 Mixture-of-Transformers 交互；
- 训练不是一上来全部 joint，而是**先 depth-only adaptation，再 RGB/depth/action joint training**。主模型约 6B；PrismWAM* 是 15B，初始化自 WAN-2.1-14B，并进行了额外 video-only pretraining。citeturn20view1

真正值得你们看的不是排行榜数字，而是它的消融。

在 clean-only RoboTwin、global batch 256 的控制实验中：

| PrismWAM 消融 | Clean | Random |
|---|---:|---:|
| Full | 87.4 | **54.1** |
| 去掉 RGB stream | 79.1 | 40.1 |
| **去掉 Depth stream** | 83.0 | **2.4** |
| 改变 RGB→Action routing | 89.1 | 40.3 |
| **去掉 depth-only stage** | 89.3 | **24.7** |

citeturn19view2

注意最后一行：

**去掉物理表征预适配后，Clean 反而从 87.4 提到 89.3，但 Random 从 54.1 掉到 24.7。**

这对你们非常有意义。

它说明一个模型：

> 可以在 Clean 上看起来“Action 学得更快、更好”，  
> 但实际上把真正支撑 OOD 的物理 representation 弄坏了。

这也让我更加不支持你们直接把 EA→Shared gradient 从 100% 降到 1%。你们 B 组已经出现 Relation/Plan 学习明显减速，这和 PrismWAM 的证据方向是吻合的：**不要为了 Clean Action optimization 轻易牺牲结构表征。**

另一方面，你们现在的 **前 2K WAM 不更新、Sidecar 仍完整 forward/backward/write**，虽然和 PrismWAM 的 depth-only adaptation 不完全相同，但思想是同一类：

\[
\text{先让结构/物理路径适应固定 Backbone}
\rightarrow
\text{再 Joint Adaptation}
\]

所以我建议**保留现在的 2K warmup，不要推翻。**

### PatchWAM：当前最应该直接抄的其实不是架构，而是它的 C2R augmentation

PatchWAM 的架构想法很漂亮：它不用单独的 Action Expert，而是通过 **Action-as-Patch** 把动作转换成和视觉 patch 兼容的 token，让同一个 FLUX.2 4B backbone 同时去生成 future image 和 action；future/action noisy tokens 之间可以直接相互 attention。citeturn17search0turn20view2

但对你们而言，真正重要的是作者自己明确写出来：

> **PatchWAM 的 C2R 较小下降不能仅归因于模型架构，因为它专门使用了很强的 C2R image augmentation。** citeturn19view1

它的 Clean2Random run 对 **80% 的样本**做增强，而且特意**不做 crop 和 rotation**，只改变 appearance，这一点特别适合你们现有的 2D grounding 标签：

- brightness：0.6–1.4
- contrast：0.6–1.4
- saturation：0.6–1.4
- hue：±36°
- gamma：0.7–1.4
- exposure：±0.5 stops
- red/blue temperature：±15%
- blur：\(p=0.3\)
- Gaussian noise：\(\sigma=0.03\)
- AdaIN：\(p=0.5\)
- FFT amplitude randomization：\(p=0.5\)，amplitude × 0.5–1.5，phase 保持不动。citeturn19view0turn20view2

尤其 AdaIN / FFT 这两个操作很值得注意：

**它们显著改变 style / texture / frequency statistics，但不移动物体边界。** citeturn19view1

这几乎就是为你们 ReLAC 量身定制：

\[
\text{外观被强烈扰动}
\quad+\quad
\text{Grounding / Contact / Relation GT 不变}
\]

迫使 Sidecar 学：

> **“这个东西还是任务对象，即使颜色、光照、texture 和背景变了。”**

而不是记住 Clean domain appearance。

我认为这是目前**最确定、最便宜、最值得立即实验的 trick**。

另外，PatchWAM 自己 C2R 用的 global batch 也是 **256**。PrismWAM 的 RoboTwin C2R 训练同样是 global batch 256。citeturn20view1turn20view2

所以从这些强模型来看，我现在**没有理由把 Batch 512 当成 C2R 提升的核心方案**。

### CloudRobo-WM 和 MWA-Phoenix：共同信号是“不要只靠 RGB appearance”

CloudRobo-WM 的公开实现说明它使用：

- Wan2.2 video backbone / VAE；
- DINOv3 semantic features；
- Depth Anything 3 depth estimation。citeturn17search2

排行榜定义里明确说明：如果 depth 是从 RGB 估计出来的，而不是传感器 observed depth，它不会被归到“Depth input”一栏，因此 CloudRobo-WM 在排行榜仍可以显示为不使用外部 depth。citeturn20view0

这个方法现在公开的技术材料还不足以让我可靠地还原全部 Attention 路由，所以我不会虚构它的完整结构。但从其公开依赖至少能看出一个非常明确的设计方向：

\[
RGB
\rightarrow
\begin{cases}
\text{semantic features}\\
\text{geometry/depth features}
\end{cases}
\rightarrow Action
\]

MWA-Phoenix 的公开项目同样强调 **structural + semantic representation**，使用 Phoenix Encoder / CS-VAE 将几何、时序和语义信息作为比纯 appearance 更稳定的表征，然后再与 action/world modeling 结合。它同时具有规模非常大的预训练配置，因此它的榜单表现不能简单归因于某个小模块。citeturn11view0

它们共同给你们一个很强的提示：

**ReLAC 现在已有“semantic relation”，但几何监督还可以更强。**

你们目前主要有：

- 2D grounding；
- near/contact/goal；
- object moving；
- future direction。

我认为下一步最值得增加的不是 dense depth model，而是**任务相关的低维 2.5D / 3D relation supervision**。

后面我会给具体方案。

### ME-Dex：和你们 ReLAC 的思想其实非常接近

ME-Dex 用的是 Video Expert + Tactile Expert + Action Expert 的 Mixture-of-Transformers，并用 shared attention 让三种模态在中间层交换信息；三个分支共同用 Flow Matching 学 video、tactile 和 action。citeturn16academia13

最有意思的是 RoboTwin：

**训练阶段利用 tactile/force 信息帮助塑造表示，但测试可以不给真实 tactile observation。**

PrismWAM 的论文在对照中也明确把 ME-Dex 描述为“training tactile supervision, zero tactile input at evaluation”的方法。citeturn16search1

这和 ReLAC 的核心哲学几乎是同一个类别：

\[
\text{Simulator-only privileged physical labels}
\rightarrow
\text{shape latent representation during training}
\rightarrow
\text{no privileged sensor required at inference}
\]

你们用：

- object state；
- contact；
- moving；
- goal relation；
- future physical direction。

ME-Dex 用 tactile。

所以它反而支持我的判断：

**不要轻易把 EA shared gradient 削掉。你们应该考虑让 simulator privileged labels 更贴近控制，而不是让它们影响更弱。**

ME-Dex 当前仓库甚至已经放出了更新版本的 RoboTwin checkpoints，报告的 Randomized 高于你截图中的旧条目，因此它也是一个仍快速变化的方法，截图值和仓库最新值不应混用。citeturn18search1

### ViGAR：对你们最值得借的不是完整 Planner，而是“Progress 应该显式进入 Action”

ViGAR 把 WAM 分成：

\[
\text{Current Observation + Instruction}
\rightarrow
\text{Visual Subgoal Planner}
\rightarrow
\text{Next Visual Subgoal}
\]

再：

\[
\text{Visual Subgoal + Observation}
\rightarrow
\text{Subgoal Executor}
\rightarrow
\text{Future Visual Trajectory + Action}
\]

其 RoboTwin 训练数据还带有 **stage annotations**。citeturn20view4turn16search7

它在截图中是：

- Clean 82.00
- Random 67.02

也就是说 Clean 并不突出，但 C2R 很强。

这提示的是：

> **任务进度 / 当前 subtask 是一种比 appearance 更稳定的控制变量。**

你们已经有 Summary token、Relation tokens、Plan tokens，本质上已经有做这个的基础。

我不建议你们现在为了赶 CVPR 再造一个 ViGAR visual planner。

但是：

**可以让已有 Summary / Relation latent 显式表示当前 progress / stage，并让它进入 Action Write。**

这是很小但很合理的改动。

### UniWAM：两个工程 trick 值得留作第二梯队

UniWAM 联合了 Physical Reasoner、World Generator 和 Action Predictor，并通过 joint attention 交换信息；它还使用 physical-language supervision、human/robot/VQA 多源数据。citeturn16academia14turn19view5

其 post-training 里有两个很实用的 trick。

第一个是 **Future-frame Noise Augmentation**：

\[
\tilde Z =
s Z +(1-s)\epsilon,\qquad
s\sim U(0.5,1)
\]

以 0.5 概率进一步污染 future visual latent，而保持当前 observation 干净。这样 Action 不会过度依赖一个近乎完美、但真实推理时不可靠的 future latent。citeturn19view4turn20view3

第二个是 **History-conditioned Flow Matching**：不用纯 Gaussian 作为 Action FM 起点，而是从上一段 action history 加小噪声开始：

\[
A^{1}=A_{\text{history}}+\epsilon,\qquad
\epsilon\sim\mathcal{N}(0,0.02^2I)
\]

让动作生成有更强的时序连续性。citeturn20view3

我认为 Future Visual Noise 比 History FM 更适合你们当前短期实验，因为前者改动更局部。

### GO-3 Nano Preview：现在不要根据榜单猜架构

你截图里 GO-3 Nano Preview 已经达到：

**81.48% C2R / 85.01% mean。**

但截至我本次 2026 年 10 月 27 日的检索，我没有找到能可靠核验 **GO-3 Nano Preview** 具体网络结构和 RoboTwin training recipe 的一手论文/技术报告内容。

官方排行榜要求公开代码、weights 和 technical report 才能正式列入，但最新条目和搜索引擎/项目索引存在更新时差。citeturn20view0

因此我不会用 GO-3 的成绩倒推出它“肯定用了某某机制”。

这一项暂时只能作为性能参考，而不能作为可复制技术来源。

## 这些论文其实在告诉我们同一件事

把这些模型排在一起后，我认为高 C2R 方法正在收敛到四类设计。

### 外观和物理状态必须解耦

PrismWAM 的 depth stream、CloudRobo-WM 的 DINO + estimated depth、Phoenix 的 semantic/structural representation，都在尝试减少策略对原始 RGB appearance 的依赖。citeturn19view2turn17search2

PatchWAM 更直接：不改几何，只大幅修改 style statistics。citeturn20view2

因此 C2R 的基本问题可以写成：

\[
I = I_{\rm task} + I_{\rm nuisance}
\]

Clean 训练非常容易学到：

\[
A=f(I_{\rm task},I_{\rm nuisance})
\]

而 Randomization 希望模型接近：

\[
A=f(I_{\rm task})
\]

你们的 Relation Latents 天然应该成为这个 \(I_{\rm task}\) 的压缩表示。

### 结构信息必须真正进入控制路径

这也是你们已经取得进展的地方。

你们新的 Warmup 已经证明：

\[
E_{\rm relation}
\xrightarrow{\rm Write}
H_{\rm action}
\]

不是形式上的接口，而是真正会降低 Action FM loss 的接口。

PatchWAM 的统一 Action-as-Patch、PrismWAM 的 action routing，以及 ME-Dex 的 shared attention，本质上都在解决同一问题：**physical representation 不能只是 auxiliary prediction，要和 Action computation 发生强耦合。** citeturn17search0turn16academia13turn19view2

因此现在不应该继续问：

> 怎么让 Write 更大？

应该问：

> **写进去的东西是不是最 task-relevant 的？**

### C2R 里“找到正确对象”比“理解整张图”更关键

RoboTwin 的 randomization 明确包括 clutter 和 distractor objects。citeturn22academia5

所以一个政策即使对整张图有不错的表示，只要它在：

> 红方块 vs 红色 distractor  
> bottle A vs bottle B  
> target bowl vs background object

之间 target selection 不稳，动作就会彻底失败。

这正是你们 Ground Tokens 应该发挥最大的地方。

我认为这是你们当前 ReLAC **最值得强化、也最容易形成论文差异化的地方**。

### Staged training 比暴力联合优化更可靠

PrismWAM 的 depth-only stage 消融是最强证据之一。citeturn19view2

你们自己的 C 组又发现：

- 正常解冻 WAM 后第 2 次 update，global grad norm 冲到 12.67；
- 300-step LR ramp 后最高约 0.94；
- 100 次更新内从 13 次 clipping 降到 0。

两个来源都指向：

\[
\boxed{\text{Backbone transition deserves explicit treatment}}
\]

所以现在的 warmup 思路应该**强化，而不是删除**。

## 我最推荐给 ReLAC 的结构修改

### 第一项：Grounding-Guided Relational Read

这是我最推荐的 Sidecar 小结构改动。

目前如果你们的 Grounding 流程是：

\[
V
\rightarrow E_{\rm ground}
\rightarrow \hat{x},\hat{y}
\]

然后 Grounding Loss 只约束最后一个 head，那么它存在一个问题：

**模型即便能从 Ground token 解码出正确位置，这个定位结果未必真正改变 Relation/Plan 从视觉 token 中读取什么。**

我建议把它变成：

\[
\boxed{
\text{Ground first}
\rightarrow
\text{use Ground attention to route visual information}
\rightarrow
\text{Relate / Plan}
\rightarrow
\text{Write Action}
}
\]

假设视觉 patch 为：

\[
V=\{v_1,\dots,v_N\}
\]

Ground query：

\[
q_g=W_q E_g
\]

计算：

\[
s_i=
\frac{q_g^\top W_kv_i}{\sqrt d}
\]

\[
A_g=\operatorname{softmax}(s)
\]

然后直接用 GT object 2D position / bbox 把这个 Attention Map 监督成一个 target heatmap：

\[
L_{\rm attn}
=
KL(M_{\rm gt}\|A_g)
\]

如果现在只有 \((x,y)\)，不需要 segmentation mask。

直接以 \((x,y)\) 为中心构造 Gaussian heatmap：

\[
M_i
\propto
\exp
\left(
-\frac{\|p_i-p_{\rm gt}\|^2}{2\sigma^2}
\right)
\]

这自然会把**所有 distractor patches 变成负样本**。

然后 Relation / Plan Attention 不再完全自由读取整张图，而加入 Ground bias：

\[
S_{\rm relation}
=
\frac{Q_rK_v^\top}{\sqrt d}
+
\lambda \log(A_g+\epsilon)
\]

其中：

\[
\lambda=0
\]

初始化。

这点非常关键。

**零初始化 \(\lambda\)** 意味着新模型开始训练时完全等价于现有 ReLAC。

训练学到 Grounding 确实有用之后，才逐渐打开 object-focused routing。

所以风险远低于直接 hard mask visual tokens。

### 如果现在 Sidecar 没有显式语言输入，再加一个很小的 Instruction Injection

如果 Sidecar 当前只是通过 WAM visual hidden **间接**接触语言，而 Ground tokens 没有直接 task instruction conditioning，我建议做：

\[
E_g^0
=
E_{g,\rm learned}
+
\gamma_t W_t e_{\rm text}
\]

其中 \(e_{\rm text}\) 直接复用 WAM 已经计算好的 pooled text representation，不新增语言模型。

同样：

\[
\gamma_t=0
\]

初始化。

这样：

> “push the red cube”

会直接参与“哪个 visual patch 是当前目标”的选择。

而不是希望 Sidecar 自己从被融合过很多层的 WAM hidden 中重新恢复任务身份。

如果现在已经有直接 Text→EA Cross-Attention，这一项就不用重复加。

### 四个 Ground Tokens 不要平均做同一件事

如果现有标签允许，我会给它们实体角色：

\[
E_G=
[
G_{\rm object},
G_{\rm goal},
G_{\rm leftEE},
G_{\rm rightEE}
]
\]

或者根据任务数据定义最稳定的四个角色。

Action 真正需要的不是“图中存在什么”，而是：

\[
\text{EEF}
\leftrightarrow
\text{Object}
\leftrightarrow
\text{Goal}
\]

这个 relational graph。

这比继续加 latent 数量更有价值。

## 第二项：把 2D Grounding 升级为很轻的 2.5D / Relative-3D Supervision

这是我认为你们目前 Sidecar 最明显的潜在缺口。

RoboTwin Randomization 不只有颜色和 clutter，还有 **tabletop height** 等几何变化。citeturn22academia5

2D Grounding 能回答：

> “物体在图像哪里？”

但不能很好回答：

> “物体相对末端还有多远？”  
> “目标在物体左上还是右下？”  
> “桌子高度是不是变了？”  
> “EEF 应该沿哪个 3D 方向靠近？”

PrismWAM 的 depth 消融说明，几何 representation 对其 C2R 至关重要。citeturn19view2

但你们没有必要为此增加一个完整 Depth Encoder。

仿真里已经有 GT state，因此可以极便宜地增加：

### Ground token

除了：

\[
(\hat u,\hat v)
\]

再预测：

\[
\hat z
\]

即 object / goal 的 camera depth。

### Relation token

预测：

\[
\Delta p_{\rm EEF\rightarrow obj}
=
p_{\rm obj}-p_{\rm EEF}
\]

和：

\[
\Delta p_{\rm obj\rightarrow goal}
=
p_{\rm goal}-p_{\rm obj}
\]

不一定直接回归米制 XYZ。

更稳定的方法是拆成：

\[
d=\|\Delta p\|
\]

和：

\[
\hat{r}=\frac{\Delta p}{\|\Delta p\|}
\]

Loss：

\[
L_{\rm geo}
=
\lambda_{\rm dir}
(1-\cos(\hat r,r))
+
\lambda_d
\operatorname{Huber}(
\log\hat d-\log d
)
\]

这和你们已经在用的 future local direction 非常兼容。

关键在于：

**这些 GT 仍然只是 training supervision。**

推理时：

- 不需要 simulator state；
- 不需要 depth camera；
- 不需要 Depth Anything；
- 不增加额外 foundation model。

它只是迫使 ReLAC latent 从 RGB 中学出：

> task-specific metric-ish geometry.

这正是你们论文的思想。

我甚至认为这个方向比引入完整 DepthAnything 更适合 ReLAC 的科研故事。

## 第三项：直接复制 PatchWAM 的 C2R augmentation

这项我建议马上实现。

### ReLAC-C2R Aug

训练样本概率：

\[
p_{\rm aug}=0.8
\]

对于一个 camera view：

- 当前 observation；
- history frames；
- future target frame；

**全部使用相同的 augmentation 参数。**

不同 camera view 可以独立采样 style 参数。

这点和 PatchWAM C2R recipe 保持一致。citeturn20view2

不要做：

- crop；
- rotation；
- random translation。

原因不是这些增强一定不好，而是你们现在有 precise Grounding supervision，最省风险的办法就是**不改变 geometry**。

使用：

```text
brightness    [0.6, 1.4]
contrast      [0.6, 1.4]
saturation    [0.6, 1.4]
hue           ±36°
gamma         [0.7, 1.4]
exposure      ±0.5 stop
R/B scale     ±15%
blur          p=0.3
Gaussian σ    0.03

AdaIN         p=0.5
FFT amplitude p=0.5, scale=[0.5, 1.5]
```

这些是 PatchWAM 公布的 C2R 配方。citeturn19view0turn20view2

### 对 ReLAC 来说，这个 augmentation 比普通 WAM 更有意义

因为：

\[
I
\xrightarrow{\rm strong\ style}
I'
\]

但：

\[
Y_{\rm grounding}
,\ 
Y_{\rm relation}
,\ 
Y_{\rm contact}
,\ 
Y_{\rm plan}
\]

全部保持不变。

因此 Sidecar 的监督天然形成：

\[
E(I)\approx E(I')
\]

的任务语义约束。

甚至不用先增加额外 consistency loss。

**现有 EA supervision 本身就是一个 domain-invariant representation learner。**

我认为这一点非常值得写进论文。

## 训练策略上，我会这样改，而不是继续 B/D/E 那几个方向

基于你们已经完成的五组实验，我现在会明确做减法。

### EA Shared Gradient 不降到 1%

B 已经看到：

- Plan 后段 +23.35% loss；
- Relation +40.19%；
- 差距随着 annealing 完成而扩大；
- Shared parameter displacement 也明显偏离 A。

所以当前没有理由继续：

\[
100\%\rightarrow1\%
\]

尤其 PrismWAM 的 staging 消融进一步说明，结构/物理 representation 的持续训练很可能正是 C2R 的来源。citeturn19view2

**保持 EA shared gradient = 100%。**

如果需要解决不同 EA 项的不平衡，应在 Relation / Plan 内部处理，而不是把全部 EA 降低。

### Write LR 不 ×2

D 已经给出了非常好的否定结果：

- shared trajectory 基本没变；
- Write displacement ×1.58；
- β displacement ×1.91；
- 但 β 反而更快趋向 0；
- 实际 Write 更弱；
- fixed FM 没改善。

所以不要再“强迫模型写更多”。

当前问题是 information quality，不是 residual magnitude。

### E 的 token gradient ×2 不是优先路线

目前 E 和 A 很接近，没有持续 Action advantage。

在 deadline 下，不继续围绕“Action 梯度到底够不够大”做很多小倍率搜索。

### C 的想法保留，但把 ramp 缩短

C 最有价值的发现不是 Loss，而是它证明：

> **WAM freeze→unfreeze 存在真实 optimizer shock。**

A 第二个 update：

\[
\|g\|\approx12.67
\]

C：

\[
\|g\|\approx0.73
\]

这是非常清楚的优化问题。

但原 C：

\[
300\text{ updates}:0.1\times LR\rightarrow1\times LR
\]

导致前 1K 的 LR integral 只有 A 的 86.37%，所以后面 Action adaptation 落后不能简单归因于“ramp 不好”。

我建议最终版本尝试：

\[
\boxed{
100\text{-update ramp}:
0.1\times\rightarrow1.0\times
}
\]

假设近似线性，前 1K 的累计 LR 代理大约从：

\[
86.5\%
\]

提升到：

\[
95.5\%
\]

同时第一步依然保留已经验证过安全的 0.1× LR。

这比把 batch 256 改成 512 更有针对性。

### 不要把 Batch 512 放进下一条 Golden Run

PatchWAM C2R 用的是 global batch 256。citeturn20view2

PrismWAM C2R 也是 global batch 256。citeturn20view1

而你们已经找到导致 spike 的直接原因和直接修复方案。

所以现在把 batch 翻倍：

- 会改变每 epoch optimizer updates；
- 会改变有效训练预算；
- 会再引入一个实验变量；
- 没有证据它能直接改善 C2R。

**保持 256。**

## 我建议你们马上启动的下一轮实验

如果目标明确是：

> **赶 CVPR，在有限完整训练机会下，把 C2R 尽可能往上推。**

那么我不建议继续做十几个单点 trick。

我会把实验分成一个**立即启动的低风险候选**和一个**结构候选**。

### 立即启动：ReLAC-C2R-AUG

从你们现有的相同 **2K Warmup checkpoint** 分叉。

配置：

| 项目 | 配置 |
|---|---|
| ReLAC architecture | **不改** |
| 2K warmup | 保留现方案 |
| Global batch | **256** |
| EA shared gradient | **100%** |
| Write LR | **1×** |
| Write gradient multiplier | **1×** |
| Gate | 不改 |
| C2R augmentation | **PatchWAM strong recipe** |
| WAM unfreeze | **100-update 0.1→1 LR ramp** |
| Loss weights | 先保持现状 |
| 训练数据 | 同一 Clean2500 |

这里同时用了 Augmentation + shorter ramp。

严格科研消融确实是两个变量，但它们分别解决两个已经有明确证据的问题：

1. **appearance/domain shift**
2. **WAM thaw shock**

而且都不改变最终 inference architecture。

如果你们现在目标是抢最佳模型，我认为这个组合值得直接成为 production candidate。

后面再用短 ablation 把两者分开即可。

### 同时实现结构候选：ReLAC-GGR

全称可以叫：

**Grounding-Guided Relational Read**

它是在当前：

> Read → Refine → Write

中把 Read 改成：

> **Ground → Read → Relate → Write**

核心只加：

\[
A_g=\operatorname{softmax}(Q_gK_v^T)
\]

Ground attention supervision，以及：

\[
S_{\rm relation}
\leftarrow
S_{\rm relation}
+
\lambda\log(A_g+\epsilon)
\]

并令：

\[
\lambda_0=0
\]

如果 Sidecar 没有 direct language conditioning，再把 pooled instruction 加到 Ground token，gate 同样 zero-init。

**这是我认为当前最值得拿来形成 ReLAC 自己技术贡献的改动。**

不是因为 leaderboard 某一篇论文已经这样做，而是因为几篇高 C2R 工作共同说明：

- geometry/structure 比 raw appearance 稳定；
- target/subgoal semantics 很重要；
- structured state 必须参与 Action；
- RoboTwin C2R 有大量 clutter/distractors。citeturn22academia5turn19view2turn20view4

而你们已经拥有 GT Grounding，这意味着你们恰好有条件用很小的代价实现这件事。

### 如果还有一个小改动名额：加 2.5D Relation，不加新 Backbone

我会加：

\[
(u,v,z)_{\rm object}
\]

\[
\Delta p_{\rm EE\rightarrow object}
\]

\[
\Delta p_{\rm object\rightarrow goal}
\]

三个简单 training-only supervision。

不要加 dense depth decoder。

不要现在接 Depth Anything。

不要再加一个大 encoder。

这样既继承 Prism/CloudRobo 的 geometry insight，又保持 ReLAC 是 lightweight relational conditioning framework。

### Progress supervision 只有一种情况下值得现在加

**如果 RoboTwin 数据或你们现有 preprocessing 已经能稳定拿到 task-specific stage ID。**

那就直接让 Summary Token 预测：

\[
s_t\in\{1,\dots,K_{\rm task}\}
\]

或者 coarse progress。

ViGAR 的训练明确使用 stage annotations，而它的核心就是 subgoal decomposition。citeturn16search7turn20view4

但我不建议为了赶这一轮，手工给 50 个任务设计一套统一的：

> approach → contact → move → place

规则。

因为 press / open / push / shake / pick-place 的状态机并不完全相同。

如果没有可靠 stage GT，就先不要加。

### 一个几乎不需要训练的并行测试：更频繁 Replan

如果你们当前 Action chunk 每次执行较长 prefix，比如 24–32 steps，可以直接对现有最佳 checkpoint 测：

\[
K_{\rm execute}
=
\{8,16,K_{\rm current}\}
\]

预测 horizon 不改，只减少真正执行的前缀，然后重新观察。

这个实验不要求模型重新训练。

Randomized 中：

- target pose 更变化；
- clutter 更多；
- 错一点后需要重新定位。

理论上更频繁视觉闭环可能帮助纠偏；但 PatchWAM、PrismWAM 各自使用的 horizon/replan 并不相同，没有一个论文层面保证“越短越好”，因此这里必须用你们自己的 closed-loop 数据决定。PatchWAM 使用 16-step chunk，而 PrismWAM 的 RoboTwin 配置是 horizon 32、每 24 个动作重新规划。citeturn20view1turn20view2

这项可以和训练并行做。

## 最终我会怎样安排你们接下来这几轮

我不会再根据 **1K training Action loss 排名**决定模型。

你们的 C 已经证明这个标准不可靠：

> C 明显更稳定，却因为累计 LR 较小在 1K 后 Action loss 落后。

而 Augmentation 更典型：

**更强的数据增强很可能让 training loss 变高，却让 C2R 变好。**

所以新的 screen 必须直接包含一个小型 C2R closed-loop。

### Screen 候选

我建议只保留三个：

| Candidate | 目的 |
|---|---|
| **A** 当前最佳原配方 | Control |
| **F** Strong C2R Aug + 100-step WAM LR ramp | 最低风险性能候选 |
| **G** F + Grounding-Guided Relational Read | ReLAC 结构候选 |

如果 2.5D head 已经非常容易从 simulator state 得到，可以放进 G；否则不要因为它阻塞 F 的训练。

### 短训期间看什么

不要把 Total Loss 当主指标。

重点看：

**一，Grounding。**

不仅看 Clean grounding。

把同一输入做 appearance augmentation 后，检查：

\[
\|\hat p_{\rm clean}-\hat p_{\rm aug}\|
\]

以及 target attention 是否仍落在正确物体。

**二，Write-Off。**

继续做：

\[
L_{\rm Action}^{Full}
-
L_{\rm Action}^{Off}
\]

确保结构增强之后 Sidecar 仍然被 Action 使用。

**三，真实 Write Ratio。**

确认没有掉回 BF16 几乎写不进去的状态。

不要求越大越好。

**四，C2R distractor-heavy mini benchmark。**

这是最重要的。

从 50 个任务中根据现有 per-task 结果，选出：

- C2C 高；
- C2R 掉得最多；
- 明显涉及多个相似物体、目标容器或 clutter；

的一小组任务。

固定相同 random seeds 做 paired rollout。

这会比全 50-task 低样本快速 eval 更快检测“Grounding 是否真的解决 distractor”。

### Go / No-Go

我的建议不是设一个伪精确统计阈值，而是看三件事情是否同时成立：

\[
\text{C2R ↑}
\]

同时：

\[
\text{C2C 不发生系统退化}
\]

并且：

\[
\text{Full > Write-Off}
\]

如果 G 的 grounding 指标漂亮，但 Full/Off Action 没变化，说明它仍然只是一个好的 auxiliary perception module。

如果 F 的 Training Action loss 更高，但 C2R 明显赢，不要因为 loss 把它淘汰。

最终 winner 才重新从完整起点跑正式 10-Epoch。

## 我最推荐的最终版本

如果让我现在替你锁一个**最符合当前实验、排行榜趋势、CVPR 时间限制，又具有论文辨识度**的版本，我会做：

# ReLAC-C2R

### Simulation-Supervised Multi-Stage Relational Latents for Robust Action Conditioning

整体是：

\[
\text{RGB + Instruction}
\]

↓  

**Strong Appearance Randomization**

让 texture / color / lighting 不可靠

↓

**Task-Conditioned Ground Tokens**

“我要操作哪个物体 / 哪个 goal？”

↓

**Grounding-Guided Relational Read**

Suppress distractors

↓

**Privileged Relational Supervision**

2D Grounding  
+ Relative 3D Geometry  
+ Near / Contact / Moving / Goal  
+ Future Motion Direction

↓

**Multi-Stage Relational Latents**

Block 7 → 15 → 23

↓

**Gated Action Conditioning**

\[
H_A'
=
H_A+
\beta W(CrossAttn(H_A,E,E))
\]

↓

**OpenWAM Action Generation**

训练：

\[
\boxed{
2K\ \text{Delayed WAM Update}
+
100\text{-update short WAM LR ramp}
+
Joint Training
}
\]

这套逻辑就非常完整了：

> **Appearance can change; task relations should not.**

而且这不是为了排行榜临时拼起来的故事。

它和你们最开始想做的事情——**用 simulator privileged physical supervision 学到可迁移的关系表示，再让关系表示真正指导 WAM Action Generation**——完全一致。

从前列工作的证据来看，我现在最有信心的判断是：

**你们不要再花主要精力提升 Write 强度。你们已经证明 Write 能影响 Action。接下来的瓶颈是让写进去的 latent 对 Randomized scene 更 invariant、更 target-specific、更 geometric。**

因此，假如现在只能立刻启动一个新实验，我选：

\[
\boxed{
\textbf{当前 ReLAC}
+
\textbf{PatchWAM-style Strong C2R Augmentation}
+
\textbf{100-step WAM LR Ramp}
}
\]

这是最可靠的低风险性能实验。

而紧接着最值得做的结构实验是：

\[
\boxed{
\textbf{Instruction-conditioned Grounding-Guided Relational Read}
}
\]

如果它在 distractor-heavy C2R tasks 上成立，我认为它不只是一个 engineering trick，反而可能成为 **ReLAC 比普通 Sidecar / Adapter 更有说服力的核心机制**：不是“预测 grounding”，而是**利用 grounding 去控制模型从复杂随机场景中读取什么，并进一步指导动作生成**。