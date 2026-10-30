# OpenWAM-alpha + EA-RSIG-Lite 架构图生成记录

2026-10-16。使用 imagegen 技能的内置 image_gen 工具。原图 docs/image.png 仅作 OpenWAM 双分支结构参考，未覆盖原文件。

## 初稿提示词

Use case: scientific-educational / infographic-diagram.
Create a NEW, highly legible, publication-quality architecture diagram explaining the actual implemented OpenWAM-alpha + EA-RSIG-Lite C joint model. The supplied image is a REFERENCE for the original OpenWAM dual video/action architecture, NOT a requirement to preserve its cramped layout. Redraw with a generous white canvas, flat vector-style boxes, thin clean orthogonal arrows, consistent modern sans-serif typography, substantial whitespace. Large landscape image, preferably 3840x2560. Chinese explanatory labels with short English module names. Prioritize technical correctness and readable text over decoration. No robot clipart or unrelated illustrations.

TITLE: "OpenWAM-alpha + EA-RSIG-Lite"
SUBTITLE: "当前实现：C joint ｜绿色模块为我们的新增部分"
Legend: blue = OpenWAM视频分支; purple = OpenWAM动作分支; green = 新增EA模块; dashed red arrows = 仅训练监督. Main forward arrows must be solid.

Three clearly separated panels:
(A) 主训练结构 — the left 60% of the image, flowing bottom to top like the reference.
(B) EA-RSIG-Lite 内部结构 — right 40%, expanded readable diagram.
(C) 训练目标与推理边界 — horizontal bottom band.

PANEL A main structure:
At bottom two original input paths:
Left: "三视角视频" small subtitle "33帧跨度 → stride 4 → 9帧". Arrow to blue "Wan VAE 编码（冻结）". From its video latent path add noise only to future frames, label "未来latent加噪 τv；当前首帧保持干净", then "Video patch embedding". On the far left draw a distinct clean-current-frame branch from the VAE current latent: "当前首帧 z_t（干净）" → "共享 patch embedding" → thin green connector going to Panel B input. IMPORTANT: this branch is before all Video DiT blocks, not from intermediate DiT states. EA never sees future frames/noisy action as input.
Right: "未来32步绝对EEF动作" subtitle "20D有效动作 → 80D统一空间" → "动作加噪 τa" → "Action Encoder".
Above, two aligned columns Video stream and Action stream enter a gray outlined box labelled "第1个联合块 · block 0". Inside include two small channel-specific sequences from bottom to top:
Video "AdaLN → QKV", Action "AdaLN → QKV";
one shared gray horizontal bar "Shared Video–Action Attention";
then on each side "Cross-Attn → AdaLN → FFN".
A separate top-center condition box "任务文本 + 当前Proprio" connects to both streams' Cross-Attn with thin gray arrows.
Directly ABOVE the block0 box, place a vivid GREEN residual adapter ONLY on the PURPLE Action stream, callout "② 唯一注入点：第1个联合块之后".
Its exact internal labels: "Q = Action hidden"; "K,V = EA 交互token"; "8-head Cross-Attn"; "Linear（零初始化）"; a visible plus-circle on the Action stream with identity bypass arrow. Label "h_a ← h_a + Δh_EA". A green solid arrow from Panel B's 15-token output connects here; its direction must clearly be EA → adapter, not vice versa. The Video stream goes straight past this adapter with no direct green module.
Both streams continue into ONE clearly labeled gray shared outer box "后续联合块 · blocks 1…L−1" showing continued shared Video–Action attention. Output two heads "Video head：latent流速度" and "Action head：动作流速度". Each has a short red dashed link to its matching training loss at bottom band. Do NOT label these outputs as already decoded RGB or executable clean actions in training.
Small green callout near the left clean branch: "① 特征来源：仅当前干净观察". Small green callout near main adapter: "不向原联合序列插入EA token；保留原attention mask与RoPE".

PANEL B green light-tinted framed zoom:
Top input rows:
"当前三视角patch" → "每视角14×14；合计588 tokens" → "LayerNorm + Linear → 512D".
"任务文本" → "Masked mean + Linear → 通道调制" feeds this visual tensor.
"当前双臂TCP [B,2,10] + arm mask" → "EEF MLP: 10→512→512".
Then show the four core components, linked correctly and not just a sequential black-box MLP:
1. "Ground · 空间注意力" with subtitle "4个可学习Object/Goal query × 三视角网格"; output "4个语义slot + 存在/有效性门控". Ground consumes visual grid, not GT boxes.
2. "Relation · 关系MLP" consumes EEF embeddings + Object/Goal slots, labelled "1536→1024→512 + LayerNorm"; outputs "2个EEF–Object + 2个Object–Goal".
3. "Progress · 注意力汇聚" consumes 4 relation tokens; subtitle "1个learned query + 8-head Attn + FFN"; output "1个进展token".
4. "Plan · 多时间尺度MLP" consumes EEF/Object/Goal/Progress; subtitle "预测8 / 16 / 32步：方向 + 运动概率"; output "6个计划token（3时刻×2臂）".
At bottom of B, join the projected/gated outputs into a prominent outlined green box:
"15 × 512 交互token"
"4 Ground + 4 Relation + 1 Progress + 6 Plan"
Give Ground/Relation/Progress outputs respective direct paths to this assembly, not just Plan alone.
A single solid green connector leaves the output toward the Panel A Action cross-attention adapter.
Small note: "空间定位与动作读取用注意力；关系与计划预测用MLP".
Show only the active Lite structure, no legacy H18 adapter, no direct video residual, no full extra observation DiT.

PANEL C:
Left: "原始EA oracle → 离线重建8/16/32标签 → labels/*.npz（130D监督）". Dashed RED arrows go ONLY to EA auxiliary loss, NEVER to Panel B forward inputs or Action features.
Middle in clear typography:
"L = L_video + L_action + 0.025 L_ground + 0.01 L_relation + 0.05 L_plan"
"joint更新：Video DiT + Action DiT + EA；VAE/文本编码器冻结"
Right: two very short boundary statements:
"训练：EA标签仅用于loss"
"推理：当前RGB + 文本 + 当前EEF → 预测EA → 条件化动作去噪"
At the very bottom one quiet caption: "Baseline 对照：关闭所有绿色EA路径；使用相同数据、Foundation、增强与训练预算。"

Enforce arrows, labels, dataflow, and panel boundaries exactly. No arrows from EA oracle/GT labels into EA forward computation. No future or action input into the clean observation EA extractor. No EA video residual. No implication that adding EA means replacing the original shared attention. Make this visually cleaner and easier to follow than the source reference, with distinctly larger EA details and two numbered callouts for our additions.

## 连线修订提示词

undefined

