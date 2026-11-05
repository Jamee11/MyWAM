# ReLAC 论文叙事、实现证据与初稿写作说明

日期：2026-11-05，UTC。证据冻结时间见 [`results_snapshot.json`](../papers/relac_cvpr2027/evidence/results_snapshot.json)。这是对当前工作树、已有原始实验记录和一手文献的研究整合，不是新训练或真机实验报告。LaTeX 初稿在 [`papers/relac_cvpr2027`](../papers/relac_cvpr2027/README.md)，图表逐项设计见 [`FIGURE_TABLE_PLAN_CN.md`](../papers/relac_cvpr2027/FIGURE_TABLE_PLAN_CN.md)。

## 1. 先确定这篇论文究竟要证明什么

建议保留 **ReLAC**，展开为 **Relational Latents for Action Conditioning**。推荐题目：

> **ReLAC: Grounding World–Action Models with Simulation-Supervised Relational Latents**

原题 *Simulation-Supervised Relational Latents for Action Conditioning in World–Action Models* 也准确，但偏长。新题把方法的核心作用放在前面；正文第一次出现时解释缩写即可。“Simulation-Supervised”指新增监督的来源，不表示底座只在仿真中训练。OpenWAM Foundation 本身继承人类视频和真实机器人数据，不能把基于它的策略称为“从未看过真实世界的纯仿真模型”。

论文的核心问题应是：**已有视频世界先验，怎样把其中的空间和时间知识转成对当前操控决策有用、可用仿真精确监督、又能从真实可观测输入预测的交互表征？**

建议主张：在保留 WAM 全局生成与动作能力的基础上，用对象/目标定位、末端—对象和对象—目标关系、多时距运动意图，训练一个面向操控的中间接口；检验它是否改善外观变化下的闭环控制，以及目标真机任务的示范效率。

这里不能先假定“完整图像的未来都不重要”。障碍物、自由空间、另一只手、容器结构也可能决定动作。更严谨的表述是：**任务相关交互状态可能比所有像素变化更适合作为新增动作条件，但全局视觉仍然保留。** 当前实现也确实保留了 WAM→Action 的原有通路，没有把关系 latent 设为唯一信息瓶颈。

### 1.1 目标、当前实现和证据之间的对应

| 用户希望具备的能力 | 当前代码实际具备 | 当前可以写到什么程度 |
|---|---|---|
| 预测物体 | 每臂 Object/Goal 角色的二维软定位、可见性和交互轨道有效性 | task-grounded localization；不是开放世界检测、完整 3D tracking 或任意对象分割 |
| 预测 relation | EEF–Object near/contact；Object–Goal proximity | 对指定交互关系的监督；不是完整场景关系图或因果图 |
| 预测 task progress | 一个共享 Summary latent；goal proximity 可作局部状态线索 | **没有显式 progress GT、progress head 或 progress loss**；Summary 不能改名后就当作进度预测 |
| 预测 future plan | 8/16/32 原生控制步后，EEF 与 Object 的是否移动和局部单位位移方向 | 多时距运动意图；没有位移幅度、未来旋转、夹爪事件、完整子目标序列或反事实动作结果 |
| 从全图聚焦局部 | 基础版 latent 自由读取全幅当前帧；新版用预测热图引导后续 Read | 基础版是语义监督，不能声称它已强制裁剪/聚焦；引导版提供明确软空间计算路径，闭环增益仍待测 |
| 单臂、双臂泛化 | 统一槽位、臂 mask，同一 Sidecar 支持 Franka/Aloha/Piper | 证明接口能容纳这些本体；不等于零样本跨本体迁移 |
| sim2real / few-shot | 已有服务、客户端、输入输出及部分仿真诊断 | 真实任务成功率、0/5/10/20 示范效率曲线仍是待填结果 |

**当前最合适的论文版本是“实现和现有结果完整、关键因果对照与真机结果显式留空的初稿”，不是把研究目标写成已经完成的结论。**

## 2. 一条能从原理走到真机的完整叙事

### 2.1 引言的四段逻辑

第一段说明 WAM 的机会。视频生成预训练能提供外观、空间和时序知识；专用动作专家把这些知识转成控制。基础工作应准确引用 [OpenWAM](https://arxiv.org/abs/2609.07398)，不要把其预训练、MoT 架构或 Flow Matching 作为本文贡献。

第二段提出接口问题。像素变化中同时存在任务交互与背景变化。桌面纹理、光照变化不必改变抓取目标，而接触、相对位姿和目标区域改变往往必须改变动作。问题不是世界模型完全不懂物理，而是学习出的高维表征缺少一个与操控明确对齐的训练约束与读写接口。

第三段给出 ReLAC。仿真提供当前对象/目标框、关系和专家未来运动，模型只能从当前 RGB、语言和本体状态预测这些量；15 个角色 latent 在三个深度读取当前帧表征并写入动作专家。可选 Grounding-guided Read 把预测定位变成后续关系/意图查询的空间先验，减少“定位头预测得好，却未被动作通路真正使用”的自由度。

第四段说明验证链条。先在单臂和双臂仿真检验任务成功与随机化鲁棒性，再通过训练期匹配消融区分“辅助监督改善表征”和“在线 latent 条件改善控制”，最后在相同真机任务、相同示范预算下检验迁移。当前初稿已填入可靠仿真结果；未完成的匹配控制、引导版、进度和真机数据用 TBD 标识。

### 2.2 三项贡献的写法

1. **方法贡献**：给现有 WAM 增加仿真监督的角色关系 latent 与多层 Read/Refine/Write 接口，保留原世界—动作联合建模及可部署观测合同。
2. **监督与数据贡献**：明确对象/目标、关系和多时距运动的构造、缺失标签 mask、单双臂角色对应，以及按物理家族划分的数据组织。数据规模本身不能替代方法验证；若数据不公开，应避免承诺“发布新 benchmark”。
3. **经验贡献**：系统检验闭环随机化、训练监督与运行时条件的不同作用，并设计从目标本体仿真微调到 0/5/10/20 示范真机适配的证据链。只有真机结果补齐后，才能把真实迁移优势写成已验证贡献。

不能写“首次用 latent 连接世界模型和动作”。[LD4WAM](https://arxiv.org/abs/2608.22403) 已明确研究 latent dynamics→action；ReLAC 的差别应定位到**仿真显式角色监督、当前观测因果读取、关系/意图的组织及空间引导**。也不宜把“用了 object token”作为唯一新颖性；[Slot Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html) 和 [FOCUS](https://arxiv.org/abs/2307.02427) 已覆盖对象中心表示与世界模型的基本思想。

### 2.3 核心假说和可证伪预测

设当前观测为 \(o_t=(I_t^{1:C},q_t,\ell,m_t)\)，其中包含 RGB、本体状态、指令和有效维度 mask。令 \(z_t\) 表示新增关系 latent，动作来自 \(\pi_\theta(V_t,z_t,q_t,\ell)\)，其中 \(V_t\) 是仍然保留的 WAM 全局表征。

假说是：对于保持目标身份、可达性和任务含义的扰动 \(\mathcal T\)，关系监督可能使 \(z_t\) 和最终控制对无关外观更稳定。这里不能直接声明 \(z(\mathcal T(o))=z(o)\)，也不能从损失形式推出跨域不变性定理。改变目标颜色指代、遮挡接触点或移动障碍物时，动作本来就可能需要变化。

对应的可检验预测有：同增强/同数据下的完整方法优于 Vanilla；有监督优于同容量无监督 token；空间引导能改善角色定位与其下游查询的一致性，并在闭环中带来收益；适配优势随真实示范数变化仍能保留。若结果仅支持第一项或仅表现为训练正则化，论文应据此收窄机制主张。

## 3. 当前方法：按源码而不是旧文档重建

权威源码：[`relational_sidecar.py`](../openwam/model/modules/relational_sidecar.py)、[架构接入](../openwam/model/architectures/dual_system/relational_sidecar.py)、[标签构造](../openwam/dataloader/ea_rsig_lite.py)、[双轨损失](../openwam/model/modules/ea_rsig_v3.py)。部分旧文档把 Grounding 简写为 EEF/Object；**当前 Sidecar 的四个定位角色实际是两个臂对齐轨道的 Object/Goal，EEF 来自 proprio。** 论文按当前实现写，不照搬这个简写。

### 3.1 基座保留什么

OpenWAM 的 Wan 视频主干与 Action 主干通过 MoT 联合注意力交换信息。当前视频主干有 30 个 block，新增模块在零起算 7/15/23 后接入，即第 8/16/24 层。不是把 WAM 缩成三层，也不是新增另一套完整视频模型。

保留原 Video/Action Flow Matching、80D 状态/动作槽位、相机布局、语言编码和原 WAM 直达 Action 的路径。未来视频生成训练仍在；部署时可以不解码视频，但不能把“不输出像素视频”误写成“不计算世界模型”。

### 3.2 15 个 token 的真实职责

| 类型 | 数量 | 排列/读出 |
|---|---:|---|
| Grounding | 4 | Object-L、Goal-L、Object-R、Goal-R；二维热图、presence、每臂 track-valid |
| Relation | 4 | EEF–Object-L、Object–Goal-L、EEF–Object-R、Object–Goal-R；near/contact/proximity |
| Summary | 1 | 共享交互摘要；无独立百分比进度标签 |
| Plan | 6 | horizon 8/16/32 × 两臂；每 token 同时读出 EEF/Object 的方向与 moving |

token 宽度 512、8 个注意力头，三处接入共享同一组持续更新的 token，不是 45 个独立 token。新增参数约 24.29M，其中共享辅助 heads 约 0.53M。这个参数量不能直接换算为端到端延迟或 FLOPs 降幅。

初始化由可学习 role embedding、masked text mean 和“当前归一化 proprio＋维度 mask”的投影组成。不存在的臂对应 token 被 mask；不是用仿真 object ID 或真实世界不可获得的目标位姿初始化。

### 3.3 Read、Refine、Write 的计算与因果边界

每处 Read 只提取受保护的当前帧 \(H_t^{(l)}\)，投影到 512D 并加二维位置/视角编码；token 以 query 读取。随后全连接 token self-attention 和 FFN 进行 Refine。Action hidden 作为 Write 的 query，从 latent 读取 K/V，以 \(\beta_l\) 门控残差写入。

初始 Read gate 为 .01，Write gate 为 0，输出投影保持非零随机初始化；因此模型初始化时新增写入不改变继承的动作前向。双零初始化会阻断 gate 学习，所以不能把零 gate 理解为全部新参数都为零。

当前帧保护使用原 `mutual + first_frame_causal` mask 和 clean-prefix 时间调制。测试中有独立扰动未来视频、action noise、视频/动作 timestep 和 padding text 的因果检查。论文还应保留一个正式可重复的实际模型干预表；tiny 测试不替代全部真实部署情形。

关键限制：Sidecar 没有直接读取真实未来帧或生成未来帧，也没有独立的历史记忆/速度输入。它从预训练和联合训练后的**当前帧特征**预测专家轨迹中的未来运动。不能把多层深度读取称为“看到了三段未来”，不能把 observation-conditioned future prediction 称为任意候选动作的反事实动力学模型。

辅助 head 的数值输出在基础版不直接回灌 Action；写入的是 latent。因而关闭一个预测 head 与重置相应 latent 含义完全不同。当前 Refine 只有有效性 mask，没有显式 EO/OG typed graph adjacency；给 token 命名并不保证信息不在其他 token 或 WAM 中冗余编码。

### 3.4 Grounding-guided Read：已经实现，结果仍需独立验证

当前源码中已存在，不应为了写论文再实现一套。第一个 tap 保持原样；后两个 tap 用前一 tap 预测的 Object/Goal 热图，给 Relation/Plan 的视觉 attention logits 加空间 bias。Grounding/Summary 仍自由读取，原 WAM 全图上下文也保留。

对一个角色，设热图为 \(g_i\)，可预测置信度为 presence × track-valid × observed-arm-valid，则先与均匀分布混合得到 \(\tilde g_i\)，再使用

\[
b_i=\eta\log\{(1-u)N\tilde g_i+u\},\qquad u=.5.
\]

\(N\) 为当前帧 patch 数，训练前 1000 个 joint optimizer update 将 \(\eta\) 从 0 线性升到 .5。EO 查询使用对应 Object prior，OG 与 Plan 查询使用 Object/Goal 均值 prior。预测不确定时退回均匀上下文，**不把 GT 框塞进 inference attention**。

这是真正把“定位”接到“后续读取”的路径，但依然是软引导，不是硬裁剪；错误热图可能误导下游，置信度也未必校准。需要 Guidance-Off、Write-Off、定位误差与闭环结果共同验证。当前 76.02% 和 73.13% 均不属于引导版，不能用它们证明此扩展有效。

源头与已有训练记录：[`GROUNDED_READ_CLUSTER_RUN_20261103.md`](GROUNDED_READ_CLUSTER_RUN_20261103.md)。本次只使用已落盘记录，不把历史 Slurm ID/进度文件当作当前实时存活证明。

## 4. 数据生成：论文必须讲明白的监督合同

### 4.1 从专家轨迹到训练窗口

原生环境生成并验证专家轨迹，保存 RGB、当前 EEF、本体/相机参数、交互对象/目标角色、对象姿态、接触与可见框。离线转换只生成数值监督和清单，不改写原 HDF5；数据来源、构造代码、文件大小/mtime 或 hash 与 manifest 绑定。

每个窗口取当前 proprio，未来 32 个动作；视频帧锚点为 0/4/8/…/32 共 9 帧，意图标签锚点为 8/16/32。尾部不足的未来通过 mask 排除，不能把补齐的静止尾巴当成正确的未来运动。

图像是高 384、宽 320 的画布：上方头部/前视 256×320，下方两个腕部各 128×160。ManiSkill/单臂缺失第三视角在拼图时补黑；legacy midtrain ColorJitter会使该槽变灰，不能声称所有训练阶段增强后始终纯黑。bbox 先在源视角内归一化，再映射到实际 L 形画布和 runtime patch 网格；不能默认固定 14×14。

### 4.2 标签的精确定义

对每臂交互轨道，仿真给出 \(p_e,R_e,p_o,R_o,p_g\)：

- Grounding：Object/Goal 可见 bbox 映射为分视角截断的二维 Gaussian soft target；另监督角色 presence 和 track validity。
- Relation：\(r_{eo}=R_e^\top(p_o-p_e)\)，当前只由其距离构造 \(\mathrm{near}=\mathbb 1[\|r_{eo}\|<.10\mathrm m]\)；另有接触二分类和 \(\mathrm{prox}=\mathrm{clip}(1-\|p_g-p_o\|/.5,0,1)\)。**当前没有对完整相对 XYZ 向量的回归 head。** near 也不是接触的逻辑前提，参考点不在接触面时可出现 contact=1、near=0。
- Plan：对 role \(r\in\{\mathrm{EEF},\mathrm{Object}\}\)，\(\Delta^{r}_{t,h}=R_r(t)^\top[p_r(t+h)-p_r(t)]\)。位移超过 1mm 判 moving；moving 为真时才监督单位方向。要求两端有效且 `role_id` 相同，角色切换时不跨对象计算方向。

独立辅助 target 为 130D：track-valid 2、relative XYZ 6、contact 2、contact-valid 2、goal proximity 2、goal-valid 2、direction 36、moving 12、future-valid 6、bbox 48、bbox-valid 12。**这个 130D 不拼入 80D 状态输入。**

不可见目标、无效轨道、缺失臂、不可获得接触标签与尾部未来要分别 mask。目标距离不是任务进度百分比；进入目标区域并不总等于释放、稳定放置或多阶段任务完成。

### 4.3 动作统一的范围

每臂 EEF10 为位置 3、旋转矩阵前两列 6、夹爪 1。左右臂映射到模型 80D 的 0:10、34:44，其他槽位和缺失臂的损失无效。部署必须使用该 checkpoint 的训练统计；四元数读取 wxyz，客户端/ROS 所需 xyzw 在边界显式转换。

三个来源的编码相同，但标签语义并非完全相同：

| 来源 | 当前 state | future action 标签 | 不能忽略的差别 |
|---|---|---|---|
| RoboTwin Aloha/Piper | 当前实际 endpose reference | 专家未来实际到达的绝对 EEF | 推理将预测到达位姿再当控制目标交给规划器，不保证重执行等于原专家轨迹 |
| ManiSkill | 当前实际 Panda TCP | 从 `pd_ee_delta_pose` 命令重建的绝对控制目标 | 训练原生 20Hz；目标与 achieved 不能互换 |
| RLBench | 当前实际 base-frame EEF | joint target 经对应 FK 得到的绝对 EEF 控制目标 | FK/工具点、本体和时间对齐需按来源记录 |

RoboTwin 的辅助关系 TCP 使用局部 x 方向 .12m 偏移；控制 endpose 不叠加这个偏移。32 个原生控制步也不必在所有来源代表相同秒数；不能根据视频 fps 猜 RoboTwin 控制频率。

### 4.4 两条训练数据支线必须分开

**Aloha Clean-only 专项线**：50 任务，每任务 50 轨迹，共 2500 条、552275 帧、549775 个训练窗口。数据目录名的 4350 不是此实验实际轨迹数。当前 76.02% 属于这条训练线，不是 multi-sim midtrain 的下游效果。

**多仿真/多本体 midtrain 线**：冻结归档 52204 个外观轨迹版本，train/val/test 为 39816/6161/6227。重复外观版本不等于独立物理轨迹。

| 来源 | 任务 | Train | Val | Test | Train 窗口 |
|---|---:|---:|---:|---:|---:|
| ManiSkill Franka | 8 | 16000 | 2000 | 2000 | 2671589 |
| RoboTwin Aloha | 50 | 9584 | 1227 | 1189 | 2124695 |
| RoboTwin Piper | 35 | 9566 | 1172 | 1262 | 1404781 |
| RLBench Franka | 18 | 4666 | 1762 | 1776 | 1080908 |
| 总计 | 111 个 source-task 配对 | 39816 | 6161 | 6227 | 7281973 |

111 不是 111 个互不重复的任务语义，Aloha/Piper 有任务重叠。四来源按窗口等权混采，名义各 25%，不是按 episode 数量比例抽样。“一个 mixture epoch”也不等于每个来源全部遍历一次。midtrain 最终 56892 microstep / accumulation4 = 14223 optimizer updates；之后 ManiSkill 目标微调 20872 microstep = 10436 updates，Piper 两 epoch 21952 microstep = 10976 updates。

split 要按物理家族、重复外观版本、EEF 轨迹签名等分组。ManiSkill 评测清单已有 ID/family/physical-seed/numeric/media 与 train 交集为零的记录；本次另外从四个 worker JSONL 重算了 1600 个唯一 ID 的结果。不同目录名/不同 seed 数字本身不能证明无泄漏。

实际多源midtrain采用4卡×32×accumulation4/global512，WAM LR2e-5、Sidecar LR1e-5，前500个optimizer update仅延迟WAM更新，使用普通ColorJitter(.2/.2/.2/0)。之后ManiSkill目标适配使用4卡×32×accumulation2/global256、1epoch、WAM LR1e-4/Sidecar LR5e-5、完整PatchWAM增强，**不重复冻结WAM 2K**。不能把Aloha专项的2K warmup或appearance配方无差别套到所有阶段。

midtrain还有一个具体历史观察域差：`PreparedSimDataset`先拼图补黑，再调用不带缺失视角mask的ColorJitter，contrast<1可使黑槽变灰。既有真实样本检查达到7–29/255；当前源码调用仍如此。后续PatchWAM路径显式传入相机有效性，但不能追溯修改已经训练的midtrain权重。此项不是已证明的成功率致因；不能在论文中以“所有阶段缺失槽完全一致”掩盖它。证据见[1epoch启动与读取审计](MULTISIM_1EPOCH_B512_LAUNCH_20261025.md)。

multi-sim 与 ManiSkill 目标训练包含随机化域，因此 HardRandom 结果应称为**已见任务及随机化类型中的 held-out physical-family 评估**，不能称“只用 Clean 训练后零样本面对 HardRandom”，也不能称未见任务泛化。

构造/归档记录见 [`MULTISIM_SIDECAR_MIDTRAIN_20261024.md`](MULTISIM_SIDECAR_MIDTRAIN_20261024.md)、[动作/存储审计](MULTISIM_MIDTRAIN_ACTION_AND_STORAGE_AUDIT_20261025.md)、[精确组成](MIDTRAIN_DATA_COMPOSITION_20261102.md)。归档训练可读与“全部原生重处理/回放依赖已经独立闭包验证”仍是不同的验收范围。

## 5. 训练目标、优化经验和已经失败的假设

基础联合目标为

\[
L=L_V+L_A+0.5[.025G_2+.01R_2+.05P_2+.1(.025G_0+.01R_1)].
\]

\(G\) 为位置 soft CE＋presence/track BCE，\(R\) 为 contact BCE＋.5 near BCE＋.5 proximity smooth-L1，\(P\) 为 moving BCE＋masked cosine direction loss。下标 0/1/2 对应三个 tap。完整 joint 时，EA 可沿 Read 影响视频主干，Action 可沿 Write 影响 latent；小 Write gate 不意味着辅助梯度不会影响底座。

最终 appearance+ramp 42960 方案从 Foundation 开始，四卡 ×32×accumulation2/global256，AdamW 的 WAM LR1e-4、Sidecar LR5e-5、weight decay .01、cosine、5% LR warmup、BF16、ZeRO2；前 2000 个 optimizer update 暂不更新 WAM，之后用 100-update WAM LR ramp。**这 2K 包含在总 21480 updates/10epoch 内。** 完整反向、clip 和“跳过 optimizer 参数/动量/衰减更新”不等于关闭 WAM 反向，也不等于只把 LR 设零。

原 Full21480 则从一个已完成 2K 预热的权重重载，重建 optimizer 后额外 joint10epoch，总曝光约 10.93epoch；两者不能只按文件名 21480/42960 排序训练量。C2R-init10740 又是官方 C2R 起点、2K 新 Sidecar 预热、低 LR 额外 joint5epoch。

### 5.1 应保留在论文讨论中的经验

| 经验 | 证据与含义 | 不应推出 |
|---|---|---|
| 训练 loss 降不保证随机化 SR 持续升 | 旧连续训练 32k→最终42960 的 Clean/Randomized 取舍；多轮离线/闭环不一致 | 越晚 checkpoint 必然更好 |
| gate 符号/大小不是贡献指标 | 既有真实残差测量、Full/Off 同输入 FM 比较 | 负 beta 有害，或放大 beta 一定有效 |
| 监督可能主导共享梯度 | 若干旧小样本 probe 的 EA/Action 梯度比很大 | 仅凭标量 loss 就确认梯度冲突；少量 episode 代表整套训练 |
| 强外观增强值得作为强基线 | PatchWAM 参考 recipe 已接入；当前组合 Randomized 较高 | 所有增益都来自关系 token，或强 hue 一定保留颜色指令语义 |
| 运行时移除某些通路可能不掉分 | 19 组诊断及两个 checkpoint 的 Refine-Off 方向不同 | 模块训练无用，或每个命名 token 都必须不可替代 |
| 表征拟合不等于控制 | 训练 Clean 窗口亚毫米差、接触关键阶段和闭环恢复是不同量 | 小 XYZ 平均误差证明任务可完成 |

PatchWAM 这里只作为外观 randomizer 的代码来源，**没有把 WAM 主干替换成 PatchWAM 模型**。来源固定 revision `dea19d21182a26a8ee16efc2c4125ee7aecf44fb`，说明见 [接入记录](PATCHWAM_C2R_AUGMENTATION_20261027.md)。三相机独立采样、同相机 clip 时序共享；hue/AdaIN/FFT 对颜色指代和跨视角一致性的风险继续保留。

XA 同时改变 alignment 与增强，修正渲染异常条件后的全任务速测为 72.53%，不能沿用旧 72.60%，也不能据此单独认定 alignment 无用。三阶段全 EA 监督、heads-only、梯度退火属于不同训练变体，不能把它们拼成最终单一方法的默认配方。

## 6. 当前仿真结果：哪些可以直接填进初稿

### 6.1 RoboTwin 全 50 任务、双域、每条件 100 条

以下均覆盖 100 个任务×域条件；不是 100 个不同任务。本文对原结果逐条件重算，检查汇总与成功数一致；Full21480 和最终42960均有已落盘的独立原生日志/seed验收。官方行来自作者发布表，属于外部参考。

| 方法/配方 | Clean /5000 | Randomized /5000 | 综合 /10000 |
|---|---:|---:|---:|
| 官方 OpenWAM-C2R 发布参考 | 4468，89.36% | 2436，48.72% | 6904，69.04% |
| 旧 Full-backward32k | 4401，88.02% | 2618，52.36% | 7019，70.19% |
| 原 Full21480，额外 joint10epoch | 4439，88.78% | 3093，61.86% | 7532，75.32% |
| 最终 appearance+ramp42960，总10epoch | 4401，88.02% | 3201，64.02% | 7602，76.02% |
| Vanilla＋相同增强/训练预算 | TBD | TBD | TBD |
| 相同容量但无 EA 监督 | TBD | TBD | TBD |
| EA-only 训练、不写入 Action | TBD | TBD | TBD |
| ReLAC＋Grounding-guided Read | TBD | TBD | TBD |

最终方案相对发布参考综合 +6.98pp、Randomized +15.30pp、Clean −1.34pp。相对原 Full21480 综合 +.70pp、Randomized +2.16pp、Clean −.76pp。**这些是配方层面的描述性差值，不是关系模块的单变量因果增益，也没有多训练 seed 的显著性证明。**

本地 100 条采用 native30＋native70 接续，保留官方专家筛 seed、任务预算与成功判据；每条件 100 个唯一有效 seed。但分段语言 RNG/指令池 cap 与连续 native100 不同，且官方发布表声明 seed0、本地有效 seed 从100000系列开始。评测参考价值不等于逐 episode、语言、噪声严格配对。

旧摘要曾列“68条件 83.15%”或“71条件 77.70%”；这些只是阶段子集。现在已有完整全套，应以 75.32%/76.02% 替代对这两个权重全套表现的旧认识，不能把高的子集值移进主表。

权威路径：

- [`full21480_remaining32_supp70_20261104/comparison.json`](../outputs/robotwin_eval/full21480_remaining32_supp70_20261104/comparison.json)。
- [`patchwam_ramp100_final42960_supp70_20261103/comparison.json`](../outputs/robotwin_eval/patchwam_ramp100_final42960_supp70_20261103/comparison.json)。
- [本文冻结的逐条件 CSV](../papers/relac_cvpr2027/evidence/robotwin100.csv)。

### 6.2 运行时消融：两个 checkpoint 的结论并不相同

| 权重与运行方式，均50任务×双域×30 | Clean | Randomized | 综合 |
|---|---:|---:|---:|
| 原 Full21480 / Full | 88.67% | 61.00% | 74.83% |
| 同权重 / Refine-Off | 89.40% | 62.00% | 75.70% |
| 最终42960 / Full | 87.87% | 64.53% | 76.20% |
| 同最终权重 / Refine-Off | 88.67% | 62.27% | 75.47% |

Refine-Off 只关闭三个 token self-attention 残差分支，保留 Read、FFN、Write。在旧权重上综合 +.87pp，在最终权重上 −.73pp、Randomized −2.27pp。因此“关掉 Refine 必然更好”不能作为结论。新42960 Refine-Off 的100个最终有效attempt，已在本次独立检查 run exit0、native分母/成功率、30唯一seed和client日志渲染错误；证据在 [`refineoff42960_independent_audit.json`](../papers/relac_cvpr2027/evidence/refineoff42960_independent_audit.json)。这仍不补回历史未冻结的语言/噪声配对，也不构成显著性证明。

旧 10 任务、每域10条的19组诊断中 Full77%、RelationReset81.5%、RefineOff81%、WriteOff80%、SidecarOff77.5%。这是3800条总诊断，但每组只有200条，且任务被选择、候选很多。它更适合发现机制问题，不能替代全任务排名或在同一测试集反复选择最终方法。

All-Reset 可经 LayerNorm/MHA bias 保留常量残差，不能等同 Sidecar-Off；Sidecar-Off 又保留联合训练后的 WAM，不能等同从 Foundation 训练的 Vanilla。关闭辅助数值 head 的输出本来不改变基础版动作，不能用这种“消融”说明对应训练目标没有贡献。

最终42960 Sidecar-Off 在本次读取时仍有未完成条件，因此初稿不放阶段成功率、不按已有易任务子集给它排名。后续完成结果应作为新快照替换占位，不能自动使用不断变化的 live 数字。

### 6.3 ManiSkill：1600 条原始记录重算

当前 archive8任务协议：每任务100 Clean＋100 HardRandom，32步预测、replan8、denoise10、600步预算；Clean来自 held-out val＋test，HardRandom来自test且D25%/M75%。这是定制数据/环境协议，应称 **ManiSkill3-based eight-task suite**，不能冒称官方统一排行榜。

| 任务 | Clean /100 | HardRandom /100 |
|---|---:|---:|
| Lift peg upright | 72 | 72 |
| Pick cube | 93 | 94 |
| Pick cuboid | 80 | 70 |
| Pick sphere | 92 | 81 |
| Place sphere | 36 | 41 |
| Pull cube | 63 | 67 |
| Push cube | 80 | 76 |
| Stack cube | 73 | 80 |
| 总计 | 589/800＝73.63% | 581/800＝72.63% |

合计1170/1600＝**73.125%（四舍五入73.13%）**。结果说明此接口可用于单臂闭环；没有相同预训练、增强、数据预算的 baseline 结果，因此不能单凭这个数字写“提高了多少”。尤其 place_sphere 只有36%/41%，当前真机任务若选它必须正视仿真能力本身不足。

历史六任务：Sidecar25410为587/600，baseline21180为576/600；训练预算不匹配，环境、任务、随机化和200/600步协议也与当前八任务不同。DiT4DiT旧六任务200步协议的v1/v2高分同样不能与当前73.13%直接比。历史经验可以解释设计来源，不能并列为本文主表基线。

来源：[原始 summary](../outputs/maniskill_eval/final20872_l20_gpu3_novideo_20261102T130608Z/run/summary.json)、[逐任务冻结 CSV](../papers/relac_cvpr2027/evidence/maniskill.csv)。本次直接重算四worker的1600行；未重新运行模型。

Piper已有模型与原生评测框架，本地最近可用的全套评测记录仍不足以证明35任务×双域×100已经全部完成。论文保留TBD；不把click_bell双域各1次smoke或部分任务高分当全套性能。

## 7. 从仿真到真机：怎样形成可发表的闭环

### 7.1 实验链条及每一步能证明什么

建议主线：**同一 Foundation → multi-sim relational midtrain → 目标本体仿真微调 → 0条目标真机示范 → 5/10/20条示范适配 → 独立真实测试。** 另一支是 Foundation 直接做相同目标仿真微调和相同真实适配，用来衡量 midtrain 的增量。

目标平台优先选已有可验证控制和相机接口的一套单臂系统，完成2–3个任务；双臂Aloha仿真提供协作复杂度证据。Piper真机若硬件/遥操作/标定更成熟可作为后续扩展，但不必为了故事同时启动两套硬件。共享网络/训练协议不等于一个权重零样本适配所有本体。

0-shot精确定义：**0条目标真实任务动作示范用于适配**。如果使用真实相机标定、手动选择场景、目标域图像或真机验证集选checkpoint，都要独立记录。它不表示基础模型没有真实训练数据，也不表示完全没有目标域接触。

### 7.2 已有真机诊断不能被叙事跳过

目前已验证的是HTTP、固定输入、有限动作输出、部分FK/IK软件链路及独立客户端；这些不证明物理任务成功。最新单臂诊断还指出：

- 一个真实place_sphere初始EEF与该任务最近训练初始XYZ约差158mm；其朝向与全部训练初始旋转最近差70.56°。虽然归一化未越界，任务条件状态仍明显偏离训练分布。
- 视觉域有目标marker、视场、尺度和腕部夹爪可见性差别。替换图像/状态会改变预测，但物理不一致的离线拼接只能证明敏感性，不能定位单一真实失败原因。
- 同一训练scene、无cache HTTP模型、20Hz虚拟控制、300步诊断中，prefix1/4/8未完成，prefix16/32分别228/220步成功。单scene训练内证据不能外推八任务，且该实验没有使用正式600步预算，也不模拟真实HTTP等待期间物理继续运行。
- 训练20Hz下32步是1.6秒；旧参考真机逐点等待会把它拉长到数十秒。控制点频率、插值、重规划频率、夹爪时序和推理延迟要分别量测。短前缀并不总能推进接触阶段，长前缀又可能增加开环误差，必须实际比较。
- 参考工程的arm-base/mobile-base、link8/TCP、姿态IK检查与夹爪反馈问题有软件修复和离线证据，但现场TF、物理TCP/桌面、碰撞与控制器响应仍未全部确认。

**较新的抖动记录更新了早期判断。** 11月4日下午绿色指令的新初态已经接近训练初态，模型也出现闭合/抬升预测，但现场仍抓空和抖动；不能继续用“早期初始姿态偏离”解释全部剩余问题。20个绿色请求chunk中，方向反转中位20次、路径累计92.67mm但首尾净位移7.44mm。相同20Hz单场景仿真实际TCP二阶差分中位为模型1.008mm、专家.136mm；这说明目标/执行轨迹存在粗糙度，终点误差小或仿真success都不能证明中途平滑。

11月5日新增的已落盘精度诊断进一步报告：全2671589个原始训练动作空间二阶差分RMS为.7940mm；仅经原归一化、BF16舍入、反归一化就增至2.5735mm。复位附近x/z的BF16格点约2.707/2.919mm。144个train/test窗口的reader/部署拼图、proprio、prompt、mask和action对齐检查通过，只排除该范围的软件接口差异。单scene六组10/20/30去噪×BF16/FP32动作累积均成功，20步较10步的一个粗糙度指标下降，30步未继续改善；完整936次输入配对与288次训练checkpoint比较在本次读取时尚待最终冻结。

因此，应将**语义/空间选对目标、预测精度与时序连续性、低层跟踪/接触**作为三条相互作用但不可互相代替的证据链。关系定位更好并不自动消除BF16台阶或控制器振荡；FP32动作累积也尚未证明能根治问题，更不能从仍在下降的action loss推断“多训一个epoch必然修好”。以上后续诊断没有被拼成总体成功率或论文新增方法。来源：[11月4日抖动/录像诊断](MANISKILL20872_JITTER_AND_VIDEO_DIAGNOSIS_20261104.md)、[11月5日精度/训练阶段诊断](MANISKILL20872_DENOISE_TRAINING_JITTER_DIAGNOSIS_20261105.md)。

应引用：[真机问题清单](MANISKILL_REAL_ROBOT_ISSUE_CHECKLIST_20261104.md)、[pose/chunk审计](MANISKILL20872_POSE_CHUNK_ACTION_AUDIT_20261104.md)、[新客户端](MANISKILL_CLEAN_CLIENT_20261104.md)。这些是控制合同与实验有效性问题，不应包装为关系表征已经解决的能力。

### 7.3 Few-shot实验的最小可执行设计

每任务建议先固定20个独立真实测试初始化，包含物体位置、背景/光照、干扰和目标位置；根据实际资源增加次数。0/5/10/20是训练示范数，测试重置数单列，不能把两者混淆。测试条件不得用于挑超参或删失败。

同一任务的5⊂10⊂20示范集嵌套；不同方法共享同一组示范、适配步数、动作语义、数据增强和评测重置。至少报告按任务成功数/分母，以及跨示范子集/训练seed的不确定性；一次20次试验的二项区间不能代表训练随机性。

三条曲线足以回答主要问题：Foundation＋target sim SFT；multi-sim Vanilla＋同target SFT；multi-sim ReLAC＋同target SFT。如果算力无法覆盖三条，应明确哪项归因尚未完成，而不是把数据/预训练/模块增益合并。

真实示范没有仿真oracle：接触、bbox、未来Object轨迹等不可得时，对应辅助loss必须mask；不能把缺失项填0当负例。可先做主任务适配并明确哪些参数更新；如使用检测器、人工标注或源仿真回放，要把成本和来源单列。

真实失败分类至少包括目标识别/定位、接近/抓取、持物、放置/释放、碰撞/IK、时间同步、恢复失败。成功判据预先定义为任务完成和稳定保持，而不是模型head超过某阈值。

## 8. progress与扩展future plan：怎样补全愿景而不改写既有事实

以下均为**论文设计提案，未在本次修改或训练**。若要落地，需按AGENTS规则单独说明并确认核心改动，不能为补论文段落擅自改变当前训练。

### 8.1 Progress应表示任务里程碑

不要默认用 \(t/T\) 作为进度：不同速度的同一状态会有不同标签，重试/绕障/交接也可能非单调。更合适的是任务定义的里程碑向量，例如对象已被正确夹持、对象离开支撑、已运输到目标、已释放且稳定；双臂任务可用部分有序的协作事件，而非强制所有任务四阶段串行。

保留Summary latent，为其增加任务条件的事件/progress readout；训练使用仿真状态构造的有效里程碑mask和BCE/有序分类。若对计划也使用进度，应通过该预测latent/概率显式接入，不能只在图里画一条不存在的箭头。GT进度同样只用于loss。

必须验证同一物理状态在变速、暂停和恢复中预测是否一致，以及它是否改善闭环；只展示沿专家时间上升的曲线会被审稿人质疑为时钟捷径。若最终没有独立实验证据，主文仍称interaction summary，进度留在future work。

### 8.2 Future plan可以逐步增加可执行语义

当前direction＋moving已提供轻量运动意图，但EEF向目标走多少、怎样旋转、何时夹紧/释放对接触任务很关键。一个适度扩展是对每臂、每horizon预测局部位移幅度/相对旋转与夹爪事件，保留对象future-valid与role-switch mask。不能把需要任意新对象轨迹GT的方案默认为真实数据可得。

如要声称“预测候选动作导致的未来关系”，则必须显式输入候选动作、训练对应transition模型并验证干预/rollout；仅从观测预测专家未来不是这个问题。初稿没有宣称已经做到该扩展。

计划的数值head是否直接用于Action也是独立设计选择：当前latent写入保留更高维信息，但可解释数值只作监督。若将预测几何作为显式动作条件，应比较latent-only、numeric-only、combined，检验预测误差传播，而不是先假定数值更可解释就一定更好。

## 9. 最少还需要哪些实验，才能把初稿变成强论文

| 优先级 | 对照/证据 | 解决的审稿疑问 |
|---|---|---|
| P0 | Vanilla＋相同PatchWAM增强、数据、共享WAM更新预算与完整评测 | 提升是否只是增强/额外微调/更好底座 |
| P0 | 同容量token、无EA监督 | 新参数/训练容量是否足以解释收益 |
| P0 | EA-only（训练期监督、关闭Action Write）与完整方法 | 训练表征价值与在线条件价值能否分开 |
| P0 | 已有Grounding-guided版 vs 同起点、同增强无引导 | “局部聚焦”是否有实际计算路径和闭环收益 |
| P0 | 至少一套真机、2–3任务、0/5/10/20示范成对测试 | sim2real/示范效率是否真实成立 |
| P1 | 冻结独立manifest，更多训练seed/任务bootstrap与原始结果 | 小差值是否稳定，是否反复在测试集选最优 |
| P1 | 预测定位/关系/意图质量＋一致性/失败阶段分析 | latent是否学到所声称的角色，以及错误怎样影响控制 |
| P1 | 同设备/同输入的端到端延迟、显存、失败率 | 24M参数与三次读写的实际代价 |
| 条件项 | 独立progress监督/运动幅度旋转事件扩展 | 只有决定把这些写成贡献时才必须补 |

预算说明要拆成总样本曝光、WAM optimizer updates、Sidecar updates、预热和计算时长。建议主因果对照从同一Foundation/WAM状态开始，以相同joint数据和WAM更新预算训练；额外Sidecar-only预热单列，另可给Vanilla额外训练的计算预算对照。不要用“seed相同”声称不同架构的增强/noise流全程逐bit相同。

当前matched Vanilla＋PatchWAM作业和Grounding-guided作业已有配置及落盘进度证据，但本次没有验证其最新远端终态或闭环结果；不能填造数值。新baseline在初始化、冻结预算等方面的差别也需最终以resolved config核对。

## 10. 审稿问题与应答原则

| 可能的问题 | 用什么回答 | 需要避免的说法 |
|---|---|---|
| 这是不是普通auxiliary loss？ | 同容量、EA-only、完整写入、引导版的训练和闭环对照 | 仅给token命名就宣称新关系推理 |
| Relation/Plan真的被Action使用吗？ | 训练消融＋运行时干预＋逐阶段预测/动作分析 | 只有预测准确率，或把所有reset等同语义删除 |
| 原WAM本来就有全局注意力，为何还需局部？ | 给出空间先验怎样影响后续查询及其一致性/鲁棒性证据 | 硬说背景无用、原WAM不懂物理 |
| 这些不是已有object-centric/latent world-model思路吗？ | 明确与LD4WAM/FOCUS/Slot Attention在监督、读取和部署接口上的差别 | “首次”泛化表述 |
| 模型是否偷看GT future/object pose？ | forward输入合同、mask机制、未来/噪声干预；GT只进loss | 把未来监督等同部署未来输入 |
| 76.02%是否证明模块优于OpenWAM？ | 它证明当前配方结果；匹配基线补齐后才谈模块增益 | 直接把+6.98pp归因关系latent |
| 单臂是否一样强？ | 报八任务全部结果，包含36%/41%的place_sphere | 只用强任务均值或历史六任务高分 |
| generalization到底是什么？ | 分别定义随机化、物理家族、物体/任务、本体和真实域 | 把所有Randomized统称unseen objects/tasks |
| 真机有多少数据？ | 目标域动作示范、图像/标定/验证接触分别披露 | Foundation见过真实数据却称完全sim-only |
| 有理论保证吗？ | 说明结构和监督的归纳偏置，用实验证伪 | 无证据的充分统计量/因果不变性/信息瓶颈定理 |

[DAgger](https://proceedings.mlr.press/v15/ross11a.html) 支持讨论模仿学习闭环分布偏移，但不能证明本项目已经定位到同一唯一失败原因。[rliable](https://arxiv.org/abs/2108.13264) 支持报告评估不确定性的原则；task bootstrap不能补回缺失训练seed，也不能修复多次测试集选型。

## 11. 初稿结构与图表安排

正文按CVPR英文写作：Abstract；Introduction；Related work；Method；Simulation supervision and training；Experiments；Discussion and real-world transfer；Conclusion。方法集中说明已实现base与独立标注的guided扩展，进度/扩展plan放补充材料的“planned extension”。

正文图表有：完整计算路径图、真实逐任务ManiSkill图、两个权重的Refine-Off对比图；全套RoboTwin表、数据组成表、单臂表、匹配消融空表、真机0/5/10/20空表。空白处写TBD，不画虚构few-shot曲线或假的attention热图。补充材料提供标签表、mask、控制语义、配方时间线、19组诊断、progress设计和定性图的采样标准。

图1应优先让审稿人看懂：RGB/Text/Proprio→WAM→Action主路仍在；当前帧→15latents→三次Action Write；GT supervision只在训练；预测heatmap→后续Read用虚线标明该扩展尚未闭环验证。不要把simulator GT到部署attention画成实线。

真机few-shot图最终应画真实点＋清楚定义的区间，0/5/10/20共用测试条件；目前仅给空表和设计。定性图需要成功与失败、Clean/Randomized/Real一致的同一展示协议；没有实测热图时只放明确标记为schematic的架构示意。

模板严格使用用户zip中的 `cvpr.sty`、`ieeenat_fullname.bst`，保留原始zip全部内容的独立副本；review模式、10pt、twocolumn、letterpaper、原字号/边距和页眉。正文不超过8页，参考文献另计；补充材料单独PDF。不能通过负间距、改字号/边距或压缩样式来伪装满足页数。

## 12. 本次新整合结论、未决项与下一步

本次只做已有证据的读取、重新汇总与论文写作，没有改动模型/训练/数据/部署核心，没有启动或停止已有实验，没有操作硬件。新增的独立Refine-Off审计是文件检查，不是新rollout。

新整合的主要结论：旧21480全100条为75.32%，最终42960全100条为76.02%；二者30条子集表现不能代替完整100条。Refine-Off的符号随权重改变，不能沿用旧诊断的单一叙事。ManiSkill1600条为1170成功，最弱任务必须纳入迁移决策；最新抖动/精度证据说明语义聚焦、连续动作精度和控制执行需要分别验证。Guided Read、显式progress、扩展plan、目标真机优势与模块单独增益均仍需证据。

本次写作已考虑《思考与隐患》中的相关未决项：near-plane .02/.1；achieved/commanded EEF与回放；BF16/write残差与cache近似；角色/目标mask、颜色增强语义；训练预算与fresh optimizer；物理家族split；渲染错误与实际分母；多重比较与训练seed；真实base/TCP/夹爪/频率/碰撞；多来源物理时间和动作语义；归档原生依赖闭包。**除本文具体文件审计范围外，没有因写作把这些项宣布解决。**

下一步优先补齐既有匹配基线和guided版结果，在独立manifest上验证关键差值；真机先固定平台和控制合同，按相同示范预算形成可比较曲线。若结果不支持在线关系模块优势，应如实将贡献定位为训练期关系监督/表征，而不是继续叠加模块或在同一测试集上寻找有利删token组合。

## 13. 一手文献与本地证据入口

- [OpenWAM论文](https://arxiv.org/abs/2609.07398)：底座与WAM预训练；[发布C2R逐任务参考](https://github.com/OpenWAM-Official/OpenWAM/issues/46#issuecomment-6118279946)：外部对照。
- [RoboTwin2.0](https://arxiv.org/abs/2506.18088)、[ManiSkill3](https://arxiv.org/abs/2410.00425)：仿真平台背景；本文定制数据协议另行披露。
- [LD4WAM](https://arxiv.org/abs/2608.22403)：最接近的latent dynamics→action相关工作；[DINO-WM](https://arxiv.org/abs/2411.04983)、[V-JEPA2](https://arxiv.org/abs/2506.09985)：特征预测与规划的相关背景，不能把它们的结果当ReLAC证据。
- [Slot Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)、[FOCUS](https://arxiv.org/abs/2307.02427)：对象中心表示与世界模型。
- [Domain randomization](https://arxiv.org/abs/1703.06907)、[Diffusion Policy](https://arxiv.org/abs/2303.04137)、[Flow Matching](https://arxiv.org/abs/2210.02747)：训练/控制相关基础。
- [既有完整实现审计](OPENWAM_SIDECAR_IMPLEMENTATION_EXPERIMENTS_AND_DIAGNOSTICS_20261102.md)、[后续机制分析](SIDECAR_NEXT_STEPS_DIT4DIT_V1_V2_V3_EVIDENCE_20261103.md)、[真机部署复核](SIM2REAL_READINESS_REVIEW_20261031T171430.md)、[风险清单](思考与隐患.md)：用于衔接历史经验；较早的阶段成绩由本次原始结果冻结快照更新。

一手网页核查日期为2026-11-05。参考文献不虚构录用信息；未核验正式发表venue时，LaTeX按arXiv记录引用。所有当前实验数值在 `evidence/results_snapshot.json` / CSV中有来源；导出脚本不会运行策略。
