# 下一轮 Sidecar 实验：三阶段完整监督与逐层插入的取舍

日期：2026-10-31 UTC。授权范围：重读报告、分析实现与提出方案；未修改模型/训练配置，未启动训练。

## 建议

若用户希望在当前 XA 完成后、闭环评测进行期间，用同一8卡探索一个独立的 Sidecar 方案，首选：**原始 warmup2K + 原成功版完整 PatchWAM + 三阶段完整辅助监督，暂不开 XA**。保持3个Read/Refine/Write位置和共享heads，只改变各阶段辅助loss分配。此为低改动、结果较易解释的结构假设检验，不是已验证的成功率提升方案。

这与此前“恢复完整增强+XA”的建议服务于不同问题：后者是现有XA主线最有直接外部依据的延伸；本次推荐是用户新提出的Sidecar设计的独立实验。仅有一个名额且希望独立于XA最终结果推进时，优先三阶段完整监督；不把两个改动叠在同一次训练。

## 核对了什么

- docs/deep-research-report (5).md 全文；report (1) 的 per-channel gate、更多heads/Write层风险；report (2)/(4) 的Grounding-guided Read、2.5D关系、progress标签与增强建议；初版report的历史梯度/优化假设。report (2)/(4) SHA相同，不算两份独立证据。
- openwam/model/modules/relational_sidecar.py 的 SidecarStage、EAHeads、共享head、15tokens与Write；openwam/model/architectures/dual_system/relational_sidecar.py 的tap_blocks、前向、辅助loss聚合；相关测试入口。
- docs/思考与隐患.md、最新XA与旧21480趋势文档。没有新增GPU诊断，最新XA观察仍以既有快照为限。
- 复查PICA-WAM原文 https://arxiv.org/html/2610.26813v1 ：支持增强与Action对齐组合及paired诊断；不提供我们“三层全监督”或“24层Sidecar”的直接消融证据。报告内部不可解析cite标记不能当可追溯原始出处。

## 当前结构的关键澄清

不是“浅层只能预测Ground，中层只能预测Progress，深层才能预测Plan”。

1. Transformer blocks 7/15/23（零起算，即第8/16/24层）各做一次Read/Refine/Write。每次更新同一组持续传递的15tokens：4Ground、4Relation、1Summary、6Plan。
2. 每个stage输出后都调用同一个EAHeads，预测Ground、Relation和Plan。每个stage也都把全部有效tokens作为Action Write的KV；并非浅层只写Ground、中层只写Progress。
3. 实际辅助loss是：最终stage全部任务 + .1×浅层Ground + .1×中层Relation。浅层Plan、中层Ground/Plan缺直接loss，但不是完全无梯度：深层EA和主Action路径仍可经持续tokens反传。
4. 这里中层Relation包括near/contact/goal proximity，不是已经定义完备的任务阶段Progress分类。Summary token参与交互，但当前EAHeads没有独立Summary/progress预测head。要做显式阶段进度，需先验证可靠标签，不能把Relation改个名字当已实现。
5. 因此“三阶段全监督”实际是监督分配消融，推理结构、参数、权重格式可以保持不变；不是新增三个全任务sidecar或让每个token承担全部任务。

## 方案比较

|方案|针对的问题|起点兼容与成本|主要不确定性|优先级|
|---|---|---|---|---|
|3阶段均监督Ground/Relation/Plan|每次Write之前的tokens是否足够有任务语义|可完整加载旧warmup2K；不加参数、不加推理forward|早层未来Plan可能难；EA可能过约束浅层；语义监督不自动带来外观不变性|本次独立Sidecar实验首选|
|每个Transformer block插Read/Refine/Write，阶段专门监督|交互频度是否不足|独立参数由3组stage增至主干每个block一组；共享参数虽省但读到的层分布不同；均有训练/推理额外开销|当前无频度瓶颈证据；新增层无预热；多次写入累积；兼容/导出/探针均要改|暂缓|
|每block插入且每处全任务监督|交互频度+监督密度|以上成本再加更密集loss，原代码明确约束3tap|一次改变太多，难判断原因；浅层过约束与辅助梯度叠加|最后考虑|
|成功版完整增强+当前XA|外观鲁棒性，修复当前XA增强混杂|无需改推理；需EMA teacher训练成本|当前XA最终闭环未验；完整Hue等可能改变语义|XA主线后续候选|
|Grounding-guided Read|让目标定位真正控制Relation/Plan读取，减少干扰物竞争|有现有bbox/heatmap，但Read attention并非当前head heatmap，需要新路由|定位错会强化错对象；缺失/遮挡、多角色/视图处理，偏置开关不能省|若错误集中于目标/干扰物再做|
|Per-channel gate|标量gate限制通道选择|旧scalar广播初始化可保持函数，参数少|已有倍率诊断仅反对盲目放大，不能证明或否定向量gate；没有当前通道瓶颈证据|第二梯队|
|2.5D/3D或显式Progress|几何精度/接触阶段表征不足|新head与标签/坐标/遮挡mask契约|有标签不等于可靠且可直接训练；无误差分类依据|先诊断，再决定|

“每层”在本表指每个Transformer block；若只是现有3stage都做全部预测，则等价于第一项。

## 为什么这次选三阶段完整监督

每次Write已经使用全部token，浅/中阶段却没有完整的直接监督，完整监督可以检验“浅层写给Action的信息语义是否不够成熟”。这是合理机制假设，不是目前已测出的瓶颈。浅层缺直接Plan loss也不证明浅层Plan无用。

该候选比24次插入更容易隔离变量、保留预热权重与推理成本。它不会直接补足外观一致性约束，不能预先承诺C2R改善。已知成功增强配方保持原样，可避免重演“改loss同时减弱增强”的混杂。

## 一个具体且可复核的训练提案（未实现）

- 初始化：同一原始pretrain-origin warmup2K（checkpoint_step_4000，acc2=2000更新，SHA49b6c013...8383835b）；不用新C2R warmup，不从最终21480继续训，不额外重做2K冻结。
- 保持：原成功10epoch完整PatchWAM、tap7/15/23、15角色tokens、3组stage、共享heads、gate、主干与Sidecar LR、warmup/cosine、全局batch256、相同数据及mask；8×batch32/acc1优先，和旧基线一致。无新teacher、无XA。
- 唯一改变：浅/中/深均计算Ground/Relation/Plan辅助loss并反传。保留任务内旧系数Ground .025、Relation .01、Plan .05，整体EA .5。
- 不直接把3层loss相加使EA隐性放大。建议浅/中/深按 .20/.30/.50 分配，各任务再乘原跨层总系数预算：Ground1.1、Relation1.1、Plan1.0。即

  `L_EA_new = .5 * sum_s a_s * [1.1*.025*G_s + 1.1*.01*R_s + .05*P_s]`

  其中a=(.2,.3,.5)。旧版为

  `L_EA_old = .5 * [.025*(G_deep+.1*G_shallow) + .01*(R_deep+.1*R_middle) + .05*P_deep]`。

  两者每个任务的系数总和相同，但实际loss大小和共享梯度仍可能不同，不能称梯度守恒。新分配明显提高浅中层监督、保留深层最大份额；这个权重是本次设计建议，不是report或论文验证的最优值。若用等权三层，浅层约束更强，也会进一步削弱最终层监督，本轮不优先。
- 不再叠旧deep监督，否则重复计数。旧默认loss路径完整保留，新配置单独启用。
- 10epoch21480更新，5000/10000/15000/20000/final保存；不把短训预算用于重建更短cosine。按用户偏好直接正式训练，不另加10步训练验收；实现层面仍须做CPU loss数学/梯度和默认兼容检查，生产监控OOM/有限loss。
- 先在本地增量实现并记录MODIFICATION_LOG，经用户确认后同步独立远端快照；原成功baseline与XA快照不覆盖。当前XA完成释放8卡后才能启动，闭环评测可在现有评测资源并行。

预期改动边界：architecture配置读取与compute_loss、一个独立训练配置/启动入口、已有test_relational_sidecar监督/梯度/默认兼容测试，以及修改/实验文档。modules中SidecarStage/EAHeads与推理调用不需要变；部署仅新代码快照/配置，旧模型权重完全兼容。若实际实现发现必须改推理层，应重新审查方案而不是顺手扩展。

## 评价和决策

- 每个stage分开记录Ground/Relation/Plan；报告归一化后的EA总量，不能把重新加权后数值下降当质量提升。
- 同输入/noise/timestep的Action FM、采样动作前8/全32段误差、按窗口P90/最差窗口、Full/WriteOff增益，以及Clean/Randomized分域闭环。辅助head分数不是控制收益。
- 重点检验早层Plan监督是否改善早期Write带来的动作收益，是否伤害Ground或Clean表现。若只改善EA指标而不改善Action/Randomized，则不给“语义更完整导致更好控制”的结论。
- 最终与旧21480同预算/相同评测协议比；历史单次训练有随机性，不是多seed因果结论。5K等中途与旧最终比较只作进展诊断，不作为严格优劣判决。
- 三阶段监督若有效，再单独研究6tap（而非直接24tap），并设计新增stage的初始化/短适配、参数量和计算量对照；不与全任务监督首轮捆绑。

本轮未实现、未启动实验，没有新的闭环结果。报告(5)的历史Ramp100/同2K full-state建议针对当时A/B/C背景，不能机械替代目前已成功的“原warmup2K权重+新optimizer+无重复冻结/ramp”配方。


### 实现核对补充：主干层数

已部署XA自包含config的WanModel为30个block；本提案仍仅选零起算7/15/23。前文“24层Sidecar”仅沿用讨论中的密集插入举例，不能当当前主干总层数；若真的每block插入，应为30处。本次不做密集插入。

### 2026-10-31 用户确认后的实现

用户已批准三阶段完整监督方案并要求串接当前XA之后。已本地增量实现、CPU52项通过、远端配置/原始warmupSHA复验，并部署独立等待队列；实际GPU训练尚未开始。配置、公式、路径和监控详见docs/SIDECAR_ALLSTAGE_WARMUP2K_10EPOCH_20261031.md。本文件上文“未实现/未启动”描述保留为提案阶段的历史状态。
