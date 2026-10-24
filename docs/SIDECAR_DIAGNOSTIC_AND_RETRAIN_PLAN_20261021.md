# Sidecar诊断与下一轮训练决策计划

> 2026-10-23新增：最早isolation warmup2K版本的24000微步/12000更新checkpoint已完成
> 同样本25GT窗口Full/Off及32batch梯度诊断。XYZ比旧42960差，夹爪略好、EA主导有所下降但仍存在，
> 尚无整体改善或新闭环SR结论。不是后来frozen-backbone版本，且训练预算/LR/aux不同。
> 详见[24000诊断对照](SIDECAR_ISOLATION2K_24000_DIAGNOSTICS_20261023.md)；原方案与历史结果保留。

时间：2026-10-21T13:36:25.479830+00:00。状态：源码、数据契约与现有实验结果已核对；本计划中的Strength Sweep、梯度分解、完整GT表征诊断尚未执行。本轮不修改生产模型或训练配置，不启动重训。后续先实现独立诊断，再依据证据选择训练改动，保留现有版本和已交付warmup包。

## 1. 当前能说什么、不能说什么

42960诊断已有证据：无cache三层实际mean r为2.69e-6–9.41e-6；18真实初始观测中，Full/WriteOff最终action相对差异mean0.2381%，当前cache下0.2883%。18对闭环Full13/18、Off14/18，17对相同，唯一分歧hammer/Randomized/seed100001偏向Off。18对初始原始输入哈希全部一致。说明直接write弱且确有动作影响，本小样本未显示净闭环收益，不证明写回有害或Sidecar整体无用。

需要区分三种价值：①推理时关系条件write的增益；②EA辅助监督对backbone训练的正则化/表征增益；③有GT支持的关系预测能力和诊断可解释性。当前runtime WriteOff只干预①，不能消除②；③预测准确也不自动证明机器人控制更好。

## 2. 已核实的实现约束

- 三次write在zero-based block7/15/23之后：actions + beta * projection(attention(tokens))。当前beta零初始化，projection正常初始化，不是双零死锁；不能只把旧checkpoint beta改成1。
- Sidecar只读首帧hidden。mutual跨模态mask明确禁止首帧看Action，first_frame_causal保护首帧。按此图结构，EA head预期不依赖任何Write参数；EA→Write梯度应是None/零，需要真实探针验证。Action→EA prediction heads同样预期不可达。两种结构性零都不是bug。
- Read/Refine生成的tokens同时送EA heads和Write，属于Action/EA共享路径；梯度方向冲突在这些共同可达参数上统计，不能给零向量强算cosine。
- 旧42960配置lambda_action=lambda_video=1。因此必须分解Action、Video、EA三类梯度，Video Loss也可能通过未来video对Action的注意力影响write。不能只比较Action和EA。
- 当前compute_loss返回的loss_action/loss_video/loss_ea日志已经detach，不能直接autograd.grad这些日志。诊断应在独立进程包裹原损失计算与heads.losses捕获活tensor，保留生产返回契约；重建加权EA含deep后校验与原total一致。复用原action masking/scheduler权重，避免手写另一个MSE冒充训练loss。
- 新2K隔离warmup在前2000 optimizer updates跳过全部write、detach EA读取，backbone仍训练原FM；Read/Refine/heads由EA训练，整个Write适配器没有这条Action训练路径。2000之后才第一次学习write。它保护早期backbone，但本身不保证write更强。acc=2时2000 updates是4000 microsteps。
- 新warmup配置还同时改了Sidecar LR为5e-5和aux总scale为0.5；旧配置与新版直接比较不能把变化全归因于warmup。需要一份相同LR/scale、isolation=0的控制。

代码依据：openwam/model/modules/relational_sidecar.py；openwam/model/architectures/dual_system/relational_sidecar.py；openwam/model/architectures/base.py；openwam/model/architectures/utils/mask_modes.py；openwam/train/utils/optimizer_groups.py。

## 3. 先固定GT与诊断样本，不用“动作变化”替代“动作变好”

当前manifest outputs/sidecar/aloha_4350/manifest.ready.json含50任务2500条，全是Clean。raw/aloha下未发现demo_randomized。已查看一条HDF5：有endpose、ea_rsig_oracle_v3，后者含双臂EEF/Object/Goal pose、track_valid/contact、goal_valid、bbox/visibility等；可复用targets_from_oracle和现有130D标签契约。

当前动作契约是absolute_eef_t_plus_1_to_t_plus_32，official_robotwin_endpose_reference、wxyz；proprio/EA TCP含局部x方向0.12m偏置转换。计划方向是each_role_current_local_rotation，horizon=[8,16,32]。动作GT复用当前reader的未来endpose序列，不把TCP与endpose混用，也不把控制器命令或相邻帧差分替换为既有标签。

上轮18个评估场景只保留初始RGB/proprio/prompt，没有专家完整轨迹或oracle监督。不能拿它们计算Action GT误差或Relation/Plan精度，不能把Clean加colorjitter称为真实Randomized GT。

执行顺序：

1. 首轮工具/梯度可行性用现有Clean三任务，各2条episode、每条2个有完整未来32步的anchor，共12窗口；明确这些episode参与过旧模型训练，只报告拟合与机制诊断。
2. 独立评估集沿用当前官方RoboTwin运行时、三个任务、两域，各5条专家可解episode，固定manifest。先复用旧18场景中每条件3条，再按固定seed扫描规则补到5条，不按模型成绩挑样本。
3. 每episode最多4个预先固定anchor，保留完整32步future，尽量不重叠；不足就减少并报告实际数量，不靠padding制造未来GT。总目标30episode、最多120窗口。所有scale/模型复用同一窗口表、prompt、noise和mask。
4. 在专家轨迹中同步保存RGB、原endpose/夹爪及oracle标签。首次只采一条Clean与一条Randomized，验收RGB编码、EEF/TCP、role匹配、contact单位、goal_valid、坐标系/时间对齐，再扩充。已有v6 oracle采集器在当前官方运行时的复用入口尚未核实；必要时仅适配三任务，缺失GT字段标记unsupported，不能填0冒充负例。
5. 新GT只存独立diagnostic目录，不加入训练集、不覆盖raw/labels或原manifest。以episode为划分/置信区间单位；同一episode多个窗口不视为独立episode。

## 4. P0-A：Write Strength Sweep

使用只读42960，一次载入，实例内对write执行s*beta，finally恢复；s=[0,0.5,1,2,5]，所有层统一缩放。s=0是仅关write，不跳过read/refine。先复用现有WriteProbe扩展倍率，保证s=1与旧Full一致、s=0与旧WriteOff一致、运行前后所有权重不变。

两组互补指标：

- 真正的10步生成：输入只有当前RGB/proprio/prompt，GT future只用来评分，不能喂入未来专家图像。固定相同初始noise seed、10步和sigma schedule。主实验cache off；选出候选倍率后再在当前cache+FFN配置复测，后者的cache路径可能分叉。
- 固定GT/noise的teacher-forced FM：用训练定义的noisy action/video输入、相同action/video时间步，计算原weighted action velocity loss和active维度误差；按Action噪声低/中/高分组，记录Video时间步，另在小子集检查Action/Video时间步组合。它衡量局部去噪，不等价于完整生成和闭环。

评分分开报告：双臂XYZ误差mm、SO(3)旋转角误差deg、gripper MAE与固定阈值的开合F1；按arm present与有效未来mask处理，不能把inactive arm或padding加进去。按预测horizon分别报告前8步和全部32步，防止后段误差掩盖部署实际执行段。保留每个窗口相对s=1的paired误差，按episode聚合/重采样区间，不能只报告平均相对action改变量。

每个倍率同步测真实r分布，而不是将旧r简单乘s：BF16舍入和下游状态变化可能非线性。5倍当前弱写回仍可能很小，若曲线平坦且r仍很低，只能说该范围无收益。确有必要再在离线小子集扩到10/20，报告实际r及有限性；没有GT误差改善前不直接把大倍率上机器人闭环。

只有在held-out GT上存在较一致的XYZ/旋转/夹爪改善，才选一个倍率进入少量固定manifest闭环；不能在同一批窗口选最佳倍率又把选择成绩当独立检验。至少留独立episode作确认。

## 5. P0-B：Action / Video / EA梯度与优化分析

先在上述12个Clean窗口用B=1完成显存探针，再在多个episode和任务扩展至32个固定microbatch、每批3个固定Action噪声层级。后续加入有真实GT的Randomized，不从单batch的梯度方向推论整个训练。

独立离线进程，关cache与compile以保留可审计图；对backbone参数不求梯度以省参数grad存储，但不能detach穿过backbone到Write的梯度。使用checkpointing、需要时saved-tensor CPU offload；梯度探针不是训练，绝不optimizer.step或写回checkpoint。与原forward和loss数值对照通过后才解释结果。两卡可用显存动态检查，不停止其他任务，不把“只反传Sidecar”预先说成必然能装下。

分组：每层beta、Write attention/query/projection、Read/projector/alpha、Refine/FFN、context/role tokens、heads；另在小子集检查backbone代表层，诊断EA对backbone的正则影响。每组统计三种loss的norm/RMS、None/非零比例、相对参数norm、加权贡献；共享非零路径统计cosine及负cosine比例，按任务/噪声层级列出分布。

beta单独记录dL_action/dbeta、dL_video/dbeta、dL_EA/dbeta；局部“放大现有有符号写回”的一阶指标是sum(beta_l*dL/dbeta_l)=dL/ds|s=1。正值表示局部增大s提高该loss，负值表示降低；beta可能为负，不能只看梯度正负或gate绝对值。以有限差分及Strength Sweep交叉验证，BF16附近有限差分不平滑时明确标注。

再核对LR、weight_decay、global clipping与optimizer moment：42960有模型权重，但当前目录完整accelerator state只看到36000/40000，不能用40000的Adam状态宣称恢复了42960真实更新。优先读取日志与相同步数状态；无对应state时只给SGD一阶方向或明确标成模拟Adam更新。对比FP32 master/optimizer状态与实际BF16参数变化，不能把lr*raw_grad当Adam真实更新。

必要时用36000/40000/42960相同窗口复测，区分后期LR已很小与始终学不会。不能由最终checkpoint的梯度反推出整个早期训练历史。

## 6. P1：关系表征质量与“是否真在用关系”

使用有标签的同一专家轨迹；固定有效mask、阈值、时间与horizon，报告每类正负支持数。类别失衡时不只报accuracy；与类别先验/常量预测、静止预测等简单基线对照。

|模块|现有输出对应的GT指标|边界|
|---|---|---|
|Grounding|有效GT框内attention mass、peak-in-box、presence F1、track-valid F1；按三视角/遮挡分组|当前无bbox回归头或role分类头，不能直接报BBox IoU/Role Accuracy；若以后加heatmap-to-box规则，先冻结解码并说明|
|Relation|masked Contact precision/recall/F1、Near balanced accuracy/F1、goal proximity MAE|contact_valid、track_valid、goal_valid分别处理；固定near=0.10m与原标签一致|
|Plan|各horizon、各arm、object/goal的moving F1；仅GT moving且future-valid上方向cosine/角度|plan是role自身局部坐标，不直接与world EEF delta比较；静止片段不计方向分母|
|Summary|暂不设progress accuracy|当前没有可支撑该指标的GT|

每个tap都可看已有head输出，但中间head仅接受特定deep supervision，不能把中间表现差直接当最终表征失败。Clean和Randomized专家轨迹可能长短/阶段分布不同，按任务/有效角色/阶段分层，避免把接触率差异误当域退化。另做相同几何的外观干预才可讨论外观鲁棒性，不能假定不同域专家天然动作逐帧配对。

若预测准确却动作收益弱，再做write输入的token干预：保持read/refine正常，只在write处替换plan/relation token为匹配任务、阶段和norm的其他样本token，与原token对照GT动作误差。先做一组最小干预，不先铺满组合；跨样本置换也可能OOD，必须与尺度/分布匹配控制一起解释。作用是检验Action是否依赖具体关系内容，而不是仅多了一条adapter通路。

## 7. P1：2K Warmup比较

使用已经实现的隔离warmup，不另堆同类机制。优先读取另一服务器实际产出的配置/指标/checkpoint；本机未核实其启动与完成状态，不能默认已有结果。

必须比较相同LR=5e-5、aux_scale=0.5下isolation=0 vs2000，而旧42960作为历史参照。固定Foundation初始化、数据、更新数/有效样本数、增强、scheduler、batch/accum、loss定义和eval协议。重点观察隔离结束前、首次接入、接入后早期和稳定期；不是只看最终SR。

记录r、Action/EA/Video梯度、GT动作误差、关系质量和配对SR。关注write在2000之前从未训练而2000之后突接入的风险；warmup本身不等于Conditioning已准备好。若只能拿到不同LR/scale/训练预算的两组，明确是整套recipe对比，不声称warmup单因素收益。

## 8. 诊断以后才选择训练修改

|诊断结果|优先最小修改|不应立即做|
|---|---|---|
|GT关系可靠，s增大改善GT动作并获独立确认|检查gate/Write优化与数值；考虑Write独立参数组、平滑接入，保留预训练先验|直接把全部EA LR、aux权重和gate同时放大|
|GT关系可靠，但放大无益或Action倾向减小写回|先检验token内容依赖、每层贡献和信息冗余，再决定阶段/角色相关融合|强迫r达到固定比例或立即叠Action–Plan loss|
|Randomized关系失准|先修标签/遮挡/goal_valid和关系表征OOD鲁棒性，必要时外观一致性监督|把不可靠条件更强地写入Action|
|Write稳定损害GT动作/闭环且可重复|保留主干先验，降低/渐进write，定位层和梯度冲突|由一条分歧宣称写回总体有害|
|Action梯度有，但Video梯度持续压制write|先验证三loss贡献及位置，单独评估写回的梯度路由方案|直接改mutual mask或停止Video loss造成大范围不兼容|

若支持重做写回初始化，候选beta=1/W=0与当前beta=0/W正常是不同训练实验；前者可以使W在首次接入获得Action梯度，但其他Write上游参数仍要等W非零，且同样受最终BF16舍入约束。不能把这个初始化直接套到已训练42960上。

更合理的目标是“可学习、任务有益且幅度受控”，不是“更大的r”。若做ramp，要有独立配置、同更新预算控制、接入瞬间数值/梯度与resume测试，不预设某个固定r百分比必然最优。单纯把加法临时算FP32后又cast回BF16也不保证小残差被保留。

## 9. P2：区分Sidecar价值的训练对照

预算足够后按相同数据/初始化/更新与评估比较：

- V：Matched Vanilla，无Sidecar。
- E：EA辅助训练、write恒关，检验辅助正则化价值。
- W：有Sidecar write、EA loss关，检验额外adapter/条件容量。
- EW：完整Sidecar。

E−V对应辅助监督路径的总训练效应，EW−W帮助判断EA语义监督的额外价值；EW−E比较完整训练recipe中的write相关效应。各组存在共同适应，不能把数值简单当彼此独立可加的贡献。单checkpoint runtime off与这些训练对照互补。已有baseline实验只有在data/stats/预算/增强等匹配核实后才能称Matched；不能仅凭step数字相似或官方成绩判定纯架构收益。

## 10. 执行顺序、产物与验收

1. 独立诊断入口复用WriteProbe/现有reader/heads.losses；先12个Clean窗口跑sweep和4窗口梯度显存探针。产物window_manifest、scale_action_errors、per_tap_r、gradient_reachability与原输出一致性检查。
2. 验收一条Clean/Randomized专家GT采集，再扩到30episode/最多120窗口；运行完整sweep与表征指标，生成按episode配对表，不先扩大闭环。
3. 若有候选倍率，在留出episode确认，再少量闭环；唯一原分歧场景优先Full/Off重复并互换仿真GPU，以检查仿真数值稳定性。没有稳定收益就保留“未发现收益”结论。
4. 归纳A/B/C/D与三loss梯度证据，给出单一训练改动的具体diff与理由后实现；保留新旧可切换入口。真实多卡跨阶段、保存/恢复与短训验收通过，才安排正式训练。

计划新增代码限benchmarks/robotwin诊断与tests；损失捕获优先实例局部包装，不改核心compute_loss日志契约。若确需共享loss helper，再说明范围并最小提取，避免复制训练loss。无理由修改根README、现有训练包、旧ckpt或正式队列。每个新非平凡分支留一个小测试：倍率/参数恢复、活梯度与结构性零、单位/有效mask、结果完整性。每阶段更新本文件及docs/思考与隐患.md，代码重大变更记MODIFICATION_LOG.md。

资源沿用用户授权GPU4/7但每阶段重新检查容量；尽量一次载入、缓存固定prompt/编码后的专家样本，主分析eager/no-cache，部署复测再启用已验证FFN/cache。先实测一个窗口的载入、forward/backward与显存峰值再估工期；不把无optimizer更新等同于零显存成本，不占用或停止未知作业。

目前未完成：新sweep/梯度/GT关系诊断与新训练实验均未运行；以上是可执行计划与源码核查结论，不是新实验结果。不存在“保证提升成功率”的承诺。

## 11. 2026-10-22：诊断完成后的结构审计与下一版选择

前文保留2026-10-21诊断开始前的计划状态。Sweep、六episode新场景GT、18对Full/WriteOff和32 batch梯度诊断现已完成，结果分别以SIDECAR42960_GT_DIAGNOSTICS_20261021.md、SIDECAR42960_WRITE_DIAGNOSTIC_20261021.md及SIDECAR42960_GRADIENT32_20261021.md为准。本节重审已有证据和源码，不是重新运行这些实验。

### 结构与容量

只读取固定42960 safetensors的274288字节header、按shape计数，Sidecar为24,290,323参数。三阶段Read/Refine合计14,999,043，Write合计6,308,355，文本/本体/role token初始化2,450,432，辅助heads532,493（2.19%）。三阶段参数独立，各7,102,466；heads跨阶段共享。

每阶段已有Read cross-attention、Refine self-attention+FFN和Write cross-attention；hidden512、8heads，FFN512→1024→512。Video hidden3072投影到512，Action hidden1024通过512维注意力再投影回1024。三个阶段位于block7/15/23之后，已是一套Transformer式条件分支，而不是只有MLP辅助头。15tokens为Ground4、Relation4、Summary1、Plan6；Plan按三horizon×双臂组织，每token预测EEF/Object方向与moving。写回使用latent tokens，不是直接把head预测概率作为动作条件。

tokens只在一次forward内跨tap持续，没有跨控制时刻的持久记忆；Read只看当前首帧，Write的query来自Action hidden，所以写回已有动作/去噪状态依赖。Relation监督没有完整相对XYZ回归，Plan没有运动幅度监督。高F1/cosine不证明表征包含全部精细控制信息，也不证明超越预训练主干的信息。暂无容量不足的直接证据，不因SR欠佳先增加Transformer深度。

### 最优先的训练改动

用户选择先只实施“前2000 optimizer updates完整前向/反向、Sidecar正常读写，WAM仅不参数更新；随后joint”。沿用Sidecar LR5e-5、WAM base LR1e-4和原scheduler、Action/Video系数各1、aux_loss_scale=.5、原三类EA内部权重、ColorJitter和Aloha50/2500数据。保留beta0/正常投影的单零初始化；不添加WAM-only第三阶段、不改变网络、不同时加入heads-only路由或Task Gate。

与旧isolation前2K绕过write相比，本方案让写回接口从warmup阶段就获得FM监督，同时固定其所适配的WAM权重。beta初始0的第一步FM只先训练gate，之后写回投影/上游才获得这条梯度；这不意味着断图。2K是工程选择，尚未证明最优。

实际旧helper仍清零WAM叶子梯度，虽然不阻断Sidecar激活梯度，但不满足用户“保留完整参数反向”的澄清。本次仅去掉该清零hook，保留AdamW step-pre-hook：WAM正常反向/归约/clip后跳过参数、动量、计数与衰减更新。永久冻结的VAE/text模块仍按原合同处理。acc2下2K对应4000microsteps；10epoch预算内WAM少2000次更新，必须披露。

### 关于权重与后续决策

已有32 batch显示旧42960共享表征的加权EA梯度比Action大多个数量级；统一降低Sidecar LR不改变同一共享参数所接收的相对梯度来源，aux从1降至.5也只减半。这个结果不能直接当作Adam实际更新倍数：还取决于历史moment、逐坐标预条件和clip，原诊断没有恢复optimizer。

先验证这版正常写回warmup，记录新2K状态的实际写入r、clip及参数更新，而不是把旧42960结论直接套到新状态。若仍有EA主导且动作无收益，唯一优先候选仍是Stage2 EA仅更新heads、共享tokens和Write继续由Action+Video训练；从同一份新2K完整状态分两路，各续训1000更新。两路保留同一数值loss系数与LR，以隔离梯度路由因素。该候选本次不实现；它可能造成关系遗忘，不能称为已确定更优的最终recipe。

不再做Gate倍率扫描，不强迫r达到固定百分比。确认方向是XYZ/旋转/夹爪GT误差与三任务双域每条件10场景的配对闭环（60场景/模型），r只是机制指标。随后需要matched Vanilla检验增量控制价值。若路由无收益，再依据新证据选择一项写回初始化或几何监督改动；不一次叠加扩网络、调Gate和新loss。

当前第一步实现/工程验收记录在ROBOTWIN_SIDECAR_FROZEN_BACKBONE_WARMUP_20261022.md；尚无本新版本训练效果，当前训练切换仍保持暂停。

## 12. 2026-10-23：当前实现复核与20天论文计划

上节末尾为历史状态。新full-backward warmup已正式运行，02:41 UTC为micro6926/optimizer3463，已过2K；有4000微步/2000更新完整状态。gate较旧joint明显增大，但尚未测新模型实际write-r/闭环收益。WAM首次恢复更新后曾出现短暂Action loss与梯度尖峰，随后恢复，提出仅对WAM增加短LR ramp的独立候选；不与Stage2 heads-only梯度路由混作同一改动。

完整实现边界、已完成ManiSkill587/600结果、监督粒度/near/控制回放问题、匹配对照矩阵及单臂真机20天安排，见[Sidecar当前实现与20天计划](SIDECAR_IMPLEMENTATION_REVIEW_AND_20DAY_PLAN_20261023.md)。用户确认已有可控单臂真机与相机/控制接口，要求本轮只分析和文档、不修改代码或启动新实验。训练日志只读证据见`docs/evidence/sidecar_review_20261023/training_log_snapshot.json`。
