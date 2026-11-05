# 下一轮主实验决策：Grounding引导关系读取

日期：2026-11-03。范围：现有代码/报告/逐episode结果分析与训练提案；本轮不改core、不写生效训练配置、不启动训练。用户目标：保留关系表征→Action主线与模块，单次10epoch预算，尽快取得有效反馈，针对模块推理收益不稳定的问题。

## 主决策

下一轮优先 **原始warmup2K + 原成功完整PatchWAM + Grounding-guided Relational Read**。保留三stage/15tokens/Refine/FFN/原全部辅助监督与Write、原WAM视觉动作连接；只把浅层/中层预测的定位热图接到后续Relation/Plan视觉读取中。暂不叠XA、三阶段完整监督、额外3D回归、gate放大或Action→Video屏蔽。

训练期关闭Write（EA-only）是有价值的归因对照，但当前唯一主训练名额优先给这条保留在线关系条件、具有具体计算路径的新分支。随机屏蔽Action对Video读取只是间接的通道竞争假设，尚无诊断证实，撤回其作为最高优先主实验的建议。

## 已核对的事实与不能混淆之处

1. 19组是10任务×双域×10条，每组200条，共3800；不是3800条独立样本支撑每个模块效应。原始完整结果见ROBOTWIN_DIAGNOSTICS_AND_MANISKILL_FINAL_20261103.md。
2. 按(task,domain,seed,instruction)逐episode对齐，Write-Off相对Full有10次由失败转成功、4次相反，净+6；Relation-Reset为13/4，净+9，其中place_object_basket两域净+5；Refine-Off13/5净+8；All-Reset12/9净+3；Sidecar-Off8/7净+1。Write-Off对Sidecar-Off有15条结果不一致，净差5，不能只看总数差。
3. 作为描述补充，未校正的配对精确McNemar p：全域Write-Off .1796、Relation-Reset .0490、Refine-Off .0963、All-Reset .6636、Sidecar-Off1。Random Relation-Reset .0215、Write-Off .0391，但18个候选/多域比较、10任务选择及任务聚类未处理，不能当确认性显著性声明，更不能凭其中一个最优组决定删token。复算JSON：outputs/analysis/sidecar19arms_paired_reanalysis_20261103.json。
4. BF16 max_abs=.0078125不能证明只有数值噪声。该间距随数值幅度变化，同一个最大值不意味着相同变化元素或物理动作差。Summary/Position-Off都改变真实计算路径，也不能当“无效操作噪声底线”。需要相同输入/噪声/cache的Full重复作为重复性参照；改变noise seed测的是采样敏感性，两者分开。
5. 先前真实Write测量显示约68.8%–76.5%的Action hidden元素确实改变（docs/SIDECAR_APPEARANCE21480_DIAGNOSTIC_20261030.md），不支持“完全被BF16吞掉”。小beta、不同符号不是未使用/有害的充分证据；×2/5/10/100放大未显示总体收益，不再据beta做放大训练。
6. tokens在read_refine之后、Write之前reset，head输出在reset前已算好；其语义可在其他tokens及主干分布式保留。All-Reset经LayerNorm/MHA偏置可留固定Write残差，不等于Write-Off。关闭预测head不会擦去训练期监督贡献。
7. 当前最好10epoch从原Foundation-origin warmup2K权重加载，而非从Foundation重新随机开始。新的matched训练应复用该原warmup2K，避免把更换起点/额外warmup混进主对照。若研究纯no-Sidecar总因果效应，才另做从Foundation开始的严格匹配训练及计算预算说明。
8. ManiSkill place_sphere低分尚不能归因Sidecar：缺同manifest baseline与失败阶段标签。它不能作为修改RoboTwin关系通道的独立证据。

## 针对的是哪一个实际缺口

代码证据：

- openwam/model/modules/relational_sidecar.py：SidecarStage中15queries共同自由读取全部当前帧视觉features，之后Refine/FFN；EAHeads再从Ground tokens与features预测heatmap。原heatmap是读出，未作为Relation/Plan的视觉注意力先验。
- architecture的3tap依次传同一组tokens，因此Ground latent可以通过self-attention间接影响其他tokens；不能称“Grounding与其他模块完全无连接”。但预测空间分布没有直接控制后续视觉读取，这是一处可明确补足的前向关系。
- Relation辅助监督目前近/接触/goal proximity，plan为未来moving/局部方向，数据含EEF局部相对XYZ但目前未直接回归。连续几何监督也是合理候选，本轮不与路由同加，避免使结果难解释。

假设：自由全局读取可能让Relation/Plan保留任务无关外观或干扰物信息，或者让Grounding仅可被head解码、没有充分转化为Action需要的关系条件。这能解释部分现象，但当前尚未证明它就是负收益根因。新分支通过目标相关视觉路由检验和修补这个缺口，不以“保证模块消融变差”作为目标。

## 最小可实现的路由设计

采用**跨stage定位先验**，不增加第二次主干forward：

- tap7保持原计算，得到同一当前帧的预测Grounding heatmaps。
- tap15保留原Ground/Summary读取；Relation和Plan query读取视觉features时，加入tap7预测的目标/参考物空间先验，随后原Refine/FFN/EA/Write不变。
- tap23同样使用tap15的预测热图。热图在同一forward内传递，不跨chunk缓存，不使用GT热图做推理输入。
- 按现有track/role定义将object/goal热图与对应手臂的Relation/Plan queries配对；不擅自改成EEF/任意全物体槽。缺失手臂使用原proprio mask，目标/参考物有效性使用推理可得的预测与既有语义，不向推理泄漏GT labels。
- 对得到的归一化任务先验A混入50%均匀分布：P=.5A+.5/N。注意力logits加`lambda * log(N*P)`，不做hard crop/mask，仍允许读取全部patch。Ground与Summary行不加空间bias，文本/proprio/原WAM masks保持。
- lambda使用确定的训练schedule：前1000参数更新从0线性增加到.5，后续及eval固定.5。这是本次工程候选强度，不是论文已验证最优值。lambda0必须走原路径并检查输出/梯度兼容，避免将原Read拆成不同算子造成不必要数值漂移。
- 路由prior保留可微路径，原EA监督不减弱。因定位head现在参与后续读出，Action与后续关系损失可以训练其路由用途，但实际梯度效果需测量，不预称所有模块的动作贡献已得到保证。

同一个视频latent网格在三个tap之间空间位置不变，可复用热图；实现前检查三视角canvas、role映射与有效mask即可，不新增数据标注/深度模型。实现仍需注意torch.nn.MultiheadAttention批次×head mask形状、BF16/FP32 mask dtype，以及checkpoint重算保持同一prior张量；这些是必要正确性检查，不是额外实验方向。

此方案与report(2)/(4) Grounding-guided Read思想一致，但本次选择跨stage先验以完整复用现有训练热图、避免新增完整读取pass。它不是已有paper直接验证过的OpenWAM配置。

## 为什么它比其他候选更适合这次预算

- 保留原强主干通路，不需要假定Action过度依赖Video，也不会先砍掉预训练有效信息。
- 保留Ground/Relation/Summary/Plan和全部Refine/Write；让Grounding具有可检验的路由职责，而不只是希望潜在语义自己被Write学到。
- 改变写回信息的构造，而非强迫更大残差；针对当前“局部去除有时更好、整体取消没多少改变”的弱而不稳收益，减少无关读取是比直接放大更贴近目标的干预。
- 不加teacher/第二主干forward/新标签，不增加30处Sidecar。训练成本预计接近原版；实际mask实现仍需测吞吐，不承诺开销严格为零。
- 三阶段全监督是监督密度候选，已经训练完成，应并行评估；本次以原R10监督为参照，不把两个训练改变捆绑。

## 10epoch训练配方（提案，未执行）

初始化同一原始warmup2K checkpoint_step_4000（acc2=2000更新，SHA49b6c013...8383835b），不用C2R新warmup，不从原21480再训10epoch。完整原PatchWAM不变，fresh optimizer/scheduler/counters，无重复2K冻结/额外WAM ramp；8×32acc1/global256，主干/Action1e-4、Sidecar5e-5，标准5%LR warmup+cosine；10epoch21480更新，每5000保存与最终保存；EA100%、旧3stage监督分配、gate1、XA关闭。

唯一新训练因素：上述Grounding-guided Read。默认配置关闭，独立配置/输出，先本地实现/回归再同步新快照，不覆盖任何旧baseline或实验。需要用户确认后才改core。仍尽量仅占一组8卡；H100当前内网不可达是执行阻塞，不应承诺已排入可运行的H100队列。

时间：先前同机器原配方约1天量级，四卡A100可能3–4天且受争用影响；不做固定时长保证。本次不做多组长训或10步短训验收，正常实现测试后直接正式预算；5000权重自然保存时开始已有协议的小规模评估，无需等10epoch结束才第一次看结果。

## 尽快反馈而不让诊断耗尽时间

训练并行只做一个聚焦评估包：若干未用于选择新方案的held-out任务，固定scene/语言/采样noise与同cache设置。只比较原Full、新Full、新Guidance-Off、新Write-Off；第一轮先看新Full是否保Clean且改善Randomized，而不铺开19组。

现有评测疑点以少量固定输入/短闭环补齐：同Full同noise重复区分数值/仿真重复性；cache-off Full与Write-Off/Sidecar-Off同输入对照，记录RMS/物理XYZ和cache skip；换noise seed单独报告采样敏感性，不能用来抹去所有真实模块效应。该小包可与训练并行，不要求完成全部旧因果对照才开始新训练。

判断有效的主标准：正常全通路、相同评测预算下Randomized更好且Clean不出现一致退化；Guidance-Off能解释至少部分新收益。仅mask开启/被迫失去主通路时更好、gate变大、head分数更好均不足以称方法成功。初期5K只筛查明显退化，不能与旧最终21480进行等预算因果排名。

科研主线：仿真关系监督定位任务对象，定位引导关系与短期转移表征，再通过Sidecar条件动作。保持RGB-only/proprio/text推理，不增加simulator标签输入。无需声称每个token在每个任务都不可替代，目标是证明结构化关系路径带来可重复的实际收益。

参考：report(2)/(4)提供直接项目提案；OA-WAM原文https://arxiv.org/abs/2605.06481提供对象可寻址表征这一相关方向，PrismWAM原文https://arxiv.org/abs/2610.26564提供物理表征与动作路由分工的相关方向。它们不是本方案的实测收益或优先级证明。少量run不确定性的讨论见https://arxiv.org/abs/2108.13264，本轮实际结论以本地审计数据与实现为主。
