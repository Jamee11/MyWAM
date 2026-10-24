# Sidecar32k Randomized低分：现象、假设与验证顺序

时间：2026-10-19T09:15:35.825096+00:00。本次是只读分析与实验建议，未启动新的GPU消融、未更改训练或当前完整评测。

## 已有证据

完整Randomized四项：adjust_bottle37 vs官方51（-14pp）、beat_block_hammer33 vs55（-22pp）、click_alarmclock65 vs77（-12pp）、grab_roller49 vs38（+11pp）。四项同任务Clean：Sidecar98.25%、官方97.00%；Randomized：46.00% vs55.25%。这个预先由进度形成的小子集中，Clean到Randomized跌幅分别52.25pp与41.75pp，多下降10.50pp；不能外推全50任务或直接解释为Sidecar模块的因果作用。

官方Clean2Random本身也仅用Clean demos微调，故“我们只有Clean训练”不能单独解释相对官方的差距。当前32k训练config为clean_50、50任务2500轨迹，joint训练约6B参数，video/action LR均1e-4，ColorJitter brightness/contrast/saturation .2、hue0。已有训练loss/CPU验收不等价于随机场景稳健性。

## 对实际Randomized扰动的核查

本机官方demo_clean.yml与demo_randomized.yml差异为：随机背景开启、cluttered_table开启、clean_background_rate1→.02、随机光照开启、crazy_random_light_rate0→.02、random_table_height0→.03。两者random_head_camera_dis均0；不能把此次随机域直接描述为随机移动相机。_base_task.py:111使用uniform(-random_table_height,0)，即桌面基础高度偏移-3cm到0，而不是±3cm。ColorJitter不会自动覆盖新增杂物、遮挡、背景内容和桌面几何偏移。

优先假设：当前模型的目标绑定/视觉grounding或接触高度对杂物、背景与桌高变化敏感，可能学习Clean场景中的固定图像/绝对高度捷径；这些都是待验证假设，没有逐失败轨迹或单因素干预证据。需要将失败分为首次接近即偏移/抓错物体、抓取未闭合或高度错误、已抓取后放置/接触失败、动作抖动或长时间不推进。不同类别不应统一归因于视觉。

Sidecar的ground/relation/plan辅助head不直接作为动作输入，作用通过15个latent token和三处write残差实现。当前三beta约1.43e-4、3.38e-4、4.91e-5，但真实贡献还乘attention/projection输出并依赖主干hidden尺度；不能凭beta小断定无效。建议记录每层写入残差范数/动作hidden范数、full/off动作差异及grounding误差在Clean/Randomized间的变化。当前帧读取没有显式历史状态，多层读取不能直接当作时间记忆。

## 验证顺序

1. 排除基础设施无效样本。click_bell/Randomized已有OIDN/buffer崩溃，原8条排除并计划指定完整100重跑。全量搜索当前其余日志未发现相同错误；不能把所有随机域低分都推给此次异常，也不能把损坏渲染输入当模型能力下降。
2. 先做同32k、同封存expert manifest/指令/seed的full cache-on与full cache-off。原加速验收只验证相同cache配置下的FFN编译保真，没有证明cache相对完整计算不影响闭环；此前两fixture最大动作差0.03125/.01171875，且10步只重算4次。若cache-off明显恢复，则优先调部署策略，不急于改训练/结构。原版cache-on并不保证新架构相同近似误差。
3. 复用已有full/off/first_write/last_write方案，主比较统一cache-off；off只是同一joint训练权重关闭在线分支，不能冒充独立训练的原版。先看full/off是否改善或退化，再解释局部write位置。已有方案只CPU验收、GPU消融尚未启动，见SIDECAR32K_INFERENCE_ABLATION_20261019.md。作为额外建议，可用grab_roller作正向对照，避免只选择退步任务；不自行覆盖原用户选定三任务。
4. 如需区分部署/模块与主干训练，增加原版发布权重在相同新环境、同manifest/指令/seed与匹配cache设置下的对照；不能只引用官方历史百条当配对控制。若full优于off但两者均弱于原版，优先排查joint微调/数据/主干变化，而非直接责怪Sidecar在线写入。
5. 固定同一候选episode做背景/光照、杂物、桌高单因素诊断。物理扰动或移除杂物可能改变专家可完成性/接触动力学，须每个条件重新验证专家可完成性并报告接受交集，不能静默保留新条件失败seed；不改变正式benchmark协议。仅凭官方全量任务表无法定位哪种扰动导致失败。

## 证据支持后才考虑的训练改动

若主因是视觉定位/高度捷径，优先增加与当前Clean2Random设定一致的训练侧干扰鲁棒性：保几何标签一致的背景/光照增强、注意监督是否能区分目标与杂物、桌高改变时同步重算动作/关系标签。随机crop/resize/相机几何变化不能保持旧bbox/空间监督不变。若直接加入完整Randomized专家数据，实验训练设定变为Full/域适配，不能再声称Clean-only可比成绩；应另列实验。

若主因是joint遗忘或辅助优化竞争，再考虑较小主干LR、模块与主干分组学习率、分阶段训练或调整aux权重，必须用匹配更新数/数据的独立训练对照验证。若full与off都一样或写入残差确实极弱，才考虑提升在线写入的有效训练信号；不根据小beta直接放大gate。以上训练/架构/数据改动均仅建议，执行前遵守AGENTS.md单独确认及日志记录要求。

当前最有信息价值的下一步是配对cache消融与Sidecar开关消融，而不是仅凭Clean高分继续增加训练步数或先扩大模型。全量32k评测继续，不从进行中较好的任务挑分作为最终结论。
