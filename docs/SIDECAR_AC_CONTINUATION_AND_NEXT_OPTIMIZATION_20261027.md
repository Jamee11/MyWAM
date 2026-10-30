# A/C续训安排、第二份研究报告复核与下一步性能优化

记录日期：2026-10-27。按用户明确要求执行E结束后A、C各续1K更新，并分析 `deep-research-report (1).md`。本轮不叠加batch、gate、EA权重或weight decay改动。

## 1. 已执行的续训安排

E于 **08:43:14 UTC正常结束**，完成6000微步/3000 optimizer updates，最终权重与完整状态已保存；其离线诊断继续运行。A在GPU0–3接续，四卡真实恢复和首次更新已经通过。

|顺序|来源|本轮终点|新增训练量|
|---|---|---|---:|
|E|共同2K warmup完整状态|3K更新，微步6000|已完成1K|
|A续训|本轮A自己的3K完整状态|4K更新，微步8000|1K|
|C续训|本轮C自己的3K完整状态|4K更新，微步8000|1K|

这轮先执行用户正文的各+1K，**不是报告中的C直接3K→5K**。C已完成300-update ramp，恢复时进度为3000，因此其实际LR倍率保持1，不重复warmup或ramp。A/C保留各自不同的Adam moments、RNG及原21480-update/10epoch cosine。

独立输出根目录：

```text
/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_AC_extend1k_20261027T082947Z
```

保存方式：各组仅在8000微步保存一份最终权重与完整 `accel_state_step_8000`；3K源仍保留。3K权重用硬链接固定，配置/统计/tokenizer独立复制，原完整state只读引用并记录文件身份清单，避免再复制两份约110GB状态。08:50复查源文件inode、mtime、size均未改变。

监督 `scripts/supervise_sidecar_ac_continue.py`：A已CPU预加载至E自然退出后接卡，A到7400微步时预加载C，A满8000保存且exit0后释放C。当前监督PID4123250，PID使用前必须核对启动标识和命令。数值batch逐微步、图像按既定6步与续训A核对；不错误要求A/C的初始optimizer相同。

原短训入口仅增加可选来源/起止参数，默认2K→3K行为保持；原E在内存中按旧入口正常结束。12项合同/原入口/LR恢复检查及6项端点诊断检查通过（含8K endpoint）。A实际恢复验证：6000/3000、epoch1、skip1704，scheduler last_epoch12000、T_max81624，首更新3001有限。C真实恢复及两组8K终点保存继续由队列验收。

离线诊断复用原三任务、双域、25窗口×Full/Off及32固定FM样本，4K与各自3K同输入比较；`diagnostics/comparison.json`保留本次4K和`reference_3k`。CPU Loss分析拼接2K→3K旧日志及3K→4K续训日志，输出连续曲线和后段差距，等C结束自动生成 `analysis/`。仅Loss/离线GT不足以宣称SR提升，后续仍需配对闭环。

按约13.2秒/更新，两组纯计算约7.3小时，恢复/保存另计，诊断与另一组训练重叠。若C在4K后值得继续，尚余17480次更新到原10epoch终点，当前速度下约64小时计算量；实际受共享负载和保存影响。

## 2. 参考A的历史checkpoint到底在不在

旧完整full-backward训练的4K更新权重（文件名为微步8000）仍有历史诊断快照：

```text
outputs/robotwin_eval/sidecar_full_backward2k_step8000_diagnostics_20261023/checkpoint_step_8000/
```

但它来自10月22日完整run。本轮A/C共同起点是10月26日冻结的S，本轮A分支截至续训前只到3K，尚无本轮4K/5K权重。历史同配方模型不等于本轮相同输入和恢复路径的严格控制。因此这次续A到4K是有价值的，不用历史4K模型冒充本轮Control；也不重跑整条A的10epoch。

## 3. 对第二份报告的判断

主线合理：优先延长已解决过渡冲击的C，保留关系监督，减少没有证据的新trick。但需收紧几处结论：

- C已证明改善开始更新时的稳定性，尚未证明闭环SR提高。LR积分少13.63%支持适配滞后的可能性，不保证延长后追平。
- B只说明本次1%/500-update退火使共享语义学习减速，不是所有辅助梯度路由方法均无效；D/E也只覆盖本次倍率/起点/预算，不能排除所有融合改进。
- 报告的0.6%–0.9%写入及约5%–6% FM收益是旧12k微步历史诊断点；本轮同预算比较用自己的3K/4K数据，不与历史点混成单条曲线。
- 已有71.10%对官方69.04%是+2.06个百分点，但每条件30对官方100，尚不是严格Matched Vanilla的因果提升。论文仍需匹配训练/数据/预算的Vanilla与No-EA Adapter。
- 若从已受EA监督2K的S关闭EA继续训，那是晚期撤监督，不是完整的No-EA Adapter控制。真正No-EA应从匹配预训练起点明确其整个训练协议。

## 4. Gate weight decay审计：确实存在，但直接作用很小

已读取A实际ZeRO checkpoint的AdamW metadata并核对 `optimizer_groups.py`：alpha、beta属于Sidecar组（索引2，`ea_backbone_group=False`），继承 **weight_decay=0.01**，未单独排除。

按真实1000次LR记录计算仅decay的乘积：

|参数|A的纯衰减|D的纯衰减|
|---|---:|---:|
|Read alpha|0.04938%|0.04938%|
|Write beta|0.04938%|0.09873%|

D因LR翻倍额外直接收缩约0.049个百分点，远小于实测beta数十个百分点的变化。**不能把D的主要gate收缩简单归因weight decay。** 这不是说weight decay绝无间接影响，而是本次量级不足以支持把去decay列为首选。证据 `engineering/gate_weight_decay_audit.json`；A/C维持0.01，不偷偷修改。

## 5. Per-channel Gate：能做，但主要是优化重参数化

1024维hidden写入的标量beta复制成1024维向量，初始前向可以保持相同。三个scalar替换成向量后净增3069个参数，参数量很小，但有两个关键点：

1. Write projection本身已经可训练，`beta_vec * (Wz)`可写成 `diag(beta_vec) Wz`。当前W的每一行原本就可分别学习，所以静态向量gate不直接扩大这一线性写回函数的表达集合，主要改变优化路径/正则化与参数分解。不能把它说成之前各hidden channel完全没有独立调节能力；1024 hidden维也没有固定对应XYZ/rotation/gripper的语义。
2. 初始前向保持不代表完整optimizer恢复保持。scalar→vector改变Adam moments形状与ZeRO flat partition偏移，不能直接原样load旧状态，也不能只重复beta数值就声称全过程控制变量相同。需要迁移和真实保存恢复检查。

因此F是可选优化候选，不是无风险的几行结构升级。本轮不实现或排队F，不与C叠加；保留初始化方案，先验证C延长后的实际收益。

## 6. 还能怎样提高性能：建议的优先级

没有一个调整能保证再涨几个点。当前最有价值的是区分优化滞后、后期泛化变化和部署反馈频率，而不是一次修改多个参数。

|优先级|候选|依据与成本|主要限制|
|---|---|---|---|
|P0|完成A/C同预算续训与配对闭环|已有稳定化证据，成本约两组1K|按Loss、实际Write、GT和闭环综合判断|
|P0|固定开发seed选择checkpoint|历史32k总体71.10%、42960为70.00%，Random53.20→50.07|这不直接证明过拟合；最终需独立seed/匹配baseline验证|
|P1|邻近checkpoint插值/平均作一项离线候选|同一轨迹的保留权重可用，无新增训练|不等同历史EMA；非线性模型平均不保证改善，需同时看旋转/夹爪|
|P1，条件触发|后期更保守的预训练Video/WAM LR|若重复确认后期Random回落，可检验是否更好保留先验|当前C尚在追赶，不立刻进一步压LR；更低LR也可能欠拟合|
|P1，需独立数据协议|更贴近Random域的外观增强|目标直接对应泛化短板，保持EA几何监督|ColorJitter已开；不能把不变性假设或标签破坏当增强|
|P2|global batch256→512|可能减小随机梯度噪声，可用acc4而非扩大单卡batch|更新数、warmup、scheduler、恢复计数均变化，未见当前噪声是主瓶颈的证据|
|P2|静态per-channel gate|初始化可保留前向，小量参数|优化重参数化，迁移工程与涨点均需检验|
|P3|gate去decay、重做初始化、更多Write/heads|当前缺少支持主要瓶颈的证据|不挤占A/C与matched controls的预算|

### 更大Batch的具体账

当前每卡32×4卡×acc2=global256；可通过acc4变成512，通常不需要把单卡batch直接翻倍。这样改变梯度累积后的平均，不自动修复2K处Adam冷启动/LR冲击；C已提供更直接的解决方式。

从头按同一数据集训练10epoch时，global512约使更新数21480→10740，但前向/反向处理的总样本数不变，训练时间仍应按接近原80小时的量级估计，不是自动40小时。相同1000次更新时，512反而处理两倍样本、计算约两倍；公平筛选要明确等更新数还是等样本/计算预算。

同2000次warmup，256看约512000窗口、512看约1024000窗口；若按相同样本预算对齐，warmup应1000、300-update ramp应150。这是在换算计量，不是说这些数字保证最佳，也不能把新512结果与旧256混称相同训练量。

**不能在当前resume命令里直接把acc2改4。** `resume_if_configured`目前按 `global_step // grad_accum`推导opt_step；旧6000微步/3000更新在acc4下会被误算成1500，可能再次进入WAM warmup。scheduler总预算也需按既往/未来真实更新分别处理。当前专用A/C入口强制acc2，因此不会触发此问题；如未来试512，应先做独立恢复契约适配，不只是改一个环境变量。

### 学习率或增强怎么改才有信息量

若后期泛化回落重复出现，可优先考虑“只降低Video DiT实际LR、保留Action/Sidecar LR”的单变量候选，或更早进入后段衰减，检验预训练视觉先验保持。不能从本轮C的适配滞后反而推出现在应继续降低整个WAM LR。resume时旧optimizer/scheduler会覆盖配置值，必须验证真正的effective LR，不能只改yaml就声称已生效。

增强只考虑保持动作/EA标签语义的外观扰动。当前brightness/contrast/saturation已为0.2、hue为0；颜色相关任务不适合任意hue变换，几何crop/warp需要同步变换bbox监督，不能直接套用。若加入Random demonstrations，必须作为另一种训练数据协议并给Vanilla相同数据；本轮仍是原Clean2500。

不凭EA数值中Ground占比高就下调Ground权重，因为loss值不等于梯度规模。B已提示削弱共享监督可能损害语义学习；如要调整权重，应先用正确训练态路由分解真实梯度。

## 7. 部署侧的低成本候选，需单独标注

已核对当前部署配置是 `inference_horizon:32`。同步executor预测32步后消费整个buffer；即使客户端每步发观测，也不是每步重新推理，当前没有默认重叠chunk ensemble。

可以先单独比较32→16（之后才考虑8）的重规划频率，保持预测32步、动作语义和动作预算一致；完整diffusion调用约增加2倍/4倍，需测延迟/吞吐。它可能更及时反馈接触/物体状态，但这是部署策略变化，A/Vanilla/C必须同配置比较。

真正temporal ensemble还需保留重叠预测并按时间对齐。当前是绝对EEF，XYZ可在同一坐标系中加权，旋转需要合法SO(3)处理，夹爪/接触切换不宜无差别平滑，不能直接平均整个Rot6D/动作数组。这轮不改执行器或标准评估协议。

## 8. 决策节点

这轮4K后看C的Action差距是否收窄、Relation/Plan是否保持、Full/Off与XYZ/rotation/gripper是否改善。若有正向信号，再优先做三任务×双域×10个相同有效场景seed的A/C配对闭环，记录win/loss/both，不只比较两个成功率均值。

若C有可信收益或至少保持控制性能，才从其4K完整状态接回原10epoch预算；若证据接近，可再各续1K到5K，不据小样本强行选赢家。不因单次train loss高几个百分点就淘汰C，也不因曲线平滑就自动启动完整训练。

论文主线保留Matched Vanilla、关系监督与控制增量，而非将提升全部归因于LR技巧。当前先完成已授权的A/C对照，所有新增训练候选均未启动。


### 2026-10-27T08:58:52.413252+00:00 最后状态核验

原五组训练及离线诊断均完成、无失败。E全32步XYZ为Clean19.2743mm、Random26.5739mm（A分别15.6095/23.5921），固定Action FM相对A高3.07%；训练后250更新Action高2.47%，暂未显示全面优势。新A/C队列运行正常：08:57Z A为3039 optimizer updates（本轮39/1000），完整恢复与首更新验收通过，C按顺序等待；三份监督均活跃，无Traceback。C的4K实际结果及真实恢复仍待后续运行，当前不自动开启新的trick或完整长训。


### 2026-10-27T09:23:38.867547+00:00 — 用户授权C 4K后立即续原10epoch，评估并行；每2000次更新保存

用户在本轮明确要求：A/C各自3K→4K短对照照常完成，C到4K后不等待收益诊断结论，默认继续原10epoch预算，评估并行；若后续发现问题再考虑切换。这替代本文件及此前计划中“需先评估再启动长训”的本轮执行门槛，不代表已确认C提升SR。

新增独立队列root `/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_C_full_from4k_20261027T092219Z`，GPU0–3。复用专用入口的显式full_training模式，仅允许8000微步/4000optimizer updates→42960微步/21480更新。C短训到7400微步时预加载CPU权重骨架；骨架复用已固定C3K权重（硬链接，未复制环境或110GB状态），仅用于提前构建模型。必须等C4K完整状态标记、13文件、配套权重及短训torchrun真实exit0均通过后才授权GPU，随后通过原Accelerate/DeepSpeed恢复完整4K model/master/moments/RNG/scheduler。恢复counter与原cosine last_epoch16000/T_max81624、C factor=1均强制核对，warmup/ramp不重启，未改网络/loss/EA梯度/数据/增强/batch256/decay。A/C短对照的默认入口行为保持不变。

原4K输出与正式续训目录隔离。按用户最新要求，save_steps=4000微步且accumulation=2，即每2000次真实optimizer更新保存权重+完整state；末端42960微步/21480更新额外最终保存。沿用原keep_last_k_ckpts=2滚动保留，仅作用于新正式run，不删改旧3K/4K或评估输出。32K微步权重完成保存后仅硬链接固定一份weights/config/stats/tokenizer供历史最佳时点对比，不另复制optimizer，也不产生额外训练保存。

必要GPU/ZeRO恢复存在不可消除的交接时间；通过提前CPU加载缩短空窗，不声称零耗时。当前真实C4K恢复和首次4001更新仍待队列，工程测试与控制流模拟不等于GPU/SR证据。待评估结论仍保留near-plane、EEF replay、匹配Vanilla、少量GT/闭环样本等边界；不引入新trick或自动超参搜索。

跟踪入口：`outputs/sidecar/c_full_from4k_current.json`；实际状态 `/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_C_full_from4k_20261027T092219Z/job_status.json`。正式run将写在 `/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_C_full_from4k_20261027T092219Z/C_wam_ramp/runs/<timestamp>/`。原A/C4K的固定GT/FM诊断和连续Loss分析仍使用原队列root，在GPU5/6可用时运行，不等待完整训练完成。当前约6.6s/微步，剩余A约3.1h、C约3.7h及加载保存，预计UTC16:30–17:00前后可获得4K离线对比；GPU资源争用可能延后。此ETA不等于配对闭环成功率已测完。C4K到21480更新还需17480更新，纯计算约64h，加载保存/争用另计。

工程验收补记 2026-10-27T09:24:46.822810+00:00：20项CPU合同/真实子进程守卫回归/完整队列模拟通过，Ruff与编译通过。监督PID4137756已启动且状态waiting_for_C_preload_threshold；前序训练/离线诊断/Loss监督均核验存活，未提前加载额外GPU模型或停止现有作业。真实C4K恢复仍待执行。


### 2026-10-27T17:35:20.224706+00:00 — C 4K离线结果与新增强训练真实接卡

A/C均完成至8000微步/4000optimizer updates，完整状态/真实exit0及固定输入诊断均通过，无failed。少量GT窗口下，32步XYZ(mm)：A Clean12.2688/Random20.9182，C Clean13.9091/Random20.3011。C固定Action FM比A高13.16%；C Full相对自身Write-Off FM降低16.04%（A降低11.68%），但C Off也较弱，因此不能把更大Full/Off差距当C整体胜出。C训练最后250更新Action仍较A高5.94%。结论：延长后C进步且Random XYZ略好，仍无全面离线优势，更非闭环SR结论；GT仅24 heldout重叠窗口/6episode及32FM，边界保留。

新PatchWAM＋100ramp完整训练已真实接卡：旧C于16:24:09Z正常退出，新GPU gate于16:24:17Z释放（约8秒交接），新torchrun4168821/四rank已从Foundation开始更新。当前319更新/639微步，近期中位6.601s/微步，约13.20s/optimizer update，仍在前2K WAM不更新阶段，100-step ramp尚未开始。前50对最近50更新：Action 0.04375→0.04414，Plan 1.6026→1.1061，Relation 1.1568→0.7690。辅助目标下降但Action暂未下降，不据早期warmup断言增强好坏。CPU预检状态已由本次真实GPU更新解决；2K解冻、R100完整曲线、最终checkpoint和闭环收益仍未验收。

用户关注多视角一致性：已重新核对固定上游代码，p=.8按样本抽签，各相机独立采样photometric/style/Fourier参数及分支，同一相机9帧共享时间参数/噪声图；算法及参数范围相同，不是每路换不同配方。独立颜色/AdaIN可能增加跨相机外观对应难度，颜色指令也可能不满足标签语义不变，官方采用并不构成对OpenWAM的最优性证明。当前不热改在跑配方；若以后比较共享颜色/style需单独受控分支，共享hue本身也不保证保留颜色语言语义。未变更网络、数据或训练配置。
