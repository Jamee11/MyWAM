# 42960：GT动作误差与关系表征诊断

更新时间：2026-10-21T14:17:54.921886+00:00。输出：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar42960_gt_diagnostic_20261021T135950Z。实验一/三不进行optimizer更新；实验二本轮先明确独立梯度探针执行路径。

## 第一阶段：已完成的Clean训练内诊断

固定42960、三个任务adjust_bottle/place_shoe/beat_block_hammer、每任务episode0/1各两个anchor，共12窗口/6episode。源数据是旧模型已训练过的Clean，不是held-out。相同输入、seed42、十步、eager且无cache；五倍率共60组。GT由现有LiteRoboTwinSource读取t+1..t+32 absolute endpose/rot6D/raw gripper，关系标签复用带SHA验证的130D sidecar标签；GT只评分，不送部署输入。三次gate每次临时缩放后恢复，逐组检查beta恢复。首阶段未单独重跑同一Full数值重复性，将在第二阶段增加重复/首阶段参考核对。

|倍率s|XYZ平均误差mm|旋转平均误差deg|Gripper MAE|相对s=1 XYZ变化mm|XYZ较s=1更低窗口数|
|---:|---:|---:|---:|---:|---:|
|0|3.716241|0.904940|0.003435|-0.032356|9/12|
|0.5|3.709731|0.910775|0.003411|-0.038866|10/12|
|1|3.748597|0.907794|0.003405|+0.000000|0/12|
|2|3.717007|0.904754|0.003377|-0.031590|10/12|
|5|3.719209|0.911607|0.003403|-0.029388|9/12|

XYZ变化没有单调随s改善；s=0与s=2/5都略低于s=1，改善幅度约几百分之一mm，不能作为增大gate的充分证据。不能以全部窗口均值抹掉任务、方向、gripper间的取舍。统计是预测相对专家未来endpose，不是机器人真实跟踪误差或SR。

|Tap|Contact F1|Near F1|Moving F1|移动方向mean cosine|GT框内attention mean|
|---:|---:|---:|---:|---:|---:|
|7|0.800000|0.888889|0.744186|0.339610|0.425946|
|15|1.000000|1.000000|1.000000|0.995243|0.094919|
|23|1.000000|1.000000|1.000000|0.995427|0.420343|

每窗口仅取第一次真实去噪forward的三层head输出，不把10步重复读同一首帧当独立样本。Contact仅12个有效track观察、4正例，Moving为72个有效角色×horizon项、45正例，方向cosine也只在这45个移动项上统计；全部来自6条训练episode，不能外推Randomized或held-out。后两层能拟合本小样本，不等于Action使用了这些关系或获得控制收益。Plan两类运动角色按源码是EEF与Object，不是Goal；Goal是关系/grounding监督，旧规划文档相关表述以此更正。中间tap监督种类不同，不能按head分数简单排名层价值。

## 第二阶段：已启动的新专家轨迹GT采集

复用原v6 oracle（快照oracle_snapshot.py，原DiT4DiT模块不改），在与正式评估相同的官方RoboTwin commit0aeea2d669c0及A100/Curobo环境中，GPU4采Clean、GPU7采Randomized。使用上一轮sealed manifest各条件第一条，共6个场景，保存fresh expert play_once观测/原endpose/130D标签所需oracle；不伪称官方缓存轨迹，不改变训练数据。先验收EEF→TCP几何、wxyz/rot6D方向、RGB/视角/目标有效性，再导出每条最多四个完整32步窗口。

已有专家采集控制器expert_collection.json与每条件client.log，仍在运行，不能提前宣称Randomized指标。独立heldout_eval模型CPU载入与采集并行，等待所有GT完整验收后才占用GPU推理；原训练与正式评估不停止。该阶段每域只有三条episode（每任务一条），只能作为小规模新场景诊断，不是有统计把握的泛化评测。训练原数据混合来源与wrist配置可能与当前原生eval不同，Clean/Random的原生比较也不自动是仅改变外观的严格配对。

## 实验二：具体运行方式

不提交训练、不建立optimizer；使用同一只读checkpoint和固定带GT的batch。先B=1、一个窗口，初始化原训练scheduler，缓存CUDA文本embedding与GT视频VAE latent；开启autograd并仅对Sidecar参数求梯度，冻结backbone参数但保留穿过backbone的输入梯度。启用activation checkpointing，cache/compile关闭；有必要才CPU saved-tensor offload。先测峰值显存，避免按此前双卡训练OOM直接推断这个探针也不可能，或反过来保证一定装得下。

入口路径：现有reader样本→architecture.prepare_inputs→捕获未detach的action/video/EA损失→torch.autograd.grad三次；不对返回的detach日志求导，不执行optimizer.step。比较三类loss对各tap gate、Write Projection、Read/Refine的梯度及共享非零路径cosine。期望EA→Write结构性None/0、Action→EA head为None/0，不把它们算bug。用真实total loss一致性和少量有限差分交叉检查捕获逻辑。

先4个窗口的可行性探针，再32个任务均衡microbatch及低/中/高噪声分布。分析dL/ds=sum(beta*dL/dbeta)，区分增大现有有符号写回究竟降低还是提高Action/Video loss；用sweep交叉验证。无42960配套Adam状态就不报告“真实optimizer更新量”；36000/40000状态只能在对应checkpoint使用。该梯度实验尚未实际运行。

## 验证与当前边界

五项CPU测试通过：保留原write-only/异常恢复/物理单位测试，新增倍率0与off一致、倍率2后gate恢复及非法倍率拒绝，GT框区域/Contact与Moving mask/局部方向单位校验。源权重和训练配置未改；当前没有新训练结果、held-out Randomized指标或新的闭环SR。


## 2026-10-21T14:51:31.586135+00:00：原生新场景实验一/三完成

三个任务各一条Clean与一条Randomized专家轨迹，共6episode；每条4个完整32步窗口，共24窗口。另有一个训练内控制窗口。五倍率共125组生成（控制窗口包含其中5组），与第一阶段60组分开。所有新seed与各任务训练collection manifest的seed集合无交集；这是新seed场景，不表示全新任务/物体类别。6个专家轨迹都成功，但该成功率是GT采集验收，不是模型闭环SR。
六条轨迹全部通过长度/三相机/finite oracle、EEF→TCP位置（最大1.19e-7m）和wxyz/rot6D姿态一致性检查；重新构造8/16/32局部运动标签、goal-valid bbox mask。GT只用于评分。Full同进程重放逐位一致，跨两个加载进程的训练内控制也逐位一致，max_abs_error=0。

### 写回强度对GT动作误差

|域|s|XYZ前8步mm|XYZ全部32步mm|旋转全部32步deg|Gripper MAE|XYZ较s=1更低窗口数|
|---|---:|---:|---:|---:|---:|---:|
|Clean_heldout|0|4.216286|6.354801|1.399824|0.004462|5/12|
|Clean_heldout|0.5|4.197533|6.345220|1.403037|0.004481|5/12|
|Clean_heldout|1|4.226862|6.347106|1.401919|0.004541|0/12|
|Clean_heldout|2|4.182603|6.352581|1.394455|0.004481|3/12|
|Clean_heldout|5|4.186746|6.338131|1.397252|0.004469|5/12|
|Randomized_heldout|0|9.016105|12.621586|3.137911|0.008391|5/12|
|Randomized_heldout|0.5|9.041906|12.621348|3.138967|0.008400|5/12|
|Randomized_heldout|1|9.036853|12.608922|3.138657|0.008408|0/12|
|Randomized_heldout|2|9.080178|12.649299|3.135709|0.008349|3/12|
|Randomized_heldout|5|9.002958|12.599763|3.137147|0.008383|8/12|

s=5相对s=1：Clean全部32步平均XYZ仅改善0.00898mm（5/12窗口改善、median delta反而+0.00379mm），Randomized改善0.00916mm（8/12窗口改善）；s=2反而使两域平均更差。没有单调、一致的倍率收益，不能据此选择增大gate重训。窗口不能当独立episode，当前每域只有3episode，且Clean/Random有场景/动作阶段差异，不能把域间差异全归因于外观。

第一阶段实测实际mean r，s=1各tap约3.07e-6/6.08e-6/8.58e-6，s=5约3.41e-5/6.64e-5/9.36e-5；并非简单线性5倍，符合BF16小残差舍入与后续状态变化共同作用。该范围仍很弱，没有支持“强制某个r就是最优”的证据。

### 最终tap23的关系/运动质量

|域|Contact F1|Near F1|Moving F1|方向对齐mean|Ground peak-in-box|
|---|---:|---:|---:|---:|---:|
|Clean_heldout|1.000000|0.888889|0.971429|0.913037|0.950000|
|Randomized_heldout|1.000000|1.000000|0.980769|0.941373|0.900000|

每域Contact12个有效track观察（6正6负）；Moving72个有效角色×horizon项，Clean52正/Random53正；方向只在这些moving项计算。来自各3条episode，不是72条独立episode。Ground定位有效role总20个；不将其写作BBox IoU或Role Accuracy。方向指标遵循现有normalized head输出与GT的clipped dot，BF16近1的数值受量化影响。

加一个不更新policy的任务先验对照：从第一阶段12训练窗口按task/arm/role/horizon统计常量方向和moving频率，再在新场景评分。不是公平的完整学习baseline，只用于排除“所有方向都是常量”的最简单解释。

|域|常量先验Moving F1|模型Moving F1|方向支持项|常量方向mean对齐|模型在相同支持项mean对齐|
|---|---:|---:|---:|---:|---:|
|Clean_heldout|0.645833|0.971429|31|0.094677|0.932134|
|Randomized_heldout|0.849558|0.980769|52|0.126870|0.940686|

方向先验未见过的训练role/horizon不强行填方向，方向对照仅比较共同支持项；因此不能把该表的子集方向与上表全体均值混算。结果说明Sidecar在这些样本能给出超过简单task常量的关系/运动信号，但不证明其信息超越原backbone或来自视觉而非proprio/任务阶段线索。

### 已查明的监督粒度边界与下一步

当前130D标签有完整relation_xyz，但EAHeads关系loss只把其范数转成near二分类，再加contact与goal proximity；不直接回归完整相对XYZ。Plan监督是EEF/Object局部单位运动方向+moving，没有位移幅度监督。因此高F1/方向对齐不等价于学到了毫米级动作所需的全部几何信息；latent可能含这些信息，但当前监督并未要求保留。该点是待检验的机制解释，不是已证明的失败原因。
当前证据更偏向“可读关系信号存在，但直接写回未转化为可观GT动作收益”，尚不能区分写回优化不足与信息冗余。优先结合Action/Video/EA梯度、token内容干预及之后Matched Vanilla判定，不因头分数高就直接放大write或立刻加入新loss。

实验二首次真实batch遇CPU/CUDA设备不一致：独立reader返回CPU proprio_mask，正常训练由Accelerate dataloader搬运；已在诊断入口统一prepared tensor设备，不改生产训练代码。专门gradient_probe已启动复测；待完整活loss/梯度输出才能宣称真实checkpoint梯度探针通过。


## 2026-10-21T14:56:06.449540+00:00：实验二真实42960单batch可行性通过

独立gradient_probe模式复测通过；首试错误已通过搬运prepared tensor（包括proprio_mask）到architecture.device解决。没有修改生产训练路径。使用已有adjust_bottle训练episode0首窗口、B=1、无增强、seed42、eager/no-cache、冻结backbone参数并保留输入梯度、activation checkpointing；不建立optimizer、不执行step。Video训练时间步ID500，Action实际采样ID67（action时间步不会受video min/max边界限制）；两者均已记录，不能声称都固定500。

损失：Action=1.2431879e-05，Video=0.099323764，EA=0.09656024；原始活loss重建total一致。EA grounding交叉熵含软目标熵下界，不能仅由loss标量差几千倍推断梯度主导，下面直接测梯度。GPU峰值allocated=14.351GiB，覆盖GT预编码与求梯度；这是仅求Sidecar梯度且没有Adam状态的诊断，不代表全量训练显存。

|tap|Read/Refine Action梯度norm|Video梯度norm|EA梯度norm|EA/Action比|
|---:|---:|---:|---:|---:|
|7|1.32147e-07|1.86612e-06|0.00350967|26558.8|
|15|7.84522e-08|1.14724e-06|0.00480401|61234.9|
|23|4.97266e-08|6.65342e-07|0.00293342|58991.1|

|tap|dAction/dbeta|dVideo/dbeta|dEA/dbeta|
|---:|---:|---:|---:|
|7|-4.23193e-06|-0.000146866|0.0|
|15|1.37091e-06|-0.000150681|0.0|
|23|1.2219e-05|-0.000103951|None|

共同非零参数上的梯度cosine：Action–EA=0.065187，Action–Video=-0.279852，Video–EA=-0.101513。这是合并共享参数的值，不冒称每层/每批都冲突。

按当前有符号beta求局部放大全部write的dL/ds：Action=-4.46977e-09，Video=2.24063e-08，EA=0。该batch中Action方向轻微偏向增强，而Video方向相反；不等于任意有限倍率或最终EEF误差都按此改善。

三个Write Projection的Action梯度norm约8.55e-8/1.69e-7/2.24e-7，均非零，Video约1.21e-6/2.44e-6/2.31e-6。EA对前两层Write有图连接但导数为0，最后Write为None，符合首帧掩码与读写先后关系，不是梯度断路bug。所有Sidecar参数逐tensor检查与探针前完全相同，requires_grad标志已恢复；无权重更新或源checkpoint保存。

这个batch的Read/Refine上EA梯度约为Action的2.7万至6.1万倍，提示Action弱反馈与EA主导值得多batch诊断；也可能受该训练内窗口、噪声时间步和已拟合程度影响。不能据此断言整个训练都有同样比例、必须削弱EA，或已证明某个新初始化能提高成功率。

下一步实验二可直接沿用benchmarks/robotwin/sidecar_gradient_probe.py的measure(architecture, prepared)，将现有单batch入口扩到任务均衡的32批，并显式控制/记录Action与Video低中高时间步。保持只求梯度、没有optimizer；不要通过仅设video min/max假定Action也被限制，不要对detached日志求导。再根据跨batch分布选择Write初始化/独立优化或梯度路由改动，暂不启动新训练。
