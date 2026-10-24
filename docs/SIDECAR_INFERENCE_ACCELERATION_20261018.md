# Sidecar 20k 推理加速与官方环境评测（2026-10-18）

**当前方案已通过验收：Sidecar三层FFN局部compile + 官方联合cache，完整Sidecar读写保留。** 同GPU真实输入耗时0.912→0.401秒/32动作块，约2.28倍；两fixture在相同缓存条件下编译前后最大动作误差0。全模型compile尚未通过原数值门槛；本轮加速主要来自缓存。正式百条队列已开始，缓存近似计算的实际成功率仍待闭环结果。

## 目标与边界

仅适配推理：保留Sidecar权重、三个read/refine/write层（block7/15/23）、80D状态/动作mask、wxyz训练合同、自有归一化统计和原生32动作/9视频帧/10步联合去噪。模型架构、训练、监督数据、专家筛seed、RT、成功判据和动作预算均不改。20k文件名对应10000次optimizer更新。

新官方仿真为`.external/robotwin-official-20261018/RoboTwin`（0aeea2d669c0f8516f4d5785f0aa33ba812c14b4）与`.conda/robotwin-official`，PYTHONNOUSERSITE=1；模型推理沿用项目`.venv`。镜像资产已验上游revision/size/SHA。指定权重物理副本SHA256为0d67d651577ef53d5a86b0e1e03dd93f6cc7bf652c3c7af08734b7712042ea6e。产物根：`outputs/robotwin_eval/sidecar20000_officialenv_20261018/`。

## 为什么不能直接打开原版开关

原Sidecar实现主动拒绝DiT cache，并禁用原版compiled MoT loop。原版循环没有Sidecar，直接套用会跳过Sidecar计算；仅把配置enabled改成true，既不能证明真实加速，也不能证明能力保留。

新增独立`benchmarks/robotwin/sidecar_fast_runtime.py`，只有显式启动该入口才安装本进程适配，不改原架构/训练/默认部署源码。编译对象是完整Sidecar.forward，原版foundation-only compiled loop保持None。继承原联合生成循环的joint video/action velocity cache，0.99阈值、最多连续skip3，每次generate重置，端口间动作deque独立、完整预测互斥。

缓存保存的是已经过Sidecar的联合预测，不是关掉Sidecar后生成的foundation预测。但缓存是近似计算：跳过的去噪步不再重算Sidecar/主干，不能称与逐步完整计算等价或预先保证相同成功率。必须保留完整计算参考与缓存统计，以闭环评测检验实际收益。

## 验收方法

`check_sidecar_fast_runtime.py`使用两组真实HDF5 RGB观测（瓶子、单鞋），将原fixture的legacy四元数状态正确转换到Sidecar的wxyz训练合同。固定seed42，比较eager、eager+cache、compile、compile+cache。每组先记录cold调用，再做3次warm调用；torch.cuda.synchronize前后计时，统计的是完整engine.generate（一次32动作），不含WebSocket排队/仿真。

每种变体保存全部动作、每次耗时、缓存skip与实际Inductor kernel生成数。编译对照分别是compiled vs eager、compiled_cached vs eager_cached，预先设置atol0.005/rtol0.01。此门槛不是bitwise等价；缓存与非缓存之间的动作差异另外报告，不作为“缓存必须等价”的错误假设。CPU测试验证编译目标是Sidecar完整forward（非零write gate），保留active mask及禁止privileged输入。

## 首轮经验：默认BF16融合不能直接放行

同一GPU0、两fixture共6次warm调用，首轮结果：

| 变体 | 平均秒/32动作块 | 对完整计算速度比 |
|---|---:|---:|
| eager | 0.95865 | 1.00x |
| eager + cache | 0.43736 | 2.19x |
| compile | 0.62616 | 1.53x |
| compile + cache | 0.26386 | 3.63x |

缓存两fixture均10步skip6，实际重算4次。编译确实执行：6717 captured calls、6 unique graphs、765 generated kernels。但默认融合compiled vs eager的最大动作误差分别0.01953125、0.0078125；cached对应对照也达0.0078125，未通过原定容差。不能只凭速度启用或放宽门槛宣称parity通过。失败报告/全部动作保留在`fast_validation_default_fusion/`，正式launcher正确拒绝启动。

读取本机Torch2.7.1 `_inductor/config.py`：默认融合会省略BF16运算之间的downcast/upcast，改变中间舍入。第二轮使用`emulate_precision_casts=True`及`force_same_precision=True`，保留eager舍入边界；不改变权重、网络拓扑、推理步数或缓存阈值。第二轮验收状态及最终正式入口见后续追加记录，尚未出结果前不能宣称解决。

## 新环境闭环与正式统计

未加速迁移smoke：adjust_bottle/Clean 2/2、place_dual_shoes/Clean 2/2，exit0/complete=true。只证明链路可用，样本极少，不并入正式百条成绩，不推出整体效果。

正式启动器`launch_sidecar_official.py`要求CPU与GPU数值门槛通过，随后在模型服务预热完成、reset清空fixture动作后才放行客户端。全量50任务×2域×100，三个无重叠任务分区。旧环境结果、短测、失败编译尝试与正式加速队列分别保留，禁止混算。显式记录实际优化方案，不能把加速结果误称未经适配的官方Sidecar实现或纯架构消融。

若最终无法在保留Sidecar计算和可接受数值行为下加速，按用户指示使用原已有关闭cache/compile方案多GPU推进；优先尝试解决加速，不无限期因编译阻塞所有效果评估。

## 原版baseline的九小时验证安排

原先只有一个native仿真进程，五任务两域串行，无法保证九小时内完成。传统官方流程的双鞋Clean92/100已完成，Randomized继续；保留该原始进程和日志。

另行复用新下载官方OpenWAM中原有的`eval_policy_wrapper.py labtasker`入口：先从seed100000起按专家成功/规划成功连续筛100个，封存seed/指令及manifest hash，再按10条逻辑episode分片、8路评测。无重复、漏样本或按策略表现挑seed。原生成功判据/预算/RT不改，结果以官方结构化JSON的实际100条汇总，不用旧小样本_result.txt的固定100分母。

官方manifest路径还会固定instruction和CUDA RNG，并将专家筛选与policy执行分离；它是官方提供的另一条协议，不冒充传统single_eval逐位相同。传统与manifest结果分别报告，不拼接。清单/分片目录`outputs/robotwin_eval/baseline_official_manifest_deadline_20261018/`；目标2026-10-19 01:43 UTC前完成余下9个任务条件，完成时间需用实际吞吐持续验证，不作无依据保证。


### 2026-10-18T16:58:50.387754+00:00：Sidecar编译范围隔离与原版9小时并行验证

第二轮仅保留BF16舍入仍未通过原门槛（compiled max0.0234375、cached对应0.0078125）；原始产物保留fast_validation_precision_casts，不放宽容差。第三轮同一已加载模型依次测试conservative全forward（关闭pattern/epilogue融合、fusion size1）、仅Sidecar read/write、仅三层Sidecar FFN，全部保留原主干/三处读写，均按原eager/同缓存eager参考检查。新增serve_sidecar_checked.py在同一模型上完成验证后直接服务，避免重复加载；只有通过者才写最终deploy_selected.yaml/打开服务。四项CPU测试通过130.11s，相关入口ruff通过；实际GPU数值/速度仍待本轮结果。

用户进一步要求原版异常任务至多9小时出结果。原流程实际仅1个native客户端，仍原样保留；新增官方原有labtasker manifest/run_eval调度（源码不改），3个专家清单builder已实际筛seed，8个策略worker等待100条封存清单。任务为双鞋Randomized及单鞋/扫码/递积木/铃铛两域，共9项900条，每项10个无重叠10条分片。新原版模型pool GPU4、19900–19907，原19400/原native继续；部署默认cache/compile。Sidecar调整8sim（model0/1/7；sim0:2,5,0；sim1:6,2,1；sim7:6,7），原版分片8sim（3,3,3,5,5,5,6,6）以保留显存余量，不操作其他namespace/未知作业。目标截止2026-10-19T01:43UTC，须基于实际吞吐检查，不保证硬件未验证的绝对工期。

官方manifest另含固定instruction/CUDA RNG与任务状态初始化，不能声称与传统native逐位相同，也不能把两者样本拼接；分别报告。结构化JSON汇总按完整100及manifest hash/episode覆盖验证，禁止用上游小样本_result.txt固定100分母。保留native双鞋Clean92/100。目录baseline_official_manifest_deadline_20261018；进度/截止与有效结果自动更新docs/ROBOTWIN_LIVE_RESULT_ANALYSIS_20261018.md，并将新完整结果追加EXPERIMENT_RESULTS_AND_ANALYSIS。专门加速经验文档SIDECAR_INFERENCE_ACCELERATION_20261018.md已建立。


### 2026-10-18T17:07:59.134583+00:00：最终采用FFN局部compile + 联合cache，原门槛通过

本轮8变体在GPU0同两真实输入、每组3次CUDA同步warm，eager0.912392s，eager_cached0.377508s，conservative0.647082s/cached0.275161s，Sidecar读写compile0.941083s/cached0.388895s，FFN局部compile0.939386s/cached0.400740s。前两种编译范围至少一个未缓存参考未过原门槛，均保留失败，不放宽容差；FFN策略的两fixture×未缓存/同缓存四组动作对照max_abs_error全部0.0，原allclose门槛通过。FFN确有4个新Inductor kernels，非仅把enabled写true。

选择ffn：只编译3层Sidecar内部FFN，主干与三次read/refine/write全部保持原路径，未旁路/关闭Sidecar。cache开启0.99/max3，每generate reset，10步实际重算4次/复用6次已实测。组合相对完整eager约2.28倍，速度贡献主要来自cache；本轮FFN compile未观察到额外吞吐收益（相对cache-only略慢），不能称全模型compile已解决。相同缓存条件的编译保真不等于缓存与完整计算等价；缓存相对完整eager的两fixture动作最大差0.03125/0.01171875，须闭环百条判断成功率。

最终配置outputs/robotwin_eval/sidecar20000_officialenv_20261018/deploy_selected.yaml（compile.enabled=true/sidecar_strategy=ffn/cache=true），数值/耗时/全部动作证据fast_validation/result.json。checked模型GPU0直接复用为19600–19602，已预热/reset并启动正式queue0三个百条客户端：adjust_bottle两域、place_shoe/Clean；新官方Conda与runtime/PYTHONNOUSERSITE=1，原RT/预算/成功判据不改。模型GPU1/7新副本正在加载，各预热后启动queue1/queue7，合计8sim/50任务两域10000条，不混短测/旧环境。


### 2026-10-18T17:18:05.575263+00:00：正式新环境双队列启动验收

Sidecar三个模型GPU0/1/7已全部READY并以真实fixture预热/reset，三个pool配置SHA一致d1de0dd89a16eae0a9cb08b685c265100b17b766c77c6d0cf27d35dca70ceb6b。queue0/1/7分别3/3/2个百条作业启动，父进程实查全部存活，共8路覆盖100个唯一task/mode、10000条；queue0已完成真实成功episode，新增各路仍有初始化过程，不能把启动当任务完成。当前任务瓶子两域、单鞋Clean、双鞋两域、扫码Clean、锤子两域。

原版传统官方single_eval仍保留；新增官方manifest并行组8个run_eval子进程已实际创建，双鞋Randomized100专家清单封存，递积木Randomized100亦完成并开始下一专家任务，3个清单builder继续。run.json当前无errors。目标截止2026-10-19T01:43UTC；实际耗时须看分片吞吐，不把队列创建称已完成。每个清单全100，逻辑0–99分成10条且验证无重复；分析仅将协调器确认exit0/manifest/逻辑覆盖的完整100条纳入替换，不以progress=100代替验收。

启动验收证据outputs/robotwin_eval/sidecar20000_officialenv_20261018/launch_acceptance_snapshot.json，包含pid/存活/配置hash/当前baseline任务/显存。四项针对性CPU测试通过130.11s，七个新文件ruff全通过。加速经验与策略限制见SIDECAR_INFERENCE_ACCELERATION_20261018.md；实时结果、共同完整任务对比、替换平均及截止预测自动写ROBOTWIN_LIVE_RESULT_ANALYSIS_20261018.md，新完整成绩自动追加EXPERIMENT_RESULTS_AND_ANALYSIS.md。
