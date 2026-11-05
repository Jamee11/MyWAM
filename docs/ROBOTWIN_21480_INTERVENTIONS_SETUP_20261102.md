# 原21480 Sidecar推理干预与XA单卡/split对照

用户2026-11-02授权推理阶段干预。原项目模型/训练代码和权重不变，所有适配位于独立评测目录。主模型固定为`outputs/1030-warmup2k-to-full10epoch/checkpoint_step_21480.safetensors`（原速测74.83%，SHA b141060b4ab33b0a2542effad452203b2ee5a5d287e5fd38ef092abe7326de7e）。

## XA部署数值对照

XA最终速测已100条件/3000条完成，独立audit通过：72.60%，Clean89.00%，Randomized56.20%。比原21480综合低2.23pp、Randomized低4.80pp；仅此结果不能归因于split。

同一XA权重（SHA f55e7641237f28bb14f004237b2cafaaad8dd7fc55d9c9d43537340075ede1d7），单卡GPU4与此前split副本保持同BF16/config/stats/cache/seed。两组固定fixture生成的动作已逐元素一致，最大差值0。证据`outputs/robotwin_eval/xa_single_split_parity_20261102/single_vs_split_actions.json`。

另启动一条`place_burger_fries / demo_clean`仿真，先用原生专家筛seed并封存1条指令/场景manifest；执行单卡策略动作，每次replan将相同实际RGB/proprio/语言请求同时发送单卡39600与split39500，对比完整32步动作chunk。逐chunk记录input hash/shape/max abs/exact结果，若有差异保存原始输入与两组动作。不把独立rollout的不同输入混成部署误差。该单episode只是数值诊断，不是单卡/split总体成功率对照。

目录：`outputs/robotwin_eval/xa_single_split_parity_20261102`，最终`live_parity_result.json`；结束后按PID/start_tick精确释放两份闲置XA对照服务，不影响原21480干预模型。

## 三组干预

|组别|精确操作|保留内容|
|---|---|---|
|Full|不修改Sidecar推理|全部3个Read/Refine/Write tap|
|Write-Off|将实例`write_action(index, actions, tokens, valid)`替换为恒等返回actions|Read/Refine、关系tokens、其余WAM推理|
|Relation-Reset|每tap完成read/refine后，clone tokens并将`tokens[:,4:8]`置零；修改后的tokens继续进入Write与后续tap|其余Ground/Summary/Plan token、Write计算、模型参数|

不采用仅交换关系token顺序：Write attention对KV行顺序可具有置换不变性，容易造成无效干预。Relation-Reset确实去掉指定latent信息，但也可能产生分布变化；它不是仅关闭辅助关系预测head。Write-Off组用于检验本checkpoint对直接写入的依赖，不能代替无Sidecar重训基线；干预降SR不能单独证明token特定语义正确。

## 首轮配对闭环速测

10任务×Clean/Randomized×3组×10条＝600条。10任务为：`adjust_bottle`、`beat_block_hammer`、`handover_block`、`place_bread_skillet`、`place_burger_fries`、`put_bottles_dustbin`、`place_object_basket`、`place_cans_plasticbox`、`stack_blocks_three`、`open_microwave`。覆盖抓取/交接/放置/堆叠/铰接及外观干扰，任务在干预结果出现前固定。

每task/domain先复用现有RoboTwin manifest builder的原生专家筛选与100候选语言路径，固定有效seed及unseen指令，再让三组执行相同不可变manifest/hash。保留实际EEF、RT、原成功判据/动作预算/物理规划；policy默认噪声seed42、10步去噪、horizon32、cache .99/max3、compile off、无视频。复用已验证chunk传输与中间采图gate。为配对生成的新Full结果是主要参照，不将历史全任务74.83%与10任务选择集直接比较。

三模型分别独立canonical和4会话，共12策略会话，4090最多12路队列（另有一条临时XA数值诊断仿真）。Full优先核对历史21480固定fixture动作。Write-Off/Relation-Reset不要求与Full动作相同，要求自身重复稳定。模型固定权重不加训练增强/EMA teacher。

首先生成20份manifest，ready后弹性执行60份策略条件。按实时6GiB准入、每GPU最多4路，容量失败保留日志重排，最多20attempt。每份结果验收进程exit0、manifest hash、10条对应seed/指令逐条一致、成功计数与逐episode结果一致。报告只在三组相同task/domain全部完成后计算配对均值，并保存Full成功/干预失败及反向不一致次数。

## 文件与进度

- 主目录：`outputs/robotwin_eval/sidecar21480_interventions_20261102`。
- `experiment.json`：checkpoint、任务与预算；`models/*/intervention.py`：实例干预；`intervention_scope_checks.json`：CPU作用范围检查。
- `state.json`、`comparison.json`、`schedule.py.log`、各attempt的manifest/result/client.log；实时文档`docs/ROBOTWIN_21480_INTERVENTIONS_LIVE_20261102.md`。
- 10条/条件只能快速识别较大效应，小差值不能确定显著性；如有值得研究的方向，后续应扩展同manifest样本或复验，而非从单次小样本推结构普遍因果。

## 启动验收与初始进度

2026-11-02：三组固定fixture重复/数值检查通过，Full与原21480历史canonical逐元素一致；Write-Off和Relation-Reset在相同fixture下产生不同engine动作，证明干预非无效hook（结果intervention_action_effects.json）。CPU三组作用范围检查通过。运行12路队列，已封存10份manifest；首个完整策略条件beat_block_hammer/Clean/Full为10/10，但尚无三组齐全的配对条件，不能报告干预胜负。真实XA对照已完成7chunk/224步预测动作逐元素相等；单episode成功，原XA单卡/split服务已按精确PID释放。

## 2026-11-02T06:45Z：59/60组完成，恢复最后卡住的Full组

已配对齐全19/20条件：Full147/190=77.37%，Write-Off152/190=80.00%，Relation-Reset156/190=82.11%。Clean100条分别92/91/93成功，Randomized90条分别55/61/63成功。最后place_bread_skillet/Randomized/Full在初始化后约3小时无progress/动作记录；按PID/start_tick验证后停止其单个自有进程组，保留日志和stalled_recovery.json，调度器按同一manifest自动重跑。59组结果未重测，原21480剩余32条件继续等干预60组完整验收。该19条件初筛不支持“关闭Write明显损害性能”的预期，尚不能据10条/条件宣称干预稳定获益。
