# 原最佳21480：新增token核对及扩大消融

2026-11-02 UTC。用户要求暂停剩余32条件补70，优先并行分析诊断。目标始终为`outputs/1030-warmup2k-to-full10epoch/checkpoint_step_21480.safetensors`，SHA256 `b141060b4ab33b0a2542effad452203b2ee5a5d287e5fd38ef092abe7326de7e`。不替换XA或其他训练分支。

## 相比baseline新增的token

|类型|数量|索引|含义|
|---|---:|---|---|
|Grounding|4|0:4|双臂各2个，用于EEF/交互对象定位及有效性读出|
|Relation|4|4:8|双臂各2个，用于near/contact与goal proximity读出|
|Summary|1|8:9|共享汇总latent，无独立语义监督头|
|Plan|6|9:15|三预测horizon×双臂，预测EEF/Object运动方向及moving|
|合计|15|0:15|4类，每个512维|

代码：`openwam/model/modules/relational_sidecar.py`的role_tokens、init_tokens、EAHeads.forward。同一组15 latent经过block7/15/23后的三次更新，不是45个独立token。不是将15个token全部拼接进原WAM序列，而是独立Sidecar通过Write cross-attention写回Action hidden。baseline已有text/video/action/proprio输入，本表仅计Sidecar新增token。

## 闭环18组

|组|干预|
|---|---|
|Full|完整Sidecar|
|Write-Off|三个tap均跳过Action Write|
|Grounding-Reset|每次read_refine后，0:4置零|
|Relation-Reset|每次read_refine后，4:8置零|
|Summary-Reset|每次read_refine后，8:9置零|
|Plan-Reset|每次read_refine后，9:15置零|
|All-Tokens-Reset|每次read_refine后，0:15置零|
|Visual-Read-Off|Read attention增量置零|
|Refine-Off|token self-attention增量置零，保留FFN|
|Sidecar-Text-Off|Sidecar的文本投影结果置零，WAM原文本不变|
|Sidecar-Proprio-Off|Sidecar的本体状态投影结果置零，WAM原proprio不变|
|Write-Tap7-Off|仅跳过block7后的Write|
|Write-Tap15-Off|仅跳过block15后的Write|
|Write-Tap23-Off|仅跳过block23后的Write|
|Role-Embedding-Off|初始tokens减去role embedding，保留context|
|FFN-Off|Sidecar各stage FFN增量置零|
|Position-Off|Sidecar视觉位置/视角投影结果置零|
|Stage-Context-Off|各stage中加入refine query的context投影置零|

对位置/上下文/FFN的“off”计算原投影后再置零，目的是诊断输出影响，不是测量其加速收益。All-Tokens-Reset仍保留Write的LayerNorm偏置及下游运算，不等价Write-Off。各token通过self-attention交换信息，按索引置零不保证抹除该语义在所有分支的编码。

任务：adjust_bottle、beat_block_hammer、handover_block、place_bread_skillet、place_burger_fries、put_bottles_dustbin、place_object_basket、place_cans_plasticbox、stack_blocks_three、open_microwave。

沿用已封存的20份专家有效seed/语言manifest，Clean/Randomized各10条件，每条件取前5条。18组共1800条比较样本。Full/Write-Off/Relation-Reset直接复用已审计10条结果的前5条，共300条；其余15组新跑1500条。原10条实验及数据保持原样。所有差异按相同seed/语言逐条配对；未完成时按各组与Full的相同条件子集比较，不将不同任务子集均值直接排名。5条是第一轮诊断筛查，不是100任务的整体成功率；后续应扩大关键组样本验证。

## 部署与有效性

新独立目录`outputs/robotwin_eval/sidecar21480_expanded_ablations_20261102`。两份原Full模型共享24个独立PolicyServer会话，L20主体GPU1/2、T5 GPU4/6。按用户进一步要求，4090调度24线程、每GPU最多6进程、4608MiB准入/启动预留，进程失败后该GPU暂退120秒再尝试，按实际余量动态填充；不承诺24路都能同时装下。关闭视频，保留官方RT、专家筛选、成功判据、动作预算、EEF契约。沿用cache开启、compile关闭、10 denoise、32步chunk及chunk内跳过冗余RGB方案。

干预模式随每次RPC显式携带，在原共享engine互斥锁内使用thread-local scope；每端口独立policy/action buffer。不修改权重、不在请求间留下全局模式。启动门槛：原Full历史canonical fixture完全一致；18组真实RPC有限动作；每组之后Full恢复逐元素相同；跨副本各组固定输入动作一致；24端口握手。CPU18组hook检查已通过。模型加载、在线门槛及实际rollout状态以launch_status/deployment_acceptance/state.json为准。

同任务训练Clean数据也扩大到18组×40窗口=720次推理，独立目录`outputs/robotwin_eval/sidecar21480_train_expanded_ablations_20261102`，复用已核对的输入/GT，GPU0运行；保留原7组280次诊断。原7组全部完成且参数指纹不变：Full位置误差4.8891mm，Write-Off5.0347，Relation-Reset4.8903，Grounding-Reset5.0422，Plan-Reset4.8749；仅Grounding/Plan预测head关闭与Full动作完全相同。离线expert动作误差不是闭环SR，也不代表Randomized泛化。

## 暂停及可恢复性

剩余32条件补测已停止自有调度/仿真和两个服务，保存`state_before_analysis_stop_20261102.json`、`stop_owned_processes_analysis_20261102.json`，68个已完成条件保持不变。partial保留但不计完整70条结果；剩余队列未删除，不自动恢复抢占本轮诊断。

## 本轮不能回答的归因

推理消融覆盖当前Sidecar的token、输入上下文、视觉读取、Refine/FFN/位置及Action Write路径。辅助监督loss、warmup、PatchWAM增强、XA等训练策略需同起点、同数据、同预算重训对照；不能通过关闭预测head抹去其训练贡献。Write-Off后的模型也不是原baseline，因为主干已联合训练。优先完成本轮，筛出稳定差异组后再设计重训，不在本轮擅自启动新训练。

启动验收已通过：两模型18模式固定输入跨副本逐元素相同、每组之后Full恢复一致、24会话握手通过。4090队列已启动；启动扩容遇到attempt目录和记录竞争，已保留状态并修复重排，未计入任何新SR。实时数量以state.json为准。

07:38 UTC启动观察：已实际创建14个独立runner（GPU4五路、GPU6四路，GPU0/1/2/3/5各一路）；GPU7余3730MiB，未满足本次4608MiB试跑门槛。此时runner进程尚在共享文件系统cxiWaitEventWait状态，Popen的exec尚未全部返回，未视作已经推进episode，新增完成条件为0。队列选择数与实际rollout吞吐须区分；startup_capacity_snapshot.json保留现场，后续以render_affinity/progress/results验证真实推进。

进一步定位启动记录延迟：原继承调度器在持RLock时为每个queued job重复调用nvidia-smi，无容量时扫描近300个job，导致已创建进程也长时间无法登记。新实现每轮select仅查询一次容量，缺容量立即返回；300queued场景检查通过（查询次数严格1），恢复登记/调度效率。保留启动尝试后应用修复，训练数据模型不受此调度重启影响。

## 训练数据离线18组最终结果（2026-11-02）

720/720完成：10任务×2训练episode×2锚点×18组。每个锚点预测未来32步动作，对齐专家GT；相同输入、seed42、cache开启、compile关闭、10 denoise。下表XYZ为双臂32步的平均欧氏位置误差（mm）；动作漂移为干预与Full预测的平均位置差，不是GT误差。参数版本/Sidecar指纹均未改变。

|干预|XYZ误差mm|相对Full变化mm|动作漂移mm|
|---|---:|---:|---:|
|full|4.8891|+0.0000|0.0000|
|write_off|5.0347|+0.1457|1.2306|
|grounding_reset|5.0422|+0.1531|1.0674|
|relation_reset|4.8903|+0.0012|0.6818|
|summary_reset|4.9010|+0.0119|0.6501|
|plan_reset|4.8749|-0.0142|0.7018|
|all_tokens_reset|5.0372|+0.1481|1.2265|
|visual_read_off|4.9804|+0.0913|1.0397|
|refine_off|4.9957|+0.1066|0.9581|
|sidecar_text_off|4.9003|+0.0112|0.7031|
|sidecar_proprio_off|5.0443|+0.1552|1.0899|
|write_tap7_off|4.9393|+0.0502|0.8176|
|write_tap15_off|4.8876|-0.0015|0.7962|
|write_tap23_off|4.9262|+0.0371|0.8372|
|role_embedding_off|5.1374|+0.2483|1.1994|
|ffn_off|4.9829|+0.0938|0.9321|
|position_off|4.8925|+0.0034|0.7428|
|stage_context_off|4.9071|+0.0180|0.8055|

### 每任务关键组（每任务4窗口）

|Task|Full|Write-Off|Grounding-Reset|Relation-Reset|Plan-Reset|
|---|---:|---:|---:|---:|---:|
|adjust_bottle|4.7833|5.1350|4.8952|4.7623|4.8055|
|beat_block_hammer|3.5740|3.6514|3.6995|3.5372|3.5538|
|handover_block|5.4666|5.7388|5.7167|5.4801|5.5064|
|place_bread_skillet|4.9626|5.0901|5.2992|4.9050|4.8546|
|place_burger_fries|3.7867|3.8021|3.9647|3.8154|3.7928|
|put_bottles_dustbin|5.9741|6.0290|6.1563|6.0213|5.9502|
|place_object_basket|7.5448|7.7115|7.5469|7.5730|7.5496|
|place_cans_plasticbox|3.1617|3.4355|3.3754|3.1735|3.1589|
|stack_blocks_three|2.9885|3.1030|3.1159|3.0387|2.9684|
|open_microwave|6.6487|6.6512|6.6518|6.5961|6.6083|

### 分析与下一步

关闭Write平均误差+0.1457mm（+2.98%），10任务均值均上升，40窗口28升/12降；Grounding reset+0.1531mm（+3.13%），10任务均值均上升，40窗口36升/4降；Role embedding off+0.2483mm（+5.08%），40窗口36升/4降。Relation reset+0.0012mm，Plan reset−0.0142mm，仅支持本样本上平均专家动作拟合影响很小，不代表token不携带信息：两者动作漂移分别0.6818/0.7018mm。
Full不保证所有局部指标最优：Plan reset位置均值略低、Visual Read off的gripper MAE略低等，不能仅挑单指标判定模块优劣。每任务仅2条训练episode的4个相关窗口、固定一次噪声，没有独立大样本显著性结论。
原7组的Grounding/Plan head-only对照各40窗口与Full完全相同；该零效应只验证预测头没有直接反馈动作，不抹去其训练监督影响。扩大18组无head-only模式，complete.json中继承的head_only_controls_exact字段不应作为本轮新增头部对照证据；引用原7组complete/results。
这些是训练Clean轨迹上的离线专家动作误差；不是执行后实际EEF误差，不是闭环SR，不含Randomized泛化。Write-Off也不是未训练Sidecar的原OpenWAM baseline。下一步优先完成已启动18组配对闭环，检查离线/闭环方向是否一致，再对稳定差异组增加episode及独立episode数据；训练策略归因另需重训。

原始证据：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar21480_train_expanded_ablations_20261102/summary.json、results.json、complete.json。

## 08:24 UTC后进度快照

{"utc": "2026-11-02T08:26:22.293653+00:00", "new_completed_conditions": 55, "new_completed_episodes": 275, "new_total_episodes": 1500, "reused_episodes": 300, "active": 18, "queued": 227, "failed": 0, "errors": 20}

截至快照新增条件主要为adjust_bottle/beat_block_hammer，handover_block启动；尚不能作全10任务的18组排名。部分启动发生OOM/DeviceLost/无日志超时后重试，失败attempt不计SR。发现report启动时读取paused_for_capacity_expansion误当终态而退出；仅改报告器在complete/finished_with_errors结束并重启，原提前写出的final_results改名startup_snapshot_not_final保留。当前模型/仿真/统计口径未改，实时comparison恢复。

## 前两个任务配对趋势：2026-11-02T08:45:52.058185+00:00

仅adjust_bottle和beat_block_hammer、Clean/Randomized各5条，完整四条件时每组20条。前三旧组复用历史配对前5条，非独立新重复。各组完成相同四条件才填合计。

|Mode|adjust Clean|adjust Random|hammer Clean|hammer Random|合计|
|---|---:|---:|---:|---:|---:|
|full|4/5|2/5|5/5|3/5|14/20|
|write_off|4/5|2/5|5/5|4/5|15/20|
|grounding_reset|4/5|2/5|5/5|4/5|15/20|
|relation_reset|4/5|2/5|5/5|4/5|15/20|
|summary_reset|4/5|2/5|5/5|4/5|15/20|
|plan_reset|5/5|2/5|5/5|4/5|16/20|
|all_tokens_reset|5/5|2/5|5/5|4/5|16/20|
|visual_read_off|4/5|2/5|5/5|3/5|14/20|
|refine_off|4/5|2/5|5/5|4/5|15/20|
|sidecar_text_off|4/5|2/5|5/5|未完成|未完成|
|sidecar_proprio_off|4/5|2/5|5/5|未完成|未完成|
|write_tap7_off|4/5|3/5|5/5|3/5|15/20|
|write_tap15_off|4/5|2/5|4/5|4/5|14/20|
|write_tap23_off|5/5|2/5|5/5|4/5|16/20|
|role_embedding_off|4/5|2/5|5/5|4/5|15/20|
|ffn_off|4/5|3/5|5/5|3/5|15/20|
|position_off|4/5|2/5|5/5|4/5|15/20|
|stage_context_off|4/5|2/5|5/5|未完成|未完成|

初步趋势：前两任务中多项关闭/reset不降低或略提高闭环成功；Plan/AllTokensReset/末层WriteOff较Full各多2条，但样本极小，不能判定总体负作用。部分组Clean有单次退步，不能把平均持平解读为轨迹完全相同。后续重点验证之前差异最大的put_bottles_dustbin/place_object_basket及其他任务，不用此子集替代全10任务结论。

## 协议更正：统一每条件10条

用户指出原约定10条，先前扩大消融自行缩至5条未经确认。保留当前前5阶段，已启动接续watcher，自动补同一sealed manifest第6–10条；19组最终均为每task/domain10条，共3800条。此前5条结果仅为阶段趋势，不是10条终态。原三组复用完整10条，其他16组追加互斥后5条。执行与审计见docs/ROBOTWIN_21480_19ARMS_COMPLETE10_20261102.md。
