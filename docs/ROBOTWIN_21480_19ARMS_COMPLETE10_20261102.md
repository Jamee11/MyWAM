# 原21480消融：统一恢复每任务/每域10条

2026-11-02 UTC。用户指出原始三组协议是每task/domain/arm10条。扩大18组时自行改为5条筛查未经事先确认；本文件更正最终目标为19组均10条。此前所有前5条结果保留并标为阶段性数据，不冒充10条终态，也不覆盖原三组完整10条。

## 目标与数据

19组=原18组+Sidecar All-Off。10任务×Clean/Randomized×每条件10条×19组=3800条比较样本。相同原sealed manifest全部10个专家有效seed及指令，推理seed42与既有推理/RT/成功/动作预算/视频关闭设置不变。

- 原三组Full/Write-Off/Relation-Reset原来就已完成10条：直接复用原始完整600条，无新增重复。
- 其他16组的前5条（共1600条）继续当前两个独立队列，保留全部有效结果。
- 新追加同manifest第6–10条（逻辑episode index5:10）：16组共1600条新评估；原三组后5条直接复用历史300条。
- 最终共3800条，其中新增3200条，复用原三组600条。新后半轮共1900条记录，其中真正新跑1600条。

## 执行与会话隔离

独立目录`outputs/robotwin_eval/sidecar21480_19arms_complete10_20261102`。CPU watcher `resume_after_first5.py`已挂接，等18组前5队列及All-Off前5队列全部完成并逐native exit0/manifest hash/seed/指令审计后，才复用空闲的原24+4端口启动后5队列；不重启现有仿真、不改变在途结果、不与前半轮共享同一个活动会话。

新增runner使用现有官方manifest wrapper的`--total-episodes 10 --episode-start 5 --num-episodes 5`。每次native分片test_num仍为5，最终逻辑集合为固定10条；绝不把start_seed简单+5，避免专家筛选造成seed跳号。20份manifest的前后半集合互斥、恰好覆盖原10条，已通过`selection_checks.json`。All-Off使用其独立专用端口，不发送18组干预模式；其他模式仍走已验收请求级隔离hook。

后5汇报`comparison.json`仅为后半轮；最终`merge_report.py`重新读取实际原生结果，将前5+后5逐episode核对至原manifest全部10条；只把完整10条的条件加入`merged10_results.json`，并按相同条件的Full比较。380个条件/3800条全部通过后才产生`final_independent_audit.json`。

## 当前状态与下一步

目前是已配置并启动CPU接续等待，不是后5仿真已经启动；实时查看`queue_status.json`和`queue_process.json`。原第一轮两队列继续跑。此前4–7小时等估时是针对前5阶段，不应作为本次完整10条的完成承诺；按新总量与实时吞吐重新估算。

原18组说明和All-Off说明中5条安排均为第一阶段历史记录，最终协议以本更正为准。后续按10条终态报告，阶段结果明确注明分母。

## 2026-11-02 10:05 UTC：渲染完整性检查与复测更正

检查发现“进程exit0+seed/指令匹配”不足以确保图像有效。前5阶段205个已记录完成条件中，42个日志出现明确`OIDN Error:`或`ErrorDeviceLost`，包括旧三组的adjust_bottle/Randomized/RelationReset、handover_block/Randomized/Full、place_burger_fries/Randomized/WriteOff。渲染错误可能发生于部分帧/初始化阶段，不能断言每条都已损坏，但在确认前一律隔离，不纳入可靠闭环均值。原始结果/日志保留，不能据旧表宣称已经完成稳健因果结论。训练数据离线720次诊断不经过本仿真渲染路径，不受这项隔离影响。

新增独立render_guard持续审计，针对当前明确渲染失败的自有worker按PID/start_ticks/runner路径校验后TERM，并在未退出时升级KILL，让原调度器及时重排，避免等20分钟。其他仿真、模型、训练不停止。report器仅重启CPU报告进程，剔除隔离条件，并在Full条件隔离时不做该条件配对。

接续watcher只重启CPU本身：前5轮结束后再次全量扫描日志。对隔离条件，后续不再拼接可疑前5，而是同一原manifest完整重跑10条；其他条件仍原前5+互斥后5。旧三组中被隔离的条件也完整重测10，避免把可能受影响的历史10条直接复用。后续调度验收新增日志无明确OIDN/Vulkan错误门槛；最终merge仅包含渲染日志干净且native exit0、原seed/指令/hash匹配的10条。最终目标仍380条件/3800有效条，实际尝试数会因重试而增加。

20份manifest×19组的范围回归通过：隔离条件0:10，其余0:5+5:10，全部恰好覆盖原10条。当前42隔离条件（其中3旧三组）若此刻开始后半轮，需要新跑1825条；最终数随前5完成后的审计更新。文件：quarantined_conditions.json、render_guard_actions.json、render_repair_selection_checks.json、merge_report.py。此前“新增3200/复用600”数量需扣除隔离历史条件并加入重测，不再作为固定有效数据数量。

## 11:47 UTC阶段效果：剔除渲染日志异常后的配对比较

仍为每条件前5条，完整10条未完成。各行仅与相同任务/domain/seed/语言的Full比较；各行任务子集不同，不能按干预SR跨行排序。42个明确渲染错误日志条件及对应配对Full条件排除，等待完整10条重测。

|Mode|配对条数|Full SR|干预SR|差值pp|
|---|---:|---:|---:|---:|
|full|95|76.84%|76.84%|+0.00|
|write_off|90|77.78%|81.11%|+3.33|
|grounding_reset|50|80.00%|82.00%|+2.00|
|relation_reset|90|78.89%|86.67%|+7.78|
|summary_reset|50|78.00%|80.00%|+2.00|
|plan_reset|50|80.00%|84.00%|+4.00|
|all_tokens_reset|45|77.78%|84.44%|+6.67|
|visual_read_off|45|77.78%|75.56%|-2.22|
|refine_off|45|73.33%|80.00%|+6.67|
|sidecar_text_off|45|80.00%|82.22%|+2.22|
|sidecar_proprio_off|50|80.00%|80.00%|+0.00|
|write_tap7_off|45|75.56%|80.00%|+4.44|
|write_tap15_off|35|91.43%|88.57%|-2.86|
|write_tap23_off|30|90.00%|86.67%|-3.33|
|role_embedding_off|35|74.29%|68.57%|-5.71|
|ffn_off|35|74.29%|74.29%|+0.00|
|position_off|35|80.00%|74.29%|-5.71|
|stage_context_off|40|77.50%|75.00%|-2.50|
|sidecar_off|40|82.50%|80.00%|-2.50|

主要趋势：Relation reset和Write off仍在各自匹配子集上略高于Full，差异主要在Randomized；部分分支如位置/角色embedding和视觉Read关闭会降低对应子集SR，不能认为所有Sidecar组件都带来负作用。All-Off与Full同40条中32对33成功（80%对82.5%）；同40条Write-Off也是33/40。All-Off/Write-Off两固定fixture完全相同不外推任意状态或完整rollout，闭环三条不同结局涉及beat_block_hammer Random、place_bread_skillet Clean、place_burger_fries Clean，需结合轨迹/新样本排查，不能把仅1条净差当稳定退化证据。
每条件5条、各组任务子集不均、待重测42条件意味着结果仍是阶段趋势。后续以render-clean完整10条统一结果为准，离线训练拟合收益与闭环泛化贡献分开解释。

## 12:34 UTC进度与All-Off缓冲区故障接续

{"utc": "2026-11-02T12:35:24.791545+00:00", "rows": [{"run": "sidecar21480_expanded_ablations_20261102", "complete_conditions": 255, "total_conditions": 360, "render_clean_conditions": 213, "active": 15, "terminal": 0}, {"run": "sidecar21480_all_off_20261102", "complete_conditions": 13, "total_conditions": 20, "render_clean_conditions": 13, "active": 4, "terminal": 1}], "recorded_episodes": 1340, "render_clean_episodes": 1130, "quarantined_conditions": 42, "active": 19, "tail_started": false}

All-Off/place_burger_fries/Randomized因cannot create buffer和DeviceLost等渲染基础设施错误到初始20attempt上限，不作为模型失败。新增watcher等当前队列正常worker全部退出再只重排终态infra失败，保留已完成条件与所有尝试；未来重试profile用8192MiB启动余量、4线程，错误日志验收门槛，保留原seed/指令/推理协议。不中断当前4路与其他18组。待当前队列drain后再启动，不声称已救活该条件。

## 12:36 UTC完整ETA与公平候选比较

{"utc": "2026-11-02T12:38:14.705823+00:00", "common_conditions": 7, "episodes_per_mode": 35, "successes": {"full": 28, "write_off": 29, "relation_reset": 30, "plan_reset": 30, "all_tokens_reset": 31, "sidecar_off": 27}, "rows": [{"task": "adjust_bottle", "domain": "demo_clean", "successes": {"full": 4, "write_off": 4, "relation_reset": 4, "plan_reset": 5, "all_tokens_reset": 5, "sidecar_off": 4}}, {"task": "beat_block_hammer", "domain": "demo_randomized", "successes": {"full": 3, "write_off": 4, "relation_reset": 4, "plan_reset": 4, "all_tokens_reset": 4, "sidecar_off": 3}}, {"task": "place_bread_skillet", "domain": "demo_clean", "successes": {"full": 5, "write_off": 5, "relation_reset": 5, "plan_reset": 5, "all_tokens_reset": 4, "sidecar_off": 4}}, {"task": "place_bread_skillet", "domain": "demo_randomized", "successes": {"full": 4, "write_off": 4, "relation_reset": 4, "plan_reset": 4, "all_tokens_reset": 4, "sidecar_off": 4}}, {"task": "place_burger_fries", "domain": "demo_clean", "successes": {"full": 5, "write_off": 4, "relation_reset": 5, "plan_reset": 5, "all_tokens_reset": 5, "sidecar_off": 5}}, {"task": "put_bottles_dustbin", "domain": "demo_clean", "successes": {"full": 5, "write_off": 5, "relation_reset": 5, "plan_reset": 4, "all_tokens_reset": 5, "sidecar_off": 5}}, {"task": "put_bottles_dustbin", "domain": "demo_randomized", "successes": {"full": 2, "write_off": 3, "relation_reset": 3, "plan_reset": 3, "all_tokens_reset": 4, "sidecar_off": 2}}], "scope": "6 candidates same render-clean conditions, first5 only; not19-arm whole-suite ranking"}

近期有效前5结果吞吐约125–155条/h（10:05为815无明确渲染错误，12:35为1130；11:47为1005）。完整3800目标已知1130前5+285未隔离历史后5≈1415条，剩约2385条；按近期实测暂估16–22小时，含渲染重试波动。此前4–7小时对完整10条偏乐观，不能延用。实查4090当前18个runner；显存free多数0.09–1.9GiB，GPU7约3.7GiB，不具备再加一个完整典型5–6GiB峰值仿真的余量。现有进程继续运行，空出后自动补位；不以显存占满代替有效吞吐。

## 2026-11-02T13:43:53.329004+00:00阶段进展

{"utc": "2026-11-02T13:43:53.329004+00:00", "phase": "first5; full10 continuation pending", "recorded_conditions": 304, "quarantined_conditions": 42, "common6_episodes": 40, "common6_successes": {"full": 33, "write_off": 33, "relation_reset": 35, "plan_reset": 35, "all_tokens_reset": 36, "sidecar_off": 32}, "common6_conditions": [{"task": "adjust_bottle", "domain": "demo_clean", "successes": {"full": 4, "write_off": 4, "relation_reset": 4, "plan_reset": 5, "all_tokens_reset": 5, "sidecar_off": 4}}, {"task": "beat_block_hammer", "domain": "demo_randomized", "successes": {"full": 3, "write_off": 4, "relation_reset": 4, "plan_reset": 4, "all_tokens_reset": 4, "sidecar_off": 3}}, {"task": "place_bread_skillet", "domain": "demo_clean", "successes": {"full": 5, "write_off": 5, "relation_reset": 5, "plan_reset": 5, "all_tokens_reset": 4, "sidecar_off": 4}}, {"task": "place_bread_skillet", "domain": "demo_randomized", "successes": {"full": 4, "write_off": 4, "relation_reset": 4, "plan_reset": 4, "all_tokens_reset": 4, "sidecar_off": 4}}, {"task": "place_burger_fries", "domain": "demo_clean", "successes": {"full": 5, "write_off": 4, "relation_reset": 5, "plan_reset": 5, "all_tokens_reset": 5, "sidecar_off": 5}}, {"task": "place_object_basket", "domain": "demo_clean", "successes": {"full": 5, "write_off": 4, "relation_reset": 5, "plan_reset": 5, "all_tokens_reset": 5, "sidecar_off": 5}}, {"task": "put_bottles_dustbin", "domain": "demo_clean", "successes": {"full": 5, "write_off": 5, "relation_reset": 5, "plan_reset": 4, "all_tokens_reset": 5, "sidecar_off": 5}}, {"task": "put_bottles_dustbin", "domain": "demo_randomized", "successes": {"full": 2, "write_off": 3, "relation_reset": 3, "plan_reset": 3, "all_tokens_reset": 4, "sidecar_off": 2}}]}

共同40条中AllTokensReset36/40，Relation/Plan35/40，Full/Write33/40，AllOff32/40；仅阶段趋势，不是19组最终统一10条排名。42渲染异常条件仍排除，后续完整同seed重测；前5轮收尾及All-Off单独基础设施重试等待继续。

## 2026-11-02T14:40:57.770565+00:00剩余时间与终态基础设施失败接续

{"utc": "2026-11-02T14:40:57.770565+00:00", "robotwin_first5_recorded_episodes": 1675, "robotwin_first5_clean_episodes": 1465, "robotwin_full_target": 3800, "robotwin_estimated_hours_remaining": [14, 18], "robotwin_recent_clean_episodes_per_hour": 155, "maniskill_completed": 166, "maniskill_target": 1600, "maniskill_estimated_hours_remaining": [12, 16], "maniskill_recent_episodes_per_hour": 120}

All-Off已有2条件在初始20attempt后因渲染错误终态失败。取消另一套“补前5失败再重试”CPU watcher，交由同一最终10条修复队列处理；现有正常模型和worker不停止。首轮收尾条件允许complete或已确认renderer-failed，后者无可用结果，最终必须原manifest完整重跑10条，不伪造前5。render_integrity纳入终态渲染失败，合并只读取最终无错误的10条。最终新一波准入8192MiB，避免延续已观察到的低余量反复渲染失败。3800条身份/范围回归通过。最近1小时155、2小时325条有效新增，完整估计14–18小时；第一轮剩余预计约1.5–3小时，非完整全部结束时间。

## 2026-11-02 17:37 UTC接续修复并真实启动

前5阶段最后一个常规条件16:44完成，All-Off3个条件重试耗尽，其中place_object_basket/Randomized末次-15为无进度超时、没有错误关键词，被过窄的failed_renderer分类漏掉，造成约53分钟接续空等。已改为所有没有有效结果的终态failed统一同seed完整重测10条；正常complete且0成功的策略条件不受影响。无需错误文本分类，不伪造结果，380条件完整性判定通过。45修复条件=42日志标记完成条件+3缺失结果，后半需要真实新增1840条，57个历史后5条件直接复用。17:37:40尾轮已启动，模型和ManiSkill均未重启。记录queue_status/state/first5_independent_audit/render_repair_plan/merge_report。
