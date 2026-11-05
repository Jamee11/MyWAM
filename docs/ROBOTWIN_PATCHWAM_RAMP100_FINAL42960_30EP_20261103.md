# PatchWAM ramp100最终42960：完整50任务速测

2026-11-03用户授权使用最终权重：
`outputs/sidecar/aloha50_patchwam_c2r_ramp100_10epochs_20261027T144939Z/F_patchwam_ramp100/runs/2026-10-27_14-57-15/checkpoint_step_42960.safetensors`。

目标50tasks×Clean/Randomized×30=3000。保留完整Sidecar，不采用Refine-Off；复用此前cache .99/max3、Eager、10denoise、sync32、384×320、原生expert筛seed/RT/成功判据/动作预算、无视频、现有chunk通信及中间camera采集优化。历史native30语言RNG未冻结，因此旧权重比较是相同协议，非逐条同指令严格配对。当前Refine-Off收尾不停止。

独立队列：`outputs/robotwin_eval/patchwam_ramp100_final42960_all50_30ep_20261103`。先将最终权重与同目录config/stats/tokenizer/Sidecar contract放入独立RAM目录，逐字节SHA核对；原权重及旧评估不覆盖。后台launch.py正在执行，不把staging标作评估启动成功。

复用此前已验证的T5拆分放置：首份主模型GPU2/T5 GPU4；XA修复任务已完成，检查专用XA PID/start_ticks和入口后仅退休该服务，腾出主模型GPU0/T5 GPU3启动第二份。固定输入动作repeat检查通过后才开放会话，完整权重加载无量化/裁剪。旧单卡vs split固定输入maxabs0证据仅覆盖相应fixture；新权重仍须自身数值检查，不外推整条轨迹逐bit保证。

计划两份×12端口，4090映射42100–42111/42120–42131。复用已在Refine-Off运行的native30调度脚本，按实时显存空余6144MiB起步，不强行结束正常旧仿真；最多24独立模型会话，25调度线程不会让多个sim共享live action deque。渲染错误attempt拒绝计分并原任务重排，原生完整30、exit0、无渲染错误才纳入SR。已有启动数与并发量以launch_status/state实际文件为准，不把配置上限当真实在跑数。

结果看comparison.json和state.json。下一步确认首份/第二份GPU固定输入验收与真实仿真启动，随后汇报已完成相同任务条件和原21480/早期PatchWAM成绩；最终必须100条件3000条完整。


07:50 UTC恢复：初始两份服务启动均失败（replica0首次推理CUBLAS_STATUS_ALLOC_FAILED，replica1加载阶段显存OOM，未启动仿真/无SR）；显存期间有其他进程变化，不结束他人任务。保留失败日志，改只加载replica2，主模型GPU3/T5 GPU4，主模型allocator上限由16放至18GiB（物理卡此时22GiB空余），无模型数学/精度改变。新16端口41140–41155映射42140–42155；后台验收通过才启动队列。此前复制/双端SHA已通过，原Refine-Off收尾未动。


## Refine-Off完成后自动扩容（用户2026-11-03明确授权）

追加监督器`expand_after_refine.py`（记录expansion_watcher_process.json），等待Refine-Off所有100条件及同队列XA修复均为有效complete，随后只退休确切PID/start_ticks/入口匹配的旧Refine服务，不停仍在跑的仿真。主模型/T5优先2/6，再1/4；保留当前PatchWAM副本3/4。新增每份12独立会话，总目标3份模型/40会话；若一份加载失败，仅用成功新增的一份激活，总28会话。使用同一新42960 RAM权重/SHA、deploy配置、完整Sidecar和canonical固定输入动作验收；未引入Refine-Off、quantization或不同cache。

实际仿真数由4090实时可用显存决定，6144MiB启动预留、180秒后释放预留；不因显存低而结束正常worker，错误attempt仍按原规则原任务重试。40会话不是声称40路同时可容纳，CPU线程40也不是实际并发数。

调度器现有端口表在启动时读取，单改JSON不能令旧进程接入新模型，因此增加独立`schedule_expanded.py`和一次性交接入口`handoff_expand_remote.py`。交接只SIGSTOP旧控制器（不SIGSTOP或kill仿真），保留其子进程原PID/场景/episode。新控制器继承当前queued/complete/running，忙端口预先登记；已运行仿真由新控制器只监测，其自然结束后读取旧父进程保留的/proc zombie真实exit_code，再用相同native30/log/renderer验收。全部继承子进程退出状态已消费后才退休旧控制器。新任务继续用原Popen执行，已完成条件不重跑，未完成也不清空进度。交接状态快照、PID身份和结果日志全部保留；旧report进程继续读取同一state出报告。

必要CPU验收`check_expansion_handoff.py`已通过：两个真实dummy子进程退出0/7均恢复正确；start_ticks不符拒绝；直接提取实际端口表达式验证1/2新增副本均覆盖原16端口且没有重复。没有将dummy测试当作真实GPU/调度接管通过；真实验证需待Refine结束和新增服务ready。状态看expansion_status.json，日志expand_after_refine.log、schedule_expanded.log。


## 09:26扩容已触发，09:33修复接管重试分支

Refine-Off09:18有效完成后，新增replica3主GPU2/T5 GPU6、replica4主GPU1/T5 GPU4均通过同checkpoint canonical两fixture动作检查；旧Refine服务按PID身份已退休。09:26新队列总3模型、40独立会话；继承原16sim无重置，并扩大并发。

09:30检查发现抽取monitor函数时遗漏局部attempt变量：正常成功分支不受影响，failed attempt更新queued时抛NameError，hanging_mug和open_microwave Randomized两条件有错误终态pending retry。09:32已修复仅此变量，从各job attempts恢复当前尝试数，实际函数CPU失败重排/20次耗尽/成功三分支通过；旧错误记录保留。交接repaired controller4153966（见expanded_scheduler_repaired_process.json），原26个活动sim继续原PID执行；两条明确无效且native run.json不完整的条件重入队，hanging_mug09:33已启动新attempt，未把基础设施失败计SR。控制器只读原父进程保留的真实exitcode，继承worker自然结束后再退旧控制器，原生验收不降级。

09:33状态：14/100条件322/420=76.6667%，Clean241/270=89.2593%，Randomized81/150=54%；当前同任务旧Full306/420=72.8571%，早期子集不可外推全套。27路活动、59 queued、0终态failed。40为会话容量，实际并发仍由显存/重试决定。


2026-11-03 12:24 UTC：62/100条件1860/3000条有效完成，1451成功=78.0108%；Clean897/1020=87.9412%，Randomized554/840=65.9524%。同条件原Full1426/1860=76.6667%，差+1.3441pp；Clean旧910/1020=89.2157%，-1.2745pp，Randomized旧516/840=61.4286%，+4.5238pp。当前3模型、26路活动、12 queued、0终态failed；历史2条NameError记录保留，修复后无新增controller异常/terminal pending retry。11:22:48所有26个继承仿真已完成真实exitcode消费，旧控制器退休（adopted_workers_repaired_finished.json），新controller4153966仍live。渲染/分配失败attempt继续重试，不纳入SR；当前仍是部分任务子集，不能与全套75.70% Refine-Off直接排名。


2026-11-03 13:43 UTC：88/100条件2640/3000条有效完成，2020成功76.5152%；Clean1190/1350=88.1481%，Randomized830/1290=64.3411%。相同条件原Full1987/2640=75.2652%，综合+1.25pp；Clean旧89.4074%，-1.2593pp，Randomized旧60.4651%，+3.8760pp。剩余12条件均运行，无queued/terminal failed，controller4153966 live，历史NameError记录保留且无新增；put_object_cabinet Randomized日志约6.5分钟未更新，仍在既有20分钟无进度监控内，暂不把正常规划等待当终态或重置其余worker。多卡显存已释放是收尾条件减少，不能由余量推断有未调度任务。


13:45 UTC进一步检查确认3个“日志更新但无rollout”的worker不健康：hanging_mug/Randomized、place_empty_cup/Clean、press_stapler/Randomized在Robot.reset因缺left/right_planner反复setup错误，native外层catch继续写日志，原20分钟log-growth监控不能识别。保留100MB级失败日志，仅对当前run且PID/start_ticks/entry匹配的反复缺planner错误attempt SIGTERM，由原调度重排；追加单个recover_planner_initialization.py轻量监视最近64KB，三次以上明确planner AttributeError才介入，不改官方筛seed/成功/动作逻辑。13:45已真实重启三路，PIDs597237/597233/597239，GPU3/4/2；无效旧attempt未纳入SR。根因初始Curobo planner创建失败尚未充分定位，重启后真实推进需继续核实，不能把新PID当完整恢复。


## 最终速测结果（17:12 UTC完成，17:43独立复核）

100条件3000条2286成功=76.20%；Clean1318/1500=87.8667%，Randomized968/1500=64.5333%。相对原Full21480：综合+1.3667pp、Clean-0.8pp、Randomized+3.5333pp；相对Refine-Off最终75.70%综合+0.50pp。原始最终native exit0/完整30/30unique/log无渲染错误100条件复核通过，证据final_native_render_audit.json。分段70扩展正在独立队列进行，不覆盖本3000成绩。
