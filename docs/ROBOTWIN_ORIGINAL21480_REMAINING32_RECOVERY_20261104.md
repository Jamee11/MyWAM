# 原21480剩余32条件恢复：补齐完整100条

用户2026-11-04明确授权利用4090空闲资源，恢复此前74.83%模型的剩余32条件各70条；当前新42960完整100条继续运行。

## 已核对范围

原权重：`outputs/1030-warmup2k-to-full10epoch/checkpoint_step_21480.safetensors`，SHA256 `b141060b4ab33b0a2542effad452203b2ee5a5d287e5fd38ef092abe7326de7e`。这是完整Sidecar，不是Refine-Off、XA或新42960。

旧根：`outputs/robotwin_eval/full21480_supplement70_4090_20261031`；状态为用户暂停，68条件已经30+70完整100条，32条件未完成（22queued、10interrupted）。完整68为50Clean+18Randomized，5654/6800=83.1471%，只属于该子集。旧partial全部保留，不计正式结果。

本次新根：`outputs/robotwin_eval/full21480_remaining32_supp70_20261104`。只重新跑32个Randomized条件×70=2240条；旧68条目的完整100条结果与原始文件保留且复用。68与32任务键互斥，合计100条件；所有32的前30有效seed已经与原速测状态逐项一致，仍从原末有效seed+1接续官方专家筛选，不重测前30。

已对旧68条件的前30、后70共136份日志扫描OIDN Error/ErrorDeviceLost/VK_ERROR_DEVICE_LOST/非法CUDA内存，没有发现这些错误标记；正式最终报告仍重新核验原生日志、exit0、条数和100唯一seed。不能把旧68的83.15%当全套最终平均。

## 部署与共存

复用已验证的T5拆分适配：首先主模型GPU1、T5 GPU4，16个独立policy/action-buffer端口41500–41515，4090映射42500–42515。Body allocator上限18GiB，原精度与计算不变。已有RAM权重/配套config/stats/tokenizer，固定两个原生输入必须与旧Full canonical动作逐元素一致且重复一致，之后才启动仿真。

释放资源只涉及已结束诊断的Sidecar-Off服务PID3924278，以及新42960已经不再使用的replica4服务。后者退休前检查：当前42960补70非complete条件K<=15，而该服务端口位于原first-free端口顺序index>=25；未来派发最多K−1旧活跃，会在前K个中找到空端口，所以不会再访问该池；当前live端口也无交集。当前42960其他两份服务和在跑仿真继续，既有端口顺序/队列范围不改。证据reused_idle_patch_service.json、retired_diagnostic_model.json。

42960完整100条及final_independent_audit通过后，独立`add_after_patch_complete.py`监督器再退休其余两个已结束服务，新增原Full第二份（主GPU3/T5 GPU6，41520–41535→42520–42535）。新调度器每次派发读取available_ports.json，第二份验收通过后动态扩至32会话，现有仿真不重置。若原32条件在此前已完成，监督器不再部署多余模型。

## 仿真与结果协议

复用近期70条入口与continuation_adapter：Eager、cache .99/max3、10denoise、sync32、384×320、完整Sidecar、无视频；官方RT/专家seed/成功判据/预算不变。32条件独立新attempt目录，旧10条件partial不覆写，不把其残缺结果拼入70。

动态按4090实际显存派发，6144MiB启动占位180秒，之后看真实余量；不以低显存强杀正常进程。新入口含已知renderer错误与Robot缺left/right_planner循环的恢复，错误attempt不计SR。16/32是模型独立会话容量，实际仿真并发取决于GPU余量，不声称必须启动32路。

30+70分段候选指令池/RNG与一次native100并非逐位等价；沿用既有接续协议并记录边界。最终report合并旧68与新32，逐条件100个不重复seed，完整100条件/10000条才写final_results.json和final_complete100_independent_audit.json。

## 检查、状态与后续

- recovery_selection_checks.json：68/32集合互斥且覆盖100条件、32新增2240、旧partial保留；68旧日志渲染扫描通过。
- contract_checks.json：接续覆盖/30+70不重叠和实际原adapter测试通过；源前30共有100条件有效。
- model_processes.json：首份原Full PID4113133，GPU1/T5 GPU4；activation_process.json为加载与自动启动监督器，expansion_watcher_process.json为完成42960后的第二份监督器。
- launch_status.json记录真实加载/评测启动阶段；当前初始记录是loading_split_model，不把加载当已推进rollout。
- 新state.json只含剩32个70条作业；comparison.json则汇总旧68+新完成条件的100条结果。原旧根状态/结果不覆盖。

下一步：验证第一份canonical通过与真实4090首步，然后按空余派发；待全部完成后汇报原Full完整100条与新42960完整100条的相同任务比较。此前混合条数估计74.94%不替代新的实际10000条成绩。


## 04:13 UTC实际启动通过

首份GPU1/T5 GPU4服务通过两个原生fixture各两次重复、与旧Full canonical动作逐元素一致，完整Sidecar/cache-on/eager保持。16端口全部健康，独立SSH反向隧道正常；远端scheduler1867001、report1867002已启动，9路70条补测进程已派发（source当前42960同时14路，共23路），尚在仿真初始化。dump_bin_bigbin/Randomized原30最后seed100087，接续起点100088。controller无异常，剩余23条件会按实际显存自动补位。第二份监督器4113703正在等待Patch100完成，现有32条件不等待它即可推进。

## 07:32 UTC进度

本次32条件已完成9个，16个运行、7个queued；加上保留68条件，共77/100条件已各100条，6278/7700=81.5325%。Clean已全50条件，4439/5000=88.78%；Randomized完成27条件，1839/2700=68.1111%。对应原30条同77条件81.5152%，扩样+0.0173pp；这77条件含全部Clean与部分Randomized，不能当全套最终均值或直接与新42960的95条件均值比较。

07:22采样时活动16个attempt另推进530条，尚待完整70条验收；controller/report均live，当前仅首份模型16会话，第二份监督器仍等Patch100完整审计后接续。快照outputs/analysis/aloha_two_supplements_status_20261104T0723Z.json。


2026-11-04 07:48 UTC ETA更新：完整100条条件79/100，当前16路运行/5queued；按剩余任务长度和当前attempt推进估计还需3–5小时，北京时间约2026-11-04 18:48至2026-11-04 20:48。长rollout、尾轮并发下降和近期初始化重试已纳入判断；新增失败重试仍可能延后，不按完成条件百分比线性推算。逐任务计数见本run eta_latest.json。
