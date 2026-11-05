# 原21480剩余32条件：干预实验结束后自动恢复

用户2026-11-02明确授权执行顺序：XA完整速测 → Sidecar干预评估 → 原最佳21480剩余32条件补70 → 完整100条件汇总。

XA已经100条件/3000条72.60%并通过独立audit。干预实验为10任务×双域×Full/Write-Off/Relation-Reset×10条，600条配对清单；只有80个manifest/策略job全部complete，并重新检查60份策略结果对应seed/语言/hash/成功计数/真实退出码通过，才切换补测。中途基础设施错误不被当作完成；若需要人工恢复则队列保持等待。

后台接续器：`outputs/robotwin_eval/full21480_supplement70_4090_20261031/resume_after_interventions/queue.py`；状态`status.json`，进程身份`process.json`，日志`queue.log`。当前仅CPU等待，不停止或修改在途干预任务。

原补测68个完整条件已封存：`preserved68_state.json`、`preserved68_results.json`。恢复前逐条检查该68条件与封存相同，保留原结果，剩余32条件各新跑完整70条（新增2240条）；此前未完成partial不拼接。全部从各自原30条最后有效seed+1继续专家筛选，固定续测起点不变，原30条有效seed不重复。当前原始30全任务3000条与68完整补70共4760条已验收，合计7760条完整片段，尚需2240条。

使用原最佳21480（SHA b141060b4ab33b0a2542effad452203b2ee5a5d287e5fd38ef092abe7326de7e）的正常Full路径；不从XA权重/干预数据继续、不把干预结果混入100条。保留原eager/cache .99/max3/10 denoise/horizon32/no-video/官方专家筛seed、RT、EEF、成功判据与动作预算。部署两份Full双卡放置模型，各6独立session，固定fixture对原21480历史canonical精确一致才放行。干预结束后仅退休已完成自有模型，动态等四卡放置容量；不清理未知作业。

调度复用XA积极并发：12会话、每4090最多4路、6GiB准入/256MiB低水位，容量失败暂退120秒再试、每条件最多20attempt，保留日志和部分结果。原68完整条件不会重新测试。

最终独立验收每条件原30+新70的原生结果/实际退出码、checkpoint SHA及100个唯一有序有效seed，100条件/10000条。自动生成`final_complete100_independent_audit.json`、`final_results.json`、`final_complete100_results.csv`及`docs/ROBOTWIN_FULL21480_COMPLETE100_FINAL_20261102.md`，报告Clean/Randomized/综合与官方、原30条的差值。

口径边界保持：30+70分段候选指令池与一次性native100不完全相同；只声称无重复有效seed的每条件100条扩展评估。
