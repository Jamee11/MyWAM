# Aloha补70条速度与重跑诊断

快照：2026-11-04T08:26:59.726141+00:00。本轮只读分析并保存证据，未修改评测逻辑、模型、训练或在跑部署。

|实验|已各100条条件|已完成条件均值|运行/排队|无效attempt|无效attempt内累计已记录rollout|无效worker小时|
|---|---:|---:|---:|---:|---:|---:|
|PatchWAM42960|96/100|76.75%|4/0|285|4108|164.96|
|OriginalFull21480|86/100|80.05%|14/0|54|535|21.13|

各实验完成的条件集合不同，不能直接横比当前均值。无效attempt中的rollout数包含同seed反复执行，只表示累计重复工作量，不能理解为丢失了同样数量的唯一seed。worker小时为并行进程耗时相加，不是墙钟时间。

## 实测主要耗时

|样本|动作规划/物理执行|模型RPC（含网络/排队）|获取观察|PNG编码|
|---|---:|---:|---:|---:|
|stack_bowls_three/demo_randomized|78.7%|5.7%|4.5%|0.30%|
|put_bottles_dustbin/demo_randomized|74.2%|5.6%|1.9%|0.30%|
|put_object_cabinet/demo_clean|70.8%|5.2%|2.2%|0.23%|
|place_mouse_pad/demo_randomized|79.9%|7.9%|6.2%|0.27%|
|handover_mic/demo_randomized|77.0%|7.1%|3.8%|0.25%|

仅以上五个当前attempt的wall计时，剩余包含场景/专家筛seed/初始化等；不重复加总嵌套的render_update/client_step。CPU容器有71核affinity、quota=-1、throttled=0；当前宿主可用内存约203GiB，容器memory failcnt=0，不支持把问题归为CPU配额或容器RAM上限。

## 明确的额外损耗

Original21480 place_a2b_left/Randomized attempt1已记录68/70，08:07 UTC出现OIDN Error: out of memory，被现有调度判为renderer error并整组重跑；place_object_stand attempt1记录17/70后08:12同样失败。日志与run.json保留在各任务原attempt目录。Patch无效原因：{"renderer error": 179, "20min no log progress": 54, "poisoned planner initialization": 3, "client exited 1": 49}；Full无效原因：{"renderer error": 42, "20min no log progress": 8, "client exited 1": 3, "poisoned planner initialization": 1}。

这是当前“慢”的实质原因之一，此前只用尾轮长任务解释不够完整。OIDN内存错误为实证；具体GPU峰值/资源释放及两调度器独立6144MiB预留的贡献尚未隔离，不能把所有失败都归为单一并发原因。

## 当前尾轮与处理边界

两组都没有queued条件，所有未完成条件已在跑；增加模型不会增加独立任务数。Patch余下4条件为put_bottles_dustbin/Randomized、put_object_cabinet两域、stack_bowls_three/Randomized，预算分别1700/700/1200，单路常约12–20条/h，低成功率时会执行完整动作预算。

当前保留正常在跑任务和完整原生结果校验，不回收污染attempt的partial计分，也不降低专家/RT/动作预算。后续优先降低渲染OOM和整组重跑损耗；episode级恢复只有在保存并验证seed/指令/RNG与原生序列一致后才能替换现有方案，本轮未实现或宣称该修复。完整证据：outputs/analysis/aloha_supplement_slowdown_20261104/audit.json。
