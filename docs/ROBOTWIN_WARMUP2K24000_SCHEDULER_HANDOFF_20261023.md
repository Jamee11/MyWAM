# 24k评估调度修复与GPU4/5核查


## 2026-10-23T09:17:30.822608+00:00：GPU4补路、修复错误看门狗、无损接管进行中评测

用户询问GPU5/4空余及总耗时。GPU5完成place_shoe Clean30条后启动open_microwave Randomized，暂时2.2GiB占用时空闲6GiB，实际随规划/rollout上涨到约5.5GiB、余量约2.8GiB，不能再加。GPU4的hammer Clean完成释放后实际补到3仿真，整卡现约23.8GiB/24.2GiB，余量仅约0.4GiB；全局并行已10路，不再强塞该卡。

发现并已告知用户：原dispatch.py的partial解析未剥离ANSI颜色，导致启动信用未释放且last_progress未更新，45min误将6项仍推进的条件TERM（不是模型失败/OOM）。原partial日志保留，6项由原控制器启动新attempt重测，正式汇总不混入旧partial。看门狗错误发生前3项Clean完整30结果有效。

新增elastic_v2.py，ANSI剥离、动作步/episode双进度心跳、正确释放信用。控制切换采用保留native进程和真实exit状态：封存handoff_snapshot，原snapshot写入路径设排空标记以阻止旧线程继续派发；旧控制器暂停，9个在跑native子进程继续、不重启不重置模型；/proc zombie wait status由独立collector读取，仍要求真实exit0+原validate_result完整30条后写run.json并移交V2，非猜测返回码。真实fork子进程exit7与waitpid对照通过。全部旧native结束后才退出旧控制器释放slot锁；新控制器独立管理空闲slot/新任务。旧watchdog已无法再误杀，collector仅45min无日志动作进展才判停滞。

标准仿真/seed筛选/成功判据/预算/视频关闭均未改变；只有orchestration修正。原state备份state.before_elastic_v2.json，canonical state.json链接elastic_v2/state.json；live watcher保持读取，另统计physical含重测rollout与最新attempt覆盖，避免重跑后把总工作量变少或重复计入最终SR。paused_legacy_parent.json、legacy_exit_collection.json、adopted_native_exit.json、native_exit_status_collection_test.json记录身份和真实退出证据。

09:14:03 UTC快照：自08:14:38运行3565s（59m25s），实际完成229条终态（含重测/中止前终态）；当前最新attempt覆盖128条，其中3项完整90条；10路活跃，历史调度误停6次、未归为模型失败。后续继续30×100趋势任务，需待完整30结果再做官方对照；初期10–16h ETA需考虑这次调度误停损耗重估。
