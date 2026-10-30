# 当前20k：L20与4090仿真对照

2026-10-24T10:23:11.074269+00:00；complete。4条件各30条，不存视频。

有效完整条件4/4，观察终态120/120。只比较同条件完整30条，partial仅作进度。

|Task|Mode|L20 GPU|进度|L20成功/观察数|4090成功/30|L20整任务秒/条|4090整任务秒/条|
|---|---|---:|---|---:|---:|---:|---:|
|adjust_bottle|demo_randomized|2|complete|18/30|16/30|204.31|162.33|
|beat_block_hammer|demo_randomized|4|complete|7/30|8/30|215.01|154.46|
|click_alarmclock|demo_clean|6|complete|30/30|30/30|53.02|34.54|
|place_shoe|demo_clean|5|complete|27/30|28/30|71.13|91.18|

速度：整任务包含初始化/专家筛seed/所有rollout；L20另有逐动作get_obs、take_action与policy RPC计时、逐rollout计时。4090历史日志无同样分段计时，不能伪造对等分段数据。新L20四会话共用独立GPU7单模型，历史4090各模型1–2仿真，CPU/网络/背景负载不同；速度比为实际运行系统对照，不能单独归因GPU型号。原生seed扫描可能接受不同seed，记录顺序相同与交集，不假称逐场景完全配对。

输出：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward20000_l20_vs4090_4x30_20261024T083037Z`。
