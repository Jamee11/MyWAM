> 已按用户要求停止本机评测，以下为停止前的部分结果，不是完整百条成绩。

# Sidecar 32k RoboTwin 实时评测

更新时间：2026-10-19T06:52:59.910491+00:00

运行目录：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_gpu56_20261019T021700Z

32k=16000次参数更新；20k=10000次参数更新。两组使用cache+FFN局部编译、相同native协议；32k为A100，本轮20k参考为L20，硬件/编译内核及accepted seeds可能不同，不冒充逐seed因果对照。

|任务|条件|32k状态|32k成功/完成|20k完整百条|差值百分点|accepted seed交集|
|---|---|---|---:|---:|---:|---:|
|adjust_bottle|demo_clean|failed|13/14|98|—|—|
|adjust_bottle|demo_randomized|failed|2/6|34|—|—|
|place_shoe|demo_clean|running|25/27|94|—|—|
|place_shoe|demo_randomized|running|9/14|49|—|—|

已验收完整组合：0/100；当前策略完成：61/10000。未完整项不纳入最终成功率，不将运行中比例当全量宏平均。

模型/编译门槛及A100环境适配说明见ROBOTWIN_SIDECAR32K_GPU56_20261019.md。


## 2026-10-19T06:54:06.792254+00:00：用户要求停止本机32k评测

已核验PID/进程组后停止本机模型、两路仿真、调度和监控；所有目标进程均退出，GPU5约5MiB、GPU6约1045MiB。本次共完成61条，无完整百条任务：瓶子Clean13/14、Randomized2/6；单鞋Clean25/27、Randomized9/14。瓶子两路此前已退出143（SIGTERM），发起来源未确认，不判为策略失败或渲染异常。全部为部分观察，不能作为正式SR或32k对20k优劣结论。日志、checkpoint快照保留；另一台L20评测未操作。
