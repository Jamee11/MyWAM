# 单模型单仿真与双仿真吞吐实测

状态：interrupted_insufficient_capacity

32k固定权重、同一GPU6模型，独占两个策略端口；两份adjust_bottle/Clean任务，各2条native评测，仿真参数为GPU2/3（GPU7余量不足，在其子任务启动前改为GPU3，两个阶段一致）。先serial依次执行，再parallel2同时执行。与正式10000条成绩分开。

| phase | status | wall seconds | episodes/hour (pilot) | printed steps | rollout window seconds |
|---|---|---:|---:|---:|---:|
| serial | complete | 1296.3351739123464 | 11.108238278022437 | 489 | 490.58412903919816 |
| parallel2 | interrupted | pending | pending | 0 | pending |

边界：每组仅4条episode；顺序serial→parallel2可能有共享盘/库/资产缓存预热偏差；其他卡后台负载变化；GPU参数控制CUDA选择，未据此声称已经逐进程审计所有Vulkan/OIDN实际物理设备。native模型CUDA RNG未额外固定，不能称动作逐位配对。比较同seed仍须核对实际step差异。墙钟含启动、权重校验、仿真初始化与收尾；rollout window从首次观察step起估计，不是细粒度模型耗时。共享模型predict锁串行，若有加速来自流水交叠，不是模型并行计算。

证据：outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/throughput_ab_20261020/state.json；不将失败/部分pilot纳入正式SR。


最终处置：本次状态 interrupted_insufficient_capacity。完整serial 4条合计1296.34秒，其中从首个step到结束合计490.58秒；parallel第二路因GPU3余量不足尚未启动即终止，**没有有效加速比**。GPU7和GPU3容量等待及pilot中断原因见state.json，正式恢复调度已释放并重新接管；此数据不能作为多仿真快/慢结论。
