# RoboTwin仿真热点测试与中断恢复

更新：2026-10-28T13:08:13.776491+00:00。

动作仍按实际状态逐步规划、250Hz物理步进与原成功检查执行。不得把32个动作合成一次跳跃，或用前一步预测状态替代实际执行状态。

## 热点结论

同一官方clock场景/62条记录动作重放，不使用模型推理，仅作计时而非正式SR。原生warm baseline21.830s，其中规划12.718、physics1.532、camera capture4.553、RGB读回0.884；规划占take_action78.44%、physics9.45%。双臂GPU流并行局部规划11.168→9.110s，但0/62完整position/velocity轨迹逐元素相等，未应用正式评测；未区分所有差异是否固有重复漂移，不能声称改变了算法，也不能认为已证明等价。lazy RGB24.205s比warm baseline21.830s慢，未应用。首baseline31.164s受冷相机采集影响，不当加速基线。

线程1/4交替cProfile对照：34.579冷baseline、22.927单线程、25.207 warm baseline、26.438单线程复验，没稳定增益，未改正式线程设置。profile自身有开销，不与无profile绝对时间混比。868次CUDA synchronize约6.211s、trajopt约11.010s为包含关系；不能简单跳过优化迭代/碰撞检查来换速度。所有候选代码保持独立未启用，不删旧实现。

## 正式通信链路与恢复

旧加速stack_blocks_three/Randomized attempt3已真实执行2825动作/3条后，于11:01:05因GPU6 free1921MiB低于2048被保护中止（exit−15），没有实际OOM日志；原3次infra上限耗尽使条件failed。未将这些partial并正式SR。当前一份完整70的独立恢复保留原seed100036续接、候选指令70、RT/相机/Sidecar/cache/去噪10/horizon32/成功/预算、无视频，最终必须exit0/70实际记录/与旧30共100唯一有序seed才接受。

GPU6恢复上限2，准入仍7680+1024=8704MiB，仅该卡保护余量改1536MiB；GPU1保留2而非旧失败3，GPU4主队列保留1给独立恢复留物理第二slot，GPU5原1792、其他2048保持。保护只选已知PID的真正extra，starting未给PID时不误杀baseline。原配置/source/status均有recovery_guard_backup。余量放宽仍可能遇OOM，原生renderer错误/partial必须作为infra处理，不能承诺峰值已永久解决。

L20现在单卡约17GiB空闲不足新完整副本，恢复服务只将原BF16 T5置GPU0、其余同权重策略置GPU3，复用模型构建及所有参数、peer-copy输入输出，不量化/CPU推理。两原生fixture×重复的完整32动作与原单卡canonical逐元素一致且Sidecar/cache门槛通过后才启动。设置每进程GPU3/0分配上限16/12.5GiB；当前显存变化/其他训练峰值仍需监测。不干预原六模型、任何训练或其他用户作业。

恢复新接口35910–35912，SSH私有反向转发；恢复仿真GPU4独立进程，准入9472MiB连续5次、低于1536或45min无动作/episode进展才中止本进程。最初恢复启动器误用4090本机/proc检查L20 PID而退出，已修为共享canonical证据+跨机token健康接口重试，原失败日志保留。

reporter只对原terminal-failed条件读取独立recovery_overlay，完整70后再审计并并入原30；不重写控制器状态、不覆盖已完成条件、不跨checkpoint/seed，四项CPU审计通过。新旧失败历史全部保留。当前状态：running，实报仿真10路；没有整套稳态加速倍数的结论。

证据根：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/performance_recheck_20261028T0900Z`；replay与thread profile在sim_hotpath_*，恢复在recovery_stack_blocks_random_20261028T1240Z，耗时统计在rollout_breakdown_20261028.json。
