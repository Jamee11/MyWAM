# Sidecar 32k：GPU5/6 RoboTwin 评测

当前实际布局（2026-10-19T03:00Z）：**GPU5一份32k模型，GPU6两路仿真；有效队列为gpu5a/queue_cross**。所有历史queue/queue_serial保留为incomplete，不计正式结果。四模型曾全部通过数值验收，但外部显存占用增长后释放三份本次空闲模型以保障仿真。实时结果见ROBOTWIN_SIDECAR32K_LIVE_20261019.md。

2026-10-19 UTC，用户明确授权评测32k，而不是20k重复评测。运行目录：`outputs/robotwin_eval/sidecar32000_gpu56_20261019T021700Z`。

- 固定权重：训练run的checkpoint_step_32000.safetensors，24862363126 bytes，SHA256 `0f22e7954ac5278e0e2184feea65c698754e4efad6b9fc9e317399e14f898de1`。独立目录以hardlink保留weight inode，训练unlink旧文件不影响快照；config/stats/contract/tokenizer已复制。32k微步=16000次参数更新。
- 使用20k已验收的cache .99/max3、Sidecar FFN局部compile、10步、32动作、CFG1、sync、wxyz自有stats；同official RoboTwin commit 0aeea2d、新Conda包、RT32spp/depth8/oidn、seed group0、unseen指令、原step limits和专家筛选，不启用planner fallback。
- 任务与20k进度快照、无重复四分区见task_partitions.json / comparison_20k_snapshot.json；优先瓶子、单鞋、扫码、锤子、双鞋、放盘、递积木。最终每task/mode100条；完整exit0/无errors/100条才计最终SR，进行中不计总体平均。
- 本机实际A100-SXM4-80GB/sm80，不沿用L20/sm89二进制。初始5/6各约80GB空闲，随后出现其他namespace占用各22398MiB，agent未终止这些作业。四个模型CPU bootstrap已经启动；每卡第二个模型须先等首模型完成验证及free>=42000MiB，避免抢显存。队列须通过数值检查、sm80构建和显存门槛才启动。

## 已定位并解决的本机渲染阻塞

默认SAPIEN初始化exit139，显式NVIDIA ICD及bundled Vulkan仍复现。catchsegv原生栈显示ld-linux的__tls_get_addr及libnvidia-gpucomp.so.570.86.10析构链；仅凭此不将根因绝对归于某一个版本。进程环境LD_PRELOAD现有`/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.570.86.10`后，原RT32×32 RGBA探针实际exit0、RT_PROBE_PASS。不安装驱动/不更改系统文件/不改shader。证据native_backtrace.log / preload_gpucomp_probe.log / vulkaninfo.txt。正式仿真继承该局部环境。

## 本机cuRobo构建与模型检查

原kinematics .so的cuobjdump仅sm89。为不覆盖20k环境，创建独立official detached worktree及独立cuRobo源码副本；大资产和cuRobo content只读链接共享，未复制全量资产。构建sm80并将新的五模块安装到私有目录，须实际CUDA kernel smoke通过后写sm80_build_pass.json；当前构建进行中。初次只因PATH未含已有ninja退出，修正PATH后独立retry日志保留，不改已有环境。

评测工具validate_server新增可选strategies参数，默认原三种范围完全保留；本次只验证用户要求的ffn，避免重复编译已经排除的全forward/读写策略。每模型两真实fixture、eager/eager_cached/ffn/ffn_cached，原atol=.005/rtol=.01、真实kernel证据门槛不放宽；通过后才对外服务。未知/空/重复strategy的stdlib self-check通过。

## 当前状态

四模型加载、sm80构建进行中，尚无正式32k任务成功率。原Aloha训练及20k队列未停止/改动。每池server.log、fast_validation/result.json，队列parallel.json/各task result.json均在本目录；queue_launch_state.json记录实际放行，不能将CPU启动/READY当闭环完成。


## 2026-10-19T02:42Z 启动结果更新

四份32k模型均实际完成A100数值验证：相同缓存条件下FFN编译对两fixture max_abs_error=0，均生成4个Inductor kernels；各份warm cache+FFN生成均值0.257–0.282秒/chunk。缓存相对完整计算仍有近似误差，不把编译保真混作缓存等价。sm80五个cuRobo扩展和实际CUDA quaternion kernel均通过，完整日志/五个.so hash见sm80_build_pass.json。

两卡外部占用先释放、后又各回到约22GB；四份模型都保留时每卡仅剩约9GB，无法安全仿真。用户回复“先按当前可用显存跑”，因此仅释放本次自己的gpu5b/gpu6b两份空闲模型，验证产物与PID记录保留all_four_model_processes.json/capacity_release.json。保留gpu5a/gpu6a，每份两slot，四路native百条评测已创建，首批adjust_bottle Clean/Randomized、place_shoe Clean/Randomized，后续扫码/锤子等已排队。两队列各25任务，全部50任务两域无重叠；实际模拟尚处checksum/环境准备，未提前计策略episode完成。

进度与20k完整任务对照自动维护于docs/ROBOTWIN_SIDECAR32K_LIVE_20261019.md和本目录progress.json，只比较双方有效100条结果并报告accepted seed交集；未完成不当最终SR。不改20k监控或队列。


## 2026-10-19T03:00Z 资源变化与真实闭环验收进展

初始四路已进入真实策略动作，但place_shoe/Randomized在相机take_picture报cannot create buffer；保留失败产物，未当作策略失败或有效SR。只降为每卡一路时发现卡总占用仍约72GB，而本容器模型每卡约25GB、已停止的客户端PID均消失，其他namespace已增到约47GB/卡。因此buffer异常与显存压力相关，但不凭错误字符串断言全部根因。agent仅停止自己的队列/模型，不处理未知进程。

最终保留原GPU5模型（PID2125758）而不重载，释放本次GPU6模型，将两路sim统一GPU6。新输出gpu5a/queue_cross，从相同原生seed group0重新完整100条；所有旧queue/queue_serial仅incomplete历史，不拼接/择优。GPU6已恢复约24–34GB余量，两路瓶子Clean/Randomized已实际推进约74/72动作，无新buffer/NaN错误；首个episode终态尚待完成。

与20k对应作业run.json复核：normalization_stats、deploy_config、policy_config、step_limits、wrapper、adapter、eval_policy及所有共同RoboTwin关键源码SHA均相同。实际差异为32k权重、A100硬件/sm80扩展及本机loader修复；accepted seed重合范围待完整结果记录。当前模型每chunk约0.3秒，场景约每分钟20–30动作，仿真执行成为主要耗时；不降低RT/预算/评测条数冒充提速。


## 2026-10-19T03:03:29.486038+00:00：首个真实闭环终态确认

有效queue_cross的adjust_bottle/Clean在121步成功，Randomized在122步成功，各完成1/100条，均为原native专家筛选/策略执行后的真实终态；两路已继续下一episode。不能据1/1推断最终SR或32k优于20k。当前GPU5剩余约10GB、GPU6约22GB，无新buffer/NaN/OOM。数值检查与闭环链路已通过，任务级100条结果和模型效果对比仍未完成。自动队列优先完成与20k对应任务，每项100条后再形成有效对照；只读取queue_cross。


## 2026-10-19T06:54:06.792254+00:00：用户要求停止本机32k评测

已核验PID/进程组后停止本机模型、两路仿真、调度和监控；所有目标进程均退出，GPU5约5MiB、GPU6约1045MiB。本次共完成61条，无完整百条任务：瓶子Clean13/14、Randomized2/6；单鞋Clean25/27、Randomized9/14。瓶子两路此前已退出143（SIGTERM），发起来源未确认，不判为策略失败或渲染异常。全部为部分观察，不能作为正式SR或32k对20k优劣结论。日志、checkpoint快照保留；另一台L20评测未操作。
