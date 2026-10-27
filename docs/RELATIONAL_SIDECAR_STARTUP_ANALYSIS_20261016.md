# Sidecar 启动耗时分析（2026-10-16，测量中）

本次双卡5/6、真实Foundation、Aloha prepared reader、warmup/bs1/acc1/2更新已完成，全部Sidecar梯度与门控更新、两rank完整状态保存通过，不外推joint容量。

| 时间 UTC | 阶段/证据 |
| --- | --- |
| 09:26:21 | torchrun父进程开始（ps lstart） |
| 09:30:01 | rank0 worker开始，父进程启动开销约220秒 |
| 09:33:15 | Hydra日志目录创建；worker早期Python依赖开销约194秒 |
| 09:45:24 | BF16/Accelerate配置日志；距worker开始923秒，包含训练入口依赖导入与seed，不是构造模型 |
| 09:50:39 | prepared dataset返回，开始Trainer；上段315秒包含Accelerate/NCCL初始化+数据构造，现有日志未逐项分离，不全部算给数据 |
| 09:50:39–09:52:38 | Wan/VAE/T5组件构建约118秒 |
| 09:52:38–09:56:04 | Foundation权重加载约206秒 |
| 09:56:25 | DeepSpeed包装完成 |
| 09:56:42 | 两个实际optimizer updates完成；15.22秒/1.55秒，非长期吞吐 |
| 09:57:48 | 部署safetensors保存完成，保存约66秒 |
| 09:59:50 | DeepSpeed model/optimizer/scheduler/sampler/RNG完整状态保存结束，trainer_state marker证明2次更新 |

源码及现场证据：进程频繁处于cxiWaitEventWait（GPFS），读取调用持续增长；解释器base、.venv、项目和权重均位于/remote-home的GPFS。CPU回归2174项在GitHub本地盘runner只用132.91秒，不能直接当本机预期，但说明15分钟不是Sidecar模型计算的固有成本。每rank另有Inductor compile_worker --workers=32，增加导入/初始化并发开销；仅凭当前日志不能将全部延迟定量归因于此。

已做的最小变更：新训练launcher设置TORCH_WARM_POOL=0，当前torch源码证实仅跳过import-time AsyncCompile.warm_pool，保持模型/精度/loss不变。本次已启动进程不受后续环境变量改变影响。尚待下一次启动量化改善。

prepared数据已离线构造，fast preflight不扫描2500个HDF5、不重做标签/stats。不能将launcher import、全量full audit、权重加载或完整状态保存叫作每步数据构造。后续用importtime和阶段单独时间戳定位，再比较本地临时运行环境/权重缓存，保留源环境和全部原文件；不用虚假分配显存掩盖启动慢。

待完成：真实GPU恢复；启动优化前后对比；joint/四卡训练容量；长期吞吐和闭环。


### 2026-10-16 四卡joint正式训练已完成首轮更新

真实启动13:17:43 UTC，CPU-ready13:50:59（33分16秒），通知用户后由用户释放旧训练，13:52:33显存门槛放行，13:52:54四rank完成DeepSpeed包装。首个micro约13:53:36，首个optimizer update约13:53:43，合计约36分钟，含94秒交接等待。上次共享盘双卡warmup首次更新30分20秒，不能当本次四卡joint的严格性能对照。

四rank均完成至少7次真实optimizer updates；全部已记录指标有限，grad norm正值，三个write gates均已由0更新。当前batch32/rank、acc2、global256、joint可训练6045.4M参数、ColorJitter开启、10epochs共42960micro/21480updates。剔除头两micro，最近12个micro平均7.104秒，即约14.21秒/update；NVML整卡使用约64.1–65.2千MiB、GPU利用率100%。仅初期小窗口，非最大batch/长期稳定性结论。按现速10epoch纯训练约84.8小时，未含保存/数据波动/中断。

证据：docs/evidence/relational_sidecar_20261016/formal_4gpu_startup.json；原始metrics_rank0–3.jsonl位于outputs/sidecar/aloha50_C_joint_10epochs_20261016T131709Z/runs/2026-10-16_13-50-52。进程继续运行；本正式run尚未到首次save4000micro，完整checkpoint保存恢复、20次以上稳定更新和闭环效果仍待验收。未操作旧进程kill、未换临时环境。
