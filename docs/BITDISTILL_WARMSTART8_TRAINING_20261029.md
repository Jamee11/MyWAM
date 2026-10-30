# bitdistill H100 8卡：2K权重新开10 epoch

2026-10-29，用户授权在空闲整机8卡节点启动已交付的训练方案。

## 设置与来源

- 节点：`aigc24`，`172.16.80.158`；GPU 0–7。启动前八卡显存/利用率均为0，无计算进程。
- 项目：`/mnt/100T/lixiangyu/cuigm/bitdistill/OpenWAM`，环境为项目内`.venv`。
- 入口：`scripts/train_sidecar_patchwam_warmstart_8gpu.sh`，复用现有cluster metrics入口；未改模型、训练算法、数据或标签。
- 权重：`assets/openwam_ckpt/openwam_alpha/Sidecar-PatchWAM-Warmup2K/checkpoint_step_4000.safetensors`，原run的2000 optimizer updates。
- 来源SHA256：`49b6c013eb910b9d0bc3071ec40a38915631d8a54dd6b73b1942d6968383835b`。此前完整归档SHA和解压inventory通过；本次复核权重size/mtime及manifest与ckpt统计SHA相等，未重复扫描25GB权重。
- 数据：50任务、2500条HDF5，使用迁移后的`manifest.ready.json`；PatchWAM增强沿用交付配置。
- 新optimizer、scheduler、counter，从step0开始10 epoch；`finetune_ckpt_path`选上述2K目录，`resume_ckpt_path=null`。
- `batch32 × world8 × acc1 = global256`；backbone/isolation warmup均0，直接联合训练；保留标准5% LR warmup/cosine。
- 每10000更新保存，10 epoch结束按原配置保存最终产物；保留原配置的full-state选项。预计21480更新，以真实loader计算为准。
- 模型离线加载，HF镜像端点保留；数据、权重、环境、缓存和输出均在项目目录内。NFS缓存/锁限制仍适用，启动异常需按真实日志处理。

## 启动与验收

- 当前UTC启动：`2026-10-29T01:39:36.561608+00:00`；torchrun父PID：`1709151`。
- 当前Run：`outputs/sidecar/patchwam_warmstart8_aigc24_20261029T013934Z`。
- 输出：上述run的`runs/`；启动日志`launch.log`；配置快照`resolved_training_config.yaml`；来源记录`launch_state.json`。
- CPU启动前检查通过：10 epoch、global256、fresh finetune、无重复2K延迟更新、save10000、LR warmup0.05、数据计数/统计、端口空闲、无同项目训练进程、八卡空闲。
- 第一次启动尝试仅因新项目尚无`outputs/sidecar`父目录而结束，尚未产生训练进程；随后补建父目录，保留原方案启动。
- `01:17:10Z`的run `patchwam_warmstart8_aigc24_20261029T011708Z` / PID1698439在Triton编译阶段因`ld: cannot find -lcuda`退出，尚无训练更新，日志保留。驱动`libcuda.so.1`实际存在，缺少无版本链接名称。
- `01:21:16Z`修复并实测：项目内`.local/lib/libcuda.so`指向节点已有`/lib/x86_64-linux-gnu/libcuda.so.1`（实际580.159.03）；`env_openwam.sh`增加`TRITON_LIBCUDA_PATH`和`LIBRARY_PATH`，原文件备份至`outputs/setup/env_openwam.before_libcuda_20261029.sh`。`CudaUtils()`真实编译通过，记录`outputs/setup/triton_cuda_link_check.json`；未改系统库/驱动或训练逻辑。
- `01:21:17Z`的run `patchwam_warmstart8_aigc24_20261029T012116Z` / PID1700079通过八rank NCCL、2K权重加载和模型初始化，但在第一个DataLoader batch因NFS不支持HDF5文件锁退出，无optimizer更新。独立默认只读打开同一HDF5复现errno37；`locking=False`在0.13秒成功；再以`HDF5_USE_FILE_LOCKING=FALSE`验证原默认h5py API只读打开成功。
- 仅当前训练启动包装脚本增加`HDF5_USE_FILE_LOCKING=FALSE`，不修改全局shell或HDF5文件。当前部署包装脚本旧版本保存在`outputs/setup/launch_warmstart8.before_hdf5_20261029.sh`。DeepSpeed退出时的Triton autotune FileLock也可报NFS锁异常；这不是前一run的首个错误，后续仍需区别主异常与退出缓存异常。
- `01:30:46Z`的run `patchwam_warmstart8_aigc24_20261029T013044Z` / PID1704863已通过真实首batch forward/backward，但首个optimizer.step被`SidecarLRScreen.before()`的分组标记检查拦住：`backbone_warmup_updates=0`时工厂不附加`ea_backbone_group`，`control`钩子却仍要求该标记。无optimizer更新，日志保留。
- 运行时覆盖`training.sidecar_lr_screen=null`关闭该无干预诊断钩子；不修改core源码或已交付配置。20步CPU AdamW对照表明，与带正确标记的`control`相比，权重、Adam step/一二阶moments、warmup LR逐位相等，记录源端control目录的`control_hook_parity.json`；此为CPU检查，不宣称完整八卡GPU逐位等价。标准5%LR warmup/cosine和原优化器分组/LR不变。
- 复现实际入口需要该覆盖：`source env_openwam.sh; export HDF5_USE_FILE_LOCKING=FALSE; bash scripts/train_sidecar_patchwam_warmstart_8gpu.sh training.sidecar_lr_screen=null`，另选独立输出和空闲端口。包装脚本旧版本备份至`outputs/setup/launch_warmstart8.before_lrcontrol_20261029.sh`。
- 启动验收已通过：`2026-10-29T01:47:06Z`，八rank共同完成5个真实optimizer更新，micro/optimizer计数均从0开始递增，所有记录的loss/grad norm有限。

## 启动验收结果（非训练终态）

八rank在step5的归约指标完全一致：总loss `0.2973007858`、video `0.2159739435`、action `0.01763792895`、EA `0.06368890405`、grad norm `0.4853250384`。当步各rank耗时约4.446秒；最初加载/首batch开销较大，不据此承诺全程吞吐或ETA。各rank该步CUDA allocated峰值47.92GiB，观察时nvidia-smi显存约52–53GiB。

实际日志确认scheduler总计21480 optimizer更新、标准LR warmup1074更新；10epoch目标仍在运行。真实训练输出位于当前run的`runs/2026-10-29_09-43-36/`，含`metrics_rank0.jsonl`至`metrics_rank7.jsonl`。启动证据为当前run的`startup_acceptance.json`，源端另存`outputs/transfers/bitdistill_20261028T130623Z_control/startup_acceptance_snapshot.json`。

## 后续

八rank首次真实optimizer更新、有限loss、权重来源和新计数已确认；显存/步时按上述初期测量记录。正式训练继续运行至原10 epoch终点，后续检查10000/20000更新及最终权重与恢复状态。若后续失败，保留日志并区分部署/数值/容量原因，不静默改变batch、监督、warmup或模型结构。

near/contact参考点、EEF replay、强增强颜色语义和matched baseline等研究边界保留；本次启动不提供闭环SR或泛化收益结论。


## 2026-10-29T07:10Z 训练运行监督

现场两次观察八rank共同从4454推进到4459（20秒），约20.8%/第3个epoch，父PID1709151存活。各rank近200条记录步数严格递增、loss与grad norm均有限；检查的近期日志未见OOM、NCCL WARN或异常栈。最近两段100步total均值0.187793→0.186243，action均值0.001488→0.001522，按训练随机batch正常波动记录，不作策略质量推断。最近100步均值4.1798秒，线性剩余约19.77小时（不含后续checkpoint写盘与吞吐变化）。GPU采样八卡100%利用率，显存约52–53GiB；尚未达到首次10000更新保存点，checkpoint列表为空符合当前配置。未改或重启训练。证据源端outputs/transfers/bitdistill_20261028T130623Z_control/training_health_latest.json和training_advancement_latest.json。本次只确认当前训练健康，不证明后续全程稳定或闭环收益。


### 2026-10-30T04:13:36.028554+00:00：bitdistill warmup2K→联合10epoch最终权重保存已确认，本地归档进行中

2026-10-30T03:55:48Z只读复查：旧run八rank已21480更新，final checkpoint_step_21480.safetensors为24,862,363,126 bytes，safe_open可读取2229张量；两阶段接续器已写predecessor_completed.json并自动进入新的stage1。这解决原10epoch完成/最终权重保存未验事项，不代表闭环收益。退出NFS清理/Triton缓存ENOLCK继续存在，不改写成保存失败。用户要求归档至outputs/1030-warmup2k-to-full10epoch：仅最终weights+config/stats/contract/tokenizer/日志/说明，不含完整optimizer/RNG状态，服务器原件保留。SFTP单路2.03MB/s，8路总1.22MB/s未加速；配套文件已回传，最终权重使用独立旧2K副本作rsync差分参考，只有两端SHA256全匹配后改名正式文件。partial预置24.86GB不代表下载完成；传输与GPU评测尚待验收。实时证据该目录transfer/status.json及README.md。
