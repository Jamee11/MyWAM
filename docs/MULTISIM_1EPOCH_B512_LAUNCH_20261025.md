
# 2026-10-25T15:46:24.092562+00:00 多仿真 Sidecar 四卡正式启动记录

用户明确授权本机后四张A100启动，按示例将可见卡从0–3修正为4,5,6,7。启动前四卡各81153MiB空闲，端口29684可绑定，未操作前四卡或其他作业。

运行目录：`/remote-home/jinminghao/OpenWAM/outputs/midtrain/multisim_1ep_b512_warmup500_20261025T154412Z`；启动PID：664226；启动记录`launch_record.json`，入口`bootstrap.sh`。先执行CHECK_ONLY=1快检，成功后执行CHECK_ONLY=0（正式入口也按原脚本再做一次快检），输出分别preflight.log及launch.log。原始数据/代码/配置未改；仅新增本次输出与记录。

参数：用户指定.venv/bin/python、Foundation目录、项目内archive_v2 manifest；1epoch，bs32/acc4/四卡/global512，14223次optimizer更新/56892micro，MASTER_PORT29684。WAM前500optimizer updates（2000micro）完整Read/Write和Video/Action/EA反向，只跳过WAM参数/Adam状态更新；第501次更新开始联合。WAM LR2e-5，Sidecar1e-5，EA scale0.5，ColorJitter(.2,.2,.2,0)开启、BF16/ZeRO2/gradient checkpointing/fused AdamW、offload none；save4000micro=1000updates，最近3组完整状态。增加expandable_segments:True沿用已有容量调试结论，未更改模型/优化目标。解析后配置在requested_resolved_config.yaml，manifest及入口/模型/optimizer-hook SHA记录在launch_record.json。

当前状态：后台进程存活，数据快检/依赖加载中；不把进程创建当真实更新或WAM第501次联合更新验收。后续核对四rank真实metrics、显存、首更新及500-update边界。首500次warmup通过不能证明WAM Adam状态首次分配可用，先前双卡OOM记录仍保留；本次四卡目标bs32尚待实测。CPU offload不作为绕过方案。不自动改batch/预算或停止其他作业。

## 2026-10-25T16:09:53.345435+00:00 四卡真实更新验收

两次12窗口快检通过，四rank正式worker15:55前后创建，16:00:44开始Foundation重建，16:03:31起实际读取checkpoint_step_154000.safetensors，16:06:34四rank完成架构构建，随后DeepSpeed包装并进入训练。首轮micro4/opt1已通过；本次最新四rank至少micro19/opt4，全部历史metrics有限、同步更新grad norm正值，三write gates已从0变化，ea_backbone_frozen_warmup=1。loss total0.336156/video0.150327/action0.052824/weighted EA0.133005。未见OOM，GPU4–7 NVML约53–55千MiB（约52–53GiB），只是warmup当前阶段，不能外推第501次WAM Adam状态首次分配后的峰值。

初期去除前2micro均值6.716s/micro，acc4约26.86s/update；小窗口而非稳定吞吐/最终ETA。训练继续：56892micro/14223updates/1epoch，未改bs/acc/epoch/500warmup。实际权重输出`/remote-home/jinminghao/OpenWAM/outputs/midtrain/multisim_1ep_b512_warmup500_20261025T154412Z/2026-10-25_16-06-34`，launcher PID仍664226，原GPU0–3/其他作业未操作。

证据`startup_acceptance.json`、实际run config和per-rank metrics。readonly monitor_startup.py持续记录startup_monitor.json至四rank达到第501次更新或launcher退出，不终止/重启任何进程；第501次联合更新/完整保存恢复/策略闭环仍未验收。WAM warmup后的容量须以实际观察为准。

启动慢点：按用户示例，两次独立快检约前10分钟均含依赖导入，非全量数据扫描；随后四rank导入/CPU模型初始化。额外只读导入探针的单进程日志存import_profile.log，transformers和Wan text_encoder的self import各约44/41s；该探针无GPU、未实例化完整模型，不能等同全部启动墙钟拆分。不为本次运行改共享环境或核心训练逻辑，未重启已启动的正式任务。

## 2026-10-26T02:14:21.257186+00:00 进度与loss趋势分析

现场四rank同步约1360updates/14223（约9.6%），warmup flag已0；只读监控在2026-10-25T19:48:53Z确认四rank达到501更新后退出。最新记录已逾850次联合更新，未见非有限loss，第501次WAM Adam状态首次分配的四卡首轮容量问题在本bs32/acc4配置已通过；不代表更大batch容量或长期稳定性。全部rank最新metrics持续更新。

按每4micro求一次optimizer update的loss均值，再取100updates窗口；没有把单个batch与长窗口均值混用。完整曲线与逐update数值在/remote-home/jinminghao/OpenWAM/outputs/midtrain/multisim_1ep_b512_warmup500_20261025T154412Z/analysis_20261026/loss_trend.json、loss_trend.png和loss_trend.svg。

| 更新区间 | total | video | action | weighted EA |
|---|---:|---:|---:|---:|
| 1–100 | 0.345739 | 0.151246 | 0.062805 | 0.131688 |
| 401–500 | 0.318360 | 0.152264 | 0.062360 | 0.103736 |
| 501–600 | 0.241042 | 0.125299 | 0.016589 | 0.099154 |
| 901–1000 | 0.198041 | 0.103419 | 0.005385 | 0.089236 |
| 1261–1360 | 0.182426 | 0.097661 | 0.004212 | 0.080553 |

前100→最近100均值：total下降47.24%，video35.43%，action93.29%，EA38.83%。最近EA占总loss约44.16%；原始ground6.149→4.395、relation1.051→.365、plan1.651→.692，三项均下降，但原始项不同定义/尺度不能直接互比大小。前500WAM参数不更新时video/action近平台，EA下降；第501起video/action明显降低，之后持续缓降，和配置的训练阶段吻合。不是断言该结构导致SR提升，也不能以train loss诊断过拟合，尚无本run验证集/闭环结果。

已读rank0全部metrics有限；最近100次同步梯度norm范围.583–1.338（裁剪前记录，不代表超过clip1后仍未裁剪）。最近100更新约26.43s/update，剩余约94.4h，纯更新ETA北京时间10月30日08:38；额外保存/负载波动另计，建议按10月30日上午前后粗估。checkpoint_step_4000权重及对应完整state完成marker存在，marker global4000/opt1000/epoch0；未实际重载，不称GPU恢复验收通过。此次只读监控与独立绘图（复用已有robotwin环境的matplotlib，无环境安装或训练变更）。

## 2026-10-26T03:01:59.009217+00:00 当前读取方案代码核查

实际运行指针与config：archive_v2 manifest，train+selected_for_midtrain过滤，只使用39816训练轨迹，val6161/test6227不取训练。rlbench_franka/maniskill_franka/robotwin_aloha/robotwin_piper weights1/1/1/1归一化各25%，来源内随机窗口排列，不强制每个batch各占四分之一；虚拟epoch7282176窗口，较小来源重复、较大来源随机子集，并不表示逐轨迹恰好访问一次。Accelerate把同一混合索引分给四rank，rank各4个DataLoader CPU workers，pin_memory=True，bs32/acc4/global512。

NPZ一次读取某episode的state/action/targets并验SHA、按统一训练集stats归一化state/action；@lru_cache(maxsize32)的方法缓存由四来源实例共用，每个worker合计最多32条episode，淘汰后重读重验。HDF5打开句柄每来源每worker保留8个LRU，重新打开核对size/mtime；不是解码RGB帧的全量缓存。RGB每窗口从native相对路径读取所需0/4/.../32（9个）时间点，按codec区分uint8 RGB、legacy JPEG及标准JPEG的通道处理；整图384×320，ColorJitter同一随机参数用于9帧；当前proprio1×80、action32×80（无效臂/尾部mask）、EA targets1×130。GPU阶段再执行冻结VAE和文本编码，不是预存latents。启动12窗口快检不在每步重复；数值缓存miss的hash及重新打开media的stat是正常reader完整性检查。

只读核查另发现：PreparedSimDataset.__getitem__将两相机样本先补黑第三槽，但调用ColorJitter只传video，没有传现有video_jitter_exclusion_masks；对比度因子小于1时该黑槽会向图像均值变灰，不能称增强后缺失槽始终纯黑。此结论来自当前调用和PIL对比度数学，未统计运行样本发生率或实验效果，未修改正在运行的代码/训练。

## 2026-10-26 缺失视角增强：真实样本可视化

按用户要求，独立CPU诊断实际archive训练图像，不改正在运行的reader、模型或训练。六张真实首帧抽查在brightness=1、contrast=0.8、saturation=1时，缺失右下槽从RGB0变为7–29/255；contrast=1.2时仍为0。0.8是当前允许区间下限，用来展示机制，不代表训练随机采样频率或平均强度。

图像输出目录：`outputs/midtrain/multisim_1ep_b512_warmup500_20261025T154412Z/analysis_20261026/missing_view_visual_comparison/`。脚本从现有源文件AST提取并执行原始VideoColorJitter与拼图函数，避免导入完整训练栈；RLBench/ManiSkill各一条真实训练轨迹首帧，比较原图、当前无mask增强、相同参数传入现有缺失视角mask的增强。输出comparison.png/pdf、六张无标注原始分辨率PNG及provenance.json；红框/文字只在拼图上覆盖，未写入原始单图。

已确认影响是结构补黑变灰，以及补黑参与contrast均值而轻微改变有效相机区域的增强。缺失槽占画布1/6，动作、EA标签及时间对齐没有因此改变。可能增加视频分支对无意义灰度变化的拟合负担，或与推理纯黑输入产生差异；未做配对训练/闭环，不能量化SR影响或据此宣判本run无效。下一步若修改reader传mask需明确取得核心逻辑变更授权并保留本run预处理来源；此次仅生成分析产物。

同期只读进度约2928/14223 optimizer updates（20.6%），最近100更新total/video/action/weightedEA均值0.154200/0.088476/0.002968/0.062755，约27.29秒/update，剩余约85.6小时（保存/负载波动另计）。checkpoint8000完整状态marker为micro8000/opt2000/epoch0；未进行恢复重载验收。
