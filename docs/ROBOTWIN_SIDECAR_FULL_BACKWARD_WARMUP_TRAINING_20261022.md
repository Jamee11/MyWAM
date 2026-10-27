# Aloha50完整反向／WAM延迟更新2K正式训练

启动：2026-10-22T13:49:18.864562+00:00。用户明确授权GPU0/1/2/3，允许结束当前ManiSkill训练，要求新实验先加载再交接。

## 本轮设置

- 输出：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z`；启动器PID和命令：`launch.json`；权威训练日志：`train.log`及实际run目录的`metrics_rank*.jsonl`。
- 环境：已有`/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv`，未新复制环境或权重；数据与Foundation仍用共享盘原路径。HF离线、endpoint为hf-mirror。
- 独立配置：`train_relational_sidecar_aloha50_frozen_backbone_10epochs`。从Foundation新建Sidecar训练，不加载旧isolation或42960的optimizer状态。
- 数据：`/remote-home/jinminghao/datasets/robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015/raw/aloha`，manifest=`outputs/sidecar/aloha_4350/manifest.ready.json`；50任务2500条Clean，552275帧，32动作/9视频，80D合同。使用已准备的标签与normalization统计。ColorJitter开启。
- 4卡、bs32/卡、acc2、global batch256、10epochs；预期42960microsteps／21480更新。前2000次更新（4000microsteps）正常读写、正常WAM/Sidecar前反向与global clip，仅跳过WAM参数/Adam/decay更新，之后joint。WAM在预算内少2000次更新。
- WAM base LR1e-4、Sidecar5e-5，沿用cosine+线性LR warmup；Action/Video系数1，EA scale .5及原内部权重；不改gate初始化、不加heads-only、不加WAM-only第三阶段。
- 源码及配置已保存在输出source_snapshot与source_sha256.json，resolved_config.yaml为启动解析结果。CPU53项、两卡微型BF16/ZeRO2/acc2工程验收见`docs/evidence/sidecar_full_backward_warmup_20261022/validation.json`。

## 交接与验证

通过现有CPU-ready handover入口等待全部rank的模型/loader准备，之后只结束launch.json中已核对启动时间、命令和CUDA分配的旧训练。新进程在GPU0–3各空闲>=70000MiB且至少90秒通知窗口之后进入DeepSpeed。旧ManiSkill最后完整恢复状态为33880，保存文件保留，不删除训练产物。

当前新进程已启动加载，尚未确认GPU接管与真实参数更新。随后补充old-stop、gpu-release、首个有限loss/optimizer update证据；2K切换与闭环效果不从启动成功推断。所有现有GPU4–7评测保持原作业。


## 2026-10-22T13:57:58.024694+00:00：GPU交接与真实训练启动确认

- 新CPU-ready：13:54:20 UTC；按PID/start_ticks/命令/CUDA双重核对后，13:55:42向旧torchrun2430422发送SIGTERM，13:55:45旧worker及其dataloader全部退出，无残留。旧checkpoint与完整33880恢复状态保留；旧run写入user_requested_stop_20261022.json。
- 13:55:52新gate放行，各卡空闲约79–80GiB；13:56:07四rank进入DeepSpeedEngine训练。新实际run：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18`。
- 四rank均已有真实optimizer更新：记录时micro=12、optimizer=6。Action/Video/EA和total loss有限、warmup标记1，write gates离开0。total=0.378381，action=0.037368，video=0.208791，EA=0.132222。最近micro约7.33s，属于启动早期速度；不据此精确推完整工期。
- 当前GPU0–3约53–55GiB占用，训练中。启动配置再次核实bs32×4/acc2/10epochs/max_steps=null、ColorJitter、2K warmup、原loss/LR。没有停止GPU4–7其他作业。
- 证据：输出startup_validation.json、handover/old_stop_status.json、handover/gpu_release.json、train.log及四rank metrics。WAM不更新的机制已由前述CPU/微型GPU验收；本真实run未独立比较完整WAM tensor hash，也尚未到2K边界或保存首份训练checkpoint。训练继续运行，尚无效果/SR结论。


## 2026-10-22T14:00:22.534248+00:00：warmup显存低于上一实验的原因

当前GPU0–3 nvidia-smi分别53631/53510/54730/54550MiB，旧ManiSkill运行时约64391/64290/65210/64130MiB。两配置均bs32、acc2、BF16、ZeRO2、gradient checkpointing、无optimizer offload；数据任务不同，旧ColorJitter关闭、新版开启。

主因是WAM尚未建立Adam一阶/二阶FP32动量。实际安装的PyTorch Adam._init_group仅在参数grad非None且state为空时分配exp_avg/exp_avg_sq；新helper在完整backward/clip之后、Adam前跳过WAM组，因此前2K不分配这两份state，并非关闭WAM反向或减少模型。DeepSpeed ZeRO2的initialize_optimizer_states对Adam只初始化梯度分片，不提前初始化moments。

WAM trainable=4,999,787,712+1,021,035,600+331,776=6,021,155,088参数，两份FP32state四卡分片，理论每卡额外12,042,310,176bytes=11.21528GiB。对照同属非更新累积microstep的历史peak allocated：旧60,517,658,112bytes、新48,475,378,688bytes，差11.21525GiB，与理论吻合。nvidia-smi与reserved受CUDA上下文、缓存复用和batch峰值影响，不要求恰好同差值。

2000次更新后WAM首次真正Adam step将建立这两份state，预计每卡增加约11.2GiB实际占用；最终reserved/峰值仍需边界现场确认。不能以当前低显存推断整个联合阶段容量。仅只读分析和记录，没有改变训练。机器可读证据：输出memory_warmup_analysis.json。
