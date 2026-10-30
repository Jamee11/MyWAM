> 2026-10-21 更新：用户授权提前并行，已取消等待全32k结束的门槛。当前执行以 ROBOTWIN_BASELINE10740_EARLY_COEXISTENCE_20261021.md 为准；下文保留原计划历史。

# 复现 baseline10740：等待 Sidecar32k 完成后接续的十任务诊断

配置时间：2026-10-20T14:39:47.405043+00:00。用户明确授权后台等待、评估完成后有显存再接续。

## 当前状态与调度

后台监督PID `2060359`（PPID1、独立session），每30秒更新 `status.json`；当前 waiting_for_sidecar32k_completion，无baseline GPU进程。现有32k全局调度PID2037013、15个独立worker共享剩余任务队列继续工作，原episode不打断。

只有当前本机32k权威 `global_remaining_20261020/parallel.json` 中100个唯一条件全complete、每项100条、无result errors，并且全部原始run.json经真实exit0/百条验收后才启动。不能只看进程退出或状态陈旧。主机名和boot_id固定；不看外机42960结果。若原评估未完整结束，会继续等待并保留原因，不跳过失败项。

原评估完成后，仅停止本次5个已固定PID+start_ticks且仍声明32k checkpoint的模型树，检查无活跃standard_eval引用后释放显存，不动其他进程。原生仿真已退出，未知占用不终止。随后按实时GPU余量选择最多3份baseline模型，每模型3独立同步端口，共最多9sim；模型启动free>=33000MiB，sim启动free>=16000MiB、每GPU最多2sim、逐次观察分配，剩余条件全局领取。容量或外部作业不满足时等待，不强抢卡。

## 权重与执行合同

用户源：`/remote-home/jinminghao/OpenWAM/outputs/1020-baseline-reproduced/checkpoint_step_10740.safetensors`；独立固定目录 `/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/baseline10740_after_sidecar32k_20261020/checkpoint_step_10740`。权重24813767464 bytes，SHA256 `bd1ab96a0ae5697dff3b8f2cedf5519e6be66d6e8141fd2cd245a20238ea2187`。权重为只读使用hardlink（共享inode，禁止原地写入）；config/stats/tokenizer独立复制并保存hash。

header完整、2089 tensors、全BF16、无Sidecar键；配置为原版dual_system/joint_self_attn。自有统计20D EEF、旋转min/max为±1、夹爪0…1通过CPU检查；与Sidecar自有stats不同，不互换。wxyz源EEF合同、20D→80D的0…9/34…43、384×320、9视频帧、10去噪步、32动作、CFG1、sync、cache cosine .99/max3、decode_video=false、原生官方RT/专家筛seed/动作预算/成功判据和逐episode视频沿用当前流程。

每份新模型用两个真实RGB/EEF fixture，匹配cache条件比较vanilla compile与eager，atol=.005/rtol=.01且有真实kernel证据才使用compile；失败则保留原cache的eager，写validation.json和模型各自最终deploy.yaml，不放宽容差。该GPU验收须等前序结束后实测，目前没有声称加载、compile或rollout已通过。

## 任务选择及当前候选

规则预先固定：待32k全100条件完成，以每个task最差的Sidecar−官方发布差距排序，优先≤−10个百分点的10个不同任务；不足10个时按次大差距补足并明确标记；同差距按英文task名。正式启动前写一次selection.json固定，不根据新baseline结果换名单。每task两域各100条，共20条件/2000条。先派发10个入选任务差距最大条件，之后跑另一域作对照，所有worker共享队列。

| Task | 当前差距最大域 | Sidecar32k | 官方发布 | 差距(pp) |
|---|---|---:|---:|---:|
| beat_block_hammer | demo_randomized | 33% | 55% | -22 |
| open_microwave | demo_clean | 61% | 82% | -21 |
| place_bread_skillet | demo_randomized | 41% | 61% | -20 |
| hanging_mug | demo_clean | 33% | 49% | -16 |
| stamp_seal | demo_clean | 73% | 89% | -16 |
| adjust_bottle | demo_randomized | 37% | 51% | -14 |
| click_alarmclock | demo_randomized | 65% | 77% | -12 |
| handover_block | demo_randomized | 40% | 52% | -12 |
| place_burger_fries | demo_randomized | 70% | 80% | -10 |
| place_cans_plasticbox | demo_randomized | 86% | 96% | -10 |

候选只基于当前已完整条件，最终以selection.json为准。官方参照为此前已存档并逐项核对的issue46表，本次不重新下载。

## 可解释性、结果与验收

Baseline保存配置写num_epochs=5、batch64/acc1；Sidecar训练计划10epochs、batch32/acc2，且这里比较的是中途32k checkpoint。没有原baseline完整训练日志证明数据/optimizer/增强/训练预算逐项受控，自己的stats也不一致。因此这次能诊断同评估栈下不同checkpoint与官方发布值的差异，不能把所有差距单独归因模块，也不能据双方都低就认定官方虚报。当前10任务按Sidecar负差选择，结果也不是全50任务无偏总体性能。

每个condition完整exit0/100条/无errors才计分；异常保留attempt和日志、最多两次完整基础设施恢复、不拼接。每完成一项更新comparison.json及comparison.md：官方、Sidecar32k、baseline10740及两种差值；未完整项不提前计最终SR。独立输出与现有32k不覆盖，全部结束后只回收本次baseline模型。

CPU预检通过：两入口语法、100唯一条件完成门槛、99条拒绝、10个不同任务排序与阈值；权重hash/结构/stats合同通过。实际后台状态/进程已核验waiting，未启动baseline模型或评估。

文件入口：

- 监督状态：`outputs/robotwin_eval/baseline10740_after_sidecar32k_20261020/status.json`
- 监督进程：`outputs/robotwin_eval/baseline10740_after_sidecar32k_20261020/supervisor_process.json`
- 监督日志：`outputs/robotwin_eval/baseline10740_after_sidecar32k_20261020/supervisor.log`
- 正式名单（完成后生成）：`outputs/robotwin_eval/baseline10740_after_sidecar32k_20261020/selection.json`
- 任务状态（启动后生成）：`outputs/robotwin_eval/baseline10740_after_sidecar32k_20261020/parallel.json`
- 对照结果（首项完成后生成）：`outputs/robotwin_eval/baseline10740_after_sidecar32k_20261020/comparison.md`

独立单次监督有flock防重复与selection.json启动标记；异常写fatal.json，禁止无检查重复启动覆盖既有attempt。保留原AGENTS/隐患中已冻结父调度器的adoption退出码生命周期，不手动SIGCONT旧控制器。
