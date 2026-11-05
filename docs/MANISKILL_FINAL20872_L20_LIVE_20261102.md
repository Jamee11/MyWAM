# ManiSkill最终20872：L20标准八任务评估（不录视频）

2026-11-02 13:06 UTC，用户明确要求subagent在当前L20机器GPU3部署最终ManiSkill checkpoint、其他卡仿真，尽快取得成功率，并不保存视频。

## 启动与协议

- checkpoint：`outputs/maniskill_sidecar/midtrain_patchwam_1epoch_20261031T184131Z/runs/2026-10-31_18-46-16/checkpoint_step_20872.safetensors`，24,862,363,126 bytes；20872 microstep =10436 optimizer updates。
- 独立run：`outputs/maniskill_eval/final20872_l20_gpu3_novideo_20261102T130608Z`；launcher PID3941976（精确start_ticks及命令见launcher_process.json）。
- 模型GPU3，首批4路仿真GPU0，独立端口22500–22503；不触碰RoboTwin服务、远程4090仿真、训练或旧暂停ManiSkill队列。
- 原手册标准8任务×100 Clean＋100 HardRandom=1600条；Clean held-out val+test，HardRandom test D25%/M75%，原600步预算、绝对EEF、front+hand RGB与黑第三视图、预测32/replan8/denoise10、cache .99/max3和FFN编译数值验收均保留。
- 本机1600清单已生成，manifest SHA `a96d7f13dcb243f9429303a667d76d02151098a83c2946d996a31b21bf4a07aa`，ID/family/physical seed/numeric/media训练交集均0。

## 可移植性与容量处理

复用已有模型`.venv`与仿真`/remote-home/jinminghao/miniconda3/envs/maniskill-eort-v1`，后者版本实读torch2.7.1+cu128、SAPIEN3.0.3、自定义ManiSkill3.0.1、NumPy1.26.4、Gymnasium1.2.3、h5py3.16、websockets16。无需新安装或下载。

当前GPU3空闲28864MiB（约28.2GiB），不足原serve_sidecar硬编码33GiB等待门槛。按用户对低风险试跑及指定GPU3的授权，只在独立输出目录复制runner/serve入口：ROOT改为当前项目绝对路径、runner调用本副本，放置等待门槛改为28GiB；项目原脚本默认33GiB不变。模型结构、权重、推理配置/FFN数值门槛不改，加载失败保留日志，不静默降为eager。

本机实际kernel driver550.127.08，显式`MANISKILL_GPUCOMP_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.550.127.08`及`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`。使用本机真实非空驱动库，未复制/覆盖其他机器驱动。

## 不录视频

archive_eval直接创建原生env；没有RecordEpisode、FFmpeg或视频编码/写入调用，输出仅JSON/JSONL/日志。`render_mode=rgb_array`保留策略所需RGB，不等于保存视频。模型decode_video由现有deploy配置关闭。实际输出文件和当前工作目录在开始rollout后再核对无新mp4等文件。

## 当前验收边界

已真实启动原24场景delta/绝对EEF桥接检查子进程，模型和FFN阶段须等待此检查通过。尚未有策略rollout或成功率，不能把清单/启动进程当评估完成。后续依次查看`run/expert.json`、`run/server_manifest/validation/result.json`、`run/worker*/episodes.jsonl`，最终仅完整1600覆盖及渲染日志无错误才写`run/summary.json`。

保留runner已有OIDN/ErrorDeviceLost/VK_ERROR_DEVICE_LOST/非法CUDA访问门槛与一次分片重试，不重复叠加新guard。控制回放只是桥接检查，不计策略SR；全程显存峰值、550驱动上的FFN数值一致与1600闭环仍待实际验证。

2026-11-02追加：GPU3初始28864MiB在CUDA context创建后可能低于28GiB，父线程明确授权按真实余量低风险监测试跑；在模型尚未启动前将独立副本门槛改27GiB，原33GiB保持。初始28配置保存在launch_config_initial_threshold28.json，当前launch_config.json为27。副本文件名capacity28为初始命名，实际内容/证据以当前配置为准；所有FFN/cache/控制门槛不变。

## 实际启动验收通过：2026-11-02T13:22:26.085518+00:00

24场景原delta/绝对EEF桥接全部passed，最大TCP差0.012531201355159283m。GPU3单卡完整模型加载成功，无split/量化/关闭FFN；现场GPU3used41307MiB/free4283MiB，原基线used16726MiB，增量约24GiB，未发生OOM。两native fixture的FFN与eager、FFN+cache与eager+cache逐元素相同/maxabs0，生成4个内核，cached温推理均值0.4181秒。cache相对无cache本身仍是近似（其中fixture最大差0.021484375），保留原协议，不夸大为cache无影响。

13:17:34四个独立会话端口22500–22503开放；四仿真PID3944740–3944743已实际请求模型并推进原生环境。当前6条完成/5成功，仅lift_peg_upright/Clean/test等最初样本，不报告为全1600 SR。已逐ID/task/condition/split/domain/seed对齐冻结清单、检查600步上限；四worker日志无明确渲染错误，输出目录无mp4/avi/mkv/webm。

验收快照startup_acceptance.json；实时逐条结果run/worker*/episodes.jsonl；最终run/summary.json仅1600完整覆盖后生成。原runner继续后台运行，后续首批峰值/整体SR仍待完成。

## 2026-11-02：按用户要求精简后续验收

当前后台模型与4路仿真不重启。已将此run的成功24场景/FFN记录注册为日常复用参考；新入口不再需要输出目录中的capacity副本，默认直接尝试显存分配。未变的数据直接复用1600条清单，未变的环境/结构/推理与控制代码复用既有验收及编译缓存；新权重仅一次32×10有限动作推理。源权重始终正常加载，不复用旧权重的预测结果。

入口scripts/eval_maniskill_archive.sh自动读取outputs/maniskill_eval/validated_run.txt。首次或关键条件改变才完整重验；REVALIDATE=1可显式重验。无新增调度锁、复杂守护或GPU完整重复实验。18项相关CPU功能检查通过，统一入口CHECK_ONLY生成的1600条清单与当前计划完全一致；新复用路径未另开GPU任务。详见迁机手册“日常启动”段落。

## 2026-11-02T13:43:53.329004+00:00策略阶段结果

{"utc": "2026-11-02T13:43:53.329004+00:00", "completed": 48, "successes": 38, "target": 1600, "by_task_condition": {"lift_peg_upright/Clean": {"episodes": 48, "successes": 38}}, "scope": "Partial task coverage, not suite average"}

四路继续推进，worker日志无OIDN/DeviceLost/非法内存访问，尚无完整summary；均首任务Clean，不外推HardRandom或八任务均值。

2026-11-02T14:40:57.770565+00:00：当前完成166/1600，约120条/h；完整暂估仍需12–16小时，任务难度变化可能影响。后台4路持续，无新验收/重启。
