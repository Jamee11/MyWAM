# ManiSkill Sidecar 25410 与 baseline 14K 的相同协议评估

## 真实启动验收更新

04:26 UTC：首批两个PushCube正式策略episode均成功结束，12worker持续存活，
无failure.json；首个完整视频H.264/512×256通过ffprobe。此处只确认首批闭环，
不以2条推断最终600条SR。协调器PPID1，后台不依赖当前对话存活。

04:20 UTC：原生单臂GPU数值gate已通过，两个fixtures×无cache/同cache参考的
四组FFN比较max_abs_error均0，实际生成4个Inductor kernels。Sidecar三tap为
7/15/23且enabled=true，实际write gates约+0.0006714/-0.0003624/+0.0003185，
replan8/cache .99 max3由服务实际配置确认。产物为本次根目录下
`server_gpu6_manifest/{validation/result.json,sidecar_contract.json}`。
该fixture编译检查不等于缓存与完整计算等价，也不证明模块带来SR收益。

04:23 UTC：12个正式worker存活且均开始写真实rollout视频，未见failure.json；
尚无完整episode终态，600条持续运行。当前GPU5约22GiB空闲、GPU6约12GiB
空闲，其他作业未操作。以下“待数值gate”的段落属于启动时历史状态。

2026-10-22 UTC，用户要求沿用 baseline 评估逻辑，测试完整 Sidecar checkpoint，
尽量公平对比。固定权重：
`outputs/maniskill_sidecar/joint_20261019T222758Z/2026-10-19_22-33-43/checkpoint_step_25410.safetensors`。

## 已核对的一致项

- 直接复用 `benchmarks/maniskill/run_eval.py` 和未经修改的 `single_eval.py`；
  观测/动作桥接源码 SHA256 与正式 baseline run 相同。
- 同六任务 color-distractors、各100条、相同 successful-controller test seed
  manifest/顺序，共600条。200控制步、replan8、32动作chunk、10步去噪、seed42、
  CFG1、sync、cache .99/max3、原成功判据、H.264视频均保持。
- front/hand RGB、384×320 L画布、训练同款文本、实际base-frame TCP EEF10。
  `pd_ee_pose`、column-Rot6D→XYZ Euler、夹爪裁剪/计数逻辑完全复用。
- 两份checkpoint的`normalization_stats.npy`逐字节SHA256相同；33帧/stride4、
  unify0:10、尺寸/相机布局/增强配置相同。没有给Sidecar添加object oracle输入。
- 本次只加载完整Sidecar，三处tap7/15/23全部保留，不改write gate、不做消融。

## 必须公开的区别

- Sidecar step25410是微步数；保存状态为`opt_step=12708, epoch=5`。
  原baseline step14000对应14000次更新。评估协议一致不等于训练预算完全一致，
  不能把SR差异全部归因于Sidecar结构。
- baseline使用完整MoT compile；Sidecar复用已有评测专用FFN compile，保留完整
  Sidecar计算并保持同一joint velocity cache。不能用原baseline编译循环绕过
  Sidecar。正式放行前用两个原生ManiSkill RGB/EEF10 fixture做eager/FFN×cache
  on/off验证，比较匹配cache参考，沿用atol .005/rtol .01及真实kernel门槛。
  此门槛只证明本fixture编译保真，不证明cache相对完整计算无误差。
- 当前GPU5/6已有其他作业。仅GPU6新增一份模型，GPU5运行12路仿真（每任务
  50+50互斥分片）；与baseline的双模型布局不同。每端口独立policy/动作队列，
  所有模型请求串行、每generation重置cache，固定推理seed。不能宣称跨硬件负载
  逐动作bitwise复现或相同墙钟吞吐。

## 实现与启动

新增`benchmarks/maniskill/serve_sidecar.py`，复用已有Sidecar runtime和数值检查器；
检查器仅增加可选native inputs，默认RoboTwin调用路径不变。原模型/训练/数据
核心不修改。GPU放置前等待至少33GiB空闲，不结束其他作业。
CPU回归14项通过，包括原pool隔离、Sidecar完整计算/输入mask、原生单臂fixtures、
预算不一致拒绝、600条分片覆盖。原客户端的pandas依赖修复保持不变。

启动协调器PID3221029，权威输出：
`outputs/maniskill_eval/sidecar25410_six_task_100_replan8_max200_matched_20261022`。
日志：`outputs/maniskill_eval/sidecar25410_matched_20261022.launch.log`。
`fairness.json`保存输入/统计/seeds/更新数审计；`run.json`记录权重/源码/配置哈希；
固定权重hardlink只读，防训练keep-last清理，不原地改写。

当前已启动，真实模型数值门槛、首批策略闭环和600条终态待验收，不报告SR。
完成后自动输出逐任务`summary.json`、与baseline逐seed对照的`comparison.json/md`。
下一步先验收Sidecar分支实际启用和首个完整episode，再以完整100条/任务报告。

```bash
.venv/bin/python -u benchmarks/maniskill/run_eval.py \
  --policy sidecar \
  --checkpoint outputs/maniskill_sidecar/joint_20261019T222758Z/2026-10-19_22-33-43/checkpoint_step_25410.safetensors \
  --output outputs/maniskill_eval/sidecar25410_matched_NEW \
  --gpus 6 --sim-gpus 5 --workers-per-task 2 \
  --episodes 100 --max-steps 200 --replan-every 8 --denoise-steps 10 \
  --port 21980 --expert-report outputs/maniskill_baseline/expert_replay.json
```
