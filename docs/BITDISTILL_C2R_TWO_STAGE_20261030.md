# 官方 RoboTwin 微调权重 → Sidecar 2K → 低学习率联合 5 epoch

2026-10-30 UTC。用户授权在本地保留配置并同步 aigc24，在当前八卡实验完成后串行执行。
当前阶段：配置及自动接续脚本已编写，工程验证和远端排队结果见文末。

## 训练方案

官方起点：`assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Clean2Random/checkpoint_step_10740.safetensors`。
这是官方 RoboTwin 微调后的策略，不是 Foundation，也不加载当前 10 epoch 实验的权重。
已校验转移归档中的 SHA256 为 `2cab63b550ff45a52fbde4adc40ee2bdb7d4bafe5fa0dd6879e41e22ca2ff95f`。
2089 张官方张量均为 BF16，原权重没有 Sidecar；仅初始化新的 Sidecar，保留官方 policy heads。

| 阶段 | 初始化 | 更新与停止 | 峰值学习率 | 学习率调度 |
|---|---|---|---|---|
| 1 | 官方10740 + 新Sidecar | 完整反传和全局裁剪；仅Sidecar更新；严格2000 optimizer updates结束 | Sidecar `5e-5`；原策略LR `1e-4`，但跳过AdamW更新/衰减/动量 | 本阶段2000步独立cosine，标准5% LR warmup=100步 |
| 2 | 显式重新加载本次阶段1的2000步权重 | 新optimizer/scheduler/step0，Video+Action+Sidecar联合5epoch，预计10740更新 | Video/Action/base `2.5e-5`；Sidecar `1.25e-5`，均为此前1/4 | 本阶段10740步cosine，标准5% LR warmup=537步 |

两阶段均为8×batch32、acc1、global256；同一50任务2500条Clean HDF5及EA标签，PatchWAM增强不变。
阶段1使用 `stage=joint, backbone_warmup_updates=2000, isolation_warmup_updates=0`。
不能用 `stage=warmup` / requires_grad=False / no_grad / LR=0 替代：本实验要求原策略参与反传，
然后由现有AdamW pre-hook跳过原策略更新。第一阶段最后一个batch使用completed_updates=1999，
因此不会混入第2001次联合更新；保存后退出进程，阶段2通过finetune而非resume加载。

架构及EA权重/梯度比例不变。现有Sidecar直接Write仅进入Action hidden，但MoT双向注意力和
联合损失仍可间接影响Video；这不等价于“只影响Action且Video完全隔离”。本轮不叠加该结构消融，
也不额外冻结Video，以便先检验官方微调起点+较低LR的组合。

官方统计文件与当前数据统计文件SHA不同；实际使用的EEF min/max逐元素完全一致。
mean/std/分位数和metadata不同，但现有读取器使用min-max，本轮不迁移或重算归一化。

## 文件与执行

- `configs/train_sidecar_c2r_warmup2k_8gpu.yaml`
- `configs/train_sidecar_c2r_joint5ep_8gpu.yaml`
- `scripts/supervise_sidecar_c2r_two_stage.py`
- `tests/test_sidecar_c2r_two_stage.py`

所有远端文件均在 `/mnt/100T/lixiangyu/cuigm/bitdistill/OpenWAM` 内。
队列根位于 `outputs/sidecar/c2r_two_stage8_<UTC>`；阶段输出分别在 `stage1/runs/<timestamp>` 和
`stage2/runs/<timestamp>`，各自保留config、tokenizer、stats、checkpoint和每rank指标，不覆盖当前实验。

仅在aigc24运行；控制器端口29630防止同节点重复启动，torchrun端口29631。
接续器等待原PID及身份退出，再核对原八rank全部21480步和final保存日志/权重文件。
随后等待八张H100全部没有计算进程、已用≤128MiB、空闲≥78000MiB；不停止其它用户作业。
此前eval_standby应随原训练退出自动释放；如果未释放则队列等待，不强行杀进程。
NFS不支持flock，因此用节点本地socket互斥；这不提供集群调度器级排他锁，仍有外部资源争用风险。

阶段1结束须满足torchrun退出0、八rank全部2000步、final权重保存成功，并逐张量比较所有官方张量
与阶段1权重完全相同，才能启动阶段2。任何失败均停队列并保留日志，不自动重试或跳过验收。
阶段2须八rank全部10740步和final保存成功才标记complete；初始启动后未完成时不能声称上述门槛已通过。

新实验只保存weights，不保存完整优化器状态，以减少共享NFS的大量optimizer写入。
阶段1保留2000，阶段2保留2000/4000/6000/8000/10000/final10740；保留6份，不清理前一实验。
代价是中途故障不能精确resume optimizer/RNG；可单独从已有weights重新finetune，但不得伪称无缝恢复。
原trainer在2000边界会周期保存后再final保存同一路径，遵循原实现，只有final日志和进程退出均成功才放行。

部署环境复用项目内CUDA链接和缓存，启动前设置 `HDF5_USE_FILE_LOCKING=FALSE`（仅只读HDF5），
`sidecar_lr_screen=null`规避已有zero-warmup/control元数据兼容问题，HF offline且保留镜像endpoint。
不修改core model、optimizer hook、trainer或原配置。归档mtime/size与已有已验证SHA绑定，部署时复核。

在服务器项目根，检查配置：

```bash
source env_openwam.sh
export EA_SIDECAR_MANIFEST="$PWD/datasets/robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015/openwam_sidecar/aloha50/manifest.ready.json"
.venv/bin/python scripts/supervise_sidecar_c2r_two_stage.py \
  "$PWD/outputs/sidecar/c2r_two_stage8_<UTC>" \
  --predecessor-state "$PWD/outputs/setup/latest_warmstart8_launch.json" --check-only
```

实际运行去掉 `--check-only`，由部署步骤脱离终端启动并留存controller.log；不要重复启动已存在队列。
`status.json`是当前控制器阶段，`stage1/launch.log`和`stage2/launch.log`是训练日志；尚未启动的阶段没有日志。

## 验证与结论边界

本地检查覆盖配置继承和1/4 LR、原策略逐张量不变门槛、完整八rank/final保存门槛、忙卡不抢占，
并复跑已有tiny真实模型的full-forward/backward、全局裁剪、AdamW跳过及2000边界恢复测试。
这些是CPU/配置工程验证，不等于H100新实验启动、两阶段完成或策略收益；闭环评测需独立配对。


本地工程验收：`tests/test_sidecar_c2r_two_stage.py` 与已有 `tests/test_sidecar_backbone_warmup.py` 合计 **12 passed (179.69s)**。部署前核对17个依赖core/入口/父配置文件，本地与aigc24 SHA256全部相同；无需覆盖远端现有core或配置。远端项目共享盘约7957GiB空闲，高于本次独立权重保留的220GiB准入门槛。


## 远端已部署并排队（2026-10-30T02:26:58Z）

五个新增文件同步后逐文件SHA校验通过，远端两份配置CHECK_ONLY通过，官方权重/50任务2500条/min-max映射/磁盘门槛preflight通过。控制器PID2201774存活，状态`waiting_for_existing_training`，等待旧PID1709151；尚未启动新的GPU训练。

队列：`outputs/sidecar/c2r_two_stage8_20261030T022635Z`。
控制器日志：`outputs/setup/c2r_two_stage_controller_c2r_two_stage8_20261030T022635Z.log`。
当前状态：队列内`status.json`；阶段1/2启动后分别查看`stage1/launch.log`、`stage2/launch.log`。
原实验在本次准备期已恢复到20293步，未启停或修改；调度先等待旧实验完整21480步及final，再等八卡释放。新H100首批更新、2000步官方张量冻结验收、第二阶段10740步及闭环收益均未完成，不能据排队通过宣称实验结果。


## 2026-10-30T07:40:49Z 两阶段真实运行验收

阶段1已完成2000 optimizer updates，八rank均2000、torchrun退出0，最终权重：
`/mnt/100T/lixiangyu/cuigm/bitdistill/OpenWAM/outputs/sidecar/c2r_two_stage8_20261030T022635Z/stage1/runs/2026-10-30_11-52-00/checkpoint_step_2000.safetensors`。
接续器`stage1_frozen_policy_verification.json`证明官方2089张量逐张量完全不变，新增140个Sidecar张量；warmup末步ea_backbone_frozen_warmup=1，符合full-backward但原策略不更新的用户合同。
随后显式重新加载本次2000权重并进入stage2；当前新torchrun PID2284976，八rank均1064/10740更新、epoch0，最近100步全部指标有限，均速3.998s/update，八卡利用率100%。当前total0.188114/action0.00152773，日志LR约2.48e-5；约余10.75小时计算时间，另加保存。
阶段1退出仍出现已知NFS Triton autotune锁异常，但退出0且权重保存和冻结验收通过，未按缓存清理告警重启训练。阶段2未完成、不宣称闭环收益；继续既定低LR/5epoch配置并验收最终10740步，不根据单批loss临时调参。
