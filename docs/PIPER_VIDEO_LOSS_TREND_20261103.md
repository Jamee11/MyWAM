# Piper Video loss 趋势核查（2026-11-03）

观测时间：2026-11-03T14:35:33.527471+00:00；进度 7572/10976 optimizer updates。只读训练日志及实现；未修改运行中的训练。

证据：`outputs/piper_sidecar/midtrain_patchwam_2epochs_20261102T101245Z/runs/2026-11-02_10-18-00/metrics_rank0.jsonl`；汇总：`outputs/piper_sidecar/midtrain_patchwam_2epochs_20261102T101245Z/analysis_video_loss_20261103.json`。下表为 rank0 训练损失，按完整 optimizer update 包含两次 micro 的均值，不是全局四卡验证集指标。

| 更新区间 | Video | Action | EA（已加权） |
|---|---:|---:|---:|
| 501–1000 | 0.145279 | 0.003333 | 0.055117 |
| 2501–3000 | 0.135987 | 0.001942 | 0.051507 |
| 4989–5488 | 0.127966 | 0.001250 | 0.049957 |
| 5489–5988 | 0.127487 | 0.001100 | 0.049526 |
| 6501–7000 | 0.124635 | 0.000920 | 0.049174 |
| 7001–7500 | 0.122422 | 0.000810 | 0.049044 |

结论：Video loss 从 warmup 后501–1000更新的均值下降到7001–7500更新，下降 15.73%；第二epoch仍下降，未见平台或epoch边界反弹。Action下降更快，但两者数值不能直接用于判断梯度主导性。

计算口径：`openwam/model/architectures/base.py` 每样本随机采样 video FM timestep 和 Gaussian noise，目标为 noise−video latents。Video loss 是潜空间速度预测 MSE，经过 scheduler timestep 权重、有效帧 mask 和样本平均；不是 RGB 重建误差或成功率。记录值已乘 lambda_video，本实验为1。Action目标空间及归一化不同。

解释边界：随机 timestep/noise、不同 batch 使逐步值有波动；PatchWAM 的外观变化可能增加预测难度，但本次没有同条件无增强对照，不能归因。当前曲线不支持“Video分支停止学习”。尚未检查本轮各损失梯度，不能从损失大小判断目标冲突。

下一步：保持本次两epoch训练配方完成，不为压低数字而临时改loss权重。效果判断以固定未训练数据的相同噪声/timestep验证、预测视频以及配对闭环成功率为准；本次未执行这些额外实验。

## 用户提供的 reproduced baseline 日志核查

文件：`outputs/1020-baseline-reproduced/train.log`，同目录 `config.yaml`。该日志只有初始化、调度器和保存事件，没有逐步 Video/Action/total loss，因此不能从这份文件计算 baseline loss 曲线或断言 Piper loss 高于 baseline。最终保存10740已记录，表明这次运行到达最终保存点。

配置显示 baseline 为 Aloha clean_50、Foundation154000初始化、原 joint_self_attn、无辅助监督、普通ColorJitter、5epochs、bs64/rank×4卡×acc1/global256；当前Piper为midtrain56892初始化、Piper Clean+Random、Sidecar、PatchWAM、2epochs、bs32/rank×4卡×acc2/global256。两者初始化、机器人、数据分布、增强和预算不同，不能作Sidecar单变量归因。两者lambda_video/lambda_action均为1，原联合FM训练范式一致，但未验证远端历史源码与当前源码逐行等价。

后续需要原机本次运行目录的 `metrics_rank0.jsonl`（若有），或含 tqdm Video/Action 数值的 Slurm stdout/stderr / 终端完整记录；当前train.log不能替代这些数值证据。当前Piper曲线持续下降的结论仍成立；尚未完成baseline数值对比。

## 更新：通过 ssh-56 tmux 找到历史 loss

远端 Slurm job1259925 的 `.out` 中保存路径、开始时间和最终10740均与baseline_clean50/2026-10-15_21-25-35吻合。真实loss位于 `/home/mingfeihangroup/xiaoming/projects/OpenWAM/slurm-openwam-alpha-baseline-1259925.err`，不在Hydra train.log。`.err`约12.47MB，85934次进度条刷新，按显示step保留最后一次得到0–10740共10741条。进度条仅4位小数且刷新顺序可能使显示step与本批指标存在一格偏移，适合趋势参考，不当作原始高精度metrics。分段每500步均齐全。汇总已保存 `outputs/1020-baseline-reproduced/loss_summary_from_slurm.json`。

| updates | baseline Video | baseline Action |
|---|---:|---:|
| 1–500 | 0.153408 | 0.007473 |
| 501–1000 | 0.136618 | 0.003264 |
| 2501–3000 | 0.115071 | 0.001588 |
| 5001–5500 | 0.100245 | 0.000901 |
| 7001–7500 | 0.092269 | 0.000576 |
| 10241–10740 | 0.086192 | 0.000351 |

修正此前证据缺口：现在已有baseline训练loss趋势。baseline下降确实比当前Piper快、后期值更低；但Aloha仅Clean/5epoch/Foundation初始化/普通CJ，与Piper混合域/2epoch/midtrain初始化/PatchWAM不匹配，不能据此归因为Sidecar拖累Video分支。baseline初500的Video0.1534至末500的0.0862仍明显非零，Video比Action大也不是Sidecar独有。
