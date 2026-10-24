# Sidecar 32k附近训练日志：收敛分析

分析时间：2026-10-19T09:42:59.787523+00:00；指标快照截止micro 36030 / optimizer 18015。

结论：训练优化稳定，已进入后期，但32k时Action、Relation、Plan仍有持续改善，不是所有指标均已平台。grounding约3.386的非零值很可能主要来自高斯软标签交叉熵的目标熵；不能将其直接解释为定位没学会，也不能从训练loss推出Randomized泛化已经收敛。

来源：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_C_joint_10epochs_20261016T131709Z/runs/2026-10-16_13-50-52/metrics_rank{0,1,2,3}.jsonl` 与上层 `train.log`。四rank均从micro1连续到36030，无重复/缺失/NaN/Inf；跨rank全部已记录metric最大差为0，代码已先gather求均值，因此不能把四份日志当作四个独立样本。

32k、36k为microstep文件命名，分别对应16000、18000次optimizer更新。计划42960micro（21480更新、10epochs），当前约83.87%，零起始epoch8即第9个epoch。36k完整trainer_state已保存，训练仍进行。

## 同口径窗口统计

各checkpoint列均为此前1000个microstep均值；最后一列是32k窗口到36k窗口的相对变化，不是单batch损失比较。

| Loss | 20k | 32k | 36k | 32k→36k |
|---|---:|---:|---:|---:|
| Total | 0.18931 | 0.1743848 | 0.1712344 | -1.807% |
| Video | 0.09205473 | 0.07945679 | 0.07677628 | -3.374% |
| Action | 0.000709477 | 0.0003217114 | 0.0002616599 | -18.666% |
| EA weighted | 0.09654582 | 0.09460633 | 0.09419641 | -0.433% |
| Ground raw | 3.390111 | 3.386451 | 3.386046 | -0.012% |
| Relation raw | 0.009229039 | 0.002042896 | 0.001164382 | -43.003% |
| Plan raw | 0.06315087 | 0.02836702 | 0.02062866 | -27.279% |

Total与Video后期下降变慢；Action32k→36k仍降18.67%，Relation降43.00%，Plan降27.28%。因此从日志看，继续已有训练计划仍可能改善训练拟合，但这些相对降幅不能直接换算闭环SR增益。不同epoch/批次/噪声与timestep也会影响窗口平均，曲线有周期性变化，不作严格最优点判定。

## 总loss为什么显得接近平

源码显示total≈video+action+EA，EA=0.025×ground+0.01×relation+0.05×plan+0.1×deep，其中deep本身已包含早层ground/relation权重。日志分解最大数值误差7.45e-8，与公式一致。36k weighted EA约0.09420，占total约55%；其非零底部主要来自ground相关项，故总loss看起来下降缓慢时，Action/Plan仍可能显著改善。数值占比不是梯度贡献占比，不能据此直接改loss权重。

## Grounding损失的下界核查

实现为soft-target空间交叉熵 + presence BCE + track-valid BCE。空间CE=H(q)+KL(q||p)，Gaussian目标有非零熵，预测与目标完全一致也不应期待loss归零。WanVAE38空间压缩16、DiT patch2，当前384×320画布为12×10 token grid。

CPU确定性抽样50任务各首/中/末3个episode、每episode20帧，共3000帧、6407个有效role；按现有arm/goal masks与同一canvas Gaussian公式，目标熵均值3.381714、标准差0.410624，接近日志36k ground 3.386046。这支持ground非零值很大部分是目标熵的解释。抽样并非训练batch同权重、未读模型预测，不能直接把两均值相减称精确KL或断言定位已完美；后续应同batch报告CE−H(q)、中心误差/目标区attention mass，并在未训练Clean/Randomized场景验证。

## 稳定性与gate

所有四rank loss与lr/耗时记录有限。更新步总grad_norm中位数0.106950，最近1000micro对应更新步中位数0.060676、最大0.170022；没有零梯度的optimizer更新。非更新micro的grad_norm=0是accumulation2的日志逻辑，不是梯度消失。全程preclip norm>1共202次/约18015更新、最大2.318515；后期没有梯度爆炸趋势。

学习率已从峰值1e-4降到7.8776831e-06。三层read/write gates均已更新，write beta可过零或变号；单看标量大小/符号不表示分支失效或发散，应测实际残差/hidden比例和full/off闭环差异。

## 训练收敛不等于泛化收敛

未发现独立validation loss、held-out episode验证或Randomized验证曲线；trainer中的validation用语是配置检查，不是held-out评估。现有日志能证明训练集拟合改善与优化稳定，不能据此判定过拟合是否发生、最佳泛化checkpoint或Randomized继续训练一定变好。32k闭环随机域偏低应继续做配对cache/Sidecar干预与失败类型分析。

建议先完成已计划10epochs，保留32k/36k/最终权重用于固定协议闭环比较；是否延长训练应由独立验证/闭环结果决定。新增未参与训练的episode做验证，按完整episode划分，避免相邻窗口泄漏；不能事后抽已训练窗口称独立验证。没有足够证据仅因ground plateau停止训练，或仅因Action loss继续下降就延长训练。此轮未修改训练、checkpoint内容或评测进程。

## 图与可复核数据

- [六项loss曲线](evidence/sidecar_training_loss_20261019/loss_curves.png)，100micro均值，Action/Relation/Plan使用对数纵轴；虚线为20k/32k/36k。
- [gate与学习率](evidence/sidecar_training_loss_20261019/gates_and_lr.png)。
- [窗口统计CSV](evidence/sidecar_training_loss_20261019/window_statistics.csv)。
- [审计JSON](evidence/sidecar_training_loss_20261019/summary.json)。
- [目标熵抽样](evidence/sidecar_training_loss_20261019/target_entropy_sample.json)。
- 两个CPU分析脚本同目录，均未加载模型/GPU或写训练原始文件。
