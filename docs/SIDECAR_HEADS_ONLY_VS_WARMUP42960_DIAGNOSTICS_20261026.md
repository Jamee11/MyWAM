# Heads-only42960与原full-backward warmup42960诊断比较

更新于2026-10-26T04:09:38.574956+00:00。用户要求优先复用旧方案，简单比较两份最终模型。当前正在执行离线测试，不启动训练或修改生产模型。

## 对象与配对条件

- 原warmup：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_42960.safetensors`。
- Heads-only：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_heads_only_from40000_to10epochs_20261025T171338Z/runs/from40000/checkpoint_step_42960.safetensors`。
- 两者同一40k完整状态起点，heads-only只在最后1480次optimizer更新改变EA heads输入梯度路由；WAM、Read/Refine/Write和Action/Video仍训练。原训练未中断，分支恢复了Adam及scheduler。
- 原10epoch/42960微步/21480更新、bs32acc2/global256、ColorJitter等保持。训练日志按相同step/LR对齐，但恢复后的worker增强/输入未保证逐位相同。

## 40k后训练loss

| 项 | 原训练 | Heads-only | 变化 |
|---|---:|---:|---:|
| Total | 0.12413925 | 0.12398847 | -0.121% |
| Action FM | 0.00022535889 | 0.00022790145 | +1.128% |
| Video FM | 0.076788035 | 0.076643598 | -0.188% |
| 加权EA | 0.047125854 | 0.04711697 | -0.019% |
| Relation | 0.0015361864 | 0.0015660968 | +1.947% |
| Plan | 0.020653813 | 0.020290732 | -1.758% |

共同40001–42960，共2960微步；optimizer计数完全一致，学习率最大差0，全部loss有限。最后500微步Action仍+0.980%，Video−0.537%，EA−0.040%。没有持续、明显改善Action训练目标的证据；不把这些未严格输入配对的日志差当成动作精度或成功率。

## 写回gate与adapter

| Block | 原Write beta | Heads-only beta | 绝对值变化 |
|---|---:|---:|---:|
| 7 | -0.002868652 | -0.002838135 | -1.064% |
| 15 | -0.002868652 | -0.002853394 | -0.532% |
| 23 | -0.003021240 | -0.003005981 | -0.505% |

Read alpha前两层相同，第三层从.0334473到.0332031。beta不代表真实write-r或控制收益。

相对权重变化定义为||W42960−W40000||/||W40000||，不是范数差。

| 权重 | 原B7/15/23 (%) | Heads-only B7/15/23 (%) |
|---|---|---|
| Read projector | 0.3695 / 0.3634 / 0.4362 | 0.2656 / 0.2097 / 0.1211 |
| Write output | 0.1625 / 0.1716 / 0.1597 | 0.1690 / 0.1662 / 0.1685 |
| Write query | 0.0537 / 0.1037 / 0.1345 | 0.0542 / 0.0965 / 0.1278 |

共享Read投影在heads-only中移动较少，尤其第三层；Write输出映射的移动量仍约.16–.17%，没有大幅增加。两份最终映射的相对L2差约.16–.44%（Read/Write output），不能以范数接近认定没有学习，也不能由这些变化推出实际残差更强。

## 复用既有离线诊断

- 使用原8k/12k/16k/20k诊断的同一25个GT输入：Clean heldout12、Randomized heldout12，加1个训练见过的重复检查，不混进heldout均值；来自adjust_bottle/place_shoe/beat_block_hammer的六条专家episode。
- 固定seed42、10步去噪、32动作chunk、eager/no-cache，Full与Write Off两种模式；两模型各50次生成。探针原位记录三tap实际hidden差、原生残差、改变元素比例及分位数。
- 复用32个固定FM样本（Clean16/Random16，接触阶段和action timestep覆盖），同prepared输入/RNG，Full/Off两次，测实际加权Action/Video/EA目标；全程inference_mode，无optimizer。
- 统计相对专家EEF的位置mm、SO(3)旋转角和gripper MAE，分别看前8/完整32步；比较Full/Off直接作用及两模型Full/Full差异，同时复核关系和计划指标。
- 模型分别部署本机空闲GPU5/6；训练GPU0–3保留，其他作业不终止。只复用旧bootstrap/探针/分析器，新增的调用与汇总文件保存在独立诊断输出。
- 输出：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_heads_only_vs_warmup42960_20261026T040447Z`。输入与源码SHA、两权重SHA、原始逐窗口动作和测量均独立保存。

这是六条轨迹的离线小样本，窗口重叠且不是独立试验；不能给出闭环成功率或统计显著性，不与当前cache开启的跨机正式评估混算。初步优劣须等待这轮配对结果。

## 2026-10-26T04:18:19.298519+00:00 离线诊断已完成

**结论：晚期heads-only没有明显增强写回；Randomized位置/旋转有小幅改善，Clean位置和两域夹爪略差，不能认定全面优于原warmup。** 原始50次生成+32FM对两模型均完整；重复推理逐位一致、Write Off真实delta为零、Sidecar参数及原始输入SHA不变。两份完成标记齐全、进程已退出并释放GPU5/6，训练0–3继续。

### 真实写回与动作敏感度

下表mean按24个heldout窗口×10去噪时刻汇总，排除1个训练见过的检查样本。ratio=||H_after−H_before||/||H_before||，不是beta。

| Block | 原真实ratio (%) | Heads-only (%) | 相对变化 |
|---|---:|---:|---:|
| 7 | 0.1848 | 0.1907 | +3.19% |
| 15 | 0.1645 | 0.1674 | +1.80% |
| 23 | 0.2639 | 0.2627 | -0.45% |

两版均约0.16%–0.26%，明显不是最早无此warmup版本的1e-6量级。前两层仅略增、第三层略降，没有数量级增强。原生gated residual与实际hidden差的均值接近，不能把效果弱简单归因于BF16把残差整体舍入消失；实际改变元素比例约65%–72%。这不代表写回已提供足够控制收益。

| 域 | 原Full/Off动作相对变化 (%) | Heads-only (%) | 原/新XYZ动作差 (mm) |
|---|---:|---:|---|
| Clean | 0.3791 | 0.3912 | 1.2353 / 1.2515 |
| Randomized | 0.3426 | 0.3403 | 0.9991 / 0.9571 |

归一化相对Action差覆盖有效20D映射，XYZ差为Full与Off输出的距离，**不是**相对专家的误差改善量。写回确实改变动作，强度两者接近。

### Full模型相对专家动作的误差

| 域/范围 | 项 | 原warmup | Heads-only | 新−原 |
|---|---|---:|---:|---:|
| Clean/8步 | XYZ mm | 4.448951 | 4.611182 | +0.162231 |
| Clean/8步 | SO(3) deg | 0.587710 | 0.579016 | -0.008694 |
| Clean/8步 | Gripper MAE | 0.001866 | 0.001969 | +0.000103 |
| Clean/32步 | XYZ mm | 7.041945 | 7.087876 | +0.045931 |
| Clean/32步 | SO(3) deg | 1.361374 | 1.338497 | -0.022877 |
| Clean/32步 | Gripper MAE | 0.004300 | 0.004377 | +0.000077 |
| Randomized/8步 | XYZ mm | 10.750844 | 10.525326 | -0.225518 |
| Randomized/8步 | SO(3) deg | 2.660307 | 2.591475 | -0.068832 |
| Randomized/8步 | Gripper MAE | 0.011225 | 0.011510 | +0.000285 |
| Randomized/32步 | XYZ mm | 14.915013 | 14.727107 | -0.187906 |
| Randomized/32步 | SO(3) deg | 3.674472 | 3.588007 | -0.086465 |
| Randomized/32步 | Gripper MAE | 0.009614 | 0.009858 | +0.000244 |

Randomized32步XYZ平均改善0.188mm（约1.26%），Clean略差0.046mm（约0.65%）；Randomized前8步改善0.226mm（约2.10%），Clean前8步差0.162mm（约3.65%）。两域32步旋转均略好，两域gripper MAE均略差。32步XYZ在两域均各7/12窗口更好，但重叠窗口不是独立episode，不能当作成功率。

按任务32步XYZ（原→新，mm）：

| 任务 | Clean | Randomized |
|---|---|---|
| adjust_bottle | 5.5416 → 5.4192 | 10.8579 → 10.8082 |
| place_shoe | 10.2923 → 10.5269 | 18.8541 → 18.3869 |
| beat_block_hammer | 5.2920 → 5.3175 | 15.0330 → 14.9861 |

三任务Randomized均略好，主要均值改善来自place_shoe；Clean中place_shoe与hammer略差。

### 写回是否帮助动作，而不只是改变动作

Full−Off的专家XYZ误差差值，负值表示开启Write后更接近GT：

| 域/范围 | 原Full−Off (mm) | Heads-only Full−Off (mm) |
|---|---:|---:|
| Clean/8步 | -0.229298 | -0.187625 |
| Clean/32步 | -0.158272 | -0.151335 |
| Randomized/8步 | +0.059495 | -0.118290 |
| Randomized/32步 | +0.026246 | -0.043011 |

两版Clean的Write均有小幅正作用。Randomized原版XYZ的Write小幅负作用，heads-only转为小幅正作用；但Randomized旋转两版的Write仍分别增加约.0380/.0396度误差，不能称写回全面改善。

Randomized中关闭Write时，两模型XYZ仍是14.8888→14.7701mm，已有约.119mm差异。因此Full改善约.188mm并非全由Write造成；主干参数变化也参与，不能把收益归因于gate增大。两模型Full输出本身平均XYZ距离Clean1.201mm、Random1.425mm，仍大于其对GT误差的净改善。

### 固定32个样本的Action FM

| 域 | 模式 | 原FM | Heads-only FM | 新相对变化 | 新更优批数 |
|---|---|---:|---:|---:|---|
| Clean | full | 0.00035557 | 0.00038142 | +7.269% | 6/16 |
| Clean | write_off | 0.00035920 | 0.00038367 | +6.812% | 6/16 |
| Randomized | full | 0.00048212 | 0.00046637 | -3.267% | 6/16 |
| Randomized | write_off | 0.00047819 | 0.00046610 | -2.528% | 4/16 |

这是实际加权Action flow-matching目标，噪声输入已配对，比训练日志更适合此处比较。Clean均值差约7.27%，Randomized均值好约3.27%，但两域均只有6/16样本更优，均值可能受少量高误差样本影响；全32平均新比原高约1.21%。不能说Action目标全面改善。

### 表征是否被detach破坏

| 指标/末层 | 原Clean→新Clean | 原Randomized→新Randomized |
|---|---|---|
| Contact F1 | 1.000000 → 1.000000 | 1.000000 → 1.000000 |
| Moving F1 | 0.962264 → 0.962264 | 0.971429 → 0.971429 |
| 方向cos | 0.814237 → 0.815885 | 0.884619 → 0.885667 |
| Ground峰值命中率 | 0.950000 → 0.950000 | 0.950000 → 0.950000 |
| Presence accuracy | 1.000000 → 1.000000 | 0.541667 → 0.541667 |
| Track accuracy | 1.000000 → 1.000000 | 0.500000 → 0.500000 |

晚期detach没有在这些样本上造成明显语义遗忘；Contact/Moving等基本保持。但按现有GT可见性/有效性mask定义，Randomized Presence约54.2%、Track50%，两版共同存在大量误报。Contact F1=1只在其有效GT mask内统计，不能代表全部物体/关系都可靠，也不能把该共同短板归于heads-only。

### 判断与下一步

1. 本轮hard heads-only主要改变了共享Read优化方向，并未把Write作用显著放大。暂不建议因本结果放大gate或再改网络。
2. Randomized位置/旋转及配对FM有弱正向信号，但Clean和gripper存在代价，整体接近持平。优先由同场景、同协议闭环决定实际优劣，不能从离线误差换算SR。
3. 这是40k后1480次低LR更新的晚期硬路由试验，不足以判定已启动的全程rho1→.01退火策略；现有完整实验继续，保留两版最终checkpoint作为比较。
4. 若后续仍有同类Randomized失败，应检查可见性/有效性预测误报及实际是否影响Write，而不只看Contact/Moving漂亮指标；这次不扩诊断网格或修改训练。

完整指标：`outputs/robotwin_eval/sidecar_heads_only_vs_warmup42960_20261026T040447Z/comparison.json`；逐窗口动作差、32FM、写回分位数和原始测量都保留。完整验收：`analysis_acceptance.json`。

图：[离线对照PNG](../outputs/robotwin_eval/sidecar_heads_only_vs_warmup42960_20261026T040447Z/comparison.png) / [PDF](../outputs/robotwin_eval/sidecar_heads_only_vs_warmup42960_20261026T040447Z/comparison.pdf)；[40k后loss曲线](../outputs/sidecar/aloha50_heads_only_from40000_to10epochs_20261025T171338Z/trend_analysis_20261026/final/loss_trends.png)。
