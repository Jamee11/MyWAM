# Full-backward / frozen-backbone warmup2K：checkpoint12000诊断

对象：`outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_12000.safetensors`。
SHA256：2862651985c9e86a1756a6e91a4ce6ed8d796124a429f6af95d10f4ff0dbcfa7。

## 完成结果：2026-10-23

50组GT生成、32组同输入/RNG Action FM配对、32批梯度全部完成。
Full重复逐位一致，50个输入与前轮完全一致；FM和梯度前后参数未更新。
独立模型已退出，GPU5/6各1MiB；训练PID3501138仍在GPU0–3运行。

核心结论：这个full-backward warmup版本已解决此前“实际写入近乎为零”的现象，
写回对本轮Action有一致性更强的局部正收益；但整体预测精度未全面超过旧checkpoint，
不能等同于闭环成功率提高，也不能作等预算warmup因果归因。

### 原生残差与真实hidden变化

以下24个held-out窗口来自6条专家episode；每tap记录240次前向，不是240次独立rollout。
数值为各次范数或范数比的算术均值，不用均值范数相除代替均值比。
真实比例=||实际加法后的hidden−原hidden||/||原hidden||，包括BF16舍入。

| Block | 原hidden L2 | 投影输出L2 | beta乘后原生残差L2 | 实际delta L2 | 乘后/hidden | 实际delta/hidden | 元素变化率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 310.882 | 226.560 | 1.85269 | 1.83939 | 0.6159% | 0.6118% | 86.64% |
| 15 | 380.261 | 247.719 | 2.44934 | 2.44491 | 0.6542% | 0.6530% | 89.01% |
| 23 | 348.097 | 315.955 | 3.12403 | 3.11913 | 0.9002% | 0.8988% | 90.51% |

hidden RMS分别1.71739/2.10066/1.92298；乘gate后残差RMS为
0.0102348/0.0135308/0.0172580；实际delta RMS为0.0101613/0.0135063/0.0172309。
旧isolation24000真实比例仅0.00004684%/0.00055555%/0.00032897%，
因此新版本不再是“大部分残差低于BF16有效变化”的情况。
projection权重范数只略增，但实际输出范数由旧28.12/31.52/32.23变为226.56/247.72/315.96，
说明变化不只来自beta，也来自adapter输入/输出表征尺度；不能只检查权重L2。
加载后FP32 norm为18.7848/18.9230/19.0309，直接header分析数值有微小归约数值差，
RMS一致，不影响结论。read alpha和write beta见后续参数表。

### Action局部收益与绝对精度分别看

GT全32步XYZ平均误差（mm，越低越好）：

| 模型/条件 | Clean | Randomized |
| --- | ---: | ---: |
| 旧joint42960 Full | 6.3471 | 12.6089 |
| isolation24000 Full | 7.7236 | 14.5418 |
| 本轮12000 WriteOff | 11.3453 | 15.8867 |
| 本轮12000 Full | 11.1658 | 15.6497 |
| 本轮Full−Off | −0.1795 | −0.2369 |

本轮Full相对Off降低XYZ误差约1.58%/1.49%，逐窗口6/12与8/12更好。
前8步Full误差6.8368/8.7570mm，Off6.9372/9.0354mm；Randomized短时
比isolation24000的10.6309mm更好，但全32步仍更差，不能笼统称全部指标改善。
旋转Full Clean3.5976°/Random3.7518°（Off3.5224°/3.7807°）；夹爪MAE
Full0.0058063/0.0093980（Off0.0057865/0.0094215），Clean旋转和夹爪未改善。

32组原训练Action FM loss配对（同prepared输入、同噪声，非部署生成）：

| 子集 | Full loss | WriteOff loss | 相对变化 | Full更好 |
| --- | ---: | ---: | ---: | ---: |
| 全部 | 0.000804685 | 0.000853529 | −5.72% | 29/32 |
| Clean | 0.000624641 | 0.000659966 | −5.35% | 15/16 |
| Randomized | 0.000984729 | 0.001047093 | −5.96% | 14/16 |

旧isolation24000同配对loss整体为Full略差0.037%，本轮写回局部收益明显更一致。
但本轮Full绝对FM loss仍高于isolation的0.000658114；“写回有帮助”不等于
“整个checkpoint已更强”。FM含带噪专家未来视频，不能等同在线仅观察当前帧的闭环效果。

### 关系表征和梯度

最后tap23 Contact F1两域均1；Moving F1两域均0.9714；方向余弦
Clean0.8228/Random0.8941；ground峰值命中0.95/0.90。
方向精度低于isolation24000的0.8555/0.9375和旧42960的0.9130/0.9414；
Randomized定位命中高于isolation的0.75，但不能称表征全面提高。

实际加权EA/Action梯度范数比中位数：

| 子集 | Refine7 | Refine15 | Refine23 | Plan tokens |
| --- | ---: | ---: | ---: | ---: |
| 全部32批 | 492 | 520 | 690 | 943 |
| Clean16批 | 112 | 130 | 176 | 236 |
| Randomized16批 | 2207 | 1914 | 2367 | 3284 |

相比isolation Clean Refine2.3万–4.5万，Action梯度相对辅助目标的弱势明显缩小，
但共享模块仍被EA梯度主导，特别是Randomized。Action梯度可达全部Refine和Write
projection，EA到三个Write projection仍为结构性零。只是即时梯度，不是Adam更新归因。
aux=.5、ground=.025、relation=.01、plan=.05、deep=.1均按实际加载配置验证，
Action/Video外部权重均1；未把内部权重误认为全部最终有效权重。

### 下一步与边界

建议同一训练run在更晚、接近旧模型WAM更新预算的checkpoint重复同协议，观察
0.6%–0.9%写回及Action局部收益是否保持，再做Full/Off同seed配对闭环。
本轮不增加结构或手动放大gate；六条episode、重叠窗口、无独立噪声重复及near差异
仍限制外推。未跑新闭环，没有本checkpoint成功率结论。历史原始结果与训练均保留。

## 训练身份与预算

- 完整恢复状态trainer_state为global_step12000、opt_step6000、epoch2。
- `backbone_warmup_updates=2000`、`isolation_warmup_updates=0`：前2K正常读写/完整反向，
  仅WAM不做参数/Adam/衰减更新，之后joint；不是此前绕过write的isolation版本。
- 因此Sidecar更新6000次，WAM只在后4000次更新；不能与isolation24000的12000次更新作纯单因素对照。
- Sidecar LR5e-5、WAM LR1e-4、aux=.5；内部ground .025、relation .01、plan .05、deep .1。
- 4000微步/2000更新的beta为−.02734/−.03931/−.03735；12000时降至下表数值。
  主干解冻后的Action loss短暂尖峰此前已记录，本轮不改训练策略。
- normalization_stats.npy SHA与旧两模型相同。

## 实际gate与adapter系数

以下直接读safetensors标量和Write projection权重，而不是由配置初值推断。

| Block | 新read alpha | 新write beta | isolation24000 beta | 新projection权重L2 |
| --- | ---: | ---: | ---: | ---: |
| 7 | .05688477 | −.00817871 | +.0000474453 | 18.7842 |
| 15 | .09179688 | −.00988770 | −.0002632141 | 18.9224 |
| 23 | .02514648 | −.00988770 | −.0001735687 | 19.0303 |

write beta绝对值约为isolation的172/38/57倍；read alpha不随之增大。
projection权重L2旧约18.39，新约18.78–19.03，没有同量级放大。
alpha/beta是有符号的特征混合系数，不是概率或实际范数比；负beta本身不说明有害。
实际残差还取决于tokens、attention、projection输出和hidden尺度，必须测forward。

## 配对诊断协议

复用此前相同25GT窗口（24held-out+1训练内控制），seed42、10步、eager/no-cache、
Full与WriteOff各一次，共50组。记录gate相乘的FP32参考与原生dtype残差、实际加法delta、
hidden/projection范数和RMS、数值变化元素比例，以及相对GT的前8/全32步位置、旋转、夹爪误差。

32组Action FM配对复用同prepared输入和CPU/CUDA RNG；Video ID500、Action ID100/500/900，
原loss/mask/权重。再运行同32批的三loss梯度分析，记录共享/Write组及EA结构性零。
不optimizer.step，不写回checkpoint，不启动此前已停止的4090正式评测。

新模型与旧模型不能要求动作逐位相同：只核对50个输入样本完全一致；
新模型自身Full重复仍须逐位一致。已固定checkpoint hardlink、config/stats/tokenizer和源码快照，
保护训练keep-last删除源文件；硬链接只读使用，禁止原地改写。

## 启动记录及产物

输出：`outputs/robotwin_eval/sidecar_full_backward2k_step12000_diagnostics_20261023`。
独立诊断PID3659793，GPU5编码文本/GPU6推理及梯度，GPU0–3训练不操作，4/7其他作业保留。
复用前一轮bootstrap，按plan传checkpoint/更新数、记录adapter参数、允许不同权重输入对照，
可选32梯度阶段；没有修改模型、训练或原诊断结果。新增入口语法检查通过。

- training_audit.json：恢复计数与2K边界、12000步日志快照。
- adapter_checkpoint_comparison.json：新12000、isolation24000、旧42960的alpha/beta/投影范数。
- results.json、summary.json：GT/真实写回；adapter_parameters.json：加载后实际系数。
- paired_action_fm.json：32组Full/Off原训练Action/Video/EA loss。
- gradient32/batch_*.json、summary.json：梯度对照。
- source_snapshot、source_sha256.json、plan.json：代码与样本/权重身份。
- complete.json：全部配置阶段完成后才生成。

启动时模型加载中；现已全部完成，最终结论见文首完成结果。
小样本、窗口重叠、训练预算与数据/相机near差异等边界仍保留，不做纯warmup因果归因。
