# Isolation warmup2K checkpoint24000：复用旧42960诊断

## 六类实验覆盖审计（按用户追问补充）

本轮没有完整复刻用户列出的全部六类实验。此前“诊断完成”仅指已启动的GT Full/Off和32batch梯度，
范围表达不够明确；complete.json也只标注这两个阶段，不代表全六项完成。

| 原实验 | 新24000实际覆盖 | 未完成部分 |
| --- | --- | --- |
| 完整RoboTwin10000条 | 未运行 | 无新全任务成功率，不可沿用32k成绩 |
| 真实写回强度 | 已在25个GT窗口测三tap实际r | 未复刻原18初态、无cache/cache配对动作相对变化及单forward重放 |
| Full/WriteOff配对闭环18对 | 未运行 | 无新Full/Off成功数，不能把旧13/18和14/18套给新模型 |
| Write Strength Sweep 0/.5/1/2/5 | 只做0/1 | .5/2/5未运行；遵循总计划后续“不再做Gate倍率扫描”的收敛决定，不能声称完整复刻 |
| 关系与未来运动质量 | 已完成相同六条专家轨迹、24窗口及训练内控制 | 只报告这组小样本，不外推全任务 |
| 32batch加权梯度 | 已完成相同窗口/任务/阶段/噪声与三loss拆分 | 无Adam恢复或真实参数更新实验，与旧口径一致 |

因此尚不能回答新模型闭环是否更好。优先缺项是同18初态的写回/动作影响和Full/Off配对闭环；
完整10000条属于正式大规模评估，不能把离线诊断等同于它，也不应无说明自动恢复此前已停止队列。
如要求完整复刻原五倍率，应另行明确放弃“不再扫倍率”的缩减范围，并保持相同窗口和噪声。

## 已完成结果：2026-10-23 UTC

**这份24000 isolation checkpoint尚未显示整体变好：相同GT上XYZ/旋转误差更大，夹爪略好；
EA主导有所减轻但仍是数量级差异，直接write仍没有稳定可观的动作收益。**
这不等于已证明warmup方案无效：新模型12000次更新，旧42960训练更久；Sidecar LR与aux系数也不同。

真实结果：25个窗口×Full/Off共50组、32个固定梯度batch均完成；Full重复逐位一致，
所有梯度batch参数未变，EA→Write仍结构性零。模型已正常退出，GPU5/6各1MiB；
GPU0–3训练仍在运行。CPU11项回归通过，未做闭环或新SR实验。
checkpoint SHA256：`c8835677cb59324ce3f2bf8c01e80178e36397ac3379d7bc1ca3d0d6ea23de76`。

### 1. 相同GT的Full动作误差（旧42960 → 新24000）

| 域 | 前8步XYZ mm | 全32步XYZ mm | 全32步XYZ增幅 | 全32步旋转deg | Gripper MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 4.227 → 5.490 | 6.347 → 7.724 | +21.7% | 1.402 → 1.737 | 0.004541 → 0.004239 |
| Randomized | 9.037 → 10.631 | 12.609 → 14.542 | +15.3% | 3.139 → 3.211 | 0.008408 → 0.007970 |

主要执行段前8步也变差，不是仅后24步拖累。夹爪MAE两域略好，但不能抵消位置误差升高。
这些是相对专家未来目标的开环生成误差，不是已执行TCP跟踪误差或闭环成功率。

### 2. WriteOff干预：仍然没有稳定实质收益

数值为Full−Off的全32步XYZ误差，负值表示Full更好，单位mm：

| 域 | 旧42960 | 新24000 |
| --- | ---: | ---: |
| Clean | -0.007695 | 0.007602 |
| Randomized | -0.012664 | -0.005690 |

新模型Clean开write平均反而多0.00760mm误差；Randomized少0.00569mm，
仅微米级且方向不一致。不能据此选更大gate，也不证明write总体有害。
新模型Full相对Off在Clean6/12窗口更好，Randomized7/12更好；每域仍仅3条episode，
多个窗口不是独立样本，未作收益显著性或优化倍率选择。

实际GT窗口write-r（hidden真实BF16加法后的范数比，非beta大小）：

| 域/模型 | tap7 | tap15 | tap23 |
| --- | ---: | ---: | ---: |
| Clean / 旧 | 3.237e-6 | 6.635e-6 | 9.513e-6 |
| Clean / 新 | 4.956e-7 | 6.150e-6 | 3.374e-6 |
| Randomized / 旧 | 2.820e-6 | 6.072e-6 | 9.575e-6 |
| Randomized / 新 | 4.412e-7 | 4.961e-6 | 3.206e-6 |

新模型三tap均未比旧模型增强，尤其首/末层更弱。但更大r本身不是优化目标，
也不能从弱write否定EA辅助监督训练主干的作用。

### 3. EA/Action梯度：有所下降，仍未解决主导

最贴近实际Clean训练分布的16个microbatch，以下为逐batch加权EA/Action norm比的中位数：

| 共享参数组 | 旧42960 | 新24000 |
| --- | ---: | ---: |
| tap7/relational_refinement | 35635 | 27406 |
| tap15/relational_refinement | 38619 | 23020 |
| tap23/relational_refinement | 57451 | 44973 |
| plan_tokens | 81100 | 60341 |

Clean仍约2.3万–6.0万倍；32batch（含Randomized）Refine三层中位数旧约43.5万/25.9万/37.9万，
新约13.3万/6.55万/13.2万，Plan旧75.5万→新30.7万。
新模型aux从1降到.5会直接影响此比例，不能把下降独归warmup。
这是梯度norm比例，不是Adam真实更新量；无optimizer恢复，不宣称已测出训练参数更新主导倍数。
Action到共享/Write路径仍有梯度，EA到Write仍零/None，未发现新断图问题。

局部dAction/ds均值由旧+1.0704e-7变为新−8.8088e-9；新21/32批偏向微小增强，
但Video方向均值+2.1910e-8相反。这是s=1处FM梯度，不是GT轨迹误差导数，
没有因此重新放大gate或扫描更多倍率。

### 4. 最终tap23表征质量

| 域 | 模型 | Contact F1 | Near F1 | Moving F1 | 方向对齐mean | Ground peak-in-box |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 旧42960 | 1.000 | 0.889 | 0.971 | 0.913 | 0.95 |
| Clean | 新24000 | 1.000 | 0.889 | 0.962 | 0.856 | 1.00 |
| Randomized | 旧42960 | 1.000 | 1.000 | 0.981 | 0.941 | 0.90 |
| Randomized | 新24000 | 1.000 | 1.000 | 0.991 | 0.938 | 0.75 |

指标有升有降：Randomized Moving略升，但ground命中率下降；Clean ground略升，但运动方向对齐下降。
不能称为关系表征全面变好，更不能由高Contact F1推出控制成功。

### 验收、解释与下一步

- `complete.json`、GT `results.json`、`gradient32/progress.json`均确认完成；
  `comparison_42960.json`保留逐域、逐组对照及一致性检查。
- 所有旧/新50项样本键、payload与GT路径逐项匹配；seed42、时间步、32梯度输入及噪声seed保持。
- 诊断原progress总数写死5倍率125组，本次修正为实际2倍率50组及完成状态；
  此为进度显示错误，真实results/summary已有50组，不改变任何推理或分母。
- 当前结果不支持宣称这份中期isolation2K模型优于旧最终42960，也不支持“只要warmup就解决EA主导”。
- 优先取得更后期、尽量同optimizer更新预算的checkpoint，复用固定样本再次比较；
  目前不扩网络、不强行增大gate、不叠加新的训练改动。
- 本轮没有新闭环成功率；若后期GT改善，再做独立配对闭环确认，而不是把开环指标当SR。


2026-10-23 UTC。用户要求诊断最早“前2K都训练、但不写回”的版本，
不是后来full-backward/frozen-backbone版本。只做独立诊断，不更新权重或改变现有训练。

## 已核对的训练身份

- 来源：`outputs/1022-sidecar-warmup2k/checkpoint_step_24000.safetensors`。
- config为`isolation_warmup_updates=2000`、Sidecar LR5e-5、aux_loss_scale=.5、ColorJitter开启。
- rank0日志micro4000/optimizer2000仍isolation=1且三beta=0；micro4001起isolation=0，
  micro4002/optimizer2001三beta开始变化。
- checkpoint步数24000是微步，日志对应optimizer12000；不当作24000次参数更新。
- 新旧normalization_stats.npy SHA均为
  `5bc16e47680eda1f3279204fe44537f037bba3407509a8fb2821e1d63cf320f8`。
- 新beta：+4.74453e-5、-2.63214e-4、-1.73569e-4；
  旧42960：-2.03133e-4、+3.79562e-4、-4.78745e-4。
  数值来自safetensors实际标量，不由模型文件名推断。

## 匹配诊断协议

复用`SIDECAR42960_GT_DIAGNOSTICS_20261021.md`与`SIDECAR42960_GRADIENT32_20261021.md`
对应原始样本，不重采、更不根据模型成绩筛选：

1. GT动作/关系：三个任务各Clean/Randomized一条专家轨迹，24个完整32步窗口，
   加一训练内控制窗口，共25。只跑s=0/1（WriteOff/Full），seed42、10步、eager/no-cache。
   记录前8/全部32步XYZ、旋转、夹爪误差、三tap实际write-r及关系预测。
2. 三loss梯度：旧32个B=1窗口、原Action ID100/500/900、Video500及逐批噪声seed，
   复用同一reader/坐标系/mask，按新checkpoint实际aux=.5测加权EA梯度。
   主干不存参数梯度但保留中间输入反传，不建optimizer、不step，逐批检查参数未变。
3. GT只用于评分，不进入部署输入；仍只有当前RGB、文本、proprio/mask。

原工具默认4/7和42960目录，现仅给诊断工具增加plan可选encoder_gpu/model_gpu/scales/
checkpoint_dir，默认旧行为不变；summary按实际倍率汇总，避免为比较重新扫描大倍率。
11项CPU回归通过，包括无多倍率扫描的Full/Off汇总。未改模型/训练核心。

## 执行与产物

输出根：`outputs/robotwin_eval/sidecar_isolation2k_step24000_diagnostics_20261023`。
独立模型PID3627516，CUDA_VISIBLE_DEVICES=5,6；GPU5编码文本，GPU6运行模型和梯度。
A100 GPU0–3训练保持，4/7已有其他作业不操作；没有重启已停止的正式评测或4090任务。
独立hardlink固定24000权重，config/stats/tokenizer快照；权重只读使用，不原地改写。

启动命令（现已启动，不要重复执行）：

```bash
CUDA_VISIBLE_DEVICES=5,6 HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv/bin/python -u \
  outputs/robotwin_eval/sidecar_isolation2k_step24000_diagnostics_20261023/bootstrap.py
```

- `training_audit.json`：日志边界、微步/更新数、实际gate与统计哈希。
- `plan.json`、`inputs_ready.json`：GT协议及固定窗口；`results.json`、`summary.json`：GT/表征结果。
- `gradient32/{plan.json,inputs_ready.json,runtime_contract.json,batch_*.json,summary.json}`：梯度证据。
- `complete.json`：两阶段均正常完成后生成；仅有PID或CPU加载不等于结果完成。

## 启动时边界与后续（历史状态，最终以顶部结果为准）

当前是已启动、模型加载中，尚无新的GT/梯度结论，不能因gate或训练loss变化宣称变好。
与旧42960的训练预算、Sidecar LR、aux系数均不同，只能比较两个checkpoint在相同样本上的
诊断差异，不作warmup单因素因果归因。六episode窗口重叠且每域仅三episode，
GT误差并非闭环SR；本轮尚未进行新Full/Off闭环或50任务评测。

待结果完整后，先比较同域同窗口Full误差、Full相对Off收益和真实write-r，再看EA/Action
梯度比例、方向与结构性零，决定是否值得进一步配对闭环；不因r小或梯度比高直接改网络。
