# Warmup系列训练量与RoboTwin评估汇总

快照：2026-10-31 05:37 UTC。所有百分比仅统计完整验收条件。一个条件为一个任务×一个域。

同数据与global batch256下，2148 optimizer updates约为1 epoch，2000更新warmup约0.931 epoch。旧四卡acc2文件名计microsteps（42960=21480更新=10epoch）；新八卡acc1文件名计optimizer updates。表中epoch为数据暴露量的近似换算，不是直接把文件名step视为更新数。官方既有预训练不计入本次新增训练量。

|训练方案|Checkpoint|实际训练量（epoch）|评测覆盖/每条件条数|Clean|Randomized|综合|官方同范围综合|差值pp|
|---|---:|---|---|---:|---:|---:|---:|---:|
|官方发布|官方权重|官方既有训练|100条件/100|89.36%|48.72%|69.04%|69.04%|—|
|原Full-backward warmup2K，连续训练|32000|0.93 warmup +6.52 joint =7.45|100条件/30|89.00%|53.20%|71.10%|69.04%|+2.06|
|同一32000权重，补满100条|32000|同上7.45|100条件/100（30+70）|88.02%|52.36%|70.19%|69.04%|+1.15|
|同一连续训练|36000|0.93 warmup +7.45 joint =8.38|100条件/30|89.33%|51.20%|70.27%|69.04%|+1.23|
|同一连续训练，heads-only分支起点|40000|0.93 warmup +8.38 joint =9.31|未找到独立完整评测|—|—|—|—|—|
|同一连续训练最终权重|42960|0.93 warmup +9.07 joint =10.00|100条件/30|89.93%|50.07%|70.00%|69.04%|+0.96|
|从40000接续，EA辅助监督改heads-only|42960|原9.31 +接续0.69 =总10.00|100条件/30|90.07%|50.67%|70.37%|69.04%|+1.33|
|重新加载warmup2K权重，新优化器/计步后joint10epoch|21480（1030归档）|0.93 warmup +新10.00 joint =10.93|100条件/30|88.67%|61.00%|74.83%|69.04%|+5.79|
|C：warmup2K后WAM LR ramp300，再续至10epoch的计划|无完整最终评测|短训实际至约1.86；后续10epoch长训计划取消|仅离线诊断，未找到全任务SR|—|—|—|—|—|
|PatchWAM增强 +warmup2K +WAM LR ramp100，中间权重|20000|已训练4.66（包含0.93 warmup）|100条件/30|84.20%|55.33%|69.77%|69.04%|+0.73|
|同一ramp100训练，中间权重|24000|已训练5.59（包含0.93 warmup）|56条件/30，中止|83.76%|53.07%|70.06%|66.98%|+3.08|
|官方C2R权重起点，新Sidecar warmup2K，重载后低LR joint5epoch|10740（1031归档）|新增0.93 warmup +5.00 joint =5.93|18条件/30，运行中|84.76%|38.33%|74.44%|73.00%|+1.44|

原连续训练的2K warmup包含在10epoch预算中；1030重载方案的10epoch是warmup之后额外训练。EA heads-only仅限制辅助loss回传到辅助heads，Action/Video任务损失仍更新原联合网络，不能称整个模型只训练heads。

300-update ramp是WAM实际LR从原调度的0.1倍恢复至1倍，不是额外冻结300步，也不是新增300次数据以外的阶段。该C分支短训至4K更新（约1.86epoch）；长训监督在创建子模型前取消，见取消文件。正式增强版本改用100-update ramp（约0.047epoch），不是300-update（约0.140epoch）。其32k曾排队后在仿真前取消，因此没有SR；未找到该版本10epoch最终权重的闭环完整评测。

当前10740只完成18个条件，其中14个Clean、4个Randomized，不能与全任务综合直接排名。在这18个相同条件上，1030的21480为72.59%，当前为74.44%（+1.85pp）。30条与官方100条、候选语言池与未固定的语言RNG存在采样差别；32k100为30+70扩展，heads-only最终含10条件L20/90条件4090。

证据：

- `docs/ROBOTWIN_32K_36K_42960_HEADSONLY_SUMMARY_20261028.md`
- `docs/ROBOTWIN_FULLBACKWARD32K_COMPLETE100_FINAL_RESULTS_20261028.md`
- `docs/ROBOTWIN_SIDECAR_FULL_BACKWARD_WARMUP_TRAINING_20261022.md`
- `docs/SIDECAR_HEADS_ONLY_FROM40000_EXPERIMENT_20261025.md`
- `outputs/1030-warmup2k-to-full10epoch/README.md`
- `docs/PATCHWAM_C2R_AUGMENTATION_20261027.md`
- `outputs/sidecar/aloha50_C_full_from4k_20261027T092219Z/cancelled.json`
- `docs/BITDISTILL_C2R_TWO_STAGE_20261030.md`
- `outputs/robotwin_eval/warmup2k_fromc2r_5epoch_10740_4090_all100_30ep_20261031/comparison.json`
