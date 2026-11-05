# 19组诊断＋近期完整速测＋baseline汇总

快照：2026-11-04T02:14:10.552951+00:00

## 对照定义

- 官方参考：OpenWAM-Alpha-Sim-RoboTwin-Clean2Random，50任务×双域×100条，Clean89.36/Randomized48.72/综合69.04%。本次重新在线核对官方Issue46表。
- Original Full21480：本项目完整Sidecar，原74.83%模型，是19组诊断与Refine-Off的对照，不能称作无Sidecar baseline。
- 本地原版baseline10740：不带Sidecar，只有预选10任务×双域×100条，综合67.70%，不能与50任务全套平均直接混排。

## 1. 最近完整50任务速测

全部本地行：50tasks×Clean/Randomized×30=3000条。差值均为百分点。官方每条件100条，故是发布参考，不是同seed同语言的配对因果对照。

|实验|Checkpoint|Clean|Randomized|综合|对原Full21480|对官方参考|
|---|---|---:|---:|---:|---:|---:|
|Official OpenWAM-C2R|发布权重|89.36%|48.72%|69.04%|−5.79|—|
|Original Full21480|21480|88.67%|61.00%|74.83%|+0.00|+5.79|
|C2R-init joint5epoch|10740|89.13%|61.13%|75.13%|+0.30|+6.09|
|XA21480 corrected|21480|89.00%|56.07%|72.53%|-2.30|+3.49|
|Refine-Off21480|21480|89.40%|62.00%|75.70%|+0.87|+6.66|
|PatchWAM ramp100 final42960|42960 micro=21480 optimizer|87.87%|64.53%|76.20%|+1.37|+7.16|

XA采用渲染异常条件完整重测11/30后的修正成绩2176/3000=72.5333%，不是旧72.60%。旧异常13/30与原始文件保留，汇总仅叠加经验证的新结果。

训练量与起点：

- Original Full21480：原warmup2K权重重载，额外joint10epoch。
- C2R-init joint5epoch：官方C2R初始化，warmup2K后额外joint5epoch。
- XA21480 corrected：Action alignment；训练增强也变化。
- Refine-Off21480：同原Full权重，推理关闭三tap Refine。
- PatchWAM ramp100 final42960：Foundation起点，总10epoch内含2K冻结，WAM ramp100。

42960是microstep，等于21480 optimizer updates；它的2K冻结包含在总10epoch中。原1030的21480是optimizer updates，表示重载warmup后再额外10epoch。两者并非完全匹配的训练消融，原Full也使用PatchWAM，不能把差值全部归因于“新增PatchWAM增强”。C2R5epoch起点/LR不同，XA还改了增强；官方比较不等价matched no-Sidecar训练。

## 2. 19组固定manifest诊断

同一原Full21480权重，10个选定任务×双域×10=每组200条，19组共3800条；最终seed/指令/原生退出与渲染日志审计通过。本表只与同次Full77%比较，不能直接减官方全50任务69.04%。

|模式|Clean|Randomized|综合|对同次Full|Randomized差值|
|---|---:|---:|---:|---:|---:|
|full|92.0%|62.0%|77.0%|+0.0|+0.0|
|write_off|91.0%|69.0%|80.0%|+3.0|+7.0|
|grounding_reset|92.0%|68.0%|80.0%|+3.0|+6.0|
|relation_reset|93.0%|70.0%|81.5%|+4.5|+8.0|
|summary_reset|90.0%|71.0%|80.5%|+3.5|+9.0|
|plan_reset|93.0%|68.0%|80.5%|+3.5|+6.0|
|all_tokens_reset|91.0%|66.0%|78.5%|+1.5|+4.0|
|visual_read_off|90.0%|66.0%|78.0%|+1.0|+4.0|
|refine_off|93.0%|69.0%|81.0%|+4.0|+7.0|
|sidecar_text_off|93.0%|68.0%|80.5%|+3.5|+6.0|
|sidecar_proprio_off|91.0%|67.0%|79.0%|+2.0|+5.0|
|write_tap7_off|91.0%|66.0%|78.5%|+1.5|+4.0|
|write_tap15_off|93.0%|68.0%|80.5%|+3.5|+6.0|
|write_tap23_off|91.0%|68.0%|79.5%|+2.5|+6.0|
|role_embedding_off|88.0%|66.0%|77.0%|+0.0|+4.0|
|ffn_off|92.0%|68.0%|80.0%|+3.0|+6.0|
|position_off|91.0%|65.0%|78.0%|+1.0|+3.0|
|stage_context_off|90.0%|65.0%|77.5%|+0.5|+3.0|
|sidecar_off|92.0%|63.0%|77.5%|+0.5|+1.0|

Sidecar-Off77.5%并列第16（15组更高、1组相同、2组更低）。Relation-Reset81.5%为这组最高，Refine-Off81%第二。All-Reset78.5%并不等价于完整关闭；LN/MHA bias和训练后非零Write gate留下常量隐空间残差。各mode详细实现见ROBOTWIN_19ARM_IMPLEMENTATION_AUDIT_20261102.md。

诊断任务：adjust_bottle、beat_block_hammer、handover_block、place_bread_skillet、place_burger_fries、put_bottles_dustbin、place_object_basket、place_cans_plasticbox、stack_blocks_three、open_microwave。

## 3. 本地原版baseline的同任务比较

固定原baseline的10任务×双域20条件。此表只统一任务/域，旧baseline每条件100、其他每条件30；旧baseline使用完整compile+cache/自有stats，训练初始化/预算不同，因此不能声称纯Sidecar架构因果增益。

|实验|每条件条数|Clean|Randomized|综合|对本地baseline|
|---|---:|---:|---:|---:|---:|
|Local baseline10740 (100ep)|100|89.10%|46.30%|67.70%|-0.00|
|Official same10tasks (100ep)|100|87.90%|51.80%|69.85%|+2.15|
|Original Full21480|30|85.67%|60.00%|72.83%|+5.13|
|C2R-init joint5epoch|30|86.67%|62.00%|74.33%|+6.63|
|XA21480 corrected|30|85.33%|53.00%|69.17%|+1.47|
|Refine-Off21480|30|83.67%|59.67%|71.67%|+3.97|
|PatchWAM ramp100 final42960|30|84.33%|61.33%|72.83%|+5.13|

原baseline任务：adjust_bottle、beat_block_hammer、click_alarmclock、handover_block、hanging_mug、open_microwave、place_bread_skillet、place_burger_fries、place_cans_plasticbox、stamp_seal。与上面的诊断10任务集合不同。

## 4. 新42960补至100条的阶段结果

- all：71条件，5517/7100=77.7042%；同subset原30条77.6995%。
- demo_clean：39条件，3486/3900=89.3846%；同subset原30条88.8889%。
- demo_randomized：32条件，2031/3200=63.4688%；同subset原30条64.0625%。

此时71/100条件已合并100条，尚非全套10000最终平均；综合77.7042%对同条件速测77.6995%，基本持平。不要把77.70%直接对全套速测76.20%宣称扩样提分，覆盖不同。原Full21480的68条件补测等权替换估计74.94%同样不是完整100条全套结果。

## 5. 结论与下一步

1. 当前完整速测最高为PatchWAM最终42960：76.20%，相对原Full+1.37pp；主要来自Randomized+3.53pp，Clean−0.80pp。比Refine-Off高0.50pp（15/3000），没有独立重复时不能称稳定显著优势。
2. Refine-Off是同一权重的直接推理干预，完整速测75.70%对Full74.83%，+0.87pp；比10任务诊断+4pp收窄。说明自由Refine存在改善空间，尚非全局有害证明。
3. 完整Sidecar-Off只净增1/200，不能据此推翻训练期表征价值；关闭分支仍用joint-trained主干。Write-Off与完整Off路径及闭环结果不同，现有两fixture动作exact不能外推全状态。
4. XA修正72.53%对原Full−2.30pp，主要Randomized−4.93pp；考虑增强/teacher等变化，不能只归因alignment目标本身，但现有配置不适合成为主结果。
5. C2R-init joint5epoch75.13%与原Full74.83%相差0.30pp；初始化/训练预算同时变化，证据支持效果接近。
6. 完成当前42960完整100条后优先与原32k完整100及原21480已补条件做同任务100vs100；保留正在运行的Grounding-guided Read和无Sidecar PatchWAM训练对照，不再叠加多个新长训练变量。
7. 19组中18个干预有16个高于Full、2个持平，单次开发样本及从多种干预中选择最佳的偏差必须考虑；同Full重复闭环/no-op对照尚未建立，不能把这种方向一致性自动解释为全部模块均有害。
8. 在本地baseline的10任务子集上，新42960与原Full都72.83%，Refine-Off71.67%反而比原Full低1.17pp；全套正收益不是普遍逐任务改善。
9. 论文可先展示全套结果与诊断，但需明确辅助训练收益和在线依赖不同。完整方案、同数据预算的无Sidecar训练、聚焦Guidance-Off/Write-Off才是后续机制证据；不追求19种每项runtime ablation均掉分。

## 6. 来源

- 官方：https://github.com/OpenWAM-Official/OpenWAM/issues/46#issuecomment-6118279946
- 19组：outputs/robotwin_eval/sidecar21480_19arms_complete10_20261102/merged10_results.json；final_independent_audit.json
- 各速测、修正和本地baseline的绝对路径见本次snapshot.json；CSV附表同目录。
- 正在运行的新42960补70：outputs/robotwin_eval/patchwam_ramp100_final42960_supp70_20261103/comparison.json。

