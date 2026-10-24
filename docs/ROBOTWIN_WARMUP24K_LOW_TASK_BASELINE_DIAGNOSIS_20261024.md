# Warmup24k低分任务：官方权重与baseline10740交叉核对

核对：2026-10-24T05:39:32.386413+00:00。当前full-backward20k仍在独立评估；下表来自已完成的24k/32k与历史baseline，不将新20k partial混入。选择规则：24k相对官方低至少10pp，共11条件；属于事后问题定位，不代表总体均值。

全部单位%。—=未完成该条件复测，不能填旧环境数值冒充新环境。

|Task|Mode|官方发布|官方ckpt旧环境|官方ckpt新环境|自训10740新环境|初始32k|warmup24k|
|---|---|---:|---:|---:|---:|---:|---:|
|open_microwave|demo_clean|82.00|53.00|—|72.00|61.00|23.33|
|beat_block_hammer|demo_randomized|55.00|42.00|—|51.00|33.00|26.67|
|stamp_seal|demo_randomized|29.00|28.00|—|38.00|28.00|10.00|
|stamp_seal|demo_clean|89.00|83.00|—|88.00|73.00|73.33|
|place_mouse_pad|demo_clean|90.00|64.00|—|—|83.00|76.67|
|hanging_mug|demo_clean|49.00|47.00|—|49.00|33.00|36.67|
|put_bottles_dustbin|demo_randomized|52.00|28.00|—|—|37.00|40.00|
|place_bread_skillet|demo_randomized|61.00|25.00|—|48.00|41.00|50.00|
|handover_block|demo_clean|84.00|86.00|85.00|95.00|93.00|73.33|
|place_object_scale|demo_clean|94.00|88.00|—|—|94.00|83.33|
|open_laptop|demo_randomized|57.00|53.00|—|—|57.00|46.67|

主要证据：open_microwave/Clean10740=72 vs24k=23.33（差−48.67pp），beat_block_hammer/Randomized51 vs26.67（−24.33pp），stamp_seal/Randomized38 vs10（−28pp），stamp_seal/Clean88 vs73.33（−14.67pp）。hanging_mug/Clean10740=49与官方相同、24k36.67；handover_block/Clean官方新环境复测85、10740=95、24k73.33。不能将这些结果统一解释为“本地仿真只能支持到24k这个成功率”。相反place_bread_skillet/Randomized：10740=48、24k=50、官方61，共同下降，仍可能有训练/任务/评估共性因素。

官方同一发布权重5任务双域旧44.3→新70.7%，官方该子集72.1%，证明历史旧环境/执行栈问题确有显著影响。并非全部50任务的官方权重新环境复测；本表11条件仅handover_block/Clean具有该新环境复测，其余不可将旧值当新值。9项manifest+1传统native的协议限制保留。

10740完整20条件：67.70%，官方同范围69.85%；32k同范围62.80%；warmup24k在同20条件61.50%。这十任务按32k低分预选，不能外推50任务。24k全100条件68.20%不应与该20条件67.70%直接当同集合比较。

因果边界：10740和32k主要L20仿真，24k当前4090；样本100 vs30、训练预算/stats、compile配置、accepted seed/指令/RNG未配对。证据倾向checkpoint特定任务能力差异，不能单独认定Sidecar结构或warmup导致，也尚未完全排除当前4090配置交互。下一步若需定因，应在当前4090相同native流程固定场景/指令/RNG/accepted manifest，对open_microwave Clean、beat_block_hammer Randomized、stamp_seal Randomized做官方checkpoint/10740/24k配对对照。此次未额外启动这些权重，保持用户当前20k主队列。

GPU1/4资源扩容：开始时分别1路/2路、free10372/10205MiB，用户授权各补一路；改为2路/3路并由独立resource supervisor监控，<2048MiB回退对应cap并只停最新自有新增仿真，保留partial、不混最终SR。原官方任务/预算/seed筛选/视频关闭/模型不变。GPU4过去3路存在cannot create buffer历史，扩容是受监督试验，不能承诺长期满载安全。具体配置和轨迹见本轮gpu14_capacity_trial_current.json及目标目录。

来源：docs/ROBOTWIN_BASELINE_RETEST_ANALYSIS_20261019.md；outputs/robotwin_eval/baseline_official_comparison_20261019/per_condition_comparison.csv；outputs/robotwin_eval/baseline10740_officialcompile_early_20261021/final_results_20261022.json；32k/24k封存final_results.json。全部原始结果保持。


2026-10-24T05:41:09.762356+00:00：扩容已实际生效，GPU1=2、GPU4=3，总10路，新增双域put_bottles_dustbin真实动作推进；当前infra errors=0。2s资源守护低于2GiB或渲染buffer异常自动恢复原cap，只停止新增自有进程。证据`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward20000_4090_all100_30ep_20261024T052041Z/gpu14_capacity_trial_20261024T053746Z/action_acceptance.json`。
