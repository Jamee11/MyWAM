# Baseline10740十任务双域最终结果

核对时间：2026-10-22T14:17:02.241914+00:00

20条件各100条、2000条全部完整通过原生run.json和退出码核验，无errors。仅为固定选定的十任务，不是全50任务结果。

|Setting|Baseline|官方同范围|差值pp|32k同范围|
|---|---:|---:|---:|---:|
|demo_clean|89.10%|87.90%|+1.20|84.50%|
|demo_randomized|46.30%|51.80%|-5.50|41.10%|
|all|67.70%|69.85%|-2.15|62.80%|

|Task|Setting|Baseline|官方|32k|
|---|---|---:|---:|---:|
|adjust_bottle|demo_clean|96%|90%|97%|
|adjust_bottle|demo_randomized|27%|51%|37%|
|beat_block_hammer|demo_clean|100%|99%|98%|
|beat_block_hammer|demo_randomized|51%|55%|33%|
|click_alarmclock|demo_clean|100%|99%|98%|
|click_alarmclock|demo_randomized|67%|77%|65%|
|handover_block|demo_clean|95%|84%|93%|
|handover_block|demo_randomized|37%|52%|40%|
|hanging_mug|demo_clean|49%|49%|33%|
|hanging_mug|demo_randomized|10%|10%|8%|
|open_microwave|demo_clean|72%|82%|61%|
|open_microwave|demo_randomized|3%|7%|3%|
|place_bread_skillet|demo_clean|95%|92%|94%|
|place_bread_skillet|demo_randomized|48%|61%|41%|
|place_burger_fries|demo_clean|97%|97%|99%|
|place_burger_fries|demo_randomized|83%|80%|70%|
|place_cans_plasticbox|demo_clean|99%|98%|99%|
|place_cans_plasticbox|demo_randomized|99%|96%|86%|
|stamp_seal|demo_clean|88%|89%|73%|
|stamp_seal|demo_randomized|38%|29%|28%|

Baseline使用自身stats/EEF合同、官方完整compile+cache；RT/专家筛seed/预算/成功判据沿原批次，后续任务关闭视频。训练预算和统计不同，十任务由32k相对官方负差预选，不能把该均值外推全50或单独归因模块。10月22日03:48组会快照保留19/20状态，不回写历史快照；本报告为全部完成后的更新。
