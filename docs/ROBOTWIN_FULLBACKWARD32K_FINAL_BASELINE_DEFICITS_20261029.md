# Full-backward32k完整100条：低于baseline的条件分析

核对时间：2026-10-29T14:06:59.399589+00:00。本次32k指full-backward warmup2k的checkpoint_step_32000，非2026-10-16初始joint方案同名32k。

最终7019/10000=70.19%；官方69.04%，+1.15pp。Clean88.02% vs89.36%（−1.34pp）；Randomized52.36% vs48.72%（+3.64pp）。100条件中54高于官方、12持平、34低于；34项中17项低至少5pp。

## 所有低于官方发布baseline的条件

### Clean（19项）

|Task|当前100条SR|官方|差值pp|原30条SR|
|---|---:|---:|---:|---:|
|open_microwave|58.00%|82.00%|-24.00|60.00%|
|hanging_mug|31.00%|49.00%|-18.00|30.00%|
|place_can_basket|77.00%|87.00%|-10.00|86.67%|
|place_dual_shoes|80.00%|89.00%|-9.00|73.33%|
|place_mouse_pad|81.00%|90.00%|-9.00|86.67%|
|stamp_seal|80.00%|89.00%|-9.00|90.00%|
|place_object_basket|87.00%|95.00%|-8.00|96.67%|
|move_stapler_pad|68.00%|76.00%|-8.00|80.00%|
|dump_bin_bigbin|93.00%|99.00%|-6.00|93.33%|
|stack_bowls_three|78.00%|83.00%|-5.00|73.33%|
|put_object_cabinet|43.00%|48.00%|-5.00|46.67%|
|blocks_ranking_size|51.00%|55.00%|-4.00|53.33%|
|place_shoe|94.00%|98.00%|-4.00|96.67%|
|move_pillbottle_pad|93.00%|96.00%|-3.00|93.33%|
|place_fan|91.00%|93.00%|-2.00|90.00%|
|scan_object|92.00%|93.00%|-1.00|93.33%|
|lift_pot|99.00%|100.00%|-1.00|96.67%|
|place_phone_stand|84.00%|85.00%|-1.00|83.33%|
|rotate_qrcode|89.00%|90.00%|-1.00|93.33%|

### Randomized（15项）

|Task|当前100条SR|官方|差值pp|原30条SR|
|---|---:|---:|---:|---:|
|handover_block|40.00%|52.00%|-12.00|26.67%|
|place_bread_skillet|52.00%|61.00%|-9.00|60.00%|
|place_cans_plasticbox|88.00%|96.00%|-8.00|73.33%|
|place_burger_fries|72.00%|80.00%|-8.00|73.33%|
|beat_block_hammer|48.00%|55.00%|-7.00|46.67%|
|click_alarmclock|71.00%|77.00%|-6.00|80.00%|
|scan_object|56.00%|60.00%|-4.00|60.00%|
|open_microwave|5.00%|7.00%|-2.00|6.67%|
|stack_blocks_two|87.00%|89.00%|-2.00|86.67%|
|place_dual_shoes|17.00%|19.00%|-2.00|26.67%|
|press_stapler|51.00%|53.00%|-2.00|56.67%|
|put_bottles_dustbin|51.00%|52.00%|-1.00|46.67%|
|shake_bottle_horizontally|92.00%|93.00%|-1.00|86.67%|
|place_mouse_pad|20.00%|21.00%|-1.00|30.00%|
|place_object_scale|37.00%|38.00%|-1.00|30.00%|

## 对照我们训练的baseline10740：仅共同20条件

10740旧报告中的32k列属于旧方案，不能直接挪用。本表用本次完整32k重新join。baseline10740和当前推理/训练预算不同，未固定逐episode语言/scene manifest，不能据此单独归因模块。

|Task|Mode|10740|当前32k|差值pp|
|---|---|---:|---:|---:|
|hanging_mug|demo_clean|49.00%|31.00%|-18.00|
|open_microwave|demo_clean|72.00%|58.00%|-14.00|
|place_burger_fries|demo_randomized|83.00%|72.00%|-11.00|
|place_cans_plasticbox|demo_randomized|99.00%|88.00%|-11.00|
|stamp_seal|demo_clean|88.00%|80.00%|-8.00|
|stamp_seal|demo_randomized|38.00%|30.00%|-8.00|
|beat_block_hammer|demo_randomized|51.00%|48.00%|-3.00|
|handover_block|demo_clean|95.00%|92.00%|-3.00|
|place_bread_skillet|demo_clean|95.00%|92.00%|-3.00|
|beat_block_hammer|demo_clean|100.00%|100.00%|+0.00|
|click_alarmclock|demo_clean|100.00%|100.00%|+0.00|
|place_cans_plasticbox|demo_clean|99.00%|99.00%|+0.00|
|adjust_bottle|demo_clean|96.00%|98.00%|+2.00|
|open_microwave|demo_randomized|3.00%|5.00%|+2.00|
|handover_block|demo_randomized|37.00%|40.00%|+3.00|
|place_burger_fries|demo_clean|97.00%|100.00%|+3.00|
|click_alarmclock|demo_randomized|67.00%|71.00%|+4.00|
|place_bread_skillet|demo_randomized|48.00%|52.00%|+4.00|
|hanging_mug|demo_randomized|10.00%|15.00%|+5.00|
|adjust_bottle|demo_randomized|27.00%|62.00%|+35.00|

共同20条件均值：32k66.65% vs10740为67.70%（−1.05pp）；Clean85.00% vs89.10%（−4.10pp）；Randomized48.30% vs46.30%（+2.00pp）。低于10740共9条件。

## 当前32k补齐100条 vs原30条：另一个比较口径

全100条件原30条71.10%→完整100条70.19%（−0.91pp）；这是同一ckpt增大样本，不是更换checkpoint后退化。前30与补70候选指令池不同，后70不是完全同分布的确定性重放，不能据此单独定位seed抽样/语言影响。

|Task|Mode|原30条|完整100条|差值pp|
|---|---|---:|---:|---:|
|place_object_basket|demo_randomized|63.33%|49.00%|-14.33|
|move_stapler_pad|demo_clean|80.00%|68.00%|-12.00|
|grab_roller|demo_randomized|50.00%|39.00%|-11.00|
|lift_pot|demo_randomized|36.67%|26.00%|-10.67|
|place_mouse_pad|demo_randomized|30.00%|20.00%|-10.00|
|stamp_seal|demo_clean|90.00%|80.00%|-10.00|
|place_can_basket|demo_clean|86.67%|77.00%|-9.67|
|place_object_basket|demo_clean|96.67%|87.00%|-9.67|
|place_dual_shoes|demo_randomized|26.67%|17.00%|-9.67|
|stack_bowls_two|demo_randomized|90.00%|81.00%|-9.00|
|click_alarmclock|demo_randomized|80.00%|71.00%|-9.00|
|place_bread_skillet|demo_randomized|60.00%|52.00%|-8.00|
|move_pillbottle_pad|demo_randomized|50.00%|42.00%|-8.00|
|handover_mic|demo_clean|100.00%|93.00%|-7.00|
|stamp_seal|demo_randomized|36.67%|30.00%|-6.67|
|place_shoe|demo_randomized|63.33%|57.00%|-6.33|
|place_object_scale|demo_clean|100.00%|94.00%|-6.00|
|place_mouse_pad|demo_clean|86.67%|81.00%|-5.67|
|move_can_pot|demo_randomized|46.67%|41.00%|-5.67|
|press_stapler|demo_randomized|56.67%|51.00%|-5.67|

判断：官方差额的主要绝对短板是Clean open_microwave/hanging_mug；Randomized不是整体落后，但交接/容器放置/工具接触有部分负差。任务名只能提出可能涉及门/接触/对齐、交接和物体放置的观察方向，没有失败轨迹分型证据，不把任务类别直接当原因。小幅1–4pp差异仅100条时不宜强归因。后续优先保存失败动作/轻量关键帧做分型，继续matched vanilla与环境/语言口径记录；不以本统计改变标准评测。
