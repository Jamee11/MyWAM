# 同一32k checkpoint：30条速测与完整100条对比

更新时间：2026-10-29T14:08:42.957841+00:00。仅比较full-backward warmup2k checkpoint_step_32000自身，不与官方或10740 baseline比较。100条包含原30条与续接70条，不是另外独立测试100条。

全100条件：71.10%→70.19%，−0.91pp；Clean89.00%→88.02%，−0.98pp；Randomized53.20%→52.36%，−0.84pp。52条件下降（Clean24/Randomized28）、37上升、11持平；20条件下降至少5pp，其中6条件至少10pp。其余32个下降条件幅度不足5pp。

## 全部52个下降条件

|Task|模式|原30成功/30|原30成功率|完整100成功率|变化pp|
|---|---|---:|---:|---:|---:|
|place_object_basket|demo_randomized|19/30|63.33%|49.00%|-14.33|
|move_stapler_pad|demo_clean|24/30|80.00%|68.00%|-12.00|
|grab_roller|demo_randomized|15/30|50.00%|39.00%|-11.00|
|lift_pot|demo_randomized|11/30|36.67%|26.00%|-10.67|
|place_mouse_pad|demo_randomized|9/30|30.00%|20.00%|-10.00|
|stamp_seal|demo_clean|27/30|90.00%|80.00%|-10.00|
|place_can_basket|demo_clean|26/30|86.67%|77.00%|-9.67|
|place_object_basket|demo_clean|29/30|96.67%|87.00%|-9.67|
|place_dual_shoes|demo_randomized|8/30|26.67%|17.00%|-9.67|
|stack_bowls_two|demo_randomized|27/30|90.00%|81.00%|-9.00|
|click_alarmclock|demo_randomized|24/30|80.00%|71.00%|-9.00|
|place_bread_skillet|demo_randomized|18/30|60.00%|52.00%|-8.00|
|move_pillbottle_pad|demo_randomized|15/30|50.00%|42.00%|-8.00|
|handover_mic|demo_clean|30/30|100.00%|93.00%|-7.00|
|stamp_seal|demo_randomized|11/30|36.67%|30.00%|-6.67|
|place_shoe|demo_randomized|19/30|63.33%|57.00%|-6.33|
|place_object_scale|demo_clean|30/30|100.00%|94.00%|-6.00|
|place_mouse_pad|demo_clean|26/30|86.67%|81.00%|-5.67|
|move_can_pot|demo_randomized|14/30|46.67%|41.00%|-5.67|
|press_stapler|demo_randomized|17/30|56.67%|51.00%|-5.67|
|place_bread_skillet|demo_clean|29/30|96.67%|92.00%|-4.67|
|pick_dual_bottles|demo_randomized|26/30|86.67%|82.00%|-4.67|
|move_playingcard_away|demo_randomized|17/30|56.67%|52.00%|-4.67|
|rotate_qrcode|demo_clean|28/30|93.33%|89.00%|-4.33|
|place_can_basket|demo_randomized|15/30|50.00%|46.00%|-4.00|
|scan_object|demo_randomized|18/30|60.00%|56.00%|-4.00|
|put_object_cabinet|demo_clean|14/30|46.67%|43.00%|-3.67|
|stack_blocks_three|demo_randomized|22/30|73.33%|70.00%|-3.33|
|click_bell|demo_randomized|25/30|83.33%|80.00%|-3.33|
|open_laptop|demo_clean|29/30|96.67%|94.00%|-2.67|
|place_shoe|demo_clean|29/30|96.67%|94.00%|-2.67|
|place_a2b_left|demo_clean|29/30|96.67%|94.00%|-2.67|
|blocks_ranking_size|demo_clean|16/30|53.33%|51.00%|-2.33|
|place_container_plate|demo_randomized|25/30|83.33%|81.00%|-2.33|
|open_microwave|demo_clean|18/30|60.00%|58.00%|-2.00|
|blocks_ranking_rgb|demo_clean|30/30|100.00%|98.00%|-2.00|
|put_object_cabinet|demo_randomized|6/30|20.00%|18.00%|-2.00|
|adjust_bottle|demo_clean|30/30|100.00%|98.00%|-2.00|
|move_stapler_pad|demo_randomized|6/30|20.00%|18.00%|-2.00|
|hanging_mug|demo_randomized|5/30|16.67%|15.00%|-1.67|
|open_microwave|demo_randomized|2/30|6.67%|5.00%|-1.67|
|handover_block|demo_clean|28/30|93.33%|92.00%|-1.33|
|open_laptop|demo_randomized|22/30|73.33%|72.00%|-1.33|
|place_burger_fries|demo_randomized|22/30|73.33%|72.00%|-1.33|
|scan_object|demo_clean|28/30|93.33%|92.00%|-1.33|
|place_cans_plasticbox|demo_clean|30/30|100.00%|99.00%|-1.00|
|place_empty_cup|demo_randomized|24/30|80.00%|79.00%|-1.00|
|move_playingcard_away|demo_clean|30/30|100.00%|99.00%|-1.00|
|place_object_stand|demo_clean|29/30|96.67%|96.00%|-0.67|
|shake_bottle|demo_randomized|28/30|93.33%|93.00%|-0.33|
|dump_bin_bigbin|demo_clean|28/30|93.33%|93.00%|-0.33|
|move_pillbottle_pad|demo_clean|28/30|93.33%|93.00%|-0.33|

解释边界：同一权重增加样本，不能描述为模型训练后退化。原30与补70候选语言池上限不同，未冻结确切指令，不能把所有变化归因于随机抽样；也不能证明性能/通信优化改变了结果。30条每次成功对应3.33pp，扩大样本后有上有下，整体下降0.91pp；明显个别任务需查实际accepted seeds/指令/失败轨迹。
