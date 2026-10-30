# Sidecar fullbackward36000：50任务双域×30条趋势评估

更新时间：2026-10-27T10:39:36.234168+00:00。状态：complete。

完整条件：100/100；观察终态：2108成功 / 3000条；当前仿真：0路。

实际执行终态（含中断及重测）：3011条；当前各条件最新attempt覆盖：3000条。

只将完整30条条件计入该条件最终SR；上述混合partial仅用于进度，不能当全任务成功率。计划3000条，未使用100条正式分母。

权重：/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_36000.safetensors；完整Sidecar、cache .99/max3开启、FFN compile关闭。模型L20 本次通过验收的GPU（见deployment.json）；仿真4090；不存视频。

保留官方expert筛seed、unseen、seed0、RT32spp/OIDN、near0.1、原动作预算/成功判据。CUDA和Vulkan渲染PCI逐进程断言，不修改上游源。

调度实际启用的独占policy端口、共享100条件队列，先覆盖全部就绪模型；实际并行受4090显存控制，单路7680MiB+卡余量1024MiB。GPU7未达到门槛暂不放仿真。

输出：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward36000_4090_all100_30ep_20261026T170539Z`；live_progress.json实时剥离ANSI后读取原始日志，state.json用于调度/完整条件；原始client.log保留。

|Task|Mode|状态|成功/观察条数|当前动作步|
|---|---|---|---:|---|
|adjust_bottle|demo_clean|complete|30/30|114/400|
|adjust_bottle|demo_randomized|complete|18/30|123/400|
|beat_block_hammer|demo_randomized|complete|11/30|216/400|
|beat_block_hammer|demo_clean|complete|30/30|110/400|
|click_alarmclock|demo_clean|complete|30/30|70/400|
|click_alarmclock|demo_randomized|complete|25/30|400/400|
|place_shoe|demo_randomized|complete|21/30|230/500|
|place_shoe|demo_clean|complete|29/30|143/500|
|put_bottles_dustbin|demo_clean|complete|27/30|648/1700|
|put_bottles_dustbin|demo_randomized|complete|15/30|1700/1700|
|open_microwave|demo_randomized|complete|3/30|1500/1500|
|open_microwave|demo_clean|complete|20/30|1500/1500|
|blocks_ranking_rgb|demo_clean|complete|29/30|426/1200|
|blocks_ranking_rgb|demo_randomized|complete|17/30|1200/1200|
|blocks_ranking_size|demo_randomized|complete|2/30|1200/1200|
|blocks_ranking_size|demo_clean|complete|15/30|446/1200|
|stack_blocks_three|demo_clean|complete|27/30|1200/1200|
|stack_blocks_three|demo_randomized|complete|16/30|581/1200|
|stack_bowls_three|demo_randomized|complete|23/30|458/1200|
|stack_bowls_three|demo_clean|complete|25/30|432/1200|
|hanging_mug|demo_clean|complete|7/30|900/900|
|hanging_mug|demo_randomized|complete|2/30|900/900|
|stack_bowls_two|demo_randomized|complete|26/30|308/900|
|stack_bowls_two|demo_clean|complete|29/30|302/900|
|handover_block|demo_clean|complete|29/30|283/800|
|handover_block|demo_randomized|complete|10/30|800/800|
|place_cans_plasticbox|demo_randomized|complete|26/30|232/800|
|place_cans_plasticbox|demo_clean|complete|30/30|240/800|
|stack_blocks_two|demo_clean|complete|30/30|285/800|
|stack_blocks_two|demo_randomized|complete|23/30|305/800|
|open_laptop|demo_randomized|complete|20/30|156/700|
|open_laptop|demo_clean|complete|28/30|203/700|
|place_bread_basket|demo_clean|complete|28/30|137/700|
|place_bread_basket|demo_randomized|complete|24/30|383/700|
|place_can_basket|demo_randomized|complete|18/30|259/700|
|place_can_basket|demo_clean|complete|26/30|228/700|
|place_object_basket|demo_clean|complete|26/30|227/700|
|place_object_basket|demo_randomized|complete|14/30|700/700|
|put_object_cabinet|demo_randomized|complete|4/30|262/700|
|put_object_cabinet|demo_clean|complete|14/30|700/700|
|shake_bottle_horizontally|demo_clean|complete|30/30|98/700|
|shake_bottle_horizontally|demo_randomized|complete|27/30|97/700|
|shake_bottle|demo_randomized|complete|26/30|99/700|
|shake_bottle|demo_clean|complete|30/30|101/700|
|dump_bin_bigbin|demo_clean|complete|28/30|107/600|
|dump_bin_bigbin|demo_randomized|complete|21/30|600/600|
|handover_mic|demo_randomized|complete|6/30|200/600|
|handover_mic|demo_clean|complete|29/30|202/600|
|place_dual_shoes|demo_clean|complete|25/30|254/600|
|place_dual_shoes|demo_randomized|complete|7/30|215/600|
|place_bread_skillet|demo_randomized|complete|13/30|500/500|
|place_bread_skillet|demo_clean|complete|27/30|150/500|
|place_burger_fries|demo_clean|complete|30/30|210/500|
|place_burger_fries|demo_randomized|complete|22/30|228/500|
|place_empty_cup|demo_randomized|complete|22/30|500/500|
|place_empty_cup|demo_clean|complete|29/30|146/500|
|scan_object|demo_clean|complete|28/30|161/500|
|scan_object|demo_randomized|complete|15/30|500/500|
|click_bell|demo_randomized|complete|25/30|69/400|
|click_bell|demo_clean|complete|30/30|59/400|
|grab_roller|demo_clean|complete|30/30|90/400|
|grab_roller|demo_randomized|complete|14/30|94/400|
|lift_pot|demo_randomized|complete|11/30|400/400|
|lift_pot|demo_clean|complete|30/30|113/400|
|move_can_pot|demo_clean|complete|30/30|144/400|
|move_can_pot|demo_randomized|complete|18/30|400/400|
|move_pillbottle_pad|demo_randomized|complete|16/30|400/400|
|move_pillbottle_pad|demo_clean|complete|29/30|135/400|
|move_playingcard_away|demo_clean|complete|30/30|97/400|
|move_playingcard_away|demo_randomized|complete|13/30|400/400|
|move_stapler_pad|demo_randomized|complete|2/30|400/400|
|move_stapler_pad|demo_clean|complete|22/30|128/400|
|pick_diverse_bottles|demo_clean|complete|23/30|112/400|
|pick_diverse_bottles|demo_randomized|complete|21/30|400/400|
|pick_dual_bottles|demo_randomized|complete|27/30|113/400|
|pick_dual_bottles|demo_clean|complete|29/30|110/400|
|place_a2b_left|demo_clean|complete|29/30|400/400|
|place_a2b_left|demo_randomized|complete|6/30|137/400|
|place_a2b_right|demo_randomized|complete|11/30|257/400|
|place_a2b_right|demo_clean|complete|25/30|135/400|
|place_container_plate|demo_clean|complete|29/30|134/400|
|place_container_plate|demo_randomized|complete|23/30|400/400|
|place_fan|demo_randomized|complete|13/30|162/400|
|place_fan|demo_clean|complete|28/30|123/400|
|place_mouse_pad|demo_clean|complete|27/30|135/400|
|place_mouse_pad|demo_randomized|complete|6/30|142/400|
|place_object_scale|demo_randomized|complete|11/30|400/400|
|place_object_scale|demo_clean|complete|30/30|125/400|
|place_object_stand|demo_clean|complete|29/30|130/400|
|place_object_stand|demo_randomized|complete|17/30|224/400|
|place_phone_stand|demo_randomized|complete|7/30|400/400|
|place_phone_stand|demo_clean|complete|25/30|112/400|
|press_stapler|demo_clean|complete|30/30|94/400|
|press_stapler|demo_randomized|complete|17/30|400/400|
|rotate_qrcode|demo_randomized|complete|9/30|400/400|
|rotate_qrcode|demo_clean|complete|28/30|159/400|
|stamp_seal|demo_clean|complete|27/30|134/400|
|stamp_seal|demo_randomized|complete|8/30|400/400|
|turn_switch|demo_randomized|complete|16/30|400/400|
|turn_switch|demo_clean|complete|15/30|400/400|
