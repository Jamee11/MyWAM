# Sidecar full_backward20000：50任务双域×30条趋势评估

更新时间：2026-10-24T14:46:10.105258+00:00。状态：interrupted。

完整条件：35/100；观察终态：822成功 / 1203条；当前仿真：0路。

实际执行终态（含中断及重测）：1203条；当前各条件最新attempt覆盖：1203条。

只将完整30条条件计入该条件最终SR；上述混合partial仅用于进度，不能当全任务成功率。计划3000条，未使用100条正式分母。

权重：/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_20000.safetensors；完整Sidecar、cache .99/max3开启、FFN compile关闭。模型L20 GPU0–6；仿真4090；不存视频。

保留官方expert筛seed、unseen、seed0、RT32spp/OIDN、near0.1、原动作预算/成功判据。CUDA和Vulkan渲染PCI逐进程断言，不修改上游源。

调度21个独占policy端口、共享100条件队列，先覆盖七份不同模型；实际并行受4090显存控制，单路7680MiB+卡余量1024MiB。GPU7未达到门槛暂不放仿真。

输出：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward20000_4090_all100_30ep_20261024T052041Z`；live_progress.json实时剥离ANSI后读取原始日志，state.json用于调度/完整条件；原始client.log保留。

|Task|Mode|状态|成功/观察条数|当前动作步|
|---|---|---|---:|---|
|adjust_bottle|demo_clean|complete|30/30|120/400|
|adjust_bottle|demo_randomized|complete|16/30|126/400|
|beat_block_hammer|demo_randomized|complete|8/30|400/400|
|beat_block_hammer|demo_clean|complete|30/30|113/400|
|click_alarmclock|demo_clean|complete|30/30|110/400|
|click_alarmclock|demo_randomized|complete|25/30|80/400|
|place_shoe|demo_randomized|complete|22/30|196/500|
|place_shoe|demo_clean|complete|28/30|145/500|
|put_bottles_dustbin|demo_clean|complete|23/30|1700/1700|
|put_bottles_dustbin|demo_randomized|complete|8/30|500/1700|
|open_microwave|demo_randomized|complete|0/30|1500/1500|
|open_microwave|demo_clean|complete|11/30|1187/1500|
|blocks_ranking_rgb|demo_clean|complete|27/30|425/1200|
|blocks_ranking_rgb|demo_randomized|complete|15/30|1200/1200|
|blocks_ranking_size|demo_randomized|complete|4/30|1200/1200|
|blocks_ranking_size|demo_clean|complete|13/30|1200/1200|
|stack_blocks_three|demo_clean|complete|28/30|1200/1200|
|stack_blocks_three|demo_randomized|complete|15/30|436/1200|
|stack_bowls_three|demo_randomized|complete|21/30|448/1200|
|stack_bowls_three|demo_clean|complete|24/30|454/1200|
|hanging_mug|demo_clean|complete|8/30|900/900|
|hanging_mug|demo_randomized|complete|6/30|900/900|
|stack_bowls_two|demo_randomized|complete|28/30|481/900|
|stack_bowls_two|demo_clean|complete|29/30|304/900|
|handover_block|demo_clean|complete|26/30|254/800|
|handover_block|demo_randomized|complete|7/30|800/800|
|place_cans_plasticbox|demo_randomized|complete|27/30|234/800|
|place_cans_plasticbox|demo_clean|complete|30/30|229/800|
|stack_blocks_two|demo_clean|complete|29/30|296/800|
|stack_blocks_two|demo_randomized|complete|27/30|294/800|
|open_laptop|demo_randomized|complete|18/30|700/700|
|open_laptop|demo_clean|complete|27/30|354/700|
|place_bread_basket|demo_clean|complete|28/30|226/700|
|place_bread_basket|demo_randomized|complete|20/30|221/700|
|place_can_basket|demo_randomized|interrupted|14/24|231/700|
|place_can_basket|demo_clean|interrupted|9/20|87/700|
|place_object_basket|demo_clean|interrupted|26/29|119/700|
|place_object_basket|demo_randomized|interrupted|8/14|154/700|
|put_object_cabinet|demo_randomized|interrupted|0/6|118/700|
|put_object_cabinet|demo_clean|interrupted|3/12|235/700|
|shake_bottle_horizontally|demo_clean|complete|30/30|99/700|
|shake_bottle_horizontally|demo_randomized|interrupted|25/27|6/700|
|shake_bottle|demo_randomized|interrupted|19/21|124/700|
|shake_bottle|demo_clean|queued|—||
|dump_bin_bigbin|demo_clean|queued|—||
|dump_bin_bigbin|demo_randomized|queued|—||
|handover_mic|demo_randomized|queued|—||
|handover_mic|demo_clean|queued|—||
|place_dual_shoes|demo_clean|queued|—||
|place_dual_shoes|demo_randomized|queued|—||
|place_bread_skillet|demo_randomized|queued|—||
|place_bread_skillet|demo_clean|queued|—||
|place_burger_fries|demo_clean|queued|—||
|place_burger_fries|demo_randomized|queued|—||
|place_empty_cup|demo_randomized|queued|—||
|place_empty_cup|demo_clean|queued|—||
|scan_object|demo_clean|queued|—||
|scan_object|demo_randomized|queued|—||
|click_bell|demo_randomized|queued|—||
|click_bell|demo_clean|queued|—||
|grab_roller|demo_clean|queued|—||
|grab_roller|demo_randomized|queued|—||
|lift_pot|demo_randomized|queued|—||
|lift_pot|demo_clean|queued|—||
|move_can_pot|demo_clean|queued|—||
|move_can_pot|demo_randomized|queued|—||
|move_pillbottle_pad|demo_randomized|queued|—||
|move_pillbottle_pad|demo_clean|queued|—||
|move_playingcard_away|demo_clean|queued|—||
|move_playingcard_away|demo_randomized|queued|—||
|move_stapler_pad|demo_randomized|queued|—||
|move_stapler_pad|demo_clean|queued|—||
|pick_diverse_bottles|demo_clean|queued|—||
|pick_diverse_bottles|demo_randomized|queued|—||
|pick_dual_bottles|demo_randomized|queued|—||
|pick_dual_bottles|demo_clean|queued|—||
|place_a2b_left|demo_clean|queued|—||
|place_a2b_left|demo_randomized|queued|—||
|place_a2b_right|demo_randomized|queued|—||
|place_a2b_right|demo_clean|queued|—||
|place_container_plate|demo_clean|queued|—||
|place_container_plate|demo_randomized|queued|—||
|place_fan|demo_randomized|queued|—||
|place_fan|demo_clean|queued|—||
|place_mouse_pad|demo_clean|queued|—||
|place_mouse_pad|demo_randomized|queued|—||
|place_object_scale|demo_randomized|queued|—||
|place_object_scale|demo_clean|queued|—||
|place_object_stand|demo_clean|queued|—||
|place_object_stand|demo_randomized|queued|—||
|place_phone_stand|demo_randomized|queued|—||
|place_phone_stand|demo_clean|queued|—||
|press_stapler|demo_clean|queued|—||
|press_stapler|demo_randomized|queued|—||
|rotate_qrcode|demo_randomized|queued|—||
|rotate_qrcode|demo_clean|queued|—||
|stamp_seal|demo_clean|queued|—||
|stamp_seal|demo_randomized|queued|—||
|turn_switch|demo_randomized|queued|—||
|turn_switch|demo_clean|queued|—||
