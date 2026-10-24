# Sidecar full_backward24000：50任务双域×30条趋势评估

更新时间：2026-10-24T16:25:11.326538+00:00。状态：running。

完整条件：7/100；观察终态：222成功 / 305条；当前仿真：9路。

实际执行终态（含中断及重测）：305条；当前各条件最新attempt覆盖：305条。

只将完整30条条件计入该条件最终SR；上述混合partial仅用于进度，不能当全任务成功率。计划3000条，未使用100条正式分母。

权重：/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_24000.safetensors；完整Sidecar、cache .99/max3开启、FFN compile关闭。模型L20 GPU0–7；仿真4090；不存视频。

保留官方expert筛seed、unseen、seed0、RT32spp/OIDN、near0.1、原动作预算/成功判据。CUDA和Vulkan渲染PCI逐进程断言，不修改上游源。

调度24个独占policy端口、共享100条件队列，先覆盖八份不同模型；实际并行受4090显存控制，单路7680MiB+卡余量1024MiB。GPU7未达到门槛暂不放仿真。

输出：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward24000_4090_all100_30ep_20261024T144937Z`；live_progress.json实时剥离ANSI后读取原始日志，state.json用于调度/完整条件；原始client.log保留。

|Task|Mode|状态|成功/观察条数|当前动作步|
|---|---|---|---:|---|
|adjust_bottle|demo_clean|complete|30/30|122/400|
|adjust_bottle|demo_randomized|complete|17/30|124/400|
|beat_block_hammer|demo_randomized|running|9/29|400/400|
|beat_block_hammer|demo_clean|complete|28/30|111/400|
|click_alarmclock|demo_clean|complete|30/30|63/400|
|click_alarmclock|demo_randomized|complete|23/30|400/400|
|place_shoe|demo_randomized|complete|15/30|500/500|
|place_shoe|demo_clean|complete|29/30|135/500|
|put_bottles_dustbin|demo_clean|running|13/16|705/1700|
|put_bottles_dustbin|demo_randomized|running|0/5|1700/1700|
|open_microwave|demo_randomized|running|0/4|1260/1500|
|open_microwave|demo_clean|running|8/10|424/1500|
|blocks_ranking_rgb|demo_clean|running|14/15|122/1200|
|blocks_ranking_rgb|demo_randomized|running|0/6|89/1200|
|blocks_ranking_size|demo_randomized|queued|2/5|318/1200|
|blocks_ranking_size|demo_clean|running|3/4|261/1200|
|stack_blocks_three|demo_clean|running|1/1|54/1200|
|stack_blocks_three|demo_randomized|queued|—||
|stack_bowls_three|demo_randomized|queued|—||
|stack_bowls_three|demo_clean|queued|—||
|hanging_mug|demo_clean|queued|—||
|hanging_mug|demo_randomized|queued|—||
|stack_bowls_two|demo_randomized|queued|—||
|stack_bowls_two|demo_clean|queued|—||
|handover_block|demo_clean|queued|—||
|handover_block|demo_randomized|queued|—||
|place_cans_plasticbox|demo_randomized|queued|—||
|place_cans_plasticbox|demo_clean|queued|—||
|stack_blocks_two|demo_clean|queued|—||
|stack_blocks_two|demo_randomized|queued|—||
|open_laptop|demo_randomized|queued|—||
|open_laptop|demo_clean|queued|—||
|place_bread_basket|demo_clean|queued|—||
|place_bread_basket|demo_randomized|queued|—||
|place_can_basket|demo_randomized|queued|—||
|place_can_basket|demo_clean|queued|—||
|place_object_basket|demo_clean|queued|—||
|place_object_basket|demo_randomized|queued|—||
|put_object_cabinet|demo_randomized|queued|—||
|put_object_cabinet|demo_clean|queued|—||
|shake_bottle_horizontally|demo_clean|queued|—||
|shake_bottle_horizontally|demo_randomized|queued|—||
|shake_bottle|demo_randomized|queued|—||
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
