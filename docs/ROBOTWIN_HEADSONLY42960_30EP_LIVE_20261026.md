# Sidecar headsonly42960：50任务双域×30条趋势评估

更新时间：2026-10-27T01:58:46.254363+00:00。状态：interrupted。

完整条件：10/100；观察终态：320成功 / 444条；当前仿真：0路。

实际执行终态（含中断及重测）：444条；当前各条件最新attempt覆盖：444条。

只将完整30条条件计入该条件最终SR；上述混合partial仅用于进度，不能当全任务成功率。计划3000条，未使用100条正式分母。

权重：/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_heads_only_from40000_to10epochs_20261025T171338Z/runs/from40000/checkpoint_step_42960.safetensors；完整Sidecar、cache .99/max3开启、FFN compile关闭。模型L20 本次通过验收的GPU（见deployment.json）；仿真L20；不存视频。

保留官方expert筛seed、unseen、seed0、RT32spp/OIDN、near0.1、原动作预算/成功判据。CUDA和Vulkan渲染PCI逐进程断言，不修改上游源。

调度实际启用的独占policy端口、共享100条件队列，先覆盖全部就绪模型；实际并行受L20显存控制，单路7680MiB+卡余量2048MiB，新增进程最低空闲10240MiB。仿真按本组L20容量策略动态分配。

输出：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/headsonly42960_l20_all100_30ep_20261026T170539Z`；live_progress.json实时剥离ANSI后读取原始日志，state.json用于调度/完整条件；原始client.log保留。

|Task|Mode|状态|成功/观察条数|当前动作步|
|---|---|---|---:|---|
|adjust_bottle|demo_clean|complete|29/30|118/400|
|adjust_bottle|demo_randomized|complete|17/30|123/400|
|beat_block_hammer|demo_randomized|complete|10/30|116/400|
|beat_block_hammer|demo_clean|complete|29/30|109/400|
|click_alarmclock|demo_clean|complete|30/30|63/400|
|click_alarmclock|demo_randomized|complete|26/30|58/400|
|place_shoe|demo_randomized|complete|19/30|261/500|
|place_shoe|demo_clean|complete|29/30|137/500|
|put_bottles_dustbin|demo_clean|interrupted|21/25|605/1700|
|put_bottles_dustbin|demo_randomized|interrupted|9/16|1047/1700|
|open_microwave|demo_randomized|interrupted|0/11|1377/1500|
|open_microwave|demo_clean|interrupted|11/20|301/1500|
|blocks_ranking_rgb|demo_clean|complete|27/30|450/1200|
|blocks_ranking_rgb|demo_randomized|interrupted|7/16|984/1200|
|blocks_ranking_size|demo_randomized|interrupted|1/15|252/1200|
|blocks_ranking_size|demo_clean|interrupted|12/23|228/1200|
|stack_blocks_three|demo_clean|complete|30/30|445/1200|
|stack_blocks_three|demo_randomized|interrupted|5/7|278/1200|
|stack_bowls_three|demo_randomized|interrupted|5/7|20/1200|
|stack_bowls_three|demo_clean|interrupted|2/3|139/1200|
|hanging_mug|demo_clean|interrupted|1/1|142/900|
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
