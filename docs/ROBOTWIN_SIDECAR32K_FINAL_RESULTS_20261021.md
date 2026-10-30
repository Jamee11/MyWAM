# Sidecar32k RoboTwin 完整评估结果

核对时间：2026-10-21T13:21:24.388375+00:00

50任务 × Clean/Randomized × 100条，10000条全部完成。100个原始run.json的complete/实际条数/成功数与控制器逐项核对通过，无最终失败条件。

|范围|成功数/总数|32k|官方|差值pp|
|---|---:|---:|---:|---:|
|Clean|4383/5000|87.66%|89.36%|-1.70|
|Randomized|2265/5000|45.30%|48.72%|-3.42|
|综合|6648/10000|66.48%|69.04%|-2.56|

协议保持原官方环境/RT/专家筛seed/动作预算/成功判据，Sidecar三tap、FFN compile及cache .99/max3。官方数值来自已封存issue46对照CSV，不代表相同accepted seed/指令/CUDA RNG逐例配对；现有baseline10740的训练预算和统计亦不同，不能纯归因Sidecar结构。

|Task|32k Clean|官方 Clean|差值pp|32k Randomized|官方 Randomized|差值pp|
|---|---:|---:|---:|---:|---:|---:|
|adjust_bottle|97%|90%|+7|37%|51%|-14|
|beat_block_hammer|98%|99%|-1|33%|55%|-22|
|blocks_ranking_rgb|95%|96%|-1|46%|47%|-1|
|blocks_ranking_size|62%|55%|+7|15%|14%|+1|
|click_alarmclock|98%|99%|-1|65%|77%|-12|
|click_bell|100%|100%|+0|71%|75%|-4|
|dump_bin_bigbin|91%|99%|-8|64%|69%|-5|
|grab_roller|100%|100%|+0|49%|38%|+11|
|handover_block|93%|84%|+9|40%|52%|-12|
|handover_mic|92%|90%|+2|15%|22%|-7|
|hanging_mug|33%|49%|-16|8%|10%|-2|
|lift_pot|100%|100%|+0|16%|6%|+10|
|move_can_pot|100%|100%|+0|38%|29%|+9|
|move_pillbottle_pad|96%|96%|+0|31%|31%|+0|
|move_playingcard_away|99%|97%|+2|53%|49%|+4|
|move_stapler_pad|72%|76%|-4|16%|16%|+0|
|open_laptop|92%|93%|-1|57%|57%|+0|
|open_microwave|61%|82%|-21|3%|7%|-4|
|pick_diverse_bottles|78%|77%|+1|61%|62%|-1|
|pick_dual_bottles|92%|85%|+7|72%|73%|-1|
|place_a2b_left|94%|87%|+7|21%|30%|-9|
|place_a2b_right|97%|89%|+8|23%|30%|-7|
|place_bread_basket|90%|92%|-2|76%|74%|+2|
|place_bread_skillet|94%|92%|+2|41%|61%|-20|
|place_burger_fries|99%|97%|+2|70%|80%|-10|
|place_can_basket|75%|87%|-12|38%|45%|-7|
|place_cans_plasticbox|99%|98%|+1|86%|96%|-10|
|place_container_plate|97%|98%|-1|79%|76%|+3|
|place_dual_shoes|88%|89%|-1|10%|19%|-9|
|place_empty_cup|100%|100%|+0|77%|67%|+10|
|place_fan|89%|93%|-4|42%|40%|+2|
|place_mouse_pad|83%|90%|-7|19%|21%|-2|
|place_object_basket|88%|95%|-7|30%|38%|-8|
|place_object_scale|94%|94%|+0|37%|38%|-1|
|place_object_stand|99%|96%|+3|63%|65%|-2|
|place_phone_stand|83%|85%|-2|14%|22%|-8|
|place_shoe|92%|98%|-6|66%|51%|+15|
|press_stapler|94%|90%|+4|46%|53%|-7|
|put_bottles_dustbin|80%|83%|-3|37%|52%|-15|
|put_object_cabinet|45%|48%|-3|13%|17%|-4|
|rotate_qrcode|85%|90%|-5|12%|7%|+5|
|scan_object|87%|93%|-6|52%|60%|-8|
|shake_bottle|100%|100%|+0|91%|91%|+0|
|shake_bottle_horizontally|100%|100%|+0|90%|93%|-3|
|stack_blocks_three|93%|95%|-2|49%|66%|-17|
|stack_blocks_two|100%|100%|+0|82%|89%|-7|
|stack_bowls_three|76%|83%|-7|73%|73%|+0|
|stack_bowls_two|93%|96%|-3|76%|74%|+2|
|stamp_seal|73%|89%|-16|28%|29%|-1|
|turn_switch|47%|54%|-7|34%|39%|-5|

完整JSON/CSV：outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/final_results_20261021.{json,csv}。接下来继续固定十任务baseline10740双域对照，等百条完成再比较。
