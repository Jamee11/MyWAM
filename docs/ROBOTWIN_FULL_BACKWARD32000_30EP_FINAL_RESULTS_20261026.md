# Full-backward warmup2k32k：全50任务双域×30条最终结果

核对：2026-10-26T01:00:04.871785+00:00。开始2026-10-25T07:40:06.467136+00:00，结束2026-10-25T21:57:41.423473+00:00，耗时14.2930小时。100/100条件、3000有效条全部完成；100份native exit0/实际30分母/成功数/accepted seed递增独立复验通过。2133次成功；无最终失败条件。4次基础设施/资源中断均已重测，丢弃的18条不混入正式SR，物理执行3018条。

|方案|每条件条数|Clean|Randomized|综合|
|---|---:|---:|---:|---:|
|官方发布|100|89.36%|48.72%|69.04%|
|初始Sidecar32k完整|100|87.66%|45.30%|66.48%|
|初始Sidecar32k前30|30|87.80%|45.53%|66.67%|
|isolation warmup24k|30|87.13%|49.27%|68.20%|
|full-backward warmup24k|30|85.80%|46.47%|66.13%|
|full-backward warmup32k|30|89.00%|53.20%|71.10%|

当前32k对官方综合+2.06pp（Clean−0.36、Random+4.48）；对isolation24k综合+2.90pp；对同run full-backward24k+4.97pp；对初始32k前30+4.43pp（Random+7.67pp）。同一方案继续训练24k→32k本批有改善，整体提升主要在Randomized；这是已完成全100条件观测，不是早期子集均值。每项30对官方100、accepted seed/场景/指令/RNG与硬件负载未完全配对；不能据此证明warmup/Sidecar单因素因果或统计显著性。

|Task|新32k Clean|官方 C|isolation24k C|full-backward24k C|新32k Random|官方 R|isolation24k R|full-backward24k R|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|adjust_bottle|100.00%|90.00%|90.00%|100.00%|53.33%|51.00%|50.00%|56.67%|
|beat_block_hammer|100.00%|99.00%|93.33%|93.33%|46.67%|55.00%|26.67%|33.33%|
|blocks_ranking_rgb|100.00%|96.00%|93.33%|86.67%|53.33%|47.00%|43.33%|40.00%|
|blocks_ranking_size|53.33%|55.00%|60.00%|43.33%|20.00%|14.00%|13.33%|10.00%|
|click_alarmclock|100.00%|99.00%|100.00%|100.00%|80.00%|77.00%|76.67%|76.67%|
|click_bell|100.00%|100.00%|100.00%|100.00%|83.33%|75.00%|80.00%|90.00%|
|dump_bin_bigbin|93.33%|99.00%|100.00%|96.67%|76.67%|69.00%|76.67%|73.33%|
|grab_roller|100.00%|100.00%|100.00%|100.00%|50.00%|38.00%|46.67%|40.00%|
|handover_block|93.33%|84.00%|73.33%|83.33%|26.67%|52.00%|60.00%|43.33%|
|handover_mic|100.00%|90.00%|100.00%|96.67%|23.33%|22.00%|23.33%|10.00%|
|hanging_mug|30.00%|49.00%|36.67%|6.67%|16.67%|10.00%|3.33%|13.33%|
|lift_pot|96.67%|100.00%|100.00%|100.00%|36.67%|6.00%|3.33%|26.67%|
|move_can_pot|100.00%|100.00%|96.67%|96.67%|46.67%|29.00%|33.33%|43.33%|
|move_pillbottle_pad|93.33%|96.00%|93.33%|90.00%|50.00%|31.00%|43.33%|36.67%|
|move_playingcard_away|100.00%|97.00%|96.67%|93.33%|56.67%|49.00%|50.00%|53.33%|
|move_stapler_pad|80.00%|76.00%|66.67%|50.00%|20.00%|16.00%|13.33%|10.00%|
|open_laptop|96.67%|93.00%|100.00%|96.67%|73.33%|57.00%|46.67%|70.00%|
|open_microwave|60.00%|82.00%|23.33%|73.33%|6.67%|7.00%|0.00%|10.00%|
|pick_diverse_bottles|76.67%|77.00%|90.00%|66.67%|63.33%|62.00%|76.67%|56.67%|
|pick_dual_bottles|90.00%|85.00%|86.67%|86.67%|86.67%|73.00%|86.67%|76.67%|
|place_a2b_left|96.67%|87.00%|100.00%|90.00%|36.67%|30.00%|30.00%|33.33%|
|place_a2b_right|93.33%|89.00%|86.67%|90.00%|26.67%|30.00%|40.00%|40.00%|
|place_bread_basket|93.33%|92.00%|86.67%|90.00%|76.67%|74.00%|70.00%|70.00%|
|place_bread_skillet|96.67%|92.00%|93.33%|93.33%|60.00%|61.00%|50.00%|53.33%|
|place_burger_fries|100.00%|97.00%|100.00%|96.67%|73.33%|80.00%|80.00%|66.67%|
|place_can_basket|86.67%|87.00%|86.67%|73.33%|50.00%|45.00%|60.00%|40.00%|
|place_cans_plasticbox|100.00%|98.00%|100.00%|100.00%|73.33%|96.00%|90.00%|86.67%|
|place_container_plate|96.67%|98.00%|96.67%|90.00%|83.33%|76.00%|80.00%|73.33%|
|place_dual_shoes|73.33%|89.00%|83.33%|90.00%|26.67%|19.00%|20.00%|10.00%|
|place_empty_cup|100.00%|100.00%|96.67%|96.67%|80.00%|67.00%|73.33%|80.00%|
|place_fan|90.00%|93.00%|86.67%|90.00%|40.00%|40.00%|40.00%|36.67%|
|place_mouse_pad|86.67%|90.00%|76.67%|80.00%|30.00%|21.00%|30.00%|23.33%|
|place_object_basket|96.67%|95.00%|96.67%|80.00%|63.33%|38.00%|43.33%|53.33%|
|place_object_scale|100.00%|94.00%|83.33%|90.00%|30.00%|38.00%|36.67%|33.33%|
|place_object_stand|96.67%|96.00%|93.33%|96.67%|66.67%|65.00%|80.00%|63.33%|
|place_phone_stand|83.33%|85.00%|83.33%|83.33%|26.67%|22.00%|23.33%|20.00%|
|place_shoe|96.67%|98.00%|100.00%|96.67%|63.33%|51.00%|70.00%|50.00%|
|press_stapler|93.33%|90.00%|93.33%|96.67%|56.67%|53.00%|46.67%|40.00%|
|put_bottles_dustbin|80.00%|83.00%|90.00%|83.33%|46.67%|52.00%|40.00%|36.67%|
|put_object_cabinet|46.67%|48.00%|53.33%|50.00%|20.00%|17.00%|26.67%|16.67%|
|rotate_qrcode|93.33%|90.00%|86.67%|90.00%|16.67%|7.00%|3.33%|16.67%|
|scan_object|93.33%|93.00%|83.33%|93.33%|60.00%|60.00%|53.33%|40.00%|
|shake_bottle|100.00%|100.00%|100.00%|100.00%|93.33%|91.00%|93.33%|90.00%|
|shake_bottle_horizontally|100.00%|100.00%|100.00%|100.00%|86.67%|93.00%|93.33%|86.67%|
|stack_blocks_three|93.33%|95.00%|90.00%|96.67%|73.33%|66.00%|63.33%|40.00%|
|stack_blocks_two|100.00%|100.00%|100.00%|100.00%|86.67%|89.00%|86.67%|80.00%|
|stack_bowls_three|73.33%|83.00%|86.67%|76.67%|73.33%|73.00%|66.67%|56.67%|
|stack_bowls_two|93.33%|96.00%|93.33%|100.00%|90.00%|74.00%|76.67%|76.67%|
|stamp_seal|90.00%|89.00%|73.33%|76.67%|36.67%|29.00%|10.00%|20.00%|
|turn_switch|43.33%|54.00%|56.67%|40.00%|43.33%|39.00%|33.33%|20.00%|

主要短板：handover_block Random26.67 vs官方52，place_cans_plasticbox Random73.33 vs96，open_microwave Clean60 vs82，hanging_mug Clean30 vs49。整体改善不表示每任务都改善。

本轮保持完整Sidecar三tap、cache .99/max3、compile=false、10denoise/32chunk、RT32/OIDN/near.1/官方expert筛seed/unseen/seed0/预算和成功判据，视频关闭。未改模型、训练、数据或上游源；旧20k/24k/L20对照不混入本轮。权重SHA `2a74859c3c36aa8d9e21af3d98e034230ca367ada0b61403c33f14f86a3adeb9`。

中断历史：[{"task": "stack_bowls_three", "mode": "demo_clean", "returncode": -15, "errors": ["client exited -15", "expected 30 episode records, got [1]", "missing or ambiguous native result completion marker"]}, {"task": "shake_bottle_horizontally", "mode": "demo_randomized", "returncode": -15, "errors": ["client exited -15", "expected 30 episode records, got [1, 2, 3, 4, 5, 6, 7, 8]", "missing or ambiguous native result completion marker"]}, {"task": "place_bread_skillet", "mode": "demo_randomized", "returncode": -15, "errors": ["client exited -15", "expected 30 episode records, got [1, 2, 3, 4, 5]", "missing or ambiguous native result completion marker"]}, {"task": "grab_roller", "mode": "demo_randomized", "returncode": -15, "errors": ["client exited -15", "expected 30 episode records, got [1, 2, 3, 4]", "missing or ambiguous native result completion marker"]}]

证据：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward32000_4090_all100_30ep_20261025T073709Z/final_results.json`、final_all_scheme_comparison_20261026.json/CSV、final_independent_audit_20261026.json及原始run/client.log。下一步可按用户决定复测低分任务或对关键结论做同manifest配对/更多样本；不自动扩100条或更改训练。
