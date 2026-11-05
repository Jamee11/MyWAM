# Midtrain训练数据组成（2026-11-02核查）

源manifest SHA256：`a96d7f13dcb243f9429303a667d76d02151098a83c2946d996a31b21bf4a07aa`，与实际midtrain保存的data contract一致。以下主表只统计split=train且selected_for_midtrain未禁用的episode；不含val/test。Random为所有非Clean域的合计。

| 来源 | 任务数 | 归档总条目 | Train Clean | Train Random | Train合计 | Val | Test | Train窗口 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| maniskill_franka | 8 | 20000 | 4000 | 12000 | 16000 | 2000 | 2000 | 2671589 |
| robotwin_aloha | 50 | 12000 | 1998 | 7586 | 9584 | 1227 | 1189 | 2124695 |
| robotwin_piper | 35 | 12000 | 1988 | 7578 | 9566 | 1172 | 1262 | 1404781 |
| rlbench_franka | 18 | 8204 | 778 | 3888 | 4666 | 1762 | 1776 | 1080908 |

归档全组52204条，实际train池39816条；ManiSkill+Aloha+Piper三组实际train共35150条。条目含不同域版本，不能等同于全部独立物理轨迹。旧Aloha 2500条Clean专项微调不采用本次midtrain split口径，不能混为同一数据量。

## 实际域细分

| 来源 | Collection | Domain | Train条目 |
|---|---|---|---:|
| maniskill_franka | maniskill_base | B | 800 |
| maniskill_franka | maniskill_base | C | 800 |
| maniskill_franka | maniskill_base | Clean | 4000 |
| maniskill_franka | maniskill_base | D | 800 |
| maniskill_franka | maniskill_base | M | 800 |
| maniskill_franka | maniskill_base | O | 800 |
| maniskill_franka | maniskill_hard | D | 2000 |
| maniskill_franka | maniskill_hard | M | 6000 |
| robotwin_aloha | robotwin_base | Clean | 1998 |
| robotwin_aloha | robotwin_base | DR2500/B | 501 |
| robotwin_aloha | robotwin_base | DR2500/C | 500 |
| robotwin_aloha | robotwin_base | DR2500/M | 494 |
| robotwin_aloha | robotwin_base | DR2500/O | 493 |
| robotwin_aloha | robotwin_base | DR7500/B | 1003 |
| robotwin_aloha | robotwin_base | DR7500/C | 997 |
| robotwin_aloha | robotwin_base | DR7500/M | 995 |
| robotwin_aloha | robotwin_base | DR7500/O | 1001 |
| robotwin_aloha | robotwin_clutter | fresh_clutter_randomized | 1602 |
| robotwin_piper | robotwin_base | Clean | 1988 |
| robotwin_piper | robotwin_base | DR/B | 1505 |
| robotwin_piper | robotwin_base | DR/C | 1493 |
| robotwin_piper | robotwin_base | DR/M | 1478 |
| robotwin_piper | robotwin_base | DR/O | 1500 |
| robotwin_piper | robotwin_clutter | fresh_clutter_randomized | 1602 |
| rlbench_franka | rlbench | B | 778 |
| rlbench_franka | rlbench | C | 778 |
| rlbench_franka | rlbench | Clean | 778 |
| rlbench_franka | rlbench | D | 777 |
| rlbench_franka | rlbench | M | 777 |
| rlbench_franka | rlbench | O | 778 |

## 逐任务明细

| 来源 | 任务 | Clean | 普通Random | HardRandom | FreshClutter | 合计 | 窗口 |
|---|---|---:|---:|---:|---:|---:|---:|
| maniskill_franka | lift_peg_upright | 500 | 500 | 1000 | 0 | 2000 | 499141 |
| maniskill_franka | pick_cube | 500 | 500 | 1000 | 0 | 2000 | 252036 |
| maniskill_franka | pick_cuboid | 500 | 500 | 1000 | 0 | 2000 | 287100 |
| maniskill_franka | pick_sphere | 500 | 500 | 1000 | 0 | 2000 | 252126 |
| maniskill_franka | place_sphere | 500 | 500 | 1000 | 0 | 2000 | 384840 |
| maniskill_franka | pull_cube | 500 | 500 | 1000 | 0 | 2000 | 383767 |
| maniskill_franka | push_cube | 500 | 500 | 1000 | 0 | 2000 | 277588 |
| maniskill_franka | stack_cube | 500 | 500 | 1000 | 0 | 2000 | 334991 |
| robotwin_aloha | adjust_bottle | 41 | 122 | 0 | 31 | 194 | 27408 |
| robotwin_aloha | beat_block_hammer | 43 | 130 | 0 | 29 | 202 | 22930 |
| robotwin_aloha | blocks_ranking_rgb | 44 | 132 | 0 | 28 | 204 | 93837 |
| robotwin_aloha | blocks_ranking_size | 41 | 123 | 0 | 31 | 195 | 90415 |
| robotwin_aloha | click_alarmclock | 42 | 121 | 0 | 30 | 193 | 16376 |
| robotwin_aloha | click_bell | 41 | 122 | 0 | 31 | 194 | 14983 |
| robotwin_aloha | dump_bin_bigbin | 42 | 130 | 0 | 30 | 202 | 47174 |
| robotwin_aloha | grab_roller | 42 | 124 | 0 | 30 | 196 | 18385 |
| robotwin_aloha | handover_block | 40 | 120 | 0 | 32 | 192 | 54174 |
| robotwin_aloha | handover_mic | 38 | 113 | 0 | 34 | 185 | 40999 |
| robotwin_aloha | hanging_mug | 39 | 117 | 0 | 33 | 189 | 63828 |
| robotwin_aloha | lift_pot | 40 | 121 | 0 | 32 | 193 | 21508 |
| robotwin_aloha | move_can_pot | 40 | 119 | 0 | 32 | 191 | 28808 |
| robotwin_aloha | move_pillbottle_pad | 36 | 110 | 0 | 36 | 182 | 26679 |
| robotwin_aloha | move_playingcard_away | 40 | 121 | 0 | 32 | 193 | 22645 |
| robotwin_aloha | move_stapler_pad | 44 | 132 | 0 | 28 | 204 | 31895 |
| robotwin_aloha | open_laptop | 39 | 116 | 0 | 33 | 188 | 39468 |
| robotwin_aloha | open_microwave | 37 | 115 | 0 | 35 | 187 | 110857 |
| robotwin_aloha | pick_diverse_bottles | 38 | 112 | 0 | 34 | 184 | 22300 |
| robotwin_aloha | pick_dual_bottles | 38 | 114 | 0 | 34 | 186 | 22861 |
| robotwin_aloha | place_a2b_left | 38 | 112 | 0 | 34 | 184 | 27226 |
| robotwin_aloha | place_a2b_right | 42 | 125 | 0 | 30 | 197 | 29318 |
| robotwin_aloha | place_bread_basket | 42 | 126 | 0 | 30 | 198 | 47748 |
| robotwin_aloha | place_bread_skillet | 40 | 122 | 0 | 32 | 194 | 32339 |
| robotwin_aloha | place_burger_fries | 41 | 128 | 0 | 31 | 200 | 48194 |
| robotwin_aloha | place_can_basket | 42 | 125 | 0 | 30 | 197 | 49394 |
| robotwin_aloha | place_cans_plasticbox | 43 | 130 | 0 | 29 | 202 | 57798 |
| robotwin_aloha | place_container_plate | 40 | 121 | 0 | 32 | 193 | 30522 |
| robotwin_aloha | place_dual_shoes | 38 | 109 | 0 | 34 | 181 | 41279 |
| robotwin_aloha | place_empty_cup | 42 | 123 | 0 | 30 | 195 | 33510 |
| robotwin_aloha | place_fan | 39 | 118 | 0 | 33 | 190 | 27891 |
| robotwin_aloha | place_mouse_pad | 38 | 109 | 0 | 34 | 181 | 27423 |
| robotwin_aloha | place_object_basket | 44 | 132 | 0 | 28 | 204 | 50309 |
| robotwin_aloha | place_object_scale | 42 | 126 | 0 | 30 | 198 | 28813 |
| robotwin_aloha | place_object_stand | 42 | 126 | 0 | 30 | 198 | 27567 |
| robotwin_aloha | place_phone_stand | 40 | 119 | 0 | 32 | 191 | 24171 |
| robotwin_aloha | place_shoe | 41 | 125 | 0 | 31 | 197 | 35297 |
| robotwin_aloha | press_stapler | 40 | 124 | 0 | 32 | 196 | 23329 |
| robotwin_aloha | put_bottles_dustbin | 39 | 113 | 0 | 33 | 185 | 115825 |
| robotwin_aloha | put_object_cabinet | 36 | 108 | 0 | 36 | 180 | 48779 |
| robotwin_aloha | rotate_qrcode | 39 | 118 | 0 | 33 | 190 | 29493 |
| robotwin_aloha | scan_object | 37 | 111 | 0 | 35 | 183 | 30833 |
| robotwin_aloha | shake_bottle | 38 | 109 | 0 | 34 | 181 | 45060 |
| robotwin_aloha | shake_bottle_horizontally | 39 | 118 | 0 | 33 | 190 | 52880 |
| robotwin_aloha | stack_blocks_three | 40 | 120 | 0 | 32 | 192 | 90775 |
| robotwin_aloha | stack_blocks_two | 40 | 120 | 0 | 32 | 192 | 60165 |
| robotwin_aloha | stack_bowls_three | 39 | 120 | 0 | 33 | 192 | 90575 |
| robotwin_aloha | stack_bowls_two | 36 | 105 | 0 | 36 | 177 | 55532 |
| robotwin_aloha | stamp_seal | 38 | 117 | 0 | 34 | 189 | 27457 |
| robotwin_aloha | turn_switch | 38 | 111 | 0 | 34 | 183 | 17663 |
| robotwin_piper | beat_block_hammer | 60 | 176 | 0 | 44 | 280 | 21174 |
| robotwin_piper | blocks_ranking_rgb | 56 | 167 | 0 | 48 | 271 | 87625 |
| robotwin_piper | blocks_ranking_size | 56 | 173 | 0 | 48 | 277 | 89344 |
| robotwin_piper | click_bell | 60 | 175 | 0 | 44 | 279 | 15074 |
| robotwin_piper | grab_roller | 59 | 175 | 0 | 45 | 279 | 17687 |
| robotwin_piper | handover_block | 57 | 170 | 0 | 46 | 273 | 57093 |
| robotwin_piper | handover_mic | 57 | 166 | 0 | 46 | 269 | 42918 |
| robotwin_piper | lift_pot | 60 | 179 | 0 | 43 | 282 | 23175 |
| robotwin_piper | move_can_pot | 58 | 174 | 0 | 45 | 277 | 28849 |
| robotwin_piper | move_pillbottle_pad | 59 | 176 | 0 | 44 | 279 | 28653 |
| robotwin_piper | move_playingcard_away | 60 | 178 | 0 | 43 | 281 | 23681 |
| robotwin_piper | move_stapler_pad | 58 | 166 | 0 | 45 | 269 | 29202 |
| robotwin_piper | open_laptop | 56 | 170 | 0 | 47 | 273 | 46215 |
| robotwin_piper | pick_diverse_bottles | 54 | 163 | 0 | 49 | 266 | 22520 |
| robotwin_piper | pick_dual_bottles | 58 | 173 | 0 | 45 | 276 | 23156 |
| robotwin_piper | place_a2b_left | 59 | 183 | 0 | 43 | 285 | 29284 |
| robotwin_piper | place_a2b_right | 57 | 176 | 0 | 45 | 278 | 28751 |
| robotwin_piper | place_bread_basket | 56 | 174 | 0 | 46 | 276 | 43413 |
| robotwin_piper | place_burger_fries | 57 | 177 | 0 | 45 | 279 | 46600 |
| robotwin_piper | place_container_plate | 56 | 169 | 0 | 46 | 271 | 29268 |
| robotwin_piper | place_dual_shoes | 56 | 172 | 0 | 46 | 274 | 43200 |
| robotwin_piper | place_empty_cup | 55 | 170 | 0 | 47 | 272 | 35056 |
| robotwin_piper | place_mouse_pad | 54 | 160 | 0 | 48 | 262 | 26521 |
| robotwin_piper | place_object_stand | 55 | 166 | 0 | 47 | 268 | 27200 |
| robotwin_piper | place_phone_stand | 58 | 175 | 0 | 44 | 277 | 26892 |
| robotwin_piper | place_shoe | 57 | 167 | 0 | 45 | 269 | 34361 |
| robotwin_piper | press_stapler | 54 | 162 | 0 | 48 | 264 | 25529 |
| robotwin_piper | put_bottles_dustbin | 58 | 171 | 0 | 44 | 273 | 120108 |
| robotwin_piper | shake_bottle | 57 | 170 | 0 | 45 | 272 | 39498 |
| robotwin_piper | shake_bottle_horizontally | 57 | 174 | 0 | 45 | 276 | 44729 |
| robotwin_piper | stack_blocks_two | 54 | 158 | 0 | 48 | 260 | 58078 |
| robotwin_piper | stack_bowls_three | 56 | 167 | 0 | 46 | 269 | 86035 |
| robotwin_piper | stack_bowls_two | 55 | 165 | 0 | 47 | 267 | 57034 |
| robotwin_piper | stamp_seal | 54 | 169 | 0 | 48 | 271 | 27012 |
| robotwin_piper | turn_switch | 55 | 170 | 0 | 47 | 272 | 19846 |
| rlbench_franka | close_jar | 44 | 220 | 0 | 0 | 264 | 44734 |
| rlbench_franka | insert_onto_square_peg | 44 | 220 | 0 | 0 | 264 | 49932 |
| rlbench_franka | light_bulb_in | 44 | 220 | 0 | 0 | 264 | 61986 |
| rlbench_franka | meat_off_grill | 44 | 218 | 0 | 0 | 262 | 37486 |
| rlbench_franka | open_drawer | 43 | 215 | 0 | 0 | 258 | 31824 |
| rlbench_franka | place_cups | 43 | 215 | 0 | 0 | 258 | 107476 |
| rlbench_franka | place_shape_in_shape_sorter | 43 | 215 | 0 | 0 | 258 | 48962 |
| rlbench_franka | place_wine_at_rack_location | 43 | 215 | 0 | 0 | 258 | 55858 |
| rlbench_franka | push_buttons | 43 | 215 | 0 | 0 | 258 | 37440 |
| rlbench_franka | put_groceries_in_cupboard | 43 | 215 | 0 | 0 | 258 | 58606 |
| rlbench_franka | put_item_in_drawer | 43 | 215 | 0 | 0 | 258 | 116018 |
| rlbench_franka | put_money_in_safe | 43 | 215 | 0 | 0 | 258 | 49998 |
| rlbench_franka | reach_and_drag | 43 | 215 | 0 | 0 | 258 | 47284 |
| rlbench_franka | slide_block_to_color_target | 43 | 215 | 0 | 0 | 258 | 40882 |
| rlbench_franka | stack_blocks | 43 | 215 | 0 | 0 | 258 | 140790 |
| rlbench_franka | stack_cups | 43 | 215 | 0 | 0 | 258 | 78782 |
| rlbench_franka | sweep_to_dustpan_of_size | 43 | 215 | 0 | 0 | 258 | 35094 |
| rlbench_franka | turn_tap | 43 | 215 | 0 | 0 | 258 | 37756 |

## 数量与实际采样的区别

midtrain四个来源weights=[1,1,1,1]，按动作窗口等权混采，来源各约25%，不是按episode数量比例混采。全train可用窗口7281973，每源名义采样1820493个窗口（全局512尾批对齐额外204个窗口）；one mixture epoch不是每个来源分别完整遍历一遍。这里给出训练池episode，不冒称逐batch消费日志覆盖。

详细机器可读文件：`/remote-home/jinminghao/OpenWAM/outputs/analysis/midtrain_data_composition_20261102` 下summary.json、group_counts.csv、domain_counts.csv、task_counts.csv、train_episodes.csv（39816行，含实际HDF5与numeric相对路径）。本次只读统计，没有修改训练/样本/采样比例。
