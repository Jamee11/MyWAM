# PatchWAM24k 与既往两版warmup24k：相同条件30条比较

快照：2026-10-30T04:07:50.258071+00:00；仅当前已完整完成的13条件，不拿旧全任务均值对比新局部。

旧isolation24k：`outputs/1022-sidecar-warmup2k/checkpoint_step_24000.safetensors`。
旧full-backward24k：`outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_24000.safetensors`。
新PatchWAM24k：`outputs/sidecar/aloha50_patchwam_c2r_ramp100_10epochs_20261027T144939Z/F_patchwam_ramp100/runs/2026-10-27_14-57-15/checkpoint_step_24000.safetensors`。

|范围|共同条件|新PatchWAM24k|旧isolation24k|旧full-backward24k|
|---|---:|---:|---:|---:|
|综合|13|61.03%|60.26%|54.10%|
|Clean|7|75.71%|82.86%|78.10%|
|Randomized|6|43.89%|33.89%|26.11%|

|Task|Mode|新PatchWAM24k成功/30|旧isolation24k成功/30|旧full-backward24k成功/30|
|---|---|---:|---:|---:|
|place_mouse_pad|demo_randomized|12|9|7|
|move_stapler_pad|demo_randomized|5|4|3|
|place_phone_stand|demo_randomized|9|7|6|
|place_fan|demo_randomized|16|12|11|
|pick_diverse_bottles|demo_randomized|19|23|17|
|place_dual_shoes|demo_randomized|18|6|3|
|place_mouse_pad|demo_clean|25|23|24|
|move_stapler_pad|demo_clean|18|20|15|
|place_phone_stand|demo_clean|23|25|25|
|place_fan|demo_clean|23|26|27|
|blocks_ranking_rgb|demo_clean|25|28|26|
|pick_diverse_bottles|demo_clean|18|27|20|
|place_dual_shoes|demo_clean|27|25|27|

观察：目前对旧isolation综合+0.77pp，Clean−7.14、Randomized+10；对旧full-backward综合+6.92，Clean−2.38、Randomized+17.78。双鞋Randomized增益较大，pick_diverse_bottles两域下降、Clean尤其明显。仅13个优先条件，不能外推全任务或据任务名认定增强原因；确切语言/accepted seed/RNG和推理通信版本未严格逐episode配对。旧isolation使用最终100行CSV，而非仍停早期单条件的comparison_latest.json。等完整100条件再判断总体。

## 更新：当前已完成子集（2026-10-30 07:12 UTC）

按当前PatchWAM24k已完成的53个共同条件、每条件30条重新对齐两版旧warmup24k；旧实验均从各自最终结果取数。当前PatchWAM共1590条、成功1102条。以下是相同条件子集结果，不与旧checkpoint的全100条件均值混比：

|范围|共同条件|当前PatchWAM24k|旧isolation24k|旧full-backward24k|
|---|---:|---:|---:|---:|
|综合|53|69.31% (1102/1590)|65.03% (1034/1590)|63.90% (1016/1590)|
|Clean|29|82.87% (721/870)|86.09% (749/870)|84.94% (739/870)|
|Randomized|24|52.92% (381/720)|39.58% (285/720)|38.47% (277/720)|

相同子集差值：PatchWAM24k相对旧isolation24k综合+4.28个百分点、Clean−3.22、Randomized+13.33；相对旧full-backward24k综合+5.41、Clean−2.07、Randomized+14.44。当前优势主要来自Randomized，Clean略低；不能据此说整体100任务评测已确定胜出。任务条件差值也有正有负，例如相对旧isolation的`pick_diverse_bottles / demo_clean`少9/30，而`place_dual_shoes / demo_randomized`多12/30。样本条件未逐episode固定指令、seed与随机流，且仍有47个条件未完成，以上是阶段性趋势而非因果结论。下一步待全100条件完成后，用相同100条件重算三版结果。
