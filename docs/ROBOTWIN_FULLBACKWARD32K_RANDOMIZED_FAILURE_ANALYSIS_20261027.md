# full-backward32k：Randomized失败类型与归因边界

时间：2026-10-27T13:18:23.716541+00:00。对象为最近与36k比较的full-backward warmup2k 32k，不是最早初始模块32k。

权重：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_32000.safetensors`。评测来源：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward32000_4090_all100_30ep_20261025T073709Z`。权重SHA256 `2a74859c3c36aa8d9e21af3d98e034230ca367ada0b61403c33f14f86a3adeb9`。

## 可直接确认的统计

Clean1335/1500=89.00%，165失败；Randomized798/1500=53.20%，702失败，成功率下降35.80个百分点，失败数约为Clean的4.25倍。已经逐条解析50个Randomized条件共1500条seed/累积成功记录，核对每条delta为0/1和任务总成功数。702个失败全部在达到官方动作预算时未满足成功条件；100条件最终native均exit0且complete，不能把702称程序崩溃，也不能因超时确定是错抓、滑落、交接失败还是终端差一点。

17/50个Randomized任务成功率不高于40%；最差10个任务贡献239/702=34.05%失败，失败并非集中于一种任务。

Clean≥80%且Randomized≤40%的12个任务贡献255/702=36.32%的Randomized失败。此阈值是描述性筛选，不能把该集合的均值当作某种随机化的因果效应。

## 随机化实际改变什么

已核对50个实际runtime配置，Randomized域设置完全一致：`random_background=true`、`cluttered_table=true`、`clean_background_rate=.02`、`random_light=true`、`crazy_random_light_rate=.02`、`random_table_height=.03`、`random_head_camera_dis=0`。

对应官方`envs/_base_task.py`：墙面与桌面各抽取eval模式的`unseen`背景纹理；各有2%机会回退无纹理。干扰物有2%机会整组跳过，否则目标为最多10个、最多50次放置尝试，实际可能少于10。灯光颜色/强度随机，2%概率启用每次render更新的动态灯光；桌高偏移实际为Uniform[-.03,0]米，不是±3cm。相机位移设0，所以本次Randomized不是头相机位姿扰动实验。

这些因素是同时开启的，**当前结果不是“干扰物组”“新外观组”“高度组”分别评测**。场景增加物体、纹理与灯光可影响识别/定位，桌高也改变几何关系，但不能由目前日志给各因素分配失败百分比。

## 哪些任务最敏感

下表按任务需求人工描述，表示任务涉及的能力，不表示已经观察到该能力就是失败环节：

|Task|任务需求|Clean|Randomized|随机域失败/30|
|---|---|---:|---:|---:|
|rotate_qrcode|抓取后姿态对齐|93.33%|16.67%|25/30|
|handover_mic|双臂交接|100.00%|23.33%|23/30|
|handover_block|双臂交接|93.33%|26.67%|22/30|
|place_object_scale|目标定位/精确放置|100.00%|30.00%|21/30|
|move_stapler_pad|物体与小目标相对定位|80.00%|20.00%|24/30|
|place_a2b_right|目标辨识/左右关系放置|93.33%|26.67%|22/30|
|place_a2b_left|目标辨识/左右关系放置|96.67%|36.67%|19/30|
|place_phone_stand|物体—支架对齐|83.33%|26.67%|22/30|
|place_mouse_pad|位置与朝向对准|86.67%|30.00%|21/30|
|lift_pot|双臂抓取搬运|96.67%|36.67%|19/30|
|stamp_seal|小目标接触对准|90.00%|36.67%|19/30|
|place_fan|抓取与姿态放置|90.00%|40.00%|18/30|

Clean下已经较弱的另5个任务需要单独看：open_microwave60→6.67%、hanging_mug30→16.67%、blocks_ranking_size53.33→20%、put_object_cabinet46.67→20%、turn_switch43.33→43.33%。这5个任务Randomized共118失败，已有Clean能力不足，不能把其所有失败归于新增随机化。

存在相对稳健的对照：shake_bottle100→93.33%、stack_bowls_two93.33→90%、pick_dual_bottles90→86.67%、shake_bottle_horizontally100→86.67%。相同Randomized设置下任务敏感程度不同，支持任务需求与扰动相互作用的假说，不支持统一“随机域全部无法工作”。

## 是否换了训练未见过的目标物体

训练launch.json明确manifest为outputs/sidecar/aloha_4350/manifest.ready.json，manifest的source_root指向official_clean数据raw/aloha，2500场景。训练config是clean_50、同一source，含0.2 brightness/contrast/saturation color jitter，不等于有干扰物/纹理/几何随机化训练。

目标型号随机选择来自各task的load_actors，Clean和Randomized共用；不是Randomized专属开关。逐个检查下面5个任务源码及该训练源50个场景的scene_info：

|Task|训练已出现的目标型号|
|---|---|
|hanging_mug|039_mug/base0–9，覆盖源码选择的10种杯型|
|rotate_qrcode|070_paymentsign/base0–3，覆盖源码选择的4型|
|move_stapler_pad|048_stapler/base0–6，覆盖源码选择的7型|
|place_phone_stand|077_phone/base0/1/2/4和078_phonestand/base1/2|
|open_microwave|044_microwave/base0/1|

上述5个任务的训练原始场景均无clutter和墙/桌纹理。因此，这些任务的大幅下降不能简单称为“评测使用了训练完全未见的目标型号”。训练见过型号仍可能遇到新的姿态、组合、光照及遮挡外观；没有逐seed eval asset元数据，不能把此结论推广成全部50任务的每一个资产均在训练中覆盖。

## 现有记录不能回答的部分

原生eval_policy.py将episode_info用于指令生成，没有把每seed的clutter/texture、灯光、桌高、目标资产型号和成功原因写为评测结果。当前留下seed、成功/失败、动作步及最终结果，未存视频。Base_Task虽然在内存里有cluttered_table_info/texture_info，不能因此声称原评测已经保存这些字段。故目前无法给出“干扰物造成X%、外观造成Y%”或逐条物理失败阶段统计。

最合理的当前解释是：关系定位、双臂交接、姿态/小目标对准任务对混合场景域变化较敏感；干扰物引起的选择/定位问题、纹理与光照变化、桌高几何偏移均为待验证候选。目标型号未见不是目前已核查5任务的主要可支持解释。此为假说排序，不是故障归因完成。

## 建议的最小区分实验（本轮未启动）

选择rotate_qrcode、handover_mic、place_mouse_pad/scale，再加稳健对照shake_bottle。用固定目标资产、初始位姿、目标姿态和动作预算，分别测Clean、仅clutter、仅背景/灯光、仅桌高、完整Randomized；必要时再把背景与光照拆开以区分它们。先每组10个配对场景，保留官方专家可行性检查，并区分被专家排除的场景和策略失败。

不能仅“相同seed”就视为配对：不同随机化开关会消耗不同数量的随机数，Clean/Randomized的专家可行seed集合也可能不同；需固定并记录场景清单/独立随机数流，或可验证的场景状态恢复。日志保存每回合资产/纹理/灯光/桌高、目标/干扰物位置、阶段事件，并按关键阶段少量截图，无需恢复全程视频。然后才可定量区分干扰物、外观、几何及交互项影响。任何改动原生评测逻辑的实现先按项目要求确认，不改现有正在跑的42960。

## 全50任务统计

|Task|Clean|Randomized|下降百分点|随机域失败/30|
|---|---:|---:|---:|---:|
|open_microwave|60.00%|6.67%|53.33|28/30|
|hanging_mug|30.00%|16.67%|13.33|25/30|
|rotate_qrcode|93.33%|16.67%|76.67|25/30|
|blocks_ranking_size|53.33%|20.00%|33.33|24/30|
|put_object_cabinet|46.67%|20.00%|26.67|24/30|
|move_stapler_pad|80.00%|20.00%|60.00|24/30|
|handover_mic|100.00%|23.33%|76.67|23/30|
|handover_block|93.33%|26.67%|66.67|22/30|
|place_dual_shoes|73.33%|26.67%|46.67|22/30|
|place_a2b_right|93.33%|26.67%|66.67|22/30|
|place_phone_stand|83.33%|26.67%|56.67|22/30|
|place_mouse_pad|86.67%|30.00%|56.67|21/30|
|place_object_scale|100.00%|30.00%|70.00|21/30|
|lift_pot|96.67%|36.67%|60.00|19/30|
|place_a2b_left|96.67%|36.67%|60.00|19/30|
|stamp_seal|90.00%|36.67%|53.33|19/30|
|place_fan|90.00%|40.00%|50.00|18/30|
|turn_switch|43.33%|43.33%|0.00|17/30|
|beat_block_hammer|100.00%|46.67%|53.33|16/30|
|put_bottles_dustbin|80.00%|46.67%|33.33|16/30|
|move_can_pot|100.00%|46.67%|53.33|16/30|
|place_can_basket|86.67%|50.00%|36.67|15/30|
|grab_roller|100.00%|50.00%|50.00|15/30|
|move_pillbottle_pad|93.33%|50.00%|43.33|15/30|
|adjust_bottle|100.00%|53.33%|46.67|14/30|
|blocks_ranking_rgb|100.00%|53.33%|46.67|14/30|
|move_playingcard_away|100.00%|56.67%|43.33|13/30|
|press_stapler|93.33%|56.67%|36.67|13/30|
|place_bread_skillet|96.67%|60.00%|36.67|12/30|
|scan_object|93.33%|60.00%|33.33|12/30|
|place_shoe|96.67%|63.33%|33.33|11/30|
|place_object_basket|96.67%|63.33%|33.33|11/30|
|pick_diverse_bottles|76.67%|63.33%|13.33|11/30|
|place_object_stand|96.67%|66.67%|30.00|10/30|
|stack_blocks_three|93.33%|73.33%|20.00|8/30|
|stack_bowls_three|73.33%|73.33%|0.00|8/30|
|place_cans_plasticbox|100.00%|73.33%|26.67|8/30|
|open_laptop|96.67%|73.33%|23.33|8/30|
|place_burger_fries|100.00%|73.33%|26.67|8/30|
|place_bread_basket|93.33%|76.67%|16.67|7/30|
|dump_bin_bigbin|93.33%|76.67%|16.67|7/30|
|click_alarmclock|100.00%|80.00%|20.00|6/30|
|place_empty_cup|100.00%|80.00%|20.00|6/30|
|click_bell|100.00%|83.33%|16.67|5/30|
|place_container_plate|96.67%|83.33%|13.33|5/30|
|stack_blocks_two|100.00%|86.67%|13.33|4/30|
|shake_bottle_horizontally|100.00%|86.67%|13.33|4/30|
|pick_dual_bottles|90.00%|86.67%|3.33|4/30|
|stack_bowls_two|93.33%|90.00%|3.33|3/30|
|shake_bottle|100.00%|93.33%|6.67|2/30|

逐任务CSV、1500条episode seed/成败/预算CSV及训练覆盖证据：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward32000_4090_all100_30ep_20261025T073709Z/failure_analysis_20261027T131556Z`。只读原日志和代码，本轮仅新增分析文档/派生表格，未修改任何模型、训练、数据或运行中评测。
