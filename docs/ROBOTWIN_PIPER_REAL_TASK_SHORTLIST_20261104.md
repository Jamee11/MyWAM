# Piper真机任务与道具候选

时间：2026-11-04T13:00:41.832739+00:00。范围只限当前Piper35任务，不要求双臂同时操作；所有成功率为21952仿真已完整完成的100条/域，不是Aloha成绩或真机成功率。

|任务|Clean|Randomized|道具与实现要点|
|---|---:|---:|---|
|click_bell|100%|100%|桌面接待铃，仿真固定底座；成功判据为接触铃顶，不验证铃声。|
|grab_roller|98%|99%|木质擀面杖，双臂抓两端抬起；资产长约23.2/25.1cm，直径约3.1–3.5cm。|
|move_playingcard_away|95%|96%|整副带纸盒扑克牌，非单张；资产纸盒约9.7×6–6.7×2–2.2cm。|
|move_stapler_pad|91%|未完成|订书机+11×6cm彩色薄垫；示例订书机资产0约9.6×2.3×4.8cm，其他六款外形不同。|
|press_stapler|未完成|未完成|固定订书机；源码检测顶端接触，不是装订纸张质量。|
|pick_dual_bottles|未完成|未完成|固定001_bottle/13可乐瓶及16绿色塑料瓶。缩放包围盒推得前者高24.8cm/直径约6.8cm，后者高25.5cm/直径约9.2cm；绿色瓶未确认商品型号。|
|shake_bottle|未完成|未完成|可乐等塑料瓶，资产0–19；源码质量设10g，成功判据主要为瓶子高度，不验证完整摇晃动作。|
|shake_bottle_horizontally|未完成|未完成|同shake_bottle；抓取后有横置旋转，对真机先缓做抓取抬起更合适。|
|lift_pot|59%|83%|双耳轻锅，双臂同时抓把手抬起；当前Piper Clean59%，不作为第一优先。|
|handover_block|未完成|未完成|红色长方体6×6×20cm及蓝色10×10×1cm目标块，需交接和放置，后续阶段再做。|

推荐首轮click_bell、grab_roller、move_playingcard_away；move_stapler_pad作为第二批。瓶类商品容易采购，但本checkpoint对应任务结果还未完成，先列候选。

尺寸来自官方本地assets/objects/model_data的extents逐轴乘scale，均为缩放包围盒估计，不是已确认商品尺寸；固定品牌、型号、包装与实测质量后再对照。500mL可口可乐是厂家列出的常规规格：[中粮产品页](https://productandservice.cofco.com/search/single-products/2836)，不能凭容量认定等于资产13。当前Piper任务不含place_cans_plasticbox，不把Aloha高分移到Piper。

源码：.external/robotwin-official-20261018/RoboTwin/envs/<task>.py；物体语义来自description/objects_description/<asset>/baseN.json。没有改变模型、训练、评测或真机控制。当前仿真筛选不能代替真机相机/EEF工具点/夹爪与物体几何对齐。
