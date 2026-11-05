# 原21480训练集离线token干预诊断

用户2026-11-02授权同10任务训练数据分析，并要求补Grounding与Plan。固定原最佳21480权重，10任务每任务前2个训练episode、各起点/中段2窗口，共40个训练已见Clean窗口。复用既有sidecar_gt_diagnostics准备逻辑，验证HDF5元数据及EA标签SHA；相同原始三视图240×320、当前EEF、32步专家未来EEF目标（anchor+1至anchor+32）。训练数据来自已有aloha_4350/manifest.ready.json指向的官方Clean数据，不新增训练、不改数据。

|组别|操作|
|---|---|
|Full|原推理|
|Write-Off|3tap直接Action写入恒等旁路|
|Relation-Reset|每tap read/refine后tokens[:,4:8]置零|
|Grounding-Reset|每tap read/refine后tokens[:,0:4]置零|
|Plan-Reset|每tap read/refine后tokens[:,9:15]置零|
|Grounding-Head-Off|只把辅助grounding预测输出置零，tokens和Write不变|
|Plan-Head-Off|只把辅助plan方向/移动预测置零，tokens和Write不变|

头部对照预计与Full动作逐元素相同，代码会直接断言；预测head是读出，并非动作写入路径。Token干预保留其他角色/全部模型参数，不等于完整切除某语义在分布式表征里的所有来源。Grounding-Reset与Plan-Reset和Relation-Reset均作用于每tap之后、Write之前及下个tap的传入token，范围一致。

40窗口×7组＝280次生成，同图像/状态/语言/噪声seed42、10步去噪、cache .99/max3、compile关闭。T5先在GPU0编码固定prompt后回CPU，仅缓存文本embedding留GPU，随后策略主体搬GPU0；模型计算/精度不变。保留parameter _version及Sidecar fingerprint不变验收。

记录专家GT XYZ误差（毫米）、旋转误差、夹爪误差及相对Full动作漂移，输出每任务和总体均值。输入为训练已见Clean名义图像，没有加训练时随机增强；这不是Randomized泛化测试，也不是闭环成功率。低离线动作误差不保证闭环更成功。用于判断先前闭环干预差异是否也伴随训练专家动作拟合的变化。

独立目录outputs/robotwin_eval/sidecar21480_train_interventions_20261102；plan、inputs_ready、progress、results、summary、complete与launch日志保留。当前三组闭环干预600条已完成，新增两个token组本轮先做训练集离线诊断，不误称其已有闭环SR。
