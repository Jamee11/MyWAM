# 训练与评估图像格式、尺寸和腕部近裁剪面对齐核查

时间：2026-10-22T14:34:04.316060+00:00。当前仅只读分析与独立CPU图像预处理检查，不修改任何评估/训练配置。

## 格式与几何链路

|阶段|训练（本次Sidecar数据）|当前官方RoboTwin评估|
|---|---|---|
|数据来源|raw/aloha HDF5，各相机JPEG bytes|实时SAPIEN get_obs，不读取训练HDF5/MP4|
|原始数组|JPEG按训练cv2.imdecode还原RGB uint8|Color RGBA转uint8、取前三通道，RGB uint8|
|原始尺寸|抽查三路各H240×W320×3|官方Clean/Randomized均D435，配置H240×W320×3|
|传输|本地解码为PIL RGB|三路分别编码无损PNG→base64→WebSocket；客户端不resize|
|拼接|assemble_multiview_layout|服务端ObsPreprocessor调用同一assemble_multiview_layout|
|槽位|head H256×W320，left/right各H128×W160|相同|
|最终图像|H384×W320×3|相同|

因此，没有发现“评估使用不同分辨率或错误拼接尺寸”这一问题。JPEG训练图像含原有有损压缩，PNG只保证评估传输不再损失原RGB；同尺寸不意味着不同场景的像素相同。训练随机ColorJitter与评估关闭增强也须分开表述。

## 已发现的真实相机参数差异：腕部near

本次数据README注明Wrist near=0.02m，SOURCE_EVIDENCE/raw_sources/*/ea_rsig_v3_collection_manifest.json还保存wrist_near_m=0.02、wrist_near_correction及head_near_m约0.1。当前官方RoboTwin/envs/camera/camera.py:81设置near,far=0.1,100，并在左右腕部scene.add_camera调用中传同一个near。当前评估wrapper没有修改该值。

|参数|此训练数据|当前官方评估|
|---|---:|---:|
|主相机near|0.1m|0.1m|
|左/右腕部near|0.02m|0.1m|

这是同一240×320分辨率下的可见内容差异：相机前方深度2–10cm的表面，在两种near下可能被不同裁剪。不能通过resize消除，也不能据此立即认定SR降低由它造成。需要相同scene/camera pose下0.02/0.1图像与闭环对照量化。

评估代码此处遵守官方0.1；训练数据此前重渲染时修改为0.02。二者并不矛盾，但说明“尽量对齐官方评估”不等于“与自采训练数据全部相机参数一致”。之前笼统称相机输入已整体一致不够严谨，应限定为已核查的尺寸、相机顺序和拼接；现将near差异明确记录，不再遗漏。

证据：outputs/robotwin_eval/training_image_preview_20261022/camera_clipping_alignment_evidence.json；同目录预处理parity结果生成后另记。此结论针对给定4350集合与当前L20官方评估，未假定来源未完整提供的baseline10740训练数据也完全相同。

下一步应保留已完成的官方标准成绩，独立固定场景比较near后再决定是否做训练相机参数匹配的诊断；不能静默改near并把两种评测拼成一个成绩。本次没有改变near、数据、模型或评测进程。


补充验证：使用此前实际HDF5解码并保存的首帧/第70帧三路RGB，调用真实encode_numpy_b64进行PNG/base64编码，再调用真实ObsPreprocessor.preprocess，输出均为H384×W320；与训练assemble_multiview_layout生成的参考PNG逐像素max_abs_difference=0。为避免加载无关模型依赖，通过文件加载实际模块，未改动图像函数。该检查仅证明同RGB输入下传输/解码/几何预处理一致，不消除训练JPEG压缩、场景差异或腕部near0.02/0.1的内容差异。证据training_image_preview_20261022/train_eval_png_preprocess_parity.json。
