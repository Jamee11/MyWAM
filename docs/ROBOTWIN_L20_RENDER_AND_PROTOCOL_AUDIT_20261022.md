# L20渲染与RoboTwin协议核查

核查时间：2026-10-22T08:59:45.037497+00:00。只读核查源码、原评测记录、官方文档和当前Vulkan能力，未修改评测进程/配置。

## 已确认

- 当前8张NVIDIA L20，驱动550.127.08。vulkaninfo现场返回NVIDIA L20及VK_KHR_acceleration_structure（revision13）、VK_KHR_ray_tracing_pipeline（revision1）；L20具备此渲染栈所需光追API能力。
- 官方RoboTwin envs/_base_task.py setup_scene无条件使用rt shader、32 samples/pixel、path depth8、OIDN去噪。该文件SHA与已完成32k run.json记录一致，当前RoboTwin git tracked diff为空。
- standard_eval.py设置SAPIEN_DISABLE_RAY_TRACING=0、ROBOTWIN_ENABLE_PLANNER_FALLBACK=0。不能只凭环境变量证明RT；本次同时核查实际shader设置、Vulkan扩展和已有runtime_smoke.json真实RT图像有限值证据。
- 相机config、demo_clean/randomized、robot.py当前SHA与32k原记录一致。保存视频关闭只修改新任务私有task config的eval_video_log，不改变渲染/相机/物理步进。
- RoboTwin commit0aeea2d669c0f8516f4d5785f0aa33ba812c14b4、Python3.10、SAPIEN3.0.0b1、cuRobo v0.7.8、warp1.13.0、mplib0.2.1、仿真Torch2.4.1+cu121，与OpenWAM当前README verified stack对应。L20 cuRobo扩展独立sm89构建，不复用A100 sm80二进制。

## 官方硬件文字的含义

[RoboTwin安装文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)列出Rendering:NVIDIA或AMD、Ray tracing:NVIDIA RTX或AMD equivalent、Denoising:NVIDIA；RT驱动>=470、OIDN>=520；GPU simulation需NVIDIA；Docker应提供graphics capability。没有指定4090作为评估唯一合规型号。

[NVIDIA官方架构参考](https://docs.nvidia.com/ai-enterprise/release-7/latest/infra-software/vgpu/reference/index.html)把L20列在Ada Lovelace/Advanced ray tracing and AI系列。结合当前Vulkan能力和真实RT运行证据，不能因为型号不含RTX就认为L20无法按该RT协议评估。这里没有宣称RoboTwin发布过L20与4090的数值等价认证，也不推论任意数据中心卡都具备相同渲染能力。CUDA/cuRobo整套仍需相应NVIDIA环境，不能据渲染文档AMD条目推论整个OpenWAM流程可直接改成AMD。

[OpenWAM官方适配器文档](https://github.com/OpenWAM-Official/OpenWAM/blob/main/benchmarks/robotwin/README.md)允许独立模型/仿真环境和headless Vulkan，列出上述verified versions，未指定4090必须使用。最新RoboTwin HEAD不是必要目标，复现这里应对齐OpenWAM声明验证过的commit。

## 尚不能称完全一致的事项

|范围|当前证据与限制|
|---|---|
|跨硬件/驱动图像|未对L20与官方评测硬件/驱动下的同scene、相机、光照、RT随机状态做逐像素对照；RT设置一致不等于图像逐位相同。官方当次完整硬件/驱动快照未建立。|
|资产来源已核验|三类资产来自官方TianxingChen/RoboTwin2.0固定revision 3dc3b798668feb99ac61cc9086d84cbcc3d79186，下载验收记录SHA256/大小与官方LFS清单一致，解包文件manifest已保存。目前无资产错误或混用的证据。官方发布成绩当次没有另给资产版本锁定，这是历史复现信息边界，不应据此将当前资产列为异常或要求重下。|
|随机性与并发|同seed组和官方专家筛选不保证全部accepted seed/指令/模型CUDA RNG一致；共享模型端口独立buffer、推理互斥、每generation清cache，但缺严格单/多sim同输入闭环SR配对对照。|
|推理加速|Baseline使用原版完整MoT compile/cache，Sidecar只compileFFN、使用同cache；已有FFN匹配同cache eager的数值证据，仍缺Sidecar cache-on/off同manifest闭环SR消融。|
|EEF与训练合同|自训练reader wxyz与评估adapter对应，不能为复制发布模型默认配置而盲改xyzw；achieved EEF监督经绝对命令执行的物理跟踪误差未完整量化。训练预算/自有stats/权重不同并非纯评测硬件差异。|
|视频保存|后续Baseline关闭视频，早先任务/32k沿原配置；省去编码写盘但保留渲染。不是RT降级；未做严格video on/off SR/耗时配对A/B。|

当前证据不支持“L20不能正确RT，因此成功率较低”的结论；同样也未证明硬件/驱动对输入和闭环SR完全无影响。同步评测按动作步预算，硬件慢主要延长墙钟时间；超时/进程异常须按基础设施另查，不能等同策略失败或自动缩短上游动作预算。

后续如需归因硬件，应固定同一官方checkpoint、manifest/指令/动作噪声，对L20与目标RTX机器先比原始三相机RGB，再比策略动作与闭环成功率；现阶段不修改正在运行的正式实验。

证据：benchmarks/robotwin/standard_eval.py、.external/robotwin-official-20261018/RoboTwin/envs/_base_task.py:214、outputs/robotwin_eval/rebuild_official_20261018/runtime_smoke.json、Sidecar32k原始run.json、baseline10740根official_source_comparison.json。


## 2026-10-22T14:34:04.316060+00:00：图像尺寸链路一致，但发现训练/官方评估腕部near差异

训练raw JPEG三相机H240×W320，评估D435实时RGB同尺寸、无损PNG/base64传输；两端同一assemble_multiview_layout输出H384×W320，未发现分辨率不匹配。但给定数据采集manifest明确wrist_near_m=0.02、head约0.1，而当前官方camera.py:81及双腕相机构造使用near0.1，wrapper无相关修改。这是具体可见内容分布差异，不能再把“符合官方相机配置”外推“与训练采集全部相机参数一致”。影响大小及SR因果未测，保持为必须考虑的未解决项。证据和范围见docs/ROBOTWIN_TRAIN_EVAL_IMAGE_ALIGNMENT_20261022.md与camera_clipping_alignment_evidence.json。只读核查，不改现行near/训练/资产/已完成成绩；下一步需独立同scene近裁剪面对照，不混入原官方协议。
