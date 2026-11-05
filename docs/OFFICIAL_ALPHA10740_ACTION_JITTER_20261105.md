# 官方OpenWAM-Alpha Clean2Random10740动作诊断

用户于2026-11-05要求完整测试指定官方发布权重，补齐上一轮动作误差/抖动对照。输出独立保存在`outputs/analysis/official_alpha_jitter_20261105/`，原实验结果不覆盖。

## 权重与输入

官方文件：`assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Clean2Random/checkpoint_step_10740.safetensors`。原下载验收记录SHA256为`2cab63b550ff45a52fbde4adc40ee2bdb7d4bafe5fa0dd6879e41e22ca2ff95f`，发布revision为`9bb966823547bc959800b45e9bb4005e9c2a3db3`。不要与本地复现/预热后同名10740混淆。官方config是`joint_self_attn`、80D统一动作/proprio、双臂绝对EEF20、三视图，没有Sidecar；推理时明确检查Sidecar未启用。

复用上一轮Aloha42960实际使用的同10条本地Clean-50轨迹、同80个完整32点窗口，raw state、prompt、GT与图片来源相同；使用官方自带normalization_stats.npy，其范围与本地Aloha统计一致。官方包没有可逐条核验的原始训练manifest，所以这组称为“同批本地Aloha Clean观测对照”，不额外声称每条观测必然属于官方实际训练清单。

与上一轮统一：BF16、seed42、CFG1、32点、cache/compile关闭，10/20/30去噪。复用当前项目loader/engine的生成路径，不当作官方发表成功率协议。GPU3主体/T5 GPU4，显存准入后才放置；不停止其他作业、不改5901或默认模型/训练/数据/部署代码，所有新代码仅独立分析目录。

## 四元数接口必须分别核验

风险文档2026-10-16条目已指出：官方原版reader/default adapter直接按xyzw解释数组；本地原始RoboTwin数组实际为wxyz，本地Lite reader与Piper预处理会先重排再转6D。模型看到的都是6D旋转，区别发生在数组与旋转之间的边界映射。只要训练、推理与执行配套，编码约定不同本身不等于训练错误。仅凭发布配置不能证明官方原训练数据采用哪种数组顺序，不能擅自给官方发布模型套用本地物理重排后把差表现归因训练不足。

因此本次保留同一权重的两个独立接口实验，每个均为80窗口×10/20/30，共480次预测、960臂窗口：

- `official/`：本地物理wxyz重排探针。保持上一轮Aloha的raw20状态、GT和图片完全相同，不当作官方默认协议成绩。
- `official_legacy/`：原版reader的raw-array-as-xyzw约定。只在本进程reader实例绑定原版读取函数；图像/XYZ/夹爪/文本/时间窗口保持，state与GT的6D旋转按该约定一起转换。这个实验与“物理重排”分开统计，不改全局reader或默认部署。legacy仅为文件别名，不表示该约定错误或被弃用。

旋转误差在各自编码配套的GT上计算；四元数分量置换保持单位四元数内积的绝对值，因此配套变换后的SO(3)角误差具有可比性。两套JSON分别保留原始张量解释，后续重用必须按相同边界转换，不能混用。

每个接口预计30个原始对照视频、1个三步数并排、2个与Aloha同20步的lift_pot/beat_block_hammer并排；合计66个。输入和结果分别验收，当前进行中的中间结果不称最终完成。

## 指标与视频口径

对齐单臂/Piper/Aloha：XYZ向量RMS误差、空间二阶差分粗糙度、高频误差（SG7/阶2）、二阶差分误差、旋转误差、夹爪误差、反转、GT和BF16舍入GT。双臂分别统计，另提供GT路径较长臂和GT路径<1mm的近静止窗口。

视频是专家未来图像+预测/GT目标曲线，**不是策略闭环驱动物理的rollout**。每条由8个可能重叠的独立窗口拼接，段间图像跳变/重复不是策略运动；图按窗口范围缩放，幅值以mm指标为准。播放10fps不是物理执行频率。

官方与Aloha可在同输入上配对比较，但架构、训练预算/初始化与增强不同，不能单独归因Sidecar或epoch。ManiSkill/Piper还有机器人、数据、stats与采样差异。near-plane/相机域差异仍未关闭；本次保持同本地观测，不改相机来规避差异。

完成后保留每窗口完整预测/GT action、state/prompt、视频、CSV、四模型对照、输入契约和权重身份验收。所有数值以完整完成标志与验收结果为准，不把加载中视为完成。

## 完整结果：官方默认数组解释

两个接口均已完成10轨迹/80不同窗口×三种去噪，共480次32×20预测、960臂窗口、66视频。原始60视频逐一验收，6并排视频也通过完整帧数检查（各256帧）。官方文件SHA与发布验收记录一致，min/max范围与Aloha统计最大差为0。

以下均为每窗口、每物理臂指标再取均值；位置为XYZ向量RMS，粗糙度为空间二阶差分RMS。单位mm，旋转单位度。

|模型/接口|去噪|位置误差|旋转误差|目标粗糙度|高频误差|
|---|---:|---:|---:|---:|---:|
|官方默认raw-array-as-xyzw|10|5.9627|0.8658|3.5928|1.1355|
|官方默认raw-array-as-xyzw|20|5.8124|0.8435|2.7244|0.8382|
|官方默认raw-array-as-xyzw|30|5.9065|0.8575|3.0182|0.9206|
|本地Aloha42960/配套wxyz|20|4.2351|0.5231|2.8127|0.8560|
|官方/本地物理重排探针|20|26.7437|29.5447|3.2201|1.0139|

官方默认接口10→20高频误差降低约26.2%，30没有进一步改善；位置改善较小。与同窗口的Aloha42960相比，官方位置/角度拟合较差，高频残差却基本同量级。这不是单一架构/epoch消融，不能据此把差别唯一归因Sidecar或训练时长。

对GT整32点路径<1mm的68个近静止臂窗口，官方默认20步的位置误差1.3050mm、粗糙度2.1009mm、高频误差0.6889mm、反转次数均值22.68；GT粗糙度约0.0001mm。因此原版无Sidecar模型也有残余回摆，任务成功不能证明动作目标无噪声，方向反转次数也不能脱离幅度解释。本次没有原生闭环成功率或真实机器人振幅测量。

## 对“四元数问题是否存在于本地训练”的答复

**当前列出的Aloha42960、Piper21952、ManiSkill20872，没有发现这种四元数顺序混用错误。** Aloha的Lite reader明确把本地wxyz重排后转6D；Piper在离线准备numeric state/action时完成转换，训练直接读6D；ManiSkill使用独立仿真位姿转换，原专家replay/坐标检查已留证。这不等于训练已经充分，也不表示所有真机TF/相机/TCP问题已排除。

本次官方权重的区别是**发布权重与本地接口的配套问题**。在相同本地图像、XYZ、夹爪、指令和时间窗口下，仅换state/GT旋转编码，完整20步平均旋转误差从29.54°降至0.84°、位置从26.74降至5.81mm。当前数据明确支持采用原版reader/default adapter的配套数组解释；不能将套用本地重排后的大误差当作官方模型能力成绩。

这也不证明官方训练时“用了错误四元数”：只要训练编码、推理上报和执行解码相互配套，不同边界编码可自洽。官方原训练数据的逐条数组来源仍没有公开manifest可核验。min/max统计相同也不能证明旋转编码相同，因为旋转范围被固定为[-1,1]。

## 数据、视频与复核

- 主结果：`summary_legacy.json`、`aligned_metrics_legacy.csv`；物理重排探针：`summary.json`、`aligned_metrics.csv`。后者不能冒充官方默认成绩。
- 全部对照：`COMPARISON_ALL_CONTRACTS.md`；原始窗口JSON在`official_legacy/n10|n20|n30/任务目录/`与`official/n10|n20|n30/任务目录/`。
- `input_contract_audit_legacy.json`：80窗口通过实际原版读取函数、归一化state、GT和文本校验；与前一轮Aloha保持相同图片SHA/XYZ/夹爪/文本，仅旋转编码按接口有意不同。三步数输入完全配对。
- `input_contract_audit.json`：另一组80窗口通过本地物理重排合同，与上一轮Aloha完整raw20状态/GT/指令相同。
- `checkpoint_identity.json`与`combined_video_audit.json`分别保留发布文件SHA和6并排视频的解码/帧数证据。

建议先看官方默认接口与Aloha42960同窗口20步的两个视频：

- [lift_pot](../outputs/analysis/official_alpha_jitter_20261105/lift_pot_official10740_legacy_vs_aloha42960_n20.mp4)
- [beat_block_hammer](../outputs/analysis/official_alpha_jitter_20261105/beat_block_hammer_official10740_legacy_vs_aloha42960_n20.mp4)

所有视频都是离线训练观测/目标曲线对照，8窗口拼接、按窗口缩放，不是策略驱动物理的rollout；不从播放速度或段间图像跳变推断实际抖动。两临时模型已结束，未修改任何本地训练/默认部署或启动新训练。后续分析优先使用官方默认接口结果，保留另一接口作为合同失配对照；增加epoch能否改善剩余高频误差仍需同模型受控续训。
