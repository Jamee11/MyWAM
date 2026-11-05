# Piper21952 / Aloha42960 与 ManiSkill20872 动作诊断对照

2026-11-05，用户授权各约10条实际训练轨迹，保存视频与动作误差/抖动，要求可与单臂诊断对应。独立输出：`outputs/analysis/robotwin_two_ckpt_jitter_20261105/`。

## 数据与协议

- Piper：指定midtrain_patchwam_2epochs最终21952；PreparedSimDataset的robotwin_piper/train，优先覆盖lift_pot、beat_block_hammer等原录像任务，10任务各1条Clean训练轨迹。
- Aloha：指定F_patchwam_ramp100最终42960；必须用实际Clean-50清单`outputs/sidecar/aloha_4350/manifest.ready.json`及EARSIGLiteDataset/LiteRoboTwinSource。不能把同名archive轨迹直接当作这份checkpoint的训练样本。
- 使用各checkpoint自己的归一化统计。原架构Full Sidecar、BF16、seed42、CFG1、cache与compile关闭、32点，分别10/20/30去噪。
- 每条轨迹均匀取8个完整32点窗口，输入为该窗口首帧训练拼图、真实proprio与原训练指令；GT未来只用于误差和视频显示，不输入模型生成路径。
- 对齐ManiSkill的XYZ向量RMS误差、空间二阶差分RMS、高频误差（SG窗口7/阶2）、二阶差分误差、反转、夹爪误差，以及同stats下BF16舍入GT。双臂分别统计，另根据GT路径选择主运动臂；不能让静止手稀释平均值。
- ManiSkill原诊断是96 train窗口/3阶段，本次是各10轨迹/8窗口；机器人、任务、action语义、stats和采样阶段仍不同，因此是统一指标下的诊断对照，不是严格单因素训练epoch实验。

## 视频含义

新视频是**离线专家观测下的预测对照**：左侧为真实训练轨迹图像，右侧绿线为专家目标、红线为预测目标。黄色文字明确标注NOT predicted physics。10fps只用于播放，不是物理控制频率。不能称为策略驱动物理的rollout，不能从这些视频统计任务成功率。

原Piper视频诊断入口使用chunk_transport与官方录像，没有启用逐步action/观测落盘。当前目录检索未找到可与四条指定失败视频逐帧对应的action/状态轨迹；run.json可确认结果/seed，mp4可分析画面，但不能从中精确恢复原模型预测。本次新实验保存每个窗口完整state/prompt/预测32×20/专家32×20与源轨迹身份，后续不再依赖视频猜动作。

## 资源与修改范围

独立进程Piper主体GPU3/T5 GPU4。Aloha原计划GPU1/2，CPU加载期间GPU1被其他作业占用，随后仅退休自己的CPU预加载，改为在GPU3/4接续Piper；不停止其他作业。模型加载与分卡复用现有部署代码，所有新实验代码只在outputs/analysis内。5901、现有正式评估、原模型/训练/数据/部署源码保持。旧尝试日志保留。

两个模型各n10/n20/n30均已完成，每组80窗口/10轨迹，共480次32×20预测、960臂窗口统计。60个原始视频帧数全部验收；另有2个lift_pot三步数并排视频，各2880×384、256帧。完整索引为`outputs/analysis/robotwin_two_ckpt_jitter_20261105/VIDEO_INDEX.md`。

## 三模型统一指标结果

RoboTwin下表为两条物理手臂分别计算再取均值，ManiSkill为单臂。RoboTwin每个去噪设置80个独立观测窗口（160臂窗口），ManiSkill原诊断96个train窗口。单位mm，粗糙度为目标位置空间二阶差分RMS，不是真机振幅。

|模型|去噪|位置向量RMS误差|预测粗糙度|专家粗糙度|高频误差|
|---|---:|---:|---:|---:|---:|
|ManiSkill20872|10|9.3814|5.7004|0.3104|1.8898|
|ManiSkill20872|20|9.2961|4.2764|0.3104|1.4031|
|ManiSkill20872|30|9.4577|4.7949|0.3104|1.5479|
|Piper21952|10|8.5690|5.8505|2.8850|1.7609|
|Piper21952|20|8.3736|5.0931|2.8850|1.4901|
|Piper21952|30|8.5243|5.2738|2.8850|1.5423|
|Aloha42960|10|4.2567|3.4671|0.9036|1.0900|
|Aloha42960|20|4.2351|2.8127|0.9036|0.8560|
|Aloha42960|30|4.3761|3.1675|0.9036|0.9568|

三模型本次都以20步最平稳，30步没有继续改善。Piper10→20高频误差下降约15.4%，Aloha下降约21.5%；位置RMS分别只下降约2.28%/0.51%。Aloha拟合误差与残差较低，但不同任务、数据、初始化、归一化、运动/时间分布均为混淆因素，不能把差异单独归因于训练epoch。

尤其不能只比较预测粗糙度：Piper专家本身的二阶差分比ManiSkill高得多。高频误差与二阶差分误差均按预测减对应GT计算；逐臂、GT路径较长一臂、旋转误差、夹爪误差、原GT及BF16舍入GT均保留在`summary.json`/`aligned_metrics.csv`。

三个checkpoint的阶段epoch分别1/2/10，micro/optimizer步分别20872/10436、21952/10976、42960/21480。Aloha还有此前初始化/预热阶段，不能把epoch数当作等比例优化预算，更不能把此次跨模型对照当成同一个模型续训实验。

## 静止目标也有高频回摆

按GT整32点路径<1mm选取近静止臂窗口：Piper每设置60个、Aloha68个，GT粗糙度仅约0.0002/0.0001mm。

|模型|去噪|近静止目标粗糙度mm|高频误差mm|目标反转次数均值|
|---|---:|---:|---:|---:|
|Piper|10|4.7253|1.5308|20.52|
|Piper|20|3.5649|1.1375|17.02|
|Piper|30|3.8371|1.2281|17.77|
|Aloha|10|3.0047|0.9864|23.44|
|Aloha|20|2.2558|0.7326|23.46|
|Aloha|30|2.6535|0.8326|25.07|

所以两份RoboTwin权重也存在额外的时序残差，任务成功不等于预测目标完全平滑。Aloha回摆幅度更小，但反转次数可以更高，不能单独按反转次数排序质量。是否变成实际机械臂抖动仍取决于执行器/轨迹组织，不能用此表代替连续TCP/joint反馈。

各自统计下，对GT做BF16归一化舍入后的粗糙度均值，Piper2.9700mm、Aloha1.0393mm。它们与ManiSkill的量化背景不同；本次没有对两份RoboTwin权重做末步FP32生成干预，因此不把这些统计当作数值原因的完整因果占比。

## 原视频与新视频如何使用

旧的lift_pot seed100005及3条hammer失败有原生结果/seed和视频，但此次没有找到逐步action、状态或HDF5/NPZ/NPY/JSONL轨迹。不能把新选训练episode的预测当作旧失败当时的预测；要定位旧case的接触/抓取失败，需按相同case重新运行并保存观测、动作和实际执行反馈。

新视频左侧是专家未来图像，右侧是模型/GT的XZ目标曲线。**每个视频拼接8个独立且可能重叠的窗口；段间图像跳变/重复来自切换窗口。图按各窗口范围缩放，幅值以标注和CSV为准。** 这些不是策略控制机器人后生成的物理视频，也没有新的任务成功率。

先看同任务三步数并排：

- [Piper lift_pot](../outputs/analysis/robotwin_two_ckpt_jitter_20261105/piper_lift_pot_compare_n10_n20_n30.mp4)
- [Aloha lift_pot](../outputs/analysis/robotwin_two_ckpt_jitter_20261105/aloha_lift_pot_compare_n10_n20_n30.mp4)
- [全部62个视频](../outputs/analysis/robotwin_two_ckpt_jitter_20261105/VIDEO_INDEX.md)

## 验收与结论

`input_contract_audit.json`：passed=true，160个不同窗口的完整80D归一化状态、32点标签反归一化及指令匹配实际训练reader；三种去噪设置的原状态/指令/GT逐元素相同，首帧像素SHA可追溯。Aloha的实际训练源文件大小/mtime通过旧清单检查。模型checkpoint/normalizer完整SHA保存在`checkpoint_identity.json`。

结论：RoboTwin两份权重与单臂共享“增加到20步有所改善、30步未再改善”的模式，且即使专家目标静止也会输出回摆。Aloha当前样本中的残差较小；这支持继续检查动作生成数值/时序残差及执行器对它的处理，不能以RoboTwin可完成任务推断ManiSkill只是epoch不足。额外训练是否能改善ManiSkill，仍需同模型、同数据/归一化和固定指标的续训对照。本次未训练新模型、未改共享模型/训练/部署逻辑，两个临时模型进程均已正常退出。
