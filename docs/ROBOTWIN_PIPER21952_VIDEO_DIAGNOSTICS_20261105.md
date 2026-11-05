# Piper21952 视频诊断

2026-11-05用户要求并行录制杯子放盘子及若干低成功率任务，不改正式结果。

权重、完整Sidecar、cache/denoise/chunk/EEF/相机/专家筛seed/RT/动作预算/成功判据均沿用当前Piper21952标准评测。独立目录：outputs/robotwin_eval/piper21952_video_diagnostics_20261105。每任务Clean和Randomized各3条，6任务12条件共36条，均为官方专家筛选后的模型rollout。

place_empty_cup为杯子放杯垫；place_container_plate为杯子或碗放盘子，官方随机选择物体，不为得到指定物体而修改任务分布。

选取时正式100条结果：

|Task|Clean|Randomized|
|---|---:|---:|
|place_empty_cup|83%|84%|
|place_container_plate|99%|99%|
|lift_pot|59%|83%|
|beat_block_hammer|58%|68%|
|turn_switch|60%|80%|
|move_can_pot|89%|66%|

只在本实验私有runtime将piper_demo_clean/randomized的eval_video_log设为true，使用官方FFmpeg记录；camera gate发现录像启用后自动保留完整RGB采集，模型动作chunk协议保持。未修改共享piper_protocol.py、run_piper_native.py、默认无视频配置或正式评测。复用原服务空闲端口43512/43532/43513/43533，4路worker，GPU6/2/0/7按6GiB空余启动。主评测活跃session不复用；4是并发目标，非声称均已容纳。

原生视频位于eval/jobs/<mode>/<task>/attemptN/runtime/eval_result/.../episodeN.mp4。每条成功/失败、有效seed、控制步数见对应client.log；不将专家筛seed失败算模型失败，也不把3条诊断平均代替100条标准结果。未保证这3条必然包含失败，若缺失败样例再根据实际结果追加诊断。初始处于加载状态，不宣称已有可播放视频。

下一步：确认FFmpeg视频实际生成和可解码，整理成功/失败视频索引及可观察故障；视频表象无法单独证明坐标、IK或视觉的因果根源。


启动补充：为先拿到盘子任务视频，利用GPU5实时约10GiB空闲，另开place_container_plate/Clean优先3条，独立plate_priority目录、43515会话、PID3909530；与36条队列不合并计分，至多39条诊断。Randomized杯垫初次planner初始化缺right_planner已由原监测终止并重试，不当模型失败。视频索引VIDEO_INDEX.md及video_index.json自动更新，便捷链接放videos/<task>/<mode>/seed*_success/failure_*.mp4；生成中的未闭合视频不列已完成。


2026-11-05T04:19:39.552397+00:00：用户允许本机L20 117录像并要求不影响4090正式评估。已只终止本诊断远端controller及runner（3905413/3909530/3910723），不触及正式评估。旧录像尝试和未闭合文件保留，不作成功视频。新36条全部迁至本机l20_eval，controller81310，GPU5/3/0/1、独立会话43514/43534/43511/43531、7168MiB起步；共享Piper模型推理有轻量并发请求，不承诺正式推理延迟绝对不变，但4090没有新增录像仿真负载。已解决远端ffmpeg入口缺失：现有imageio_ffmpeg静态7.0.2新增到模拟Conda bin的ffmpeg软链接，远端-version通过，无下载、不修改编码参数或模型。索引由index_l20_videos.py维护，正式录像视频仍待实际可解码验收。


2026-11-05 07:03 UTC：L20诊断12条件全部完成，36条32成功=88.89%，两域各16/18。全部36个正式诊断视频经ffmpeg完整解码通过，证据video_decode_audit.json。杯垫和容器放盘子各6/6成功；小样本不能代表正式100条结果。低成功率任务获得4条失败视频，索引按success/failure标注；旧4090优先录像额外1条与36条主诊断分开，不计本结果。后续可根据失败视频做动作阶段分析，尚不唯一归因模型、抓取几何或控制。
