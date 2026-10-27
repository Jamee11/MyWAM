# Full-backward warmup2k20k：RoboTwin每条件30条评估

更新时间：2026-10-24T05:21:56.122895+00:00。七份模型/21路跨机动作验收已通过；4090标准仿真队列已于2026-10-24T05:24:39.708159+00:00启动，当前8路真实policy动作推进，已有成功终态。

用户指定权重：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_20000.safetensors`。SHA256：`0295ed0bebb304929bf0e6baaa94421242440979b1796ccf502c687b01a4127c`；精确20k独立固定目录，自己的config/stats/Sidecar contract。与24k归一化统计SHA相同。

50个task × demo_clean/demo_randomized ×30条 =3000条。完整Sidecar三tap7/15/23；compile=false，cache=true（.99/max3），10步去噪、32动作chunk；与24k最终有效配置一致。不存视频。官方expert筛seed、unseen、seed0、原成功判据/动作预算、RT32spp/OIDN、near.1保持。

本轮模型在L20 GPU0–6，仿真在4090 host55fda0705558。23900/23901/23902至23960/23961/23962，独立token与结果目录。七份模型必须通过本checkpoint fixture重复/跨副本精确一致、本机21路与远端21路真实PNG/state→20D动作验收；验收不等于仿真rollout。

动态共享100条件队列，从七份不同模型开始分配。初始显存预算7680MiB/路+1024MiB整卡余量，perGPU caps1/1/1/1/2/1/1/0；实际并行还受空闲显存约束，不终止未知作业。使用24k最终V3的ANSI剥离、episode/action进度看门狗（45min无进展）、启动信用、端口独占保护；全新队列无旧attempt接管。基础设施异常至多3attempt，失败partial不合并，完整30条才计SR。

输出：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward20000_4090_all100_30ep_20261024T052041Z`。模型部署：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward_warmup2k20000_deploy_20261024T051552Z`。source_evidence.json保留与24k运行脚本的差异/上游commit。live_progress.json记录进度，comparison.json仅完整条件按相同任务子集与官方/24k比较，结束复验后写final_results.json/CSV。动态任务列表见[实时表](ROBOTWIN_FULL_BACKWARD20000_30EP_LIVE_20261024.md)。

24k已完成：Clean87.13%、Randomized49.27%、整体68.20%；官方相应89.36%、48.72%、69.04%。新20k为前2k步完整反向但延迟WAM更新，旧24k为isolation warmup2k；训练步数亦不同，因此差值不能单独归因warmup方案。训练near.02与官方eval.1、cache近似、不同GPU/seed未逐条配对及每项30的小样本边界继续保留。未修改模型、训练、数据或上游官方源。


部署补记：GPU7在加载阶段遇到外部作业显存上涨，本次服务OOM退出，失败日志保留，暂用GPU0–6七份模型/21个独占slot承载队列；不计作任何任务失败。并行仿真数独立受4090显存限制，初始实际7路（4090 GPU1空闲降至7696MiB，自动等待容量）。


2026-10-24T05:28:21.325496+00:00 实机仿真验证：8路均已实际推进policy action，首个成功case为beat_block_hammer/demo_clean（seed100000，111步完成）。调度历史异常0；尚无完整30条条件，不能据此判断总体成功率。证据：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward20000_4090_all100_30ep_20261024T052041Z/first_rollout_acceptance.json`。后台实时报告与同任务对比持续更新。


2026-10-24T05:41:09.762356+00:00：扩容已实际生效，GPU1=2、GPU4=3，总10路，新增双域put_bottles_dustbin真实动作推进；当前infra errors=0。2s资源守护低于2GiB或渲染buffer异常自动恢复原cap，只停止新增自有进程。证据`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward20000_4090_all100_30ep_20261024T052041Z/gpu14_capacity_trial_20261024T053746Z/action_acceptance.json`。
