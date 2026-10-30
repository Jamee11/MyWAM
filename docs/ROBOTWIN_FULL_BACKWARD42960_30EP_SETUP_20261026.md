# Full-backward warmup2k最终42960：全50任务双域各30条评估

建立：2026-10-26T01:12:02.298834+00:00。用户指定本run最终checkpoint_step_42960，在32k完成后切换，并明确要求L20仅部署模型、所有仿真均4090。当前未启动任何L20仿真，继续原跨机方式。

权重：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_42960.safetensors`；SHA256 `f57615f4fe3a8c83be0d5bc1c86f13adcb2baa2807d00f1a22519a1025e21c6d`。精确step42960固定目录，自己的config/stats/contract、同run tokenizer，RAM副本源和目标独立SHA一致、源inode/size/mtime稳定。权重本地已有，没有HF下载。32k100×30=3000条2133成功71.10%，其全量结果与原始日志/权重保留；只退出其八份自有空闲模型服务。

范围仍为50task×Clean/Randomized×30=100条件3000条，不存视频。完整Sidecar三tap7/15/23、cache .99/max3开启、compile=false、10去噪步/32chunk。保持相同官方checkout/隔离Conda、native expert筛seed/seed0/unseen、RT32/OIDN/near.1、原动作预算/成功判据。没有改模型/训练/数据/上游或DiT4DiT评测。此轮是每条件30条全任务覆盖，不能称已完成官方每条件100条复现。

L20 GPU0–7各启动1份完整模型，27900–27972共24独立会话。规范参考在本checkpoint内部首个通过重复推理的副本原子建立，其余重复/跨副本严格逐元素相同，再验本机及4090全部实际启用端口真实RGB/state→20D动作。若GPU受未知作业影响无法就绪，保留失败副本证据，已验证副本继续，不放宽数值阈值或把部署错误算作policy失败。L20没有仿真进程；暂定多host空目录在未启动时已重命名为4090-only，不存在L20评测结果混入。

4090动态共享100条件队列，caps1/3/1/1/3/1/2/1。当前GPU7 free3730MiB不足门槛，不会派发；GPU6 free19771MiB可尝试2路，GPU1/4各尝试3路，现有资源目标最多12路，若GPU7之后自然释放且达到门槛才可能13路。不终止未知作业。每路启动预算7680MiB+1024MiB卡余量，信用到首条完整episode后释放；GPU1/4/6独立2s守护，<2048MiB或renderer buffer异常分别回退2/2/1路，仅停最新自有额外sim，中断partial不并SR。其余ANSI剥离、45min无episode/action进展、token/slot独占和CUDA/render PCI断言沿用。

新输出：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward42960_4090_all100_30ep_20261026T010703Z`；模型部署：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward_warmup2k42960_deploy_20261026T010521Z`；pointer fullbackward42960_30ep_current.json。live_progress.json用于进度；comparison.json只对完整30条件与官方同任务、同run完整32k相同子集、旧初始32k首30相同子集对照。previous32k字段明确指本run full-backward32k，不是初始32k。终态再次原生验证后自动写final_results.json/CSV。source_evidence.json保留脚本SHA/diff/官方commit。

当前模型正在加载/验收，尚无42960正式成功率。最后权重未必优于32k，等待完整结果再判断；同N不保证accepted seed/场景/指令/RNG相同，训练预算/运行负载等边界继续保留。每条件30仍有采样不确定性，不能单凭数值高低宣称单因素因果或显著性。


## 2026-10-26T01:14:44.474015+00:00：最终42960七模型/21端口验收通过，4090十路启动

GPU5加载阶段OOM，原日志与deployment_with_failed_replicas.json保留，未关闭其未知作业或降低模型能力；其余L20 GPU0/1/2/3/4/6/7七份完整Sidecar已通过本checkpoint重复/跨副本逐元素一致、本机/4090全部21路真实RGB/state→20D动作验收。bootstrap自动排除失败副本并将队列endpoint缩至21，模型失败不计task SR0。仅进度reporter纠正GPU/端口计数文案并重启，不重置模型/调度/仿真；生产核心代码与训练不变。

用户最新明确只在L20部署模型、全部4090仿真，现10路：GPU1/4/6各2路，其余0/2/3/5各1路；GPU1/4第三路等启动信用与显存容量，GPU7不足不派发，最多当前12路。dispatcher PID2723976，开始2026-10-26T01:13:03.654434+00:00；尚在场景初始化，未报正式成功率。证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward_warmup2k42960_deploy_20261026T010521Z/final_deployment_acceptance.json；后续确认全部实际policy动作，继续后台30×100。


## 2026-10-26T01:18:55.597650+00:00：最终42960十路真实闭环通过，4090已扩至十二路

当前12路4090仿真，10路实际policy动作推进，新补GPU1/4各一路初始化中；已有成功终态，video关闭/new42960 SHA/unseen/EEF/独占端口及已生成render PCI证据逐路通过，infra errors=0。每卡数量{"0": 1, "1": 3, "2": 1, "3": 1, "4": 3, "5": 1, "6": 2, "7": 0}。L20严格只有GPU0/1/2/3/4/6/7七份模型，无L20仿真；GPU5加载OOM证据保留，未停未知作业，七模型21端口均验收通过。GPU1/4/6守护2GiB或buffer异常回退2/2/1路，继续标准100条件各30条共享队列，partial不混SR。证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward42960_4090_all100_30ep_20261026T010703Z/first_rollout_acceptance.json；专用LIVE已按实际就绪模型更正文案，dispatch未重启/模型未reset；最终按同条件与本run32k和官方对比，不预判最终checkpoint更优。
