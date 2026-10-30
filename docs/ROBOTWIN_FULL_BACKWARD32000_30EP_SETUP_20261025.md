# Full-backward warmup2k32k：50任务双域各30条评估

建立：2026-10-25T07:39:20.523563+00:00。用户明确要求24k完成后评估本run最新32k，并尽量利用L20模型与4090仿真资源。已核对24k完整100/100、3000条1984成功=66.13%，只退出其八份空闲自有服务，全部结果/日志/权重保留，未知训练/其他任务不动。

权重：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_32000.safetensors`；SHA256 `2a74859c3c36aa8d9e21af3d98e034230ca367ada0b61403c33f14f86a3adeb9`；精确step32000固定目录，自己的config/stats/contract/tokenizer，RAM副本源/目标各SHA一致、源inode/size/mtime稳定。所有文件本地，无HF下载。

50task×Clean/Randomized×30=100条件3000条，不存视频；完整Sidecar三tap7/15/23，cache .99/max3开启、compile=false，10去噪步/32动作chunk。继续上游官方commit与原生expert筛seed/seed0/unseen、RT32/OIDN/near.1、原预算和成功判据，与前24k评估一致。不修改模型/训练/数据/上游及旧DiT4DiT评测。

L20八卡各启动一份完整模型，26900–26972共24个独立会话。GPU5加载前free25324MiB为较小余量，其他未知作业不终止；若外部负载使某份服务失败，保留该卡失败日志/报告，已通过验证副本继续，不把模型部署失败当任务SR0。两fixture各重复两次逐元素相等、跨副本严格相同、本机及4090所有实际启用端口真实PNG/state→20D动作验收之后才开始queue；不放宽数值门槛、不比较不同checkpoint应输出相同。

4090当前GPU1 free24189MiB、GPU4 free23210MiB，尝试各3路，GPU0/2/3/5/6各1路，GPU7余3730MiB暂不放；最多11路，实际受7680MiB启动预算+1024MiB整卡余量/首条完整episode释放信用限制。GPU1/4独立2s资源守护，低于2048MiB或渲染buffer错误回退至各2路，仅停最新自有额外sim，partial不混SR；基础设施最多3attempt。ANSI剥离、45min无episode/action进展看门狗、独占token/slot锁与渲染PCI断言沿用。不能以短暂低显存把卡塞满而反复中断。

新结果单独保存在`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward32000_4090_all100_30ep_20261025T073709Z`；部署`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward_warmup2k32000_deploy_20261025T073319Z`；pointer fullbackward32000_30ep_current.json。comparison.json只统计完整30条件，与官方同任务、已完成full-backward24k同任务对照；同时回溯旧初始Sidecar32k各任务原始前30条，原100条计数/成功/accepted seed均验证，已封存reference_initial32k_first30.json（共2000/3000=66.67%，旧全100条66.48%），新32k按同N对照，不能把仅相同N称场景/指令/RNG完全配对。这里old24k字段明确指full-backward24k而不是isolation24k。

source_evidence.json保留脚本SHA/diff/官方commit；live_progress.json与专用LIVE用于进度，后台原生完整30复验后自动输出final_results.json/CSV。当前模型加载/验收进行中，尚无新32k正式成功率。相机near训练.02与官方eval.1、cache近似、不同硬件/负载、训练预算/seed与非逐条配对的因果限制继续保留。


## 2026-10-25T07:41:58.818253+00:00：full-backward32k八模型/24端口验收通过，4090十路启动

八张L20各一份完整Sidecar32k，GPU5虽余量较小仍通过两个fixture各2次重复与跨副本逐元素一致、本机及4090全部24端口真实RGB/state→有限20D动作，SHA 2a74859c3c36aa8d9e21af3d98e034230ca367ada0b61403c33f14f86a3adeb9。dispatcher PID2394680，开始2026-10-25T07:40:06.467136+00:00。目前GPU1三路、GPU4两路，其余0/2/3/5/6各1路，总10路；GPU4第三路等待首条终态释放信用后尽量补到11。资源守护3/3→2/2回退门槛2GiB/buffer错误，未停止未知作业。此时仅验部署与队列启动，尚无完整30条件，不能报成功率。证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward_warmup2k32000_deploy_20261025T073319Z/final_deployment_acceptance.json。

同100条件前30条旧初始32k参考全量回溯完成：所有原100条计数/成功数/seed与canonical run.json相等，首30共2000/3000=66.67%，保留各条件seed；新watch_comparison同时列full-backward24k、官方和初始32k同N参考，old24k字段明确为full-backward24k，非isolation24k。样本数量对齐不等于严格场景/指令/RNG因果配对。下一步确认所有native真实动作与首条终态，再持续后台完成3000条。


## 2026-10-25T07:45:43.537355+00:00：full-backward32k十路真实闭环已验收

至少10路均出现真实policy动作，hammer Clean及alarmclock双域已出现成功终态；new32k SHA/独占端口/video关闭/unseen/EEF与实际render GPU PCI均通过，infra errors=0。每卡仿真数{"0": 1, "1": 3, "2": 1, "3": 1, "4": 3, "5": 1, "6": 1, "7": 0}，GPU4第三路按启动信用和显存门槛自动补；最多11，不人为降低渲染质量或预算。八份完整L20模型全部正常服务。first_rollout_acceptance.json位于/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward32000_4090_all100_30ep_20261025T073709Z。已有后台live_progress/同完整条件comparison/终态原生复验，未到完整30不输出阶段总体SR；旧24k/初始32k参考独立保留，所有旧训练与DiT4DiT流程未改。


## 2026-10-25T15:28:03.938755+00:00：用户再次要求利用4090空余显存，GPU1/4各补一路

启动前68.36秒历史窗口GPU1 min9758MiB、GPU4 min11405MiB，现有各2路，其他卡free605/2990/2062/1956/3958/3730MiB不再加。保留过去GPU4于10:28/GPU1于14:46三路时低于2GiB的回退记录，明确本次是用户再次授权的受监督容量试验，不把旧失败称已永久解决。新独立资源守护PID2540305（旧守护已退出，身份已验），使用原helper，live caps2/2→3/3；GPU4新增place_dual_shoes Clean、GPU1新增同task Randomized各30条，不存视频，独占原模型slot，无重复任务或改成功判据。总并行9→11，新两路目前初始化中，无新增infra error，初期低占用不当峰值证明；<2048MiB或本卡render buffer异常恢复2路并只停新增自有sim，partial不拼最终SR。证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward32000_4090_all100_30ep_20261025T073709Z/gpu14_reexpand_20261025T152635Z/launch_acceptance.json及samples.jsonl，新pointer gpu14_capacity_trial_current.json指向本试验，旧目录保留。后续跟踪首条policy/峰值和完成数，继续原100条件主队列。
