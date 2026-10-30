# 当前heads42960结束后排队32k补70

记录时间：2026-10-28T04:10:29.792364+00:00。用户明确要求保留此前32k各条件30条，排在当前heads-only42960之后补每条件70条。队列按这一原始要求准备，未将无答复当作同意更大预算的重跑100。已通过异步选项向用户说明：若要严格一次性官方100条，应改为每条件重新100，新增10000条；收到明确更改后才替换本队列。

## 已排队与范围

后台CPU监督PID **3459612**，每15秒检查当前heads42960，只有100/100条件真实complete且全部原生结果复验通过，才停止其自有模型/隧道并部署32k。当前42960继续运行，不为下一实验抢GPU，不影响其他训练/DiT4DiT/baseline。

权重：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_32000.safetensors`。SHA256 `2a74859c3c36aa8d9e21af3d98e034230ca367ada0b61403c33f14f86a3adeb9`。源第一轮评测：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward32000_4090_all100_30ep_20261025T073709Z`，原完整3000条2133成功71.10%保持。

新目录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward32000_supplement70_4090_20261028T035728Z`。监督状态`auto_transition_state.json`，活动日志`auto_transition_v2.log`，进程`auto_transition.process.json`。本队列未使用任何新GPU模型或仿真。

目标沿用当前**6份L20模型、9路4090仿真**：切换时读取实际副本和安全容量策略，实时显存不足会降低准入；模型>=26000MiB才部署，沿用当前卡优先，最多6份×3独立端口。当前GPU5额外进程已低于2GiB触发保护而禁用，未来不因瞬时空闲盲目恢复。保护/缓存/原生RT/专家筛seed/成功条件/动作预算不变，不存视频。

## 30＋70怎样计数

50个任务×Clean/Randomized两域=100条件。每个原30条完整native exit0及accepted seed清单重新核验；每条件从最后已接受seed+1开始继续官方专家筛选，跑满70条。候选seed连续推进，接受seed可以因为专家筛选有间隔，但必须有序且与原30互斥。并非从100000重复测70条。

- put_bottles_dustbin / demo_clean：原最后seed=100040，补测候选seed从100041开始。
- put_bottles_dustbin / demo_randomized：原最后seed=100045，补测候选seed从100046开始。
- open_microwave / demo_clean：原最后seed=100039，补测候选seed从100040开始。
- open_microwave / demo_randomized：原最后seed=100039，补测候选seed从100040开始。

新增7000条，合并原3000得到10000条；每条件成功数直接相加后除100，不将旧30条SR外推为100。只有完整30＋完整70进入正式100条SR，任何中断补测保留日志、重新从其固定起点跑满70，不拼接中断partial。原30目录不修改、不覆盖。每条补测的run_native通过进程内适配st_seed并使用既有70条分母覆盖，未改官方checkout或全局wrapper。

控制器和最终审计双重检查：新accepted70唯一/有序/起点正确、与旧30交集为空、合并100唯一；最后逐份重新读取两段真实退出码和native结果。长动作预算条件先开始，共享弹性队列，减少长任务尾部拖延。

## 重要口径：不等同一次性官方100条

官方`script/eval_policy.py`在生成指令时调用`generate_episode_descriptions(..., test_num)`，其第三参数是`max_descriptions`。因此原30、补70、一次性100的候选指令上限分别30/70/100。即使seed不重复，原30+补70也不是“逐项输入与官方一次100完全一致”。`now_ep_num`的重置在所查Base_Task中用于保存索引，不改变任务对象采样，但指令池上限差异确实存在。

本队列报告会明确标为**每条件100条的分段扩展评估（30+70）**，保留原生专家、成功/预算、RT及checkpoint一致，不声称严格指令池等价。旧30条没保存实际指令，无法事后将它们变成100设置下的输入。若用户选择严格官方100，应从候选100000以test_num100完整新测；旧30保留作历史，不纳入新100均值。

## 已完成的准备检查

- 100条件唯一、原3000条2133成功和seed完整清单保留；100个seed续接与错误分母拒绝检查通过。
- 真实既有`_install_test_num_override`与续接适配联合CPU检查返回正确起点与70条，不仅测试参数字符串。
- 原32k与当前4090实际runtime的demo_clean/demo_randomized/相机/embodiment/动作预算五个配置SHA相等；不会随意切资产/渲染标准。
- 模型端重复/同checkpoint跨副本和真实RGB/state动作验收由切换后执行，参考原32k的固定动作canonical；RAM权重/config/stats/contract重新哈希。
- 此前自动回收的进程退出竞态已修复，使用pidfd和退出幂等处理；新排队脚本也对已消失进程安全返回，不复用旧缺陷。

当前仅证明排队准备与CPU合同，并不证明新70条已执行。实际完成后写`final_complete100_independent_audit.json`、`final_results.json/csv`和`docs/ROBOTWIN_FULLBACKWARD32K_COMPLETE100_FINAL_RESULTS_20261028.md`。候选指令口径边界、near/EEF replay与严格配对等既有未解项持续保留。


### 2026-10-28T06:53:32.244020+00:00：heads最终70.37%，32k补70真实接续运行

heads于06:06:24完成100条件3000条2111成功70.3667%，Clean1351/1500=90.0667，Random760/1500=50.6667；对官方+1.3267pp、对原fullback42960+0.3667pp、对fullback32k30条基准−0.7333pp。保留10L20+90RTX混合硬件及30条指令池口径。独立100份验收通过后自动回收自有六服务；32k六份/18端口全部通过旧canonical和4090真实动作验收，06:14:47续测70队列已真实运行，9路。

当前补测56/7000，保留原3000后累计3056/10000；未满70的条件不计完整100条SR。实际9路日志均已完成首个episode且候选seed起点证据正确，已接受seed大于原last且无重叠；例如put_bottles_dustbin Clean100040后从100041接受，Random100045后从100046接受，open_microwave Clean从候选100040继续但专家排除后首接受100041。非所有整数都必须接受；候选延续与专家筛选均保留。基础设施错误0，分段指令上限30/70不同于native100的既有边界继续注明。


### 2026-10-28T08:22:30.450959+00:00：用户选择保持当前32k分段协议，扩至11路并监测吞吐

用户明确“先按照当前这个来”，故不实施待确认的新增70候选池100方案；当前原30池30＋补70池70/seed续接/70分母保持，差异影响大小未闭环量化，不据用户判断称已证明很小。L20仍六份32k模型、GPU1/2/4/5/6/7，每份3session合18端口，均ready/进程存活。

扩容前实际滚动吞吐15/30/60min分别100.63/98.36/99.17条/小时，全部只算新补测、不把原3000条当本轮吞吐。首批为1700/1500/1200步长预算，不能拿此100/h直接外推全部剩余任务。近30min显存最小空闲：GPU1=10482、GPU4=7551、GPU5=8849恒定、GPU6=8812MiB。按明确尽量并行授权，GPU1在>=10240MiB现场门槛下2→3，GPU5依原单路>=8704MiB门槛0→1，其他不加，总9→11。GPU5历史保护触发2032仅比2048阈值低16MiB、无实际OOM；本试验仅GPU5阈值1792MiB(1.75GiB)，其他2048，若仍低会回退0并保留partial/整70重跑，不改模型或动作判据。

只替换单个CPU容量监督，不重启服务/dispatcher/原9路。为支持GPU1两层额外进程，guard在内存低或renderer错误时3→2→1逐层退、每次只停最新自有extra，pending_stop在dispatcher记下退出前禁止重复信号/连砍；相应tier CPU回归通过。保留GPU4/6已有额外路保护。新增GPU1 stack_blocks_three Random及GPU5 stack_bowls_three Clean实际PID已启动并验证渲染亲和性，尚初始化无新完成episode，不称峰值已经证明稳定；原9任务PID相同或自然完成，100条件唯一、错误0。证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward32000_supplement70_4090_20261028T035728Z/gpu1_gpu5_extra_trial_20261028T081709Z，原代码/config/status/policy/process完整备份。

后续CPU guard重启必须先查看rollback事件及实际当前cap，已回退cap需校正trial配置后再启动，不能因清理后瞬时空闲直接重新启用已失败的级别。GPU1第三路/GPU5单路的稳态与峰值仍待监测，不承诺9→11会线性增加吞吐；用现有progress_history和本次activation时间做15/30min后观察，任务组合不同不称严格对照因果。
