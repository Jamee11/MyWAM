# PatchWAM C2R ramp100 step16000 快速评测

用户于2026-10-29授权有空闲资源时部署测试。独立根目录：`outputs/robotwin_eval/patchwam_ramp100_16000_20261029`。

checkpoint：`outputs/sidecar/aloha50_patchwam_c2r_ramp100_10epochs_20261027T144939Z/F_patchwam_ramp100/runs/2026-10-27_14-57-15/checkpoint_step_16000.safetensors`。

固定SHA256：`1f1d8d5fdcbd7277bf33855a77bec3305d58eae50a33217fbcfa0d0696dc8c9c`。将指定文件复制到独立内存盘并逐字节校验，目录只含16000权重，避免训练后续产生新checkpoint导致自动选错。训练config/normalization/contract/tokenizer保留；训练不动、不下载。

当前L20无完整单卡容量。已验证旧恢复条件stack_blocks_three Randomized完整70成功退出，才停止其专用旧恢复模型PID3516214；六份32k主服务和在途主评测保持。复用已验证容量适配：CUDA_VISIBLE_DEVICES=3,0，主体GPU3、BF16 T5 GPU0；无量化、CPU推理或模型结构改变。上限主体16GiB、T5 12.5GiB。新模型PID/时钟在model_process.json（含私有token，不展示内容）；cache开启、compile关闭、去噪10、horizon32、384×320三视图拼接、绝对EEF合同与当前评测一致。

## 验收与自动启动

prepare_model.py复制/验证后启动serve_split.py，复用现有serve_parallel、Sidecar fast runtime和原eager gate；真实两个fixture重复生成，检查完整Sidecar taps7/15/23、cache存在、eager无编译及有限/确定性动作。新canonical仅属于本checkpoint，不与32k动作数值比较。

activate.py等待验收通过，三个新session各检查重复chunk32输出精确一致；只暴露独立36910–36912端口到4090，不复用任何在途32k session。之后自动启动4090上的schedule.py。验收失败则阻止启动，原日志保留。具体实际是否ready/仿真是否推进，检查eager_validation/result.json、online_acceptance.json、activation.log、scheduler.log、state.json和progress.json；本段准备状态不等同已完成rollout。

## 评测协议与调度

每条件30条，50任务×Clean/Randomized共100条件/3000条。保留官方专家筛seed、seed组0从100000起、unseen候选指令池30、官方成功判据/动作预算/RT相机与250Hz；无视频。原生chunk32传输＋已验证相机采集gate，仅缓存动作期间跳采图，保留_update_render及随机灯光RNG、真实proprio、逐步规划/物理/成功。

优先十个Randomized任务：place_mouse_pad、move_stapler_pad、place_phone_stand、place_fan、hanging_mug、open_microwave、blocks_ranking_rgb、pick_diverse_bottles、place_dual_shoes、handover_mic；再对应Clean，再其余全部条件。plan.json为精确列表。

新服务最多3路sim、独立session，使用4090当前空闲空间，不停止任何32k在途任务。动态计入主队列同卡进程；每卡主＋新最多2路，新实验自身同卡最多1路；实时空闲≥8704MiB才准入，启动保留7680MiB预算，剩余低于1792MiB或30分钟无进展仅停止自有新进程；新attempt整30重测，不混partial，每条件最多3次尝试。瞬时显存不代表峰值永久满足，不保证线性并发增益。主32k后续任务也会按自身显存门槛判断，可能共享GPU资源，不宣称完全无性能竞争。

正式SR仅统计完整30且validate_result通过的条件，partial只展示进度。与32k的对照应优先取其原30条相同条件；原30+70完整100可另列，但不混采样精度/指令池。官方Python random指令未统一固定，相同环境seed不保证确切语言相同，当前保留既有协议，不能声称严格相同episode manifest。近裁面/EEF实际与目标差异等原未决边界继续适用。

## 下一步

确认模型验收、跨机健康和实际action/首episode，再记录新实验进度和同任务对照；先看优先任务趋势，不用不完整条件充当100任务综合SR。

## 2026-10-29T10:23:14.375190+00:00 实际部署验收

新模型PID3605375，完整Sidecar/cache/eager检查通过；两真实fixture各重复两次全部逐元素相等。三个36910–36912 session的重复32动作在线检查均精确一致，representation=eef。跨机隧道已建立，4090调度PID220104。三路已启动并通过真实ModelClient健康检查：GPU5/place_mouse_pad Randomized/PID220202，GPU4/move_stapler_pad Randomized/PID220209，GPU6/place_phone_stand Randomized/PID220210。当前仍在专家/场景初始化阶段，尚无完整episode，不虚报SR。输入合同、相机布局、绝对EEF映射和归一化统计与32k逐项相同，augmentation不在推理侧应用。六原32k模型/在途任务保留；旧恢复模型仅在完整70已验收后退休。

## 2026-10-29T11:46:44.812795+00:00 显存反复中断修正

发现GPU5的新仿真多次因free1792MiB保护中止：place_mouse_pad三次、place_fan三次、blocks_ranking_rgb一次；不是模型策略失败，partial均不计入正式成功率。初始8704MiB准入未覆盖该卡真实峰值，原调度没有失败卡隔离，导致反复回到GPU5，已确认这是资源调度问题。

为立即避免更多重复损耗，保留state/source/progress/PID完整备份及所有attempt目录，仅停止新实验自有CPU调度和三个在途新sim（原生32k模型/进程均未触碰）；当前三个未完成任务的partial不拼接结果。两完整30结果move_stapler_pad Randomized9/30、place_phone_stand Randomized9/30逐份保留。原中断/人工重启历史在capacity_fix_backup_20261029T1145Z及各attempt中保留。

新准入策略admission_policy.json由调度每次读取：仅GPU1/4/6、最低10240MiB，仍主＋新每卡最多2路、新自身同卡最多1路，实际低于1792仍保护；本轮另给非完成条件3次基础设施尝试，历史attempt全部保留。调度PID281508；place_mouse_pad Randomized重测在GPU4/PID281520，place_fan Randomized重测在GPU6/PID281523，第三路等待安全容量。待32k自然结束释放卡后自动接续，不把瞬时空闲当永久峰值保证。

当前新实验仅两个完整条件、18/60=30%，都是先选的弱项；不能当100条件综合SR。主32k截至11:41已92个完整条件、6611/9200=71.86%，官方同条件70.88%；仍在原队列继续。

### 与32k原30条同任务比较

Randomized：move_stapler_pad，32k6/30=20%，PatchWAM16000为9/30=30%，+10pp；place_phone_stand，32k8/30=26.67%，新9/30=30%，+3.33pp。两条件合计旧14/60=23.33%、新18/60=30%，多4次成功/+6.67pp。仅两个弱项、小样本，不能推出全任务提升；两边保留原生30候选池协议，但确切语言未统一冻结。32k move_stapler_pad完整100为18/100；place_phone_stand完整100此刻尚未结束，不能用其中间值冒充终值。

### 2026-10-29T13:08Z：前四个Randomized条件完整30结果

新16000：place_mouse_pad7/30、move_stapler_pad9/30、place_phone_stand9/30、place_fan19/30，合计44/120=36.67%。32k同任务原30分别9/30、6/30、8/30、12/30，合计35/120=29.17%，新多9成功/+7.5pp。mouse_pad低2成功，其余高3/1/7，不能据四个优先弱项推出全任务提升。隔离GPU5后本轮新错误0、三路在测hanging_mug/open_microwave/blocks_ranking_rgb；总观察146条，含partial不用于完整条件均值。32k主99/100条件完成、总9982/10000，最后turn_switch Randomized剩18条。
