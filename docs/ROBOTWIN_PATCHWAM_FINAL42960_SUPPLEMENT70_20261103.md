# PatchWAM最终42960：保留30条，补70条至每条件100条

用户2026-11-03授权现在利用4090空闲显卡启动，原速测收尾不停止。

## 范围与推理协议

权重：`outputs/sidecar/aloha50_patchwam_c2r_ramp100_10epochs_20261027T144939Z/F_patchwam_ramp100/runs/2026-10-27_14-57-15/checkpoint_step_42960.safetensors`。同一RAM权重与deployment SHA，复用三份真实live/原canonical验证通过的split模型：GPU3/T5 GPU4、GPU2/T5 GPU6、GPU1/T5 GPU4。

50tasks×Clean/Randomized共100条件，各保留原30条、新增70条，最终10000条（新增7000）。完整Sidecar、Eager、DiT cache .99/max3、10denoise、sync32、384×320、官方RT/专家筛seed/成功判据/动作预算保持；无视频，保留策略RGB与已有chunk/camera优化。

旧30条：`outputs/robotwin_eval/patchwam_ramp100_final42960_all50_30ep_20261103`。
新补测：`outputs/robotwin_eval/patchwam_ramp100_final42960_supp70_20261103`。

## 接续与有效性

复用之前21480/32k补70条已验证的`continuation_adapter.py`。每个条件等原30条完整有效后，设置native evaluator的st_seed=原accepted_seeds[-1]+1，然后继续官方专家检查，收集70个有效seed，不是简单把整数seed+30。未完成的原30条不猜起点，不提前启动该条件的补70条。

最终要求每条件前30/后70各自native exit0、条数完整、渲染日志无错误；70条seeds递增/唯一、起点不小于continuation_seed，和原30条无交集，合并100唯一。失败基础设施不计SR，保留日志并重试。恢复planner缺left/right_planner错误循环的逻辑直接纳入新调度monitor，不必等持续增长日志耗尽时间；正常规划/成功标准不改。

这是**native30+native70的100条接续评估**。两段保持官方seed/专家协议与各自实际分母，但语言RNG与候选指令pool cap30/70不同于一次连续native100；不能宣称逐位等价。原30条已产生、需保留时不能追溯补回连续100时的原始语言RNG状态。本次明确记录该边界，不重写原结果。

## 与30条收尾并行且不共享live会话

原调度器对40端口采用“第一个空闲端口”派发，当前未完成条件K=8。因此以后每次派发时最多K-1个原任务占用端口，第一个空闲端口一定在前K个以内。补70条固定使用原端口顺序中index>=8的后缀，并动态排除当前仍在运行的旧30条端口；由此无需停止原控制器/仿真，也不需要让两条仿真共享Policy/action deque。原100条件全部complete之后，40会话均可给新队列使用。

若原30条队列被外部新增/恢复到未完成>8，补测派发会报错，需先重新分配端口；本次不会自动扩张旧队列范围。源文件顺序保存于experiment.json，不能随意调整旧服务端口优先级。当前源队列和补队列分别只有一个controller。

实际进程数由4090实时显存决定，6144MiB启动占位180秒，之后以实际余量派发，无运行中低余量强杀。40独立会话是可用容量，不是保证40路仿真都能容纳。

## 检查与启动记录

- contract_checks.json：8端口/3保留穷举28种占用、40端口/8保留随机2000种占用验证未来原任务端口与补测不交叉；30+70唯一通过、重叠/重复拒绝；实际原continuation_adapter只覆盖st_seed、保留其他参数、拒绝非70条；全部新脚本AST通过。
- startup_source_checks.json：三份模型PID/start_ticks、checkpoint SHA与原native fixture验收一致；92个已完成原30条的native run证据匹配，尚缺8条件。
- 2026-11-03 14:04 UTC在4090启动scheduler610189、report610190；进程身份见remote_processes.json。
- 14:05 UTC16路补测进程已启动，无controller异常；首条rollout尚未完成，仿真初始化中，不能把running状态当成功率结果。初始place_mouse_pad/Randomized continuation_seed100033，原最后有效seed100032。

新run中的`comparison.json`只汇总已经完成30+70的条件；最开始可能0条件，不能把70条部分成功率与已合并100条混用。`state.json`看70条队列/活动PID，`jobs/.../continuation_seed.json`是原生入口实际安装的接续起点，`client.log`看实际rollout进度。最终report重新验收前30与后70全部原生日志，写final_independent_audit.json，目标100条件/10000条。

## 下一步

确认原生70条入口实际接续起点和首个动作/rollout，持续监视真实进度及初始化错误。原30条收尾完成后按同一端口规则扩大至全部40会话，用实际GPU峰值维持有效并发；每次汇报只比较已完成100条的相同任务域子集，待全套完成再与其他完整100条实验对比。


14:06:42 UTC实际启动验证：17路补70条进程在跑，16路已生成原生continuation_seed.json，控制器无异常。place_mouse_pad/Randomized实际记录start_seed100033/test_num70，与原30最后100032对齐，原生已执行到step75/400；不再只是初始化状态。尚无完整70条条件成绩。


14:11 UTC首次完整100条ETA：原30队列95条件2850条验收、活动约100条；补70队列23路活动、当前attempt约43条，原30仍5路，总28sim。尚需约7007条实际推进（活动部分仍待最终验收）。原30同配置历史观测约450–600条/h，线性11.7–15.6h，给尾轮/重试后预计14–18h，即UTC11月4日04:00–08:00（北京时间12:00–16:00）。补测仅启动约7分钟、尚未稳定，后续根据实际70条完成吞吐修订；不是承诺完成时刻。证据eta_20261103T1411.json。


14:19 UTC进度：原30条速测96/100条件2880条有效完成，综合2224/2880=77.2222%，Clean全50条件1318/1500=87.8667%，Randomized46条件906/1380=65.6522%，4路收尾。补70队列24路活动、76queued、controller无异常，当前各有效活动attempt累计推进约127条；尚无完整70条条件，因此尚无可报告的合并100条条件成功率。两队合计28sim，显存多数已占满，基础设施失败attempt按原队列重试、无终态failed。不能把127条partial当最终有效条数。


17:42 UTC：原30条速测100条件3000条已完整完成（76.20%）；补70队列14/100条件完成980新增条，合并这14条件1400条1171成功83.6429%，对应原30条85.4762%。该14条件是10Clean+4Randomized，C92.60%/R61.25%，不与全套76.20%直接混排。当前25路活动、61queued、活动attempt另约675条尚待完整验收，原3000+新980=3980条已通过各自完整条件验收。独立report的完整100条条件进度14%，不等于全部rollout只完成14%。部分renderer/初始化失败attempt自动重试；controller无异常，真实25进程推进。


17:46 UTC重算ETA：补测已跑3.70h，980新增条完成验收，当前活动attempt另664条；有效推进约444条/h（失败attempt丢弃部分不算，包含启动/重试耗时），剩约5356条。恒定吞吐线性12.05h，考虑尾轮并发下降与重试，剩13–16h，即北京时间11月4日15:00–18:00附近。当前26路活动；活动partial仍可能因基础设施失败重跑，ETA非承诺。证据eta_latest.json。


2026-11-04 02:09 UTC：补70已运行12.09h，71条件完成4970新增条（合并7100条5517成功77.7042%，对应原30同subset77.6995%，暂时几乎持平），26路活动/3queued/0终态failed；活动attempt另554条未最终验收。平均有效推进457条/h，最近1小时完整验收490条（7×70），最近2小时770条。剩约1476实际推进，恒定吞吐线性3.23h；预留尾轮与重试预计4–6h，即北京时间今天14:00–16:00，异常重试集中时可能延长。place_a2b_left/Randomized初始化attempt13约10分钟无新日志，仍在既有20分钟监测；其他日志大多持续推进，不把初始化等同终态完成。证据eta_latest.json。


2026-11-04 03:06 UTC：76/100条件已各100条，5809/7600=76.4342%，Clean3575/4000=89.375%，Randomized2234/3600=62.0556%。同条件原30条76.5351%，扩样差-0.1009pp；与此前71条件77.70%变化主要不可排除任务组成，不能把不同subset直接当扩样退化。新增70已有5320条完成，活动attempt另692条，剩约988实际推进；24路活动/0queued/0failed，controller610189live/无异常。全程观测有效推进461条/h，线性2.14h，但尾轮并发下降/初始化重试保留，预计仍北京时间14:00–16:00收尾。put_bottles_dustbin Randomized约12分钟无新日志，put_object_cabinet约8分钟，仍由20分钟进度监测处理，不把尚未完成初始化计为成绩。

2026-11-04 07:32 UTC：95/100条件已各100条，7284/9500=76.6737%；Clean49条件88.7551%，Randomized46条件63.8043%。对应原30条同95条件76.8070%，扩样-0.1333pp。5路运行、无queued/终态failed，剩put_object_cabinet两域，以及put_bottles_dustbin、stack_blocks_three、stack_bowls_three的Randomized。07:22采样时当前attempt分别3/70、45/70、21/70、54/70、0/70（最后一项初始化），这些partial不计正式均值。此前ETA不再视为当前预测，尾轮被长任务和初始化重试拖慢；当前report/controller仍live。快照outputs/analysis/aloha_two_supplements_status_20261104T0723Z.json。


2026-11-04 07:48 UTC ETA更新：完整100条条件95/100，当前5路运行/0queued；按剩余任务长度和当前attempt推进估计还需4–6小时，北京时间约2026-11-04 19:48至2026-11-04 21:48。长rollout、尾轮并发下降和近期初始化重试已纳入判断；新增失败重试仍可能延后，不按完成条件百分比线性推算。逐任务计数见本run eta_latest.json。
