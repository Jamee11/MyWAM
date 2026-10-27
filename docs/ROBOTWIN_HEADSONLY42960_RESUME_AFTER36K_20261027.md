# 最新状态：42960已恢复评测

核验时间：2026-10-27T12:50:54.827815+00:00。此前自动接续在旧模型退出的身份检查上失败；退出竞态已修复并经过21项真实进程检查，已恢复运行。

36k已完整完成3000条：2108成功，**70.27%**；Clean89.33%，Randomized51.20%，官方69.04%，同协议32k71.10%。完整结果见[36k最终报告](ROBOTWIN_FULLBACKWARD36000_30EP_FINAL_RESULTS_20261027.md)。

heads-only42960：L20 GPU1/2/4/5/6/7六份模型全部ready，18端口在线真RGB/state→20D动作验收通过；4090实际已启动7路native仿真，当前可用显存不支持此前9–12路，按峰值预算限制、不停止其他作业。原10个完整L20条件保持，剩90在4090完整30条；当前进程尚初始化，不把旧300条误报为新的rollout或新SR。source头部e4c8c2d…权重RAM和config/stats/contract重新哈希通过。完整Sidecar/cache开compile关、官方标准、不存视频保持。

运行目录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/headsonly42960_4090_resume_all100_30ep_20261027T100232Z`，`state.json`、`live_progress.json`、`auto_transition_state.json`、`resume_start_acceptance.json`。后续真实动作进展由只读observer写`first_rollout_acceptance.json`。原100条件汇总将混合10条件L20＋90条件4090，比较时仍需标注硬件。

以下为历史准备/恢复记录，以上文为准。

# 36k结束后自动接续heads-only42960

记录时间：2026-10-27T10:08:51.300547+00:00。用户已确认本次所说43960为此前中止的42960：
`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_heads_only_from40000_to10epochs_20261025T171338Z/runs/from40000/checkpoint_step_42960.safetensors`。

SHA256 `e4c8c2d41a7f12902dd5ca5dbd807b2ea8b1ad982909433514d9ddaf916b1f35`。

## 当前自动监督

L20后台进程PID 3397012已真实启动并核验存活，每15秒读取36k原生队列终态；当前阶段`waiting_for36k`，36k已完成97/100条件。此状态是等待，尚未宣称新的heads服务/仿真已启动。监督具有flock防重复，不使用阻塞50分钟的交互会话。失败会写failed_needs_inspection和完整traceback，保留证据，不擅自修改模型/加速/数据或终止未知作业。

监督目录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/headsonly42960_4090_resume_all100_30ep_20261027T100232Z`。状态`auto_transition_state.json`，日志`auto_transition.log`，进程`auto_transition.process.json`。准备部署目录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/headsonly42960_4090_resume_deploy_20261027T100232Z`。

## 自动执行顺序

1. 等36k真实status=complete且100/100条件，不按时间猜完成；逐份重读run.json真实exit及native30结果，核对总3000条、成功数、唯一条件。自动写36k最终JSON/CSV、独立审计和`docs/ROBOTWIN_FULLBACKWARD36000_30EP_FINAL_RESULTS_20261027.md`。若失败不切服务。
2. 仅核验并停止已完成36k自有5份模型和自有两条reverse SSH；不停止训练或其他作业。保留所有旧结果/配置/日志。
3. 重新校验heads权重RAM SHA及config/statistics/contract。重新查询每张L20显存，仅>=26000MiB的卡启动，最多8份，全部同一完整Sidecar/eager+cache，每模型3独立session（最多24），匹配现有24线程上限。端口优先30900+10×GPU，slot0–2；冲突时顺序尝试31900/32900/33900基址，选择结果写deployment。动态内存变化或失败副本有独立记录；只使用通过重复动作/跨副本/在线真请求验收的副本，不预先宣称能放满8份。
4. 新reverse SSH只映射本次独立端口；4090对实际全部端口做真RGB/state→20D动作验收，再启动原生队列及报告/只读性能观察。启动成功后才更新heads主current指针。
5. 首次在4090完全空闲于本实验时按7680MiB峰值+1024MiB余量计算容量。GPU4最多2路；GPU6此前双路触发过显存保护，仅当本次静态余量>=2×7680+3072MiB才允许2路，否则1路。GPU1上限3，其他上限2；实际不超过按预算可容纳数。有额外路时启用低于2048MiB/渲染buffer失败回退，仅处理本实验新额外进程。保留先前GPU4/6峰值风险，不按瞬时任务切换空闲盲目恢复3路。

## 接续数据规则与比较边界

先前heads10完整条件246/300=82.00%均已逐份原生验收，本次保持其原结果和路径，标注L20。余90条件全部新在4090完整测试30条；中断的144条有独立历史attempt字段及原目录，不能拼成新的30条。新3000条覆盖由300条已验收+2700条新完整评测组成，不将历史partial用于SR或吞吐分母。

因此最终100条件为**前10条件L20、其余90条件4090**，不得标为全4090结果；与父模型或官方比较必须明确此边界。每个新条件30条只是趋势统计，官方issue46为每条件100条，样本差异保留。训练/数据不改，完整Sidecar能力保持，cache .99/max3开、compile关、原始动作chunk32/10去噪步、官方expert筛seed/unseen/RT/预算/成功判据、不存视频均保持。near/EEF replay与严格配对种子等旧未解项不因自动流程视为已解决。

## 已完成的准备验收

全部脚本AST检查；100唯一条件、10 complete+90 queued，原10结果逐项相等；90 queued新attempt为空、历史中断记录独立保留；4090设备筛选及RT缓存路径使用已验证原版36k运行模板。后台监督启动时已重新验收旧10条件native结果后进入waiting_for36k，尚未消耗新增模型GPU显存。实际新副本加载/新4090 rollout验证在触发后由部署和原生监控完成，不把工程准备称为模型效果。


### 2026-10-27T12:40:43.147903+00:00：自动接续回收竞态修复并真实恢复

36k于10:39:13完成100条件3000条、2108成功70.2667%，Clean1340/1500=89.3333、Random768/1500=51.20；独立native重验通过，对官方+1.2267pp、对32k−0.8333pp。原自动接续10:39:29在停止GPU0 launcher后、再检查其child时遇进程退出过渡cmdline为空，身份断言使脚本退出，heads未按预期启动；不能把这两小时称为评测时间。原错误日志和脚本已独立保留，36k结果与heads原10条件不受影响。

现修正stop_owned：对空cmdline退出过渡重读/幂等处理；对非空不匹配仍拒绝，使用pidfd固定进程身份后发SIGTERM，避免PID复用误停。当前Python未提供os.pidfd_open，按本机x86_64内核头文件434/424使用内核接口，原生21项真实子进程退出/重复停止/错误身份不发信号检查通过。恢复PID3404251已重新核验全部36k结果，身份正确的其余4份旧模型已停止，进入heads权重RAM哈希与重新查显存阶段。只修本实验接续脚本，不改模型/评测成功判据/数据，不停止其他工作负载。每轮后续以auto_transition_state/模型验收/原生rollout为准，不预称已运行。
