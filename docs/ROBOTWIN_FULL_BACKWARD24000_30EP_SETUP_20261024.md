# Full-backward warmup2k24k：全50任务双域各30条

建立：2026-10-24T14:50:55.882374+00:00。用户明确要求停止20k，切换同一训练run的24k，以与isolation warmup24k保持训练步数一致。

权重：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_24000.safetensors`；SHA256 `d9ed4954c2088f014651d15cee769426c83de6f12674a5f3b32be093242e040d`。精确step24000硬链接单独固定目录，config/stats/contract/tokenizer自本run保留；RAM副本源文件和目标再次流式SHA一致、源inode/size/mtime稳定。不是HF下载，不消耗镜像/VPN流量。

范围：50任务×Clean/Randomized×30条=100条件3000条；不存视频。完整Sidecar三tap7/15/23；cache .99/max3开启、compile=false，10 denoise、32chunk。继续官方原生expert筛seed、seed0/unseen、RT32spp/OIDN/near.1、原动作预算与成功判据。同官方checkout/隔离Conda，不改模型/训练/数据/上游或原DiT4DiT流程。任务顺序沿用20k/前24k，四短锚点双域后长原预算交错共享队列，均完整30才计SR。

L20 GPU0–7已各启动一份新24k模型，25900–25972共24个独占会话；重复两fixture各2次、跨副本逐元素一致、本机24路/4090隧道24路真实PNG/state→20D动作验收通过后才派发。first-ready副本原子建立本checkpoint规范参考，避免固定单卡延迟拖住其他副本，不与20k或其他24k输出强求相等。旧20k八份服务（含已完成L20对照专用GPU7）均核对身份后停止，新端口/token/权重SHA隔离。

4090尽量并行：当前caps1/2/1/1/3/1/1/0，容量允许最多10路；单路启动预算7680MiB+1024MiB卡余量、信用至首条完整policy episode后释放。GPU1/4 live守护2s采样，低于2048MiB或发生render cannot create buffer回退1/2路，仅停最新自有额外sim，原partial保留不混SR，基础设施最多3attempt。GPU7当前free3730MiB不放sim；GPU6当前free11943MiB先1路，其他作业不停止。保留45min无episode或action进展看门狗及ANSI剥离、独占token/slot锁和渲染PCI断言。

旧20k已用户中止：35/100完整，1050条718成功=68.38%，当前子集官方72.49、isolation24k68.57；全部旧原始日志、35完整结果和中断partial保留，不能称20k已完成全量。stop目录由outputs/robotwin_eval/fullbackward20000_stop_current.json指向。旧20k状态interrupted，不并入新24k正式分母。

新输出：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward24000_4090_all100_30ep_20261024T144937Z`；模型部署：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward_warmup2k24000_deploy_20261024T144703Z`。配置/代码差异与官方commit见source_evidence.json；pointer fullbackward24000_30ep_current.json。live_progress.json用于进度，comparison.json只对完整同任务子集对照官方与之前isolation24k，结束原生复验后生成final_results.json/CSV。此刻模型加载/验收进行中，尚无新24k真实rollout结果。

比较边界：本次训练步数都24k，但warmup机制不同，仍须核对其他训练超参/seed；相同30样本数量不保证accepted seed/指令/RNG逐条一致。保持历史near差异、cache近似与硬件/推理负载限制，不将阶段子集外推全量或宣称单因素因果。


## 2026-10-24T14:54:51.196032+00:00：full-backward24k八模型/24端口验收通过，4090九路启动

权重SHA d9ed4954c2088f014651d15cee769426c83de6f12674a5f3b32be093242e040d，8卡全部完整Sidecar/cache开启compile关闭，两个fixture每卡各重复2次并跨卡逐元素一致；本机/4090全部24路真实RGB/state→有限20D动作通过。dispatcher PID2037254，开始2026-10-24T14:53:33.319768+00:00，首批9路，不同模型优先。live caps1/2/1/1/3/1/1/0，GPU4第3路等待启动信用与显存，2GiB/buffer-error资源守护已运行，无占用未知作业。此记录仅队列启动，尚非真实rollout成功率。

training_budget_config_comparison.json核对：new full-backward24k与isolation24k batch32/accum2/lr1e-4/epochs10一致；新backbone_warmup_updates2000/isolation0，旧isolation2000，无架构/数据修改。相同step不自动证明其他训练/场景/RNG完全受控，结果按相同完整条件比较。旧20k已中止，35项1050条718成功保留且不混入本轮。证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward_warmup2k24000_deploy_20261024T144703Z/final_deployment_acceptance.json和/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward24000_4090_all100_30ep_20261024T144937Z。下一步确认真实仿真动作/首条终态后继续后台全部100条件。


## 2026-10-24T14:57:56.349922+00:00：新full-backward24k十路已启动，至少九路真实动作推进

八份L20模型全部ready（GPU0–7），4090现10路，9路均已出现真实policy动作，新增第10路仍在初始化；click_alarmclock双域已有成功终态，非ping/fixture冒充rollout。video关闭、new24k SHA、unseen/EEF和独占端口逐路通过；已出render证据项GPU PCI均通过，infra errors=0。每卡仿真数{"0": 1, "1": 2, "2": 1, "3": 1, "4": 3, "5": 1, "6": 1, "7": 0}，GPU4首条终态释放启动信用后已自动补至3路。资源2GiB/buffer-error保护运行。证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward24000_4090_all100_30ep_20261024T144937Z/first_rollout_acceptance.json；专用LIVE/后台comparison仅完整同条件30条计SR。旧20k已中止/结果保留，新24k独立3000条继续；相同step24000、batch32/accum2配置已核对。
