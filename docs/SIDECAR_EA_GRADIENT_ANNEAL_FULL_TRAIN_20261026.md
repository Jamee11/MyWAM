# RoboTwin完整训练：共享EA梯度退火

2026-10-26 UTC。用户已明确授权“按刚才方案把下一个实验接上”，实施此前的完整训练候选。旧full-backward、40k heads-only续训、最终42960评测均保留。

## 训练配方

| 项目 | 本次设置 |
|---|---|
| 初始化 | 原OpenWAM Foundation step154000；不是40k续训 |
| 数据 | 原Aloha50 Clean2500，549,775窗口，prepared manifest |
| 卡与batch | GPU0–3，4×A100 80GB，每卡32，累积2，全局256 |
| 预算 | 10epoch，42,960微步 / 21,480次optimizer更新 |
| 前2,000次更新 | 正常Read/Write及完整反向，仅跳过WAM参数、Adam状态和衰减更新 |
| 共享EA梯度 | 前2K保持100%；随后500次更新线性降至1%，之后保持1% |
| 学习率 | WAM1e-4、Sidecar5e-5，原cosine及5% LR warmup |
| 损失 | Action/Video各1，EA总scale .5；Ground .025、Relation .01、Plan .05、deep .1 |
| 增强/合同 | ColorJitter开启；原绝对EEF、归一化、mask、三处Write与推理结构保持 |
| 保存 | 每4000微步权重及完整state，沿用keep-last策略；最终42960按原逻辑保存权重 |

设u为前向开始时**已经完成**的optimizer updates：u<2000时rho=1；u≥2000时rho=1−.99×min((u−2000)/500,1)。例如u=2000/2250/2500分别为1/.505/.01。日志中同步微步的optimizer计数已加1，监督脚本按(micro−1)//2核对，不能把微步当更新。

## 唯一策略改动

三处read/refine送往EA heads的tokens及first-frame features走`detach(x) + rho*(x-detach(x))`。前向值不变，heads参数保持原辅助监督，回到共享表征及主干的**直接EA梯度**乘rho。包括stage1 Ground、stage2 Relation中间监督。原tokens继续进入后续tap和Action Write，Action/Video反向不缩放。

默认rho=1保持旧行为；原`aux_heads_only_after_warmup`仍可独立使用，相当于联合阶段rho=0。禁止把硬heads-only和非默认软退火同时打开。没有新增模型参数、额外前向或训练样本，eval直接走原输入，checkpoint形状兼容。

这不等于将总EA loss乘.01：loss标量和heads局部梯度不降低。全局clip及Adam等优化过程仍可能造成间接耦合，1%也不代表实际参数更新或关系语义可靠性获得保证。

## 工程验收

- 56项真实tiny Wan/ActionDiT CPU检查通过，含旧heads-only和WAM warmup回归。新用例在FP32/BF16、有/无checkpointing下检查前向逐位相同、heads梯度不缩放、共享EA梯度按rho缩放、Action/Video梯度逐位保留；检查2K/2.5K边界、累积更新和optimizer恢复。
- GPU5/6空闲时完成**两rank BF16 ZeRO-2** tiny WAM短测：accumulation2、完整WAM反向、前两次WAM不更新、之后联合更新；rho1/.505/.01均覆盖，五个检查点的保存/恢复下一步逐位一致。已正常退出，不继续占用GPU5/6。
- 6项接续/监控测试通过：即使显卡已空闲仍须等前序完成；exit0+四rank终点计数+完整safetensors才允许接续，接受最终无optimizer state的正常保存行为，失败/截断权重不放行。Ruff与bash语法检查通过。
- 这些是工程检查，**不是完整Foundation真实更新、2K正式边界或闭环性能验收**。后者由本次启动及后续配对评测验证；监督脚本持续检查真实日志rho与有限性。

## 实际输出与接续

实验根：

`outputs/sidecar/aloha50_ea_grad_anneal_10epochs_20261026T022844Z`

新run在该目录的`runs/<时间>`，与旧checkpoint目录完全分开。指针`outputs/sidecar/ea_grad_anneal_current.json`；工程证据在`engineering/`，实际配置在run内`config.yaml`。

入口`scripts/train_sidecar_ea_grad_anneal_4gpu.sh`，配置`configs/train_relational_sidecar_aloha50_ea_grad_anneal_10epochs.yaml`。复用现有`/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv/bin/python`，不复制环境，权重/数据/输出仍在共享盘；离线加载本地Foundation，不下载。

监督PID3948883，训练torchrun PID3948895，2026-10-26T02:32:07Z已启动CPU模型/数据准备。当前GPU0–3的heads-only PID3907778继续。CPU预加载只额外创建较小CUDA上下文；新完整模型须同时满足前序正常完成marker、90秒通知窗口已过、四卡各free≥70000MiB，才允许DeepSpeed搬入。CPU预加载期间的显存增量不能解释为两个完整模型同时训练。

监督脚本`supervise_sidecar_ea_grad_anneal.py`不会终止前序，不因失败/空闲而盲目接续。它校验实际dataset/world/batch/warmup/rho合同；接卡后四rank首个更新均通过才写`startup_acceptance.json`。真实2K切换仍须以届时日志为准。

查看状态：

```bash
cd /remote-home/jinminghao/OpenWAM
cat outputs/sidecar/aloha50_ea_grad_anneal_10epochs_20261026T022844Z/job_status.json
tail -n 30 outputs/sidecar/aloha50_ea_grad_anneal_10epochs_20261026T022844Z/launch.log
```

## 解释与后续决策

这是从同Foundation出发、保留弱语义约束的完整训练，主对照是原full-backward的同32k/42960预算。40k heads-only只回答晚期路由问题，不是本次全程软/硬策略的严格对照。保持near、cache、执行chunk、动作预算和成功判据，比较完整同任务Clean/Random协议，并检查Contact/Moving/方向等有GT指标是否保持。

1%与500更新是固定候选，依据旧12k少量B1梯度量级提出，并非batch32或Adam最优解。不添加新结构、Gate扫描、Video权重、WAM LR ramp或新的采样策略。预计训练量参考旧run约80小时，实际加载/保存/争用另计。

未解决事项继续保留：匹配Vanilla、训练/评估腕部near .02/.1、专家绝对EEF物理回放及严格场景/RNG配对。训练loss/gate/表征准确性不能替代闭环成功率；若语义明显退化且控制无收益，应记录失败结果而非追加多个补丁。


## 2026-10-26T02:38:11.028179+00:00 启动验收更新

实际四rank已于02:36:52Z完成CPU预加载并进入handover等待，实际run：`outputs/sidecar/aloha50_ea_grad_anneal_10epochs_20261026T022844Z/runs/2026-10-26_02-36-51`。549775窗口/world4/bs32acc2/2K/500/.01合同通过；实际config与旧run差分仅新EA路由字段、输出及同manifest解析路径。证据`engineering/cpu_preload_acceptance.json`。新正式GPU训练仍待旧heads-only正常退出，不将CPU就绪或小模型GPU检查报成Foundation已更新。


## 2026-10-26T03:58:56.813204+00:00 正式接卡后进度

前序heads-only正常完成42960/21480且最终权重已保存。新实验2026-10-26T03:26:44Z完成DeepSpeed包装，03:27:25Z四rank首个真实optimizer更新通过，现micro291/42960、opt145/21480。仍处于正常Read/Write和完整反向、仅WAM跳optimizer更新的2K warmup，rho=1。四rank已观测metrics全部有限，同步微步grad_norm非零，三Write gate已从零更新至约−4.2e-4/−4.4e-4/−3.2e-4。

最近100微步均速6.583s/micro，即约13.17s/update；按当前warmup吞吐剩约6.78小时到2K，完整预算剩约78.0小时，保存/联合阶段吞吐变化另计。前50与最近50微步的加权EA均值0.13248→0.11447；Action/Video尚大致平台。仅表明启动和早期优化正常，不代表闭环收益或2K正式边界已经通过。证据`outputs/sidecar/aloha50_ea_grad_anneal_10epochs_20261026T022844Z/progress_20261026T0358Z.json`及startup_acceptance.json。未更改训练/配置/数据或进程。


## 2026-10-26T07:24:24.670262+00:00：同卡额外负载及吞吐突降只读核查

最新micro1897/opt948，仍WAM不更新的2K阶段/rho=1，未发生新策略边界。正常101–1500微步耗时中位6.559s，最近100微步14.436s，耗时×2.20，吞吐下降54.6%。突变约micro1672/UTC06:30:32（按日志累计wall时间回推近似），不是4000保存点或warmup切换；四rank同步变慢、loss有限。

GPU0–3自身rank显存仍53500/53500/54720/54540MiB、峰值allocated/reserved保持，整卡总占用多出约5.6–5.9GiB/card，pmon自身SM约30–60%而整卡100%；频率1410MHz、无power/thermal限频，CPU quota不限且无throttle、主机RAM余量充足。这强烈提示同卡额外负载/资源争用，但NVML在本容器仅显示自身rank，不能确认外部PID、是训练还是其他GPU任务，也未通过停止外部作业隔离因果贡献。旧heads-only及本次诊断进程均已不在，当前仅本训练4rank；其他同名python主要为DataLoader子进程，不冒充多个训练。

若当前速度长期持续，剩余计算粗估约164.7小时，对原速度约74.8小时；保存/后续联合阶段/争用变化另计。建议先协调0–3独占并复查速度，不修改训练超参或误杀未知作业。本轮只读、未停止或重启任何进程。证据`outputs/sidecar/aloha50_ea_grad_anneal_10epochs_20261026T022844Z/throughput_contention_audit_20261026T0723Z.json`；负载归属与释放后恢复仍未解决。
