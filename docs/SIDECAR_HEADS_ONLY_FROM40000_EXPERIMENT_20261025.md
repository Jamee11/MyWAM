# Sidecar heads-only：从40k独立恢复到10 epoch

日期：2026-10-25。用户授权保留原训练逻辑，新增联合阶段EA heads-only路由，从指定40k完整状态继续至原定10 epoch，并要求新旧checkpoint目录严格分离。用户随后明确：等待GPU0–3旧实验正常结束后自动接续，不结束旧实验。

## 辅助heads有多大，detach后还能学什么

本轮从真实 `checkpoint_step_40000.safetensors` header逐shape核算，并由新模型CPU构造再次核对：

| 部分 | 参数量 |
|---|---:|
| Ground query/key 两个投影 | 525,312 |
| Presence / track / near / contact / goal读出 | 3,077 |
| Plan读出 | 4,104 |
| **全部辅助heads** | **532,493** |
| 全部Sidecar | 24,290,323 |

heads约0.53M，占Sidecar的2.1922%，三tap共用一套，不能乘三。Ground头占绝大多数参数，关系与Plan读出相对轻量。

本改动detach的是heads的tokens/features输入，不冻结heads参数，也不detach logits或EA loss。heads继续接受完整原权重的Ground/Relation/Plan/deep监督；共享Read/Refine/Plan仍随Action+Video训练，Write继续使用原tokens。

**能保证的是梯度路由，不是关系能力保持。** 输入特征若仍可读出关系，heads可以继续适配；若上游为了动作目标丢失关系信息，只有这0.53M读出参数不能强制它恢复。因此这是一项从已接受长期EA监督的40k模型出发的短对照，必须在后续检查关系指标与闭环，不能因heads仍有loss就宣称关系可靠。

## 具体实现及保留项

- 新开关 `model.architecture.relational_sidecar.aux_heads_only_after_warmup` 默认false；旧训练默认行为保留。
- 启用时，仅训练态、joint且warmup结束后，三次heads调用读取detach的tokens/features。跨tap持续tokens和三处Write不detach。
- final三类EA loss与中间deep supervision全部覆盖；Action/Video损失、mask、绝对EEF、数据、模型宽度和参数名/shape不变。
- 启用时新增指标 `ea_aux_heads_only`，本次恢复后应为1；原实验默认不新增此指标。
- 推理前向数值没有因此变化。诊断实际训练梯度时须处于train模式；eval模式不启用训练路由，不能用旧eval-mode梯度探针误判本功能未生效。

代码：`openwam/model/modules/relational_sidecar.py`、`openwam/model/architectures/dual_system/relational_sidecar.py`；新增独立配置及resume入口复用原GPU handover与metrics。

## 独立状态与路径

源run：

```text
/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18
```

精确来源：`accel_state_step_40000`，`global_step=40000 / opt_step=20000 / epoch=9`。原13个完整状态文件共110,179,810,280字节；连同权重与必要资产共21文件、135,063,633,621字节，独立物理复制、源不变/独立inode和写后SHA256回读通过。

新实验根：

```text
/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_heads_only_from40000_to10epochs_20261025T171338Z
```

- 新训练/新checkpoint：`runs/from40000/`。
- 初始40k参考：`initial_checkpoint/`；40k权重SHA256为 `29a975b7def92777fd5e7ae62463c304b691e7f1402ccf3635fd6716184fe153`。
- 初始权重经验证后移出活动训练目录，避免被分支keep-last/finalize清理；`fork_state/initial_weights_relocation.json`记录原复制目标、保留路径与SHA。没有和原run共享文件inode。
- `fork_plan.json`、`fork_ready.json`、`fork_state/copy.sqlite`：来源、复制与配置证明。
- `configuration_diff.json`：只改复制config的EA路由和finetune/resume/output记账字段；原config字节不改。
- `source_before/`、`source_after/`、`source_sha256.json`：源码版本记录。

源run、源checkpoint和当前训练进程均保留。新resume入口强制自己的run、精确40k状态、world4/bs32/acc2和549,775个原Aloha窗口，防止恢复到别处或把新权重写回原目录。完整状态恢复不用活动目录里的safetensors，保留在initial_checkpoint的40k权重可独立作后续参考。

## 训练预算与恢复语义

| 项目 | 本次设置 |
|---|---|
| 数据 | 原Aloha50 Clean2500、549,775训练窗口 |
| GPU / batch | 原四卡、每卡32、累积2、global256 |
| WAM / Sidecar LR | 原1e-4 / 5e-5及原cosine状态继续 |
| Loss / 增强 | 原Action=1、Video=1、EA scale=.5与内部权重；ColorJitter保持 |
| 原warmup | 保留这份模型的2,000-update历史；不重新做500或2K warmup |
| 起点 | micro40,000 / optimizer20,000 / epoch9 |
| 终点 | micro42,960 / optimizer21,480 / 总10 epoch |
| 剩余 | 2,960 microsteps / 1,480 optimizer updates |

**不是从40k再跑10 epoch。** 不重设LR、不重置Adam、不缩短max_steps重建cosine；调度器构建实测仍为21,480 updates、warmup1,074、eta_min1e-6。按原约6.65–6.9秒/微步，接续后计算约5.5–5.7小时，另加真实状态恢复与最后保存。

此阶段LR已很低，短试没有收益不能直接否定“从warmup结束就启用路由”的全程方案。继承的Adam历史和heads参与全局clip的间接影响仍保留，不把“EA直接共享梯度为零”说成不存在任何数值影响。

## 当前启动状态与自动接续

启动时间2026-10-25T17:27:01Z，torchrun PID3907778，monitor PID3907770。复用现有本地runtime：`/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv/bin/python`，没有新复制环境、没有模型下载。

CPU模型/数据/输出契约已通过，`cpu_contract.json`实测heads532,493、Sidecar24,290,323、dataset549,775。当前尚未开始GPU更新，等待GPU0–3各空闲至少70,000MiB；不终止原3501138，不在旧实验未释放显存时加载新GPU模型。

用户已确认这个接续安排。后台控制器只等待、记录和运行本分支，不自动杀其他作业、不改batch、不因超时重启。

动态证据：

- `job_status.json`：当前阶段，最终训练成功也只标记training-only，不宣称成功率。
- `handover/cpu_ready.json`、`handover/gpu_release.json`：CPU就绪与实际显存放行。
- `resume_verified.json`：只在真实Accelerate完整状态恢复成功、计数40k/20k/epoch9且heads-only激活后写出；目前不能用CPU准备替代此证据。
- `runs/from40000/metrics_rank*.jsonl`：恢复后的真实loss、step和`ea_aux_heads_only`。
- `launch.log`、`launch.json`、`monitor.log`：启动和错误日志。

当前已运行，不要再次提交同一个分支；相同输出目录不能启动第二个writer。

## 已完成的验证与后续判定

41项CPU工程回归通过：实际tiny Wan/ActionDiT、FP32/BF16、gradient checkpointing、同权重前向逐位一致、三tap及deep梯度只到heads、Action与Video仍到Write/共享表征、500-update边界、optimizer恢复后heads更新和输出路径守卫。Ruff与shell语法检查通过。真实40k四卡完整恢复与最终训练尚未完成，不把这些CPU证据当成功率或真实多卡恢复成功。

训练完成后比较三个固定点：初始40k、原run最终42960、heads-only最终42960，使用同输入/噪声/归一化/部署设置。先看Action XYZ/旋转/夹爪和Full/Off局部贡献，再复用现有Ground峰值命中、Contact F1、Moving F1及8/16/32方向cos检查是否语义遗忘，最后看配对闭环。训练loss下降、gate变大或head loss存在均不代替SR。

原run是连续训练，新分支经恢复；原worker增强随机状态不保证逐bit重现，不能把任意细小差异都归因路由。保持相同优化器/调度/预算减少混杂，若差距很小，再考虑同样恢复但关闭路由的控制。near差异、实测EEF再执行的控制误差、cache近似及完整匹配Vanilla仍是独立限制。


## 2026-10-26：恢复后显存比原run多约7GiB的只读核查

当前GPU0–3各约72504MiB，四卡各一份本分支worker；旧训练进程已退出，未发现这些卡上额外训练副本。模型/batch配置差分仍只有EA路由与恢复/输出记账项。

对齐同一microstep的300次optimizer更新（rank0），peak_allocated从原约59.189GiB变为新约59.207GiB，平均仅+18.426MiB；peak_reserved从62.246GiB变为69.172GiB，+6.926GiB。四rank同样表现：实际张量峰值接近，增加主要落在allocator预留池。注意它们是每次更新的峰值计数，不是同一时刻的实时tensor快照。

新run首次40001预留61.541GiB，首次恢复后的optimizer更新40002升至69.172GiB，此后到本次已观测约41600余步保持同一预留水平。与恢复后的分配历史/缓存池扩张相符；没有记录allocator snapshot，不能断言是哪块临时tensor或把全部原因严格归于checkpoint加载或detach之一。不存在“detach新增了6–7GiB模型参数/激活”的数据依据，也未观察到持续增长趋势。

同范围近期微步均速原6.666s、新6.591s，未见因这次预留增加而明显变慢；任务/worker增强未逐bit匹配，不能将小差异当速度提升。当前整卡free约8651MiB，不据此承诺未来所有batch绝无OOM。没有为了降低nvidia-smi数字去中断训练、注入empty_cache或改allocator配置。

证据：新实验根memory_audit_20261026/summary.json；官方解释见 https://docs.pytorch.org/docs/2.10/notes/cuda.html 的Memory management：reserved包含缓存allocator管理的显存，allocated统计tensor占用，未使用缓存也会出现在nvidia-smi。
