# Sidecar 五组同起点 1K 更新短训：执行与判断

记录时间：2026-10-26T13:41Z。用户授权原四组串行短训，并增加 Main-task Gradient Gain 第五组。

## 当前执行状态

已在 GPU0–3 启动 A_control；13:41Z 四个 rank 均到 micro4074 / optimizer2037，loss 有限，最近20微步耗时中位约6.62秒。四卡真实恢复、原 scheduler 和首次联合更新已通过。旧 EA-anneal 完整训练在新模型 CPU 权重就绪后按核验的 PID/start token 停止；旧 checkpoint 保留。

独立实验根目录：

```text
/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_four_arm_1k_20261026T131106Z
```

统一起点 S 为用户指定 `aloha50_ea_grad_anneal_10epochs_20261026T022844Z/runs/2026-10-26_02-36-51/accel_state_step_4000`，即 **4000 微步 / 2000 次 optimizer 更新**。此前 EA 路由尚未退火，正常读写与完整反向，WAM 不更新。因此可作为五组共同 warmup 末端，不能与旧 isolation/no-write warmup 混称。

S 的权重、四 rank 完整优化器/RNG/scheduler 和部署资产共21文件、86,894,380,606字节，已经逐文件复制、SHA写后回读及独立inode核验，固定在本实验 `common/`。五组读取同一份独立副本，不覆盖原实验。

## 五组唯一干预

| 顺序 | 目录 | 相对 A 的修改 | 要回答的问题 |
|---|---|---|---|
| A | A_control | 原配方，EA 共享梯度100% | 恢复后原方案参照 |
| B | B_ea01 | 2K后500更新内，EA对共享特征的梯度1→0.01；heads梯度保留 | 减弱辅助主导能否改善动作 |
| C | C_wam_ramp | WAM实际LR在300更新内从原cosine的0.1倍恢复到1倍 | 缓和WAM开始更新的边界冲击是否有益 |
| D | D_write2 | 仅三个Write模块真实AdamW LR×2 | 写回参数是否适配不足 |
| E | E_gain2 | Write输入token支路的反向梯度×2 | 保留EA监督时，增强主任务对共享关系表征的塑造是否有益 |

B 是用户所指的“head-only / 1%退火”，不是严格 detach/rho=0。C、D、E 都保持 EA 共享梯度100%，互不叠加；E 也不是此前规划的多项联合候选。组合实验应在单项结果之后再决定。

每组均从 S 恢复，**再训1000 optimizer updates**，到 micro6000 / optimizer3000 停止。数据、ColorJitter、global batch256（32×4×acc2）、Action/Video/EA loss权重、梯度裁剪、原10epoch/21480更新cosine不变。原 cosine 的内部 T_max=81624、last_epoch=8000 必须在恢复时核验，不用短训 max_steps 重建 scheduler。

按用户2026-10-26T16:33Z的新要求，**B、C、D、E只在micro6000（本轮满1000次更新）保存最终权重和一份完整 `accel_state_step_6000`**，取消4500、5000中间权重。A保持原保存安排，已有文件保留。四组共少写8份中间权重，按当前文件大小减少198,898,905,008字节，约198.9GB磁盘空间；不代表训练显存相应减少。需要继续时沿获选分支完整状态推进到原10epoch终点，不重跑warmup、不额外增加10epoch。

## Gain×2 的含义、优先级与风险

只在训练且2K WAM延迟更新结束后，对送入Write的token使用：

```python
z_for_write = z.detach() + 2 * (z - z.detach())
```

前向值仍为原 z；传给后续Read/Refine和辅助heads的是原 z。此算子放大经Write回传的主任务梯度，包含Action以及可达的Video路径，**不放大Write参数自身的局部梯度，也不直接修改EA梯度**。推理时关闭该梯度干预，不需要额外部署分支。

这与D的Write LR×2不同，也与B的EA缩小不同：E保持完整关系监督，同时让动作目标对关系表征有更大发言权。技术实现很小，值得作为第五组筛选，并非不可实现。

优先级仍低于WAM Ramp：后者有2K边界瞬态的直接证据；Gain是否有益依赖当前表征与主任务的匹配。主要风险：经Read回到WAM的相关主任务梯度也会改变；全局裁剪可能间接缩小其它梯度；Adam归一化意味着梯度×2不等于参数更新×2；增大低质量条件的训练影响可能伤害控制。局部雅可比×2不代表所有完整计算路径或所有参数的最终梯度/更新都精确×2。

因此先测2，不同时扫4、不同时叠EA退火或LR修改。不能凭训练loss降低、gate增大或写回变强就判胜。

## 接续与控制变量验收

当前四组监督 PID4026489；E跟进监督 PID4012262；离线诊断监督 PID4012263。PID仅为启动记录，后续操作须核对启动标识和命令，不可盲目复用。

16:39Z为使已经CPU预加载的B也使用新保存逻辑，只重载了尚未授权GPU/尚无训练更新的B，新torchrun4026490。A的torchrun4006583及其训练rank保持运行，由新监督附接。原监督4006566暂时暂停以保留A结束时内核真实exit status；新监督核验A正常exit0后终止原监督并接B，避免把缺失退出码假装成功。B旧预加载日志/标记移到 `B_ea01/preload_before_endpoint_only_policy/`，旧run目录保留。附接退出状态及原入口/跟进检查7项通过，保存条件静态检查确认B–E仅6000。变更记录 `checkpoint_policy.json`；监督日志 `supervisor_endpoint_only.log`。GPU真实端点保存及B后续恢复仍待顺序执行验收。

- 四组主监督 `scripts/supervise_sidecar_four_arm.py` 已运行；E由 `scripts/supervise_sidecar_gain_fifth.py` 接在D之后，不重启主队列。
- 当前组到micro5400时预加载下一组CPU模型和CUDA上下文，等待当前组6000完整保存、正常退出后再进入GPU恢复；优化器恢复仍有必要耗时，不能承诺显存交接零间隙。
- 每rank记录实际batch数值指纹；选定6个微步额外记录完整增强图像指纹。B–E逐步与A核对，若不同直接失败，避免暗中比较不同输入。
- 每rank核对恢复后Sidecar fp32 master与Adam状态指纹；原始分组和moment历史保持。D在ZeRO同步前以相同AdamW functional计算Write片段的真实2倍LR更新，普通moments只推进一次，不用梯度×2代替LR×2。
- C临时应用effective LR，AdamW结束后恢复scheduled LR，再推进原cosine，避免递归scheduler吞入倍率。
- 本轮31项Gain/LR/路由/入口检查通过，另3项跟进/诊断脚本检查通过；两份日志部分测试重叠。GPU5/6双rank BF16+ZeRO2+acc2验证更新/moment、组布局、保存恢复下一步一致。A真实四卡恢复已通过；B–E完整真实起点恢复和更新仍需队列执行验收。

状态入口：`job_status.json`、`fifth_status.json`、各组 `resume_verified.json` / `startup_acceptance.json` / `completed.json`。工程证据在 `engineering/`，包括配置差异、测试和源码SHA。

## 自动离线筛选与决策

`scripts/diagnose_sidecar_four_arm.py` 等待各组完成，使用GPU5或6中空闲显存足够的一张，串行复用已经验收的 GT/Full-Off bootstrap，不停止其它作业。固定每组6000权重，避免后续续训选到新的latest权重。

每组50次生成（25窗口×Full/Off），其中24 held-out窗口来自三任务、Clean/Randomized各三条专家episode，另1训练seen重复控制；同输入/噪声/10步推理，核对重复一致、Off真实delta为零、模型参数未更新。统计Block7/15/23真实写入、前8/全32步XYZ、SO(3)、Gripper、关系/运动指标，并跑同一32样本的配对Action FM。输入文件哈希固定，源数据和模型不更新。诊断使用eager/no-cache以便观测，不改生产评估参数。

结果：`diagnostics/<arm>/`，合并比较 `diagnostics/comparison.json`，排队状态 `diagnostics/job_status.json`。同时按相同微步窗口对照训练Action/Video/EA loss；不直接用EA绝对loss比较B的优化梯度强度，因为其前向loss未缩放。

优先筛掉明显发散、Action恶化或关系质量明显退化的候选。若多个候选接近，不强行凭小数点排序：最多选最有依据的一至两组续1K或进行同seed配对闭环；再决定一次完整训练及是否组合。上述窗口高度相关，离线误差不等于闭环SR，也不能证明全50任务约2pp提升。此轮没有自动挤占其它session正在用的L20/4090闭环资源。

当前约6.62秒/微步时，每组纯计算约3.68小时，五组约18.4小时，加加载/保存开销；E相对原四组多约3.7小时。共享卡争用会改变用时。

保持未解决事项：匹配Vanilla归因、train/eval wrist near-plane差异、真实EEF物理回放及坐标/TCP契约、小样本开发集选择偏差。训练仍依赖原official_clean4350路径，不能删除外部原始数据。


## 2026-10-27T01:46:15.032892+00:00：前三组完成，第四组进行中（离线初步结果）
A/B/C均完成额外1000更新到micro6000/opt3000，完整状态已保存、真实exit0；三组50生成与32固定FM诊断完成，无失败，实际训练batch指纹与A一致。B/C各仅保存6000最终权重。D正在GPU0–3训练，01:44Z到micro4404/opt2202，即本轮202/1000；E按序等待。原暂停监督4006566已退出，当前主监督4026489、E4012262、诊断4012263均正常。
|组|Clean前8步XYZ mm|Random前8步XYZ mm|Clean全32步XYZ mm|Random全32步XYZ mm|固定Action FM Full|
|---|---:|---:|---:|---:|---:|
|A_control|9.7377|13.0052|15.6095|23.5921|0.00158841|
|B_ea01|10.5879|11.6870|15.9206|39.0051|0.00168015|
|C_wam_ramp|9.2408|12.4923|21.8857|24.4472|0.00167009|

初步：B改善Random前8步位置误差，但全32步Random明显差于A；C改善前8步位置均值，但32步与旋转等并无全面优势。当前不能宣布B/C优于A。FM、前8步和全32步排名不一致，进一步说明不能单按某一离线loss选方案；继续D/E不改配方，完成后统一筛选和配对闭环。样本仍为六episode的相关窗口，不作SR/显著性结论。

D真实四卡Write LR已核验：已记录更新均effective=2×scheduled；四rank Write片段元素合计6308355（rank0为0、其余各2102785），与目标Write全部参数数目一致。当前每微步中位6.599秒，D余约2.93小时，D+E纯计算约6.59小时，预计整体还需约7小时含I/O/交接；不是包含闭环评估的承诺时间。证据为本root diagnostics/comparison.json、各组completed.json和D lr_rank0–3.jsonl。


## 2026-10-27T07:03:24.677592+00:00：前四组完成，E进行中

A/B/C/D均已完成额外1000更新至6000微步/3000更新，真实exit0、完整续训状态及各50生成/32FM离线诊断通过；四组实际训练batch指纹匹配。B/C/D仅保存最终6000权重。E当前5139微步/2569更新，本轮569/1000，日志Gain=2、loss有限；第五组和诊断监督正常。原四组主监督完成退出属于预期，不是队列失败。

D相对A：Full32 XYZ Clean 15.6095→14.3899mm；Random 23.5921→23.6625mm。旋转误差Clean 2.7223→3.5380°、Random 4.5210→5.9242°；固定Action FM Full 0.00158841→0.00203281。Full相对Off的FM收益12.93%→1.26%。因此D有Clean位置改善，但没有全指标收益，不能只凭该项判Write LR×2更优或立即全训。

07:02Z每更新约13.70s，E训练余约99分钟，另加保存及末次离线诊断（前三组约7.6–8.1分钟，D约10.6分钟），预计约2小时收齐五组。本轮只做状态/离线复核和记录，无代码、训练配置、数据或进程修改；继续E并等待统一比较。仍是六episode相关窗口的诊断，不是闭环SR，不据此做最终方案选择。证据diagnostics/comparison.json及E metrics_rank0.jsonl。


## 2026-10-27T07:13:30.083978+00:00：A–D离线诊断集中对比，E待完成

四组均同一2K完整状态起点，续训1000更新；每组50次Full/Off生成、32固定FM对完整。实际训练batch/抽样增强指纹与A一致，诊断输入、噪声、10步推理配对，重复、Off零delta及未更新模型检查通过。六条held-out专家episode的24个重叠窗口用于下表，额外一条seen重复只用于控制；不是24个独立episode。三任务为adjust_bottle、place_shoe、beat_block_hammer。

所有位置误差单位mm、越低越好；Action FM为32个固定样本的Full平均，不能与训练loss绝对数值混比。

|组|Clean前8步|Random前8步|Clean全32步|Random全32步|Action FM|FM相对A|
|---|---:|---:|---:|---:|---:|---:|
|A_control|9.7377|13.0052|15.6095|23.5921|0.00158841|+0.00%|
|B_ea01|10.5879|11.6870|15.9206|39.0051|0.00168015|+5.78%|
|C_wam_ramp|9.2408|12.4923|21.8857|24.4472|0.00167009|+5.14%|
|D_write2|9.5037|13.0633|14.3899|23.6625|0.00203281|+27.98%|

真实写入delta/Action hidden范数比均值，百分比；这是实际探针测量，不是gate值估算。

|组|Block7|Block15|Block23|Full相对自身Off的Action FM降低|
|---|---:|---:|---:|---:|
|A_control|1.8935%|2.4528%|3.1650%|12.93%|
|B_ea01|1.4646%|1.2389%|2.2620%|14.80%|
|C_wam_ramp|1.9564%|2.7821%|3.8059%|13.18%|
|D_write2|0.6959%|1.1161%|1.8542%|1.26%|

初步解释与下一步：

- A在此3K更新阶段的写入已为约1.89/2.45/3.16%，且Full比自身Off的FM低12.93%，不能沿用最初旧42960近乎零写回结论。这个阶段的非零写回和离线收益不证明长期或闭环SR提高。
- B：Random前8步位置较A低约10.1%，但Random全32步明显更差，固定FM总均值+5.78%。减少EA共享梯度没有自动提高所有动作指标；本批方向cosine和ground hit也有下降，Contact F1=1不能掩盖其它语义指标。不能据此直接证明1%早期退火总体有害或闭环更差。
- C：前8步位置两域均低于A，Random固定FM低12.86%，但Clean固定FM高18.75%、总体高5.14%，长horizon误差不占优。前200微步Action平均loss低26.54%，末1000微步高8.63%；这支持过渡期变缓和的局部现象，尚未证明1K后整体受益。原1K内有效WAM步长较小的预算差异需保留。
- D：Clean32位置更低，Random基本持平，但旋转/Random夹爪及固定FM更差。三个真实Write比率反而低于A，beta绝对值也更小；实际LR×2并不等于网络最终写回更强。Full相对Off的FM收益只1.26%，仍不能据此归因模型整体变化或判定Sidecar训练无用。
- 当前没有一个候选在所有关键指标全面领先A。前8步、32步及FM排序不同，应等E完整结果后选少量候选做同seed闭环，避免把离线任一表的最小值当最终SR。此时不叠加多个trick，不启动新的网络/全训改动。

表示指标按现有GT mask定义：Random Presence/Track准确率A为75.0/62.5%，B75.0/70.8%，C81.25/83.3%，D62.5/54.2%；对应可见性/有效性标签，不直接等同物理物体是否存在。全部Contact F1=1只来自每域12个有效contact判断，本身不证明全面OOD鲁棒性；near阈值契约、EEF回放、匹配Vanilla和闭环仍待验证。

本次只记录已完成诊断，不改变任何训练或评估进程；E继续，最后权重生成后由既有监督自动补齐同一流程。完整机器可读结果为本root diagnostics/comparison.json，各组原始results/paired_action_fm及工程校验保留。


### 2026-10-27T07:46:23.522608+00:00 Loss动态分析补充

已按用户要求对齐各组完整训练loss与优化动态，详见[五组Loss与参数位移分析](SIDECAR_FIVE_ARM_LOSS_DYNAMICS_20261027.md)。C的稳定化已确认，但终点排序仍待延长/闭环；B有辅助语义平台化，D的Write变化主要加速gate收缩，E最终仍待。E结束后CPU自动补齐final数值，不启动新训练。


### 2026-10-27T08:58:52.413252+00:00 最后状态核验

原五组训练及离线诊断均完成、无失败。E全32步XYZ为Clean19.2743mm、Random26.5739mm（A分别15.6095/23.5921），固定Action FM相对A高3.07%；训练后250更新Action高2.47%，暂未显示全面优势。新A/C队列运行正常：08:57Z A为3039 optimizer updates（本轮39/1000），完整恢复与首更新验收通过，C按顺序等待；三份监督均活跃，无Traceback。C的4K实际结果及真实恢复仍待后续运行，当前不自动开启新的trick或完整长训。


## 2026-10-28T02:38:31.562355+00:00 五组最终综合复核

A–E最终均完整1000更新，原始四rank日志/optimizer和batch指纹、Full/Off/32FM验收重新通过；E结果已纳入，不沿用中期partial结论。完整的Loss、前8/32步XYZ/旋转/夹爪、分任务、关系、真实write及A/C4K延长结果，统一见[最终综合对比](SIDECAR_ABCDE_COMPREHENSIVE_COMPARISON_20261028.md)。本次仅CPU分析和文档记录，无训练/代码/数据或进程变更。
