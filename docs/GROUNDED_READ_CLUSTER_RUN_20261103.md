# Grounding-guided Read：mhan0035四卡A100，两阶段单作业

用户已确认目标根目录 `/home/mingfeihangroup/xiaoming`，`mhan0035`是Slurm分区名；当前不存在`/home/mhan0035/xiaoming`。本次使用已连接SFTP-34和ssh-56会话，更新指定项目下的独立代码快照，旧默认源码/配置/入口保留。

## 实现

默认关闭`model.architecture.relational_sidecar.grounding_guided_read`，无新增parameter/buffer，旧权重严格加载；零强度走原Read调用。块7 Grounding预测引导块15 Relation/Plan Read，块15预测引导块23；不跨chunk缓存。每臂Object/Goal的预测热图按现有role顺序配对；presence与track预测置信度、当前观察得到的arm mask控制无效/低置信角色回退均匀。Relation的near/contact token用Object先验，goal token与Plan用Object/Goal混合；Ground/Summary queries不加bias。无GT输入、额外encoder/teacher/loss，原Video↔Action mask、Refine、FFN、Write与全部EA监督保持。

先验混50%uniform，attention bias = strength×log(N×P)，有均匀下界；joint训练强度前1000参数更新0→.5，之后/eval固定.5。已有辅助head输出新增参与后续主任务梯度，不代表已证明因果收益。

## 两阶段：一个sbatch，4张A100顺序执行

1. **Warmup**：从目标机`ckpts/OpenWAM-Alpha-Pretrain-Foundation-Model`加载，Sidecar随机初始化；完整前后向，仅Sidecar更新，WAM不更新（含moment/weight decay）；旧完整PatchWAM与EA，Grounding引导关闭。4×batch32×acc2=global256。保持10epoch的21480更新scheduler horizon、标准5%LR warmup，在2000参数更新/4000micro后停止。权重按optimizer编号保存为`checkpoint_step_2000.safetensors`，附tokenizer/config/norm/contract。
2. 4rank均到2000、权重marker/完整头与长度通过后，重新加载这份权重，创建**新的optimizer/scheduler/counter**。不重复冻结/2K预热，开启上述Grounding引导，完整训练10epoch=21480参数更新/42960micro；全局batch仍256。WAM/Action LR1e-4、Sidecar5e-5，5%LR warmup+cosine，完整原PatchWAM，XA关闭，不叠三阶段完整监督或gate放大。

因此总预算为预热2K + 额外完整10epoch，预热不是“联合10epoch中的前2K”。两段均weights-only；joint每5000更新和结束保存，保留最近5份，不导出optimizer大文件。只支持显式从权重新开run，不冒称精确resume。

Slurm：scripts/cluster/sidecar_grounded_clean50.sbatch，分区mhan0035、QoS normal、单节点/一个srun任务启动torchrun4rank、CPU32、mem0，沿用旧脚本不固定物理GPU、不强绑节点、不设time（该分区已查MaxTime=UNLIMITED/DefaultTime=NONE）。实际资源/提交结果以集群为准。

## 目标布局

```text
/home/mingfeihangroup/xiaoming/
  projects/OpenWAM/
    .venv/                            # 复用既有环境，不重装
    .experiments/grounded_read_20261103/  # 本次完整训练源码快照
    scripts/cluster/sidecar_grounded_clean50.sbatch  # 新增提交入口
    transfers/grounded_read_20261103/   # 代码包/哈希/安装提交说明
    outputs/sidecar/grounded_read_job<jobid>/
      warmup/launch.log
      warmup/runs/<timestamp>/checkpoint_step_2000.safetensors
      joint/launch.log
      joint/runs/<timestamp>/checkpoint_step_21480.safetensors
      final_checkpoint.txt
  datasets/robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015/
    openwam_sidecar/aloha50/manifest.ready.json
  ckpts/OpenWAM-Alpha-Pretrain-Foundation-Model/
```

独立快照是为了避免目标项目2026-10-22旧版和本地当前版混用，默认旧sbatch/代码保持不变。新增sbatch允许OPENWAM_CODE指定快照，OPENWAM_VENV仍指向项目既有.venv；Slurm在项目根目录提交，输出也保留在项目根下面。

## 提交

部署完成后可直接执行目标项目内本次transfer目录的SUBMIT.sh；等价命令：

```bash
cd /home/mingfeihangroup/xiaoming/projects/OpenWAM
sbatch --export=ALL,OPENWAM_CODE="$PWD/.experiments/grounded_read_20261103" \
  scripts/cluster/sidecar_grounded_clean50.sbatch
```

脚本自动从项目根推导数据/ckpt路径。既有manifest已从SFTP读回，50tasks/2500episodes，raw/labels/stats都已经指向正确目标目录，无需重定位或重传数据。

路径覆盖沿用OPENWAM_CLUSTER_BASE、OPENWAM_DATASET、OPENWAM_FOUNDATION、OPENWAM_MANIFEST、OPENWAM_VENV/PYTHON_BIN、OPENWAM_OUTPUT。需要module环境时指定OPENWAM_CLUSTER_SETUP。不是要求所有机器固定同一套路径。

查看：`squeue -u xiaoming`；`tail -f slurm-openwam-grounded-2k-10ep-<jobid>.out`。子阶段日志由Slurm输出和warmup/launch.log、joint/launch.log保存；4rank metrics内`ea_grounding_guidance_strength`表示实际joint强度。warmup 2000等于4000micro；joint 21480等于42960micro。

## 已验证与边界

- CPU相关回归第一批63通过；额外warmup/LR/接续检查28通过（两批有重叠，不将91称为独立测试数）。含FP32/BF16、checkpointing、零强度输出/梯度/RNG不变、query角色/缺失手臂回退、Action梯度到Grounding head、无新增权重keys、默认原配置、脚本单作业先后次序/失败不进入第二段、4rank完成/截断文件拒绝。
- 实际单A100 tiny Wan/ActionDiT BF16：2次预热更新主干逐位不变且有梯度，保存state_dict严格重载；新optimizer下强度0/.25/.5三次更新/反传通过，峰值allocated约21MB（不含CUDA上下文）。不是完整Foundation模型容量/多卡吞吐证明。
- SFTP读取目标Foundation config，真实两阶段配置合并验证保留重建components/tokenizer，第二阶段backbone_warmup覆盖为0、Grounding引导启用；未试图复用旧optimizer。
- 原数据/标签/统计/augmentation语义不改，现有颜色增强语义风险与near/EEF参考点限制仍保留。
- 完整四卡生产更新、耗时、最终保存和闭环效果以目标Slurm实跑为准。仅在登录节点做小规模配置/文件验证，所有GPU训练通过sbatch提交。

本地证据：outputs/grounded_read_setup_20261103/，含tests.log、stage_tests.log、cuda_smoke.json/log、目标manifest/Foundation config/venv版本和target_config_acceptance.json。代码归档及源文件哈希在outputs/transfers/grounded_read_20261103/。

## 2026-11-03实际部署与提交

最终合并回归为77项通过（24.02秒）。代码包SHA256为`db08b1e9f6005647c1d0066aebf9698eee98a7a1940f3394e9366800913b4993`，629个源码文件逐文件哈希在目标端全部通过；目标原默认Sidecar模块和旧sbatch的哈希保持。真实目标50任务/2500条数据与标签均存在，normalization SHA匹配，Foundation 80D权重头及24,813,767,464字节长度有效，两阶段Hydra配置解析通过，复用torch2.7.1+cu128/accelerate1.15.0/deepspeed0.18.9。

通过`sbatch --test-only`后只提交一个实际作业：**1310504**，分区mhan0035、4×A100、节点gnode06，2026-11-03 04:24:50 UTC进入RUNNING并开始加载Foundation做warmup。正式联合段已配置`save_optimizer_steps: 5000`，不是每5000 microstep；预热仅到2000参数更新后保存交接，联合段5000/10000/15000/20000和最终21480保存，不写optimizer state。完整生产首更新与两阶段最终完成仍以实际日志为准。

实际输出：`/home/mingfeihangroup/xiaoming/projects/OpenWAM/outputs/sidecar/grounded_read_job1310504`。Slurm日志：项目根`slurm-openwam-grounded-2k-10ep-1310504.out/.err`；详细阶段日志`warmup/launch.log`和之后的`joint/launch.log`。目标transfer目录保留`submission.json`、`target_validation.json`、`source_verification.log`、`sbatch_test_only.log`，本地`outputs/grounded_read_setup_20261103/`同步证据。

### 04:32 UTC四卡生产首段监督结果

04:32:22 UTC四rank全部连续完成11次参数更新（23 micro，下一次累积中）；各rank更新序号从1连续、全部历史loss/指标有限，实际更新处grad norm为1.189–1.506，`ea_backbone_frozen_warmup`始终1。Write beta从零变为非零，符合Sidecar已更新；这里没有导出运行中WAM权重，不把freeze指标当作生产权重逐位比较。最近reduced总loss0.38868、Action0.04670、EA0.13201，仅说明数值健康。

排除首次初始化后中位耗时每micro约6.55秒，即每参数更新约13.1秒；单卡最大allocated48.06GiB、最大reserved51.81GiB，四rank持续推进，未见OOM/通信/非有限值异常。证据`outputs/grounded_read_setup_20261103/startup_observation.json`，目标同名文件位于transfer目录。真实四卡预热启动已通过，作业继续运行；完整2K保存、自动重载后的Grounding引导联合训练、最终checkpoint及闭环效果仍待实际运行，不能从11次预热更新声称这些阶段已通过。

### 04:41 UTC耗时估计

实时作业1310504仍RUNNING，warmup52/2000参数更新、105micro；最近100micro均值6.5446秒，对应13.0892秒/参数更新。按相同吞吐外推：预热剩7.08小时，额外21480更新的10epoch需78.10小时，总剩约85.18小时，理论完成时间北京时间2026-11-07 01:52。考虑联合段实际吞吐与权重保存/重载开销，当前给出剩3.5–4天（北京时间11月7日凌晨至中午附近）的预算范围；joint尚未启动，这不是实测joint速度，开始后应据新日志校准。当前loss有限，仅报告训练运行健康，不推断策略效果。

### 07:04 UTC进度

作业1310504在gnode06保持RUNNING（2小时39分），四rank均warmup708/2000更新、1416micro，完成35.4%；日志在检查前约3秒更新。最近100micro指标均有限，平均6.545秒/micro，参数更新约13.09秒；主干预热冻结标记1，最近总loss0.33978、Action0.02737、EA0.09573，更新处最近平均梯度范数1.186。最大allocated48.06GiB/reserved51.81GiB，Slurm stderr为空。预热剩约4.70小时，按当前速度预计北京时间今天19:46左右到2000，然后保存/重载进入joint；目前还没有checkpoint，符合2000步才首次保存设置。joint未启动，全程完成时间估计仍是北京时间11月7日前后；不由当前loss推断闭环收益。证据outputs/grounded_read_setup_20261103/progress_latest.json（当前快照，后续查询可更新）。


### 2026-11-03 12:27 UTC：2K保存/自动重载及正式joint实跑

作业1310504仍RUNNING/gnode06，预热四rank全部2000更新/4000micro；checkpoint_step_2000.safetensors为24,862,363,126字节且complete marker存在，既有阶段交接检查通过后正式joint已四rank达到166/21480更新（332micro，10epoch总预算0.77%）。joint实际ea_grounding_guidance_strength=0.0825，符合前1000更新0→.5的165次已完成更新强度；Action/EA/总loss约0.006566/0.061075/0.219387，近期指标有限。最近13.117秒/更新，最大allocated59.28GiB/reserved62.25GiB，日志约2.5秒内更新，Slurm stderr为空。解决真实预热保存/自动重载/joint四卡首更新未验，未证明引导闭环收益；按当前速度joint余77.66小时，理论北京时间11月7日02:07结束。joint尚未到5000故没有joint checkpoint。证据outputs/grounded_read_setup_20261103/progress_both_latest.json。


### 2026-11-04 09:19 UTC：正式joint完成27.4%，首次5000权重已保存

作业1310504仍RUNNING，四rank均5885/21480更新（11770/11771micro，下一micro边界采样，更新计数一致），日志延迟0–6.4秒；2K预热已结束。当前guidance=0.5，最近100micro所有指标有限，均值Action0.001216/EA0.05009/总loss0.18353。最近13.133秒/更新，最大allocated59.28GiB/reserved62.25GiB。joint checkpoint_step_5000.safetensors 24,862,363,126字节，complete marker和safetensors头/数据长度核对通过，2229 tensors含140 Sidecar tensors；这是保存结构检查，未做全文件hash/闭环评测。解决joint首次5000权重保存未验，后续10000/最终保存与闭环收益仍待实测。当前剩56.89小时，理论北京时间11月7日02:13完成。证据outputs/grounded_read_setup_20261103/progress_both_20261104T091935Z.json和checkpoint5000_validation.json。


### 2026-11-05 09:21 UTC进度

作业1310504仍RUNNING/gnode06，四rank均12463/21480参数更新，完成58.02%，近期100micro指标全部有限、日志延迟约2.1秒。最近13.135秒/更新，按当前速度剩32.9小时，预计北京时间11-07 02:15结束。正式5000和10000权重的complete marker/文件bytes及safetensors头-数据长度检查均通过；仅结构保存验收，未做全文件hash或闭环性能验证。两组继续运行，总8GPU；Sidecar当前guidance0.5。证据outputs/grounded_read_setup_20261103/progress_both_20261105T092135Z.json。
