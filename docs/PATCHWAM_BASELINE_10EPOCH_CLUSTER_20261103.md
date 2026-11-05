# 原版OpenWAM + PatchWAM：4卡10epoch Sidecar整体消融

用户要求在mhan0035分区再提交一份原版WAM、PatchWAM增强、无完整Sidecar的10epoch训练，并明确不绑定gnode06；新增节点尚未出现在分区中，可以先PENDING等待资源。保留正在运行的Grounding-guided Read作业1310504，不设置对它的结束依赖，不抢占/停止其他作业。

## 实验设置与可比范围

- 与Sidecar实验使用同一Foundation `checkpoint_step_154000.safetensors`、同一50任务/2500条Clean数据和normalization、相同80D绝对EEF/三相机/33帧契约、完整PatchWAM，ColorJitter关闭。
- 原版`joint_self_attn`架构，不实例化Sidecar/EA heads、不读取EA标签输入、不增加EA/alignment损失；保留原Video/Action MoT与主干损失。复用既有metrics入口只记录原训练指标，入口文件名中的sidecar不会创建该模块。
- 4×batch32×acc2=global256，完整10epoch=21480参数更新/42960micro；WAM/Action学习率1e-4、5%线性warmup+cosine、BF16/ZeRO2/checkpointing/seed42均与联合训练相同。新optimizer/scheduler/step0，直接从Foundation训练；不执行没有Sidecar可训练的2K预热。
- 每5000参数更新和最终21480导出权重，保留最近5份，附config/tokenizer/normalization，不保存optimizer大state。

这是“原版WAM+同训练配方”对完整Sidecar方案的训练期消融，不是加载已训练Sidecar再在推理时关闭。两者主干更新预算一致；完整方案另有2000次Sidecar-only更新，因此不声称总训练计算量完全相同。共同seed不保证不同模型下每次增强随机实现逐位配对。最终需同闭环任务/seed/cache/渲染协议比较；训练loss不能单独证明模块收益。

## 新增入口与目标布局

仅新增配置`configs/train_alpha_patchwam_baseline_10epoch_4gpu.yaml`与`scripts/cluster/baseline_patchwam_clean50.sbatch`，继承已有完整PatchWAM配方；默认模型、旧5epoch baseline、当前Sidecar配置/运行快照不修改。新增独立baseline快照复用已验证的Grounded源码归档，再加入本次文件，固定共同训练代码版本。

目标项目`/home/mingfeihangroup/xiaoming/projects/OpenWAM`；源码`.experiments/patchwam_baseline_20261103`，交付`transfers/patchwam_baseline_20261103`，输出`outputs/baseline_patchwam_job<jobid>`。复用项目`.venv`与既有数据/Foundation目录，不上传大文件。

```bash
cd /home/mingfeihangroup/xiaoming/projects/OpenWAM
sbatch --export=ALL,OPENWAM_CODE="$PWD/.experiments/patchwam_baseline_20261103" \
  scripts/cluster/baseline_patchwam_clean50.sbatch
```

SBATCH仅partition=mhan0035、QoS normal、nodes1、GPU4、CPU32、mem0，不设nodelist/排除节点/指定reservation/依赖/优先级/time。排队后能否运行取决于新节点实际加入此分区及可分配资源；不将成功提交当作GPU已启动。日志项目根`slurm-openwam-patchwam-baseline10ep-<jobid>.out/.err`及每rank metrics，权重保存在输出`runs/<timestamp>`。

## 本地检查

13项检查通过（103.51秒）：配置逐字段对比当前joint的数据/增强/主干训练预算、旧5epoch baseline配置保持、tiny原版模型确实没有Sidecar参数/EA损失且能反传更新Action参数、已有PatchWAM的时序一致性/相机拼接/标签不变等测试。SBATCH语法通过。新快照以当前Grounded归档SHA256 `db08b1e9f6005647c1d0066aebf9698eee98a7a1940f3394e9366800913b4993`为共同源码基底；已核对原版模型、reader、trainer、metrics入口与本地相同。用户配额当前533/1024GB，Sidecar与baseline预期新增权重合计约275GB，不写optimizer状态。证据outputs/patchwam_baseline_setup_20261103/local_verification.json。实际四卡更新仍待排到资源。

## 实际提交：作业1310914

2026-11-03 08:05:24 UTC（北京时间16:05）部署校验及`sbatch --test-only`通过后，已提交唯一实际baseline作业**1310914**。归档1,594,273字节，SHA256 `78bc2ec6924a04cef96947b3b9f0fbda3c100c99a905e0a9f340ce3de3a943f9`，目标端634个源码哈希全部通过；目标配置确认原版架构/无Sidecar/无EA、50任务2500条/normalization SHA、PatchWAM、global256/10epoch/save5000。Foundation与数据原地复用。

Slurm记录`Partition=mhan0035`、`gres/gpu=4`、`ReqNodeList=(null)`、`ExcNodeList=(null)`，没有绑定gnode06。当前PENDING等待调度；同一检查中原作业1310504仍RUNNING，不改动该作业。新增节点需由集群实际注册并加入mhan0035后才能参与调度，不能把PENDING称作训练已启动。

实际输出根`/home/mingfeihangroup/xiaoming/projects/OpenWAM/outputs/baseline_patchwam_job1310914`；日志项目根`slurm-openwam-patchwam-baseline10ep-1310914.out/.err`，待分配资源后创建。目标transfer保留submission.json/target_validation.json/resolved_config.yaml/job_state.txt/queue.txt/source_verification.log，本地副本在outputs/patchwam_baseline_setup_20261103。后续节点分配/首更新/最终权重和闭环效果尚未验证。


### 2026-11-03 12:27 UTC：已获资源、真实四卡更新

作业1310914已RUNNING，Slurm实际选择gnode06（提交时未绑定节点）；已运行3小时19分，四rank均895/21480更新、1790micro，完成完整10epoch预算4.17%，日志约1秒内更新。metrics仅原版Video/Action FM与梯度，没有EA指标；最近总loss0.130525、Video0.124602、Action0.005924，近期所有指标有限。最近13.053秒/更新，最大allocated59.15GiB/reserved62.97GiB，无已观察到的异常/非有限值；stderr主要为正常tqdm进度，不能以非空stderr判故障。解决该baseline四卡首更新/分配未验，尚未到5000首次保存，完整checkpoint与闭环性能仍待验证。按当前速度剩74.64小时，理论北京时间11月6日23:05结束。当前Sidecar另4卡同样RUNNING，合计8卡。证据outputs/grounded_read_setup_20261103/progress_both_latest.json；两组步数不同且Sidecar总loss含EA，不比较当前总loss判断结构优劣。


### 2026-11-04 09:19 UTC：原版完成30.92%，首次5000权重已保存

作业1310914仍RUNNING，四rank均6641/21480更新（13282micro），日志延迟约1.8秒；最近100micro指标全部有限，均值Action0.001223/Video0.13022/总loss0.13145。最近13.074秒/更新，最大allocated59.15GiB/reserved62.97GiB。checkpoint_step_5000.safetensors 24,813,767,464字节，complete marker及safetensors头/数据长度通过，2089 tensors/Sidecar tensors为0；未做全文件hash和闭环评测。解决原版首次5000权重保存未验，余下保存与性能仍待实测。剩53.89小时，理论北京时间11月6日23:13结束。两组同用8卡，当前滚动loss来自不同训练步数/样本且Sidecar含EA，不能据此判定结构优劣。证据outputs/grounded_read_setup_20261103/progress_both_20261104T091935Z.json和checkpoint5000_validation.json。


### 2026-11-05 09:21 UTC进度

原版作业1310914仍RUNNING/gnode06，四rank均13254/21480参数更新，完成61.7%，近期100micro指标全部有限、日志延迟约3.5秒。最近13.062秒/更新，按当前速度剩29.85小时，预计北京时间11-06 23:12结束。正式5000和10000权重的complete marker/文件bytes及safetensors头-数据长度检查均通过；未做全文件hash/最终保存/闭环评测，不凭当前loss比较结构收益。证据outputs/grounded_read_setup_20261103/progress_both_20261105T092135Z.json。
