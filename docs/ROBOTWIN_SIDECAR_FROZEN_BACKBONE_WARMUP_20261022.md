# Sidecar 先适配固定 WAM：2K warmup 新训练版本

2026-10-22 UTC。用户明确要求新增此训练方案，保留原联合训练及旧 isolation warmup；本轮实现与工程短测，不启动或替换正式训练。

当前行为已按用户后续澄清修订：完整计算并保留WAM真实参数梯度，参与正常梯度归约与全局裁剪，只在AdamW执行前跳过WAM组。下文09:24及之前的验收记录属于“叶子梯度清零”的初版；完整反向版本的独立验收见文末，不混用两版证据。

## 推荐方案

前2000次 optimizer updates：主干照常参与前向与梯度传递，但其参数、Adam动量/step和weight decay均不更新；Sidecar正常读、精炼、预测和写回，接收原Action/Video/EA损失的梯度。完成2000次更新后，直接联合更新WAM与Sidecar。

不增加“只更新WAM、冻结Sidecar”的第三阶段。此时Sidecar已经围绕固定主干适配，再冻结Sidecar让主干改变特征分布，可能重新产生不匹配；当前没有收益证据，且增加阶段长度等超参。若解冻确实出现loss/实际写入幅度突变，再考虑短暂降低并逐步恢复WAM学习率、同时保持Sidecar更新，而不是默认增加交替冻结。

2K沿用既有口径，指**2000次真实optimizer更新**，不是文件名中的2000 microsteps。Aloha当前acc=2，前4000 microsteps冻结主干，从4001 microstep起联合训练。warmup包含在10epochs内，不额外追加训练。总microsteps 42960、总optimizer updates 21480，WAM实际更新19480次；Sidecar更新21480次。对比时必须披露WAM更新预算减少2000，不称两者的WAM更新数相同。

## 与旧方案比较

|项目|旧 isolation warmup|新 frozen-backbone warmup|
|---|---|---|
|前2K主干权重|按原Action/Video FM更新|完全固定|
|Sidecar读取主干特征|detach后读取|正常读取，保留完整梯度通路|
|Sidecar辅助预测|正常|正常|
|三处Write|完全绕过|正常调用|
|Write的Action/Video学习|2K之后才开始|从第一阶段开始|
|2K后|完整联合|完整联合|
|主要优点|主干先适应目标数据，避免随机分支写入|Sidecar面对固定特征，读写接口同时适配，保护预训练主干|
|主要代价|Write未热身，切换时首次接入；主干特征在移动|主干少2000次更新；已有EA主导与小Gate问题不会自动消失|

针对“随机Sidecar尚未适配却从一开始与主干共同改变”这一问题，新方案机制上更直接，优先做短预算对照；不宣称已有成功率提升证据。旧方案仍有作为控制组的价值。

仍保留现有单零初始化：beta=0、Write Projection正常初始化。第一步残差为零，但beta有FM梯度；beta更新后Projection/上游关系表征才逐渐获得写回路径梯度。这是正常的零残差启动，不是“双零死分支”，没有强行放大Gate。

## 最小实现及边界

- 新独立配置 `configs/train_relational_sidecar_aloha50_frozen_backbone_10epochs.yaml` 继承旧isolation配置，仅将isolation_warmup_updates=0、backbone_warmup_updates=2000并改输出目录。
- 继续Sidecar LR 5e-5、WAM 1e-4、aux_loss_scale=.5、原标签/损失定义、ColorJitter开启、Aloha50/2500、bs32×4卡×acc2=全局256、10epochs；不叠加上轮提出的EA-heads-only路由或新Task Gate。
- 原模型/旧配置默认backbone_warmup_updates=0，行为不变。新旧两种warmup互斥；静态stage=warmup/action_only或禁用Sidecar时拒绝新模式。
- 不在ZeRO构造后改变requires_grad或重建分组。所有将来需要训练的WAM参数从开始就注册；前2K保留WAM真实参数梯度，并和Sidecar一起正常归约、裁剪。Sidecar的Action/Video/EA反向路径与正常joint一致。
- 使用原生AdamW的step-pre-hook，对冻结WAM组清除optimizer梯度：跳过动量/计数/weight decay更新，而不是只设LR=0或只把梯度设0。ZeRO-2每组的FP32扁平参数同样保留分组标记。永久冻结的VAE/text encoder等仍不进入optimizer。
- 此实现完整计算并归约主干参数梯度，仅在optimizer消费梯度前跳过WAM更新。它是严格的**参数更新冻结**，不节省正常反向的计算或显存；相较初版，WAM参与global clip可能改变Sidecar最终更新幅度。
- 使用已有trainer在每个microstep前传入的completed_optimizer_updates作边界，累积窗口内不切换；resume恢复同一计数与optimizer状态。阶段不新增模型tensor，checkpoint键兼容。日志新增ea_backbone_frozen_warmup=1/0。
- 全局学习率scheduler在warmup期间仍按已有更新数推进；主干解冻时采用该时刻的计划学习率，不重置scheduler，也没有隐含第二次LR warmup。Sidecar/主干Adam状态年龄不同是预期。
- 推理eval仍是完整Sidecar，不依赖计数；只加载weights做finetune会从0开始warmup，接续原训练必须使用完整resume，而非手动改step文件名。

## 启动与版本切换

当前服务器：新入口复用现有四卡正式launcher和CPU-ready交接门槛。先CHECK_ONLY，实际启动需另行安排空闲卡，不能直接覆盖当前ManiSkill任务。

```bash
cd /remote-home/jinminghao/OpenWAM
export EA_HANDOVER_DIR="$PWD/outputs/sidecar/handover_frozen2k_$(date -u +%Y%m%dT%H%M%SZ)"
CHECK_ONLY=1 bash scripts/train_relational_sidecar_aloha50_frozen_backbone_10epochs.sh
# 正式启动时用全新EA_HANDOVER_DIR，将CHECK_ONLY改0；本轮未启动。
```

目标Slurm环境复用原分配、路径与preflight，现有launcher增加可选OPENWAM_TRAIN_CONFIG，默认仍旧isolation配置：

```bash
cd /home/mingfeihangroup/xiaoming/projects/OpenWAM
OPENWAM_TRAIN_CONFIG=train_relational_sidecar_aloha50_frozen_backbone_10epochs \
OPENWAM_OUTPUT="$PWD/outputs/sidecar/aloha50_frozen_backbone2k_10epochs" \
CHECK_ONLY=1 bash scripts/cluster/train_sidecar_isolated.sh

# 待正式授权/安排GPU后，使用现有sbatch：
# export OPENWAM_TRAIN_CONFIG=train_relational_sidecar_aloha50_frozen_backbone_10epochs
# export OPENWAM_OUTPUT="$PWD/outputs/sidecar/aloha50_frozen_backbone2k_10epochs"
# sbatch --job-name=openwam-sidecar-frozen2k scripts/cluster/sidecar_isolated_clean50.sbatch
```

切回旧方案使用 `train_relational_sidecar_aloha50_isolated_10epochs`，原脚本默认行为不变；原始联合配置也保留。新profile从Foundation初始化，不覆盖运行中模型或原训练产物。

## 验证与下一步

CPU真实tiny Wan/ActionDiT已检查：新模式前向等于正常joint；相同输入下Sidecar梯度等于正常joint；主干叶子梯度为零、权重/Adam状态不变；1999→2000后主干恢复更新；checkpointing开关；阈值前/边界/后的完整optimizer保存恢复与连续下一步逐位一致；eval忽略冻结计数；拒绝混用warmup。相关旧架构/optimizer/cluster回归50项通过。

两卡BF16 ZeRO-2微型真实Wan验证正在执行，证据在`docs/evidence/sidecar_backbone_warmup_20261022`；初次独立harness遗漏输入设备搬运导致失败，已仅修正test helper，不修改生产数据路径，原失败日志保留。最终验收见后续记录。

下一步在相同数据顺序、增强、总预算、loss/LR和初始化下对比旧isolation与新frozen模式，先看跨边界稳定性，再配对闭环。旧42960梯度分析不代表这两种新warmup已有训练结果；不存在“更合理就保证更优”。仍需matched Vanilla及写回实际r、关系遗忘、Action XYZ/rotation/gripper和成功率证据。


## 最终工程验收（2026-10-22T09:24:45.484493+00:00）

- 51项CPU回归通过，覆盖实际tiny Wan前向、Sidecar梯度、主干权重/Adam冻结、1999→2000、checkpointing、阶段前/边界/后的Adam恢复、旧joint/isolation/参数组与Slurm默认/新profile分配。
- 物理GPU5/6两卡真实BF16 ZeRO-2、fused AdamW、acc2通过。阈值缩为2个optimizer updates：完成1/2更新时WAM逐位未变且无Adam历史、Sidecar已变；第3次WAM/Sidecar都更新。分别在1/2/3更新保存并重建模型/optimizer，恢复后下一次更新与不中断原模型逐位一致；两rank均pass、进程正常exit0。缩阈值验证相同计数逻辑，不声称实际训练了2000步。
- GPU fixture使用真实微型Wan/ActionDiT与原compute_loss；没有加载全量Foundation、真实数据dataloader或LR scheduler。CPU保留现有LR scheduler回归；不能把此结果称为全量四卡容量/性能/完整trainer启动验收。
- 新Hydra配置resolve和shell语法检查通过，源SHA记录validation.json。未启动正式训练，当前ManiSkill/Robotwin评估进程不停止/不改配。
- 失败的第一次GPU输入CPU/CUDA错误仅在test helper修正，原日志保留；最终成功结果见rank0.json/rank1.json。证据目录含少量tiny恢复checkpoint，不包含全量24GB权重。

范围内已解决：ZeRO-2两卡静态参数分组下的严格更新冻结、梯度累积边界、阶段前后optimizer状态恢复。仍待：全量四卡启动/显存/耗时、完整数据训练、Stage2关系遗忘或loss突变、闭环收益和matched Vanilla因果对照。此版本严格遵循用户新提出的warmup策略，未额外加入上轮EA梯度路由候选。


## 完整反向修订验收（2026-10-22T13:42:33.785223+00:00）

本次仅修正WAM叶子梯度清零，使前2K与正常joint具有相同前向/真实反向/全局裁剪，optimizer执行前才跳过WAM组。53项CPU测试通过；同输入下全部参数梯度、clip结果及Sidecar更新与joint逐位一致，WAM权重/Adam保持不变，阈值后恢复更新。两个checkpointing模式及阶段前/边界/后完整Adam恢复均通过。

GPU5/6两卡真实tiny Wan/ActionDiT、BF16、fused AdamW、ZeRO2、acc2，阶段阈值缩为2：第1/2更新WAM真实反向非零但权重/Adam不动、Sidecar更新，第3更新WAM恢复更新；两rank在1/2/3更新保存重载后下一步均与连续模型逐位一致，进程exit0。Hydra解析/shell语法通过。独立证据：docs/evidence/sidecar_full_backward_warmup_20261022/validation.json、cpu_tests.log、rank0.json、rank1.json。

这是微型工程验收，未加载完整Foundation/真实数据四卡训练，GPUfixture无LR scheduler；尚无正式训练或SR收益。既有训练进程未操作。原09:48增量包不含这次修订，保留历史归档；源端配置/脚本仍使用train_relational_sidecar_aloha50_frozen_backbone_10epochs。
