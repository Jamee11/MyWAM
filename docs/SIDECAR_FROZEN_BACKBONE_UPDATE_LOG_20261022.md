
### 2026-10-22：基于旧isolation2K包增量升级至frozen-backbone2K

保留旧不写回方案；新增前2000次optimizer updates中Sidecar正常读/预测/写入、WAM参数/Adam动量/衰减不更新，之后直接joint。未添加WAM-only第三阶段或EA-heads-only路由；沿用数据/增强/LR/aux/10epochs，永久冻结模块不变。新增GPU-8A100分区、单节点4GPU、10天上限Slurm脚本；保留调度器GPU分配，不锁gnode06。

依据：原write弱、辅助梯度主导的诊断，以及用户要求保护预训练主干、提前适配Write。CPU51项与两卡tiny Wan BF16/ZeRO2/acc2跨阶段和Adam恢复已通过；新Slurm与旧交付包上的增量集成验收见随包VALIDATION.json。未在目标集群提交或完成全量四卡训练，暂无SR收益证据。

风险保留：冻结仍计算/归约主干梯度，不承诺省显存；全局LR scheduler继续推进；10epoch内WAM比旧方案少更新2000次；Stage2的EA主导和关系遗忘风险不自动解决。数据、checkpoint、venv均复用；不能用旧isolation完整state续跑来冒充新策略从头训练。详见docs/ROBOTWIN_SIDECAR_FROZEN_BACKBONE_WARMUP_20261022.md。


### 2026-10-22T13:42:33.785223+00:00：按用户澄清保留完整WAM反向，仅跳过前2K参数更新

原因：初版helper在WAM叶子参数上清零梯度，虽然保留Sidecar激活反传，却没有完整保留用户要求的WAM真实参数梯度。用户明确选择先实现正常读写/完整反向的warmup，不叠加其他实验。

修改：移除sidecar_warmup.py中的叶子梯度清零hook；保留原生AdamW step-pre-hook，WAM正常前向、反向、累积、归约和全局裁剪后，仅在optimizer执行前跳过WAM组，参数、动量、step和decay均不更新。Sidecar正常Read/Refine/Head/Write并接收Action/Video/EA梯度，达到2000 optimizer updates后恢复joint。沿用现有独立profile/启动脚本、LR、损失、Gate初始化、ColorJitter和10epoch预算；不加heads-only或第三阶段。永久冻结模块不变。

工程验收：53项CPU测试通过，检查全部真实梯度与joint一致、共同clip后仍一致、Sidecar一次更新与joint逐位一致、WAM权重/Adam不动、1999→2000边界及恢复模型/完整Adam状态一致。GPU5/6真实tiny Wan/ActionDiT、BF16/fused AdamW/ZeRO2/acc2缩阈值2通过：两rank均捕获非零有限WAM反向梯度，warmup仅Sidecar变、joint均变；1/2/3更新处保存重载后下一更新逐位一致。测试进程exit0。Hydra解析及三个入口shell语法通过。证据docs/evidence/sidecar_full_backward_warmup_20261022/validation.json。

边界：WAM重新参与global clip，Sidecar裁剪后更新可能不同于初版；正常反向不节省显存。全局scheduler继续推进、10epoch内WAM少2000次更新、Stage2 EA主导风险保留。GPUfixture不含全量Foundation、真实数据或LR scheduler；未启动正式训练、未停止已有ManiSkill四卡作业、未证明闭环收益。09:48历史传输包未重新打包，仍保留旧叶梯度清零逻辑，迁移文档已标明。
