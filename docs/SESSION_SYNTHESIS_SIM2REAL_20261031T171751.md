# 三session综合分析
时间：2026-10-31T17:17:51.991840+08:00
服务器：rlbench-simpler-117 / 2ed14158a666；rlbench-simpler-A100 / 7372e55579bc。
项目均为 /remote-home/jinminghao/OpenWAM。本轮只读历史/代码/既有结果，无新实验、无代码或训练修改。

## 来源与范围
解析117 session 01a142a1-dceb-7380-a880-46b4e9016b08（311用户消息/232 final），A100 sessions 01a13092-50a3-7102-9169-59200cca4f90 与01a181e3-94c3-7402-bc19-4a6ace7bdc89。定向提取用户决策、近期final及真机/midtrain相关段落，未逐条人工审阅所有工具原文。117 memory目录为空；A100 MEMORY.md与rollout_summaries/2026-10-12T14-44-17-wnq8-openwam_sidecar_analysis_and_bitdistill_migration.md有第一长session摘要，未发现XA独立摘要。历史快照不冒充当前H100在线状态。

## 修正与新增结论
1. 74.83来自PatchWAM预热2K权重，新optimizer联合10epoch，总数据暴露约10.93epoch；不能与包含warmup的原10epoch当等预算，也不是多仿真midtrain下游成绩。新optimizer、LR重启、增强、额外预算共同变化，不能单独归因Sidecar/warmup。
2. 1031官方C2R→新Sidecar2K→低LR5epoch历史仍在评估；54/100条件79.57%只是偏Clean子集，不能与74.83全任务排名。旧32k、warmstart21480、multisim56892三条谱系必须分清。
3. 最新21480实际Write/hidden约0.270%，非旧模型接近零量级；gate倍率0/1/2/5/10/100的GT位置误差17.275/17.227/17.291/18.219/20.738/92.193mm。保持倍率1；这些是12相关窗口离线指标，非SR。
4. XA已实现并正式训练；最新查阅历史快照7676/21480，非本次直连H100核实。8→20、EMA .9999、512 projector，8x16acc2/global256。5K相对2K起点温和漂移22.428→2.249mm，强光49.598→11.029mm，但混有联合训练和增强变化；强光还未优于旧10epoch final14.218mm的匹配预算结论。应同起点/同增强/同更新lambda0对照，先闭环再扩展。
5. XA权重.5封顶，加权loss约EA5%，未达到80%目标；不按loss比机械放大到8。EMA .9999在5000更新后保留约60.65%初始化权重影响（参数EMA数学系数，非模型功能占比）；需判断teacher滞后，不据此立刻改decay。对齐loss直接经student8及更浅路径，不能称直接监督所有深层。
6. 进度仍无显式GT；plan仍是8/16/32步moving+单位方向，没有幅度/旋转/夹爪事件。定位是辅助热图、双角色轨道，不是通用3D object tracker。当前论文表述应与实现一致。

## 部署决策
主线：现有multisim56892→当前最成熟目标本体仿真微调→零示范真机基线→同示范0/5/10或20 few-shot。与Foundation→同数据同配方目标微调比较，验证midtrain增量。沿用已初始化Sidecar，不能无依据再次随机初始化/重复2K冻结；新优化器/stat转换/有效动作槽位需显式记录。
历史已确认有可控单臂及接口，因此先2–3单臂任务；只有Piper硬件、标定、遥操作明显更成熟才改主线，不同时扩两个平台。双臂额外同步/双base/TCP/碰撞/夹爪合同。
先解决已有真机诊断：目标靶圈缺失、相机尺度/腕部夹爪不可见、base identity未标定、20Hz训练与1秒单步执行差异。把物理控制频率与重规划频率分开，量测延迟，短prefix闭环验证，保留硬件急停/IK/碰撞验收。
真实数据无oracle需正确mask缺失辅助项，不填零负例；相对几何/里程碑进度/事件plan按失败证据后置，当前不扩模型。
论文最少证据：matched Vanilla+完整方法、同容量无EA、XA同增强lambda0，完整RoboTwin与真实数据效率成对结果。运行时write-off不能替代训练消融。优先部署和闭环，不继续堆离线probe。

## 文档与回滚
仅新增此分析；现有代码/风险清单不改。若撤销可经审阅删除此新增文档，无既有文件备份需求。本地对应记录保存在/Users/xiaoming/modified_files/。
