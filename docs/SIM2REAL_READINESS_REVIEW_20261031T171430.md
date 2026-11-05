# OpenWAM 实现与真机部署复核

时间：2026-10-31T17:14:30.349166+08:00。SSH alias rlbench-simpler-117，实际hostname 2ed14158a666，项目 /remote-home/jinminghao/OpenWAM。
本轮只读分析代码与既有实验，未运行新实验、未改代码/配置/训练/部署、未操作机器人。以下建议未经实验验证。

## 已核查证据
- docs/ROBOTWIN_ALL_EVAL_SUMMARY_FROM32K_20261031.md：1030 warmup2K+full10epoch21480，全50任务双域每条件30次，Clean88.67/Random61.00/overall74.83；官方100次参考89.36/48.72/69.04。语言池、预算、增强及推理差异限制归因，需matched Vanilla。74.83不是multisim midtrain的下游结果。
- outputs/midtrain/multisim_1ep_b512_warmup500_20261025T154412Z/launch.log末尾确认56892 microsteps及Saved final；最终checkpoint_step_56892.safetensors存在。acc4对应14223更新。未全量hash/加载最终权重或恢复，因此不是完整checkpoint可用性验收。文档记录四来源等权、global512、WAM前500更新延迟；当前证据不足证明下游收益。
- openwam/model/modules/relational_sidecar.py：15 tokens/3 stages/512维，Ground4、Relation4、Summary1、Plan6；当前帧读取，latent写入Action；无显式progress监督。ea_rsig_v3.py计划loss仅moving及方向，非完整轨迹。
- docs/SIDECAR_RELATION_TOKEN_AND_DUAL_OBJECT_AUDIT_20261028.md：双轨不是通用关系图；near固定参考点0.1m，存在contact=1 near=0；应先验证参考点/尺度，不能直接调整阈值。
- docs/MULTISIM_MIDTRAIN_ACTION_AND_STORAGE_AUDIT_20261025.md：RoboTwin未来实测绝对EEF，MS/RLB控制目标；编码一致不代表控制语义一致。腕相机near .02/.1差异和multisim缺失槽增强问题仍需保留。
- docs/MANISKILL_REAL_ROBOT_INPUT_OUTPUT_DIAGNOSIS_20261027.md：已有接口字节级预处理对齐，但历史真机画面缺目标区域、视角/夹爪尺度不同；CHUNK_STEPS=1 STEP_SECONDS=1.0与20Hz训练不同；base identity标定假设未验证。HTTP成功不证明真机成功；未验证今天真实客户端配置。

## 建议与最小实验
1. 冻结74.83方案作为主参考，完成同协议matched Vanilla；30次先筛选，最终统一seeds/语言/缓存/执行协议，报告逐任务差异与不确定性。
2. 真机只选当前最成熟且具备可验证控制接口的平台，2–3任务；midtrain→目标本体仿真finetune与Foundation→同配方finetune比较，避免将本体适配收益误归因midtrain。单臂和双臂Piper不同时扩展。
3. 目标仿真微调保留现有架构，独立目录、新optimizer、明确stats转换及动作槽位，短validation选点；不自动照搬2K冻结或高LR。训练数据摄像头/base/TCP/夹爪/频率与真机对齐，验证专家动作回放。
4. 部署先校准base/TCP/相机/桌面，核对实际客户端，量测延迟；保持训练动作时间含义，以短prefix重规划做受控验证，不能简单慢放。控制器限位/碰撞/停止和硬件急停验收保留。
5. 采0/5/10或20条真实示范形成数据效率曲线；缺特权标签时通过有效mask禁用对应辅助loss，不能零填成负例。先保守action适配，必要时小比例源仿真回放防遗忘，是否支持需实现前核查。
6. 进度/几何/计划扩展为条件性后续：仅在明确失败需要时增加一种监督。progress应面向任务里程碑而非时间比例；几何可用相对位移幅度/旋转/夹爪事件；保留当前baseline。
7. 论文主张聚焦仿真特权关系监督改善鲁棒控制及真实数据效率；需要matched Vanilla、无EA同容量对照，runtime write-off仅补充机制证据。

## 时间优先级
10/31–11/02锁定平台与2–3任务、核实控制链并取得成对首轮结果；11/03–11/07完成仿真目标本体适配及few-shot；11/08–11/12独立测试/关键消融；11/13起冻结结果与写作。均为建议安排，不是已授权训练计划。

## 回滚与范围
仅新增本文件，无覆盖、删除或代码改动；如需撤销，审阅后删除本次新增文档即可。未修改现有风险清单，不把未解决问题标记完成。
