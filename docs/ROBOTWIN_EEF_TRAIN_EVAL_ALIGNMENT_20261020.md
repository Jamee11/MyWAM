# RoboTwin Sidecar：训练与评估绝对 EEF 语义核查

日期：2026-10-20 UTC。范围：Aloha Sidecar 训练 run `aloha50_C_joint_10epochs_20261016T131709Z/runs/2026-10-16_13-50-52` 与本机 32k 官方环境评估。只读检查原数据与实现，不修改训练、模型或评估协议。

## 直接结论

用户关注的是控制目标与实际执行位姿的差别，而非仅坐标/四元数格式。

| 环节 | 实际采用的 EEF |
|---|---|
| 训练当前状态 proprio(t) | raw HDF5 `endpose/{left,right}_endpose[t]`：仿真 link 的实际世界位姿，按 RoboTwin endpose 参考点和轴定义转换 |
| 训练动作监督 action(t) | 同一实际位姿序列的 t+1…t+32，绝对位姿；不是原专家规划器保存的期望命令 |
| 评估当前状态 proprio(t) | `get_obs → get_arm_pose → get_{left,right}_ee_pose → _trans_endpose`，读取实际 link `global_pose` |
| 评估动作执行 | 模型预测绝对位姿，经解码作为 `take_action(action_type='ee')` 的目标，重新调用左右臂规划器，驱动关节后得到新的实际位姿 |

因此，没有发现“训练状态是 IK 目标、评估状态是实际位置”的混用；但确实存在 **训练监督是专家实际到达位姿，评估将预测位姿作为控制目标重新执行** 的区别。即使模型完美复现标签，也不能推出再次执行后的轨迹完全等于专家原轨迹。规划、控制跟踪、动力学与接触误差仍可能影响成功率；现有代码检查不能宣称这些误差可忽略。

概念上训练的是 `observed_EEF(t) → observed_EEF_expert(t+1:t+32)`，执行后则得到 `controller(predicted_target) → observed_EEF_new`。预测误差与后者的控制跟踪误差应分开测量。

## 证据链

- `outputs/sidecar/aloha_4350/manifest.ready.json` 指向用户指定数据集的 `raw/aloha`，50 任务、2500 轨迹、552275 帧。不是将目录主 qpos 字段或旧 drive-target FK 特征当成这次训练的 EEF。
- `openwam/dataloader/ea_rsig_lite.py:LiteRoboTwinSource._read_eef_actions` 直接读取 raw endpose。`openwam/dataloader/robotwin.py:_build_sample` 先归一化和映射，再以第 0 帧作 proprio、第 1…32 帧作 action；视频从同窗口取 0/4/…/32 帧。
- 采集侧本地 `/remote-home/jinminghao/DiT4DiT/.external/RoboTwin/envs/_base_task.py:get_obs/get_arm_pose` 和 `envs/robot/robot.py:_trans_endpose` 读取实际 `left_ee/right_ee.global_pose`。这是实际位姿的来源证据，不是根据字段名推测。
- 当前官方环境 `.external/robotwin-official-20261018/RoboTwin` 使用同一 EEF getter 机制。`envs/_base_task.py:take_action` 对 EE 目标调用 `left_plan_path/right_plan_path`，再执行规划关节轨迹，不能把目标值视为执行后的测量值。
- 一个容易混淆的地方：`get_left/right_arm_jointState` 实际读取 `joint.get_drive_target()`；该字段名虽然含 State，却是关节驱动目标。此次 EEF reader 没有由它做 FK。夹爪字段也是官方报告的控制开度，不能把整个 20D 向量一概描述为实际关节传感读数。
- 训练与当前 Sidecar client 显式按 raw wxyz 解释四元数，转换到 column-Rot6D；下发时转回 wxyz。Sidecar 当前状态和 action 使用官方 endpose 参考点；辅助关系标签使用沿该坐标局部 x 方向偏移 0.12 m 的 TCP。两者定义不同且用途明确，不应把 12 cm 偏移加到控制 action。
- checkpoint 保存的自有 min-max 统计用于训练和部署；20D 的左臂 0…9 / 右臂 10…19 映射至模型 80D 的 0…9 / 34…43。两侧均以 BF16 输入模型；格式往返误差、BF16 舍入和物理跟踪误差不是同一种误差。

## 尚未解决的控制层问题与下一步

本次没有执行新的物理回放，未得到“目标位姿 vs 执行后实际位姿”的 mm/degree 分布，不能保证此项不会影响训练所得策略的闭环表现，也不能据此认定当前成功率下降由它造成。

下一项有针对性的验证应在独立输出中，用专家实际未来 EEF 标签直接替代模型预测，经当前官方 `take_action('ee')` 完整回放，逐动作记录目标和执行后实际位姿的欧氏位置误差、旋转测地误差、规划失败、接触阶段误差和最终成功率。同时核对采集帧间隔与评估每动作的实际物理推进时间。不得把当前时刻的控制目标与下一时刻实测随意相减而混入时间偏移，也不得先改标签来源或控制器再称公平验证。

若该专家标签回放仍明显漂移，才有证据进一步区分控制协议、采样节奏和 achieved-pose 监督的限制；若误差小，则继续从模型预测与泛化分析。更换成原 command EEF 标签会改变训练数据语义，须先单独确认并保留当前版本。

数值核查程序和结果保存在 `docs/evidence/robotwin_eef_alignment_20261020/`；其几何往返检查不等于物理跟踪验证。

补充：用户澄清后聚焦 command/achieved 语义；可选全量数值程序在共享盘依赖导入阶段中止，未产出数值结果（见 evidence/status.json）。本文不声称数值审计或物理跟踪验证已通过。
