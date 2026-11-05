# ManiSkill20872 初始位姿、chunk闭环与action链路审计

完成时间：2026-11-04T13:32:51.597577+00:00。统一排查清单：[20项问题、证据和验证方法](MANISKILL_REAL_ROBOT_ISSUE_CHECKLIST_20261104.md)。

## 范围

沿用现有5901服务，GPU6主体/T5GPU2、20872、10步去噪、CFG1、eager、无DiT cache、seed42。仅增加outputs下独立诊断脚本和文档，不改模型/训练/部署共享逻辑、参考RobotWorkspace或现场客户端，不操作真机。仿真只用GPU5临时单进程，已退出，原模型服务保持ready。

## 1. 初始EEF到底是否超出训练范围

本次扫描全部选入ManiSkill微调训练的16000条trajectory，共2687589个state（含各episode末帧，训练action窗口2671589个）。这不是所有早期多机器人预训练数据的全集。

| 范围 | x (m) | z (m) |
|---|---|---|
| 全8任务所有训练时刻 | .38044–.80005 | .00939–.32776 |
| place_sphere全部2000条轨迹所有时刻 | .50646–.71353 | .01396–.22006 |
| place_sphere全部2000条初始状态 | .58712–.64165 | .13002–.22006 |
| 本次真机最早place_sphere状态 | .45538 | .23770 |

因此：真机初始XYZ位于全8任务的总体轴对齐范围内，但其x/z超出了place_sphere全部训练轨迹的对应范围，尤其偏离任务初态。它与最近一条place_sphere训练初始XYZ相距约158.15mm。矩形范围不代表联合状态分布或可达性，不能由“位于全任务范围内”推出该任务条件处于训练分布。

朝向也有区别：真机初态与全部16000条训练初始旋转之间，最近SO(3)角差70.56°，中位78.61°；place_sphere最近70.56°、中位78.56°。此前对50条轨迹全部阶段的抽样最近角差5.7°，是不同统计口径，不能用来否认初始姿态差异。

部署min-max归一化后的real0为约[.42658,-.03159,-.64585,.97986,.19948,-.00936,.19946,-.97532,.09473,1]，全部在有效范围内。这是任务状态分布偏离，不是已经发现归一化越界/裁剪。

## 2. 一条真实训练初始EEF

来自用户指定family的实际训练版本：place_sphere/train/clean/slot0016/seed105001600_speed0.35/Clean/replay.h5，frame0。对应B目录没有被该family的训练manifest选入。

```text
frame: model_robot_base
point: panda_hand_tcp
position xyz (metres):
[0.611998618, -0.008745952, 0.170624807]
quaternion xyzw:
[0.710785428, 0.703373709, -0.006999531, 0.000712831]
gripper: 1.0 (open)
EEF10:
[0.611998618, -0.008745952, 0.170624807,
 0.010432864, 0.999885559, -0.010953102,
 0.999905527, -0.010529834, -0.008833232,
 1.0]
```

完整源记录、旋转矩阵、两种四元数顺序和训练关节角见`outputs/analysis/maniskill20872_pose_chunk_audit_20261104/reference_reset_pose.json`及`geometry_audit.json`。优先用四元数避免Euler约定混淆。

这组数是模型base中的策略TCP目标，不是直接发给MoveIt的link8目标。需先确认真实base和TCP，再按目标链转换并验收可达性/桌面/夹爪空间。不能仅把上报EEF改成这组数字。参考FR3用相同训练关节角得到的TCP z约.168625m，比仿真该TCP低2mm；不要混用“同关节角”和“同TCP目标”两种复位口径。

## 3. 相同场景的真实闭环chunk实验

复用archive_protocol/single_eval的原生环境注册、状态恢复、EEF→pd_ee_pose动作转换。固定上述训练episode/seed105001600，恢复归档初始env_states；proprio与归档相差<1e-5，front首帧RGB像素MAE=0（该指标只检查front）。每轮重新采集当前front/wrist与EEF，经当前HTTP模型预测32步，执行前K步，再取新观测，直至原生success或300个仿真控制步。

动作在ManiSkill默认20Hz虚拟物理时间下执行；原生SimConfig.control_freq=20，PlaceSphere未改该值。等待HTTP时物理暂停，不模拟真机每点1.2s/推理期间持续物理/人工回车。无未来GT/物体标签进入模型；GT只用于独立专家桥接验收和分析。模型请求exp_id为offline_chunk_rollout_20261104。

| 每轮前缀 | 成功/步数 | 曾抓住 | 推理次数 | 最小TCP→球距离 mm |
|---|---|---|---:|---:|
| 专家GT | 成功，200步 | 是 | 0 | 2.19 |
| 1 | 300步内未完成 | 否 | 300 | 64.48 |
| 4 | 300步内未完成 | 否 | 75 | 57.38 |
| 8 | 300步内未完成 | 否 | 38 | 56.11 |
| 16 | 成功，228步 | 是 | 15 | 4.71 |
| 32 | 成功，220步 | 是 | 7 | 4.70 |

两份仿真日志无Traceback、OIDN/ErrorDeviceLost/非法CUDA内存标记；进程均exit0。专家200步成功先验证控制桥。由此可见，至少这个scene里，执行短前缀本身会影响闭环成败；不能把该scene失败只归为sim-to-real域差。16/32是待扩样候选，不是已经确认的现场最优参数。1/4/8只是300步诊断窗口内未完成，未运行原600步正式预算，不能报告全任务成功率或断言永远失败。

完整32步在训练20Hz下覆盖1.6s，但参考真机代码32×sleep1.2s至少38.4s，尚未算IK/人工交互。直接把CHUNK_STEPS改32不等于本实验条件，也不能保证改善真机。应分离前缀长度、逐点时序、夹爪事件和推理停顿，而非一次同时修改。

## 4. 固定实拍图像，替换输入位姿的离线敏感性

固定训练原文指令，同一real0图像，仅替换EEF位置/旋转为上述训练frame0。比较的是预测相对各自输入状态的位移。

| 替换状态部分 | 首步相对位移XYZ mm | 32步相对轨迹对原状态的RMS差 mm |
|---|---|---:|
| 不替换 | [4.86, -5.01, -19.45] | 0.00 |
| 只替换XYZ | [-0.15, -8.01, -4.92] | 19.39 |
| 只替换旋转 | [-3.26, -5.01, -13.61] | 33.65 |
| XYZ和旋转同时替换 | [-0.15, -2.78, 6.76] | 32.19 |

同样的2×4对照也做了sim图像，结果在state_sensitivity.json及8份npy。输入位姿对前下探有显著影响；但替换状态同时保持原图，构造的是物理不一致的条件，只能证明条件敏感度，不能证明真实复位后会抓对。真实复位后wrist图像也会改变。请求exp_id=offline_pose_factorial_20261104，未发送动作给机器人。

## 5. action转换与参考工程核查

### 数值链路

参考infer_openwam.py当前将action解释为绝对TCP，而非delta；3D旋转使用前两列，经Gram-Schmidt还原SO(3)，转给ROS的四元数是xyzw。上报和执行的代数链应为：

```text
T_model_tcp = T_model_realbase @ T_realbase_link8 @ T_link8_tcp
T_realbase_link8_target = inverse(T_model_realbase) @ T_model_tcp_pred @ inverse(T_link8_tcp)
```

参考URDF计算出的link8→TCP确为Rz(-45°)、z=.1054m，与适配器常数一致。3008个已有真实action的目标转换往返maxabs约2.5e-16。用归档203帧关节角在参考FR3与仿真做独立FK比较，位置差约2mm、旋转最大约.00006°。这些结果未发现6D列/行、四元数顺序、绝对/delta或105.4mm工具偏移方向的基础错误；不证明真实硬件标定正确。

### 必须核对的参考配置

参考FkArm加载robot_arm_only.urdf，根base_link与fr3_link0重合；参考部分MoveIt launch却加载整机robot.urdf.xacro，base_link→fr3_link0为[.212713696,-.002069014,.776600015]m。客户端IK的PoseStamped.header.frame_id同样写base_link。如果现场组合与此一致，数值来自arm-base却被当成mobile-base，坐标会错。用户已明确此工程不是现场副本，因此只能记为高优先级条件问题，不能认定真实机器人已有约.8m误差。需以现场IK服务实际robot_description/TF及同q的MoveIt FK验证。

参考IK失败会回退q_seed；适配器只在XYZ位移>5mm且关节未变时捕获此情况，原位旋转可能漏检；到达后仅检查位置≤1cm，不检查姿态误差。参考IK avoid_collisions=False。另有参考驱动closeGripper使用move(0,.1)、参考图像对未核验时间戳、夹爪命令在运动等待后发布，以及command_estimate与训练连续夹爪状态不一致。均已列入统一清单，尚未改代码或宣称为现场故障。

## 6. 错误变换如何确定

按下列证据层级验收，软件往返不能替代物理测量：

1. 固定现场代码/URDF/参数版本，确认FK根帧、IK请求帧、模型base、工具点名称。
2. 同一q下比较客户端FK与现场MoveIt FK完整4×4目标；用TF确认base_link和fr3_link0关系。两者不同先解决frame定义。
3. 用独立已知位置/测量验证真实TCP；固定TCP改变姿态检查实际抓取中心是否漂移，核验真实工具偏移。
4. 在确认后的低风险范围分别验证x/y/z方向和尺度、原位姿态变化，并记录IK error_code、目标/实际位置与旋转误差。不能只看由同一URDF算出的反馈位置。
5. 对连续前缀单独验收逐点IK、FK误差、关节速度/加速度、碰撞与实际完成点数；夹爪指令/宽度反馈与动作时间对齐。

## 7. 后续工作顺序

先按20项清单的P0收集现场真实配置/时序并完成坐标验收。真机暂停期间，可继续在多个train/held-out scene按相同20Hz、600步预算比较8/16/32，隔离当前无cache与历史cache设置差异，观察接近、闭合、持物、释放四阶段。历史1600份汇总显示place_sphere为Clean36/100、HardRandom41/100（旧FFN/cache/replan8配置），提示该任务本来不够稳定，不能把别的任务总体SR或单窗口低误差当成该任务能力保证。

现场恢复时先使用核对后的训练初始姿态，并复采真实观测，再做固定输入和预测TCP位置检查。相机难大幅调整时，优先确认camera→TCP几何、图像方向/FOV/裁剪和起始姿态对wrist观测的影响，避免同时盲调多个参数。之后再决定连续前缀执行方案、反馈改进或真实数据适配；本轮不改模型和现场控制。

## 8. 证据索引

目录：outputs/analysis/maniskill20872_pose_chunk_audit_20261104/

- scan.json、all_initial_eef10.npy：全量状态/初始状态与已存真实chunk统计。
- task_range_normalization.json、initial_orientation.json：任务全部状态范围、归一化与朝向差异。
- reference_reset_pose.json：真实训练初始EEF与源episode。
- geometry_audit.json、reference_frame_configuration.json：参考URDF/FK/帧定义核查。
- chunk_rollout_results.json、chunk16_results.json、rollout_*_trace.json、rollout_*_final.png：闭环逐步记录与结果。
- state_sensitivity.json、real_*.npy、sim_*.npy：2×4固定图像/位姿敏感性对照。
- issue_checklist.json：与Markdown对应的20项结构化问题。
- experiment_audit.json：模型身份、源码SHA、实验范围与日志检查。
