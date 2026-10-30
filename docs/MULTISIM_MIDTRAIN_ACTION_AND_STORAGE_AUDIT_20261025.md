# 多仿真midtrain：绝对EEF语义、存储与评测split核查

时间：2026-10-25T05:01:08.543305+00:00。本轮只读核查与记录，未修改原训练/数据，未启动midtrain。

## 需要纠正的表述

现有OpenWAM Aloha训练本来就是绝对EEF。所谓“沿用旧Aloha方式”指现有reader，不能让人理解为旧delta版本。openwam/dataloader/robotwin.py明确proprio=raw_actions[0:1]、action=raw_actions[1:num_frames]；ea_rsig_lite.py的LiteRoboTwinSource读取原始endpose的绝对xyz+wxyz并转Rot6D。新v1的RoboTwin同样是绝对EEF，不是delta。

“absolute/relative”与“commanded/achieved”是两条独立语义轴：RoboTwin读取未来实测绝对位姿；RLBench/ManiSkill准备绝对控制目标。张量/损失接口兼容不等于控制目标语义统一。当前v1只通过其明确声明的混合reference契约和格式验收，不能称全部来源已统一成absolute commanded EEF。既有Aloha训练/ckpt不因此自动无效；不能为新midtrain静默改动既有baseline。

## 原始命令的恢复路径

已抽查Aloha/Piper HDF5有joint_action/left_arm、right_arm、vector，Piper另有joint_actual/vector。本项目官方RoboTwin robot.py的get_left/right_arm_jointState读取joint.get_drive_target，而get_arm_pose→get_left_ee_pose→_trans_endpose读取left/right_ee.global_pose。DiT4DiT采集manifest也明确joint_action是commanded target、joint_actual是measured qpos。因此可研究从真实drive-target经对应URDF/FK重建absolute commanded EEF；不能声称原始控制目标不存在，也不能把endpose简单改名当作command。

已有Aloha/Piper FK实现：DiT4DiT/examples/RoboTwin2/scripts/derive_robotwin_base_delta_ee.py、piper_kinematics.py。后者有单独Piper工具旋转与双臂base变换。实际转换前必须逐源核对机器人版本/基座/TCP/夹爪以及观测与target的时间配对；planner轨迹保存频率与图像采样频率也要核对。不得未经验证直接取同索引或下一索引。

## 数据存储

根目录openwam_sidecar_multisim_midtrain_v1_20261024中numeric是新生成，media是48,093个独立源HDF5的硬链接，逻辑载体合计约2.102 TiB。抽查Aloha/Piper为is_symlink=False、相同inode、nlink=2。硬链接可在原路径被删除后继续存在，但与源共享同一份内容，修改任一端会影响另一端，因此不是物理独立副本。新目录相对路径读取和迁移smoke通过，只证明逻辑自包含，不证明已有独立payload副本。若用户要求完全独立存储，应创建新版本并物理复制或提取规范化媒体，不采用软链接/硬链接替代独立复制。

## ManiSkill评测与split

已核对sidecar25410_six_task_100_replan8_max200_matched_20261022/run.json及benchmarks/maniskill/single_eval.py：六任务使用旧EORT color_distractors测试数据的saved_trajectory_seeds，seed从2000000起，在仿真中重新reset并闭环执行。push/pick来自maniskill_eort_color_distractors_v4_20260728，其他四任务来自maniskill_eort_task_suite_v5_20260730。不是读取专家test动作作为策略评测，也不是本次新20k数据的test目录。本机本轮未发现活跃ManiSkill评估client；若用户指远端新评测，需另外核对对应run，不能把已存六任务记录冒充实时进程。

Midtrain不必机械保留80/10/10，但必须先冻结独立的验证/闭环seed协议并排除任务/物体/物理轨迹家族重叠；不同目录和不同数字seed不自动证明独立。不能在未验证eval是否重合前把所有val/test并入训练后仍宣称原test泛化。建议至少保留小型物理家族级validation；其余数据可在明确实验协议后纳入midtrain。
# 2026-10-25 用户确认后的执行契约

本节优先于下文较早的候选建议：RoboTwin 保持 OpenWAM 当前训练约定的绝对 EEF，**不改成 delta EE，也不默认改成 FK 重建的 commanded EEF**。观察输入为 `endpose[t]`，动作标签为从 `endpose[t+1]` 开始的未来 32 个实测绝对位姿，末尾补齐位置的动作维度由 mask 排除。原始关节命令与实测 endpose 均保留，二者不混称。

已用当前真实部署的反归一化、EEF20→EEF16 转换与四元数重排链路，检查 Aloha/Piper × 普通/杂乱数据 × 首/中/末帧共 12 组：位置往返最大误差小于 8.5e-8 m，旋转小于 1.6e-7 rad，夹爪一致。证据：`docs/evidence/multisim_archive_v2_20261025/robotwin_absolute_eef_contract.json`。仅证明动作编码、坐标表示与训练/部署接口一致；不证明绝对目标执行后的实测轨迹逐点重合，也不替代仿真专家回放或策略成功率测试。

训练/测试须同时保持：raw wxyz 四元数正确转列式 Rot6D；复用训练统计反归一化；部署使用 `action_type: ee`；采用相同 endpose 工具参考，不能再叠加 EA 标签里的 0.12 m TCP 偏移。已有训练腕相机 near=0.02 与官方评估 near=0.1 的差异仍独立保留为未解决因素，不能由上述动作检查推导图像域一致。

独立归档目标：`/remote-home/jinminghao/OpenWAM/datasets/multisim_sidecar_archive_v2_20261025`。复制原生数据、回放代码/资产、原始元信息以及训练 numeric/statistics，训练媒体直接引用归档内 native 文件；不复制旧 media 硬链接作为独立副本。保留 ManiSkill/RLBench test 与 val，完整归档和训练选样分离。读入器支持 `selected_for_midtrain=false`：训练排除这些样本，显式 test/val 读取仍可访问；旧清单默认行为不变。

**当前状态：BUILDING。原数据不得清理。** 复制任务逐文件写后 SHA256 回读，但全部文件复制、运行时路径依赖闭包和原生回放验收尚未完成；此处不宣称归档已经自包含可回放。训练配额、现有单源训练与网络/损失保持不变。



## 2026-10-26T07:44:11.438213+00:00 归档完整性与旧目录清理范围复核

此节更新前文BUILDING状态：训练数据已于10月25日发布TRAINING_DATA_READY，52204选样、train39816/val6161/test6227。今天只读核对五份独立复制ledger：native A647575/B591296/追加pilot9400/replay61093/training104413条全部verified，复制时已逐文件SHA写后回读；当前样例为nlink1、非symlink且与源inode独立。未重新对全部TB数据做一次全量哈希，不把历史复制证明说成今天全量回放验收。

归档范围是source_roots.json明确的五个来源子树（RLB production、MS-base raw、MS-hard production、RT20k、RT-clutter4k），补充accepted样本引用的212历史pilot attempts及指定旧六任务test；不等于/remote-home/jinminghao/datasets、DiT4DiT/datasets的所有内容。native保留原始字段，training提供OpenWAM absolute EEF/130D标签/statistics，test/val保留。复制的replay代码/资产不等于已验证可脱离原目录独立执行；training_release中的native_replay_dependency_closure_verified仍False，仍不能保证所有重处理/原生回放无需旧环境/路径。

另已发现**当前GPU0–3 Aloha训练实际仍打开外部official_clean4350数据**，不是v2 manifest：/remote-home/jinminghao/datasets/robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015。本轮采样DataLoader fd确认读取其HDF5。它并非这五来源直接归档范围，当前不能清理；其它旧项目/评估路径也须逐目录核对引用，不能笼统称全部重复。

本次不删除任何数据、不改训练路径或归档内容。下一步若清理，应先完成归档内独立重处理/原生replay验收、逐目录覆盖和活跃依赖清单；通过后仅讨论明确重复的子目录。证据：outputs/archive_v2_20261025/cleanup_readiness_audit_20261026.json。
