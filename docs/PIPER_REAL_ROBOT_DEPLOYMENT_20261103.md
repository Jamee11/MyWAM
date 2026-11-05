# Piper 双臂 OpenWAM 部署记录

## 2026-11-03：输入输出与坐标系只读核对

目标 checkpoint：`outputs/piper_sidecar/midtrain_patchwam_2epochs_20261102T101245Z/runs/2026-11-02_10-18-00/checkpoint_step_12000.safetensors`。
真机连接：用户现有 tmux `piper-62:0.0`，`ubuntu@114.214.215.88`。
原 `/home/ubuntu/kai0_repo` 仅查看文件、导入检查；没有修改代码，没有启动 ROS/CAN 控制或发布动作。

### 已确认

- checkpoint 契约：双臂 EEF20，左10维后右10维；xyz（米）、旋转矩阵前两列、夹爪0闭1开。内部槽位0–9/34–43、80维；当前两臂均有效；完整预测32步。
- archive reader按 head/left/right 三视角拼接384×320画布；不是OpenPI的224×224单视图预处理。
- `scripts/prepare_multisim_sidecar_data.py::robotwin_numeric` 直接使用HDF5 `endpose/{left,right}_endpose`，没有转回各自基座系；action为下一帧实际到达的EEF，而非原始关节命令。
- 原采集源码 `/remote-home/jinminghao/DiT4DiT/.external/RoboTwin/envs/_base_task.py::get_arm_pose` 调用 `get_left/right_ee_pose`；`envs/robot/robot.py::_trans_endpose` 使用 `ee.global_pose`，无逆基座变换。这是仿真共享世界坐标系的绝对位姿，不是每只机械臂的局部base坐标。
- Piper配置的`gripper_bias=.12`与`is_endpose=False`中的减`.12`抵消，位置为link6世界位置；global_trans_matrix与delta_matrix相乘为单位阵。与真机控制器反馈工具点是否相同仍待验证，不能自行补12cm。
- 从当前训练manifest中第一条Piper train轨迹实读首帧：左xyz `[-0.30141637,-0.39504284,0.95340055]`，右xyz `[0.29858366,-0.39504284,0.95340055]`。原始HDF5为`DiT4DiT/datasets/robotwin2_ea_rsig_v3_wrist020_20k_20261001/raw/piper/EA_RSIG_v3/Clean/beat_block_hammer/ea_rsig_piper_clean/data/episode0.hdf5`。
- 真机ROS1驱动`PublishSlaveArmEndPose`将控制器EEF反馈从微米/毫度转换为米/四元数，发布`/puppet/end_pose_left`和`/puppet/end_pose_right`，类型`PoseStamped`；夹爪需另从JointState读取。
- ROS1 `source /opt/ros/noetic/setup.bash`及原工作空间devel后，`kai0_inference0/bin/python`导入rospy/cv2/numpy/requests/scipy/PosCmd通过；原驱动的`PosCmd`经IK下发关节控制。尚未验收实际IK有效性或碰撞。

### 结论和未决项

不能将真机局部base位姿原样当作模型共享世界坐标位姿。需要输入与输出互逆的刚体转换。
但这不等于必须重新做手眼标定：可先从仿真安装位姿和一致的工具点定义解析构造转换，再用已知关节姿态验证。
若真机机械结构/工具点/双臂安装与仿真不同，再确定需要补测哪些参数；不以identity或相似外观代替验证。

服务/客户端正在新增，尚未完成测试与远端client_repo交付。当前GPU0–7均有大量占用，未启动模型加载或停止其他任务。
下一步：实现与测试三视角/归一化/动作契约，复制独立client_repo，验证转换往返与HTTP通信；现场已知姿态对齐后才由用户启动运动验证。

## 2026-11-03 14:00 UTC：独立框架和非硬件通信验收

已新增`benchmarks/real_robot/piper/serve_piper_flask.py`与`run_piper_flask.sh`，复用现有loader/engine，不改模型、训练或共享部署核心。
客户端源码保存在`benchmarks/real_robot/piper/client_repo`，已复制到远端`/home/ubuntu/client_repo`并独立初始化git。
复制保留Kai0原客户端/ROS1驱动参考；运行时只复用原环境和已编译消息，未安装依赖、未改原代码。

Flask服务显式三视图384×320、原20→80统计/两臂mask、完整32步动作；保留请求和响应。
客户端读取实际驱动PoseStamped+夹爪反馈，做双向坐标转换，复用Kai0时间平滑；默认dry-run无控制publisher，执行需实际几何确认和工作范围、基本位移/转角阈值。没有额外多层确认开关，不自动归零/使能。

验证：`tests/benchmarks/test_piper_real_robot.py`6项＋`test_real_robot_http.py`7项均通过；shell语法与server/client help通过。
远端原ROS1环境真实运行`client_repo/tests/smoke_ros.py`两次通过，每次2轮HTTP请求，私有master11411、模拟三图/双EEF反馈、0控制publisher，退出后释放测试进程。
详细证据和源码路径见新repo的`VALIDATION.md`和README。
原仓库diff内容SHA前后均为`a59aeba9347a00d9ec6c0cc54304e02eba822a7c2961c9496f209b9a613a9819`；参考客户端SHA前后均为`258ac089b6f3a40256e0355d24e1838dc5f9136fe1e482550470227d6820c5c9`。

未完成门槛：实际checkpoint GPU生成与真实网络/现场几何/控制验收。GPU0–7现有任务均保留，当前剩余显存不足35GiB；未加载模型、不影响原训练或现有Kai0部署。
