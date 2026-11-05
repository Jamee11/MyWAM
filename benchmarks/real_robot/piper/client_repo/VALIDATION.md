# 验证状态（2026-11-03）

- 实际存放于`ubuntu@114.214.215.88:/home/ubuntu/client_repo`，已独立`git init`。
- 原`/home/ubuntu/kai0_repo`仅查看/复制；复制前后`git diff --binary`的SHA256均为
  `a59aeba9347a00d9ec6c0cc54304e02eba822a7c2961c9496f209b9a613a9819`。
- 原Kai0参考客户端及新repo中的副本SHA256均为
  `258ac089b6f3a40256e0355d24e1838dc5f9136fe1e482550470227d6820c5c9`。
- 使用原`kai0_inference0`环境与ROS Noetic消息，client `--help`通过。
- `python tests/smoke_ros.py`实际通过：私有ROS master、模拟三视图/双臂位姿与夹爪、真实HTTP客户端两轮请求、32×20动作响应，控制publisher为0；日志`logs/smoke_ros/client.log`。
- GPU服务器本地6项Piper测试与7项已有真机HTTP测试合计13项通过；覆盖坐标/工具变换往返、运动阈值、原时间平滑、HTTP认证/记录、三视图逐像素拼接一致性、实际checkpoint统计的20→80维归一化/缺省维mask一致性。
- **未验收**：真实Piper12000模型加载/生成、公网映射/SSH隧道、真实机器人反馈、工具点/坐标现场核对、控制执行及任务成功率。所有GPU当前空闲显存不足35GiB，未挤占现有训练。
- `robot.json`保持`geometry_verified=false`和未填workspace；软件模拟成功不意味着可以绕过现场几何验证直接执行。
- 额外只读URDF比较：仿真与真机URDF并非逐字相同；20个固定随机关节姿态的静态刚体拟合后FK矩阵元素最大残差约.0041，不能据此声称完全等价或已完成真实控制器工具点验证。未将拟合值自动写入配置。
