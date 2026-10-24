# 真机client文件包

文件包包含访问凭证，请仅通过可信渠道传到自己的真机，不公开上传，不提交git。

- openwam_robot_client.py：真机client。
- openwam_real_contract.py：配套几何和安全检查。
- openwam_api_token：已经生成的服务器访问凭证，不要随意改写。
- robot.openwam.json：配置模板，仍未标定，不是可以直接执行运动的真机参数。

把包内文件放到原RobotWorkspace的scripts目录，或使用--robot-scripts指向其中convert.py的位置。
在原ROS2/RobotWorkspace环境执行python3 openwam_robot_client.py --help可查看参数。

注意：

1. 默认不会发布运动指令，不要先加--execute。
2. 先配置真实夹爪反馈；如果硬件没有反馈，需要明确允许估计值并确认当前开合状态，不能伪称实测。
3. 模型基座到真机基座、link8到TCP的两个identity矩阵都是占位；工作空间也尚未填写。
4. 完成标定前不能将calibration_confirmed改为true来绕过检查。
5. 公网210.45.70.21:29928到服务器内网端口的映射仍需确认；打包不改变当前服务监听。
6. Flask服务只提供推理；HTTP成功不证明硬件执行或sim-to-real成功。
