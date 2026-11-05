# OpenWAM 双臂 Piper 客户端（独立于 kai0_repo）

服务端在GPU机器的OpenWAM仓库运行；此仓库放机器人机器`/home/ubuntu/client_repo`。
不改`/home/ubuntu/kai0_repo`的代码或环境。ROS驱动/三相机继续沿用原启动方式。

## 1. 输入输出

| 项目 | 本实现 |
|---|---|
| 状态 | 左臂10维＋右臂10维：xyz（米）、旋转矩阵前两列、夹爪0闭1开 |
| 观测 | `/puppet/end_pose_left/right`和`/puppet/joint_left/right`；三路RGB图像 |
| 图像 | 原尺寸PNG传输；server按训练逻辑拼384×320，front在上、left/right腕部在下 |
| 动作 | 一次返回32×20的绝对EEF，不是14维关节，也不是50步 |
| 控制 | 转回各自真机基座系，发布原`/control/end_pose_left/right`的`PosCmd`，沿用驱动IK |
| 平滑 | 从用户给定Kai0脚本复制`StreamActionBuffer`；对EEF20插值，执行前正交化Rot6D |

不自动reset、不自动enable。只推理模式没有控制publisher。任何推理/执行错误停止，不重放旧动作。

## 2. 启动模型服务（GPU服务器）

等待一张GPU至少空闲35GiB；不要挤占正在训练的GPU。进入OpenWAM，使用现有模型运行环境，
准备16字符以上私有token文件（例如复用现有Flask部署的token文件），然后：

```bash
cd /remote-home/jinminghao/OpenWAM
GPU_ID=7 TOKEN_FILE=/你的私有路径/api_token \
  bash benchmarks/real_robot/piper/run_piper_flask.sh
```

默认checkpoint为Piper12000，监听`0.0.0.0:5902`。可用`CKPT`/`PORT`/`PYTHON_BIN`覆盖。
`GET /health`和`POST /api/inference`都使用`Authorization: Bearer <token>`。
每个请求的三幅PNG、原始state/prompt和完整预测记录在
`outputs/real_robot/piper12000/inference_records/<id>/`。调用engine返回完整chunk，避免逐动作重复HTTP。

公网访问需要把公网端口映射到服务端5902；仅监听0.0.0.0不等于公网映射已建立。
若没有现成映射，可在机器人端使用原有SSH隧道方式：

```bash
ssh -p 35601 -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18000:127.0.0.1:5902 root@210.45.70.21
```

这是示例跳板端口，使用当前能登录GPU服务器的实际SSH端口。隧道加密；不要通过公网明文HTTP传token/画面。

## 3. 启动机器人客户端

在原来的终端按旧流程启动三相机、Piper驱动；本仓库不替你使能或控制硬件。
本实现复用`kai0_inference0`环境和现有编译的ROS消息，不安装包、不修改环境。

```bash
cd ~/client_repo
source scripts/activate.sh
export OPENWAM_API_TOKEN='与服务器token文件内容相同的值'
python inference/agilex_inference_openwam.py \
  --host 127.0.0.1 --port 18000 --prompt 'stack the blocks' --steps 2
```

这一步只推理。若直接访问映射端口，把host/port换成对应公网地址和端口，不要使用旧Kai0服务端口。
HTTP会绕过终端的Clash代理。

## 4. 坐标核对后执行

`robot.json`集中放话题、两臂`model_from_base`、`feedback_from_tool`、夹爪最大开度和运动阈值。
默认基座矩阵来自仿真Piper安装：左右x=−0.3/+0.3、y=−0.45、z=.75、绕z旋转90°；
这是一套**模型虚拟坐标**，不是声称测得真机的外参。工具矩阵暂为identity。
训练位置是RoboTwin link6参考点，不要自行加12cm TCP偏移。

在已知关节姿态对比控制器反馈与仿真/URDF末端定义，确认上述工具点/轴向可对应；如果不同，只改此处矩阵。
当前真机URDF与仿真URDF不完全相同，不能凭identity就视为通过现场验证。
填入每臂实测安全工作范围`workspace_xyz=[[xmin,ymin,zmin],[xmax,ymax,zmax]]`（各自真机基座系），
确认后将`geometry_verified`改为true。默认单步位移2cm、转角.15rad；超过阈值拒绝，不偷偷裁剪坐标。
这几个字段是唯一的现场执行门槛，不另设多层确认/审核流程。

```bash
python inference/agilex_inference_openwam.py \
  --host 127.0.0.1 --port 18000 --prompt 'stack the blocks' \
  --execute --chunk_size 4 --steps 1 --publish_rate 30
```

第一次只执行4步；现场验证后可以用更长prefix和时间平滑：

```bash
python inference/agilex_inference_openwam.py \
  --host 127.0.0.1 --port 18000 --prompt 'stack the blocks' \
  --execute --use_temporal_smoothing --publish_rate 30 --inference_rate 2 \
  --latency_k 0 --min_smooth_steps 12 --chunk_size 32
```

`chunk_size`是每轮取出的预测前缀（1..32），不是模型生成长度；`steps`是推理轮数，0为持续运行。
`latency_k`保留Kai0语义，不直接沿用50步模型的24步裁剪，默认0以免剪掉大部分32步预测。
30Hz是现有客户端控制频率；训练archive未声明固定control_hz，不声称已做训练时间轴匹配。
没有环境碰撞规划；现场急停、双臂间距/工作台与驱动IK安全仍需实际确认。

## 5. 不碰硬件的验证

```bash
cd ~/client_repo
source scripts/activate.sh
python tests/smoke_ros.py
```

此测试另起`127.0.0.1:11411`私有ROS master，模拟三相机/双臂反馈和HTTP hold策略；
运行真实client两次请求，检查控制publisher为0。不是实际checkpoint或真机成功率验证。
ROS master/测试HTTP服务退出后自动关闭，日志在`logs/smoke_ros/`。

`reference/`保留原始Kai0客户端、ROS1驱动和真机URDF的副本以便核对；不在新客户端中导入或启动它们。
