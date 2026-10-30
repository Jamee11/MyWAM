# ManiSkill Sidecar / baseline：Flask server + RobotWorkspace client

## 2026-10-27：GPU4 Sidecar33880 带输入输出留存的当前部署

当前服务 PID `4066208`，`CUDA_VISIBLE_DEVICES=4`，监听 `0.0.0.0:5901`；
公网 `http://210.45.70.21:29928/health` 未认证返回401，携带既有token后
返回同一PID、`sidecar`、`checkpoint_step_33880.safetensors`。此前下文的
GPU7/127.0.0.1/公网502描述仅是2026-10-24的历史快照，不代表当前状态。
未停止或修改其他GPU作业；本服务就绪后GPU4总显存占用约44.1GiB，
其中启动前已有约19.1GiB占用。未连接或驱动真机。

当前服务从 `benchmarks/real_robot/serve_maniskill_flask.py` 启动，使用
`/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv/bin/python`，
checkpoint为 `outputs/maniskill_sidecar/joint_20261019T222758Z/2026-10-19_22-33-43/checkpoint_step_33880.safetensors`，
token复用私有文件 `outputs/real_robot/maniskill_sidecar33880_20261024/api_token`。
启动参数为 `--policy sidecar --host 0.0.0.0 --port 5901`，
manifest和日志分别在 `outputs/real_robot/maniskill_sidecar33880_gpu4_20261027/server_manifest.json`
与同目录 `server.log`。

每次通过校验的推理请求自动留存在该目录的 `inference_records/<UTC时间>_<随机ID>/`：
`request.json` 为server收到的原始任务/EEF10请求，`front.*` 和 `wrist.*`
为原始上传图像字节，`response.json` 为返回的完整32步action及元信息；
不保存Authorization/token。一次原生ManiSkill样本经认证HTTP推理返回有限的
32×10动作，约2.55秒；已逐字节核对图像、请求和响应落盘与实际发送/返回一致。
记录目录权限700；日志将持续增长，后续需监控磁盘空间。该验证不证明
真机画面、运动或任务成功率。公网为明文HTTP，访问凭证和图像有传输暴露风险，
仅在受控网络下调试；尚未从真机网络独立验收。

## 用户要求的简化版：推荐使用单文件infer_openwam.py

用户希望沿用原infer.py，不再额外传token/config/几何模块路径，因此新增薄适配：
`benchmarks/real_robot/infer_openwam.py`。它继承真机现有infer.Infer，复用ROS话题、
FkArm、IK client与publisher，不覆盖原文件；不再调用原相机delta apply()或自动reset。
后文多文件client保留为较早方案/验证记录，真机优先用本节单文件。

可直接复制的私有版本：`outputs/real_robot/maniskill_sidecar33880_20261024/infer_openwam.py`。
其中已嵌入当前服务访问凭证，权限600，只交给自己的真机，禁止公开上传或提交git；
项目源码模板不含凭证，读取OPENWAM_API_TOKEN环境变量。凭证没有出现在诊断输出。

把这一个文件放到真机原infer.py、convert.py旁边，在原ROS环境执行：

```bash
python3 infer_openwam.py
```

按提示输入任务描述、当前夹爪开/关状态、推理轮数。不需要--token-file、--config、
--robot-scripts或额外几何文件。夹爪输入明确标记command_estimate，不伪称实测反馈。

所有参数集中脚本顶部：API_URL、TASK、EXECUTE、两个坐标/TCP变换、工作空间与步长限制。
默认EXECUTE=False，不下发运动；原Infer构造仍创建原有ROS publisher并等待既有IK服务，
因此它不是无需RobotWorkspace的独立离线client。真正运动前仍需确认base/TCP变换并填写
工作空间、CALIBRATION_CONFIRMED，最后显式启用EXECUTE并交互确认；不能靠identity占位跳过标定。
前述硬件急停/连续碰撞/实体验证边界不变。

动作逻辑只保留两个必要变化：发送当前模型坐标系EEF10状态；将预测的绝对TCP转换为
真实link8目标直接求IK，不做delta累加，不套相机系cam2base。不新增控制框架。
IK失败无旧解回退，整prefix验证后才下发，逐步验证反馈/七轴限位/跳变/速度与观测时效。
单文件新增3项测试+既有6项HTTP测试共9通过；仅CPU/mock验证，无实体运动。

公网只读检查：`http://210.45.70.21:29928/health`返回502，独特probe未进入本服务日志，
说明当前公网入口尚未转发到本Sidecar服务。本轮没有猜测映射或修改已有监听端口；
当前仍是127.0.0.1:5901。需确认公网29928的后端端口/转发规则后才能从该地址连接。

## 范围与当前状态

### 2026-10-24 08:57 UTC：当前验收结果

- Sidecar33880已在GPU7运行，PID3723989，HTTP `127.0.0.1:5901`。
- 权重SHA256：dc1645504f06e27983080b4b9181df8dd73ce2b3fcd871d6711c91f5c046b3c1。
- 两条原生ManiSkill PushCube/PickCube观测经认证HTTP真实推理通过，输出32×10且有限；
  第一条重复逐位一致，未认证请求401。首次请求约2.82秒，后一任务约.80秒（非吞吐基准）。
- `outputs/real_robot/maniskill_sidecar33880_20261024/http_smoke.json`及两个smoke_response保存证据；
  私密token未写入这些记录。token权限600，Flask依赖独立target安装并记录版本。
- 6项CPU测试通过，client无ROS环境也可运行`--help`；Python语法与shell语法检查通过。
- GPU7总占用约65.9GiB中的原有约40GiB作业保留；本进程约24GiB，非全生命周期峰值保证。
  0–3训练3501138继续，GPU4/5/6及其他进程未操作；baseline没有加载。
- 本轮只证明模型加载、绝对EEF10 HTTP协议与推理链路，**没有ROS硬件/实际IK/实体运动验收**。

按用户最新要求：**当前只部署Sidecar33880到GPU7，baseline21180仅保留启动选项，不加载。**
不改模型、训练、原评估或其他项目；Flask只是现有loader/preprocessor/engine外的HTTP适配。
真机不在当前服务器，本轮不会连接或驱动机械臂。通信成功不是硬件验收或sim-to-real成功。

文件：

- `benchmarks/real_robot/serve_maniskill_flask.py`：Flask服务，认证、单请求锁、完整32步chunk。
- `benchmarks/real_robot/run_maniskill_flask.sh`：Sidecar/baseline共用入口，默认GPU7。
- `benchmarks/real_robot/openwam_robot_client.py`：独立ROS2 client，默认dry-run，无自动reset。
- `benchmarks/real_robot/openwam_real_contract.py`：独立NumPy几何/安全检查，随client一起复制。
- `benchmarks/real_robot/robot.example.json`：真机配置模板，未确认标定时禁止运动。

Sidecar：`outputs/maniskill_sidecar/joint_20261019T222758Z/2026-10-19_22-33-43/checkpoint_step_33880.safetensors`。
baseline：`outputs/maniskill_baseline/six_task_bs64_10epochs_20261017T050252Z/runs/2026-10-17_05-16-28/checkpoint_step_21180.safetensors`。
两者均从**指定文件**加载config和normalization_stats，不选择目录里最新checkpoint。
模型输出由原engine反归一化，HTTP层不能再做一次旧UniVLA的7维delta反归一化。

## 输入输出契约

| 项目 | 契约 |
| --- | --- |
| RGB | 当前front与wrist；原训练预处理组成384×320 L形画布，右下黑填充 |
| 文本 | 原始英文任务描述；server用训练模板包一遍 |
| state | 当前实际TCP的10维：xyz米 + 旋转矩阵第一列3维 + 第二列3维 + 开爪比例 |
| action | 32×10，同上表示，**模型机器人基座系下的绝对TCP目标**，不是delta/RPY/相机系 |
| gripper | 0闭、1开；输出轻微越界只在client夹爪执行边界裁剪，不裁剪位置 |
| 模型内部80维 | 由checkpoint原normalizer/unify映射处理；client只发10维，不手填80维 |
| 推理 | 10步、CFG1、sync、eager、DiT cache关闭，无GT物体/EA标签输入 |

每个HTTP请求均以最新观测独立生成完整chunk，不保留server动作队列。client默认仅执行前1步，
可显式选1–8步；不会把剩余chunk在之后重放。两模型的Flask传输/预处理相同，但不能将此前
其他checkpoint的仿真SR当成本33880的真机成功率。

## 服务器启动

当前新增Flask依赖安装在`outputs/real_robot/http_deps`，不修改训练进程所用环境。
复建时在OpenWAM已有模型环境执行：

```bash
cd /remote-home/jinminghao/OpenWAM
PYTHON_BIN=/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv/bin/python
"$PYTHON_BIN" -m pip install --target outputs/real_robot/http_deps \
  -r benchmarks/real_robot/requirements-server.txt

MODEL=sidecar GPU_ID=7 PORT=5901 \
TOKEN_FILE=/absolute/path/to/private_api_token \
MANIFEST=outputs/real_robot/sidecar_http_manifest.json \
PYTHON_BIN="$PYTHON_BIN" \
bash benchmarks/real_robot/run_maniskill_flask.sh
```

token文件应为权限600、至少16字符的随机秘密，不放进git、命令行参数或公开日志。
当前token位于`outputs/real_robot/maniskill_sidecar33880_20261024/api_token`；经安全渠道单独传到真机。
当前启动日志/manifest在同目录，初始PID3723989，端口127.0.0.1:5901；以health和后续验收记录为准。
服务需要启动前至少35GiB空闲，CPU加载后再次检查；本进程PyTorch allocator上限34GiB，
为GPU7已有作业保留余量。不会结束其他GPU进程。HF强制离线并保留mirror配置，不下载模型。

默认只监听本机。真机上使用SSH隧道（服务器地址/SSH端口按实际填写）：

```bash
ssh -N -L 5901:127.0.0.1:5901 YOUR_A100_SSH_HOST
```

若确需直接内网访问，server设`HOST=0.0.0.0`并限制防火墙来源；不要把明文HTTP/token暴露公网。
Flask开发服务只用于受控单客户端实验；不启reloader、多worker或并发模型实例。

## 真机准备与标定：不能照抄旧delta client

将以下三个文件复制到真机，建议与原`infer.py`/`convert.py`放在同一scripts目录；不覆盖旧infer.py：

```text
openwam_robot_client.py
openwam_real_contract.py
robot.example.json -> 复制为 robot.openwam.json 后编辑
```

client使用现有ROS2 Humble/RobotWorkspace环境及已有`convert.FkArm`、NumPy、SciPy、requests、urchin。
参考接口保持：`/mk1000/joint_states`、两路CompressedImage、`/pc_arm/compute_ik`、
`/mk1000/fr3_arm_controller/joint_trajectory`、`/gripper_control_signal`。
关节按name取出七个值，单位由配置显式声明，默认ROS标准rad；不按消息前七项猜测。

必须确认：

1. `T_model_from_real_base`：模型基座←真机基座的刚体变换，平移单位米。
2. `T_link_from_tcp`：真机link8←策略TCP的固定工具变换；模型训练控制点为`panda_hand_tcp`，
   不可直接假设等于link8。模板identity只是占位，**不是已完成标定**。
3. `workspace_xyz`：真机基座系下TCP的实测允许范围，`[[xmin,ymin,zmin],[xmax,ymax,zmax]]`。
4. `gripper_feedback_topic`：真实反馈Float32，设置闭/开数值以归一化到0/1。
   原参考client只有命令记忆值，不能冒充实测。如硬件确无反馈，必须显式设置
   `allow_estimated_gripper=true`并确认`initial_gripper_open_fraction`；请求/日志标为command_estimate。

确认两个变换、单位、工具点与边界后，才将`calibration_confirmed`设为true。公式为：

```text
state TCP = T_model_from_real_base @ FK(real joints, link8) @ T_link_from_tcp
target link8 = inverse(T_model_from_real_base) @ predicted_absolute_TCP @ inverse(T_link_from_tcp)
```

不把输出加到当前位置，不套旧cam2base/相机delta旋转逻辑；相机外参不能直接充当模型基座映射。
没有确定这些参数时可做通信dry-run，但其预测值不能当作已对齐真机动作质量。

## 先dry-run，再单步运动

真机启动相机/关节驱动，source已有ROS2与RobotWorkspace环境，开好SSH隧道后：

```bash
python3 openwam_robot_client.py \
  --url http://127.0.0.1:5901 --token-file ./openwam_api_token \
  --config ./robot.openwam.json --robot-scripts "$WORKDIR/src/application/robot_bringup/scripts" \
  --task 'pick up the red cube and move it to the elevated goal' --rounds 1
```

默认**无运动publisher、无IK调用、无自动复位**，只读取状态、推理并写`openwam_predictions.jsonl`。
检查返回32×10、基座/工具点、目标位置、旋转、夹爪方向和延迟；保留人工急停。
全部确认后，同命令额外加：

```bash
--execute --execute-steps 1
```

还需交互输入`EXECUTE`。每轮确认在采集新观测前进行，避免等待人工确认后执行过期图像。
默认TCP步长≤2cm、旋转≤.15rad、七轴步长≤.08rad且速度≤.3rad/s；工作空间与URDF限位同时检查。
先验证所选整个prefix及所有IK，任一步失败不发布；IK启用collision检查，不回退旧解。
推理/IK超5秒、图像/关节过期、双相机不同步、真机状态漂移或关节跟踪超时都会停止后续命令。
每步用反馈确认到达，夹爪在该步到达后下发二值0/1；夹爪本身到位尚需硬件反馈/人工确认。
Ctrl+C只停止后续软件命令，**不是硬件急停或控制器撤销接口**；操作员必须能用硬件急停。
IK的collision检查也不等于JointTrajectory整段连续碰撞规划；必须有正确的碰撞场景、
经过验证的底层安全控制和现场监督，不应把本client当作经认证的安全控制器。

模型仿真数据约20Hz；client初次硬件验证默认.2秒/步、只执行1步并重规划，这是保守慢速探测，
并非已证明与仿真20Hz等价的控制协议。接触动力学、视角、背景、工具和机器人差异都可能影响效果。

## baseline后续切换（当前不部署）

在确认Sidecar服务退出并释放GPU后，以同一入口`MODEL=baseline`启动，client加
`--expected-policy baseline`；默认使用用户指定21180文件。不要在当前GPU7余量下同时加载两份完整模型。
本轮不宣称baseline21180已进行真实GPU推理验收。

## HTTP协议与验收证据

`GET /health`与`POST /api/inference`均需`Authorization: Bearer TOKEN`。
POST沿用multipart字段`main_images`、`wrist_images`、`json`，但json必须包含
`task_description/state/exp_id/request_id/schema/gripper_state_source`；旧delta client请求会被拒绝。
响应为带contract/checkpoint SHA/request_id的对象，`actions`是32×10，而不是旧版裸7维delta数组。
错误/忙时不返回动作，client不自动重试POST、不复用旧响应。

CPU测试：`tests/benchmarks/test_real_robot_http.py`覆盖刚体/工具偏置、绝对而非delta、七关节限幅、
认证/输入契约、busy拒绝排队、失败不返回动作、标定门槛/NaN门槛、旧响应拒绝及整prefix IK失败零发布。
真实checkpoint的HTTP health/推理已通过，见文首smoke证据；硬件ROS/IK/碰撞场景/实体执行未在本服务器验收。
