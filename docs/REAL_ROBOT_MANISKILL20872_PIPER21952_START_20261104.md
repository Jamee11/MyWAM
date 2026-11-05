# 真机部署启动：单臂 ManiSkill20872 / 双臂 Piper21952

本文固定使用用户指定的两份新 Sidecar 权重，不使用旧33880/12000。仅提供启动流程，不自动启动模型或机械臂。

| 场景 | checkpoint | 服务端口 | 机器人客户端 |
|---|---|---|---|
| 单臂 ManiSkill | `outputs/maniskill_sidecar/midtrain_patchwam_1epoch_20261031T184131Z/runs/2026-10-31_18-46-16/checkpoint_step_20872.safetensors` | 5901 | 已调通的 OpenWAM `infer_openwam.py` |
| 双臂 Piper | `outputs/piper_sidecar/midtrain_patchwam_2epochs_20261102T101245Z/runs/2026-11-02_10-18-00/checkpoint_step_21952.safetensors` | 5902 | `/home/ubuntu/client_repo/inference/agilex_inference_openwam.py` |

**机器人 client 不需要因更换这两份权重再改输入输出格式。**
单臂仍是front+wrist、绝对EEF10；双臂仍是三相机、绝对EEF20（左10+右10）。
client只需指向对应服务地址、使用同一token，保留已核对的坐标/工具点参数。
双臂必须使用新`client_repo`，不能使用旧kai0的14维关节动作client。原`kai0_repo`不改动。

## 1. GPU服务器：公共准备

下面每个模型服务使用一个终端。GPU编号只是例子，先用`nvidia-smi`选一张至少空闲35GiB的卡；不要结束其他训练。
如同时部署两份模型，应分别选择有余量的GPU。

```bash
cd /remote-home/jinminghao/OpenWAM
export PYTHON_BIN=/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv/bin/python
export PYTHONPATH="$PWD/outputs/real_robot/http_deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKEN_FILE="$PWD/outputs/real_robot/deploy_20261104/api_token"
mkdir -p "$(dirname "$TOKEN_FILE")"
# 首次生成；已有文件则保留，两个服务可共用这个token。不要把真实token提交到Git。
if [ ! -s "$TOKEN_FILE" ]; then
  (umask 077; openssl rand -hex 32 > "$TOKEN_FILE")
fi
```

使用现有环境，不要求重装Flask。token文件不会打印；机器人端需要私下复制同一个文件，命名为`~/openwam_api_token`。

## 2. 单臂 ManiSkill20872：服务端

### 2.1 首次准备单臂部署副本（只做一次）

新checkpoint的训练配置沿用20维双臂容器，但训练样本只有左臂有效。
这里复用仿真评估已有的单臂适配：选择左臂10维归一化统计、屏蔽右臂槽位，模型本身仍是80维。
**这不是相机投影，不改网络或权重，也不需要client补零/改状态格式。**

命令只在`outputs/real_robot/maniskill20872_flask/deployment`产生部署副本；不修改原训练目录。
权重使用硬链接，禁止原地写入部署目录中的`.safetensors`，否则也会影响原始同inode文件。

```bash
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
from types import SimpleNamespace
from omegaconf import OmegaConf
from benchmarks.maniskill.archive_eval import prepare

source = Path('outputs/maniskill_sidecar/midtrain_patchwam_1epoch_20261031T184131Z/runs/2026-10-31_18-46-16/checkpoint_step_20872.safetensors')
output = Path('outputs/real_robot/maniskill20872_flask/deployment')
prepare(SimpleNamespace(checkpoint=source, output=output))

# 仅部署副本：使用旧Flask入口认可的单臂类型名；不是更改训练配置或重新训练。
path = output / 'checkpoint/config.yaml'
cfg = OmegaConf.load(path)
cfg.dataloader.type = 'maniskill_sidecar'
OmegaConf.save(cfg, path)
print('Prepared:', output / 'checkpoint' / source.name)
PY
```

如果提示部署目录已存在，不需要重复准备；此前已成功执行本节时，直接进入2.2。

### 2.2 启动服务

```bash
GPU_ID=7 MODEL=sidecar HOST=0.0.0.0 PORT=5901 \
CKPT=./outputs/real_robot/maniskill20872_flask/deployment/checkpoint/checkpoint_step_20872.safetensors \
MANIFEST=./outputs/real_robot/maniskill20872_flask/server_manifest.json \
bash benchmarks/real_robot/run_maniskill_flask.sh
```

该命令沿用第1节导出的`PYTHON_BIN`和`TOKEN_FILE`。完整预测为32×10；实际执行多少步由client的`CHUNK_STEPS`决定。
请求的front/wrist、EEF、任务和预测动作保存在`outputs/real_robot/maniskill20872_flask/inference_records/`。

## 3. 双臂 Piper21952：服务端

在另一个GPU服务器终端先执行第1节公共准备，再运行：

```bash
CUDA_VISIBLE_DEVICES=6 "$PYTHON_BIN" benchmarks/real_robot/piper/serve_piper_flask.py \
  --checkpoint ./outputs/piper_sidecar/midtrain_patchwam_2epochs_20261102T101245Z/runs/2026-11-02_10-18-00/checkpoint_step_21952.safetensors \
  --host 0.0.0.0 --port 5902 \
  --token-file "$TOKEN_FILE" \
  --record-dir ./outputs/real_robot/piper21952_flask/inference_records
```

Piper无需单臂适配，直接加载原checkpoint；输入三视图+双臂EEF20，返回32×20。
这里显式指定21952的记录目录，避免与旧12000日志混在一起。

## 4. 网络连接与健康检查

`0.0.0.0`表示监听所有网卡，不代表公网端口映射已自动建立。
如果已有公网映射，client直接使用映射地址；例如旧`210.45.70.21:29928`只有在仍映射到本机5901时才能用于单臂，不能假定现在仍有效。

没有公网映射时，可以在机器人端用原来的SSH隧道方案。将`SSH_PORT`改成当前确实能登录GPU服务器的端口：

```bash
# 机器人端独立终端运行；保持此连接打开。
SSH_PORT=35601
ssh -p "$SSH_PORT" -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18001:127.0.0.1:5901 \
  -L 127.0.0.1:18002:127.0.0.1:5902 \
  root@210.45.70.21
```

于是单臂client连接`127.0.0.1:18001`，Piper client连接`127.0.0.1:18002`。
两个机器人在不同机器上时，各自在其机器建立需要的隧道即可。隧道加密；不要在无保护公网明文HTTP上传token/画面。

在机器人端检查模型身份（当前Python环境需有requests）：

```bash
python - <<'PY'
from pathlib import Path
import requests
token = Path.home().joinpath('openwam_api_token').read_text().strip()
s = requests.Session()
s.trust_env = False
for port in (18001, 18002):
    r = s.get(f'http://127.0.0.1:{port}/health',
              headers={'Authorization': 'Bearer '+token}, timeout=10)
    r.raise_for_status()
    info = r.json()
    print(port, info['checkpoint'])
PY
```

应该分别显示`checkpoint_step_20872.safetensors`和`checkpoint_step_21952.safetensors`。
只部署其中一个时，只检查对应端口。HTTP ready只说明模型服务可达，不是任务成功率。

## 5. 单臂机械臂端：沿用已调通的client

无需更改EEF10、相机拼接、mask或归一化；这些都由server处理。
保留当前已调通文件中的基座/TCP转换和阈值，不要用仓库模板覆盖现场参数。

在当前`infer_openwam.py`顶部，将`API_URL`设为实际服务地址。
若使用上面的隧道，就是`API_URL = 'http://127.0.0.1:18001'`；用公网映射则填对应地址。
token可通过已有环境变量传入：

```bash
conda activate calib
export OPENWAM_API_TOKEN="$(cat ~/openwam_api_token)"
python /home/rpp/rpp_ws/src/application/robot_bringup/scripts/infer_openwam.py
```

在client交互提示中输入任务、夹爪状态和轮数，按现有流程确认执行。
`CHUNK_STEPS`决定每次执行预测的前几步；先保持现场已验证值，不因新权重直接扩大动作范围。
已有client如果可正常运行就不必换文件；需要核对干净实现时参考`benchmarks/real_robot/infer_openwam_minimal.py`，它依赖同目录原`infer.py`。

## 6. 双臂 Piper端：使用独立client_repo

先按原流程启动三相机和双臂ROS1驱动；本节不启动、不使能、不复位硬件，也不修改`~/kai0_repo`。

### 6.1 先只推理两轮

```bash
cd /home/ubuntu/client_repo
source scripts/activate.sh
export OPENWAM_API_TOKEN="$(cat ~/openwam_api_token)"
python inference/agilex_inference_openwam.py \
  --host 127.0.0.1 --port 18002 \
  --prompt 'stack the blocks' --steps 2
```

不带`--execute`只采集观测、请求动作、打印预测，不创建控制publisher。
新checkpoint沿用同一EEF20接口，无需修改client代码；任务文本换成你现场要测试的任务。

### 6.2 现场坐标核对后，执行一个4步前缀

先在`robot.json`保留/核对两臂基座与工具点转换，填写各自基座系下的安全`workspace_xyz`；
确认完成才设置`geometry_verified=true`。这不是因换权重重新标定，而是前一轮尚未完成的Piper首次真机几何验收。

```bash
python inference/agilex_inference_openwam.py \
  --host 127.0.0.1 --port 18002 --prompt 'stack the blocks' \
  --execute --chunk_size 4 --steps 1 --publish_rate 30
```

### 6.3 已完成运动验证后，连续推理与时间平滑

```bash
python inference/agilex_inference_openwam.py \
  --host 127.0.0.1 --port 18002 --prompt 'stack the blocks' \
  --execute --use_temporal_smoothing \
  --publish_rate 30 --inference_rate 2 \
  --latency_k 0 --min_smooth_steps 12 --chunk_size 32
```

这里不是旧kai0的50步关节chunk；OpenWAM每轮最多32步，`chunk_size`只能为1..32。
30Hz为沿用客户端参数，尚未证明与archive训练数据的实际时间轴匹配。
原版Kai0、环境、驱动不修改；新客户端只通过既有话题通信。

## 7. 本次已核对与尚未验证

- 两份指定checkpoint都存在，safetensors头声明长度与实际文件长度一致，config/stats/tokenizer配套存在。
- 单臂适配复用`benchmarks/maniskill/archive_eval.py::prepare`；只调整部署副本，保留原80维网络和左臂mask语义。
- 本文10个shell代码块及内嵌Python语法检查通过；用临时权重占位文件、20872真实配置/统计验证了准备流程和Flask配置条件，归一化左10维与训练一致，其余70个槽位为0，原配置不变。这是CPU配置验收，不是实际模型加载。
- 未实际启动这两份新权重、未验证公网映射或隧道，也未下发机械臂动作。
- Piper此前通过的是模拟ROS/HTTP链路，不等于实际模型生成、真机坐标和驱动IK/碰撞安全已验收。


## 8. 2026-11-04 本机实际部署更新（临时两卡）

按用户最终要求，单臂20872已实际部署到GPU2和GPU6：**GPU6主体 / GPU2 T5**，监听`0.0.0.0:5901`，PID4157435。原token未变（与10月24日/27日均一致），继续逐次保存原始front/wrist、请求JSON及完整动作响应。三次归档观测HTTP推理/32×10/重复一致/记录原文核对通过，服务保持运行。原单卡方案保留；本次仅可选分卡放置，不改模型/训练/客户端协议。

当前环境使用`.venv/bin/python`，第1节旧`/dev/shm/.../venv`已不存在；实际启动命令、配套SHA、记录与重启说明见[MANISKILL20872_REAL_ROBOT_GPU26_DEPLOYMENT_20261104.md](MANISKILL20872_REAL_ROBOT_GPU26_DEPLOYMENT_20261104.md)。Piper未在本轮启动；未下发任何真机动作。
