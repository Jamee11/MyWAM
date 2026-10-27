# RoboTwin：A100/L20 模型服务 + RTX 4090 仿真

**用户当前部署请直接按第4节操作**：已知4090入口为
`ssh root@210.45.70.21 -p 26199`，推荐模型端主动建立反向隧道。

2026-10-22 UTC。用户指定 Sidecar42960：
`outputs/sidecar/aloha50_C_joint_10epochs_20261016T131709Z/runs/2026-10-16_13-50-52/checkpoint_step_42960.safetensors`。
使用已封存的同一权重目录
`outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z/checkpoint_step_42960`，
不复制另一份 25GB 权重，不修改训练输出。权重 SHA256 应为
`501addbed687e6d7587af60ab4480b9c79eaa2c8867d723b74d54e0acc9deb55`。

## 拓扑与边界

4090：RoboTwin RT 渲染/物理/cuRobo → 原生三视角 RGB H240×W320 →
原有 PNG/base64 WebSocket → SSH 隧道 → A100/L20：原模型预处理 H384×W320、
去噪及动作预测 → 同隧道返回绝对 EEF 动作 → 4090 执行。
权重、归一化和 tokenizer 仅由模型服务器加载；4090 不加载模型。

复用原 `single_eval.sh`、`eval_policy_wrapper.py`、适配器与共享模型池。
新增入口不改模型/训练/原评测源文件。Sidecar 使用已有 FFN compile + cache，
启动时保留真实模型数值/kernel 验证门槛；baseline 使用原完整 compile 路径。
每个并行 simulator 必须独占一个模型 slot/端口，不能多个 simulator 共用端口。
模型内部执行仍互斥、每 slot 有独立 policy/action buffer。

默认不存视频，保留 RT、官方专家筛 seed、unseen 指令、seed group 0、成功判据和
动作预算。默认 1 episode 只验链路；正式成绩必须显式指定 100。
相机保持官方 near=0.1；不混入独立 wrist-near 诊断。
源文件指纹、模型 session、耗时、实际条数和原生结果记录在每次 run.json。
跨机共享的 ready token 是 session 标识，并非鉴权密码；服务保持 loopback，经 SSH 访问。

## 1. 模型端：L20 / A100

在项目根目录，先以 `nvidia-smi` 选有足够余量的卡。将 `GPU_ID` 设为实际空闲卡。
输出目录必须是新目录。使用 tmux 保持服务前台运行，查看 `server.log`；
出现 `session_slot0.json` 才表示真实模型校验与端口检查通过。

```bash
cd /remote-home/jinminghao/OpenWAM
GPU_ID=0  # 示例，必须根据实时资源调整
RUN=outputs/robotwin_eval/split_host42960_$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python benchmarks/robotwin/remote_eval.py model \
  --kind sidecar --gpu "$GPU_ID" --port 22700 --slots 2 \
  --checkpoint outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z/checkpoint_step_42960 \
  --deploy-config outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z/deploy.yaml \
  --output "$RUN"
```

默认部署前要求 33000MiB 空闲，无余量直接退出，不结束其他作业。
每次部署独立 Inductor cache，A100/sm80 与 L20/sm89 不共用编译产物。
不同模型使用不同输出目录与端口。baseline 用 `--kind baseline`、该 checkpoint
自己的 artifacts 和官方 deploy 配置；不得套用 Sidecar 的 FFN 配置。

## 2. 4090 端：SSH 隧道

用户确认共享盘路径相同，可直接使用同一份文件。4090 需能 SSH 到模型机器。
公钥添加完成后，将下例 `user@L20_HOST` 和 SSH 端口替换为实际值。
隧道另开 tmux 窗口保持运行，失败不会静默转连其他模型。

```bash
cd /remote-home/jinminghao/OpenWAM
bash benchmarks/robotwin/remote_tunnel.sh user@L20_HOST 22 22700:22700 22701:22701
```

共享磁盘不保证共享 Conda 二进制能在异机运行。4090 首先检查 GLIBC、驱动、
Python/torch/SAPIEN/cuRobo 导入、Vulkan RT 和真实规划/渲染；不能仅凭 import
或 ping 声称仿真已通过。4090 与 L20 都是 sm89，但首次部署仍隔离扩展缓存，
不要覆盖旧 `.so` 或安装到原环境。A100 上只做模型，不要求其承担 RT 仿真。

## 3. 4090 端：连接检查与单任务闭环

`RUN` 指向步骤1实际生成的共享输出路径；`SIM_PYTHON` 首先检查可执行/导入。
CUDA_HOME 应为目标机可用的 toolkit；不要盲目沿用本机驱动/preload 设置。

```bash
cd /remote-home/jinminghao/OpenWAM
RUN=/absolute/path/to/actual/split_host42960_run
SIM_PYTHON=/remote-home/jinminghao/OpenWAM/.conda/robotwin-official/bin/python
export TORCH_CUDA_ARCH_LIST=8.9
"$SIM_PYTHON" benchmarks/robotwin/remote_eval.py probe \
  --session "$RUN/session_slot0.json" --port 22700
"$SIM_PYTHON" benchmarks/robotwin/remote_eval.py sim \
  --session "$RUN/session_slot0.json" --port 22700 --gpu 0 \
  --robotwin-path /remote-home/jinminghao/OpenWAM/.external/robotwin-official-20261018/RoboTwin \
  --task adjust_bottle --mode demo_clean --episodes 1 \
  --output "$RUN/4090_adjust_bottle_clean_smoke"
```

首次 1 条闭环后，再以新输出目录 `--episodes 100` 正式运行。
第二个 simulator 用 `session_slot1.json` / 22701，建议先用不同 task/mode，
避免误把同一 task 的两次 100 条相加成 200 条独立场景。
同一任务要拆分并行时，应使用已有 sealed manifest/shard 工具，不能简单重复启动。
本机文件锁只覆盖同一个 simulator 主机；不同 simulator 主机也必须人工分配独占 slot。

## 当前验证及待完成

部署入口、SSH 转发入口和标准仿真调用入口已新增。
本机通过现有真实 L20 模型做只读 ping/slot 身份与 TCP 转发检查；
这些检查不 reset、不执行预测，不构成 42960 新服务或 4090 闭环验收。
实际验证证据见 `outputs/robotwin_eval/split_host_setup_20261022/validation.json`。
初次搭建时等待 SSH 地址；后续已收到地址并完成第4节的真实隧道检查。
42960新服务及4090策略闭环仍未启动，不能提前报告异机评测速度或 SR。
接下来依次完成：4090 环境预检 → 新 42960 模型 ready → 隧道 → 单条闭环 →
同模型/同任务/同 seed 的 L20 与 4090 仿真计时。
计时需区分冷启动/编译、模型响应、网络往返、完整 rollout 与多路吞吐。

## 4. 已知4090地址的直接部署命令与参数表

2026-10-22现场检查：当前L20 GPU0空闲21287MiB（约20.8GiB）、利用率98%，
不足新模型33000MiB启动门槛；不在这张卡强行启动。4090 GPU1空闲23969MiB，
是当时可选的仿真卡，启动前重新检查，不代表已预订。

| 参数 | 在哪里设置 | 当前值/规则 |
|---|---|---|
| 项目根目录 | 两端 `cd` | `/remote-home/jinminghao/OpenWAM`；共享路径相同 |
| 模型Python | 模型端启动命令 | `.venv/bin/python`；异机需能运行CUDA模型环境 |
| 模型GPU | 模型端 `--gpu` | 物理编号，例如空闲的 `0`；与仿真GPU独立 |
| 模型端基础端口 | 模型端 `--port` | `22700`；2 slots实际使用22700、22701 |
| 并行slot数量 | 模型端 `--slots` | `2`；增加后必须同步增加隧道映射，给每个sim独占slot |
| 4090 SSH用户名/IP | 反向隧道第1参数 | `root@210.45.70.21` |
| 4090 SSH端口 | 反向隧道第2参数 | `26199`，用于SSH连接，不是模型服务端口 |
| 隧道端口对 | 反向隧道后续参数 | `4090本地端口:模型端口`，如 `22700:22700` |
| 仿真GPU | 4090命令 `--gpu` | 示例 `1`；只运行物理/规划/渲染 |
| 仿真连接端口 | 4090命令 `--port` | 隧道左侧端口，如22700；host保持127.0.0.1 |
| checkpoint | 模型端 `--checkpoint` | 封存42960目录，内含权重、config、stats、tokenizer |
| 模型配置 | 模型端 `--deploy-config` | 42960现有deploy.yaml；不需要改原文件 |
| 实验目录 | 模型端 `--output`、4090 `RUN` | 两端指向同一个新共享目录，不能复用已存在run |
| session文件 | 4090 `--session` | slot0→session_slot0.json；slot1→session_slot1.json |
| 仿真Python | 4090命令解释器 | `.conda/robotwin-official/bin/python` |
| RoboTwin目录 | 4090 `--robotwin-path` | `.external/robotwin-official-20261018/RoboTwin`的绝对路径 |
| CUDA toolkit | 4090 `CUDA_HOME` / `PATH` | 目标机实际可用toolkit；若需JIT编译，确认nvcc可用 |
| 任务/域/条数 | 4090 `--task/--mode/--episodes` | adjust_bottle/demo_clean/1先验链路；正式改100 |

**A：在选定的模型机器启动服务。** 换A100或L20只需登录目标机器，检查模型Python和GPU。
`MODEL_GPU=0`是占位示例，不能用于当前余量不足的L20 GPU0。

```bash
cd /remote-home/jinminghao/OpenWAM
MODEL_GPU=0
MODEL_PORT=22700
RUN="$(pwd)/outputs/robotwin_eval/split_host42960_$(date -u +%Y%m%dT%H%M%SZ)"
echo "共享实验目录：$RUN"
.venv/bin/python benchmarks/robotwin/remote_eval.py model \
  --kind sidecar --gpu "$MODEL_GPU" --port "$MODEL_PORT" --slots 2 \
  --checkpoint outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z/checkpoint_step_42960 \
  --deploy-config outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z/deploy.yaml \
  --output "$RUN"
```

**B：仍在模型机器，另开终端建立反向隧道。** 不需要填写模型机器的SSH地址。
下面两对端口应对应步骤A的模型端口；`26199`只在这条命令中使用。

```bash
cd /remote-home/jinminghao/OpenWAM
bash benchmarks/robotwin/remote_reverse_tunnel.sh \
  root@210.45.70.21 26199 22700:22700 22701:22701
```

隧道在前台持续运行；两台机器需要长期运行的终端都可用tmux保持。
换模型机器时先结束旧反向隧道，释放4090对应端口；或选择新的左侧端口，
如 `22800:22700`、`22801:22701`，同时将4090的 `--port`改为22800/22801。
多个模型机器同时连同一4090时，左侧端口必须互不相同。

**C：登录4090，先ping后单条闭环。** 把 `RUN`改成步骤A打印的真实完整路径。

```bash
ssh root@210.45.70.21 -p 26199
cd /remote-home/jinminghao/OpenWAM
RUN=/替换为步骤A打印的完整实验目录
SIM_GPU=1
SIM_PORT=22700
SIM_PYTHON="$(pwd)/.conda/robotwin-official/bin/python"
export TORCH_CUDA_ARCH_LIST=8.9
"$SIM_PYTHON" benchmarks/robotwin/remote_eval.py probe \
  --session "$RUN/session_slot0.json" --port "$SIM_PORT"
"$SIM_PYTHON" benchmarks/robotwin/remote_eval.py sim \
  --session "$RUN/session_slot0.json" --port "$SIM_PORT" --gpu "$SIM_GPU" \
  --robotwin-path "$(pwd)/.external/robotwin-official-20261018/RoboTwin" \
  --task adjust_bottle --mode demo_clean --episodes 1 \
  --output "$RUN/4090_adjust_bottle_clean_smoke"
```

本次真实检查已通过：4090 SSH登录、共享项目目录、共享Python启动及包版本读取
（Python3.10.22、torch2.4.1、SAPIEN3.0b1、Warp1.13、websockets16.1.1）；
从当前L20临时反向转发 `4090:22790 → L20:21652`，4090实际运行薄client ping，
返回正确旧32k模型session/slot，随后已停止临时隧道、释放22790。
这是旧服务的真实网络验收，没有reset/obs/predict，未将旧模型作为42960测试。
证据：`outputs/robotwin_eval/split_host_setup_20261022/ssh_reverse_validation.json`。
真实42960加载/数值校验、4090 CUDA/cuRobo/RT及首条策略闭环仍待执行。

## 5. nohup隧道显示Stopped的处理

用户A100端出现多次后台隧道Stopped，日志仅nohup ignoring input。
这是后台SSH尝试终端交互的典型现象，实际触发项仍需前台SSH确认。
2026-10-22两个隧道脚本已加 `-n -o BatchMode=yes -o ConnectTimeout=15`，
不读取stdin、不请求密码/主机确认；认证/host-key失败直接退出并保留错误。
不同机器SSH身份不同，L20已能免密登录不代表A100也已经配置完成。

仅结束重复隧道，不结束模型。以用户所贴PID为例，在同一A100终端先确认：

```bash
ps -p 3543437,3543539,3543667 -o pid,stat,args
```

确认它们确为本次重复的隧道后，发送TERM再CONT使暂停进程处理退出信号：

```bash
kill -TERM 3543437 3543539 3543667
kill -CONT 3543437 3543539 3543667 2>/dev/null
```

不要对全部ssh进程使用pkill，以免影响登录/其他隧道。
在A100前台确认目标主机身份、完成认证：

```bash
ssh -p 26199 root@210.45.70.21 'echo SSH_OK'
ssh -n -o BatchMode=yes -o ConnectTimeout=15 -p 26199 root@210.45.70.21 'echo KEY_AUTH_OK'
```

第二条成功后，再后台启动一次修复后的隧道脚本。
如果仅密码登录可用，可使用下面命令让SSH在前台先认证，再自行进入后台；
不要再加nohup或末尾&，也不要与脚本同时启动相同映射：

```bash
ssh -fNT -p 26199 -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:22700:127.0.0.1:22700 \
  -R 127.0.0.1:22701:127.0.0.1:22701 \
  -R 127.0.0.1:22702:127.0.0.1:22702 \
  -R 127.0.0.1:22710:127.0.0.1:22710 \
  -R 127.0.0.1:22711:127.0.0.1:22711 \
  -R 127.0.0.1:22712:127.0.0.1:22712 \
  root@210.45.70.21
```

如果出现remote port forwarding failed，先检查4090上对应端口监听及其归属；
停止本实验遗留隧道或换新的左侧端口，不结束未知监听进程。
