# ManiSkill20872 真机 HTTP 临时分卡部署

2026-11-04 已实际启动并完成 HTTP 推理验收。按用户最终要求使用 GPU2 和 GPU6；根据实时余量分配为 **GPU6 模型主体、GPU2 T5**。

| 项目 | 当前值 |
|---|---|
| 服务 | `0.0.0.0:5901` |
| 进程 | PID `4157435`；完整身份和日志见 `outputs/real_robot/maniskill20872_flask/process.json` |
| CUDA 可见顺序 | `6,2`；逻辑 `cuda:0` 为主体，`cuda:1` 为 T5 |
| 权重 | `checkpoint_step_20872.safetensors`，24,862,363,126 bytes |
| 权重 SHA256 | `62d66aea890035c98acd20728f6ff134e9bd6ea0368c8bd460f3934cfefc8eb8` |
| 请求 | front/wrist 原始图像、任务文本、绝对 EEF10 |
| 返回 | 完整 `32×10` 物理尺度绝对 EEF 动作 |
| 推理 | BF16、10 denoise steps、CFG=1、eager、DiT cache 关闭、视频解码关闭 |
| allocator 上限 | 本次主体 15.0215 GiB，T5 12.5 GiB |

## 实现范围

仅为当前显存不足增加可选的 `--split-text-encoder` 和独立启动脚本 `benchmarks/real_robot/run_maniskill_flask_split.sh`。放置方法沿用现有 eval 的 `serve_split.py`：T5 参数直接从 CPU 放到文本卡，编码结果传回主体卡；本次检查了全部参数的设备和 dtype。

保留原单卡默认启动方式和原 `run_maniskill_flask.sh`。额外的 allocator 余量和等待参数均为显式可选；原单卡默认 35 GiB 准入、4 GiB 余量保持。模型、训练、权重、归一化、相机预处理、Sidecar 计算和客户端接口不改。容量波动时，临时分卡进程保留 CPU 模型等待，不结束其他作业。

单臂部署副本按原启动文档的 `archive_eval.prepare` 生成，只选择左臂统计和槽位；80D 网络保持。原训练 config/stats SHA 未变，持久部署权重为原文件硬链接，不得原地写入。为加快共享文件系统读取，本次另将相同配套和权重复制到独立内存盘目录；复制流的大小/SHA 与实际加载后健康接口一致。路径和证据在 `outputs/real_robot/maniskill20872_flask/gpu2_ram_checkpoint.json`。内存盘副本随服务器重启失效，持久副本仍保留。

## Token 与逐次记录

**Token 未更换。** `outputs/real_robot/deploy_20261104/api_token` 与 2026-10-24、2026-10-27 两份旧真机 token 内容完全一致，权限 `0600`；未打印 token。

每次有效推理请求在以下目录建立独立子目录：

```text
outputs/real_robot/maniskill20872_flask/inference_records/
  <UTC时间_唯一ID>/
    request.json
    front.png 或 front.jpg
    wrist.png 或 wrist.jpg
    response.json
```

`request.json` 包含任务、EEF 状态、夹爪状态来源及请求标识；图像保存 HTTP 收到的原始字节；成功后保存包含完整动作和模型身份的响应。推理失败时已落盘的输入仍保留。此次三个请求的 JSON、两张图像和响应均逐字节/逐字段核对通过。

## 验收结果与证据

- 25 项原有 HTTP/单臂 archive 回归通过；Python/bash 语法检查通过。
- 小模型在真实两卡上的 T5 输入/输出转移检查通过，输出与参考逐元素一致；单卡/两卡容量等待分支的模拟检查通过。
- 未认证 `/health` 返回 401；原 token 认证后返回指定 checkpoint、SHA、PID、`CUDA_VISIBLE_DEVICES=6,2`。
- 使用未连接仿真的真实归档首帧：push_cube、pick_cube、再次 push_cube；三个 HTTP 请求均 200、有限 `32×10` 动作、夹爪越界数 0。
- 两次 push_cube 动作逐元素一致，maxabs=0。三次服务端延迟分别约 8.183 s（首次）、1.144 s、1.075 s；只是本次小试观测，不作为稳态吞吐结论。
- 服务保持运行；旧的本轮等待/部署进程均按 PID 和 start_ticks 身份停止，历史日志保留。其他作业未操作。

主要证据都在 `outputs/real_robot/maniskill20872_flask/`：

| 文件 | 内容 |
|---|---|
| `deployment_acceptance.json` | 最终在线健康、进程身份、token 一致、GPU 快照和验收状态 |
| `http_smoke.json` | 三次实际推理、重复一致性和原始记录路径 |
| `cpu_and_routing_validation.json` | 回归、设备转移检查和最终源码 SHA |
| `preparation_check.json` | 原 config/stats 不变、部署统计为原左10维 |
| `server_manifest.json` | 权重/统计 SHA、推理参数、分卡和 allocator 信息 |
| `server_gpu6_t5gpu2_20261104T093543Z.log` | 当前服务日志 |

## 后续重启

当前服务已运行，不要重复启动。以后停止本次服务后，可用持久部署副本启动相同模式：

```bash
cd /remote-home/jinminghao/OpenWAM
TOKEN_FILE="$PWD/outputs/real_robot/deploy_20261104/api_token" \
PYTHON_BIN="$PWD/.venv/bin/python" \
GPU_IDS=6,2 MIN_FREE_GIB=15 MAX_MEMORY_GIB=15.25 \
CKPT="$PWD/outputs/real_robot/maniskill20872_flask/deployment/checkpoint/checkpoint_step_20872.safetensors" \
HOST=0.0.0.0 PORT=5901 \
bash benchmarks/real_robot/run_maniskill_flask_split.sh
```

脚本保持 `hf-mirror.com` 和离线加载设置。客户端继续使用原 token、原 EEF10/front+wrist 格式及已验证的现场参数，仅需服务地址指向该服务器 5901。

本次只完成实际模型 HTTP 推理和文件记录验收，未连接或驱动机器人，未复验公网映射，不代表真机任务成功率或所有未来并发下的显存峰值。真机基座/工具点、IK/碰撞、控制时序以及记录目录长期磁盘增长等既有未决继续保留。下一步沿用现有客户端在现场先检查观测与预测；本次无需修改客户端格式。
