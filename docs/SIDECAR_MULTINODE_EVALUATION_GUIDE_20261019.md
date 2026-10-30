# Sidecar 20k：另一台机器启动标准评测

2026-10-19 UTC。命令已对照现有入口参数核查，尚未在目标机器执行验收。此指南评测当前 Sidecar 20k，不是旧环境流程，也不是原版 baseline。

## 先确认任务归属

当前本机50任务、Clean/Randomized均已排队。另一台机器不能直接再跑 `--tasks all` 并将结果合并；必须先将接管任务从本机待运行队列移交。现有队列没有跨机器锁，单独创建新输出目录也不会阻止重复任务。

外机先准备环境和模型服务，再确定英文任务列表，确认本机不再调度这些任务后启动正式队列。不要直接编辑运行中 `parallel.json` 来移交：它是状态快照，进程内队列不会随文件变化。可以独立短测验证链路，但不得并入正式分数。不终止其他人的作业。

结果唯一键为 `(checkpoint SHA256, deploy配置, 仿真环境, task, mode, seed group)`。意外重复运行须保留全部记录并指定有效运行，禁止择优取成绩。

## 代码、权重和环境

工作目录为 OpenWAM 根目录；异机路径不同时自行替换。必须包含当前未提交的 Sidecar 实现及评测工具，仅 clone 官方 OpenWAM 不够。优先共享当前工作区只读代码；若复制，连同未提交文件、配置、模型配套文件及链接目标复制，不要覆盖另一机器既有项目。

- 权重目录：`outputs/robotwin_eval/sidecar20000_currentenv_20261018/checkpoint_step_20000/`。需包含权重、config、stats、tokenizer等全部配套文件；其中存在指向训练 run 的软链接，异机必须保证目标存在。
- 权重文件：`checkpoint_step_20000.safetensors`，24,862,363,126 bytes。
- 权重 SHA256：`0d67d651577ef53d5a86b0e1e03dd93f6cc7bf652c3c7af08734b7712042ea6e`。
- 正式推理配置：`outputs/robotwin_eval/sidecar20000_officialenv_20261018/deploy_selected.yaml`。
- 客户端配置：`benchmarks/robotwin/policy_config_relational_sidecar.yml`，使用 Sidecar 的 wxyz 合同和自有归一化统计，不能换成原版配置。
- RoboTwin：`.external/robotwin-official-20261018/RoboTwin`，commit `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4`，包含已验收资产及本机路径配置。
- 模型环境：当前 `.venv` 为 Torch 2.7.1+cu128。
- 仿真环境：当前 `.conda/robotwin-official`，Python 3.10.22、Torch 2.4.1+cu121、SAPIEN 3.0.0b1、warp-lang 1.13.0、mplib 0.2.1、numpy 1.26.4、scipy 1.10.1、cuRobo 0.7.8、PyTorch3D 0.7.8。

新机器不能仅凭共享目录存在就认定环境可用：检查 GPU、驱动、Vulkan/RT、GLIBC、Python prefix、cuRobo/PyTorch3D 扩展与 Torch ABI。A100 的架构是8.0，L20是8.9，不要照搬L20编译二进制或架构参数；必要时在目标机器独立重建环境和扩展，保留原机器文件。路径变更时仅在独立 RoboTwin 副本重新生成资产/机器人配置路径。原资产验收证据在 `outputs/robotwin_eval/rebuild_official_20261018/`，依赖版本可查 `environment_installed.json` 和 `isolated_import_paths.json`。

不需要重新下载模型/资产。确需 Hugging Face 下载时先使用 `HF_ENDPOINT=https://hf-mirror.com`；镜像失败不得自动回退官方/CDN，需用户单独批准。

## 在目标机器准备变量

下面是1个模型服务、2路仿真的模板；根据实际空闲 GPU 调整。L20每份模型约24 GiB，仿真每路约5–8 GiB；建议仿真预留8–10 GiB。`SIM_GPUS=1,1` 表示两路均用物理GPU1，不是GPU编号范围。

```bash
cd /remote-home/jinminghao/OpenWAM
export ROOT="$PWD"
export MODEL_PY="$ROOT/.venv/bin/python"
export SIM_PY="$ROOT/.conda/robotwin-official/bin/python"
export ROBOTWIN="$ROOT/.external/robotwin-official-20261018/RoboTwin"
export CKPT="$ROOT/outputs/robotwin_eval/sidecar20000_currentenv_20261018/checkpoint_step_20000"
export DEPLOY="$ROOT/outputs/robotwin_eval/sidecar20000_officialenv_20261018/deploy_selected.yaml"
export POLICY="$ROOT/benchmarks/robotwin/policy_config_relational_sidecar.yml"
# 必须改成目标机器实际存在的 CUDA toolkit 路径。
export EVAL_CUDA_HOME=/remote-home/jinminghao/miniconda3/envs/nvcu121_cuda-12.4
export MODEL_GPU=0
export SIM_GPUS=1,1
export EVAL_PORT=20600
export EVAL_ARCH=8.9  # L20=8.9；A100=8.0
export OUT="$ROOT/outputs/robotwin_eval/sidecar20k_$(hostname -s)_$(date -u +%Y%m%dT%H%M%SZ)"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TORCH_CUDA_ARCH_LIST="$EVAL_ARCH"
export TORCHINDUCTOR_CACHE_DIR="$OUT/inductor"
mkdir -p "$OUT"
nvidia-smi
"$MODEL_PY" -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
"$SIM_PY" -c 'import sys,torch,sapien,warp,mplib; print(sys.prefix,torch.__version__,torch.cuda.is_available())'
sha256sum "$CKPT/checkpoint_step_20000.safetensors"
git -C "$ROBOTWIN" rev-parse HEAD
```

上面的 import 通过不等于闭环或 RT 验收通过。首次换机器需要另做少量闭环，并单独保存为 smoke。

## 启动模型池

仅运行专用 `sidecar_fast_runtime.py`，不要换成普通 `scripts/deploy.py`。后者不安装 Sidecar cache/compile 推理适配。

```bash
CUDA_VISIBLE_DEVICES="$MODEL_GPU" nohup "$MODEL_PY" -u \
  benchmarks/robotwin/sidecar_fast_runtime.py \
  --ckpt-dir "$CKPT" --config "$DEPLOY" \
  --port "$EVAL_PORT" --slots 2 \
  --manifest "$OUT/pool.json" --readiness-token "sidecar20k-$(hostname -s)-$EVAL_PORT" \
  > "$OUT/server.log" 2>&1 < /dev/null &
export POOL_PID=$!
echo "$POOL_PID" > "$OUT/server.pid"
```

服务用同一权重、10步去噪、32动作、CFG=1、三处Sidecar读写完整保留；velocity cache开启0.99/max3，compile只编译Sidecar FFN，禁止误用原版foundation-only编译循环。另一个硬件/软件组合的数值和性能尚未验收，不能沿用L20的2.28倍提速结论；缓存是近似计算，不承诺与无缓存完全相同。

首次运行可能有加载和编译冷启动。`pool.json`/PONG只表示服务就绪，正式结果仍要检查闭环实际执行、成功数及完成数。如果需要GPU数值门槛复验，复用 `check_sidecar_fast_runtime.py` / `serve_sidecar_checked.py` 的验证实现，但它们含当前实验的路径/调度假设，不直接照抄运行；先核查目标机器路径与服务占用。

## 独立短测（建议首次异机执行）

在模型池已就绪后运行。短测使用独立输出，不进入正式队列统计；每个端口同一时间只允许一个客户端。

```bash
"$MODEL_PY" -u benchmarks/robotwin/standard_eval.py \
  --checkpoint "$CKPT" --output "$OUT/smoke" \
  --tasks adjust_bottle --modes demo_clean --episodes 2 \
  --server-gpu "$MODEL_GPU" --sim-gpu "${SIM_GPUS%%,*}" \
  --port "$EVAL_PORT" --reuse-server-pid "$POOL_PID" \
  --cuda-home "$EVAL_CUDA_HOME" --cuda-arch "$EVAL_ARCH" \
  --deploy-config "$DEPLOY" --policy-config "$POLICY" \
  --robotwin-path "$ROBOTWIN" --sim-python "$SIM_PY" \
  > "$OUT/smoke.log" 2>&1
```

确认 exit0、`smoke/run.json` 完整、两条实际完成、无 NaN/渲染/规划异常后再开始正式队列；成功率不作为短测放行的择优条件。

## 启动正式队列

先设置已完成移交的英文任务名，逗号分隔。下面不预填任务，防止无意重复本机队列。每任务自动运行Clean/Randomized各100条。

```bash
# 例：export TASKS='task_a,task_b'；请替换成真实且已移交的英文任务名。
: "${TASKS:?先设置已移交的任务列表}"
nohup "$MODEL_PY" -u benchmarks/robotwin/parallel_eval.py \
  --output "$OUT/queue" --checkpoint "$CKPT" \
  --pool-manifest "$OUT/pool.json" --pool-pid "$POOL_PID" \
  --sim-gpus "$SIM_GPUS" --episodes 100 \
  --cuda-home "$EVAL_CUDA_HOME" --cuda-arch "$EVAL_ARCH" \
  --deploy-config "$DEPLOY" --policy-config "$POLICY" \
  --robotwin-path "$ROBOTWIN" --sim-python "$SIM_PY" \
  --tasks "$TASKS" > "$OUT/queue.log" 2>&1 < /dev/null &
echo "$!" > "$OUT/queue.pid"
```

增加路数时，`--slots`至少等于 `SIM_GPUS` 中GPU条目数，端口段必须空闲。模型池每个slot有独立动作buffer，但共享模型计算锁；增加仿真并行不会让同一份模型并行计算，超过服务吞吐后不一定提速。需要更多模型服务时，分别建立新OUT、端口段、GPU和互斥任务列表。

**不要直接运行 `launch_sidecar_official.py`**：它是本机实验专用调度器，包含固定输出目录、历史PID清理和8卡分配，不是可移植启动入口。

## 查看与回传结果

- `server.log`：模型加载、编译、推理错误。
- `queue.log`、`queue/parallel.json`：任务、PID、complete/running/failed状态。
- `queue/jobs/<mode>/<task>/run.json`：协议、配置/权重hash、环境与结果验收。
- `queue/jobs/<mode>/<task>/<mode>/<task>/result.json`：实际完成数、成功数、accepted seeds。

只有exit0、完整100条、无errors并通过runner验收，才可计入百条成功率。专家失败跳过、策略失败计入分母，保持官方RT、动作预算、unseen指令、seed group0，不启用planner fallback。失败保留日志，不改预算、跳过失败seed或选较高分重跑。

本机现有实时监控目前仅扫描原来的三个固定queue目录，不会自动发现外机OUT。回传完整OUT路径后，需要按任务归属校验、去重后单独汇总；不要手动改本机旧结果。使用不同GPU/驱动/依赖时保留机器标签，不称作逐seed比特复现。

更多依据：`SIDECAR_INFERENCE_ACCELERATION_20261018.md`、`ROBOTWIN_DUAL_TRACK_EVALUATION_20261018.md`、`ROBOTWIN_LIVE_RESULT_ANALYSIS_20261018.md`。
