# Piper 21952：独立 RoboTwin 标准评估入口

日期：2026-11-04。用户授权在现有 Aloha 评估基础上，以最小增量建立双 Piper 流程。本次只新增 Piper 入口、部署配置、私有 task config、队列和检查；不改 Aloha 默认入口、官方 checkout、模型结构、训练数据或正在运行的 Aloha 作业。

## 权重与任务范围

目标权重：

```text
outputs/piper_sidecar/midtrain_patchwam_2epochs_20261102T101245Z/runs/2026-11-02_10-18-00/checkpoint_step_21952.safetensors
```

任务来自 `datasets/multisim_sidecar_archive_v2_20261025/manifest.ready.json` 中 `group=robotwin_piper, split=train` 的去重任务。实际 **35 tasks**，train 9566 episodes，整个 Piper 分组 12000 episodes（val 1172，test 1262）。每次启动重新从 manifest 生成清单，不复用 Aloha 的 50 任务硬编码。

|序号|Task|序号|Task|
|---:|---|---:|---|
|1|beat_block_hammer|19|place_burger_fries|
|2|blocks_ranking_rgb|20|place_container_plate|
|3|blocks_ranking_size|21|place_dual_shoes|
|4|click_bell|22|place_empty_cup|
|5|grab_roller|23|place_mouse_pad|
|6|handover_block|24|place_object_stand|
|7|handover_mic|25|place_phone_stand|
|8|lift_pot|26|place_shoe|
|9|move_can_pot|27|press_stapler|
|10|move_pillbottle_pad|28|put_bottles_dustbin|
|11|move_playingcard_away|29|shake_bottle|
|12|move_stapler_pad|30|shake_bottle_horizontally|
|13|open_laptop|31|stack_blocks_two|
|14|pick_diverse_bottles|32|stack_bowls_three|
|15|pick_dual_bottles|33|stack_bowls_two|
|16|place_a2b_left|34|stamp_seal|
|17|place_a2b_right|35|turn_switch|
|18|place_bread_basket|||

两域为官方 `demo_clean`、`demo_randomized`。正式每域每任务 100 条，共 7000 条；速测设 `EPISODES=30`，共 2100 条。这里的 35 任务结果不能直接与 Aloha 的 50 任务官方均值比较，也不把归档 `fresh_clutter_randomized` 当作与官方 `demo_randomized` 完全相同的分布。

## 协议与对齐

- 官方 checkout：`.external/robotwin-official-20261018/RoboTwin`；仿真 Python：`.conda/robotwin-official/bin/python`。
- 以官方 Clean/Randomized config 为基础，私有 runtime 只将 `embodiment` 改为 `[piper, piper, 0.6]`、`eval_video_log` 改为 `false`，文件名独立为 `piper_demo_clean.yml` / `piper_demo_randomized.yml`。0.6 是左右机器人间距，来自采集配置。随机化、D435 三视角、RT、专家筛 seed、成功判据、任务动作预算保留。
- `action_type=ee`；输入状态是实际 `endpose`，左右各 `xyz + rot6d + gripper`，合计 20D。源 quaternion 为 `wxyz`，复用现有转换。模型 80D 中启用 `[0:10,34:44]`，Piper 归一化统计随权重加载；不调用 Aloha qpos/FK。
- checkpoint contract 为 `next_achieved_absolute_eef`、`robotwin_endpose_reference`。原始 Clean/fresh-clutter 各一条 `beat_block_hammer` 全序列已核对：prepared state 与 raw/endpose 经同一转换最大差 0；prepared action 与下一帧 achieved state 最大差 0。证据：`outputs/analysis/piper_eval_contract_20261104/raw_contract_checks.json`。这是两个真实 episode 的接口证据，不外推全任务物理跟踪精度。
- 三相机原始图像均 240×320；模型布局仍 384×320。归档 codec 为 `robotwin_legacy_rgb_jpeg`，保留现有 legacy 输入约定，不额外加入颜色反转或改相机/裁剪。
- Piper archive reader 在代码内直接拼接三视图，saved dataloader 没有 `multiview` 字段。因此专用 deploy 必须显式 `dataloader.multiview: true`、`camera_layout: ['0','1','2']`，与 `PreparedSimDataset` 的布局键完全一致。三个固定 client 相机按位置映射：head→0、left wrist→1、right wrist→2，画布上部 256×320、底部左右各 128×160。
- 完整 Sidecar；Eager；DiT cache 开启（0.99/max3）；10 denoise、CFG1、sync、action chunk32；默认不存视频。复用现有 `chunk_transport.py` 和 `skip_chunk_capture.py`，只在缓冲动作间省去不必要的 RGB 采集，保留物理执行、轻量本体状态与成功检查。模型会话独立，多个仿真分别使用不同端口。
- 当前 official Piper 资产是 `planner=curobo`，归档采集代码是 `mplib_RRT`。保留当前官方资产与 planner，不为提高成功率改回历史实现；这仍是训练采集与当前闭环的已知差异，CPU 接口检查不能证明两种 planner 的闭环结果等价。

原生评估仍按官方候选 seed 顺序执行专家验证，seed-group 默认 0，使用 unseen 指令；各 task/domain 独立进程。它不是一个条件内由竞争 worker 争夺 episode 的 batch evaluator。不同 checkpoint 需要配对分析时，仍应比较实际 accepted seed 与指令，不能仅凭相同起始 seed 宣称逐条输入一致。

## 启动服务（L20/A100）

下面 GPU 与端口只是可改参数；先查看真实余量。入口自身不停止已有服务，不设置保守的固定空闲显存门槛。

```bash
cd /remote-home/jinminghao/OpenWAM
CKPT_DIR="$PWD/outputs/piper_sidecar/midtrain_patchwam_2epochs_20261102T101245Z/runs/2026-11-02_10-18-00"
MODEL_RUN="$PWD/outputs/robotwin_eval/piper21952_model_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$MODEL_RUN"

# 示例：GPU6 放主模型，GPU3 放 T5，四个独立仿真会话。
CUDA_VISIBLE_DEVICES=6,3 PIPER_SPLIT_T5=1 \
  .venv/bin/python -u benchmarks/robotwin/serve_piper_sidecar.py \
  --ckpt-dir "$CKPT_DIR" --port 43400 --slots 4 \
  --manifest "$MODEL_RUN/pool.json" \
  --readiness-token piper21952 \
  --config configs/deploy_piper_eval.yaml \
  > "$MODEL_RUN/server.log" 2>&1 &
```

`--ckpt-dir` 复用现有服务逻辑，选择目录中 step 最大的权重；当前该目录最大值为 21952。如果后续目录中新增更高 step，先建立仅含目标权重及配套 `config.yaml`、`normalization_stats.npy`、`relational_sidecar_contract.json` 的独立部署目录，再传该目录。共享文件系统可用硬链接，不需要重复下载大文件。

单卡能够容纳时，改为 `CUDA_VISIBLE_DEVICES=<GPU>` 并去掉 `PIPER_SPLIT_T5=1`。split 仅做进程内设备放置，不改变模块/权重；本次实际固定输入/闭环验收状态见下方记录，不能仅凭接口设计宣称数值验证完成。

四个会话端口为 43400–43403。更多会话增大 `--slots`；更多模型选不同 base port。多个会话共享模型权重，分别维护缓存/动作状态，不代表四份模型或 GPU batched inference。

首次载入权重需要数分钟。确认 `server.log` 已出现端口监听日志且 `pool.json` 已生成后，再启动仿真队列；仿真端现有的 300 秒健康等待用于连接服务。

## 4090 仿真与反向隧道

共享盘路径一致，4090 复用同一项目和仿真环境。模型仅监听 loopback，通过现有脚本从模型机发起反向 SSH：

```bash
# 在模型机执行；先完成 SSH 公钥/known_hosts 设置。
nohup bash benchmarks/robotwin/remote_reverse_tunnel.sh \
  root@210.45.70.21 26199 \
  43400:43400 43401:43401 43402:43402 43403:43403 \
  > "$MODEL_RUN/tunnel.log" 2>&1 &
```

在 4090 运行速测（两个 GPU 各两路，按实际峰值调整）：

```bash
cd /remote-home/jinminghao/OpenWAM
SIM_GPUS=0,0,1,1 PORTS=43400,43401,43402,43403 \
  EPISODES=30 MODEL_HOST=127.0.0.1 \
  OUTPUT="$PWD/outputs/robotwin_eval/piper21952_30ep_$(date -u +%Y%m%dT%H%M%SZ)" \
  bash scripts/eval_robotwin_piper.sh
```

脚本自身配置当前 Aloha 已验证的仿真环境：`CUDA_HOME=/remote-home/jinminghao/miniconda3/envs/nvcu121_cuda-12.4`，Curobo 的 `envs/curobo/src` 加入 `PYTHONPATH`，SAPIEN NVIDIA EGL 配置，OMP/MKL 各 4、MAX_JOBS=2，架构 8.9（4090/L20），复用 `4090_trend30_extensions` 编译缓存，并保持官方 RT 开启。因此不依赖当前 shell 偶然继承旧作业环境。迁移 CUDA/仿真环境时可覆写 `CUDA_HOME`、`SIM_PY`、`__EGL_VENDOR_LIBRARY_FILENAMES`；更换为其他仿真 GPU 架构时同时覆写 `TORCH_CUDA_ARCH_LIST`/`TORCH_EXTENSIONS_DIR`。环境选项写入 `plan.json` 便于复核。

在模型机也能通过 SSH 执行这条远端命令。SSH 连接端口 `26199`、模型会话端口 `434xx`、物理仿真 `SIM_GPUS` 分别配置，互不替代。端口不要与已有 Aloha 服务/隧道重复；每个并行仿真配一个不同会话端口，可重复使用物理 GPU 编号。

先跑单 task/domain 的小试：

```bash
SIM_GPUS=0 PORTS=43400 EPISODES=1 TASKS=click_bell \
  OUTPUT="$PWD/outputs/robotwin_eval/piper21952_clickbell_smoke_$(date -u +%Y%m%dT%H%M%SZ)" \
  bash scripts/eval_robotwin_piper.sh --modes demo_clean
```

正式评估仅将 `EPISODES=100`，使用新的输出目录；全任务队列自动均衡分发到空闲 worker。默认无视频。入口不会自行释放显卡、自动部署模型或抢占已有 Aloha 会话。

```bash
SIM_GPUS=0,0,1,1 PORTS=43400,43401,43402,43403 EPISODES=100 \
  OUTPUT="$PWD/outputs/robotwin_eval/piper21952_100ep_$(date -u +%Y%m%dT%H%M%SZ)" \
  bash scripts/eval_robotwin_piper.sh
```

## 结果与必要检查

`plan.json` 保存任务/manifest SHA、checkpoint、episode 数与协议；`state.json` 保存逐条件状态、attempt、GPU/port 和实际 accepted seeds；`summary.json` 汇总已完整通过验收的条件，`success_rate` 为 0–1 小数。逐 task/domain 的 `jobs/<mode>/<task>/attemptN/` 保存 policy、私有 runtime、原生日志及验收 `run.json`。

复用现有 `standard_eval.validate_result` 检查真实 episode 数、success 累计、accepted seed 递增、unseen 指令与 native `_result.txt` 分母。明确 renderer 错误、反复缺 planner、20 分钟无日志进展的自有 attempt 会结束并重试（默认最多 3 次），错误 attempt 不混入成功率；0 成功但正常完成的任务照常计入。没有添加全局并行上限或显存余量锁。

CPU/配置证据已完成：

- `tests/benchmarks/test_piper_robotwin_eval.py` 涵盖真实接口约束的最小 metadata fixture、任务过滤、私有配置、完整结果分母，以及 saved config 缺少 multiview 时专用 deploy 仍强制三视图的回归检查。
- 真实 manifest `CHECK_ONLY=1 EPISODES=30` 生成 35 tasks/70 conditions/2100 episodes：`outputs/robotwin_eval/piper21952_prepare_20261104/plan.json`。
- 两份配置相对官方仅 embodiment/video 两项差异，Aloha wrapper/interface/policy 与官方 demo 配置 SHA 前后不变：同目录 `runtime_validation.json`。
- 原始训练 EEF/动作/RGB 核对：`outputs/analysis/piper_eval_contract_20261104/raw_contract_checks.json`。
- 两条真实训练 episode 的 `native_fixtures.json`，通过实际 `merge_deploy_cfg`→`ObsPreprocessor`，与实际 `PreparedSimDataset.__getitem__` 未增强首帧逐像素比较：两条均 `384×320×3`、maxabs=0、不同通道值数=0。旧 single-view 路径相对训练首帧分别有 312275/364046 个通道值不同，不能用于此次 Piper 成绩。证据：`outputs/analysis/piper_eval_contract_20261104/multiview_canvas_check.json`。

复查命令：

```bash
.venv/bin/python -m pytest -q tests/benchmarks/test_piper_robotwin_eval.py
CHECK_ONLY=1 EPISODES=30 OUTPUT="outputs/robotwin_eval/piper_plan_$(date -u +%Y%m%dT%H%M%SZ)" \
  bash scripts/eval_robotwin_piper.sh
.venv/bin/python benchmarks/robotwin/check_piper_observation.py \
  --fixtures outputs/robotwin_eval/piper21952_smoke_20261104/native_fixtures.json \
  --references outputs/analysis/piper_eval_contract_20261104/raw_contract_checks.json \
  --output outputs/analysis/piper_eval_contract_20261104/multiview_canvas_check.json
```

初始 GPU6/T5 GPU3 试装显存不足；随后 GPU2/T5 GPU3 加载成功，但服务日志暴露出默认 single-view 忽略两个腕视角。初始输出及其 retry 均保留、排除，0 有效闭环条。修复仅涉及专用 deploy 配置，没有改共享 server、模型或 Aloha。正确 multiview 配置的小试根目录为 `outputs/robotwin_eval/piper21952_smoke_multiview_20261104`。

## GPU 小试记录

2026-11-04 04:54 UTC，正确配置的服务已在 L20 主 GPU2 / T5 GPU3 成功加载；两端口实际日志均确认 `multiview=True`、layout `['0','1','2']`、384×320，加载的配置 SHA 与 CPU 像素检查一致。Clean 和 fresh-clutter 两条真实训练输入均返回有限 `32×20` 动作，reset 后重复逐元素一致。首次冷请求 5.49 秒，随后三次请求 0.49/0.52/0.63 秒；不是单 rollout 耗时。

证据为上述正确 multiview 输出根目录内 `fixture_acceptance.json`、`model_server.log`、`trial_status.json`。GPU2 后续被其他作业占用，未启动 policy rollout 的本次 GPU2 仿真已结束，改在空闲 L20 GPU5 跑 `click_bell` 两域各一条；结果写入 `native_click_bell_gpu5/`。

两域小试均已完成，独立重读原生日志和 `_result.txt` 校验通过：

| Task | Domain | 成功数 / 条数 | 动作步数 / 官方预算 | Accepted seed |
|---|---|---:|---:|---:|
| click_bell | demo_clean | 1/1 | 40/400 | 100000 |
| click_bell | demo_randomized | 1/1 | 34/400 | 100000 |

两条均 exit0，原始分母、unseen 指令、accepted seed、无明确 renderer 错误通过；没有视频文件。Clean 41次 get_obs 中38次省去重复 RGB，Randomized 35次中32次省去重复 RGB，分别仍保留3次完整观察。最终证据为 `final_smoke_audit.json`。Aloha 五份默认入口/配置 SHA 与本次工作前完全一致。

服务仍保留在 L20 GPU2/T5 GPU3，PID4119493，本次实际有 **43400、43401 两个会话**；复用时仅使用这两端口。上方四端口示例需按 `--slots 4` 单独启动相应服务。仿真小试已结束，**没有启动35任务全量评估**，2/2不能外推总体成功率。当前 official curobo 的这两条闭环已验证，35任务覆盖、完整成功率及与历史 mplib_RRT 的一致性仍待正式评测。


2026-11-04T09:24:23.987826+00:00：用户要求停止Piper测试部署。已停止本次Piper21952 PID4119493（主GPU2/T5 GPU3），43400/43401均不再监听；日志、权重配套与两域小试结果保留。三份Aloha模型PID4052817/4061238/4113133仍存活。此前“Piper服务保留待命”状态至此结束。
