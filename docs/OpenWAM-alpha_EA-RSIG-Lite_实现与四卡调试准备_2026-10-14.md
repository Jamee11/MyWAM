# OpenWAM-alpha EA-RSIG-Lite：实现与四卡调试准备

2026-10-14 UTC。对应方案为 [OpenWAM-alpha_EA-RSIG-Lite_2026-10-14.md](OpenWAM-alpha_EA-RSIG-Lite_2026-10-14.md)。本次交付范围是实现调整、CPU 验证和可启动的调试入口；按用户要求，停在实际四卡调试之前。

## 当前结论与资源边界

实现和 CPU 准备已完成；四卡训练可行性、稳态速度、训练效果均未验收。现有证据证明旧版双卡 A100-80GB、no-offload 全参配置在第二步申请 FP32 flat-gradient 缓冲区时 OOM，不能据此断言 OpenWAM 无法微调。旧版首步耗时与 native14/40-step 结果不能用于 Lite 的训练排期。CPU optimizer offload、更多 GPU 和部分解冻分别需要实测；部分解冻改变可训练集合，不能声称等价复现官方全参微调。

官方基线 commit 为 `4d7f642b841913ea3c176010b692b1de3d5ef180`。Foundation 使用本地 `OpenWAM-Alpha-Pretrain-Foundation-Model`，revision `52df4e66c82c5c8b480adcc8d01f4db7415dfb56`，主权重 SHA256 `180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d`。下载完整性已验收，后续入口使用离线模式。

## 方案到实现

| 方案要求 | 当前实现 | 验证边界 |
|---|---|---|
| 保留 Foundation action/state 80D | 独立 `dual_system_ea_rsig_lite` 架构，32 个未来 action、33-state window、9 视频帧 | 实际完整 Foundation CPU 加载：2089 tensors 继承，未重置任何 policy head，仅增加 85 个 EA tensors |
| 当前观测、无额外 DiT observation pass | `first_frame_latents` 经原 patchify，按三相机分格、轻投影至 512D，加指令 FiLM | tiny 真实 Wan 检查不调用 observation DiT block；未新增 H18、未来视频 residual 或第二次 future forward |
| 15×512 EA 条件 | 复用 V3 Ground×4、Relation×4、Progress×1、Plan×6，保留相机空间网格 | 角色/无效目标/时间标签测试及实际网格预览通过；不是已训练定位效果 |
| Action hidden 单点条件化 | 第一个 mixed block 后，Action query 对 EA tokens 做独立 cross-attention，输出投影零初始化 | 原 joint attention、RoPE、mask、主序列长度与 FM 目标保持；官方 baseline 类文件未改 |
| 禁止 privileged inputs | EEF/双臂存在标志与当前 RGB/指令是输入；130D `ea_targets` 独立仅供 loss | T2 独立扰动未来视频、动作、noise、标签；生成入口拒绝 targets |
| 损失与阶段 | 原 video/action FM + ground .025 / relation .01 / plan .05；warmup/joint/action_only | CPU 两步 warmup 与 joint 梯度测试；尚无完整模型 GPU optimizer 更新 |
| A/B/C 对照 | A 原架构；B auxiliary-only；C 条件化；同 reader、数据、统计与增强 | B joint 辅助监督可达共享 patch embedding；warmup 冻结主干时不能作此解释；未用 Randomized 数据训练 |
| 可恢复与诊断 | 官方 trainer 入口，新增 EA loss/mask/gate/梯度、每步时间/峰值显存 JSONL；保存后计划退出 | CPU 实际两次 Accelerate 保存/恢复，权重、下一步 loss、LR 一致；不是 ZeRO 四卡恢复验收 |
| 部署接口 | 独立 deploy YAML、RoboTwin policy YAML、离线 chunk 检查脚本 | 配置加载/engine 参数路由/EEF 解码 CPU 检查通过；新 Lite 完整模型推理与控制闭环未跑 |

代码入口：`openwam/model/modules/ea_rsig_lite.py`、`openwam/model/architectures/dual_system/ea_rsig_lite.py`、`openwam/dataloader/ea_rsig_lite.py`。旧 V3 文件和证据保留；旧 `train_ea_rsig_v3.sh` 默认阻止误启动，只有显式 `ALLOW_LEGACY_EA_V3=1` 才运行。官方 `scripts/train.sh` 保留。

共享修改仅涉及注册、可选 EA 上下文透传、可选附加指标及 full-state 保留开关。`keep_final_resume_state` 默认 false，只有本次调试配置启用，避免官方结束清理删除恢复状态。原 baseline 四项指标运算保持。无用 legacy video 参数冻结，B 的 action/token-only 参数也冻结，避免无梯度训练参数。

## 数据与物理动作合同

原始数据只读：`/remote-home/jinminghao/datasets/robotwin2_ea_rsig_v3_clean50_v6_raw_20260921`。已生成 `outputs/ea_lite/data/manifest.json`、每轨迹 sidecar 与 Clean-only `normalization_stats.npy`：50 任务、2500 episodes、557651 源帧，官方 t+1 reader 得到 555151 个窗口（每条轨迹最后一帧不作起点）。这是重建训练索引、监督和归一化资产，未改写源 HDF5。

1. 读取现成 raw endpose，无需从 14D qpos 猜测 EEF。原采集四元数为 **wxyz**，官方 reader 转换函数接收 xyzw；Lite 在读入边界重排，模型保持官方列式 Rot6D。client 同样转换，执行边界转回该版 RoboTwin controller 的 wxyz。
2. 动作是原 RoboTwin endpose 公共参考坐标系中的 **absolute EEF targets t+1…t+32**，20D 有效槽映射进 80D（0–9、34–43），不改为 delta/chunk-relative。不能未经变换称为每只机械臂自身 base frame。真机需明确 workspace/base 刚性变换。
3. EA 所需 TCP context 由当前可观测 EE endpose 沿局部 X 加 **0.12m** 得到。全数据生成时逐帧与 oracle TCP 的位置、姿态核对，但模型输入存储的是观测推导结果；双臂存在标志为物理双臂常量。不能把 oracle EEF 直接作为可部署输入。
4. Plan 从 raw v6 当前角色身份、pose、valid 重新计算 **8/16/32**，沿当前角色局部坐标；尾段不越界，角色 ID 改变屏蔽，moving 阈值 1mm。未插值或截取旧 40-step targets。
5. 保留 V3 typed relation、contact/near、0.5m proximity 尺度与任务目标有效性规则。BBox 用每个 view 自身宽高归一化；虚拟 goal 不伪造可见框。130D 仅复用监督布局，不拼入 proprio。
6. A/B/C 使用同一统计、绝对 EEF、相机 L-canvas、增强设置（当前关闭 color jitter）。夹爪由 raw 0/1 走官方 min-max，部署反归一化保留 EEF 连续约定，未沿用旧 native14 binary patch。

## 已完成的证据

证据目录为 `outputs/ea_lite/readiness/`，所有本轮 GPU runtime 状态均为未执行。

| 证据 | 结果与范围 |
|---|---|
| `foundation_cpu.json` / `.log` | 完整 Foundation CPU 严格暖启动，80D heads 未 reset、无 meta 残留、zero-out 正确；总参数 12420900941，C warmup 可训练 13687328。不是仅加载 tiny/random 模型 |
| `preflight.json` | 50/2500 源 size/mtime、全部 sidecar SHA、统计 SHA、权重记录与 80D header 通过 |
| `data_audit.json` | 构造全部 50 readers；起点/中点/末点真实 RGB 窗口抽查，32×80、9 帧、反归一化回 raw、train/client TCP 相同、A/C 输入相同；不是遍历解码全部 RGB |
| `camera_bbox_grid.png` | 真实三相机 canvas、14×14 每 view 网格和 GT bbox 投影已人工查看；不是 trained attention 或定位精度 |
| `cpu_tests_acceptance.log` | **62 passed, 1 warning, 20.70s**。warning 为热进程中 anyio 已导入导致 pytest 无法重写其 assert，不涉及模型数值。此前冷启动完整回归及中间增量回归日志保留 |
| `config_A/B/C.yaml` | 实际 CHECK_ONLY Hydra resolve 成功，错误输出为空；默认启动器不申请 GPU |
| `sidecar_audit.json` | 全部生成标签的数值有限性、双臂常量与 v6 collection contract 复核，见实际报告 |

CPU 回归包含 T1 FP32（关闭/零初始化、训练 checkpoint 开/关）及 BF16 零初始化 identity；T2 独立因果扰动；T3 train/eval 条件一致和当前 RGB/文本/EEF 敏感性；两次 AdamW 更新梯度连通、joint 双流梯度、eval inference-mode 不保留图；两次真实 CPU 保存恢复；绝对 EEF/四元数/controller 边界数值往返；尾段/角色/虚拟 goal；四 GPU 可用性门槛的 mock 检查。四卡 mock 和 tiny 数值检查不替代完整 BF16/ZeRO 实测。

## 四卡到位后的启动与停止协议（本轮不执行）

当前可安全检查配置：

```bash
cd /remote-home/jinminghao/OpenWAM
CHECK_ONLY=1 EA_EXPERIMENT=A bash scripts/train_ea_lite_4gpu.sh
CHECK_ONLY=1 EA_EXPERIMENT=B EA_STAGE=joint bash scripts/train_ea_lite_4gpu.sh
CHECK_ONLY=1 EA_EXPERIMENT=C EA_STAGE=warmup bash scripts/train_ea_lite_4gpu.sh
```

默认 CHECK_ONLY=1。只有显式 `CHECK_ONLY=0` 才进入训练，并要求四个不同的物理 A100 ID；80GB profile 每卡至少约 70,000MiB 空闲，40GB profile 至少 35,000MiB，CPU 数据预检前后各查一次。这只是资源门槛，不是 GPU 预约或能装下模型的证明。`EA_GPU_PROFILE=40` 强制 CPU optimizer offload；80 profile 可用 `EA_OFFLOAD=cpu`，均尚未实测。

先做 A 的 G0，失败立即处理容量，不长时间盲跑。以下是未来四卡均空闲时的例子，**将 0,1,2,3 换成实际分配卡号**：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export EA_EXPERIMENT=A EA_GPU_PROFILE=80
EA_STOP_AFTER_OPT_STEPS=5 CHECK_ONLY=0 bash scripts/train_ea_lite_4gpu.sh
# 从日志取得实际 run dir（含时间戳），填入后再恢复；不要猜路径。
export EA_RUN_DIR=/absolute/path/to/the/actual/run_dir
EA_STOP_AFTER_OPT_STEPS=10 CHECK_ONLY=0 bash scripts/train_ea_lite_4gpu.sh \
  training.finetune_ckpt_path=null training.resume_ckpt_path="$EA_RUN_DIR"
CHECK_ONLY=0 bash scripts/train_ea_lite_4gpu.sh \
  training.finetune_ckpt_path=null training.resume_ckpt_path="$EA_RUN_DIR"
```

三次进程保持相同 `max_steps=160`、accum=8、数据、阶段与 LR horizon，依次在 5/10/20 optimizer steps 保存，完成两次真实恢复。`max_steps/save_steps` 是 micro-step，默认保存间隔 40 micro=5 updates；每卡 bs1×四卡×累积8=global32。这是调试预算，与论文 global256、5 epochs 不等价。不要通过把 max_steps 从40改成80来伪造恢复验证。保持同一 EA_EXPERIMENT/EA_STAGE/profile；warmup→joint 是新 run 的 finetune 初始化，不能直接恢复一个不同 trainable set 的 optimizer。

A 通过后，再用 `EA_EXPERIMENT=C EA_STAGE=warmup` 做同协议；检查第一步 zero-out 上游梯度应为零、第二步条件路径梯度出现，不能把初始化性质误判为断梯度。后续 joint 或 action_only 必须再测各自容量；warmup 通过不证明全参 joint 能跑。方案中的 100–300 步 warmup 是后续 sanity 预算，当前默认 20 updates 只是资源/接口调试。

每 rank `diagnostics_rank*.jsonl` 记录 micro/optimizer step、bs/world/accum、loss/有效 mask、selected gradient norms、逐 micro 时间与峰值 allocated/reserved。排除首轮初始化和 checkpoint 保存边界，分别统计 steady micro-step、完整累积更新、每秒样本；正式数据量按实际窗口数和样本曝光量排期。没有这份稳态证据前不填写训练天数或最大 bs。

得到训练权重后，未来执行完整 Lite 离线部署检查：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/smoke_ea_lite_deploy.py \
  "$EA_RUN_DIR" --device cuda:0
# 部署 server 使用独立 YAML（服务端直接 OmegaConf.load，不解析 Hydra defaults）。
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=0 bash scripts/deploy.sh \
  "$EA_RUN_DIR" --config configs/deploy_ea_rsig_lite.yaml
```

离线脚本检查 strict reload、当前观测且无 oracle、两 denoise steps、有限 32×20 物理 EEF，并计时；不发机器人命令。RoboTwin 客户端用 `benchmarks/robotwin/policy_config_ea_rsig_lite.yml`（state20/action_type ee/wxyz）。真实 IK 跟踪、环境闭环及真实 denoise 预算下延迟另行验收。

## 未完成的门槛

- 四卡完整 forward/backward/AdamW 多步、两次 ZeRO 保存恢复、full-model BF16 identity、offload 编译与吞吐、最大 batch、稳态峰值显存和时间。
- Lite 完整 checkpoint 生成/reload/inference、当前观测到 controller/IK 的真实轨迹误差与闭环；旧 V3 的 40×14 部署证据不算 Lite 通过。
- 训练后的 grounding/plan quality、token intervention、Clean/Randomized 配对 SR、多 seed、matched-budget A/B/C 与性能退化评估；不能保证适配不会明显影响效果。
- 真机 frame/本体/延迟/安全与 sim2real 证据。未新增 DINO、domain consistency 或 future relation transition；是否扩展由 G3 结果决定。

正式对照继续共享同一 Foundation 与 Clean 数据；warmup 额外曝光需单独计入预算，B 必须在共享层可训练的 joint 阶段解释辅助监督作用。通过上述实现准备不代表达到原方案第14节的研究成功条件。
