# ManiSkill3 六任务评估：启动流程与结果

更新日期：2026-10-22 UTC。适用项目：`/remote-home/jinminghao/OpenWAM`。

## 1. 当前结论

Baseline 14K 和 Sidecar 25410 均已完成六任务、每任务100条测试。
Baseline 成功576/600（96.00%），Sidecar 成功587/600（97.83%），
提升1.83个百分点、多成功11条，主要增益来自 PlaceSphere。
这是相同评估协议下两个固定checkpoint的对照，不是严格等训练预算的结构消融。

## 2. 评估设置与流程

| 项目 | 本次设置 |
| --- | --- |
| 环境 | ManiSkill3，自定义 `*EORTColorDistractors-v1` 六任务 |
| 任务 | PushCube、PickCube、StackCube、PlaceSphere、PickCuboid、PickSphere |
| 测试场景 | 原采集test seed manifest中，专家成功的前100个seed/任务；两模型逐任务seed顺序相同 |
| 样本量 | 每任务100条，共600条；每任务50+50互斥分片，共12路仿真 |
| 物理/渲染 | `physx_cpu` / `sapien_cuda`，默认shader |
| 模型输入 | front+wrist RGB、任务文本、真实机器人base-frame TCP10状态 |
| 图像布局 | 训练同款H384×W320 L形画布；不额外输入目标物位姿、分割或EA标签 |
| 模型动作 | 80D统一空间中有效的单臂EEF10；normalizer完成归一化与反归一化 |
| 执行桥接 | 绝对base-frame位置、column-Rot6D转XYZ Euler、夹爪[0,1]转[-1,1]，原生 `pd_ee_pose` |
| 推理预算 | 10步去噪，生成32动作，每执行8动作重新规划；固定推理seed42、CFG1 |
| episode预算 | 最多200控制步，沿用环境成功判据 |
| 加速 | joint velocity cache：cosine阈值0.99、max_skips=3；启用编译 |
| 结果 | 原始逐episode JSONL、H.264视频、分片/任务/全局summary |

统一入口为 `benchmarks/maniskill/run_eval.py`，流程如下：

1. 检查checkpoint、端口、seed清单和仿真client依赖；固定权重及config/stats/tokenizer快照。
2. 加载模型服务；每个仿真worker使用独立端口、policy和动作队列，共享模型推理。
3. 验证专家控制器回放，或显式复用此前通过的六任务 `expert_replay.json`。
4. 调用同一个 `single_eval.py` 执行闭环，写逐episode结果和视频。
5. 核对分片episode编号和seed覆盖，完整后才计入任务及最终成功率。
6. Sidecar额外核对baseline输入契约、normalization stats、seed及预算，并输出逐seed对照。

专家回放只验证执行合同，不是策略成绩。本次复用报告验证了六任务各一条轨迹、
原delta与转换后的绝对命令共12次回放均成功，最大TCP位置差约0.154毫米。
测试集经过专家成功筛选，结论限于此held-out color-distractors分布，而非任意随机场景。

## 3. 启动前准备

以下是重新完整评估的指令，不会启动训练。先进入项目并检查GPU和端口占用：

```bash
cd /remote-home/jinminghao/OpenWAM
nvidia-smi
```

- 仿真Python：`/remote-home/jinminghao/miniconda3/envs/maniskill-eort-v1/bin/python`。
- ManiSkill源码：`/remote-home/jinminghao/WAMs/ManiSkill`。
- 模型Python：baseline使用项目 `.venv/bin/python`；Sidecar正式运行使用下列已有runtime。
  `/dev/shm`环境重启后可能消失，迁机/重启后须替换为已验证的持久环境，不能只改成任意Python。
- 原test seed manifest、Foundation依赖、tokenizer、checkpoint相邻config/stats必须存在。
  runner使用离线模式，不会自动下载；镜像优先规则继续适用。
- 输出必须是新目录，runner拒绝覆盖已有目录；重复执行会重跑全部600条，不是续跑命令。
- GPU5/6需有足够空闲显存；Sidecar服务放置前等待至少33GiB空闲。
  不要为了启动评估结束其他训练/未知进程。

## 4. Baseline启动指令

复用已固定的14K快照，避免原训练目录keep-last删除旧权重。
原训练来源：
`outputs/maniskill_baseline/six_task_bs64_10epochs_20261017T050252Z/runs/2026-10-17_05-16-28/checkpoint_step_14000.safetensors`。

```bash
cd /remote-home/jinminghao/OpenWAM
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
MODEL_PY=/remote-home/jinminghao/OpenWAM/.venv/bin/python

"$MODEL_PY" -u benchmarks/maniskill/run_eval.py \
  --policy baseline \
  --checkpoint outputs/maniskill_eval/baseline14000_six_task_100_replan8_max200_gpu56_20261019/checkpoint/checkpoint_step_14000.safetensors \
  --output "outputs/maniskill_eval/baseline14000_six_task_100_${STAMP}" \
  --model-python "$MODEL_PY" \
  --gpus 5 6 --workers-per-task 2 \
  --episodes 100 --max-steps 200 --replan-every 8 --denoise-steps 10 \
  --port 21700 \
  --expert-report outputs/maniskill_baseline/expert_replay.json
```

GPU5/6各加载一个模型、各运行6路仿真；使用21700–21711端口。
这是已完成baseline的部署布局。固定快照为只读用途，不要原地修改权重。

## 5. Sidecar启动指令

保持正式Sidecar评估的布局：GPU6一份模型，GPU5运行12路仿真。

```bash
cd /remote-home/jinminghao/OpenWAM
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
MODEL_PY=/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv/bin/python

"$MODEL_PY" -u benchmarks/maniskill/run_eval.py \
  --policy sidecar \
  --checkpoint outputs/maniskill_sidecar/joint_20261019T222758Z/2026-10-19_22-33-43/checkpoint_step_25410.safetensors \
  --output "outputs/maniskill_eval/sidecar25410_six_task_100_${STAMP}" \
  --model-python "$MODEL_PY" \
  --baseline-reference outputs/maniskill_eval/baseline14000_six_task_100_replan8_max200_gpu56_20261019 \
  --gpus 6 --sim-gpus 5 --workers-per-task 2 \
  --episodes 100 --max-steps 200 --replan-every 8 --denoise-steps 10 \
  --port 21980 \
  --expert-report outputs/maniskill_baseline/expert_replay.json
```

使用21980–21991端口。Sidecar三处tap（7/15/23）完整启用，不绕过read/write分支。
复用既有FFN编译路径，并在原生ManiSkill RGB/EEF10 fixtures上先通过数值及真实kernel门槛。
本次四组编译/匹配缓存参考的最大误差均0；这不等于证明缓存相对无缓存计算无损。

需要重新执行专家gate时，省略 `--expert-report`，由runner执行回放。
以上为前台命令，长时间运行请置于持久终端；两套完整评估建议依次运行，避免争抢显存。

## 6. 已完成结果

每个任务均为100条，成功率差值单位为百分点；平均步数包含200步超时失败。

| 任务 | Baseline成功 | Sidecar成功 | 差值 | Baseline平均步数 | Sidecar平均步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PushCube | 100/100 | 100/100 | 0 | 55.38 | 52.53 |
| PickCube | 96/100 | 98/100 | +2 | 84.74 | 79.25 |
| StackCube | 93/100 | 93/100 | 0 | 105.30 | 109.37 |
| PlaceSphere | 93/100 | 100/100 | +7 | 134.55 | 116.15 |
| PickCuboid | 96/100 | 97/100 | +1 | 97.41 | 94.67 |
| PickSphere | 98/100 | 99/100 | +1 | 83.41 | 78.77 |
| **合计** | **576/600（96.00%）** | **587/600（97.83%）** | **+1.83** | — | — |

逐seed对照：Sidecar独有成功18条，baseline独有成功7条，净增11条。
StackCube虽然总成功率相同，但有3条baseline独有成功和3条Sidecar独有成功，
不是逐例结果相同。Baseline的24条、Sidecar的13条策略失败均耗尽200步。

主要结论：PlaceSphere贡献净增成功的7/11，其余增益较小；StackCube仍有7条失败，
是Sidecar当前主要剩余项。不能仅凭超时记录判断是目标选择、抓取还是恢复失败。
步数不是墙钟延迟；两种GPU布局和负载不同，不能据此比较吞吐。

## 7. 权威产物与验收口径

结果根目录（相对项目根）：

- Baseline：`outputs/maniskill_eval/baseline14000_six_task_100_replan8_max200_gpu56_20261019`。
- Sidecar：`outputs/maniskill_eval/sidecar25410_six_task_100_replan8_max200_matched_20261022`。

| 文件 | 用途 |
| --- | --- |
| `summary.json` | 最终总成绩、逐任务成绩及逐episode记录；两次均 `complete=true`、600条 |
| `policy/<task>/shard_*/episodes.jsonl` | 实时逐episode记录，含seed、success、steps |
| `policy/<task>/shard_*/episode_*.mp4` | 对应episode视频 |
| `run.json`、`deploy.yaml`、`checkpoint/` | 权重/源码/配置指纹、实际推理配置、固定模型快照 |
| Sidecar `fairness.json` | 输入契约、统计、seed与训练计数差异 |
| Sidecar `comparison.json`、`comparison.md` | 与baseline的逐任务、逐seed对照 |
| Sidecar `server_gpu6_manifest/validation/result.json` | 编译数值检查证据 |
| Sidecar `server_gpu6_manifest/sidecar_contract.json` | Sidecar分支、gate及实际推理设置 |

最终验收已核对：每任务100个不同episode编号及seed、汇总/原始成功数一致，
两次各600个非空视频，无 `failure.json`，评估协调器和模型进程已结束。
未逐个解码全部视频，也未逐条做失败行为诊断。
运行中顶层summary只聚合完整任务，应从JSONL统计实时进度，不能把暂缺汇总当作0条。

## 8. 对比边界与下一步

- 输入、normalization stats、test seeds、动作桥接、200步预算和推理参数已对齐。
- Baseline为14000次optimizer更新；Sidecar文件名25410是微步计数，实际
  `opt_step=12708, epoch=5`，不是25410次更新。
- Baseline使用完整MoT编译，Sidecar使用保留完整Sidecar计算的FFN编译；
  GPU并发部署亦不同。因此只能报告本次匹配评估协议下的checkpoint成绩，
  不把全部增益单独归因于Sidecar结构，不声称统计显著性或真机迁移收益。
- 下一步优先查看StackCube等失败视频；若需方法因果结论，应做等训练预算、
  多次重复及匹配部署优化的独立对照，不覆盖本次结果。

历史过程记录见 `MANISKILL_BASELINE_EVALUATION_20261019.md` 和
`MANISKILL_SIDECAR25410_EVALUATION_20261022.md`；其中启动时的“待完成”段落是历史状态，
最终状态以本说明及两次正式run的 `summary.json` 为准。
