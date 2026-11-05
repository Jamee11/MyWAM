# ManiSkill 八任务评估：迁机、环境和启动命令

适用：当前从 midtrain 初始化、开启 Sidecar＋PatchWAM 的 ManiSkill 微调模型。
新统一入口：`scripts/eval_maniskill_archive.sh`。
以下操作在目标机器执行；不需要恢复优化器，也不需要启动训练。

## 日常启动：复用已通过的验收

已落实AGENTS.md第12条：同一数据、环境、模型结构及推理代码，复用已经通过的清单、24场景桥接和FFN数值对照；不每次重跑。每份新权重仅做正常加载与一次原生输入推理，确认32×10动作有限，保持FFN＋cache。编译缓存也复用。当前运行的评估不重启。

本机已注册通过记录：`outputs/maniskill_eval/validated_run.txt`指向最终20872的成功启动目录。入口自动读取，无需手工复制run目录脚本。以后只换checkpoint与GPU参数即可：

```bash
CHECKPOINT=/path/to/checkpoint.safetensors MODEL_GPU=3 SIM_GPU=0 WORKERS=4 \
  bash scripts/eval_maniskill_archive.sh
```

同目录保留config/stats/tokenizer配套。SIM_PY未指定时优先项目`.venv-maniskill`，其次本机已有`maniskill-eort-v1`。继续不录像，保留策略RGB。默认直接尝试模型显存分配，不再卡33GiB固定门槛；如确实希望等待可显式设置`MANISKILL_MIN_FREE_GIB`。

只有数据版本、模型结构、推理/控制代码或环境变化时，入口才重新执行完整验收一次；单纯权重数值变化可复用。可以用`VALIDATED_RUN=/path/to/previous/run`指定已通过的参考，用`REVALIDATE=1`主动要求完整重验。结果覆盖/seed身份及实际渲染错误检查仍保留，因为它们直接影响成功率可信度。

下面各节的24场景与完整FFN流程适用于首次部署或关键条件改变，并非每次必跑。

## 1. 评估具体做什么

归档提供固定任务、语言、seed、domain、hard_index 和初始仿真状态。
每个 episode 只在开始时恢复初始状态；之后执行模型预测动作，让仿真闭环演化，
按环境原生 success 判定成功。不是沿着专家动作回放来计算模型成功率。

启动时另做专家控制回放，用于确认绝对 EEF 桥接；它不计入策略成功率。

| 项目 | 固定协议 |
|---|---|
| 任务 | push_cube、pick_cube、pick_cuboid、pick_sphere、stack_cube、place_sphere、pull_cube、lift_peg_upright |
| Clean | 每任务100条；先取全部62/63条test，再补38/37条val；8任务合计800 |
| HardRandom | 每任务100条test，hard D 25条＋hard M 75条；合计800 |
| 总量 | 1600条；Clean标为held-out val＋test，不能标成纯test |
| 数据隔离 | midtrain和本次微调均只用train；val/test未参与归一化统计；原核查记录与manifest SHA对应 |
| 环境 | 归档内Franka20K-v1 / FrankaHard10KCameraV2-v1；恢复各episode真实domain和hard_index |
| 控制预算 | 各原生metadata的600控制步，不套用旧六任务200步协议 |
| 模型输入 | front＋hand RGB、任务语言、实测base-frame TCP；无object pose/分割/EA标签输入 |
| 图像 | H384×W320的L形画布，第三视角黑色；评估不启用训练PatchWAM增强 |
| 输出执行 | 绝对base-frame TCP XYZ＋column-Rot6D＋gripper，经原桥接转pd_ee_pose |
| 推理 | 10步去噪，预测32动作，每执行8动作重规划，CFG1 |
| 加速 | 原joint velocity cache .99/max3＋Sidecar局部FFN编译；保留数值验证 |

原训练归档manifest SHA256：
`a96d7f13dcb243f9429303a667d76d02151098a83c2946d996a31b21bf4a07aa`。
原隔离证据：`outputs/maniskill_eval/archive_suite_20261102/training_exclusion_audit.json`。
部分外观版本共享物理family，但训练与本次held-out评估的family/seed无交集。
Clean与hard本身不是逐场景配对；不同checkpoint对比时应复用相同评估清单。

## 2. 目标机要有什么

### 项目代码

需要当前版本OpenWAM源码，尤其是下面这些新增/相关文件；只复制旧版GitHub源码可能缺少它们。

| 文件 | 职责 |
|---|---|
| scripts/eval_maniskill_archive.sh | 路径检查、重建本机清单、统一启动 |
| benchmarks/maniskill/archive_protocol.py | 选择held-out样本，恢复原生场景 |
| benchmarks/maniskill/archive_eval.py | 专家回放、部署副本、单路闭环 |
| benchmarks/maniskill/run_archive_eval.py | 模型服务、分片、重试、完整结果汇总 |
| benchmarks/maniskill/serve_sidecar.py | 完整Sidecar服务、原生fixture、FFN数值检查 |
| benchmarks/maniskill/single_eval.py | 原绝对EEF/相机/TCP桥接工具 |
| benchmarks/robotwin/sidecar_fast_runtime.py | Sidecar兼容cache与局部FFN运行时 |
| benchmarks/robotwin/check_sidecar_fast_runtime.py | 编译与匹配cache参考验证 |
| benchmarks/robotwin/serve_parallel.py | 一份模型、多个独立动作队列/端口 |

同时需要项目的 `openwam/`、`configs/`、`benchmarks/utils/` 和对应Python依赖。
建议同步整套源码，而不是只传入口脚本。不需要携带训练outputs中的其它大权重或optimizer state。

### 最终权重及配套文件

本机已经完成的ManiSkill最终run：

```text
/remote-home/jinminghao/OpenWAM/outputs/maniskill_sidecar/midtrain_patchwam_1epoch_20261031T184131Z/runs/2026-10-31_18-46-16/
```

从这个目录保留以下内容，放到目标机同一个checkpoint目录，例如`assets/maniskill_final/`：

```text
checkpoint_step_20872.safetensors
config.yaml
normalization_stats.npy
relational_sidecar_contract.json
tokenizer/
```

权重大小24,862,363,126 bytes；20872是microstep，对应10436次参数更新。
这份权重包含完整模型，组件规格在config中；不需要重新下载Foundation或midtrain模型。
不要改训练原件的stats/map；runner会自动生成单臂部署副本。
同一文件系统优先硬链接固定权重，跨文件系统自动复制，后者需要额外约25GB空间。

### 数据与仿真源码

默认数据目录：

```text
<PROJECT>/datasets/multisim_sidecar_archive_v2_20261025/
```

可以通过`DATA_ROOT`指定其它位置。共享盘可直接读取，无需再拷贝。
必需的内容包括：manifest、选中episode的media/numeric和邻接JSON、
`replay/ManiSkill/`（含assets）、`replay/DiT4DiT/examples/ManiSkill3/`。
**必须使用归档中的自定义ManiSkill源码**，不能只装上游最新版来替代这些环境。

迁机后不要直接复用带旧`archive_root`的plan。统一入口会从目标机DATA_ROOT重建plan，
保持episode ID/seed/数量不变。manifest中的source_h5等旧绝对路径是来源记录；
实际回放读取本机archive_root下的相对media/numeric路径，不需要全局替换来源路径。

若不是共享存储、只想传评估所需数据，可在源机器生成文件清单：

```bash
cd /remote-home/jinminghao/OpenWAM
python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/maniskill_eval/archive_suite_20261102/plan.json')
plan = json.loads(p.read_text())
root = Path(plan['archive_root'])
files = {'manifest.ready.json', 'training/normalization_stats.npy'}
for r in plan['cases']:
    files.update([r['media'], r['numeric'], str(Path(r['media']).with_suffix('.json'))])
for relative in ['replay/ManiSkill', 'replay/DiT4DiT/examples/ManiSkill3']:
    files.update(str(q.relative_to(root)) for q in (root/relative).rglob('*') if q.is_file())
init = root/'replay/DiT4DiT/examples/__init__.py'
if init.is_file():
    files.add(str(init.relative_to(root)))
Path('outputs/maniskill_eval/archive_eval_files.txt').write_text('\n'.join(sorted(files))+'\n')
print('files:', len(files), 'episodes:', len(plan['cases']))
PY
```

用这份清单做rsync/SFTP归档，必须保留相对于数据根目录的层次。
例如rsync的源根是`.../multisim_sidecar_archive_v2_20261025/`，
参数使用`-aLr --files-from=outputs/maniskill_eval/archive_eval_files.txt`。
不需要拷贝其余仿真训练集或原source.h5；这不是整个数据集的完整备份。

## 3. 两个Python环境

建议分开模型环境与仿真环境，允许同机使用不同Python：

```bash
cd /目标机器/OpenWAM
export MODEL_PY="$PWD/.venv/bin/python"          # 已能运行当前OpenWAM的环境
export SIM_PY="$PWD/.venv-maniskill/bin/python"  # 或目标机现有ManiSkill conda环境
export DATA_ROOT="$PWD/datasets/multisim_sidecar_archive_v2_20261025"
export CHECKPOINT="$PWD/assets/maniskill_final/checkpoint_step_20872.safetensors"
```

已有训练/推理环境优先复用，不要每次评估都安装依赖。
本机验证所用的关键版本为Python3.10、PyTorch2.7.1+cu128、SAPIEN3.0.3，
仿真环境NumPy1.26.4、Gymnasium1.2.3、SciPy1.15.3、websockets16.0。
模型环境的NumPy是2.2.6，所以不要直接把两套环境混装。

如果目标机没有仿真环境，下面是一次性安装命令；它不修改模型`.venv`：

```bash
python3.10 -m venv .venv-maniskill
export SIM_PY="$PWD/.venv-maniskill/bin/python"
"$SIM_PY" -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
"$SIM_PY" -m pip install \
  'numpy==1.26.4' 'sapien==3.0.3' 'gymnasium==1.2.3' \
  'scipy==1.15.3' 'h5py==3.16.0' 'Pillow==12.2.0' \
  'websockets==16.0' 'transforms3d==0.4.2' \
  'opencv-python==4.11.0.86' 'tyro==1.0.15' 'matplotlib==3.10.9'
"$SIM_PY" -m pip install -e "$DATA_ROOT/replay/ManiSkill"
```

安装命令按当前已验证版本写出，未在目标机新建环境实跑。
模型环境若尚未建立，项目已有`scripts/cluster/install_environment.sh`及锁定依赖；
已有可用模型环境不需要重建。GPU驱动/Vulkan必须支持SAPIEN渲染；只有CUDA训练可用不代表图形驱动齐全。

本机驱动需要额外预加载graphics compiler，runner会识别本机已知570库。
其它驱动若需要，可指向目标系统自己的对应库：

```bash
export MANISKILL_GPUCOMP_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.目标驱动版本
```

不要复制另一台服务器的驱动库覆盖目标系统。显式指定的库不存在时脚本会报错。
同机若有多个Vulkan ICD且驱动选择错误，可在确认文件存在后设置
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`；runner会继承该设置。
HF模型加载默认离线、mirror端点优先；本流程不自动下载缺失模型或绕过镜像。

## 4. 双卡：GPU0一份模型，GPU1运行仿真

以下是正式1600条命令。GPU编号为目标机实际分配给你的设备；没有分配的卡不要指定。
全部sim worker共享一份模型，但每个端口有独立动作队列；模型请求串行，仿真可并行。
4路会占用PORT至PORT＋3；同机并行跑另一份评估时需换一段空闲端口。

```bash
# 延续上面的MODEL_PY、SIM_PY、DATA_ROOT、CHECKPOINT设置。
export MODEL_GPU=0
export SIM_GPU=1
export WORKERS=4
export EPISODES_PER_TASK=100
export PORT=22400
export EVAL_ROOT="$PWD/outputs/maniskill_eval/final20872_dualgpu_$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/eval_maniskill_archive.sh
```

前台运行可放在目标机tmux中。若需要后台运行：

```bash
mkdir -p outputs/maniskill_eval
nohup bash scripts/eval_maniskill_archive.sh \
  > "${EVAL_ROOT}.launcher.log" 2>&1 &
echo "launcher PID=$!"
tail -F "$EVAL_ROOT/run.log"
```

前台和后台二选一，不要对同一个EVAL_ROOT执行两次。
双卡模式不把模型拆到两张GPU：MODEL_GPU放完整模型，SIM_GPU只做物理/渲染。

## 5. 同卡：GPU0放模型，也运行仿真

```bash
export MODEL_GPU=0
export SIM_GPU=0
export WORKERS=1
export EPISODES_PER_TASK=100
export PORT=22400
export EVAL_ROOT="$PWD/outputs/maniskill_eval/final20872_samegpu_$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/eval_maniskill_archive.sh
```

建议先在80GB级空闲卡上用1路仿真。模型放置前要求至少33GiB空闲，
后续编译/推理峰值及仿真还需余量；33GiB只是放置门槛，不是全程显存上界。
同卡与双卡使用同一组场景、模型、动作预算和判据，只改变硬件放置与吞吐。
确认真实峰值后，才手动提高WORKERS；不承诺任意40/48GB卡都能运行4路。

## 6. 可选：先检查路径或小规模跑通

只生成本机1600条plan并检查必要路径，不分配模型GPU：

```bash
EVAL_ROOT="$PWD/outputs/maniskill_eval/pathcheck_$(date -u +%Y%m%dT%H%M%SZ)" \
  CHECK_ONLY=1 bash scripts/eval_maniskill_archive.sh
```

CHECK_ONLY生成过的目录不能再作为正式运行目录；正式命令使用新的EVAL_ROOT。

如果先做显式小规模闭环，可用每任务每条件4条，共64条：

```bash
EVAL_ROOT="$PWD/outputs/maniskill_eval/smoke64_$(date -u +%Y%m%dT%H%M%SZ)" \
  EPISODES_PER_TASK=4 WORKERS=1 bash scripts/eval_maniskill_archive.sh
```

这里HardRandom按D1＋M3，仍覆盖两种hard域。
正式评估必须恢复`EPISODES_PER_TASK=100`；64条不与1600条报告混合。

## 7. 自动执行顺序与输出

1. 检查Python、checkpoint配套和归档路径，重建当前机器的plan。
2. 八任务分别检查Clean/hard D/hard M专家控制回放，共24场景，每场景原delta与绝对EEF两次执行。
3. 固定checkpoint，保存训练config/stats原件，派生单臂部署config/stats。
4. 模型载入MODEL_GPU；若容量不足则等待，不结束已有作业。
5. 验证Sidecar完整启用、cache匹配参考及FFN编译；失败时不开放策略服务。
6. 启动WORKERS路仿真，用1600个固定ID闭环执行；失败分片保留日志并只重试一次。
7. 核对全部ID/seed/task/domain/split无重复无遗漏后输出每任务、Clean/HardRandom成功率。

部署映射只保留原20D统计的前10维、scatter到80D模型的0:10槽，
其余70维inactive，与单臂训练mask一致。模型仍是完整80D结构，权重不变，
不能用“补零右臂且当它有效”的方式替代此步骤。

| 路径（相对EVAL_ROOT） | 内容 |
|---|---|
| plan.json | 固定评估清单；迁机后的archive_root |
| plan_summary.log | 各任务/域/split数量 |
| run.log | 主流程日志 |
| run/expert.json、expert.log | 24场景专家桥接检查，非策略成绩 |
| run/deployment/ | 固定模型、原训练config/stats、单臂部署副本 |
| run/server.log | 权重载入、显存等待、FFN验证与服务日志 |
| run/server_manifest.json | 模型服务PID及各独立端口 |
| run/worker0.log 等 | 仿真日志 |
| run/worker0/episodes.jsonl 等 | 逐episode策略结果 |
| run/worker0_retry/ 等 | 出错分片完整重试；原partial不混入最终成绩 |
| run/summary.json | 仅完整覆盖才生成的最终成功率 |
| run/failure.json | 模型/worker阶段失败详情；更早专家失败看expert.log和主日志 |

查看最终结果：

```bash
"$MODEL_PY" - "$EVAL_ROOT/run/summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d['complete'] and d['episodes'] == 1600
for r in d['conditions']:
    print(r['condition'], f"{r['successes']}/{r['episodes']}", f"{100*r['success_rate']:.2f}%")
for r in d['tasks']:
    print(r['task'], r['condition'], f"{r['successes']}/{r['episodes']}", f"{100*r['success_rate']:.2f}%")
PY
```

如果没有summary.json，先看run.log、expert.log和server.log，不把等待显存当作已开始rollout。
不要将原attempt与retry的episodes.jsonl直接相加，可能重复同一场景。
出现OIDN Error、ErrorDeviceLost、VK_ERROR_DEVICE_LOST或非法CUDA内存访问的worker日志，
即使进程exit0，也会被拒绝计分并重试；第二次仍出错则终止，不报告完整SR。
报错后保留目录，用新的EVAL_ROOT修复重跑，不覆盖原始证据。

## 8. 当前验证边界

本机已完成八任务24场景的原delta/绝对EEF回放，全部成功，
最大TCP轨迹差约12.53mm，沿用原2cm桥接检查阈值；不声称逐步轨迹完全一致。
迁机入口已用最终20872权重做无GPU路径/1600清单检查；
单臂归一化往返、缺失臂mask、图像布局、跨文件系统复制及渲染错误拒绝已有回归检查。
本次相关22项回归全部通过，证据见`docs/evidence/maniskill_archive_eval_portable_20261102/checks.json`。
当前Piper训练和已暂停的ManiSkill评估未因编写本文恢复/停止。
尚未在目标服务器验证GPU显存、FFN内核或完整1600条成功率，需由实际运行确认。
