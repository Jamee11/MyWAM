# OpenWAM ManiSkill 六任务 baseline 14K 评估

## 最终结果：2026-10-19 18:09 UTC验收

600/600条已完成，总成功576/600 = **96.0%**；顶层summary `complete=true`。
协调器、两个模型及12个worker已退出，无failure.json。逐任务100条episode ID、
test seed顺序、原始records/成功数/汇总一致，600个非空MP4数量齐全；本次未逐个
解码检查全部视频。以下为当前color-distractors held-out分布、14K中间checkpoint、
200控制步/replan8/10步去噪/cache+compile协议下的结果。

| 任务 | 成功 | 成功率 | 平均步数（含失败） |
| --- | ---: | ---: | ---: |
| PushCube | 100/100 | 100% | 55.38 |
| PickCube | 96/100 | 96% | 84.74 |
| StackCube | 93/100 | 93% | 105.30 |
| PlaceSphere | 93/100 | 93% | 134.55 |
| PickCuboid | 96/100 | 96% | 97.41 |
| PickSphere | 98/100 | 98% | 83.41 |
| 合计 | 576/600 | 96% | — |

24条失败均达到200步上限；仅据终态记录不能进一步判断抓取、目标选择或恢复
失败的具体根因。Stack/PlaceSphere各7条失败，是当前主要剩余项；PlaceSphere
平均耗时最长。下一步如需诊断，优先检查这些失败视频及200步预算影响；本次
未重跑或延长预算。该结果不代表最终10epoch checkpoint、方法相对收益或真机能力。
下文为启动过程历史记录，最终结果以本节和正式根目录summary.json为准。

## 最新执行规格：GPU5/6双模型、每任务100条

17:39 UTC真实闭环验收：双模型与12个worker均存活，六任务均已完成至少一个
正式policy episode；当前累计32条（Push9、Pick5、Stack4、PlaceSphere2、
PickCuboid6、PickSphere6），32条均成功，尚不能据早期样本推断最终SR。
没有failure.json；首个Push视频通过ffprobe（H.264、512×256、51帧）。
夹爪预测越界时按[0,1]裁剪并逐episode计数，未对绝对位置作静默裁剪。
600条run持续执行，最终仅按完整100条/任务及分片seed精确覆盖汇总。

用户随后要求每任务100 episodes；当前正式规格为六任务共600条，GPU5/6
各一份共享模型和6个独立仿真，共12路，每任务前50/后50分片无交集。
固定14K权重、200步、replan8、10步去噪和原成功判据不变。
权威输出为`outputs/maniskill_eval/baseline14000_six_task_100_replan8_max200_gpu56_20261019`；
启动日志`outputs/maniskill_eval/baseline14000_100_gpu56_20261019.launch.log`，
协调器PID2285290。下文50条规格为历史准备过程，不纳入本次600条分母。

首个12路50条版本在策略动作前因client导入训练包触发缺pandas而终止，
该失败完整保留，不计policy SR。已改用现有dependency-free评测prompt模板，
真实仿真Python的PNG/WS/prompt/EEF/Gym/ManiSkill导入检查已通过；未安装包。
新增双模型端口分配与100条分片检查，联合既有pool测试共8项通过。
真实单步hold-pose通信/视频预检已exit0：真实test seed2000000、front/hand PNG、
10D实际EEF、绝对控制器、单步物理执行与H.264视频写出均走通；输出
`outputs/maniskill_eval/client_preflight_hold_20261019`。此项用hold-pose stub，
不加载checkpoint，不作为策略SR（日志中的0/1只代表这一预检）。
17:28 UTC双模型PID2285786/2285787启动；17:34 UTC已strict加载并通过全部
12端口PONG。17:37 UTC真实checkpoint开始推理，首次compile之后正常连续
完成10步去噪/动作执行；正式闭环链路已验收，600条最终结果尚待完成。

## 评估合同

2026-10-19 UTC，用户授权从 DiT4DiT 迁移必要评估代码，测试现有原版
OpenWAM baseline 的六任务成功率。训练仍在运行，本次固定中间的
`six_task_bs64_10epochs_20261017T050252Z/runs/2026-10-17_05-16-28/checkpoint_step_14000.safetensors`。
不将它称作完成 10 epochs 的最终模型。

已核对 OpenWAM 的数据/部署路径：`ManiSkillAbsoluteDataset` → 原版
joint Video/Action DiTs → 官方 `PolicyServer` / `WAMPolicy` /
`JointInferenceEngine`。输入使用原 front/hand uint8 RGB、训练同款颜色指令
模板、实际 robot-base `panda_hand_tcp` XYZ/column-Rot6D/手指开度10D。
统一80D头仅0:10有效，现有normalizer负责训练统计的归一化、scatter和输出
gather/反归一化。没有EA、Sidecar、目标位姿、分割、接触标签作为模型条件。

关键迁移差异：训练标签是绝对controller command EEF10，不是旧DiT4DiT
delta7。独立评估bridge对Rot6D作Gram-Schmidt正交化后转intrinsic XYZ Euler，
夹爪[0,1]转[-1,1]，使用原生`pd_ee_pose`，不添加RoboTwin工具偏移。

六任务均为color-distractors环境：PushCube、PickCube、StackCube、PlaceSphere、
PickCuboid、PickSphere。复用原有successful-controller test seed manifest前50条，
总300 episodes；保持CPU physics、原默认相机shader、最多200控制步、原成功判据。
10步去噪；每执行8动作重规划，模型仍生成32动作。cache/compile沿用原版
`configs/deploy.yaml`。此replan8与既有DiT4DiT ManiSkill协议一致，但不同于
OpenWAM默认全32动作执行；后续方法比较必须记录并固定这些选项。

## 实现与证据

- 新入口：`benchmarks/maniskill/run_eval.py`、`single_eval.py`；复用现有
  WebSocket transport、PNG编码、prompt模板及EEF源合同，无模型/训练核心修改。
- 独立输出：`outputs/maniskill_eval/baseline14000_six_task_50_replan8_max200_20261019`。
  固定checkpoint hardlink与stats/config/tokenizer快照，防训练keep-last清理；
  权重inode只读，不原地修改。`run.json`记录SHA256、seed顺序、预算与源码身份。
- 模型GPU5、仿真GPU6；每任务先跑原delta/转换后绝对命令的物理专家回放，
  均成功且TCP最大差不超过2cm才进入策略评估。GPU5/6启动前各约79GiB空闲。
- 本机旧Vulkan阻塞有可复用的进程级修复：预加载现有
  `libnvidia-gpucomp.so.570.86.10`，不修改驱动/依赖。旧专家检查入口已实际
  完成六任务各一条首轨迹的双控制器回放，12次均成功，TCP最大位置差
  `0.000154223m`；证据`outputs/maniskill_baseline/expert_replay.json`，exit0。
  这是旧源脚本的独立gate；正式runner还以RGB环境对自身bridge逐任务复验。
- CPU adapter回归4项通过：绝对目标/Rot6D/夹爪往返、退化/错误动作拒绝、
  held-out successful seeds选择、训练/部署图像画布逐像素一致。
  真实GPU模型加载与正式rollout正在验收。

## 当前结果与下一步

### 用户要求GPU5/6尽快并行后的更新

14:27 UTC后，按用户新指令将正式评估调整为12路：每任务拆为两份互斥25条，
两张卡各6个仿真，GPU5共享一个模型。复用既有
`benchmarks/robotwin/serve_parallel.py`，12端口各有独立`WAMPolicy`/动作队列，
模型generate串行，禁止跨episode混用buffer。新增分片覆盖与重复拒绝测试，
连同原共享服务测试共7项通过。复用本次已通过的六任务专家报告，不重复回放；
旧专家检查脚本只新增可选`--output`以保留独立run证据。

旧串行run于策略请求前停止，未产生policy episode，历史目录和gate全部保留。
新权威根目录为
`outputs/maniskill_eval/baseline14000_six_task_50_replan8_max200_parallel12_20261019`，
新日志`outputs/maniskill_eval/baseline14000_parallel12_20261019.launch.log`；
启动PID2201861。checkpoint从旧独立快照再次固定，SHA256为
`1f3aa4dabf1c1c96df89987850086323fbd52910e49d7cc09f6563b9e978fbbe`。
尚须验收所有端口真实PONG、12个实际客户端、模型动作/渲染/视频及episode终态。
运行期间不安装或修改共享环境，不操作其他GPU作业。

当前状态是并行评估已启动、正在模型加载，尚无完整六任务策略成功率。
权威日志为`outputs/maniskill_eval/baseline14000_parallel12_20261019.launch.log`；各任务
的`policy/<task>/shard_*/episodes.jsonl`实时记录，任务`summary.json`在两个分片完成且覆盖一致后写出。
任何运行时异常保留`failure.json`，不能当作策略失败加入分母。

下一步核验真实模型首个动作及视频，完成每任务50条后核对seed顺序、视频数、
逐episode记录与summary一致，再报告六任务表及宏平均。专家成功只验证执行
合同；不证明策略能力或sim2real性能。
