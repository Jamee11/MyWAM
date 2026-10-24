# ManiSkill 六任务 OpenWAM baseline，10 epochs

2026-10-17 UTC。用户授权构造必要数据并尽快在GPU0–3启动。本文启动状态与验收结果在后续追加；本段为执行合同。

## 数据与动作

用户入口：`/remote-home/jinminghao/datasets/maniskill_eort_six_task_ea_rsig_v3_20260922`。

此目录六个LeRobot bucket为符号链接，内部动作是Panda归一化delta7；不可直接作为绝对EEF。原始HDF5在：

- `/remote-home/jinminghao/datasets/maniskill_eort_color_distractors_v4_20260728`：push_cube、pick_cube，各1000轨迹，使用raw replay HDF5。
- `/remote-home/jinminghao/datasets/maniskill_eort_task_suite_v5_20260730`：stack_cube、place_sphere、pick_cuboid、pick_sphere，各1000轨迹，使用含补采的aggregate HDF5。

精确源HDF5、trajectory和对应指令由每任务`train/color_distractors/derived_objectcentric_v4/manifest.jsonl`解析，最终逐条保存在新ready manifest中。

**本次新构造数据目录：** `/remote-home/jinminghao/OpenWAM/outputs/maniskill_baseline/data_absolute_eef_20261017/`。

其中`numeric/<task>/episodeXXXX.npz`只存actual state10和absolute command action10，`normalization_stats.npy`只从train拟合，`manifest.ready.json`记录合同/映射/哈希/源文件身份。原始RGB、depth、segmentation均不复制。baseline只读front RGB、hand RGB、当前actual EEF和文本；不读取EA/oracle作为模型条件。

坐标为robot base，末端panda_hand_tcp，rotation6D为旋转矩阵前两列，无额外工具偏移。原delta命令按采集controller裁剪/缩放后重建绝对目标，不取下一帧实际到达位姿。state[t]和action[t]配对，后者驱动t→t+1。夹爪action从[-1,1]转[0,1]，state由实际手指qpos/.04得到。80D头保留，仅0:10有效，避免改头或丢失Foundation初始化。

相机沿用384×320 L画布：上front、左下hand、右下黑色。原HDF5 uint8 RGB直接使用PIL，不进行BGR交换。33个原始时刻、stride4实际9张视频帧，32个动作；尾部pad且以时间和维度mask排除无效loss。

## 训练

独立配置：`configs/train_maniskill_baseline_10epochs.yaml`；reader配置`configs/dataloader/maniskill_absolute.yaml`。

启动器：`bash scripts/train_maniskill_baseline_4gpu.sh`。新reader由独立Python入口注册，不能直接用通用scripts/train.py替代此入口。

GPU0–3、bs64/rank、acc1、global256、epochs10、max_steps=null。原版dual_system joint_self_attn，无EA/Sidecar。加载`assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors`，新优化器从0训练。

默认AdamW lr1e-4、betas(.9,.95)、wd.01、clip1、BF16、ZeRO2、梯度检查点、5%warmup/cosine/min_ratio.01、video/action loss各1、flow shift各5、seed42。为A100容量使用CPU初始化及相同lr三组fused AdamW，无optimizer offload。

ColorJitter遵从**仓库默认关闭**，不是之前RoboTwin C2R实验中用户显式开启的override。若后续做baseline/EA对照，必须保持同样增强、数据、动作协议及训练预算。

每2000updates保存权重和完整恢复状态，保留最近2组及最终状态。满6000轨迹时共542009个动作窗口；最终每epoch及total updates以Accelerate实际日志为准。

## 验收与限制

准备时检查全部episode映射、T/T+1同步、有限值、相机shape/dtype、derived TCP对应、文本非空、统计及文件哈希。训练启动不再全量解码或重扫；worker首次打开HDF5检查size/mtime、首次载入numeric检查hash，之后LRU复用。

待追加实际controller方法同状态比较、专家物理replay、真实首中末窗口及64样本batch、四卡首更新/显存/吞吐。六任务小样本回放不能代表策略闭环成功率；本次长训练用于得到baseline，最终效果需另做独立评估。

### 已完成的合同核对与环境限制

18轨迹/1614控制步的真实PDEEPoseController._clip_and_scale_action + compute_target_pose同状态比较通过：最大位置分量误差1.86265e-9 m，最大旋转误差2.83460e-7 rad；配置为pos±.1、rot_lower=-.1、root translation/root aligned body rotation、use_target=false、panda_hand_tcp。另对2048组随机base/world姿态与超界命令验证坐标变换、裁剪和旋转正交性通过。EORT实际robot为panda_wristcam，继承Panda控制器，额外以该类有效配置复核并记录URDF路径。

六任务原HDF5中间帧拼图：`outputs/maniskill_baseline/rgb_six_tasks_preview.png`，已人工查看，front/hand/缺失黑槽位置正确，直接uint8 RGB，无RoboTwin JPEG通道逻辑。

专家物理回放尝试被本容器图形运行库阻塞：默认renderer报Vulkan ErrorIncompatibleDriver；render_backend=none时URDF视觉材质仍要求render device。未更改系统驱动、仿真代码或训练环境；不得把此情况记作专家成功/失败或回放已通过。现有实际控制器同状态目标比对已直接验证训练标签数学合同；在用户要求尽快训练baseline的范围内，数据与GPU更新验收通过后可启动训练，物理回放作为后续ManiSkill策略闭环评测前的未解决门槛。训练启动本身不依赖SAPIEN/Vulkan。

### 全量准备与真实reader验收完成；四卡加载已开始

6000/6000轨迹、542009动作窗口准备成功；源HDF5 recorded_success各任务1000/1000，仅为原数据记录，不是本次物理回放或模型评估。原manifest的源轨迹映射、shape/finite/derived TCP一致性、每个numeric SHA256和统计hash全部在准备时核对。ready manifest内保留每条精确路径。

54个真实窗口（六任务×首/中/末episode×起点/中间/末端）通过：action[t]配对、action/proprio80D仅10D有效、尾部padding/时间mask、9张384×320 RGB拼图与源像素一致、右下黑槽。list collation后64样本action为[64,32,80]、proprio为[64,1,80]。检查耗时34.46秒，证据`outputs/maniskill_baseline/data_window_checks.json`。此时间包含额外验证读取，不是训练dataloader吞吐。

PandaWristCam有效配置的18轨迹/1614步同状态controller比较复验通过，误差与前述一致，见`controller_comparison_wristcam.json`。

2026-10-17T05:09:08Z启动四卡torchrun，GPU0–3显存空闲门槛通过。持久tmux=`maniskill_baseline_20261017T050252Z`。运行根目录`outputs/maniskill_baseline/six_task_bs64_10epochs_20261017T050252Z/`，`train.log`为总日志，`launch_record.json`记录输入/参数/文件hash，训练产物写入其`runs/<timestamp>/`。本条写入时尚在模型加载，首更新、实际显存/吞吐与最终完成状态待追加。

采集版本追溯补充：读取六个源HDF5配套JSON的commit_info，push_cube/pick_cube记录94368b1ac36e5ce6f80a64b7a3c3290a1a83e298，四个aggregate记录b53792f638893816ae8644c2d30d1cc035ada6f3。逐一用git show读取这些历史commit中的pd_ee_pose.py、panda.py、panda_wristcam.py，与本次实例比对使用的当前源码逐字节/SHA256完全一致（六源×三文件）。因此动作转换不是仅凭当前版本猜历史控制逻辑；证据`outputs/maniskill_baseline/historical_controller_source_check.json`。仍不以源码一致替代完整物理replay。


## 四卡真实训练启动验收：2026-10-17T05:20Z

四个rank均已完成至少8次真实optimizer updates，micro=optimizer（acc1）；全部已记录loss/gradient norm有限且gradient norm>0，无OOM。最后总loss=0.185643，video=0.101942、action=0.083701，没有EA项。此初期loss不能证明策略成功率或收敛优劣。

原版baseline可训练6021.2M参数，Foundation结构合并0个override/0个live-only，原80D头保留。DeepSpeed四rank于05:16:55完成包装；首batch约91秒（含数据预取及CUDA冷启动）；排除头两步，已测步均值约12.942秒/update，global256约19.78samples/s。仅初期6步小窗口，不是长期基准；按此速度21180更新纯训练约76.1小时，未含保存和后续波动。

GPU0–3 NVML快照使用80963/79123/79503/79023 MiB，GPU利用率均100%；GPU0最少free190MiB，**bs64显存余量很小**，按用户指定保持64，不擅自降batch。本次记录的最大torch allocated=72.500GiB、reserved=77.596GiB；两者与NVML整卡使用含义不同。真实首更新已通过，不代表长期/首个checkpoint保存已验收。

总预算21180updates/10epoch，每2000updates存权重+完整状态，最近2组及最终状态保留。checkpoint实际输出：`/remote-home/jinminghao/OpenWAM/outputs/maniskill_baseline/six_task_bs64_10epochs_20261017T050252Z/runs/2026-10-17_05-16-28`。入口与tmux持续运行，未停止其他实验。启动证据`outputs/maniskill_baseline/startup_verification.json`；per-rank原始metrics与config/stats/tokenizer在同一checkpoint目录。

数据绝对EEF构造位置、源HDF5身份及RGB图均见前文。ColorJitter本次遵从仓库默认关闭。物理专家replay仍被Vulkan渲染依赖阻塞，策略闭环评估未执行；本轮完成的是训练数据合同与四卡真实更新验收，不能宣称10epoch已经训练完毕。


### 2026-10-17T06:29:06.440952+00:00 运行进度与耗时更新

四rank现场均到330updates，GPU0–3利用率100%，未见OOM/退出；总预算21180（第1epoch约15.6%、总约1.56%），仍处warmup。最近300updates均值12.826秒，最近100约12.880秒，最近20约13.018秒；按12.8–13.0秒估算剩余约74–76小时（约3.1天），未含checkpoint与后续波动，预计北京时间10月20日下午前后完成。当前训练循环已运行约1小时12分，最近loss总0.05499/video0.04698/action0.00801；仅训练loss，不证明闭环效果。未到首次2000步保存点，尚无checkpoint，按现速约6小时后首存。NVML使用80963/79123/79503/79023MiB；GPU0余量小的限制保持。配置/代码/训练均未改动。


### 2026-10-18T06:11:19.829649+00:00 运行进度复查

四rank同步到6924updates附近，epoch=3（第4epoch），总体32.69%，前三个epoch完成、第4个约26.9%。运行约24小时54分钟，GPU0–3利用率100%；最新总loss0.03268/video0.02880/action0.00388、grad norm0.08224。最近300步12.735s/update，剩余约50.4小时，预计北京时间10月20日16:40前后，保存及后续波动另计。现存checkpoint_step_4000/6000.safetensors（每份24813767464bytes）及对应accel_state目录，2000已按最近2组策略清理；未进行恢复运行验收。训练继续，未改配置/代码/进程。


### 2026-10-19T07:58:42Z 运行进度复查

四rank最新metrics于07:58:39Z同步到14084updates，epoch=6（第7epoch），总体66.50%，前6个epoch完成、第7个约65.0%；训练循环约50小时42分钟。最新总loss0.0214211/video0.0204331/action0.0009880、grad norm0.0528501，metrics持续更新。常规步约12.9s/update；最近300步均值13.759s，窗口跨14000步保存点，包含保存开销。剩余7096步，按当前常规速度约25.4小时，按该窗口均值约27.1小时，预计北京时间10月20日傍晚至晚间完成，后续保存及吞吐波动会改变估计。现存checkpoint_step_12000/14000.safetensors（每份24813767464bytes）及对应accel_state目录；保存后已继续到14084步，未进行恢复运行验收。此次仅状态检查与实验记录，未改配置/代码/训练进程；训练loss不代表闭环成功率。

### 2026-10-19T14:09:42Z 进度与增强配置复查

四rank最新同步15811/21180updates（74.65%），epoch=7（第8epoch），metrics更新至14:09:41Z。近300步均值12.8423秒，剩5369步约19.15小时，纯更新ETA北京时间10月20日17:19，额外保存/波动另计，预计17–19点。最新loss total0.0210065/video0.0198532/action0.00115329，grad norm0.0383739。现场读取实际run的config.yaml确认color_jitter.enabled=false，与本次ManiSkill基线遵循仓库默认关闭一致；此前RoboTwin实验显式开启不代表本run开启。此次未修改训练配置/代码/进程。
