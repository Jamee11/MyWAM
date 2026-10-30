# PatchWAM C2R 外观增强接入与验证

日期：2026-10-27 UTC。范围：实现调研报告 `deep-research-report (2).md` 中的 PatchWAM 外观增强；根据用户后续授权加入100-step WAM LR ramp及完整训练接续。不实现本轮报告中的Grounding路由、2.5D或部署修改。

## 1. 先保存原版本

增强接入前已将当前工作树保存至：

- 仓库：[Jamee11/MyWAM](https://github.com/Jamee11/MyWAM)。
- 分支：[`backup/pre-patchwam-c2r-20261027T140234Z`](https://github.com/Jamee11/MyWAM/tree/backup/pre-patchwam-c2r-20261027T140234Z)。
- 远端提交：`66bb8d2056c84c55757f1d3c7a43edc0bced57ad`。
- 完整文件树：`8b99554366c67bf534657bf2f2d33811a09ad1d1`，与本地隔离 index 快照逐树一致，共982条文件/子模块记录。448个工作树变更纳入备份。

备份包含当前训练/推理/诊断、A/C接续、配置、文档和测试；数据集、权重、环境及 outputs 不上传。主分支、当前工作目录和实际训练进程未切换。直接 git push 因已失效的 VS Code askpass socket 认证失败，改用已连接 GitHub App 的 Git 对象接口上传，并核对远端 ref/tree；没有导出凭据或强推旧分支。该 backup 分支不命中当前 CI 的 main/feature/fix push 过滤，不把“没有 CI run”说成 CI 测试通过。

## 2. 来源与准确参数

已下载完整源码到 `.external/PatchWAM`，约1.8MB；固定 revision `dea19d21182a26a8ee16efc2c4125ee7aecf44fb`。它是参考代码，不作为新的环境或运行时安装依赖。

官方文件：

- [C2R 配置](https://github.com/TianhengW/PatchWAM/blob/dea19d21182a26a8ee16efc2c4125ee7aecf44fb/configs/robotwin_c2r.yaml)。
- [AppearanceRandomizer 实现](https://github.com/TianhengW/PatchWAM/blob/dea19d21182a26a8ee16efc2c4125ee7aecf44fb/src/patchwam/data/appearance.py)。
- [训练说明](https://github.com/TianhengW/PatchWAM/blob/dea19d21182a26a8ee16efc2c4125ee7aecf44fb/docs/training_recipes.md)明确区分开发配置、CPU数据验证和仍未验证的完整GPU/闭环复现。

本地 `openwam/dataloader/transforms/patchwam_appearance.py` 与上游 appearance.py **字节一致**，SHA256 `1d9719cd379bca55085938183f4d6dbb068b07f1015225ad38cb5d1f653acf27`。MIT版权及来源保存在 `licenses/PatchWAM-data-MIT.txt`、`licenses/PatchWAM-data-NOTICE.json`。

| 操作 | 当前新增配置，逐项匹配上游 |
|---|---|
| 总增强概率 | 每个样本一次抽签，p=0.8 |
| Brightness / Contrast / Saturation | 三者均0.4，即因子[0.6,1.4]；顺序随机 |
| Hue | ±0.1周，即±36° |
| Gamma | [0.7,1.4] |
| Exposure | [-0.5,0.5] stops |
| Color temperature | R/B ±15%，G ±7.5% |
| Gaussian blur | p=0.3，sigma约[0,1.5]；保留上游小sigma分支 |
| Gaussian noise | std=0.03；同一视角沿时间共享噪声图 |
| AdaIN | p=0.5；目标mean[0.2,0.8]、std[0.05,0.4]、strength[0.4,1.0] |
| Fourier | p=0.5；amplitude jitter[0.5,1.5]、envelope strength[0,0.6]、grid=8 |
| 顺序 | Photometric → AdaIN → Fourier |
| 背景替换 / Crop / Rotation / Translation | 不启用 |

同一相机的9帧（当前帧及未来监督帧）共同处理，时序参数共享；三台相机独立采样。完整执行包含真实hue/gamma/exposure/style/FFT，不以普通ColorJitter替代这些操作。

## 3. 最小接入及边界

新增默认关闭的 `dataloader.patchwam_augmentation`，只接入现有 `EARSIGLiteDataset → LiteRoboTwinSource` 路线。它覆盖当前Aloha Sidecar的真实训练入口，也可在这个reader的auxiliary=false对照中同样启用。未扩展到其它仿真reader或修改当前多源训练。

当前reader已经生成384×320的L画布。适配器按现有几何拆成头相机256×320和两个腕相机128×160，对每个真实相机clip运行原版randomizer，然后原位拼回PIL图像。源HDF5相机是否存在用于判断padding，不用黑像素阈值猜测；缺失相机不参与统计/随机化且保持黑槽，真实相机中的黑像素仍正常增强。增强后 `first_frame_image` 同步引用增强后的第0帧。

**与PatchWAM完整数据管线的差别要保留：** 我们沿用OpenWAM既有的逐视角resize和L画布，在resize之后应用增强，而PatchWAM的recipe在自己的逐相机尺寸上增强后再compose；OpenWAM的PIL接口还需要float→uint8量化。故声称“算法、参数和相机/时间采样规则一致”，不声称两个项目的端到端像素完全相同。原版tensor算法的逐元素一致性已单独验证。

新配置明确关闭旧 `color_jitter`，两者同时启用会报错。关闭新开关时不加载新增强模块、不抽取新随机数，原增强路径保留。val和部署不启用本增强；EA bbox、relation/contact/plan、绝对EEF动作、mask、归一化、prompt格式均不重建或重解释。

## 4. 新训练配置与启动

新配置：`configs/train_relational_sidecar_aloha50_patchwam_c2r_10epochs.yaml`。

沿用当前C的训练结构：正常Read/Write及完整反向，前2000 optimizer updates跳过WAM参数更新，随后joint。用户后续明确选择“增强＋100-update ramp”完整训练，因此新配置设置 `training.sidecar_wam_ramp_updates: 100`；旧配置未指定时仍默认300，当前C短训不变。

设u为已完成optimizer更新数，新WAM实际LR为 `scheduled_cosine_lr × (0.1 + 0.9 × clamp((u-2000)/100,0,1))`。第2001次更新使用0.1倍，u=2050时0.55倍，u=2100后为1倍。Sidecar LR不受此倍率影响；optimizer step后恢复scheduled LR，避免把倍率递归混入cosine scheduler。它改变解冻更新的步长，不直接改变首个backward的梯度。100-step与300-step的实际Adam更新/状态恢复及训练工厂传参均有CPU测试。

- 同一Clean50任务/2500轨迹、549775窗口及prepared manifest。
- 每卡batch32 × 4卡 × accumulation2 = global batch256。
- 10 epochs；原模型/EA权重、AdamW、主干LR和Sidecar LR保持。
- 每4000微步=2000次optimizer更新保存，keep_last_k=2；沿用现有trainer的最终保存策略。
- 新启动脚本默认从原Foundation开始，独立带UTC时间戳输出，**不会resume进旧A/C目录或替换已有C正式续训**。它不是“从共同2K完整状态恢复”的短训对照；如后续选择该对照，应单独准备独立输出的完整状态恢复，不能直接把原run填给基础resume参数（该参数会复用原输出目录）。

配置预检（不分配训练模型）：

```bash
cd /remote-home/jinminghao/OpenWAM
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OPENWAM_VENV=/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv \
CHECK_ONLY=1 bash scripts/train_sidecar_patchwam_c2r_4gpu.sh
```

正式启动时在**另行分配的空闲四卡**上将CHECK_ONLY改0；CUDA_VISIBLE_DEVICES必须由调用者设置。脚本复用现有通用cluster训练入口，保持Slurm/手工分配的GPU编号，不固定抢0–3。另一机器把OPENWAM_VENV设为本机环境目录，并相应设置OPENWAM_MANIFEST/OPENWAM_FOUNDATION即可，不下载新的预训练模型。

早先本轮保留0–3原A/C短训及C正式续训，新增强未获得独立四卡。用户后续明确替换接续计划：保留C本次3K→4K的1K短训与评估，取消原C4K→10epoch长训，GPU0–3接入从Foundation重新开始的增强＋100-step ramp完整10epoch实验。

新队列根目录：`outputs/sidecar/aloha50_patchwam_c2r_ramp100_10epochs_20261027T144939Z`，入口指针 `outputs/sidecar/patchwam_ramp100_current.json`。监督 `scripts/supervise_sidecar_patchwam_after_c.py` 已启动四rank正式入口CPU预加载；仅核验C真实exit0、8000微步/4000更新的配套权重和完整state后才写接卡标记，正式入口再检查0–3每卡≥70000MiB。不会加载C权重或Adam历史，所有新训练计数从0开始。C离线评估在原评估队列并行继续，不占0–3或作为新训练额外等待门槛。

原C长训监督4137756在尚未创建任何子模型时按PID/start-ticks/命令身份核验停止，原目录保留并记录cancelled.json。新监督4168818、新torchrun4168821；实际状态以新root的job_status.json、handover/cpu_contract.json、cpu_acceptance.json及startup_acceptance.json为准，不把CPU预加载称为已完成GPU更新。

原图/旧ColorJitter/PatchWAM对比：`outputs/patchwam_c2r_implementation_20261027T140234Z/preview_{adjust_bottle,beat_block_hammer,place_shoe}.png`。用户要求展示后补充 `preview_adjust_bottle_multiseed.png`（seed0/2/11/42，42为未增强样本）和 `preview_adjust_bottle_temporal.png`（相对帧0/16/32）。这些是增强预览，不是新模型成功率证据。

## 5. 已完成的验证

证据根目录：`outputs/patchwam_c2r_implementation_20261027T140234Z`。

1. **首版32项CPU回归通过；100-step接续版本51项通过**：官方文件hash、完整recipe、时序/seed/范围、PIL相机适配、skip样本不量化、缺失视角、真实黑像素、双增强拒绝、真实微型HDF5、既有A/C续训合同、100-step Adam更新/恢复及新队列接卡控制流；Ruff、shell语法、Hydra配置解析和fast preflight通过。
2. **独立上游oracle**：直接导入固定revision的官方源码，与vendored类在12个seed、3相机×9帧上逐元素完全相等。
3. **真实Aloha数据**：adjust_bottle、beat_block_hammer、place_shoe各一窗口。增强后仍9帧、384×320、action[32,80]、EA[1,130]；逐项核对所有非图像字段完全相等。预览保存在 `preview_*.png`。
4. **原路径保真**：将Git备份里的旧reader动态载入，与当前新开关关闭的reader对比上述真实窗口，图像、监督和Python/Torch RNG状态均一致。范围是这些窗口，不冒称全量视觉审计。
5. **真实训练数据构造入口**：从launcher的resolved OmegaConf通过registry构造全部50任务reader，开启新增强；2个DataLoader worker实际读取3批、每批2条，形状/有限性/首帧同步全部通过。没有用单个transform测试冒充完整GPU训练。

单线程、共享盘热缓存、9条真实窗口的reader计时：原jitter中位0.095s/样本，新增强0.491s/样本；强制每次执行增强的transform本身中位约0.398s/样本。它增加CPU数据加载成本，不增加模型结构；多worker/prefetch可覆盖部分开销，但**尚不能保证正式每步训练速度不变**，需新GPU实验记录DataLoader等待和稳定step time。未调整旧训练的worker或环境来“优化”这组计时。

## 6. 结果解释与下一步

这是值得优先验证的外观鲁棒性候选，直接复用已有几何监督，不要求重构数据和模型。它不是已经验证的SR提升，也不能解决Randomized中的新干扰实体、桌面高度、物体几何和语言变化。报告中“风险最低/最可靠”应理解为相对开发成本，不能理解为只有优势。

尤其hue/强style会扰动颜色语义及局部可见性，颜色指定任务未必满足标签语义完全不变；当前按用户要求保持官方统一参数，不悄悄削弱或按任务改recipe。先看配对增强预览及少量C2C/C2R闭环，记录颜色任务是否退化。AdaIN使用整个clip统计也按上游保留，不声称完全消除了训练/推理域差。

后续在空闲四卡上先检查真实稳定更新、loader等待和保存/重载，再用配对Clean/Randomized闭环判断收益，不按更高或更低的训练loss直接选胜者。若做正式论文比较，Vanilla也要有同增强控制；增强引入新的随机过程，不宣称与旧增强具有完全相同的prompt/noise全程随机序列。

既有near-plane差异、RoboTwin EEF物理replay、matched Vanilla、少量GT窗口和sim-to-real标定边界仍保留，本轮没有解决或绕过。

验收补记 2026-10-27T15:01:01.391132+00:00：四rank CPU预加载于14:57:16Z完成，真实cpu_contract核验Foundation来源、resume=null、completed_optimizer_updates=0、WAM warmup2000、ramp100、PatchWAM augmentation实际启用、global256全部通过。新监督4168818及torchrun4168821按start-ticks核验存活；原C短训/诊断/Loss监督也均存活，当前约3630更新；原C长训监督已停止且从未生成子模型。GPU release与predecessor_verified尚不存在是按计划等待C短训结束，并非新训练已经更新。证据engineering/final_scheduling_acceptance.json。


### 2026-10-27T17:35:20.224706+00:00 — C 4K离线结果与新增强训练真实接卡

A/C均完成至8000微步/4000optimizer updates，完整状态/真实exit0及固定输入诊断均通过，无failed。少量GT窗口下，32步XYZ(mm)：A Clean12.2688/Random20.9182，C Clean13.9091/Random20.3011。C固定Action FM比A高13.16%；C Full相对自身Write-Off FM降低16.04%（A降低11.68%），但C Off也较弱，因此不能把更大Full/Off差距当C整体胜出。C训练最后250更新Action仍较A高5.94%。结论：延长后C进步且Random XYZ略好，仍无全面离线优势，更非闭环SR结论；GT仅24 heldout重叠窗口/6episode及32FM，边界保留。

新PatchWAM＋100ramp完整训练已真实接卡：旧C于16:24:09Z正常退出，新GPU gate于16:24:17Z释放（约8秒交接），新torchrun4168821/四rank已从Foundation开始更新。当前319更新/639微步，近期中位6.601s/微步，约13.20s/optimizer update，仍在前2K WAM不更新阶段，100-step ramp尚未开始。前50对最近50更新：Action 0.04375→0.04414，Plan 1.6026→1.1061，Relation 1.1568→0.7690。辅助目标下降但Action暂未下降，不据早期warmup断言增强好坏。CPU预检状态已由本次真实GPU更新解决；2K解冻、R100完整曲线、最终checkpoint和闭环收益仍未验收。

用户关注多视角一致性：已重新核对固定上游代码，p=.8按样本抽签，各相机独立采样photometric/style/Fourier参数及分支，同一相机9帧共享时间参数/噪声图；算法及参数范围相同，不是每路换不同配方。独立颜色/AdaIN可能增加跨相机外观对应难度，颜色指令也可能不满足标签语义不变，官方采用并不构成对OpenWAM的最优性证明。当前不热改在跑配方；若以后比较共享颜色/style需单独受控分支，共享hue本身也不保证保留颜色语言语义。未变更网络、数据或训练配置。


## 2026-10-28T02:08:58.532472+00:00 — 新PatchWAM＋R100与A的同更新数训练Loss对比

截至2600 optimizer updates；每个100-update窗口平均200微步的日志loss，梯度范数只取偶数微步更新边界。A的前2K取共同起点来源run，2K后取A_control；来源配置虽含1%退火，但代码在warmup阶段强制scale=1，A_control在2K后也为1。保持global256、模型、loss权重和数据集；增强输入/视频目标、随机轨迹不同，且同时增加100-update ramp，不作严格单变量或同噪声配对比较。

|更新窗口|Action A|Action 新|新相对A|Video A / 新|EA A / 新|
|---|---:|---:|---:|---|---|
|1–100|0.043437|0.044332|+2.06%|0.209436 / 0.212250|0.126434 / 0.126853|
|401–500|0.039192|0.040788|+4.07%|0.210371 / 0.212963|0.100699 / 0.101618|
|901–1000|0.026013|0.029813|+14.61%|0.210477 / 0.212469|0.079399 / 0.080888|
|1401–1500|0.020303|0.023279|+14.66%|0.209571 / 0.209421|0.065549 / 0.067809|
|1901–2000|0.015905|0.018315|+15.15%|0.209253 / 0.209331|0.061529 / 0.063639|
|2001–2010|0.045897|0.017737|-61.35%|0.330237 / 0.199971|0.066933 / 0.063350|
|2001–2100|0.012431|0.010597|-14.75%|0.183083 / 0.170781|0.061114 / 0.061962|
|2101–2200|0.004873|0.006381|+30.94%|0.151552 / 0.163119|0.058672 / 0.060188|
|2201–2300|0.004131|0.005043|+22.07%|0.144760 / 0.158945|0.057534 / 0.059058|
|2301–2400|0.003629|0.004450|+22.61%|0.139818 / 0.155008|0.056697 / 0.058269|
|2501–2600|0.003032|0.003658|+20.63%|0.135768 / 0.152502|0.055526 / 0.056896|

解冻后前100更新：A最大grad norm 12.6709，新1.0800；>1计数A 13/100，新4/100。新明显减小边界尖峰，但不是零clipping，也不能仅归因ramp（增强和训练轨迹同时变化）。

最近2501–2600：Action新相对A +20.63%，Video +12.32%，EA +2.47%；Ground +0.43%，Plan +13.34%，Relation +15.04%。两者均持续下降，新Action相对差距在2101–2200后收窄；不称全程单调。

结论：新训练更平滑、稳定期training loss仍高于A，目前没有training-loss全面领先；强增强可能提高学习难度，但高loss既不证明泛化收益也不证明退化。继续当前配方，4K更新时用关闭训练增强的固定Clean/Randomized样本、同FM噪声/时刻比较Action与关系质量，再做配对闭环。旧near-plane、EEF replay、matched Vanilla、小样本与颜色语义风险仍未解决。本轮只读取日志并保存分析，不改训练/模型/配置，不启停GPU。

证据：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_patchwam_c2r_ramp100_10epochs_20261027T144939Z/analysis/A_vs_patchwam_2600updates_20261028/comparison.json`、`update_losses.csv`；曲线采用100-update移动均值，A延伸只展示到共同2600更新，不能把更晚A训练量混入。
