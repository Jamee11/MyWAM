# 21480最终权重：配对外观压力诊断

创建：2026-10-30T12:44:33.062921+00:00。用户授权先诊断本地归档10epoch最终权重，并要求扩大外观变化；不修改现有训练。

权重：`outputs/1030-warmup2k-to-full10epoch/checkpoint_step_21480.safetensors`，归档双端SHA256已通过：`b141060b4ab33b0a2542effad452203b2ee5a5d287e5fd38ef092abe7326de7e`。本轮引用已有完整性验收，未重新读25GB文件做哈希。

独立输出：`outputs/robotwin_eval/appearance21480_20261030T124235Z`；PID395705；仅使用当前A100服务器物理GPU5、CUDA逻辑0，原GPU0–3训练和远端H100八卡训练不操作。准入40GiB空闲、PyTorch allocator上限整卡45%，每次生成前空闲低于8GiB则只中止本诊断；不终止其它作业。

## 样本与协议

复用已审计固定GT输入，3任务adjust_bottle/place_shoe/beat_block_hammer × Clean/Randomized两域 × 每episode两个窗口，共12窗口/6episode。窗口有相关性，描述性quantile不能冒充总体P95估计或闭环SR。输入与GT复制到独立目录并SHA256记录，不改原数据。

5条件：原图、温和光照(brightness1.15/contrast0.9/gamma1.1)、强光照(1.8/1.4/0.65)、强sensor(blur1.8px+noise12/255)、原PatchWAM完整外观recipe压力测试(p强制1，其余配置同训练)。均保持位置/视角几何/指令/proprio；强过曝/模糊可能丢失目标信息，PatchWAM Hue/AdaIN可能改变颜色语义，不自动视为可用于alignment的语义保持配对。

每条件Full与Write-Off各一次，总120次生成，另原图首次重复一次做逐位一致性。seed42、10step同步、32步80D统一动作/20D物理EEF、eager、DIT cache关闭、不解码视频；关闭Write时保留Read/EA模块，使用已测试的实例级WriteProbe，仅临时门控并恢复beta。

## 读出与验收

捕获第一denoise step的Action hidden block8/20、三tap Ground/Relation/Summary/Plan tokens、Action FM velocity(只计有效20D槽位)、Ground/Relation/Plan GT指标、32步预测物理动作及真实Write residual比例。固定起始action/video-future噪声和timestep做hash核对。漂移分1–8/9–16/17–24/25–32段，连同GT误差对照；Full-Off的鲁棒性差异需要和准确性一起解释。

独立入口`benchmarks/robotwin/sidecar_appearance_diagnostics.py`，不改生产model/trainer/data reader/服务。CPU新增输入扰动/特征指标检查与已有WriteProbe回归共14 passed。GPU真实结果仍在执行，首次repeat/完整120生成/Sidecar参数fingerprint以及非gate参数version不变尚待验收。当前不声明外观敏感性大小或alignment能带来收益。

`progress.json`为进度；`summary.json`每次生成增量更新；`results.json`及`features/*.npz`留存动作/特征；`images/*.png`供人工审核扰动；完成须有`complete.json`且无failed.json。

下一步先检查扰动强度与语义保持性，再报告各条件动作drift及GT误差、Full/Off、特征响应。若显著敏感优先评估Action-expert hidden跨外观alignment，允许梯度经MoT训练主干与Sidecar；不把新的loss限制在Sidecar。XA短训尚未实现/启动，不把仅alignment触点在Action解释成冻结Video或不训练主干。


## 2026-10-30T12:56:47.889511+00:00：完成与结果

120/120真实GPU生成完成，另1次原图repeat逐位相等；同窗口增强/原图起始Action与未来Video噪声及timestep一致。Sidecar SHA256 fingerprint、非gate参数版本不变；无optimizer、无训练、无仿真动作。进程395705已正常退出，释放本次模型。CPU新增第6项Capture forward透明性也通过，共15项不同测试。完整结果：`outputs/robotwin_eval/appearance21480_20261030T124235Z/RESULTS.md`，机器可读`analysis.json`与`condition_metrics.csv`，人工扰动图`appearance_contact_sheet.png`，原始120动作/GT/读出在`results.json`和`features/`。

| 条件 | Full漂移mm | Write-Off漂移mm | Full前8步漂移mm | Full对GT误差mm |
|---|---:|---:|---:|---:|
| 原图 | 0 | 0 | 0 | 12.898 |
| 温和光照 | 3.248 | 3.330 | 2.117 | 15.043 |
| 强光照 | 14.218 | 14.242 | 4.715 | 25.293 |
| 模糊+噪声 | 5.836 | 5.837 | 3.344 | 16.632 |
| 完整PatchWAM压力 | 6.436 | 6.430 | 3.443 | 16.272 |

漂移是增强与原图预测chunk的距离，不是执行误差；原图0是定义，不是无GT误差。强光窗口中位数2.641mm，Randomized place_shoe初始窗口103.763mm显著拉高均值，除去该窗口其余11窗口均值6.077mm；不能概括所有场景普遍崩溃。温和光照中位数1.369mm，仍有19.245mm的adjust_bottle Clean窗口，值得扩大独立episode验证。

Full-Off五条件60窗口对的平均距离1.114mm，实际Write增量/Action hidden范数比均值0.2703%、最大0.8095%。关闭当前Write没有消除本次外观敏感性，支持把新约束放在Action/Main路径；但Off使用已经Sidecar训练过的主干，不能据此声称Sidecar训练无效或其梯度作用小。

第一denoise step Action block20相对L2响应约1.49%/2.97%/3.06%/2.41%（温和/强光/sensor/PatchWAM），block8约0.50%/1.24%/1.52%/1.19%；深层读出更敏感，但不能仅按不同层特征标度比较就证明某层alignment最优。

下一步最小候选：先单层Action block20跨外观alignment，clean teacher stop-gradient，augmented student，配对相同物理状态、指令、动作噪声和timestep；梯度沿MoT的Video K/V回到主干并可更新Sidecar，不只训练Sidecar。保留EA/原损失/架构，不同时改detach、注入强度、路由。可从弱权重如0.05起筛，但须看梯度量级和真实显存；EMA FP32额外空间仍未验收，不直接承诺原batch可容纳。

先使用经人工确认保留任务语义的温和配对，强过曝/模糊/Hue/AdaIN继续作为压力评测；与同起点同预算augmentation-only分支比较，然后配对闭环验证。当前诊断属于旧2K→10epoch最终权重，不能替代远端当前官方C2R→2K→5epoch分支自身的诊断。XA core尚未改动或启动，本轮结果仅支持开展受控实验，不证明效果。


## 2026-10-30T12:58:38.282176+00:00：checkpoint gate与有效写回核查

直接safe_open读取21480权重：三tap7/15/23的Write beta分别-0.0026397705、-0.0025177002、+0.0037231445；Read alpha分别0.0617675781、0.1259765625、0.0130004883。beta是带符号线性乘数，不是sigmoid概率，负值不代表禁用。

每tap600次实测（12窗口×5外观×10去噪步），有效写回增量/Action hidden范数均值依次0.2482%、0.1976%、0.3651%，最大0.5940%、0.4137%、0.8095%。门控前projection/hidden范数比均值0.950、0.774、0.977，说明门控前幅度并未整体塌缩，直接写回的小幅度主要来自beta缩放。BF16加法后实际改变元素比例平均68.8%、69.9%、76.5%，不支持“全部被量化吃掉/根本没写入”。以上仅限本checkpoint/本样本，不能断言提高gate会提升闭环。证据`outputs/robotwin_eval/appearance21480_20261030T124235Z/gate_strength.json`。

补查向量而非只比较范数：增强减原图的Full/Off漂移向量之差，温和/强光/sensor/PatchWAM平均1.140/1.250/1.280/1.263mm；因此不能说Sidecar完全不改变外观响应。关闭注入使预测平均变化1.114mm，但没有明显降低本次漂移范数。

候选后续诊断：保持权重和噪声不变，独立比较Write scale=0/1/2/4，同时检查对GT误差、早期动作和语义指标；若放大有效再设计训练门控方案，若无益优先约束Action/Main跨外观表示。尚未执行scale sweep或改变训练。


## 20261030T130215Z：用户授权gate倍率诊断

复用21480 checkpoint、已保存12窗口×5外观图、seed42和10step eager，独立输出`outputs/robotwin_eval/gate21480_20261030T130215Z`。Write倍率0/1/2/5/10/100共360生成；不改Read alpha，不改变checkpoint或现有训练。复用WriteProbe实例内gate缩放/异常恢复；0/1的120生成必须逐位复现上轮Full/Off；100倍首次重复需逐位一致，输出非有限则本诊断失败且保留证据。GPU5容量门槛同前，不抢占其它作业。比较每倍率相对该倍率原图的动作漂移及独立GT误差，防止把输出塌缩当鲁棒。Sidecar fingerprint/非gate参数版本最终验收。强扰动语义及12相关窗口的闭环外推限制保持。生产model/trainer未修改，仅旧诊断入口新增可选路由至独立倍率诊断。


## 2026-10-30T13:16:28.330383+00:00：Write gate倍率诊断完成

独立输出`outputs/robotwin_eval/gate21480_20261030T130215Z`，360/360真实GPU生成完成，推理用时474.94s（不含CPU加载），进程403342已退出。×0/×1的120生成与上一轮动作逐位相同，×100首窗口重复逐位相同；所有输出有限，Sidecar fingerprint和非gate参数版本不变，86项输入/源码哈希不变。11项CPU检查通过。无训练/optimizer/仿真/部署，checkpoint未改。

12窗口×5外观条件等权，平均双臂32步物理EEF位置误差（mm）：

| Write倍率 | GT误差 | 相对×1 | 前8步GT误差 | 实际写回/hidden均值 |
|---|---:|---:|---:|---:|
| ×0 | 17.275 | +0.28% | 9.714 | 0% |
| ×1 | 17.227 | 基准 | 9.666 | 0.270% |
| ×2 | 17.291 | +0.37% | 9.747 | 0.537% |
| ×5 | 18.219 | +5.76% | 10.781 | 1.339% |
| ×10 | 20.738 | +20.38% | 13.443 | 2.660% |
| ×100 | 92.193 | +435.15% | 86.481 | 24.239% |

强光漂移×1/2/5/10/100分别14.218/14.242/14.261/14.482/16.664mm，未随gate放大改善。×2有22/60组GT误差更低，×5为6/60，×10为2/60，×100为0/60；均值不是所有窗口同向，相关小样本不做显著性断言。×100原图GT误差12.898→89.256mm，说明不是仅放大对外观的反应，也破坏原图预测。

解释：gate小但并未失效，放大后有效写回按预期增加（×100均值24.239%、最大62.376%），且动作相对×1平均改变85.345mm。当前权重下直接增大gate没有整体收益；×2接近原方案，×5起退化更清楚。Gate强度小不是“应该直接强制放大”的充分依据；可能是当前模型校准后的残差幅度。不能据此证明Sidecar表示无用、存在容量上限，或强gate重新训练必然失败。

下一步保持训练与部署×1；若探索鲁棒性，优先比较Action hidden跨外观alignment与同预算augmentation-only，而非同时改gate。若未来研究更强注入，应另立训练消融，不能把本次推理倍率结果等同训练效果。闭环收益仍未验证，强扰动语义保持和扩大独立episode样本仍未解决。完整表格/每窗口指标：`outputs/robotwin_eval/gate21480_20261030T130215Z/RESULTS.md`、`condition_metrics.csv`、`window_metrics.csv`。
