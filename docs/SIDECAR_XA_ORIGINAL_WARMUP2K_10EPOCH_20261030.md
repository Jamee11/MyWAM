# 原始warmup2K → Action跨外观alignment联合10epoch

创建：2026-10-30T13:48:28.569275+00:00。用户明确授权在本地增量实现、测试后同步H100；排除今天C2R→Sidecar2K；保存间隔5000 optimizer updates，强度适中且保留默认旧版。

## 版本、初始化与隔离

修改前505文件完整源码备份已推送：https://github.com/Jamee11/MyWAM/tree/backup/pre-alignment-20261030 ，commit `24328162a20dabb950c6bbf4c1ed44951673cb03`。本地实现位于experiment/sidecar-xa-warmup2k-10epoch-20261030。下载缓存/虚拟环境/权重不属于源码备份。

唯一生产初始化：`/mnt/100T/lixiangyu/cuigm/bitdistill/OpenWAM/assets/openwam_ckpt/openwam_alpha/Sidecar-PatchWAM-Warmup2K/checkpoint_step_4000.safetensors`。这是历史acc2的2000 optimizer updates；文件24862363126 bytes，传输清单SHA256 `49b6c013eb910b9d0bc3071ec40a38915631d8a54dd6b73b1942d6968383835b`。部署前重新校验。不是21480 final，也不是新C2R→2K。

本地真实smoke用同源原始文件加传输包内7项自包含metadata/tokenizer；原训练源config仍有占位路径，已保留未改。首次smoke因该路径失败于模型初始化、未更新参数，完整日志保留。

远端拟部署完整代码快照到项目内`.experiments/xa_original_warmup2k_<UTC>`；runtime/assets/datasets链接到本项目已有目录，生产输出仍在本项目outputs/sidecar；不覆盖aigc24正在运行的代码和输出。当前已只读确认aigc14/29/32整八卡空闲，选择aigc14生产、aigc32单卡诊断；真正启动前再次准入，未知作业不终止。

## 论文依据及本实现差异

原文：https://arxiv.org/html/2610.26813v1 ，Appendix A Table7明确0-based student8/EMA teacher20，Action stream，两层Linear-GELU-Linear宽512，EMA .9999 FP32，noisegap Uniform[0,.2)，固定lambda .05。此前20→20仅为简化试验提案，并无同层最优的直接论文证据；本用户授权版本改用有配置依据的8→20。论文的层编号是经验配置，不是对本Sidecar任务的最优性证明。

新增L_XA=mean(1-cos(projector(h_student_aug_8),stopgrad(h_EMA_nominal_20)))，屏蔽padding Action token。Teacher同物理clip、指令/proprio、同基础FM噪声，teacher sigma=max(student sigma-delta,0)，当前帧保持clean-prefix。EMA涵盖可训练WAM+Sidecar，不复制冻结T5/VAE，不加入optimizer或部署state_dict。

用户要求loss贡献略小于EA：新增平滑系数控制，以实际加权EA总损失（含deep supervision）0.8倍为目标；初始系数.05，上限.5，统计EMA .98，单更新变化≤5%，前200 optimizer updates渐入。原始/加权XA、实际权重及XA/EA比全部记录。系数上限优先于强行凑目标；标量贡献相近不等于梯度贡献相近，不能通过除以当前loss强行无限放大。

Student与teacher前向BF16，特征归一化/loss和EMA shadow累计FP32，EMA每个成功optimizer update更新一次（非microstep），且teacher前向保持并恢复模块training flag及RNG。显式hidden返回兼容gradient checkpointing重算，禁止生产路径monkey patch捕获。

## 数据、训练与保存

使用已有50任务2500条Clean HDF5及EA labels。新增retain_clean_video默认false，启用时在任何PatchWAM之前复制名义clip；动作、标签、相机布局/缺失黑槽和裁剪几何不变。

新XA配置使用中等外观扰动：p.9，brightness/contrast±.3、saturation±.15、gamma.8–1.25、exposure±.25、temperature.05、noise std.015、blur sigma.4–1.2 p.25，跨帧共享参数。禁Hue/AdaIN/FFT避免颜色指代变更；强过曝等仍作为压力诊断，不能自动视为合法训练一致性配对。增强相对旧完整PatchWAM有所变化，因此新版本与历史10epoch之间不能归因成纯alignment效果；若要因果结论需同增强的lambda0对照。

10epoch，预期21480 optimizer updates，新optimizer/计步0，无重复2K冻结，无额外ramp/heads detach/gate放大。保留旧peak LR Video/Action/base1e-4、Sidecar5e-5，标准5%LRwarmup1074+cosine。8×16acc2保持global256，给EMA显存余量；H100真实显存/吞吐尚待验收，不预先承诺一天。

每5000 optimizer updates保存，文件命名也是optimizer编号：5000/10000/15000/20000/final21480；保留5份。当前weights-only导出不含完整optimizer/RNG/EMA历史，不能精确resume；显式禁止XA走resume入口，若故障后finetune需记录fresh teacher/optimizer及LR重启，不伪装无缝恢复。

## 验收、部署与中间诊断

本地首先通过tiny真实Wan/Action/Sidecar测试：默认关闭数值/梯度/RNG/参数键一致、teacher停梯度、padding、FP32EMA小增量、8→20读出梯度路由、gradient checkpointing、旧权重仅允许新增projector、模型权重往返、5000 optimizer保存单位与独立停止不改变LR horizon。随后四空闲A100做DeepSpeed真实两次更新smoke，八H100以生产batch做10update preflight；只有finite XA/EA/main/grad和八rank共同10更新、显存余量及进程成功通过才正式启动。

监督脚本：scripts/supervise_sidecar_xa.py；启动：scripts/train_sidecar_xa_warmup2k_8gpu.sh；配置：configs/train_sidecar_xa_warmup2k_8gpu_10epochs.yaml。新增可选项缺省不启用，新版代码与旧配置兼容；旧训练配置未改。

中途诊断：scripts/watch_sidecar_xa_diagnostics.py在独立空闲H100单卡等候5K间隔权重；先诊断同一旧warmup2K起点，再依次5000/10000/15000/20000/21480。复用12窗口/6episode/3任务/两域，每权重120生成（原图+四外观×Full/Off），相同noise/seed，权重名固定、哈希记录。输出comparison.json含每个checkpoint的GT误差与appearance漂移，不以cosine loss或离线漂移替代闭环SR。不因训练期间尚无checkpoint而抢占其八卡。诊断输入/GT也放在本项目下；不下载HF资产。

## 当前状态

方案与代码已落本地；CPU首轮22+相关74项通过（后续新增metric检查与回归见更新）。真实A100 smoke正在执行；远端独立代码同步、八卡preflight及正式生产/诊断观察器尚未启动。后续证据逐项追加，不能把实现完成当成训练完成。


## 2026-10-30T13:56:33.227288+00:00：本地检查与部署准备

远端旧4000权重已重新SHA256核对，确认为49b6c013eb910b9d0bc3071ec40a38915631d8a54dd6b73b1942d6968383835b；原warmup统计与当前训练统计的EEF min/max逐元素相等。真实smoke第二次在首次forward后、更新前发现unified EEF padding为[B,32,80]，已修为按任一有效维度判断有效时间步，并补齐回归；原失败日志保留。第三次四A100 smoke正在加载同一旧权重。部署元数据本地outputs/sidecar/xa_setup_20261030/deployment.json，计划aigc14生产/aigc32单卡诊断。

8→20的alignment梯度直接作用于student第8层及其之前的Action/Video关联路径与首次Sidecar注入；teacher第20层停梯度，后续层仍由原FM/EA损失训练。不能宣称此loss直接更新所有深层模块。


## 2026-10-30T14:06:35.472420+00:00：GPU显存修正与H100用卡预算

四A100第三次真实smoke已共同完成1 optimizer update（main/EA/XA有限，grad有限，峰值约70.2GiB），第二update在创建全模型BF16 teacher副本时OOM。不是loss数值错误。已改为直接引用同卡FP32 EMA shadow，在teacher前向启用逐算子BF16 autocast且关闭AMP权重缓存，避免额外整模型BF16副本；CPU BF16真实tiny验证通过，第四次真实smoke在执行。旧失败输出全部保留。

用户新增预算约束：只允许这次新增一个8-H100训练与即将结束的C2R暂时重叠，之后尽量同时仅8H100。取消aigc32的GPU诊断安排（未启动任何GPU进程），诊断改为本地A100。新observer通过现有SSH master轮询已完成marker，使用独立旧权重副本seed做rsync可续传差分，远近SHA256一致后才正式命名并评测。SSH无密码/token写入脚本，轮询维持会话，若会话丢失则保留partial等待恢复。

新增checkpoint完成marker仅对新optimizer计步配置启用，防止评测读取未写完文件。准备固定版本部署验收；CPU综合107项通过，新增AMP精度用例与marker/计步回归也通过，未以此声称八卡生产显存已验收。


## 2026-10-30T18:02:00.409017+00:00：最终串行接续方案与启动修复

用户确认不再占用另一整节点，XA接在aigc24当前C2R完成后使用同一8H100。aigc14/29/32已被其他用户占满，14上仅等待的XA控制器已退出，未触碰其作业。C2R实查8rank10269/10740，10000权重已保存；GPU仅8个train_sidecar_metrics rank，没有额外显存占用进程，因此没有杀任何C2R训练/辅助进程。

已定位重启失败包含固定29504端口被上次启动短暂占用，以及节点主机名与hosts映射不一致的启动延迟。新XA launcher使用--standalone --local-addr=127.0.0.1，由TCPStore自动分配空闲端口；单节点GLOO/NCCL使用lo并禁用IB，不修改系统hosts。前置C2R完成验收+所有8卡释放后才开始XA10步，再自动从原传输warmup4000重新加载进入正式10epoch，不等待效果分析/人工操作。若16/acc2仅因OOM失败，自动尝试8/acc4；不改变global256或LR曲线。

本地checkpoint observer不占H100；已启动原warmup2K的120生成基线诊断（结果保留），后续继续等待XA5000间隔权重并差分传回本地A100。尚不宣称XA正式更新已经完成或效果有收益。
