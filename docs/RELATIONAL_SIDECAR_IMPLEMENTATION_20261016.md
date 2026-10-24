# Multi-Layer Relational Sidecar 实施与交接（2026-10-16）

实现依据：OpenWAM_alpha_MultiLayer_Relational_Sidecar_Codex_Impl.md。保留旧Lite与原版，新增独立版本；本次是工程搭建，不是效果评估。

## 架构和数据合同

- 原版：`model=dual_system`；旧版：`model=dual_system_ea_rsig_lite`；新版：`model=dual_system_relational_sidecar`。旧版checkpoint继续使用旧模型配置，新版不能宽松加载旧EA权重。
- 新模块注册于architecture.sidecar，15个token在block7/15/23之后持续更新；只读runtime H×W首帧，额外15token不进入MoT，不改mask/RoPE/QKV。
- 从当前归一化80D proprio+80D dimension mask和masked text mean初始化，不使用oracle EEF/arm/context。Action仅作为写入Query，不参与Z更新。输入维度mask从reader或部署normalizer恢复；GT130D独立用于loss。
- Ground4，Relation4，InteractionSummary1，Plan6。Summary没有百分比进度监督。全部stage更新全部有效token；最终三loss，stage1ground/stage2relation有0.1深监督。read alpha=.01；write beta=0，投影保持非零随机。
- 使用请求数据集 `robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015/raw/aloha`；不读取Piper，不以LeRobot 14D qpos充当EEF。原始HDF5不改写；source size/mtime、label/stats hash绑定和prepared startup复用现有实现。
- 新labels/EEF stats从此批HDF5独立构建。32绝对EEF动作、9视频、Plan8/16/32，每角色local frame，wxyz→xyzw/列Rot6D，原EE/TCP参考点分离，local-X .12m合同沿用已审计映射。
- 每视角归一化bbox在loss中显式映射到L形全画布，不再固定14×14。goal_valid=0不可形成可见goal监督；无效arm/track/未来尾部均mask。RGB沿用legacy decoder，准备程序检查全部episode三视角首尾标记并拒绝XPL-RGB1；不声称全JPEG逐帧验收。

## 一次性准备和版本切换

```bash
cd /remote-home/jinminghao/OpenWAM
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/prepare_relational_sidecar_data.py \
  --dataset-root /remote-home/jinminghao/datasets/robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015 \
  --output-dir outputs/sidecar/aloha_4350
# 准备程序拒绝覆盖已有目录；已准备好后直接使用ready manifest，不重复执行。
CHECK_ONLY=1 EA_STAGE=warmup bash scripts/train_relational_sidecar.sh
# 旧Lite入口保持原样：
CHECK_ONLY=1 bash scripts/train_ea_lite_4gpu.sh
```

默认新manifest：`outputs/sidecar/aloha_4350/manifest.ready.json`。换机器用EA_SIDECAR_MANIFEST覆盖，复用 `relocate_ea_lite_manifest.py` 迁移路径并保留纳秒mtime；不要直接放宽校验。完整代码、Foundation、raw HDF5、准备好的labels/stats/ready manifest和环境锁需分别搬迁。新版本独立配置将存入部署checkpoint的model_config，不能仅替换权重不替换config/normalizer。

## 四卡20次真实更新命令（待在空闲A100执行）

```bash
# 160 microsteps / accumulation8 = 20 optimizer updates；每40 microsteps保存完整状态。
CUDA_VISIBLE_DEVICES=0,1,2,3 CHECK_ONLY=0 EA_NUM_GPUS=4 \
  EA_GPU_NAME=A100 EA_GPU_PROFILE=80 EA_STAGE=warmup \
  EA_STOP_AFTER_OPT_STEPS=20 EA_OUTPUT_DIR=outputs/sidecar/smoke_warmup \
  bash scripts/train_relational_sidecar.sh
# 确认warmup稳定、保存恢复通过后，可新开Action适配；并非自动延续上个warmup权重：
CUDA_VISIBLE_DEVICES=0,1,2,3 CHECK_ONLY=0 EA_STAGE=action_only \
  EA_STOP_AFTER_OPT_STEPS=20 EA_OUTPUT_DIR=outputs/sidecar/smoke_action \
  bash scripts/train_relational_sidecar.sh
```

要接续warmup权重或完整状态，分别显式覆盖 `training.finetune_ckpt_path=<保存weights目录>` / `training.resume_ckpt_path=<完整状态目录>`，resume时同时设 `training.finetune_ckpt_path=null`；不要把warmup当普通SFT。`EA_EXPERIMENT=A/B/C` 分别baseline/aux-only/read-refine-write，初始调试默认bs1、acc8、BF16、ZeRO2、checkpointing、LR1e-4和相同ColorJitter；正式对比应统一训练阶段、global batch/总更新/数据/控制器。Video部分解冻、LoRA、action-plan-align不实现默认流程，不宣称硬件容量。

逐rank `diagnostics_rank*.jsonl` 含每项aux loss、三read/write gates、所有新参数梯度范数、三层token梯度范数、microstep时间、显存allocated/reserved峰值；acc8的更新时间应汇总8个microstep，不把单个microstep时间叫optimizer step时间。输出目录中的完整Accelerate状态与部署weights各有不同用途。

## 部署

`configs/deploy_relational_sidecar.yaml`关闭compile与DiTcache、CFG1、sync、32动作/9视频。沿用官方部署入口和相同绝对EEF/controller，不发送ea_targets或旧ea_effectors；当前state由normalizer变成80D proprio，当前维度mask由normalizer散射表恢复。每次MoT forward都从当前观测重建Z，三处stage均执行。调试不能用旧Lite专用请求字段或旧原生14D/40步入口。

## 验收状态

实施与测试进行中；最终数字和证据路径完成后补在本节。CPU tiny检查不能替代全量Foundation BF16、四卡ZeRO保存恢复或闭环验证。当前既有GPU评测作业不操作。

旧版远端备份：`https://github.com/Jamee11/MyWAM/tree/backup/ea-rsig-lite-pre-sidecar-20261016`，commit `cb37290f6d5c80ba3434b0d4f1c345d1ee4f9392`。远端tree回读645个文件（含初始化provenance）；本地完整代码归档644个普通文件，路径 `outputs/sidecar/backups/pre_sidecar_complete_20261016.tar.gz`，旁置SHA256。新版在独立 `feature/multilayer-relational-sidecar` 分支，main暂保留旧版。Git CLI读远端成功；写入使用已连接GitHub工具，本机VS Code旧凭据socket仍不可用，公开仓库不会自动给予Git CLI推送凭据。

只读源清点确认Aloha50任务、2500原始HDF5、35,417,456,051 bytes，所有任务均50轨迹，见 `outputs/sidecar/source_inventory.json`。此数字是文件清点，不代表本次EEF/标签重建已经通过。11个旧Lite模型/配置/启动文件已与备份逐字节相等，见 `outputs/sidecar/old_lite_preservation.json`。

首轮CPU测试因输入mask形状失败：原主干 `_proprio_sample_mask` 内部接受 `[B,1,80]`，测试与初版部署传了 `[B,80]`。已修正fixture和新架构的路由形状，保留维度mask用于Z编码；原主干不改。首轮日志 `/tmp/openwam-sidecar-tests.log`，复测 `/tmp/openwam-sidecar-tests-v2.log`。首次依赖冷加载受共享文件系统I/O拖慢，不能把该等待时间当训练步时或新增模块计算耗时。

限制与后续比较：受保护首帧在各层始终只含当前RGB/Text/Proprio。多层读取增加表示深度，并不自动提供历史速度、真实未来轨迹或准确接触状态；Plan仍依赖监督学习，效果须做配对闭环验证。alpha=0只停止Z从该层新增视觉信息，Refine仍变换Z，空间grounding head仍需该层视觉keys。跨训练阶段应以weights finetune重新建立optimizer；full-state resume要求相同训练阶段与参数组，不能直接把warmup的optimizer状态套到action_only。


## 2026-10-16 工程调试进展（追加证据）

真实Foundation CPU warmstart已通过：继承2089个tensor、没有重置原80D policy heads；新增140个EA tensor、24,290,323个Sidecar参数。完整模型12,431,044,415参数含冻结编码器，warmup仅训练Sidecar。证据：docs/evidence/relational_sidecar_20261016/foundation_cpu.json。此结果只验证加载，不验证GPU训练容量。

最新CPU组合回归40 passed、1 failed（921.03s）。新版Sidecar全部用例已通过：FP32/BF16与checkpointing开关的零写入恒等、五种独立因果扰动、三阶段完整loss/两次optimizer更新、梯度、严格权重加载、真实tiny generate以及CPU Accelerate完整状态/RNG恢复。失败是旧Lite BF16 fixture对bool调用is_floating_point；已添加Tensor类型判断，新版模型不受此fixture错误影响，远端完整复验待结果。冷依赖导入等待不当作训练耗时。

指定新集合的Aloha准备已完成：50任务、2500轨迹、552275帧、549775训练窗口，labels/stats与manifest.ready.json均位于outputs/sidecar/aloha_4350。实际reader抽查每任务首/中/末150窗口，80D EEF动作/当前state与原始endpose归一化往返通过；九帧视频、掩码及8/16/32时间锚一致。A与C在相同随机种子下的增强图像/动作/文本逐元素相同。证据：docs/evidence/relational_sidecar_20261016/data_reader_audit.json；完整样本清单保留在outputs/sidecar/data_reader_audit.json。这不代表逐帧视觉质量或闭环效果通过。

原始代码字节保存在不可变backup分支和完整tar归档。工作目录随后集成了CI排版/import清理、旧FK资产路径可迁移性修复及旧BF16测试fixture修复，因此先前11文件逐字节相等只描述CI清理前的快照，不能当当前目录字节一致证明；模型/动作/标签数学保留。备份最终commit为364a95f44660420cb30e0ed0a4bba8482a248281，包含Cosmos gitlink。

用户要求不再纠结Actions。CI修复已存fix分支，主任务转为真实GPU调试：现场仅5/6两张A100-80GB完全空闲，已启动独立两卡warmup短测（bs1/acc1/2 optimizer updates/第2步完整状态保存）。日志/tmp/openwam-sidecar-2gpu.log，输出outputs/sidecar/smoke_2gpu_warmup_20261016。未占用其他卡，不操作已有作业。当前仅通过GPU空闲预检，未把启动/加载当更新成功；四卡20步/吞吐和闭环仍待验证。


### 完整CPU与数据验收结果更新

feature提交8af2cda的完整标准CPU回归已全部通过：2174 passed / 15 skipped / 21 deselected，132.91秒；lint及train/eval import boundary也通过。run：https://github.com/Jamee11/MyWAM/actions/runs/38880662004。旧BF16 fixture修复已复验，不再仅是待验证状态。数据full preflight通过50任务/2500轨迹/552275帧：源size/纳秒mtime、全部label SHA256、stats SHA256及Foundation80D头合同均匹配。旧Lite11文件当前与原始归档比较：配置/启动器字节一致，Python可有import/排版差异但排除import的计算AST相等，逐文件证据见docs/evidence/relational_sidecar_20261016/old_lite_preservation.json。GPU两卡更新和恢复仍在执行，四卡容量/稳定训练/闭环效果未验证。


### 2026-10-16 UTC：Inductor非必要启动池
双卡首次启动现场可见每rank一个torch/_inductor/compile_worker/__main__.py --workers=32子进程，伴随GPFS I/O等待。读取当前torch async_compile.py源码确认TORCH_WARM_POOL=0只跳过import-time AsyncCompile.warm_pool；新Sidecar训练入口设置此变量，模型forward/loss/精度/优化器不改。shell语法及AST核对通过；当前已运行进程不会因此动态改变，不重启已有训练。未宣称消除全部冷启动时间；依赖导入和权重加载仍存在。GPU梯度更新/保存恢复仍待真实验证。


### 2026-10-16T13:17:09.200360+00:00：用户授权新Sidecar四卡正式10epoch训练

使用共享盘.venv，GPU0–3，指定4350集合内Aloha50任务2500轨迹，prepared manifest不重建。joint全参，ColorJitter .2/.2/.2/0、BF16/ZeRO2/checkpointing/fused AdamW、base/action/video LR1e-4、seed42沿用wrist020正式方案。旧Lite bs64曾近80GB，本次新增三层模块采用bs32/acc2/global256，既有小区间训练不能保证新batch容量；不声称已验证joint最大batch。epochs10/max_steps null，save4000micro=2000updates，最近2组+最终完整状态保留。

新增独立配置、正式入口及launcher；原trainer/旧Lite启动器不改。入口不装grad hooks，只保存轻量per-rank指标。CPU权重与optimizer准备完成后、DeepSpeed大规模GPU迁移前四rank同步；rank0写handover/cpu_ready.json，至少90秒并等待四张卡均free>=70000MiB才放行。初始CUDA/NCCL仍会产生少量显存，不把此阶段称为完整GPU迁移；用户自行结束旧进程，本agent不发kill。门槛逻辑模拟验证了90秒与最后一张卡未释放时持续等待，Ruff/shell检查通过。

启动记录：outputs/sidecar/aloha50_C_joint_10epochs_20261016T131709Z/launch_record.json；tmux=sidecar10e_20261016T131709Z；日志=outputs/sidecar/aloha50_C_joint_10epochs_20261016T131709Z/train.log。当前准备启动，首次joint更新/显存吞吐/保存恢复和闭环仍待真实结果。


### 2026-10-16 四卡joint正式训练已完成首轮更新

真实启动13:17:43 UTC，CPU-ready13:50:59（33分16秒），通知用户后由用户释放旧训练，13:52:33显存门槛放行，13:52:54四rank完成DeepSpeed包装。首个micro约13:53:36，首个optimizer update约13:53:43，合计约36分钟，含94秒交接等待。上次共享盘双卡warmup首次更新30分20秒，不能当本次四卡joint的严格性能对照。

四rank均完成至少7次真实optimizer updates；全部已记录指标有限，grad norm正值，三个write gates均已由0更新。当前batch32/rank、acc2、global256、joint可训练6045.4M参数、ColorJitter开启、10epochs共42960micro/21480updates。剔除头两micro，最近12个micro平均7.104秒，即约14.21秒/update；NVML整卡使用约64.1–65.2千MiB、GPU利用率100%。仅初期小窗口，非最大batch/长期稳定性结论。按现速10epoch纯训练约84.8小时，未含保存/数据波动/中断。

证据：docs/evidence/relational_sidecar_20261016/formal_4gpu_startup.json；原始metrics_rank0–3.jsonl位于outputs/sidecar/aloha50_C_joint_10epochs_20261016T131709Z/runs/2026-10-16_13-50-52。进程继续运行；本正式run尚未到首次save4000micro，完整checkpoint保存恢复、20次以上稳定更新和闭环效果仍待验收。未操作旧进程kill、未换临时环境。


### 2026-10-18T06:10Z：Aloha四卡正式训练中期检查

现场四rank同步推进至至少micro21737/42960、optimizer10868/21480（约50.6%，已完成5epoch、进入第6epoch），累计真实训练约40小时17分。GPU0–3整卡约63–64GiB、利用率97–100%；近100micro均值6.612秒，即约13.22秒/update，预计剩余纯训练约39小时。rank0全部历史指标有限，最近100条四rank指标亦有限；日志未检出Traceback/OOM/Nonfinite/NCCL error。最近100条均值total0.18503/video0.08868/action0.000612/EA0.09574，训练loss下降不代表闭环成功。

当前保留step16000和step20000两组部署weights，各24862363126 bytes；两组均有DeepSpeed model、四rank optimizer、四rank RNG、scheduler及trainer_state完成marker。step20000 marker确认10000次optimizer更新/epoch4。解决本配置的多epoch连续更新和真实完整状态写出问题，但未实际重启加载本run状态，不将文件存在称恢复通过；未做此模型闭环。证据docs/evidence/relational_sidecar_20261016/progress_20261018.json。本次仅只读监控及追加文档，不改变训练代码/配置/进程。
