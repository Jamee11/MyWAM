# OpenWAM-alpha + EA-RSIGv3

独立目录 `/remote-home/jinminghao/OpenWAM`。项目规则见原样复制的 `AGENTS.md`。
分析、迁移合同、来源和验收证据见 [迁移分析](docs/2026-10-13_OpenWAM_alpha_EA_RSIGv3迁移分析.md)、[隐患](docs/思考与隐患.md) 和 `MODIFICATION_LOG.md`。

当前状态：框架工程验收完成。独立环境和完整8文件权重校验通过；真实foundation继承2085 tensors，仅重置4个native政策头并新增73 EA tensors。真实数据CPU AdamW单步通过；官方双卡BF16/ZeRO2原生fused AdamW一步、完整保存、该产物在GPU上的官方严格重载/两步推理均通过，进程exit0。源数据没有重写；长期训练、闭环、真机和适配效果等价均未验证。

此扩展保留 Alpha joint-self-attention 专家，用单独 detached 当前观测 pass 产生 EA grounding/relation/progress/plan；追加 action-context tokens，并只在训练未来视频 tokens 注入残差。原版基线配置仍可使用。当前 native 14D qpos/40步/130D双轨监督属于迁移配置；80D Alpha 政策头适配及其他变化都显式记录，不称原版协议复现。

工程通过不保证实验效果等价。政策头/动作表示、40步5帧、Wan观测与EA特征几何、联合注意力融合都可能改变成绩；EA仍不是动作预测的唯一路径。效果需在相同底层数据、预算和多seed下比较原Alpha协议、native14无EA基线和EA版本，并用闭环成功率判定；当前没有该对照证据。

环境重建：

```bash
cd /remote-home/jinminghao/OpenWAM
bash scripts/install_ea_environment.sh
source .venv/bin/activate
```

权重下载（用户已批准镜像返回的 CDN 直连，禁止代理/VPN；此脚本逐重定向检查且禁用环境代理）：

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/download_assets/download_ea_foundation_mirror.py --allow-mirror-cdn
```

脚本锁定首次 inventory revision、保留 `.partial`、校验大小及 upstream hash。目标在 `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model`。权重未完成时训练入口会拒绝启动。

真实数据读取预检及 CPU 回归：

```bash
python scripts/preflight_ea_rsig_v3.py \
  /remote-home/jinminghao/DiT4DiT/datasets/robotwin2_ea_rsig_v3_aloha_10000_schema5_20260928/EA_RSIG_v3/Clean/adjust_bottle
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  python -m pytest -q tests/test_ea_foundation_mirror.py tests/test_ea_rsig_v3_core.py tests/test_ea_rsig_v3_integration.py
```

完整 foundation CPU 逐键加载检查（只有下载完成才能运行）：

```bash
python scripts/smoke_ea_foundation.py --dataset \
  /remote-home/jinminghao/DiT4DiT/datasets/robotwin2_ea_rsig_v3_aloha_10000_schema5_20260928/EA_RSIG_v3/Clean/adjust_bottle
```

`--train-step --device cpu` 执行真实数据的一次 FP32 AdamW 更新，需≥160GiB可用主存；`--train-step --device cuda:0` 执行一次 SGD 容量/梯度 smoke，需显式 `CUDA_VISIBLE_DEVICES` 选空闲 GPU且当前 ≥34GiB空闲。两者均不等价于多卡 ZeRO 容量/长期训练验收。不要据此声称已训练好或有闭环效果。

另加 `--roundtrip --roundtrip-dir <新输出目录>` 可保存完整BF16模型/配置/tokenizer/stats，用官方部署loader严格重载并从当前观测执行两步无oracle生成。该目录不可覆盖已有checkpoint；过程只做离线推理，不发送机器人指令。

对已保存的EA run可单独复验部署，无需重新训练：

```bash
CUDA_VISIBLE_DEVICES='' python scripts/smoke_ea_deploy.py <EA_RUN_DIR> \
  --dataset /remote-home/jinminghao/DiT4DiT/datasets/robotwin2_ea_rsig_v3_aloha_10000_schema5_20260928/EA_RSIG_v3/Clean/adjust_bottle \
  --device cpu --report outputs/setup/ea_deploy_report.json
```

该入口复用官方严格loader、部署config合并和JointInferenceEngine；只有报告成功写出才表示部署开环检查通过。完整模型已保存于outputs/setup/full_model_roundtrip_v2，tokenizer与foundation文件hash逐项一致；首次重载缺tokenizer失败日志保留，补齐后的部署复验通过（ea_full_deploy_v2.json，exit0）。

训练入口示例（未自动启动长期训练）：

```bash
export EA_DATASET_DIRS='["/remote-home/jinminghao/DiT4DiT/datasets/robotwin2_ea_rsig_v3_aloha_10000_schema5_20260928/EA_RSIG_v3/Clean/adjust_bottle"]'
# 只选择实际可用GPU；此命令为示例。
CUDA_VISIBLE_DEVICES=5,6 NPROC_PER_NODE=2 bash scripts/train_ea_rsig_v3.sh \
  training.initialize_model_on_cpu=true training.max_steps=20 training.save_steps=10
```

默认帧锚点0/10/20/30/40，plan10/20/40；head/left-wrist/right-wrist的384x320 L canvas。夹爪沿用旧DiT4DiT qpos/逐步delta配置的>0.5二值0/1规则（EEF context仍是原始连续夹爪读数）。原数据不改写，action/state自有stats输出于本项目 `outputs/ea_stats`，训练checkpoint会携带stats。换动作模式需同时改dataset/model合同及部署decoder，chunk-relative不自动接受。

EA入口使用PyTorch原生fused AdamW（adam_fused=true/adam_foreach=false），避免ZeRO大flat shard更新的sqrt临时量。官方其他入口保持null默认选择；模型和AdamW超参不变。双卡A100-80GB、batch1/rank、BF16/ZeRO2、CPU初始化、一步验收通过（official_gpu_train_verification.json）；不推出更大batch或长期稳定性的结论。

部署原 WebSocket/server 路径可传 native 14D `state`，同时必须传 `ea_effectors`（20元素或2x10当前EEF/FK/夹爪）和 `ea_arm_present`（2元素）。EEF必须与训练源的drive/actual、base frame及工具offset一致；禁止 `ea_targets`。本扩展要求CFG=1、DiT cache关闭，编译暂不启用；当前推理重算观测pass，速度需实测。后续应对真实训练保存的EA checkpoint执行部署往返验收；不能直接把foundation当已学会任务的EA策略。

`outputs/setup` 保存安装、下载和各层验收日志。只有独立环境/完整权重/真实数据/梯度/保存重载及部署链路都通过，才可称框架搭建完成。

Aloha/RoboTwin qpos client可选择 `benchmarks/robotwin/policy_config_ea_rsig_v3.yml`。新增 opt-in `ea_fk` 用native当前drive-qpos生成20D EEF，并校验训练URDF SHA/kinematics version；小型URDF/config和训练合同快照保存在assets/ea_robotwin。默认官方client不变。FK helper在benchmark目录只依赖numpy/scipy/yaml，避免把模型环境依赖引入仿真client。该helper不会替代Piper FK或实际关节合同；来源不一致会拒绝，不能用Aloha配置冒充Piper。

传统RoboTwin评估入口支持 `POLICY_CONFIG_PATH=$PWD/benchmarks/robotwin/policy_config_ea_rsig_v3.yml`。训练reader和client使用相同的官方RoboTwin prompt prefix，原始instruction语义和源tasks文件不变。

部署EA checkpoint使用 `bash scripts/deploy_ea_rsig_v3.sh <EA_RUN_DIR> --device cuda:0`；该薄入口固定关闭DiT cache并默认禁用编译、CFG=1，保留官方部署加载/normalizer/server路径。

实际官方训练产物：`outputs/setup/official_gpu_smoke_fused/2026-10-13_04-16-38`，含完整checkpoint/config/tokenizer/stats；GPU部署报告为 `outputs/setup/ea_official_gpu_deploy.json`，两步输出40x14有限值、夹爪0/1、alpha精确保留，exit0。这是一步工程检查产物，不是训练完成的任务策略。未启动长期训练或机器人控制。
