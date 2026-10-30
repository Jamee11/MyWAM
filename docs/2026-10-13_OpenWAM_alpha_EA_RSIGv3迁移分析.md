# OpenWAM-alpha 与 EA-RSIGv3 迁移分析

2026-10-13 UTC。目标：在独立同级 OpenWAM 目录以官方 Alpha foundation 为初始化，保留 EA-RSIGv3 的 task-grounded relational foresight 功能，并完成环境、权重、真实数据与模型验收。工程验收完成：环境、完整权重、真实数据、CPU AdamW、双卡官方BF16/ZeRO2 fused AdamW一步/完整保存，以及该官方产物的GPU严格重载/真实推理均通过。效果等价、长期训练、闭环及真机尚无证据。

## 原版逻辑与证据

论文第 5.1 节（PDF 第 17–19 页）、官方源码及实际镜像 checkpoint config 三者交叉核对：Alpha 由冻结 Wan2.2 VAE/umT5、可训练 Wan2.2-TI2V-5B 视频专家及独立动作专家组成。两专家每层各自计算 Q/K/V，再执行 joint self-attention，使用 mutual 跨模态注意力和 first-frame-causal 视频注意力。实际 checkpoint 为 30 层、视频 hidden 3072、动作 hidden 1024、24 heads；80D 动作/状态采用固定左右臂 EEF/rot6d/gripper/hand 槽位。视频与动作分别抽样噪声时间、共同计算 flow-matching loss；部署同步联合去噪。

这与 DiT4DiT 的 Cosmos 观测 H18 -> EA -> action cross-attention 不是相同 backbone 接口。不能拿含 noisy future/action 信息的 joint hidden 冒充 EA 的部署观测。论文 human+robot 预训练并不自动使现有 oracle-label 机器人数据支持 human EA 监督。

来源：[论文](https://arxiv.org/pdf/2609.07398)、[官方源码](https://github.com/OpenWAM-Official/OpenWAM)、仓库 assets/openwam_usage_docs/openwam-alpha-finetuning.md、镜像 config.yaml。

## 本地迁移合同

| 原功能 | OpenWAM 实现 | 必须成立的边界 |
| --- | --- | --- |
| detached observation H18 | 对 clean first latent 单独执行前 18 个 Wan blocks，no_grad/detach | 不读取 future/action/oracle；不是 joint hidden 切片 |
| Object/Goal grounding、typed EO/OG relation、progress、三时距 plan | 原 DiT4DiT EARSIGV3Module 逐字复制，仅改 loss helper import | 130D 双轨 targets 仅用于 loss |
| 可部署 EEF 状态 | 独立 ea_effectors(2,10)、ea_arm_present(2) | 当前 FK/夹爪/本体存在性；不是 object/goal GT |
| EA action tokens | 投影到原 action context 宽度，追加到动作专家的 cross-attention context | 文本/状态接口保留，EA不追加到视频文本context |
| 训练用视频 residual | 第 18 层后、仅训练grad模式、仅 future token 注入 | 当前clean prefix保持；eval不用残差；零初始化 alpha |
| video/action/EA loss | 保留官方 joint FM + .1*(.25 ground + .1 relation + plan) | no-gradient observation pass与trainable FM pass分开 |
| 40动作、5帧、plan10/20/40 | 读取现有schema5合同，拒绝误配 | 源数据不改、不伪造32步label |
| 多相机bbox对齐 | 原图 head/left-wrist/right-wrist -> 384x320 L canvas；EA从patchgrid按同layout拆出3view并插值到14x14 | bbox仍为各视图归一化监督，不能随意crop/reorder |

保留原 Alpha joint mutual 视频/动作专家；新增 EA 是 detached observation bottleneck，但原 joint 路径仍允许 action/video 梯度互通，所以不能宣称整个策略只有 EA 瓶颈或 action loss 全部与 video backbone 断梯度。EA auxiliary/action-token 梯度不穿过单独的观测 pass。

当前政策保留 native 14D qpos [L6,Lgrip,R6,Rgrip]，替换 foundation 80D 的 action encoder weight、decoder weight/bias、proprio encoder weight；其余 expert tensors 必须严格继承。EA warmstart 会报告逐项重置键并拒绝任何其他缺失、额外键或形状差异。后续 EA checkpoint 用严格重载。此配置是 40步 native-policy 迁移，明确不属于官方 32步/80D Alpha标准微调协议。源码还支持已有逐步 base-delta EE列；chunk-relative动作需另外绑定anchor decoder，现配置拒绝。

训练/部署保存自有 action/state stats，运动维度min-max；原qpos/逐步delta政策夹爪按旧DiT4DiT的>0.5阈值映射到0/1，state输入与action输出均保持此规则。EEF context的夹爪仍保留原始连续读数。EA/raw targets不参与归一化。deploy需要 current EEF/arm context，不接受 oracle targets。当前 EA variant 禁用编译与 DiT cache，CFG限定1；不能引用原Alpha170ms速度作为本实现测量。

## 下载与代码来源

- 上游代码 commit: 4d7f642b841913ea3c176010b692b1de3d5ef180。
- checkpoint repo: OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model。
- checkpoint revision: 52df4e66c82c5c8b480adcc8d01f4db7415dfb56。
- 主文件 checkpoint_step_154000.safetensors：24,813,767,464 bytes；SHA256 180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d。
- inventory/config/tokenizer清单由 hf-mirror 直连获取。用户明确批准镜像返回的 Xet CDN 直连，禁代理/VPN。下载器 trust_env=False、逐redirect白名单、固定revision、保留partial、大小/LFS SHA256与普通Git blob hash验收。
- 原EA core来自 DiT4DiT/model/modules/ea_rsig_v3.py；loss分组helper来自 training_memory.py；几何adapter复用 4DWAM-EA-RSIGv3/FastWAM 的 ea_bridge.py。保留 DiT4DiT LICENSE，未修改源项目。
- AGENTS.md逐字复制，核心修改由用户完整搭建/迁移授权覆盖。隐患来源快照保留，新增隐患按本地思考与隐患追踪。

## 当前验收与下一步

最新结果（下方逐次记录中的“仍在进行”属于当时状态）：

| 验收 | 当前证据 |
| --- | --- |
| 代码与规则 | 官方clone/revision、AGENTS逐字一致、源码/论文/权重合同核对 |
| 独立环境 | pip check和import通过，版本见outputs/setup/import_check.log及requirements-ea-lock.txt |
| 完整权重 | 8个上游文件大小/hash通过，主文件24,813,767,464 bytes，download_verification.json记录固定revision和直连路径 |
| 数据适配 | binary修正后50episode/7134windows元数据与3个解码窗口通过；EEF/targets和源数据逐元素相等，源Parquet不改 |
| 局部回归 | 独立环境49 passed/2 GPU skipped；修正后7个tiny模型项通过；client/FK和旧客户端接口12 passed |
| 原循环数值保真 | ea_baseline_parity_test.log：1 passed；同native14权重/输入的tiny Wan关闭EA后，视频/动作预测逐元素等于原joint-forward，仅证明此局部循环 |
| 真实完整模型 | foundation_smoke.json：2085继承tensors、4个重置政策头、新EA73 tensors，无meta残留 |
| 单步真实数据更新 | CPU FP32 AdamW，总loss3.9434447，video .7602232 / action2.8688180 / EA .3144036；三分支梯度均有限，alpha由0到-9.9988e-5 |
| 完整保存 | full_model_roundtrip_v2/checkpoint_step_1.safetensors已保存，24,869,359,726 bytes、2162 tensors，BF16 alpha=-0.0001001358；tokenizer四文件hash与foundation一致，配置和stats齐全 |
| 部署重载/生成 | ea_full_deploy_v2.json及exit0：BF16严格重载/官方JointInferenceEngine/两步生成40x14有限动作/夹爪0或1/alpha精确保留；当前观测无oracle |
| 官方GPU入口 | fused重测exit0：A100-80GB双卡、batch1/rank、BF16/ZeRO2、完整6048.9M可训练参数、AdamW一步；loss约1.3338/video .5764/action .4157，保存完整2162 tensors。首轮foreach OOM日志保留 |
| 官方产物GPU部署 | ea_official_gpu_deploy.json及exit0：严格重载官方训练产物、BF16、JointInferenceEngine、两步、40x14有限动作、夹爪0/1、alpha=+0.0001001358精确保留；测试后GPU释放 |

当前工程链路已完成。双卡证据只验证上述配置的一步容量与保存/GPU部署，不验证更大batch、部署速度或长期稳定性。长期训练、标准闭环成绩与真机验证属于后续实验，搭建验收不得声称策略已学成。

2026-10-13 用户追问实验效果，明确：上述工程检查不证明迁移效果等价。高影响变量包括native qpos/14D政策头重置、40步5帧、Wan观测第18层与特征网格、EA condition/residual与原Alpha联合路径并存。对照需控制底层数据、训练预算、多seed和同一闭环任务；原Alpha标准协议与native无EA适配之间测量适配整体影响，native无EA与EA之间隔离EA增益，逐项去掉头/时间/几何变化另需有效合同。没有这些结果不得保证“不会明显影响效果”。新增tiny数值parity仅核对关闭EA后新增循环等于原joint-forward，不替代性能实验。

部署adapter追加回归6 passed：实际ObsPreprocessor/WAMPolicy条件构造/JointInferenceEngine，检查head/left/right画面位置、当前EEF/arms及40动作/5帧传递，拒绝oracle与缺失/非有限/无效arm context。模型generate在此局部测试是stub；真实模型证据须看单独部署报告。FK拒绝actual-joint冒充drive-target专项及原payload/source-row共3 passed。

### 2026-10-13 逐次验收历史（以以上最新状态为准）

独立 .venv 安装完成，torch2.7.1+cu128 / torchvision0.22.1 / torchaudio2.7.1；依赖锁定记录在 requirements-ea-lock.txt。pip check 为 No broken requirements found；完整 import check 仍在进行。

独立环境回归 outputs/setup/independent_env_tests.log：49 passed / 2 skipped，覆盖EA原core、real tiny Wan loss/梯度与checkpoint、MoT/proprio与finetune/deploy旧路径。两个skip是官方选择性的GPU集成，不算通过。最初fixture eps/bridge_interval缺失已修复；保留失败日志。后续夹爪binary合同修正正在专门复验，不能只引用修正前49项作为修正后全部验收。

真实源数据预检（初稿连续夹爪范围）在50episode/7134windows元数据范围与三个解码sample通过；只查三个sample的图像，不是150视频全量解码。夹爪合同修正后会写独立 ea_data_binary_preflight.json。源数据现成22D context与130D labels保持原样，只有batch字段组织和camera canvas适配，未重建轨迹或监督数据。

主权重仍在固定mirror revision -> 已批准的CDN直连下载；完整hash和foundation真实加载尚未通过。目标保持active。

2026-10-13 后续合同验证：完整依赖 import check 通过（torch2.7.1+cu128、transformers5.19.0、diffusers0.41.0、accelerate1.15.0、deepspeed0.18.9、pyarrow25.0.1、av17.1.0）。修正夹爪后的真实source预检通过，EEF/130D targets与原当前行逐元素相等；新stats缓存88854ee497d8e3cd.npy，不覆盖旧调试stats。新FK helper对真实源3行误差≤2e-6通过；model tiny六项再次通过。最终合同测试7项模型检查与1项FK通过，client mock漏ping方法导致1项失败，已修正正在重跑。所有这些均为CPU/离线证据。

下载PID1609414与工具session92583仍为实际运行句柄；主权重未完整hash。下载结束后已排队执行offline完整foundation加载+一次CPU FP32 AdamW更新（16threads、≥160GiB内存门，zero/GPU不占用），queue工具session94210，日志outputs/setup/after_foundation_download.log和foundation_smoke.log。这项排队未等于执行或通过；继续时先查真实进程/报告，不重启仍live的下载。
