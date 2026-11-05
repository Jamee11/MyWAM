# 原最佳21480：Sidecar全关闭补充组

2026-11-02 UTC，用户明确要求在18组外加一组完全关闭Sidecar的闭环实验。独立目录`outputs/robotwin_eval/sidecar21480_all_off_20261102`，不重启/修改现有18组服务与仿真。

## 定义与配对

完整载入原最佳`outputs/1030-warmup2k-to-full10epoch/checkpoint_step_21480.safetensors`，SHA256 `b141060b4ab33b0a2542effad452203b2ee5a5d287e5fd38ef092abe7326de7e`后，仅将架构已有开关`sidecar_enabled=False`。复用现有forward的父类路径，完全不执行Sidecar init_tokens、Read、Refine、FFN、EA heads、Write。没有删除参数/重训/改变主干权重，不修改模型核心源码。

与All-Tokens-Reset不同：后者仍执行Sidecar与Write，并保留其偏置等计算。与Write-Off相比，本组还省去上游Sidecar计算；若Write是唯一动作作用路径，则同输入可能得到完全相同动作。必须实测，不把“不同组名”当成独立机制。Full、Write-Off、All-Off固定输入动作对照会保存在`replica0/eager_validation/actions.json`。

训练后的主干已经改变，因此All-Off不等于原版OpenWAM checkpoint，不能据此归因所有训练策略或监督损失贡献。

同18组的10任务、Clean/Randomized、每条件前5条sealed manifest，共100条。固定seed/指令/episode顺序；同推理noise42、cache开启、compile关闭、10denoise、chunk32、不存视频、同官方RT/成功/动作预算与标准EEF。复用既有专家验过的manifest，不另筛场景。报告对每个完成条件和历史Full前5条严格配对。

## 部署与验收

L20新独立模型：主体GPU2、T5 GPU4，原已验证split adapter、BF16无量化。40200–40203四会话，独立反向SSH隧道到4090；按剩余容量启动至多4路新增仿真，与原队列并行，容量失败保留日志再试。

启动门槛：Full两fixture与原canonical逐元素一致；关闭Sidecar后两次生成逐元素一致且Sidecar全部子模块及init_tokens调用为0；切回Full仍逐元素一致；记录与Write-Off差异；参数版本不变；四会话重复动作与engine参考一致。通过前不启动闭环。

状态及证据：`launch_status.json`、`model_process.json`、`replica0/server.log`、`deployment_acceptance.json`、`state.json`、`comparison.json`。最初仅启动加载，不能当成已有SR。完整100条之后与原Full/Write-Off/Relation-Reset前5条同条件比较；如发现All-Off/Write-Off差异，先解释路径/数值/cache原因，不直接当语义收益。

## 实际启动验收通过：2026-11-02T08:53:35.445089+00:00

两份固定输入下：Full与历史canonical逐元素一致；完全关闭Sidecar时所有子模块/init_tokens调用数为0；Off重复、切回Full恢复逐元素一致；参数版本未变。All-Off与Write-Off在这两个输入的32步动作逐元素完全相同，maxabs=0；与Full最大元素差均0.0078125。此等价证据仅覆盖两固定输入，不外推所有状态。4会话真实RPC验收通过，4090已启动队列：[('adjust_bottle', 'demo_clean', 4), ('adjust_bottle', 'demo_randomized', 5), ('beat_block_hammer', 'demo_clean', 2), ('beat_block_hammer', 'demo_randomized', 4)]。新组无完整SR，原18组继续运行。

## 协议更正：统一每条件10条

用户指出原约定10条，先前扩大消融自行缩至5条未经确认。保留当前前5阶段，已启动接续watcher，自动补同一sealed manifest第6–10条；19组最终均为每task/domain10条，共3800条。此前5条结果仅为阶段趋势，不是10条终态。原三组复用完整10条，其他16组追加互斥后5条。执行与审计见docs/ROBOTWIN_21480_19ARMS_COMPLETE10_20261102.md。
