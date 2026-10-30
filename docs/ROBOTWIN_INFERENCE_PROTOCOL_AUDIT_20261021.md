# 推理加速与评测协议核查（2026-10-21）

用户要求确认 baseline 为什么不能完整 compile，并排查 Sidecar 低分是否由评测差异造成。

## 结论与边界

baseline 能运行原版完整 compile。之前自动退回 eager 是额外的 eager 一致性门槛（atol=.005, rtol=.01）未通过，不是编译失败，也不是权重结构不支持。这个门槛是本地为保守数值检查添加，不是官方评测要求；不能据它未通过就说 baseline 不能使用官方加速。正式 baseline 已改为官方完整 MoT-loop compile + velocity cache .99/max3；原 eager pilot 单独保留，不混入正式结果。

目前没有证据把 Sidecar 低成功率单独归因于 compile。两真实输入、五个原32k副本共10次匹配cache的FFN编译/eager比较均逐元素相同；但 cache on/off 确实改变动作，尚无同manifest闭环A/B证明其成功率影响可忽略。

## baseline 数值差异

模型使用本机Torch2.7.1/BF16。原版完整compile实际生成693 kernels，缓存一致时预热约.26–.27s/chunk；eager+cache约.37–.38s/chunk。完整compile可运行并加速，不保证数学等价的浮点实现逐bit一致。

PyTorch2.7.1官方源码（以及本地 .venv 中同段注释）明确说明：低精度 eager 算子之间会发生FP32计算后的BF16截断；Inductor融合可以消去相邻的downcast/upcast，中间舍入因此不同。该机制与当前误差量级/取值模式相符；还未通过全模型逐算子/精度开关消融唯一定位哪一层贡献，不能声称已证明唯一根因。

来源：https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/config.py ，emulate_precision_casts 注释。

两个fixture每个32×20动作，分别仅3个标量超出上述本地阈值，全部在接近0的Rot6D分量；最大元素差均.0078125。接近0时相对容差很小，因而会失败。不能将这些数值直接解释成“7.8mm位置误差”，也不能把接近0的.0078差称为该数值处仅1 ULP。

| Fixture | 最大目标位置差 | 平均目标位置差 | 最大旋转测地差 | 平均旋转差 | 最大夹爪读数差 |
|---|---:|---:|---:|---:|---:|
| 0 | 3.699mm | 1.160mm | .417° | .172° | .00390625 |
| 1 | 3.404mm | 1.118mm | .530° | .153° | .00390625 |

位置是动作xyz欧氏距离，旋转先把column-Rot6D按正式解码同样的Gram-Schmidt正交化再算相对旋转测地角；这是预测控制目标差异，不是仿真实际跟踪误差。毫米级差异在精细接触任务可能重要，尚不能保证对SR零影响，也不能据此断言必然使SR下降。

## Sidecar既有验收重新分析

使用32k GPU0 fast_validation/result.json的已保存动作，不新占GPU或打断正式32k。

| 比较 | Fixture | 最大位置差 | 最大旋转差 | 最大动作分量差 |
|---|---:|---:|---:|---:|
| FFN compile+cache vs eager+cache | 0/1 | 0 | 0 | 0 |
| FFN compile无cache vs eager无cache | 0/1 | 0 | 0 | 0 |
| eager cache-on vs cache-off | 0 | 2.980mm | .450° | .0078125 |
| eager cache-on vs cache-off | 1 | 2.707mm | .401° | .0078125 |

另核对五个部署副本GPU0/1/4/5/7，同两个输入上FFN+cache vs eager+cache误差均0。只是2个独立输入、多副本重复，不是10个独立场景或全状态空间等价证明。

这两输入的cache统计均10步中跳过6次forward，仅重算4次。跳过步复用包含Sidecar影响的上一轮video/action速度，未把Sidecar整体关闭；但这6步不会重新计算三处关系读写。cache代码同时比较video和action速度余弦（不是只看video），仍只约束方向相似性，不保证逐维幅度或接触边界下的闭环成功率。

原Sidecar generate最初明确要求cache关闭；为用户授权的快速评估，独立runtime绕过该初始保守限制，沿用官方Base.generate缓存路径。先前数值验收证明的是“FFN编译保持同cache的eager”，没有证明“开启cache不损害Sidecar的SR”。这是需要保留的实验不确定性，不应把加速验收称为质量无损证明。

cache算法与官方文件逐字一致，但新增模型结构对同一种近似的敏感性可以不同；“官方也开cache”不能排除它影响Sidecar。是否改变SR、幅度和方向，需要固定场景/指令/seed的cache on/off闭环对照。

## 与官方实现逐项核对

以下与下载的官方OpenWAM checkout逐字相同：joint_self_attn.py、compile_options.py、dit_cache.py、model_loader.py、默认configs/deploy.yaml。BaseWAMArchitecture.generate、normalize_deploy_proprio、_resolve_inactive_action_dims的AST也完全相同；本地base文件只在训练loss增加可选forward_fn，不改变该生成循环。

engine相对官方增加EA/Sidecar特权输入拒绝及旧EA分支的上下文转发；当前baseline不进入这些分支。policy只多转发可选EA字段，当前Sidecar/baseline接口不提供旧EA字段。没有在这些diff中发现改变基础denoise schedule、Euler推进、动作32chunk或部署normalizer的变更。

保留的有意差异及限制：

- 训练reader明确按raw wxyz解释实际endpose，评估adapter相应wxyz↔Rot6D；不能为表面复制默认adapter而改成不同的训练输入语义。训练/评估EEF实际状态对齐的代码证据见既有EEF文档，物理执行误差仍未回放量化。
- Sidecar保留三处读写，只compile其FFN；baseline使用原完整MoT compile。这是结构所需的编译范围差异，在现有输入上FFN本身没有引入动作偏差。
- 共享服务有独立端口policy/buffer、predict互斥、每次generate清空DiT velocity cache；没有发现跨仿真直接沿用前一generation速度缓存的路径。此前没有完成严格单sim/多sim闭环吞吐及SR A/B，不能外推逐bit或SR完全一致。
- 官方环境/RT/预算/成功判据等保持当前原生协议，但accepted seed、指令与官方发布运行并非严格逐例配对，训练数据和预算亦未完全受控。自己的baseline配置5epochs、自有stats与Sidecar32k不同。不能只凭两checkpoint差距纯归因模块，或据两者都低认定官方数字虚报。
- 32-action开环长度沿用官方。Sidecar是否更依赖高频反馈是另一项可验证假设；不在当前标准批次中途改成replan8。

## 已做的协议修正与下一步

旧eager pilot输出完整保留在baseline10740_after_sidecar32k_20261020，终止时hammer Randomized有1个成功episode，其余两项未有episode终态；没有任何完整百条条件。由于协议修正而停止，不根据成功率择优，全部不混入新的正式批次。

新正式根：outputs/robotwin_eval/baseline10740_officialcompile_early_20261021；控制器2457886。相同固定十任务和10740权重/自有stats，继续提前与32k共存；正式始终启用原版完整compile+cache。真实推理异常/非有限值仍拒绝启动；相对eager的误差继续保存作诊断，不把额外阈值假装成官方门槛，也不静默退回eager。新启动同时记录eager/compiled×cache-off/on四组真实输入结果。当前新模型重新加载中，尚未宣称新的正式rollout开始。32k未改设置、未重跑，继续同一批次。

下一项能回答质量影响的问题应是小范围固定manifest的闭环A/B：对同一Sidecar32k权重比较cache off/on（FFN范围固定），优先hammer Randomized、skillet Randomized加一个正常任务对照；需要逐seed/指令/scene匹配，记录成功率、实际动作与完整wall time。只比较开环两输入不能代替该实验。现有32k批次保持原设置不混接；本轮未额外启动这组闭环消融，GPU优先既定baseline十任务和32k收尾。

证据在新正式根：official_source_comparison.json、baseline_original_compile_physical_difference.json、sidecar32k_existing_cache_compile_audit.json、sidecar_all_replicas_compile_check.json、retained_pilot_progress.json；原始完整动作保存在旧model0/validation.json与32k各gpu fast_validation/result.json。


## 2026-10-21 06:50 UTC 正式 baseline 阶段结果

10740 使用原版完整 MoT compile + cache，无 eager pilot 混入。实时日志：。各条件目标100，尚无完整条件，阶段百分比不代表最终结果。已部署3个模型，配置9个worker，但当前仅3个仿真实际运行，其余6个等待容量；不能将worker配置数当成活跃仿真数。Sidecar32k监控显示98/100完整条件、9964/10000 episodes，2项运行。继续既定调度与协议，待满100再正式比较。证据：outputs/robotwin_eval/baseline10740_officialcompile_early_20261021/progress_snapshot_20261021T0650.json。

阶段日志明细（去除ANSI颜色后提取）：beat_block_hammer demo_randomized: 33/66=50.0%, official=55%; place_bread_skillet demo_randomized: 27/59=45.8%, official=61%; open_microwave demo_clean: 27/35=77.1%, official=82%。尚未满100，不作最终归因。
