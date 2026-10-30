# Sidecar跨外观Action alignment：短训方案（待实施）

时间：2026-10-30T13:20:01.949674+00:00。用户询问下一步alignment测试；本文件仅记录可审阅的方案，不代表core修改或训练已获逐项确认/启动。已读docs/思考与隐患.md，查阅report(5)、本轮appearance/gate结果及现有C2R训练配置。

## 目标与起点

只回答增加跨外观Action特征约束是否优于同增强训练。Gate推理倍率保持1；不同时改变Read/Write结构、EA、detach/梯度路由。

建议使用已完成冻结验收的官方C2R→Sidecar warmup2000权重，非旧warmup2K→full10epoch的21480 final：
`/mnt/100T/lixiangyu/cuigm/bitdistill/OpenWAM/outputs/sidecar/c2r_two_stage8_20261030T022635Z/stage1/runs/2026-10-30_11-52-00/checkpoint_step_2000.safetensors`。

该路径来源于已有验收文档，本轮未重新SSH检查远端存在性/哈希。正式实施前重新验收。21480只用于已完成诊断，不能把其结果当新起点自身的鲁棒性结果。两个分支均weights-only加载同一权重，fresh optimizer/scheduler/counter；不再重复2K冻结。

## A/B设计

| 项目 | A：增强对照 | B：增强+alignment |
|---|---|---|
| 起点、数据顺序、seed、batch、LR、EA | 相同 | 相同 |
| 主训练增强 | 温和语义保持外观增强 | 完全相同 |
| Gate倍率 | 1 | 1 |
| 附加损失 | 无 | Action hidden跨外观一致性 |
| 筛查预算 | 2000 optimizer updates | 2000 optimizer updates |

两组都要新增短训，不能直接把现有完整PatchWAM的5epoch分支当作同增强对照。选择温和增强是为处理强过曝/模糊/Hue/AdaIN可能失去目标信息或改变颜色语义的问题；本次对A/B一致改变增强，因果比较只用A vs B。现有完整PatchWAM训练保留，作为外部参考。

建议第一版brightness/contrast/gamma的温和区间，保持裁剪、布局、相机身份、instruction/proprio/actions/EA labels不变；同一clip的所有帧共享随机参数，缺失相机保持黑槽。保留增强前原图；不把已增强图另增强一次当clean。

## 最小alignment实现

第一版建议EMA clean teacher与augmented student在同一Action block20（0-based，第二次Write之后）读出。L = L_video + L_action + L_EA + lambda * mean(1-cos(h_student_aug, stopgrad(h_teacher_clean)))。lambda初值0.05，需检查梯度量级；不是已证明最优值。

同层读出避免先引入8→20跨层projector；同action/video噪声与timestep避免把噪声差当外观差。Teacher无反传、eval模式；对WAM/Sidecar训练参数使用FP32 EMA（初始复制起点、decay拟用0.9999），共享冻结T5/VAE等资产，不复制八份不需要的冻结大模块。EMA严格每optimizer update更新一次，非每microstep；首次验收防止BF16直接EMA的小增量被舍入。两个teacher/student分支都含相同Sidecar/Write设定。

本方案是最小同层XA试验，不是report(5)中student8→teacher20+512projector+noisegap的完整复现；阴性结果不能否定完整方案。正式报告必须区分。

对齐触点只有Action hidden，梯度不局限在Sidecar：经Action expert及其读取的Video K/V回到主干，Sidecar也可接收相关梯度。保留Video主损失，不强制所有Video特征外观不变。推理只用student，不改变现有加载/服务契约。

## 预算、学习率与资源

复用低LR：Video/Action/base 2.5e-5，Sidecar1.25e-5；目标global batch256。优先沿用8×32/acc1，但EMA/额外前向真实显存尚未验收。先10–20步真实forward/backward/optimizer/EMA测试，记录峰值、梯度路径及步时；若不够，可将两组同时改8×16/acc2保持global256，不能只改变B。累计方式与吞吐变化另行记录。

保留5epoch/10740 optimizer update的cosine规划及537 LR warmup，以独立stop_after_optimizer_updates=2000筛查；不能直接设置max_steps=2000，因为当前openwam_trainer.py会用它缩短LR horizon。需增加可选的独立停止条件，默认不影响旧训练。500/1000/2000保存诊断权重；筛查若要继续，需保存至少2000处的optimizer/scheduler/RNG/EMA状态及可靠恢复契约，不能仅weights-only假装精确续训。若不保留精确状态，正式长训必须重新从统一起点按相同计划运行。

远端原实验正常退出且八卡准入通过后串行执行A/B，不中断原作业；本轮未检查当前远端进度/空闲，不声称已排队。代码、缓存、EMA/checkpoints、日志与环境均限制在`/mnt/100T/lixiangyu/cuigm/bitdistill/OpenWAM`内；本地保留配置和结果文档。

## 必须通过的检查

- alignment关闭时与原路径数值/梯度一致；A/B采样顺序和有效batch一致，额外teacher前向不扰动主路径RNG。
- clean/aug配对同状态/指令/相机/噪声/timestep；原图有效，缺失相机不被增强填充；EA标签不改。
- teacher/projector（本版无projector）不接收反传，student Action及Video K/V和预期Sidecar参数具有alignment梯度；共享冻结模块不被更新。
- 模型原有梯度检查点重算时不能重复追加/覆盖hidden捕获状态；按返回张量传递读出，不能照搬诊断monkey patch到生产训练。
- EMA FP32数值、rank同步、microstep更新频率、保存恢复与显存实测通过；不通过不自动换成低精度EMA冒充同方案。

## 评测与决策

先在同起点跑已完成的12窗口固定诊断作为连通性检查，再尽量扩大独立heldout episodes；原图/温和/强光/sensor/PatchWAM压力分开汇总，按任务/域看均值、中位数、最差窗口及GT误差，不能只看总平均或cosine loss。

筛查目标：B相对A外观漂移下降且原图准确性不明显退化；暂拟漂移下降约10%、原图GT误差增加不超过5%作为工程筛查线，不是统计显著性或闭环成功判据。高难窗口单独复查，不能只依赖103.76mm离群窗口的改善。保留Ground/Relation/Plan指标和动作旋转/夹爪。

离线通过后，使用固定相同task/domain/seed做配对闭环评测，再决定是否扩展完整5epoch；不凭loss下降、gate或离线漂移宣称策略改进。候选小规模闭环为3任务×2域×30episode/组，具体资源/资产存在性需实施时验收。

## 尚未执行

没有改model/trainer/data pipeline、没有生成可启动XA配置、没有启用EMA或启动新训练。实施涉及core模型读出、配对数据和训练损失/停止逻辑，遵守AGENTS.md第2条：在明确确认本方案后实施；新增重大变更必须写MODIFICATION_LOG.md并更新风险/实验记录。


最终用户指令及实际实现已覆盖本早期提案：使用原始传输warmup2K，8→20+EMA，联合10epoch，取消10步流程，串行接C2R。以docs/SIDECAR_XA_ORIGINAL_WARMUP2K_10EPOCH_20261030.md为准。
