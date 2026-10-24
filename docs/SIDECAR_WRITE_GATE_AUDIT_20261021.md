# Sidecar Action Write Gate 初始化核查

时间：2026-10-21T08:59:44.125914+00:00。用户要求检查三处Action Conditioning是否双零初始化，以及Stage2写入幅度。仅分析和CPU验证，未修改训练/推理源码、配置或交付压缩包。

## 实际实现

`openwam/model/modules/relational_sidecar.py:40-53`：三处复用ActionWriteAdapter，最终projection是`nn.Linear(dim, action_dim, bias=False)`正常默认初始化；beta由write_gate_init构造，当前配置为0。没有对这张projection做零初始化，检查architecture/ckpt loader没有发现后续统一置零；Foundation-only加载保留新Sidecar参数，Sidecar checkpoint加载则恢复已训练值。read gate alpha=.01与此write gate beta不同。

因此当前采用用户列出的第一种：beta=0，W正常初始化，初始前向残差为0。首次启用写回时beta通常能获得任务梯度；W和写适配器更上游路径此步梯度为0，在beta更新后恢复任务梯度。

隔离warmup前2000真实更新完全跳过write_action；该阶段write参数没有任务梯度是预期行为，与双零死锁不同。Sidecar read/refine/head仍用辅助监督训练。Stage2从完整写回启用后计其第一次/第二次更新。

## 实测

1. 当前ActionWriteAdapter，真实1024 action hidden、512 Sidecar hidden、8heads、三处tap7/15/23，使用合成CPU FP32输入、AdamW lr5e-5。三处第一步beta梯度非零、W梯度为0；第二步W梯度非零。
2. 对照只在临时内存中改成beta=1/W=0：第一步W有梯度，beta梯度为0；第二步beta也有梯度。二者均能启动且初始残差都为0，区别是首步先学gate还是投影。不能据此断言后一种更稳定或SR更高。
3. 临时双零负对照：连续两步beta/W梯度均0，复现用户指出的死锁。未把此设置写入项目配置。
4. tiny真实Wan/Action模型相关6项回归通过，包含隔离前向/梯度、启用边界和完整loss两步优化恢复。
5. 本次交付包内write模块sha与当前源码一致，以上初始化判断也适用于已打包版本。

最终42960 checkpoint实际tensor：

|tap|beta|projection L2 norm|非零projection元素|
|---|---:|---:|---:|
|7|-0.0002031326294|18.2641735|524288/524288|
|15|0.0003795623779|18.2731228|524288/524288|
|23|-0.0004787445068|18.2674904|524288/524288|

这证明旧run的write gate确实离开0，不是一直双零无法启动；不证明写回贡献有效。beta为无约束可学习标量，负号本身不是异常。

## 写入幅度与尚未解决事项

当前日志记录ea_write_gate_0/1/2及EA各loss，未记录实际`r_l=||H_after-H_before||/(||H_before||+eps)`。不能从beta较小就判断写入影响较小，也不能从projection梯度存在判断SR受益。建议后续只读监测三处write前后实际hidden差值，使用FP32算范数，按样本给出均值/高分位/最大值，重点查看Stage2启用附近；实际运行时及跨step分布尚未测量。r接近1可作为排查线索，不预设某一比例最优，也不自动改gate或做clamp。r衡量前向残差，不能取代EA辅助loss向主干回传的梯度影响分析。

证据：`docs/evidence/sidecar_gate_audit_20261021/gate_audit.json`。合成探针不是全模型真实activation幅度、BF16四卡ZeRO2验证或闭环效果实验；本次不因初始化无死锁就宣称已找到效果不足的原因。若后续试beta=1/W=0，应作为独立初始化对照；不能只把配置write_gate_init改成1而保留随机W，那会从首步引入随机非零写回。
