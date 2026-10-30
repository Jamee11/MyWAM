# Checkpoint 21480：DIT cache on/off 评测记录

## Cache-off 初始尝试（仅部分 rollout）

Checkpoint SHA256：`b141060b4ab33b0a2542effad452203b2ee5a5d287e5fd38ef092abe7326de7e`。2026-10-30 07:32 UTC启动两份L20模型和10路4090仿真。服务配置来自checkpoint归档`eval.yaml`：eager、compile关闭、DIT cache关闭、10步去噪、32动作chunk、无视频。发现这与此前PatchWAM24k评测不一致后，于07:40 UTC停止；没有任何条件完成30条。

|Task|Mode|Cache-off观察成功/条数|观察比例（非正式SR）|
|---|---|---:|---:|
|place_mouse_pad|Randomized|1/2|50.0%|
|move_stapler_pad|Randomized|1/2|50.0%|
|place_phone_stand|Randomized|1/2|50.0%|
|place_fan|Randomized|0/1|0.0%|
|place_mouse_pad|Clean|2/3|66.7%|
|move_stapler_pad|Clean|4/4|100.0%|
|place_phone_stand|Clean|5/5|100.0%|
|place_fan|Clean|2/3|66.7%|
|**合计**|—|**16/22**|**72.73%（仅部分rollout）**|

`hanging_mug/Randomized`和`open_microwave/Randomized`虽已启动，但停止时尚无完整策略rollout计数。每个条件分母不同，只有8/100个条件产生部分计数，0/100完整；72.73%不能作为checkpoint成功率，也不能和cache-on完整条件做优劣结论。原始状态、逐条件日志和server输出保留在实验根`cache_off_initial/`、`jobs/.../attempt1/`及`cache_off_initial_stopped.json`。

## Cache-on 对齐评测

为和此前PatchWAM24k公平比较，重启后的正式100条件评测采用此前完全相同的服务参数：eager，compile关闭，DIT cache开启、cosine阈值0.99、最多跳过3次；prompt cache开、视频关闭，其余10步去噪、32步动作chunk、EEF、官方RoboTwin环境/专家筛seed/成功判据/物理步进预算不变。初始cache-off attempt不进入正式SR。部署验收通过后再启动评测。

Cache属于近似推理加速，关闭cache代表不使用该近似。比较cache对成功率的可靠影响，需要两种设置在同一任务、相同有效seed和相同语言指令下都有足够完整样本；目前cache-off只有上述部分数据，尚不能判定哪种成功率更高。等aligned cache-on条件完成后，先核对两边逐episode输入是否可配对；若样本不足，另做固定seed/指令的配对A/B，再报告成功率差和推理耗时，避免把不同任务构成或小分母差异归因于cache。

## 同输入fixture的推理诊断（不等同SR）

两个相同captured observation fixtures，均输出32×20模型动作。cache开关输出并非逐元素相同：fixture最大绝对差分别0.01367、0.01172；全动作均值绝对差0.00159、0.00097（动作张量空间）。模型内热态`generate`耗时cache-off中位数1.632秒，cache-on 0.707秒，约2.31×加速。该计时不含仿真、网络、图像捕获和首轮CUDA预热；数值差异也未通过物理成功率界限验证。证据：实验根`cache_mode_fixture_comparison.json`。结论仅为cache确实改变了近似动作并降低这两个fixture的模型计算耗时，不能据此判断成功率孰优。
