# XA同权重单卡 vs split动作对照

记录：2026-11-02T03:45:06.886839+00:00。

|检查|输入数|动作数|逐元素一致|最大绝对差|
|---|---:|---:|---|---:|
|固定fixture|2|2×32步|是|0|
|实时同输入chunk|7|224步预测动作|是|0.0|

实时仿真：place_burger_fries/demo_clean，seed100000，原生专家有效seed、固定unseen语言；执行单卡动作，在每次replan同步将完全相同RGB/proprio/语言发往split副本，比较32步完整chunk。任务成功。224是预测动作比较总数，包含最后chunk中任务成功后未实际执行的尾部动作，不代表实际控制步数。

结论仅覆盖上述固定输入和单条轨迹，未发现split导致动作数值变化。不能证明所有场景完全等价，也不能以单episode推总体SR；XA相对原21480的Randomized降低不能直接归因split。

证据：outputs/robotwin_eval/xa_single_split_parity_20261102/single_vs_split_actions.json、live_parity_result.json、paired_rollout_retry/paired_action_chunks.jsonl。
