# 32k→36k→42960→heads-only42960：已完成30条口径汇总

统一范围：50个任务×Clean/Randomized两条件×30条=3000条；四组均100/100条件完成。下面32k为full-backward warmup2k版本。原32k正在补70条，其未完成数据不混入此表。官方公开每条件100条，作为参考单列。

|方案|Checkpoint|Clean|Randomized|整体|成功/3000|相对32k整体变化|
|---|---|---:|---:|---:|---:|---:|
|Full-backward|32k|89.00%|53.20%|71.10%|2133/3000|+0.00pp|
|Full-backward|36k|89.33%|51.20%|70.27%|2108/3000|-0.83pp|
|Full-backward|42960|89.93%|50.07%|70.00%|2100/3000|-1.10pp|
|Heads-only（from40k）|42960|90.07%|50.67%|70.37%|2111/3000|-0.73pp|
|官方发布|官方checkpoint|89.36%|48.72%|69.04%|每条件100条|—|

32k当前整体与Randomized数值最高。Full-backward32k→42960：Clean+0.93pp、Randomized−3.13pp、整体−1.10pp。Heads-only42960对原42960：Clean+0.13pp、Randomized+0.60pp、整体+0.37pp（多11/3000成功），仍低于32k0.73pp；不据小差值宣称显著收益或训练机制因果。

硬件：前三组全部4090仿真，heads-only保留10个L20完整条件、其余90为4090，属于混合硬件，不能描述为四组完全同硬件。四组均沿用官方expert筛seed、unseen指令、RT、原动作预算/成功判据、不存视频、完整Sidecar/eager+cache。30条趋势配置的候选指令上限为30，官方100条为100，因此对官方比较还存在候选指令池与样本数差异；已排队32k30+70补评同样不宣称与一次性native100逐输入等价。

原始结果与权重：

- Full-backward 32k：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_32000.safetensors`；结果`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward32000_4090_all100_30ep_20261025T073709Z`。

- Full-backward 36k：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_36000.safetensors`；结果`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward36000_4090_all100_30ep_20261026T170539Z`。

- Full-backward 42960：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_42960.safetensors`；结果`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/full_backward42960_4090_all100_30ep_20261026T010703Z`。

- Heads-only（from40k） 42960：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_heads_only_from40000_to10epochs_20261025T171338Z/runs/from40000/checkpoint_step_42960.safetensors`；结果`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/headsonly42960_4090_resume_all100_30ep_20261027T100232Z`。
