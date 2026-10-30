# Full-backward20k：L20与4090四条件对照最终分析

核对：2026-10-24T14:41:44.244791+00:00。L20于2026-10-24T10:23:00.047824+00:00结束，四条件各30条，共120条，全部原生exit0/30条/实际成功数重新验收通过，无基础设施异常或重测。相同checkpoint SHA 0295ed0bebb304929bf0e6baaa94421242440979b1796ccf502c687b01a4127c、完整Sidecar/cache开启/compile关闭、不存视频、同native官方协议。

|Task|Mode|L20成功/30|4090成功/30|L20率|4090率|L20整任务秒/条|4090整任务秒/条|L20耗时变化|accepted seed交集|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|adjust_bottle|demo_randomized|18/30|16/30|60.00%|53.33%|204.31|162.33|+25.86%|29/30|
|beat_block_hammer|demo_randomized|7/30|8/30|23.33%|26.67%|215.01|154.46|+39.21%|24/30|
|click_alarmclock|demo_clean|30/30|30/30|100.00%|100.00%|53.02|34.54|+53.49%|30/30|
|place_shoe|demo_clean|27/30|28/30|90.00%|93.33%|71.13|91.18|-21.99%|29/30|

四条件综合：L20和4090均82/120=68.33%；L20 Clean57/60=95.00，4090 Clean58/60=96.67；L20 Randomized25/60=41.67，4090 Randomized24/60=40.00。adjust_bottle Random L20多2次成功，hammer Random和shoe Clean各少1次，alarmclock Clean相同。此有限集合未见L20或4090整体SR的方向性劣化，但不是等价性证明；每项N=30、3项accepted seed顺序不同，交集29/24/30/29，场景/指令/RNG亦未严格配对。尤其hammer两卡都低（23.33/26.67），不支持把该项大幅低于官方55统一归咎4090。

速度统一整任务wall_seconds/30：四条件等权L20 135.87s/条、4090 110.63s/条，L20多耗时22.82%。三个任务4090较快，place_shoe则L20较快21.99%；不能声称4090对任何任务固定提速。整任务包含启动、专家筛seed、场景加载、推理等待与全部成功/失败rollout；不可与L20另存纯policy rollout均值直接混算。历史4090日志无同等分段计时，无法回溯真实render-only耗时。新模型四路共用engine vs历史1–2路/模型、主机CPU/负载、网络/计时仪表开销差异均存在，速度是当前系统实测，不是纯GPU型号因果。

L20逐action计时均值（get_obs / policy RPC / take_action秒）：

- adjust_bottle/demo_randomized: 0.1424 / 0.0628 / 0.4939。
- beat_block_hammer/demo_randomized: 0.1126 / 0.0562 / 0.3859。
- click_alarmclock/demo_clean: 0.1331 / 0.0715 / 0.3760。
- place_shoe/demo_clean: 0.0709 / 0.0596 / 0.1897。

下一步：继续现有20k全100条件队列；本四条件对照不并入其3000条。若需纯硬件SR/速度定因，固定同manifest/指令/RNG、推理服务负载和计时仪表，在两host相同软件/资源配置下再配对。不自行扩任务或改变现有评估协议。

证据：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward20000_l20_vs4090_4x30_20261024T083037Z/final_comparison.json`、CSV、各jobs/run.json/timing.jsonl及first_action_acceptance.json。
