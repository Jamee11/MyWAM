# 当前full-backward20k：L20与4090标准仿真对照

建立：2026-10-24T08:36:29.830617+00:00。用户授权4路L20仿真，各30条，共120条，与当前20k已完成4090结果比较。原4090主评估继续，结果/端口独立，不混入其3000条正式趋势结果。

权重：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/aloha50_full_backward_warmup2k_10epochs_20261022T134733Z/2026-10-22_13-54-18/checkpoint_step_20000.safetensors`；SHA256 `0295ed0bebb304929bf0e6baaa94421242440979b1796ccf502c687b01a4127c`。共享盘精确权重已验证RAM缓存，不下载新文件。完整Sidecar三tap、cache .99/max3、compile=false、10步/32chunk，官方同commit和同隔离Conda、标准native seed0/unseen/expert筛seed、RT32/OIDN/near.1、原动作预算与成功判据，视频关闭。

|Task|Mode|L20仿真GPU|独占端口|4090成功/30|4090率|4090整任务秒/条|
|---|---|---:|---:|---:|---:|---:|
|adjust_bottle|demo_randomized|2|24970|16/30|53.33%|162.33|
|beat_block_hammer|demo_randomized|4|24971|8/30|26.67%|154.46|
|click_alarmclock|demo_clean|6|24972|30/30|100.00%|34.54|
|place_shoe|demo_clean|5|24973|28/30|93.33%|91.18|

模型单独部署在本机L20 GPU7，一份engine串行服务4个独立PolicyServer会话，24970–24973，无SSH隧道，避免现有4090调度器随时复用同一端口。初始GPU7 free28989MiB，模型重复/跨GPU动作与同checkpoint规范参考严格逐元素比较通过后，再验四端口真实PNG/state→20D动作。若资源失败保留日志，不终止未知进程。GPU3部署期间空闲下降，place_shoe改GPU5，其他三卡2/4/6，每卡仅新增一个sim。

速度口径：四个历史4090任务原run.json wall_seconds/30（包含CUDA/场景初始化、专家筛seed、全部rollout）与L20相同整任务口径对比。L20另以process-local只读计时包装get_obs、take_action、policy RPC及每条policy rollout边界，原方法每次只调用一次、参数返回值不改、不额外渲染或抽RNG。原4090历史日志没有这些分段计时，不伪造或声称已能纯对比GPU渲染耗时。

限制：新独立模型四会话共用engine、4090历史每模型约1–2路；本机GPU与模型服务共卡负载、背景进程/CPU/网络不同，整任务速度差不能纯归因L20/4090型号。SR采用相同原生seed扫描协议，但不同硬件可能使expert接受seed不同，将报告顺序相同与交集，不自动称逐seed/指令/RNG精确配对。这是用户要求的当前部署方式系统对照；若需单变量硬件因果，需后续固定相同manifest/推理服务负载与软件/场景/指令/RNG再测。

调度：独占token+slot锁，四任务固定卡并行，各任务只有基础设施错误最多3attempt；native完整30/exit0/成功分母验证，policy失败不重试；45min无episode或动作进展才中止。每次启动要本卡至少10GiB空闲，不动未知作业。独立源码source_evidence.json记录上游commit/脚本SHA/只读timing差异/主机负载和驱动。

输出：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward20000_l20_vs4090_4x30_20261024T083037Z`。后台dispatch.py/report.py完成后自动生成final_comparison.json/CSV；[实时表](ROBOTWIN_FULLBACKWARD20K_L20_VS_4090_LIVE_20261024.md)。


## 2026-10-24T08:39:45.639214+00:00：L20四路对照已全部进入真实policy动作

GPU2 adjust_bottle/Randomized、GPU4 beat_block_hammer/Randomized、GPU6 click_alarmclock/Clean、GPU5 place_shoe/Clean全部实际推进；四路CUDA/render PCI与NVIDIA L20断言通过，独立24970–24973四会话、GPU7同20k模型通过重复/跨副本逐元素一致及四端口真实动作验收。视频关闭；每任务30，原4090队列继续。timing.jsonl的policy rollout边界/get_obs/take_action/policy_rpc均已验证记录有限非负耗时，原调用一次无额外渲染或RNG。当前infra errors=0，还没有完整30条条件，不能报告L20成功率或GPU速度因果。证据/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward20000_l20_vs4090_4x30_20261024T083037Z/first_action_acceptance.json。后台dispatch/report持续，结束自动写final_comparison.json/CSV；速度对照保留单新模型四路共享 vs历史4090每模型1–2路、CPU/网络/背景负载不同等限制。
