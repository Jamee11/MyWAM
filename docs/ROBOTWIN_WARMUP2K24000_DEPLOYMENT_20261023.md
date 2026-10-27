> 最终验收：2026-10-23T07:55:55.087507+00:00，本机8×L20的GPU0–7全部ready，8模型/24端口。最终配置是完整Sidecar、cache开启、FFN编译关闭；初期FFN尝试记录保留在下文。4090端24路真实图像/state→20D动作调用全部通过，尚未启动正式仿真评测。

# Sidecar warmup2k 24000：本机L20并行模型部署

更新时间：2026-10-23T07:32:30.097937+00:00。当前加载/数值验收中，不把进程启动当部署通过。

用户指定权重：`/remote-home/jinminghao/OpenWAM/outputs/1022-sidecar-warmup2k/checkpoint_step_24000.safetensors`。
SHA256：`c8835677cb59324ce3f2bf8c01e80178e36397ac3379d7bc1ca3d0d6ea23de76`。固定部署目录：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z`。
权重hardlink到独立目录，只读使用、不chmod原inode；config/stats/contract/tokenizer逐项复制，
没有下载或修改训练产物。实际config为isolation_warmup_updates=2000，
不与另一个full-backward/WAM延迟更新实验混为一谈。

复用remote_eval.py model与serve_remote_sidecar.py，三tap读写、FFN compile、cache .99/max3、
10步去噪、32动作chunk、自有stats、384×320拼图与wxyz EEF合同。
每GPU一份模型、3个独立同步policy端口。每份必须通过两真实观测fixture的
eager/FFN同缓存数值门槛、非零生成kernel，以及三个端口各一次真实PNG/state→20D有限动作
WebSocket请求。这里只部署服务，不启动正式仿真评测、不产生SR。

|GPU|端口|session目录|
|---|---|---|
|0|22900, 22901, 22902|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu0`|
|1|22910, 22911, 22912|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu1`|
|2|22920, 22921, 22922|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu2`|
|3|22930, 22931, 22932|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu3`|
|4|22940, 22941, 22942|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu4`|
|5|22950, 22951, 22952|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu5`|
|6|22960, 22961, 22962|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu6`|

GPU6按用户明确要求尝试：初始28270MiB空闲、util0，启动门槛本次显式降为27000MiB；
该门槛不是峰值已通过证据，必须真实加载/推理验收；失败只退出新模型。
GPU7初始29109MiB空闲但高利用率，先保留原作业，根据实测容量再决定。
GPU0/1/2/3/4/5初始33000MiB门槛。仅回收两个已确认无连接的旧32k遗留服务：
GPU1近裁剪面诊断state=failed、GPU5正式32k全量state=complete。PID/start_ticks/命令与
原评测状态、无连接检查记录在deployment.json；原日志/结果保留、其他进程不操作。

进度与验收：`deployment.json`、`gpuN/server.log`、`gpuN/fast_validation/result.json`、
`gpuN/deployment_acceptance.json`，真实动作请求之后重置各slot；不外推缓存无损/SR或并发吞吐。
`outputs/robotwin_eval/warmup2k24000_deploy_current.json`指向本次输出根。

训练腕部near .02/官方仿真near .1、跨硬件RT/RNG、cache闭环因果等未解决项继续保留；
模型服务本身不变更仿真相机或成功判据。下一步所有可用实例ready后再配置仿真/隧道及独占端口，
按用户后续评估范围执行，不静默接入旧42960选集（其SHA门槛是另一权重）。


## 2026-10-23T07:44:26.112558+00:00：24k FFN数值门槛失败，独立部署改为完整Sidecar eager+cache

GPU0–6载入成功且完成真实fixture推理，FFN4kernel已生成但同缓存参考的严格allclose门槛未通过，最大归一化动作差0.0078125；各卡一致，不是GPU6容量失败。未放宽阈值或声称SR受损。两fixture热请求均值：GPU0 eager_cached0.3634s/ffn_cached0.3758s，GPU5 .3865/.4394，GPU6 .3834/.3850；有限样本，不外推吞吐。

按用户持续部署授权及原先不影响Sidecar能力原则，仅本次输出目录新增serve_eager_cached.py/launch_eager_cached.py，三tap保持、compile=false、cache .99/max3与10步保持，自有stats/权重不变；不改生产核心代码/原入口。GPU0–7八份重新启动，两个fixture各重复两次必须与原同checkpoint eager_cached参考逐元素相等后才开放服务，再验各slot真实20D动作。原FFN失败报告与进程记录完整保留compiled_attempts。为减少GPFS mmap启动等待，将精确权重复制到/dev/shm/openwam24000_c8835677cb59324c，流式SHA/size与共享权重一致（local_weight_cache.json），不是下载、无数学变化；重启主机需重建本机RAM缓存。8卡容量与线上返回仍在验收。文档ROBOTWIN_WARMUP2K24000_DEPLOYMENT_20261023.md/本次deployment.json为权威，不把初期FFN计划当最终部署参数。


## 最终可用端口与证据

输出根：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z`。

|GPU|4090本地转发端口|模型PID|session目录|
|---|---|---:|---|
|0|22900, 22901, 22902|2964076|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu0_eager_cache_v2`|
|1|22910, 22911, 22912|2964078|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu1_eager_cache_v2`|
|2|22920, 22921, 22922|2964075|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu2_eager_cache_v2`|
|3|22930, 22931, 22932|2964077|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu3_eager_cache_v2`|
|4|22940, 22941, 22942|2964079|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu4_eager_cache_v2`|
|5|22950, 22951, 22952|2964082|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu5_eager_cache_v2`|
|6|22960, 22961, 22962|2964081|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu6_eager_cache_v2`|
|7|22970, 22971, 22972|2964080|`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar_warmup2k24000_l20_deploy_20261023T072845Z/gpu7_eager_cache_v2`|

32个真实fixture比较（8模型×2输入×2重复）均与之前保存的同checkpoint eager+cache输出逐元素相等，max_abs_error=0。
本机24端口与4090端24端口均真实reset/PNG+state/predict_once/reset通过，返回20个有限物理动作值。
4090端转发地址均为127.0.0.1；隧道保持运行，PID和全部映射见tunnel_process.json。
完整清单ready_endpoints.json，最终验收final_deployment_acceptance.json，跨机动作remote4090_acceptance.json。
每模型session_slot0/1/2.json分别对应表中三个端口，正式sim必须独占端口。

本次eager初试的模型推理本身已逐元素一致，但保存被preprocessor修改的fixture字典时遭遇ndarray JSON序列化错误导致服务退出。
已保留原脚本.v1_serialization_failed、各日志和eager_serialization_failed_attempts；修复为独立输入副本与原始JSON文本保存，
重启到gpuN_eager_cache_v2并完成上述验收，生产核心/权重/训练未改动。

以下为最终整卡快照，包含原有其他作业，不能把整卡用量全部算作本模型：

```text
index, name, memory.used [MiB], memory.free [MiB], utilization.gpu [%]
0, NVIDIA L20, 36264 MiB, 9326 MiB, 0 %
1, NVIDIA L20, 30909 MiB, 14681 MiB, 0 %
2, NVIDIA L20, 27152 MiB, 18438 MiB, 100 %
3, NVIDIA L20, 24520 MiB, 21070 MiB, 0 %
4, NVIDIA L20, 27152 MiB, 18438 MiB, 100 %
5, NVIDIA L20, 24509 MiB, 21081 MiB, 0 %
6, NVIDIA L20, 41825 MiB, 3765 MiB, 0 %
7, NVIDIA L20, 40986 MiB, 4604 MiB, 100 %
```
