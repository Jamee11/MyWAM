# 21480：补70条扩展至每条件100条

用户2026-10-31授权。精确权重：`outputs/1030-warmup2k-to-full10epoch/checkpoint_step_21480.safetensors`；SHA256 `b141060b4ab33b0a2542effad452203b2ee5a5d287e5fd38ef092abe7326de7e`。原完整评估3000条2245成功（74.83%）保留。

新目录：`outputs/robotwin_eval/full21480_supplement70_4090_20261031`。原100个条件各30条已重新从原生日志/runtime/run.json检查退出码、成功数、checkpoint SHA及30个有效seed。`plan.json`和`prefix30_reference.json`封存原结果路径、成功数、seed及续测起点。

逐条件从原最后有效seed+1继续原专家筛选，新增70条；不从100000重测。保留原seed组0、RT/相机、实际EEF、控制/规划、动作预算和成功判据。继承21480的eager/compile-off、DiT cache-on(.99/max3)、10步去噪、horizon32、完整Sidecar、无视频及已验证chunk内采图优化。

复用32k已验证的continuation_adapter与近期弹性队列。每次完整70条必须唯一有序、起点正确、与原30互斥，才能成为有效条件。中断partial不拼接；保留原attempt并整70重跑。只有完整30+70进入合并100条SR；终态再次独立读取两段原生结果和SHA，验收每条件100唯一seed，总10000条。

指令池边界：原30候选池30、补70候选池70；不宣称与一次性native100候选池/逐例语言完全一致。原30实际语言未完整冻结，无法追溯改成100设置。这是沿用前32k方式的每条件100条分段扩展评估。

先验收10740已全部完成及独立audit，再仅退出它自己的两份模型与393xx隧道，释放GPU2/4；其他GPU作业保留。目标两L20模型×6独立session，服务与重复动作/跨副本验收后启动4090弹性仿真。沿用实际失败容量：GPU1最多1路、GPU4/6最多2路，其他卡按实时余量；保留显存低水位、失败回退与日志。此上限不代表实际达到12路。

修复近期经验：复制完整observe_queue依赖；远端启动先检查依赖，等待scheduler实际写running状态后再启report/sampler，避免读取旧终态误退出。当前脚本语法、续seed adapter与100条件计划检查通过；真实模型与首rollout验收以launch_status/deployment_acceptance/state及每attempt continuation_seed.json为准。

进度：`progress.json`为新增70条进度，`comparison.json`为完整30+70的合并成绩；实时文档`docs/ROBOTWIN_FULL21480_COMPLETE100_LIVE_20261031.md`，终态`final_complete100_independent_audit.json`。

## 实际启动

两L20副本通过原21480相同fixture历史动作逐元素一致性检查及12session在线验收后，4090已启动9路、3slot待显存。首批9条件均记录正确continuation_seed（大于原last）且原生test_num=70/策略endpoint健康，通过native_startup_acceptance.json。当前在场景/专家初始化阶段，尚无新完整70条条件。

## 2026-11-01 GPU0增量仿真

用户要求有空闲尽量并行。现场6路（一个旧任务已结束），GPU0约8791MiB且无本轮sim；短暂将Randomized准入降8704，仅让GPU0 place_object_basket入场，随后恢复10240默认以避免重复试入。现7路，旧在途PID保留。新PID2381152已经通过GPU0实际render affinity；完整峰值/首rollout待运行，仍有1536MiB低水位和失败降cap。GPU4/6本轮额外并发曾剩713/93MiB，既有runtime cap降为1，不能拿瞬时闲置误报已证明可加路。证据gpu0_expansion_20261101.json。

## 2026-11-01 GPU1/GPU6按用户要求增量试跑

用户明确允许GPU1和6尝试，即使容量失败亦可重排。原调度runtime_caps已退让（GPU1=0、GPU6=1），配置更新不能修改其内存状态。新增独立schedule_adopt_v2.py：只暂停旧CPU调度器（不暂停仿真），新CPU调度器接管5个在途原PID/attempt/日志及已完成结果。旧父进程保持停止，子进程退出后成为其zombie，接管器从/proc/stat读取真实exit_code，保留0/非0退出码语义；已以真实子进程退出0和7验证。所有5个原退出码记入state后才退休旧父进程。不同锁名防止误唤醒旧CPU调度；不改模型、动作或seed。

新控制器2388201；GPU1新增place_bread_basket PID2388241和put_object_cabinet PID2388247，GPU6新增shake_bottle PID2388433。两个GPU各最多2路；只对GPU1/6允许本轮特定低头寸试入（6144/8704MiB），原1536MiB低水位/最新自有进程停止与失败降cap保留。GPU4维持1路。当前8路、59条件完成，新3路已到render affinity/camera gate，首rollout和稳态峰值尚待验；没有把进程启动当吞吐已提高证据。

证据：adoption_v2_snapshot.json、scheduler_adopt_v2_process.json、activate_adopt_v2.log、scheduler_adopt_v2.log、各client.log。首次接管检查遇到轮询nvidia-smi子进程而自动恢复旧调度，没有影响仿真；检查允许只读nvidia-smi后第二次接管成功。
