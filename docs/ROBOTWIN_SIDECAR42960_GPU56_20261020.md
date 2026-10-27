

### 2026-10-20T02:33:16.661422+00:00：最终42960权重GPU5/6独立标准评测

用户明确授权最终checkpoint_step_42960.safetensors（不是32k、不是消融）。权重固定到outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z/checkpoint_step_42960，24862363126 bytes，SHA256 501addbed687e6d7587af60ab4480b9c79eaa2c8867d723b74d54e0acc9deb55；weight只读hardlink，config/stats/contract/tokenizer独立物化。42960micro=21480参数更新。

本机A100 GPU5/6初始各约81GB空闲，使用两份模型、每模型3独立同步端口/互斥计算锁，六路sim均限制在GPU5/6，不操作0–3上的ManiSkill或其他进程。模型沿用已验证本地RAM venv，仿真用官方Conda及此前独立A100 RoboTwin/sm80扩展，五模块hash复核与已验收一致；process-only现有gpucomp预加载及NVIDIA ICD修复保留，无驱动/依赖/模型源码改动。通用fallback TORCH_EXTENSIONS_DIR仍由旧runner指向共享目录，本次用已安装sm80库，不触发/清理通用JIT。

评测完整Sidecar，10步、32动作、CFG1、sync、wxyz自有stats、384×320、joint velocity cache .99/max3及Sidecar局部FFN compile；每份新权重须两真实fixture、原atol=.005/rtol=.01和真实kernel门槛通过才放行。保留官方native专家筛seed、seed group0、unseen、RT32spp/depth8/oidn、原动作预算、无planner fallback。50任务两域各100条，共10000；任务固定无重叠两分区，优先瓶子/锤子/鞋子/扫码等。每端口独立动作buffer，缓存每generation reset；缓存近似误差仍须闭环评估，不把FFN匹配其cache参考当cache等价完整计算。

当前模型加载，队列等待numerical gate，尚无正式42960成功率；进程/命令、快照与独立监控都在新输出根。实时文档docs/ROBOTWIN_SIDECAR42960_LIVE_20261020.md只读取本次两queue，只验收完整exit0/100条/无errors；异常/部分仅观察进度，不与32k、其他机器合并/择优。未来有渲染异常需保留证据、排除异常未完整项，不降低协议冒充修复。


### 2026-10-20T02:37:45.960890+00:00：两模型真实数值验收通过、六路评测父进程已启动

两GPU分别通过两真实RGB fixture、匹配cache条件四比较的原门槛，max_abs_error全部0.0，各有4个真实Inductor kernels。warm cache+FFN generation均值GPU5约0.2619秒/GPU6约0.2649秒，不是整episode速度。每卡3路standard_eval父PID与本次输出/权重实际命令一致，任务为瓶子两域/鞋子Clean及锤子两域/扫码Clean；当前做权重校验和仿真初始化，未据父进程running声称策略episode完成。模型/队列/监控均独立后台运行，无新GPU依赖改动。


### 2026-10-20T03:53:18.972628+00:00：按用户授权利用GPU4/7扩容最终42960评测

现场GPU4/7各约40191MiB空闲、总A100-80GB；未知namespace原40GB占用不处理。新增GPU4同一42960模型PID2462558，端口21820/21821，复用现有serve_checked_ffn及完全相同checkpoint/deploy，独立gpu4/inductor；CUDA放置门槛33000MiB，尚须真实FFN数值验收和PONG。GPU7拟新增2路sim，逐任务free>=14000MiB且第二路延迟45秒；不向GPU4叠加sim，目标3模型8路（当前仍2模型6路）。

输出内expand_gpu4.py/PID2463380复用已验收expand_gpu6.py的freeze_source/exit_code/verify/save，并将helper.OUT绑定本次42960输出。只在GPU4验收后暂停本次GPU5原调度器，原三standard_eval/native客户端继续原PID，原生百条不中断；替代队列expanded_queue54接管其50个task/mode，五worker含原GPU5三个slot与GPU4两个slot，GPU6原50个不动，总100唯一条件。源三子进程终态及真实exit status记录后才回收暂停的旧调度器；不要手动SIGCONT旧调度器造成重复派发。

监控已改为读取原子queue_layout.json，未提交接管前仍读原gpu5/queue+gpu6/queue，提交后仅expanded_queue54+gpu6/queue；旧监控源文件保留watch_results_before_expansion.py。模型/训练/数据/RT/预算不改，不操作0–3的ManiSkill。独立输出脚本语法、自带100条件唯一性/重复拒绝，以及目标Python3.10真实exit7僵尸退出码检查通过；系统Python3.8不支持该helper的waitstatus_to_exitcode，控制器明确使用本地3.10，未修改系统或原helper。当前source三job PID核对存在，全部原episode保留。验证产物expansion_preflight.json。

新增卡长期容量、两路真实闭环仍待实际加载/场景创建，free瞬时满足不能保证未知作业不增长。保留日志、失败不计有效百条；无自动降低RT/动作预算或结束外部进程。


### 2026-10-20T03:58:51.913926+00:00：GPU4模型验收通过，最终42960扩至3模型/8个评测父进程

GPU4两真实fixture四个匹配cache参考比较max_abs_error全0.0、真实4个Inductor kernels，原门槛通过；warm cache+FFN约0.3327秒/chunk，不是整episode吞吐。模型仍完整三处Sidecar写入，不改变cache/动作预算/RT。GPU4当前free约15751MiB；GPU7新客户端尚在权重检查/初始化，free约40192MiB，不能据此提前保证两sim完整renderer峰值已验收。

03:57接管提交：旧GPU5调度器2439689仅SIGSTOP，原GPU5三outer PID2442123/2442126/2442125、GPU6三outer PID2442127/2442124/2442122均保留且存活。新增GPU7 shoe/Randomized PID2466152、dual_shoes/Clean PID2466329已启动，两新父进程目前初始化，不提前称已完成新episode。新权威队列expanded_queue54与gpu6/queue恰好100个唯一条件、8个running、errors为空，监控无报错。原source快照保留且不再统计冻结gpu5/queue；原三job真正退出并记录状态后才回收暂停调度器。

实际隔离子进程自检也已通过：暂停仿真stub父进程后，三子进程继续并以0/7/0退出，真实退出码完整读取，不触碰当前评测PID；证据adoption_selfcheck/result.json。扩容验收expansion_gpu4_acceptance.json、实时新状态见原实时文档。后续关注GPU7两sim真实动作、峰值及外部显存变化；尚无本次新增通道的完整百条结果。


### 2026-10-21T02:15:35.771534+00:00：最终42960与官方issue46共同完整条件对照及专家筛选说明

官方来源：https://github.com/OpenWAM-Official/OpenWAM/issues/46#issuecomment-6118279946；现场GitHub API获取并封存official_issue46_snapshot_20261021.json（50任务表）。只计本次exit0、complete=true、requested/completed均100、errors为空的共同条件；共5个Clean、3个Randomized。

|任务|条件|官方|42960|差值pp|
|---|---|---:|---:|---:|
|adjust_bottle|demo_clean|90|99|+9|
|adjust_bottle|demo_randomized|51|50|-1|
|place_shoe|demo_clean|98|94|-4|
|place_shoe|demo_randomized|51|52|+1|
|place_dual_shoes|demo_clean|89|87|-2|
|beat_block_hammer|demo_clean|99|98|-1|
|beat_block_hammer|demo_randomized|55|44|-11|
|scan_object|demo_clean|93|91|-2|

共同5个Clean等权均值93.80→93.80（0.00pp）；共同3个Randomized52.33→48.67（-3.67pp），主要负差为hammer Randomized55→44（-11pp）。不是全50平均，不用未完整比例代替成绩；也不是逐场景/指令/接受seed严格配对，不能单独归因EA或用本子集均值直接对比官方全50均值。官方训练run、硬件、精确RoboTwin revision/accepted seed清单未从该issue建立一致性，部署FFN策略也有差别；本结果仅报告值同任务描述性比较。

用户所问容器任务明确为place_container_plate。现场Traceback发生在专家play_once/grasp_actor生成pre_grasp_pose为None，尚未进入对应候选seed的学习策略rollout。实际上游script/eval_policy.py:222-252先setup/play_once专家可解性检查；UnStableError/Exception或plan_success/check_success未通过时递增candidate seed并continue，通过后才执行策略并计入100条。策略失败/超时仍计失败，不能按策略结果跳过。wrapper在此仅打印栈后raise，跳过由未改的上游native循环决定。此处上游捕获较宽，不能据遵循上游就认为所有异常均可忽略；本次具体已定位为专家抓取目标None，渲染/OOM须另行归为基础设施诊断。

现场run.json与client.log/git复核：仿真prefix .conda/robotwin-official，Python3.10.22/Torch2.4.1/SAPIEN3.0.0b1，RoboTwin HEAD 0aeea2d669c0f8516f4d5785f0aa33ba812c14b4且tracked diff为空；复用私有A100 sm80运行时，模型仍最终42960。step_limits.yml全部override注释、加载contents=null且与official checkout文件一致，沿用上游预算；RT开启、planner fallback关闭。源码目录历史名含sidecar32000不表示当前加载32k权重。

未改变当前代码/配置/种子筛选/评测进程。数值比较与官方快照存于本次输出根；下一步待更多完整百条，继续按相同条件更新；若需纯架构因果对照，另做同数据/训练预算/固定manifest的baseline。
