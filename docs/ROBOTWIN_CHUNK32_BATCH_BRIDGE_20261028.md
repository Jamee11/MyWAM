# RoboTwin chunk32 与请求合并加速实验

更新 2026-10-28T09:52:28.677106+00:00。用户已明确授权修改推理通信；评测专家筛seed、成功判据、动作预算、RT相机、checkpoint/Sidecar/cache/去噪参数、不保存视频保持。

## 当前接入状态

六个现有模型、18个session均已准备兼容加速链路，正式registry已启用，只有后续启动的native进程读取。现有在途仿真和模型均未因本次启用重启，仍使用原路径直到任务结束。不能把已启用registry写成所有正在运行的任务均已切换。

## 实现

4090本机每session仍用独立WebSocket；客户端只在原来的32动作重规划边界上传真实三图/状态并取回32动作，逐步执行，提前成功/episode/指令reset清空本地buffer。每步原生get_obs、EEF转换、take_action、物理及成功检查照常执行，不预取，不改变重规划时点。

每个模型有独立relay与4090 gateway。一条persistent WS承载有request ID/session ID的请求包；gateway最多等待2ms、每包最多8条。异步收包与单GPU顺序推理分离，其他仿真可执行动作/继续发请求，模型间并行。每条结果完成即回传，不等待同包所有推理完成；另实现grouped模式仅作对照。丢失不明确的obs请求不重发，不切回另一服务继续同一buffer；上游断线让该attempt失败，由已有原生整组重试处理。

兼容版不重启现有模型：在L20 loopback连续取原server的32个动作，保留原WAMPolicy/binary投影，再一次跨机发送。它仍做原server的逐动作预处理，因此比原生chunk接口慢；只是消除31次跨机传输及客户端PNG编码。新原生chunk服务GPU3一次预处理后经原policy取满一个buffer，已经过两原生fixture跨副本canonical gate；当前为独立测试模型，不宣称已经替换正式六模型。

## 真实GPU通信对照

同一32k SHA、固定两原生观测、三session并发、每session4轮完整32动作。下表仅含通信/模型，没有物理仿真；测试顺序/共享负载也限制因果解释。

|方案|总wall秒|相对逐动作速度|每session平均等待秒|全部32动作对原canonical|
|---|---:|---:|---:|---|
|逐动作直连|18.026|1.000×|4.209|逐元素一致|
|逐动作＋2ms合并|18.949|0.951×|4.474|逐元素一致|
|原生chunk32直连|9.622|1.873×|1.478|逐元素一致|
|原生chunk32＋合并逐条回传|9.706|1.857×|1.526|逐元素一致|
|原生chunk32＋合并整包回传|9.637|1.870×|2.287|逐元素一致|
|兼容chunk32＋合并逐条回传|13.273|1.358×|2.065|逐元素一致|

主要收益来自减少逐动作上传。2ms合并本轮未额外提升总吞吐；整包回传虽总wall近似，平均等待2.287s高于逐条回传1.526s约50%，故正式选择逐条完成即返回，避免阻塞仿真。不得将1.87×通信模型测试当1.87×rollout吞吐。

额外连续4段/不reset对照：旧协议在buffer期间改变观测、两种chunk协议仅在边界送对应观测，三种模式共12段/384动作全部仍与canonical一致。覆盖跨chunk计数、生成条件与缓存状态，不只验证每次reset后的首段。CPU真实policy/executor/ModelClient169步对照也通过，覆盖binary/indices/early termination/指令reset；请求169→8。

## 原生闭环验收与随机性边界

click_alarmclock/demo_clean，固定accepted scenes100000/100002；官方test_num30指令池仍生成，诊断只重放第一轮官方选出的同一指令，做完2个完整rollout即退出，不并入任何正式SR。初次只配scene seed发现指令不同，已保留unpaired结果但不据此计算加速比。上游_base_task.py的Python random.seed已注释，描述生成器使用random.shuffle/random.choice，故same seed并不保证same instruction。正式评测未补种或重放指令。

配对legacy/兼容版两episode初始三图SHA、EEF、prompt一致，首个实际执行action chunk逐元素相同，均2/2成功。第一条均62步，38.702→34.487s，约1.122×；第二条108→67步、54.099→27.045s，不能把其时间缩短全归通信加速。旧协议自身重复也出现32步后EEF微小漂移（约1e-5量级）和后续动作不同；这证明closed-loop跨运行无法当逐bit比较，并未进一步定位到哪一规划/物理源。确定通信等价靠固定输入GPU与连续chunk测试，不能宣称全部闭环轨迹逐元素一致。

更多native原生chunk/stream/grouped结果继续写native_pilot_paired_status.json与acceptance_summary.json；正式方案启用仅依赖已完成的兼容闭环和全部数值门槛。

## 运维与撤销

入口：本批run_native.py读取activate_transport.py和formal_transport_registry.json，按原port映射gateway端口；先核对原session checkpoint SHA及relay返回readiness token，记录transport_bridge_selection.json，不打印token。原dispatcher持有的token/slot锁及队列计划不改。

六模型原32910–32972；对应4090 gateway原端口+1000（33910–33972，按三个slot分组）；L20 relay35410/35420/35440/35450/35460/35470，SSH仍使用root@210.45.70.21:26199，loopback反向转发。正式模型服务/旧329xx隧道保留。

CPU监督supervise_formal_bridge.py分model/sim两端，独占role锁，每3秒监测子服务，仅重启已退出的bridge和已失效的专用tunnel，不重启模型。首次--replace-staged只允许registry未启用且gateway请求为0；正式启用后重启监督不要带此flag。不要按GPU/宽泛命令kill。

撤销只将registry.enabled设false，后续新任务回原路径；已使用chunk的在途session必须继续用其bridge直至任务结束，不能关闭桥或中途换协议。若要完全退役，等对应transport_bridge_selection.json所列native PID全部结束，再按PID/start_ticks停止自有CPU桥及专用隧道；不能kill原模型/共享329xx隧道。

保持已有30+70指令池30/70而非native100的差异说明、原生EEF/near等尚未决项。本轮不改变输入口径以换速度，不删旧结果/不拼被中止的partial。保护回退过的GPU容量不因瞬时空闲盲目恢复。

证据根：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/performance_recheck_20261028T0900Z`。启用记录formal_bridge_activation.json；代码/配置/状态/逐阶段GPU基准/原生actions与输入SHA均在此目录。

## 2026-10-28T10:02:01.599855+00:00：全部闭环测速完成与诊断资源退役

五方案各2条完整rollout共10条均成功；四个加速方案均通过scene seed、初始三图/EEF/prompt和首个实际执行chunk逐元素门槛。以下全为诊断，不并正式SR。后续轨迹可有重复运行漂移，不将不同步数时间比当纯提速。

|方案|scene100000步数/秒|scene100002步数/秒|同62步相对legacy速度|
|---|---:|---:|---:|
|legacy直连|62 / 38.702|108 / 54.099|1.000×|
|chunk_batch_grouped|62 / 35.007|67 / 26.942|1.106×|
|chunk_batch_stream|62 / 35.931|67 / 32.467|1.077×|
|chunk_compat_batch_stream|62 / 34.487|67 / 27.045|1.122×|
|chunk_direct|62 / 33.317|67 / 23.624|1.162×|

模型端分组测试补测（carry_*.json）发生在原生chunk直连阶段初始化/运行期间，可能增加其服务等待，故阶段耗时不是严格受控硬件加速因果；兼容版原生阶段在补测之前完成。1–2条诊断不足以估计整体吞吐或SR差异，正式各任务仍须完整跑完。

已停止仅用于测试的GPU3模型、三路私有gateway、三relay/两私有隧道（含两个模式；按记录逐个核对PID/start_ticks）。GPU3空闲恢复33773MiB，原六模型、18正式gateway/六relay、两端CPU监督与正式专用tunnel保留；原有主评测9路未为启用而中止。所有动作、状态、日志、原代码快照和统计保留。

18正式入口身份/SHA及只读完整ModelClient构造/port重映射检查通过（activation_constructor_smoke/result.json）；没有对运行中正式session执行reset/predict。启用时尚未有正式新任务产生transport_bridge_selection.json，因此不能声称已测得整套正式评测的提速。后续任务自然完成后自动读取registry切换；已有partial不为切换而中止/拼接。

代码与证据索引已写final_source_manifest.json及diagnostic_cleanup_{model,sim}.json；正式registry已启用。


## 2026-10-28T13:07:00.444552+00:00：实际加速路径耗时占比

stack_blocks_three/Randomized attempt3记录2825个动作，进程1395.128s、3完整rollout后被显存保护中止；不是完整每任务100条均值。动作执行占控制阶段65.53%/进程55.38%，取图24.49%/20.70%，编码/RPC/排队/推理9.98%/8.43%；其余15.49%为未分项残差，含启动、expert、场景重建/退出和计时未覆盖开销，不能全当expert时间。独立62动作重放中，双臂规划占take_action78.44%、纯physics9.45%；只能估计整步规划约一半、纯physics约6%，不能把跨任务不同样本拼成严格同一分项计时。证据rollout_breakdown_20261028.json。
