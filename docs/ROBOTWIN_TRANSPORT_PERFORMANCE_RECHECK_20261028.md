# RoboTwin通信冗余与当前降速复核

更新：2026-10-28T09:11:26.135978+00:00。正式队列仍使用逐动作请求；仅添加缓冲计时器，未切换chunk通信。

## 真实路径计时

当前32k，RTX4090 GPU1三仿真共卡，`stack_blocks_three/demo_randomized`。丢弃包含冷启动的第一批，采样398次take_action；不是全任务平均或硬件因果对照。

|阶段|平均秒/动作|说明|
|---|---:|---|
|take_action|0.3586|含IK/轨迹规划、物理、成功判据和内部render update|
|get_obs|0.1215|含原生相机读取/渲染，不等同纯render-only|
|客户端step|0.1266|含三图PNG、RPC、服务预处理/推理/排队|

RPC均值0.0973s，返回server latency均值0.0484s；二者差值0.0489s包含传输、序列化和共享engine锁外排队，不能全称网络耗时。每图PNG均值0.0098s，三图约三倍。表格阶段不重叠；PNG和RPC已包含在客户端step，render_update计时嵌套于其他阶段，禁止重复相加。

总计0.6068s/动作。保持get_obs及原生take_action时，即使客户端成本完全消失，该路理想上限也仅1.264倍；真实仍须模型生成，故不能承诺几倍/几十倍。

## 跨机与历史复核

独立WebSocket/SSH反向隧道测试，使用真实fixture的三图无损payload（114417bytes），回包模拟20D动作并解码PNG；不访问正式模型端口。单路32请求平均39.55ms，11路352请求平均37.87ms。不同SSH连接，未覆盖正式隧道排队或GPU推理；只能说明本独立路径未出现严重带宽拥塞。诊断echo与隧道已停止。

相同任务历史32k对比：当前/历史每秒action计数约0.708–1.420倍（排除刚重启尚未动作的探针）。有部分任务降低约14–29%，不支持普遍慢数倍。两批seed/场景/成功步数/共享GPU负载不同，不是严格配对。当前长预算任务集中，不能将整体每小时rollout和旧整套平均直接比较。见same_task_step_rates.json。

## 已做独立适配，尚未正式启用

`chunk_transport.py`为独立、显式握手启用的v1适配；原默认入口/官方代码/模型/训练不改。服务只在原sync executable buffer空时生成一次，通过现有WAMPolicy的binary投影取完整32动作；客户端本地逐步消耗，仅下一次原重规划时点上传当前观测。每步native get_obs/take_action、专家筛seed、成功检查、动作预算、相机/RT保留；不预取、不使用旧观测提前生成、不将不同session合并模型状态。

独立入口serve_chunk.py复用当前模型SHA/固定fixture/跨副本逐元素一致性门槛及原shared-engine锁；run_chunk_native.py复用原生入口。正式模型/dispatcher/隧道均未指向这两个入口。

CPU验收：真实WAMPolicy、SyncInferenceExecutor及ModelClient，合成确定性engine，169个动作、8次生成边界；输出逐元素一致、engine观测相同、action_indices与binary投影相同，early termination/reset/中途指令切换通过；客户端169次请求降为8次，PNG共531次=3×(169+8)。engine接口仅为CPU测试提供类型占位，未加载真实权重。此为CPU控制/通信等价性，不是GPU动作数值或完整仿真验收。

待验收：真实checkpoint两个fixture全部32动作逐元素一致、独立native固定seed的动作/终态/步数一致、真实端到端吞吐；reset/网络中断不得重发不明确的chunk请求，故chunk采用predict_once，失败必须重置整episode。当前不开跨sim等待拼包；是否有收益要由队列/模型计时决定，不能加全局barrier。

## 运行变更与保留

为获得即时计时，仅重启新增GPU1 stack_blocks_three Randomized PID3748448（当时约5条）；原attempt1完整保留，partial不并正式SR。队列已自动创建attempt2 PID3756264，加载observer并真实推进，其他十路保留。主动诊断中止不是策略失败；主队列基础设施错误记录会包含该中断，不能再称errors=0。后续进程自动加载同一个仅转发原调用一次的observer，每约30秒批量记录，不写每动作文件。原run_native完整备份。

## 后续与边界

按AGENTS.md第2条，正式切换核心推理通信逻辑前需解释并确认；独立方案已具备可审阅代码和CPU结果。先单路真实验收，通过后只在episode/任务边界切换成对server/client，禁止只换一端、禁止在仍有buffer的活跃session中切换。当前30+70候选指令池差异保留用户选择，不以性能改动悄悄修正；GPU余量保护、原生近裁剪/EEF未决项、原始结果保留。进一步加速需定位take_action里的规划/物理/共享GPU争用，减少通信不能替代这部分工作。

证据根：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/performance_recheck_20261028T0900Z`；真实trace：`/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/fullbackward32000_supplement70_4090_20261028T035728Z/jobs/demo_randomized/stack_blocks_three/attempt2/performance_observer.jsonl`。
