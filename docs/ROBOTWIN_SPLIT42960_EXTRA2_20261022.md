# Sidecar42960 跨机评测补充两路（2026-10-22）

## 2026-10-23 03:04 UTC：两路增加持续队列，无需手动补任务

用户指出一条lane不应只有一个任务双域。原六路已具备24项动态队列，
本次复用同一个eval_remaining24.py而非另建调度器：保留默认六端口/24项行为，
增加显式sessions和plan.expected_conditions，worker数量及最终分母随实际计划计算。
新增wait-for-capacity仅供新队列启用：每个条件前free>=11000MiB才启动，否则
保留waiting_capacity状态并每30秒等待，不因容量暂不足直接退出；空队列不会继续等待容量。
单个条件异常保留run.json/错误并继续下一条件；不把异常当0%或自动拼入完整分母。

从官方50任务100条件中排除旧28项（含8partial）、原24项及extra2已安排8项，
剩余40个未安排条件，分成GPU1/4各20项、各2000条。19任务两域+2个单域；
单域为dump_bin_bigbin/move_can_pot的Randomized，另一域已在旧记录中，故没有重复。
两份固定选集为extra2_20261022/queue_plan_gpu{1,4}_20261023.json，交集0，
与全部60项预留/历史条件交集0。保持100条/条件、原预算/RT/near/Sidecar/cache配置。

接续器在4090独立后台运行：GPU1 PID1420405等待原place_empty_cup启动器1418134，
GPU4 PID1420409等待原place_fan启动器1418137。用PID+start_ticks识别原进程，
各自完成即exec自己的长队列，不等待另一lane，不中断当前任务，不再加载模型。
原六路PID578625继续，不改其内存中的调度器；A100 GPU0–3训练PID3501138保持。

日志：extra2_20261022/longqueue_gpu{1,4}_20261023.log；
长队列输出：extra2_20261022/queue_gpu{1,4}_20261023/{state.json,summary.json,<mode>/<task>/run.json}。
当前两个接续器已存活并写WAIT_CURRENT，新40项尚未开始，不把安排队列当作已完成实验。
CPU回归2项通过：原默认六worker/24条件、显式单worker/三条件及容量等待后恢复、
异常继续/完整分母剔除、输出拒绝覆盖；两份实际计划dry-run各20项/一slot通过，
接续shell语法及错误参数拒绝检查通过。测试不调用GPU模型或仿真。

现在不需要用户反复手动启动这两路；等待当前任务、容量不足等待及正常任务切换均自动处理。
完整任务间仍新建原生仿真进程以隔离SAPIEN状态，期间显存短暂释放正常，不是队列停止。
旧8partial与本次异常条件仍独立保留，未悄悄重跑；原队列现有open_microwave Randomized
和open_laptop Clean异常不因新增队列被标成完成。后续若要补齐这些失败条件，需另行
处理资源/错误后独立重跑，不覆盖原证据。长期容量及新队列的第一条实际闭环尚待验收。

## 2026-10-23 02:55 UTC：复用空闲增量服务，继续两任务（第二批）

第一批press_stapler/rotate_qrcode四条件均完整结束，22720/22730两独立slot空闲。
用户要求检查4090 GPU1/3/4并补充仿真：GPU1 free16006MiB、GPU4 free11060MiB，
GPU3仅8850MiB，故仅GPU1/4各补一路；不在GPU3降低安全门槛，不新增模型。
A100 GPU5/6已有四份模型，各free31270MiB，且现有两服务可直接复用。

第二批：GPU1/22720=place_empty_cup，GPU4/22730=place_fan，各Randomized后Clean100条，
共4条件400条。核对与旧28项、原remaining24队列及第一批4项均无重叠。
因此跨机本次计划总量由28条件2800条增加到32条件3200条；新增结果仍独立于旧20项。
复用原run_sim.sh，仅扩展这两个固定合法任务映射，将每条件启动前显存门槛由9000
提高到11000MiB；原任务已全部结束，无运行中调用受改动影响。shell语法及错误slot/task
拒绝检查通过，使用新独占sim_activation_claim_batch2与不可覆盖的原生结果目录。

远端启动器PID1418134/1418137已返回存活，两个端口身份ping通过，实际free读回
16006/11060MiB，均已写START。此时在初始化仿真，尚不将START视为新episode成功。
新产物：extra2_20261022/sim/<mode>/{place_empty_cup,place_fan}，对应任务_launcher.log。
原六路调度器/原端口/隧道保持，A100 GPU0–3训练PID3501138仍存活，未操作。

同时查到原remaining24队列open_microwave Randomized已在61条后exit1：
相机take_picture抛RuntimeError: cannot create buffer。错误已保存在其run.json，
3/61仅partial，不作为完整百条SR。GPU0此时free仅约86–294MiB，资源压力值得警惕，
但没有把该错误未经验证直接归因CUDA OOM。本次不改渲染/相机/预算、不删除partial，
也不自动重跑或拼接该任务。后续应独立诊断并在足够余量上处理缺失条件。

本轮容量仅初始保护，不保证RT/场景/其他作业后续峰值；继续关注新卡余量及旧卡资源压力。

用户授权：在确认显存余量后补充模型/仿真，不影响A100 GPU0–3训练及原六路评测。
复用现有 `remote_eval.py model/sim`、Sidecar FFN数值gate和反向隧道，
不修改模型、训练、任务源码或原24条件调度器。

## 容量证据与边界

在4090端每约2秒采样GPU与进程显存，共45次、96.75秒。
已有六个仿真进程的最大观察值为6779MiB（6.62GiB）；
GPU1、4最低空闲分别13403MiB、11403MiB。
18:39复核二者仍空闲约12084MiB、10853MiB。
按新仿真启动前至少9000MiB空闲进行保护，先各补一个，不直接再加六路。

证据：原运行根目录下 `extra2_20261022/capacity_before.json`，含完整45次采样。
这是采样窗口内最大观察值，不是全部任务/完整生命周期的严格峰值；
其他作业增长、资产和RT/planner峰值变化仍可能导致容量不足。

## 隔离与选集

原根：`outputs/robotwin_eval/split42960_a100_gpu56_20261022T160926Z`。
增量根：其下 `extra2_20261022`。

| 模型位置 | 模型/隧道端口 | 仿真位置 | 任务 | 条件 |
| --- | ---: | --- | --- | --- |
| A100 GPU5新增一份 | 22720 | 4090 GPU1新增一路 | press_stapler | Randomized后Clean，各100条 |
| A100 GPU6新增一份 | 22730 | 4090 GPU4新增一路 | rotate_qrcode | Randomized后Clean，各100条 |

两任务共4条件、400条；与旧20完整条件、8个partial及当前24条件队列均不重叠。
原六路端口22700–22702/22710–22712不变；新端口各一个独立slot。
相同42960权重、10步去噪、cache .99/max3、FFN编译、完整Sidecar读写、
原生专家筛选/unseen/官方动作预算/RT/near0.1，视频关闭。
训练腕部near0.02与官方评估0.1的差异不在本次扩容中更改。

## 实际状态

18:39首次复核发现先前nohup启动未留下存活进程或模型产物，空日志保留，
没有把此前启动命令视作部署成功。
18:40改用 `subprocess.Popen(start_new_session=True, stdin=DEVNULL)` 独立会话，
新增launcher PID3559257/3559259已确认PPID1，部署目录已生成，正在固定权重身份及加载。
模型须通过数值gate和session验证后才启动对应仿真；此时尚不宣称新增两路闭环已通过。

原六路18:39观察完成180条、成功107条，无调度失败；仅为部分样本进度，
不能作本批最终成功率。新旧结果独立保存，不覆盖或自动拼接。

## 产物与下一步

- 新模型：`extra2_20261022/model{5,6}/{server.log,model.json,session_slot0.json,fast_validation/}`。
- 新仿真：`extra2_20261022/sim/<mode>/<task>/{run.json,client.log}`。
- 完整成绩仅计原生 `run.json` 中status=complete且completed_episodes=100的条件。
- 下一步验新服务数值gate、SSH转发、实际动作和扩容后余量，继续原六路及训练。

### 18:44 UTC：自动等待已启动，尚未宣称新增闭环成功

新增模型launcher 3559257/3559259、server 3559738/3559739持续存活，
当前模型日志尚未给出READY，仍在依赖导入/加载阶段。独立反向隧道PID3560227
已启动，只增加22720/22730，不改原六端口隧道。
4090上的两个独立启动器PID871721/871723已返回并写入WAIT_MODEL。

启动器复用本增量目录下 `run_sim.sh`，最多等待模型session30分钟；session文件
由模型数值gate和服务ping通过后生成。随后核对42960权重SHA/Sidecar/slot身份、
跨机端口ping，并在每个条件前再次要求对应GPU空闲>=9000MiB，否则退出不强启。
脚本shell语法、缺参及错误GPU/端口/task拒绝检查通过；仿真启动认领目录
`sim_activation_claim`独占创建防止本批重复启动。新目录原生runner也拒绝覆盖。
这是自动启动已安排，不是新增两条episode或八路闭环已经验收。
