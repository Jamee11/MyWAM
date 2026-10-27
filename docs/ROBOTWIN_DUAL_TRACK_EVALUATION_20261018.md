# 2026-10-18 RoboTwin 双轨评测

用户授权在OpenWAM下重新下载完整官方源码、镜像资产及新建Conda仿真环境，GPU0–3用于官方复现及对照；GPU4–7用于Sidecar20k现有仿真栈全量评测。原DiT4DiT与原run产物保留。

## 官方复现

- 源码：`.external/robotwin-official-20261018/RoboTwin`，重新git clone，固定OpenWAM验证的0aeea2d669c0f8516f4d5785f0aa33ba812c14b4；cuRobo单独clone固定v0.7.8。
- Conda prefix：`.conda/robotwin-official`，Python3.10，Torch2.4.1/torchvision0.19.1、SAPIEN3.0.0b1、Warp1.13.0、mplib0.2.1，清华镜像安装中。独立编译扩展，不写旧env/source。
- HF镜像固定资产revision3dc3b798668feb99ac61cc9086d84cbcc3d79186，三个zip上游LFS size/SHA256；保留zip/partial。镜像返回的CAS按项目既有授权直连、不用代理，不访问官方API、不自动fallback。下载及逐文件比对记录在`outputs/robotwin_eval/rebuild_official_20261018/`。
- 模型独立官方scripts/deploy.py，GPU2，端口19400，发布Clean2Random权重，默认cache/compile。优先place_dual_shoes、place_shoe、scan_object、handover_block，再用click_bell作近似匹配控制；两域各100条，不挑容易seed、不改成功判据/预算/RT。尚在环境构建及模型加载，不能称完成。

## Sidecar现有环境对比

- 指定原训练run中checkpoint_step_20000.safetensors固定暴露到`outputs/robotwin_eval/sidecar20000_currentenv_20261018/checkpoint_step_20000/`，config/stats/tokenizer/sidecar_contract引用同一原run。20k为文件命名micro-step，训练marker对应10000次optimizer更新，不能写成20000次optimizer更新。
- 两个独立模型pool：GPU7端口19200–19205、GPU5端口19300–19305；每模型6独立policy/deque、共享engine串行predict，simulator分布GPU4、6。50任务分为不重复两组各25，合计100(task,mode)×100=10000条。
- runtime `.external/RoboTwin`、Python `.venv-robotwin/bin/python`、现有资产、原生筛seed/预算/成功判据、unseen及RT与历史baseline相同。
- Sidecar使用自身已有deploy_relational_sidecar.yaml（10步/sync/32动作/9视频），无oracle或旧EA effectors，policy_config_relational_sidecar.yml按训练合同wxyz。checkpoint自有归一化stats不替换成发布模型stats。
- 关键不等价：Sidecar实现显式拒绝DiTcache，且未支持原版compile路径，因此cache/compile关闭；原发布模型的历史测试cache/compile开启。这是部署差异，不能直接叫完全相同部署条件。已额外启动发布模型在相同关闭cache/compile配置、原模型自身表示合同下的完整控制队列（GPU1模型，GPU0/1/3仿真），目录`baseline_matched_sidecar_deploy_20261018`；同环境/协议/推理设置比较，但训练数据和模型不同，不是纯Sidecar结构消融。

## 实施验收状态

评测启动器仅新增可选deploy/policy/runtime/任务子集参数，原默认保持；模型文件固定选取，复用服务比对配置SHA256。16项已有评测/隔离/互斥回归测试通过（161.49秒）。模型服务与队列创建并不代表READY、真实动作或成功率，下一步须核对strict加载、真实rollout及最终coverage。

初步新资产检查：Aloha URDF与旧资产字节相同，config.yml仅planner字符串由旧mplib_screw变为下载包curobo；本版Robot.set_planner始终创建CuroboPlanner供left/right_plan_path，planner_type只用于额外MplibPlanner，因此不能仅据这个字符串认定旧EEF实际走了mplib。完整资产比对仍在进行。


## 运行补充

官方OpenWAM亦已重新完整克隆到`.external/robotwin-official-20261018/OpenWAM`，HEAD=1e103e1c61905c230dab6b0f99980370cf2f9e3c；GPU2改为该全新仓库原始scripts/deploy.py，旧预备服务在READY前停止并保留日志。独立仿真诊断将直接执行新仓库未经修改的single_eval.sh/wrapper，不用本地分母AST修复，100条由原生main给出；单服务严格每次仅一个native客户端。诊断等候依赖/资产完整验收及action parity进程退出，防止与parity共用19400的动作缓存。

Sidecar20k权重已物理复制为独立文件24862363126 bytes，config/stats/tokenizer引用同一训练run。两个服务PONG及12原生仿真进程确认，已实际生成动作并调用take_action；beat_block_hammer Clean与Randomized分别已有首个完整episode成功（各1/1，仅链路证据，不能作任务SR）。其余任务正常推进，无当前观察到的OOM/堆栈。

镜像资产三zip均已完整下载与LFS size/SHA验证，解包/逐文件比较继续。系统bootstrap Python3.8不支持Path.is_relative_to，初次解包校验因此退出；已兼容修复并复用所有原下载zip，没有重新下载或官方fallback。后续将逐条path.resolve改为词法路径检查，避免共享盘逐级metadata RPC，仍拒绝绝对/上级路径和symlink条目。Aloha机器人包仅config.yml planner字符串与旧资产不同，URDF及其他Aloha文件相同；其余机器人差异不用于Aloha归因。


### 当前正式队列：balanced_queue7 / balanced_queue5

初始12路实测GPU4/6各6个sim时余量降至约3.2/4.4GiB，而GPU7仍空闲约20GiB。仅终止本次两个pilot队列自有进程组，模型保留；pilot日志/初始episode均保留但不计正式SR。正式重启为每pool5个sim，GPU序列4,6,7,4,6，总10路：GPU4四路/GPU6四路/GPU7两路，权重服务仍GPU5/7。正式目录balanced_queue7、balanced_queue5各25任务×两域×100；从seed group0重新独立计数，不混合pilot，不操作历史baseline尾部3项。见capacity_restart.json与balanced_queue_processes.json。


### 2026-10-18T06:43:58.768972+00:00 cuRobo源码版本表述修正

重新clone官方v0.7.8后发现其tag指向d64c4b005459db10c5dd867d8b30a87d5bda9bdb，旧cuRobo HEAD同为此提交且两者tracked diff均空。旧describe=v0.7.7-9-gd64c4b0是旧checkout缺少v0.7.8 tag造成的版本描述，不代表算法源码不同；不能将前文版本字符串差异当作cuRobo源码升级/已定位的根因。新环境已实际编译安装nvidia_curobo0.7.8（Torch2.4.1），旧扩展绑定Torch2.7.1；SAPIEN/Torch/Warp及二进制运行栈仍有差异。证据outputs/robotwin_eval/rebuild_official_20261018/curobo_source_equivalence.json及git ls-remote tag。当前正式Sidecar队列为balanced_queue7/balanced_queue5共10sim，旧12sim pilot因实测显存余量仅3–4GiB而停止并保留，不混入正式SR。


### 2026-10-18T06:47:38.838199+00:00：真实独立服务/共享池动作对照通过

fresh官方OpenWAM服务19400（GPU2）对原共享pool19000/19001（GPU0），发布同一权重、官方默认cache/compile/10步，使用两个真实HDF5首观测的三视图/EEF/固定指令，各32动作；池两端口交错请求，再reset并交换观测。四组序列均bitwise_equal=true、max_abs_error=0.0。只证明此真实fixture范围，不外推所有输入/长期共享状态或Sidecar模型；本次已明显降低“共享池本身改变原模型action”的嫌疑。证据outputs/robotwin_eval/rebuild_official_20261018/action_parity/result.json，原始请求inputs.json保留。Sidecar原20k与独立副本SHA256均0d67d651577ef53d5a86b0e1e03dd93f6cc7bf652c3c7af08734b7712042ea6e，size24862363126，copy_verification.match=true。新物体包9368文件与旧资产全同；背景纹理还在逐文件核验，官方native诊断仍等新环境完整编译验收。


### 2026-10-18T06:54:16.855785+00:00：旧baseline尾项渲染失败与完整重跑

stack_bowls_three/demo_randomized在55/92、seed100178后，下一episode获取left_camera RGB时报RuntimeError cannot create buffer，原100条作业已标failed/incomplete，不能作为最终SR。发生于Sidecar12sim pilot扩容期间，GPU6此前仅余约4GiB，可能受显存压力影响；不能仅凭异常字符串断定所有底层原因。保留全部旧日志，Sidecar已调为10sim并分散GPU7；在GPU2较大余量下以同一旧环境/权重/cache/预算/seed group0完整重跑100，目录historical_recovery_stack_bowls_three_randomized_20261018，复用原pool19013。禁止将旧92与重跑拼接、禁止选择较高分；最终汇总应以有效完整重跑替换此task/mode，其他原完成项保留。旧parallel.json继续保留失败事实，不手工改成complete。


### 2026-10-18T07:00:38.577430+00:00：官方独立环境与资产完整验收

Torch2.4.1+cu121/SAPIEN3.0.0b1/Warp1.13.0/mplib0.2.1、cuRobo0.7.8五模块、PyTorch3D0.7.8均在新prefix安装/编译完成。runtime_smoke.json确认五个本地.so实际导入、CUDA张量有效及RT32spp/depth8/oidn离屏32×32RGBA有限。完整依赖检查发现Conda默认继承/root/.local用户site；已启用PYTHONNOUSERSITE=1、在新prefix补齐缺失依赖，pip_check_isolated.log为No broken requirements found，isolated_import_paths.json确认九个关键模块均新prefix加载。旧环境不改。第一次native仅作为preflight_usersite保留，正式official_diagnostic100隔离重启，不混结果。

镜像三zip均上游size/SHA通过，物体9368/9368、背景11000/11000与旧资产同hash。Aloha除config.yml planner字段外文件相同；官方update_embodiment_config_path.py实际生成六个本地路径配置，left config替换Root前缀后与旧版逐字相同。旧流程实际引用InternVLA目录URDF/碰撞文件，已只读比对与本地副本hash相同；不能仅由副本无symlink宣称旧流程完全不依赖外部路径。Sidecar按用户要求保留该现有流程，新官方流程全部绑定新目录。资产、源码、runtime与动作对照证据见rebuild_official_20261018。

Sidecar正式10路已完成首批episode，但未满任何任务100条，不给总体效果结论；初始pilot不计。原版同cache/compile关闭控制4路继续运行，原baseline失败尾项完整100重跑继续，保留旧不完整92条及失败事实。


### 2026-10-18T07:04:32.049446+00:00：双轨真实闭环启动验收

独立官方环境正式place_dual_shoes/Clean已经专家筛seed后进入策略take_action，现场最后步数[('102', '600')]，原预算600，源码/独立服务/环境/资产均新路径；native环境PYTHONNOUSERSITE=1实查。Sidecar正式10sim与原版同部署控制4sim均推进真实闭环；当前未满单任务100，不能给总体SR。16项回归通过、六个修改/新增Python入口ruff通过；新Conda isolated pip check通过。原DiT4DiT diff hash仍为63b6f6bd72695fd2d2c573427b49e59ed3721c8d941b48864899390a8ed1f884，与首次复制记录一致。旧尾项92条渲染失败保留并完整重跑，不混入有效百条。后续按完整结果核查accepted seeds及差值，单个episode/当前子集不作效果结论。


## 当前入口和结果位置

新仿真环境使用`conda activate /remote-home/jinminghao/OpenWAM/.conda/robotwin-official`，并设置`export PYTHONNOUSERSITE=1`；大文件下载HF_ENDPOINT保持https://hf-mirror.com。当前正式诊断由run_fresh_official_diagnostics.py调度，实际执行新官方仓库原生single_eval.sh，每项完成后写result.json，保留原生_result.txt和录像。

- 官方五任务：outputs/robotwin_eval/rebuild_official_20261018/official_diagnostic100/run.json；调度日志diagnostic_queue_isolated.log。
- Sidecar正式全量：outputs/robotwin_eval/sidecar20000_currentenv_20261018/balanced_queue7/parallel.json及balanced_queue5/parallel.json；两个25任务分区共同覆盖50×2×100，不重复、不合并pilot。
- 原模型同关闭cache/compile控制：outputs/robotwin_eval/baseline_matched_sidecar_deploy_20261018/queue/parallel.json。
- 历史尾项恢复：outputs/robotwin_eval/historical_recovery_stack_bowls_three_randomized_20261018/run.json，替代原失败task/mode完整结果，不改旧失败事实。
- action_parity/result.json为四组真实动作bitwise一致证据；runtime_smoke.json为新CUDA扩展/RT证据，isolated_import_paths.json与pip_check_isolated.log为隔离和依赖证据；assets/*.manifest.json为上游revision/归档hash/逐文件比较证据。

后续结果比较必须同时报告完整任务数和accepted seed集合；传统流程每次重新专家筛选，seed group0不意味着跨进程的物理轨迹或接受集合必然逐位相同。相同真实输入动作parity通过不等于闭环轨迹比特复现；Sidecar与发布模型训练数据/表示合同亦不同，不将本组对比宣称纯架构消融。


### 2026-10-18 09:14 UTC：首个完整Sidecar结果及新官方环境初步改善

新官方环境place_dual_shoes/demo_clean已34/36（94.44%），尚未满100；旧环境该项完整13/100，官方issue46报告89/100。结合前述真实输入action parity完全一致、物体/背景资产hash一致，当前证据支持优先排查仿真运行栈差异，但单任务部分样本不能证明具体SAPIEN/Torch/Warp版本的因果作用，也不能外推其余任务。继续原生100条，不提前停表。

Sidecar正式两分区合计约370/10000条（约3.7%），首个完整task/mode为place_container_plate/Clean=99/100，无result错误，与历史发布模型该项99/100相同；同cache/compile关闭控制此项仍排队，不能称已完成匹配部署比较。其余当前部分样本：adjust_bottle/Clean=43/46、Randomized=6/21；beat_block_hammer/Clean随后55/59、Randomized=22/56；place_cans_plasticbox/Clean26/26、Randomized15/17。仍不能给50任务综合宏平均。

读取各日志相同前N完成episode进行对照：adjust_bottle/Clean前46条Sidecar43/46、匹配部署原版45/46，accepted seed前缀全部相同；Randomized前21条Sidecar6/21、原版5/21，seed前缀全部相同。beat_block_hammer/Clean前59条Sidecar55/59、原版40/59；Randomized前51条Sidecar20/51、原版18/51，但这两项accepted seed前缀不同，不能视为逐seed配对改善。总体看Sidecar部分Clean任务表现积极，Randomized瓶颈尚在，须等待完整覆盖再判断。下一步继续百条及全量队列，按有效完整结果和accepted seeds复核。


### 2026-10-18 16:13 UTC：官方双鞋百条完成、Sidecar六项完整结果

新官方环境place_dual_shoes/Clean完整92/100，result complete=true/errors=[]；旧环境13/100，官方issue46为89/100。当前Randomized3/20，仍部分样本。相同发布模型、已核对资产及真实输入动作parity条件下，新官方栈显著改善该Clean任务，但尚不能归因到某个独立依赖。后续继续原定五任务双域100。

Sidecar正式两个分区各3 complete/5 running/42 queued，共6/100项、1036/10000 episodes（10.36%）。完整结果且errors均空：adjust_bottle/Clean95；beat_block_hammer/Clean92、Randomized35；place_cans_plasticbox/Clean100；place_container_plate/Clean99、Randomized70（每项分母100）。同cache/compile关闭原模型完成4项：adjust_bottle/Clean97、Randomized33；beat_block_hammer/Clean65、Randomized33。锤子Clean完整任务差值+27个百分点，Randomized+2个百分点；专家接受seed集合可能不同，不能视为严格逐seed配对/纯架构消融。Sidecar当前仍旧环境，双鞋Clean3/31，应与新官方环境92/100区分。

历史原版99项完整、1项旧失败；指定stack_bowls_three/Randomized完整重跑68/100已完成（取代旧55/92，不拼接/选高），因此合并有效覆盖100项/10000条，总5981成功，综合59.81%，Clean80.66%、Randomized38.96%。旧parallel.json失败记录保留。

速度：各服务最后500个完整10步Joint denoising循环，读取最终tqdm rate估算循环时长：Sidecar GPU7均1.18064s、中位1.19617s；GPU5均1.53218s、中位1.53139s；原模型同cache/compile关闭均0.94623s、中位0.94029s；旧原模型cache/compile开启均0.39028s、中位0.41425s。Sidecar相对优化开启历史运行约3.03–3.93倍循环耗时，相对均关闭优化原版约1.25–1.62倍。此为现有并发负载下循环观测，非隔离硬件严格架构benchmark，也非同步CUDA事件或端到端请求延迟（不含排队/预处理/仿真）；两Sidecar GPU任务负载不同，不将全部差值归因于模块。数据来自各pool.log，未改变现行评测或插入请求。

下一步维持正式队列推进，等待更多完整task/mode，分别汇总新官方环境诊断与旧环境Sidecar/匹配部署控制，不把当前混合任务部分样本平均当全量指标。


### 2026-10-18 16:17 UTC：按用户要求计算原模型替换汇总

对原模型历史99项有效百条加指定失败项完整恢复，确认100项/10000条，Clean4033/5000=80.66%、Randomized1948/5000=38.96%、综合5981/10000=59.81%。新官方诊断当前仅place_dual_shoes/Clean完成有效百条，13/100→92/100，增加79个成功（单项+79个百分点）。仅替换这一项后：Clean4112/5000=82.24%（+1.58个百分点）；Randomized仍38.96%；综合6060/10000=60.60%（+0.79个百分点）。此为混合新旧环境的假设替换汇总，不能命名新官方环境全量成绩；Randomized及其余诊断项未满100不纳入。后续对预先指定任务所有有效完整重测均替换，无论升降，避免选择较高分。旧原始文件不改。计算快照：outputs/robotwin_eval/rebuild_official_20261018/mixed_environment_replacement_snapshot_20261018T1617Z.json。


### 2026-10-18T16:28:03.303620+00:00：按用户授权切换新环境Sidecar与独立加速验收

用户明确要求停止旧环境Sidecar及四路关闭优化原模型对照，保留新官方环境原版五任务轻量诊断，并将大部分GPU转向新环境20k Sidecar，探索cache/compile同时开启。旧三master已SIGTERM并确认无running项，产物保留为incomplete；旧19000/19500闲置模型释放，19400官方模型/原生诊断不动。先复用旧Sidecar已加载模型在新Conda/新RoboTwin做两个2条链路smoke（不并入正式成绩）。

新增仅显式运行时安装的benchmarks/robotwin/sidecar_fast_runtime.py：默认训练/部署/架构源码不改；编译对象是完整Sidecar forward，含7/15/23三处read/write，禁止复用原版foundation-only编译循环。沿用Base.generate原生联合video/action velocity缓存（0.99/max3、每generation reset），其近似跳步会改变输出，须单列加速结果且验收。check_sidecar_fast_runtime.py先以两真实RGB/正确wxyz state固定输入比较eager/缓存/compile/二者，记录CUDA同步耗时、完整动作、缓存统计和实际生成kernel；compiled对各自eager参考数值容差atol0.005/rtol0.01并非严格bitwise。尚未验收通过，不能宣称可用/提速。原文件不覆写，GPU0单独验证中。


### 2026-10-18T16:40:03.953137+00:00：新环境smoke通过；默认融合compile未通过数值门槛

Sidecar新官方仿真瓶子Clean与双鞋Clean各2/2、exit0、complete=true，均只作链路验收，不计正式任务成绩。GPU0同两真实输入、每组3次CUDA同步warm engine.generate：eager0.95865s、cache0.43736s（每10步skip6）、compile0.62616s、compile+cache0.26386s。默认Inductor融合相对各自eager出现max0.01953125/0.0078125动作误差，预设atol0.005/rtol0.01未通过；launcher正确拒绝正式队列，完整失败产物保留fast_validation_default_fusion。不能将失败改称parity通过，也未放宽门槛。

读取安装版Torch2.7.1 torch/_inductor/config.py说明：低精度融合默认省略BF16中间downcast/upcast；新增compile options emulate_precision_casts=True/force_same_precision=True以保留eager舍入边界，未改模型权重/计算拓扑/训练/缓存阈值。原服务停止后，新候选0验证及1/7服务用新选项重新加载，仍须相同门槛复验；若仍不通过则不能按精度等价放行。保留原官方19400单会话诊断。


### 2026-10-18T16:58:50.387754+00:00：Sidecar编译范围隔离与原版9小时并行验证

第二轮仅保留BF16舍入仍未通过原门槛（compiled max0.0234375、cached对应0.0078125）；原始产物保留fast_validation_precision_casts，不放宽容差。第三轮同一已加载模型依次测试conservative全forward（关闭pattern/epilogue融合、fusion size1）、仅Sidecar read/write、仅三层Sidecar FFN，全部保留原主干/三处读写，均按原eager/同缓存eager参考检查。新增serve_sidecar_checked.py在同一模型上完成验证后直接服务，避免重复加载；只有通过者才写最终deploy_selected.yaml/打开服务。四项CPU测试通过130.11s，相关入口ruff通过；实际GPU数值/速度仍待本轮结果。

用户进一步要求原版异常任务至多9小时出结果。原流程实际仅1个native客户端，仍原样保留；新增官方原有labtasker manifest/run_eval调度（源码不改），3个专家清单builder已实际筛seed，8个策略worker等待100条封存清单。任务为双鞋Randomized及单鞋/扫码/递积木/铃铛两域，共9项900条，每项10个无重叠10条分片。新原版模型pool GPU4、19900–19907，原19400/原native继续；部署默认cache/compile。Sidecar调整8sim（model0/1/7；sim0:2,5,0；sim1:6,2,1；sim7:6,7），原版分片8sim（3,3,3,5,5,5,6,6）以保留显存余量，不操作其他namespace/未知作业。目标截止2026-10-19T01:43UTC，须基于实际吞吐检查，不保证硬件未验证的绝对工期。

官方manifest另含固定instruction/CUDA RNG与任务状态初始化，不能声称与传统native逐位相同，也不能把两者样本拼接；分别报告。结构化JSON汇总按完整100及manifest hash/episode覆盖验证，禁止用上游小样本_result.txt固定100分母。保留native双鞋Clean92/100。目录baseline_official_manifest_deadline_20261018；进度/截止与有效结果自动更新docs/ROBOTWIN_LIVE_RESULT_ANALYSIS_20261018.md，并将新完整结果追加EXPERIMENT_RESULTS_AND_ANALYSIS。专门加速经验文档SIDECAR_INFERENCE_ACCELERATION_20261018.md已建立。


### 2026-10-18T17:07:59.134583+00:00：最终采用FFN局部compile + 联合cache，原门槛通过

本轮8变体在GPU0同两真实输入、每组3次CUDA同步warm，eager0.912392s，eager_cached0.377508s，conservative0.647082s/cached0.275161s，Sidecar读写compile0.941083s/cached0.388895s，FFN局部compile0.939386s/cached0.400740s。前两种编译范围至少一个未缓存参考未过原门槛，均保留失败，不放宽容差；FFN策略的两fixture×未缓存/同缓存四组动作对照max_abs_error全部0.0，原allclose门槛通过。FFN确有4个新Inductor kernels，非仅把enabled写true。

选择ffn：只编译3层Sidecar内部FFN，主干与三次read/refine/write全部保持原路径，未旁路/关闭Sidecar。cache开启0.99/max3，每generate reset，10步实际重算4次/复用6次已实测。组合相对完整eager约2.28倍，速度贡献主要来自cache；本轮FFN compile未观察到额外吞吐收益（相对cache-only略慢），不能称全模型compile已解决。相同缓存条件的编译保真不等于缓存与完整计算等价；缓存相对完整eager的两fixture动作最大差0.03125/0.01171875，须闭环百条判断成功率。

最终配置outputs/robotwin_eval/sidecar20000_officialenv_20261018/deploy_selected.yaml（compile.enabled=true/sidecar_strategy=ffn/cache=true），数值/耗时/全部动作证据fast_validation/result.json。checked模型GPU0直接复用为19600–19602，已预热/reset并启动正式queue0三个百条客户端：adjust_bottle两域、place_shoe/Clean；新官方Conda与runtime/PYTHONNOUSERSITE=1，原RT/预算/成功判据不改。模型GPU1/7新副本正在加载，各预热后启动queue1/queue7，合计8sim/50任务两域10000条，不混短测/旧环境。


### 2026-10-18T17:18:05.575263+00:00：正式新环境双队列启动验收

Sidecar三个模型GPU0/1/7已全部READY并以真实fixture预热/reset，三个pool配置SHA一致d1de0dd89a16eae0a9cb08b685c265100b17b766c77c6d0cf27d35dca70ceb6b。queue0/1/7分别3/3/2个百条作业启动，父进程实查全部存活，共8路覆盖100个唯一task/mode、10000条；queue0已完成真实成功episode，新增各路仍有初始化过程，不能把启动当任务完成。当前任务瓶子两域、单鞋Clean、双鞋两域、扫码Clean、锤子两域。

原版传统官方single_eval仍保留；新增官方manifest并行组8个run_eval子进程已实际创建，双鞋Randomized100专家清单封存，递积木Randomized100亦完成并开始下一专家任务，3个清单builder继续。run.json当前无errors。目标截止2026-10-19T01:43UTC；实际耗时须看分片吞吐，不把队列创建称已完成。每个清单全100，逻辑0–99分成10条且验证无重复；分析仅将协调器确认exit0/manifest/逻辑覆盖的完整100条纳入替换，不以progress=100代替验收。

启动验收证据outputs/robotwin_eval/sidecar20000_officialenv_20261018/launch_acceptance_snapshot.json，包含pid/存活/配置hash/当前baseline任务/显存。四项针对性CPU测试通过130.11s，七个新文件ruff全通过。加速经验与策略限制见SIDECAR_INFERENCE_ACCELERATION_20261018.md；实时结果、共同完整任务对比、替换平均及截止预测自动写ROBOTWIN_LIVE_RESULT_ANALYSIS_20261018.md，新完整成绩自动追加EXPERIMENT_RESULTS_AND_ANALYSIS.md。


### 2026-10-19T01:48:18.684698+00:00：Sidecar同一20k权重新旧环境完整任务对比

用户要求核对新测Sidecar与旧环境已完成任务的重合项。只纳入两个环境均complete、errors为空、completed_episodes=100的正式结果；旧balanced_queue5/7与新queue0/1/7，不计pilot、2条smoke或主动停止的未完成作业。当前旧环境6项完整，新环境12项完整，交集4项（3任务）：

|任务|条件|旧环境|新环境|差值（百分点）|accepted seed交集/100|
|---|---|---:|---:|---:|---:|
|adjust_bottle|Clean|95/100|98/100|+3|98|
|beat_block_hammer|Clean|92/100|99/100|+7|89|
|beat_block_hammer|Randomized|35/100|40/100|+5|89|
|place_container_plate|Clean|99/100|98/100|-1|99|

共同完整4项描述性宏平均80.25%→83.75%，+3.50个百分点（321/400→335/400，增加14次成功），3项提升、1项下降。该子集不代表50任务全量效果，百条单次运行的小幅变化不作统计显著性结论。

两边权重相同，但同时更换仿真栈与推理部署：旧cache/compile关闭，新velocity cache开启并使用Sidecar FFN局部compile；accepted seed序列均不完全一致。不能将差值全部归因于环境，也不能据此单独证明cache不影响成功率。旧place_cans_plasticbox/Clean100及place_container_plate/Randomized70已完整，新环境对应项仍未满100，不使用部分结果计算完整对照平均。

证据：outputs/robotwin_eval/sidecar20000_currentenv_20261018/balanced_queue{5,7}/parallel.json；outputs/robotwin_eval/sidecar20000_officialenv_20261018/queue{0,1,7}/parallel.json。下一步等待其余两个重合项新环境百条完成，再扩展同集合对比；持续新环境全量评测，保留全部旧产物。


### 2026-10-19T01:52:00.585442+00:00：Sidecar与原版baseline旧环境及新环境复测对齐

新环境Sidecar已完整12项、7个不同任务；下表各有效成绩均为100条。旧baseline为历史发布模型官方默认部署完整成绩，非此前cache/compile均关闭的匹配部署控制实验。

|任务|条件|原版旧环境|原版新环境已完成复测|Sidecar新环境|Sidecar减原版旧环境（百分点）|
|---|---|---:|---:|---:|---:|
|adjust_bottle|demo_clean|95%|—（未完成或未复测）|98%|+3|
|adjust_bottle|demo_randomized|38%|—（未完成或未复测）|34%|-4|
|beat_block_hammer|demo_clean|77%|—（未完成或未复测）|99%|+22|
|beat_block_hammer|demo_randomized|42%|—（未完成或未复测）|40%|-2|
|handover_block|demo_clean|86%|85%|80%|-6|
|place_container_plate|demo_clean|99%|—（未完成或未复测）|98%|-1|
|place_dual_shoes|demo_clean|13%|92%|85%|+72|
|place_dual_shoes|demo_randomized|0%|16%|10%|+10|
|place_shoe|demo_clean|50%|97%|94%|+44|
|place_shoe|demo_randomized|19%|49%|49%|+30|
|scan_object|demo_clean|66%|—（未完成或未复测）|83%|+17|
|scan_object|demo_randomized|14%|—（未完成或未复测）|53%|+39|

共同12项描述性平均：原版旧环境49.92%，Sidecar新环境68.58%，差+18.67个百分点；同时改变模型/训练与环境，不能解释为Sidecar独立收益。共同新环境完整5项：原版67.80%，Sidecar63.60%，差-4.20个百分点。两种平均使用不同子集，不可相互拼接或当全50任务综合。

新原版双鞋Clean为传统native 92/100，其余新原版为官方manifest固定指令/RNG；Sidecar为传统native，seed接受集合与训练表示/编译范围存在差异。扫码两域原版新环境未满100，暂留空，后续待完整后更新。证据：historical parallel_official100_20261016/parallel.json；baseline_official_manifest_deadline_20261018/run.json协调器验收results；sidecar20000_officialenv_20261018/queue{0,1,7}/parallel.json。


### 2026-10-19T03:06:42.967271+00:00：按用户授权停止本机20k，切换32k五模型/15路

本机L20新环境20k停止前监控快照：14个完整task/mode，1731/10000条；未完整项均保留，不拼接到32k。三队列和三个模型已停止，旧结果、权重与日志保留；旧结果监控停止避免误报仍在运行。原版manifest九项900条全部完成，释放其空闲GPU4模型；原传统native诊断及其GPU2模型保留。新复测scan_object Clean89/100、Randomized56/100；原模型旧值替换汇总62.45%，并非新环境全量成绩。

32k使用同一新官方仿真、wxyz合同、自有stats、10步/32动作/sync/CFG1/cache .99 max3/FFN局部compile；仅切换权重。已有独立快照SHA应为0f22e7954ac5278e0e2184feea65c698754e4efad6b9fc9e317399e14f898de1，启动前重新核验。输出outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019；计划模型GPU0/1/4/5/7，每模型独立数值检查通过后放行3路，共15路。每task/mode原生100条，50任务无重复分区，保留RT/专家筛seed/预算/成功判据，不改模型/训练。新32k也需实际numerical gate，不沿用20k权重验证结论。启动第一次系统Python缺yaml已保留coordinator.log，切换已有模型venv运行，不安装系统依赖。

另一机器A100独立32k队列也在运行；本次L20建立独立完整50任务结果集，不能将两机样本拼接或择优汇总。两机硬件/扩展及运行时loader差异均须保留。计划15路不代表15路已执行；以launch_state.json与实际closed-loop日志为准。后续核对每模型数值门槛、显存稳定、真实episode终态；若OOM/buffer异常仅降低本次并发并保留失败证据，不动未知作业。


2026-10-19T03:09:31.085990+00:00 L20 32k部署更新：独立权重SHA256重新完整校验通过；五份模型PID3486827/3486828/3486829/3486830/3486831，仍CPU依赖加载。外部显存短时增长后，将GPU5服务的三个仿真改到5,3,3；最终布局模型0→sim0,3,6；模型1→1,3,6；模型4→4,4,6；模型5→5,3,3；模型7→7,7,2。保留原native模型GPU2和客户端GPU3。coordinator重启仅调整调度，复用五份正在加载的模型，不重新加载权重。实时监控docs/ROBOTWIN_SIDECAR32K_L20_LIVE_20261019.md，独立产物outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019。尚未声称GPU gate/闭环通过。


### 2026-10-19T03:18:13.507605+00:00：32k L20五模型验收通过、15路已启动

五模型均strict加载32k权重，并通过两真实RGB fixture的FFN数值门槛：未缓存与各自缓存参考共四比较max_abs_error全0，真实编译kernel证据通过。相同cache设置编译保真不等于缓存与全计算等价。各GPU warm engine.generate：eager约0.863–0.895s，FFN+cache约0.364–0.396s；这是固定输入生成耗时，不是整场评测墙钟速度。

15个standard_eval父进程现场存活，五队列状态running、launch_state=launched、errors为空；尚在客户端初始化，未提前称每路已进入策略动作或有百条成绩。最终仿真布局0→0,3,0；1→1,3,6；4→4,4,6；5→5,3,3；7→7,7,2，GPU6留余量以应对其他namespace短时显存增长。首批任务与PID见outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/launch_acceptance_snapshot.json，完整参数与无重复50任务分区见launch_state.json/task_partitions.json。旧20k评测和监控已停，全部原始文件保留。后续验收实际native动作、首个episode和显存峰值，再按有效百条比较32k/20k。


2026-10-19T03:20:31.865754+00:00 32k L20：15/15 native client.log已创建，15/15已记录CUDA/Curobo prewarm；当前failed=[]。尚未以READY/启动记录充当完整episode或SR，首个闭环终态待后续监控。


2026-10-19T03:53:14.078548+00:00 L20 32k真实闭环进度：15路全部存活，策略已完成223/10000条，尚无完整百条。GPU当前余量约8.4–33.2GiB；未观察到OOM/buffer错误。place_container_plate/Clean的play_once出现target_pose为None的专家筛选异常，按原生协议跳过不可用候选seed，进程继续，非策略失败或作业退出。部分样本不报全量SR，继续原协议等待有效百条，详见独立实时文档。


### 2026-10-19T03:59:29.517622+00:00：用户授权GPU6新增32k模型，15→18路扩容准备

GPU6实测free34045MiB、原有两路sim合计约11GiB；计划新增同一32k快照模型，新增sim分别GPU2/3/5，保留当前所有评测episode与判据。GPU6模型PID3549591加载并通过原数值门槛后才放行。模型启动前free>=33000MiB、每路新sim启动前free>=10000MiB；未知namespace负载不终止。仍需实际峰值监测，free瞬时满足不保证长期容量。

现有parallel_eval队列在内存中，不能靠编辑parallel.json移交。新增outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expand_gpu6.py：GPU6验证通过后暂挂GPU5的队列控制器（仅控制器SIGSTOP），检查其各线程的子PID集合与running记录一致，原3个standard_eval/native子进程持续运行、不重跑。新控制器接管这20个task/mode，6个worker分别复用GPU5和GPU6各3独立端口。已运行3项先等待真实外层exit status和完整run.json双重验收；zombie退出码验证已用真实exit0/exit7子进程通过。17个未开始项与后续待跑项仅由新内存队列分发。源3子进程全部结束后才回收冻结控制器。其他4队列不变，整个本机结果集保持100个唯一task/mode；源snapshot与移交记录保留，监控只读替代队列expanded_queue56，不双算冻结快照。

本次仅新增推理副本与调度扩容，不改模型、训练、数据、专家筛选、RT、预算或成功判据。当前仍15路，18路须以expansion_state.json、实际新native日志与显存验收为准。后续若新模型/容量门槛失败，原15路保持原状；若已接管则以expansion_state.json为权威，避免直接恢复旧控制器导致重复调度。


### 2026-10-19T06:47:59.496219+00:00：GPU6扩容完成，18路真实闭环及首批32k百条成绩

GPU6第六模型通过同一32k数值门槛，FFN对匹配cache参考四比较max_abs_error=[0.0, 0.0, 0.0, 0.0]，真实编译证据通过。04:07:42新控制器接管GPU5源队列20项，原3个native/standard_eval PID保留，新增click_alarmclock/Randomized和grab_roller两域3路；已完成的源click_alarmclock/Clean按真实exit0/完整run.json验收后继续move_pillbottle_pad/Clean。源另两条鞋类作业继续原PID，故旧源控制器仍SIGSTOP（有意暂停控制器，不是评测卡死），待三条源子进程结束后才回收；勿擅自SIGCONT导致旧队列重复派发。其余4个旧队列不变。

现场18个运行父PID全部存在，100个唯一task/mode无重复，全部队列errors为空。GPU余量约6.4–10.5GiB，GPU6约6520MiB；尚未见OOM/buffer作业失败。新snapshot：outputs/robotwin_eval/sidecar32000_officialenv_l20_20261019/expansion_acceptance_snapshot.json，权威队列为gpu0/gpu1/gpu4/gpu7各queue与expanded_queue56；冻结旧gpu5/queue仅保留历史，不纳入汇总。监控已切换这5个逻辑队列。原launch_state.json为初始5模型/15路历史启动记录，扩容配置以expansion_state.json/expansion_acceptance_snapshot.json为准。

当前32k进度1412/10000条，3个完整百条：place_container_plate/Clean97、beat_block_hammer/Clean98、click_alarmclock/Clean98。对应20k为98/99/100，共同3个Clean条件描述性平均99.00%→97.67%（-1.33pp），并非全量或足够广泛的优劣结论。其他在测任务继续，尚无完整Randomized结果，不能据当前子集证明32k全面提升或退步。


### 2026-10-19T06:50:47.582919+00:00：两机32k评测写入路径隔离核查

用户询问A100旧32k和L20新32k是否冲突。结果根分别outputs/robotwin_eval/sidecar32000_gpu56_20261019T021700Z（有效gpu5a/queue_cross）和sidecar32000_officialenv_l20_20261019（有效gpu0/1/4/7 queue+expanded_queue56），所有task runtime/result/log/run.json均落各自根；监控文档分别ROBOTWIN_SIDECAR32K_LIVE_20261019.md和ROBOTWIN_SIDECAR32K_L20_LIVE_20261019.md。最近两边progress utc均推进至06:50，未发现成绩文件覆盖。端口21600虽同号，但各机器localhost独立。两边权重路径不同但同hardlink inode253467911、SHA相同，均只读；不能称两份物理独立权重，不得原地写入该inode。tokenizer/stats/config为各自快照。

仿真runtime/五个cuRobo.so分别在A100输出内私有RoboTwin和L20 .external/robotwin-official-20261018/RoboTwin，逐个实际resolve路径与inode不同；模型Inductor缓存分别inductor_sm80及L20各gpu*/inductor。共享项目源码、Conda/venv和资产主要只读，运行期间不更新共享依赖；共享文件系统I/O可能影响速度，不能据此称对墙钟时间绝无影响。

未完全隔离项：standard_eval.py两种official路径均将TORCH_EXTENSIONS_DIR设置为同一.cache/robotwin/torch_extensions_official，两组run.json已实证。当前使用各自安装的sm80/sm89预编译cuRobo库、日志未见触发JIT构建，暂无覆盖证据；不能宣称所有缓存完全隔离。未来若触发fallback JIT或重编译，应先按机器/架构/Torch版本隔离通用扩展缓存，不能直接清理或覆盖此共享目录。此次只核查并记录，不改正在运行进程或共享依赖。相同task/mode存在跨机重复，是独立重复实验，不能将两次百条拼为一份或择优汇总；应分机报告，可另作重复性对照。


### 2026-10-19T07:14:32.691229+00:00：32k完整评测工期估算

当前L20监控2026-10-19T07:14:14.362299+00:00，1579/10000条、5项完整、running=18、failed=0。与06:47:06的1412条相比，最近约半小时吞吐369.2条/小时；线性剩余22.8小时。考虑任务长度、Randomized频繁打满预算、专家筛选及队列尾部，面向用户的完整50任务双域10000条计划区间为还需24–36小时，非保证截止。预计再1–2小时可有约10个完整任务条件的阶段分析；与20k共同14项的完整覆盖受place_dual_shoes/Randomized（目前约23/100）拖尾，预计还需8–12小时或更长。不能用未完成项或挑选高分先结算完整benchmark。独立A100队列不并入L20吞吐估算。后续按真实完成数修正工期，监控继续。


2026-10-19T07:37:38.139533+00:00 原版预选10项复测完整分析已写docs/ROBOTWIN_BASELINE_RETEST_ANALYSIS_20261019.md：同集合44.30%→70.70%，官方72.10%；旧全量指定项替换59.81%→62.45%，不当新环境全量。传统native双鞋Randomized12与manifest16分列保留。


2026-10-19T08:05:14.396207+00:00 重核官方issue46全部50任务：旧综合59.81 vs69.04；指定10项替换后62.45。其余45任务90条件，26项低至少10pp、12项低至少20pp，不能称已全部可控；逐项见docs/ROBOTWIN_BASELINE_ALL_TASK_COMPARISON_20261019.md。未改当前评测。


2026-10-19T08:11:00.759095+00:00 32k对照官方阶段分析：完整12项，Clean同11项95.73%对官方95.09%；完整Randomized click_alarmclock65对77，低12pp。尚不能给全量综合。记录见docs/ROBOTWIN_SIDECAR32K_VS_OFFICIAL_20261019.md，不改运行进程。


### 2026-10-19T09:03:03.597376+00:00：用户要求仅对比官方；32k渲染失败与恢复排程

当前完整16条件（Clean12、Randomized4）。官方同任务Clean95.33%对Sidecar95.42%，+0.08pp；Randomized55.25%对Sidecar46.00%，-9.25pp。完整Randomized：adjust_bottle37vs51、beat_block_hammer33vs55、click_alarmclock65vs77、grab_roller49vs38。仍非全50任务平均。按用户要求，实时监控已移除20k对照，改用重新抓取issue46官方快照进行同任务比较。

发现click_bell/Randomized在6/8后renderer cannot create buffer退出1；前有OIDN filter not committed报错。原始日志保留，其他当前日志全量rg未见同类OIDN/buffer错误，失败8条排除有效进度与最终SR。现场GPU1余量约1.8GiB、GPU7约0.6GiB，NVML另见GPU0/1/3/7各新增约7.4GiB进程，容器PID映射尚未确认，不终止未知作业；显存压力是需考虑因素，未据字符串直接证明唯一根因。

已启动recovery_click_bell.py/PID3997643：等待expanded_queue56全部worker退出、GPU6三端口释放后，复用该已验收模型的一个独占端口，在GPU2/3/5中选择最大空闲且>=12GiB者，用同协议从seed group0完整重跑100条；一次指定恢复，不拼8+92、不择优。当前status=waiting_for_exclusive_endpoint，不称已开始重跑。原源队列failed标记保持，监控仅在指定恢复实际启动/验收后替代该key，保留失败历史。未增加当前并发或操作未知进程。

09:02有效任务进度2194/10000、16完整、17running、1failed，另8失败部分排除。代码仅变更独立输出目录内的监控与新增恢复调度，未改模型/训练/协议/默认评测入口。下一步继续等待任务百条与资源稳定，任何后续渲染异常均需保留并排除，不将损坏输入当策略能力下降。


2026-10-19T09:15:35.825096+00:00 Randomized低分诊断建议记录于docs/SIDECAR32K_RANDOMIZED_DIAGNOSIS_20261019.md：四任务Clean98.25 vs97.00、Randomized46 vs55.25；优先配对cache on/off与Sidecar写入消融，Randomized含杂物及桌高-3cm..0，ColorJitter未覆盖这些因素。均为假设/待实验，未改训练/进程。
