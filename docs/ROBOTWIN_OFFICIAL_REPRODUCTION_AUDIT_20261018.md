# 2026-10-18 OpenWAM / RoboTwin 官方复现审计

本次仅只读核查运行代码、官方仓库与 issue #46，新增审计记录；不改模型、评测逻辑、依赖、seed、预算或当前队列，不操作原 DiT4DiT。

## 结论和成绩口径

当前是官方发布 Clean2Random 权重在本地迁移仿真栈上的评测，不能称为官方验证环境的精确复现。已完成97个(task, mode)百条作业，无失败作业，最后3个Randomized仍运行。完整Clean50任务为80.66%，issue #46同50任务89.36%，差-8.70个百分点。完整Randomized47任务为38.7234%，官方同47任务47.7660%，差-9.0426个百分点。不能将47任务直接与官方全50任务48.72%作最终对比。当前双域均完整的同47任务为Clean80.5957%、Randomized38.7234%、综合59.6596%。

官方issue表50任务逐项求和复核为Clean89.36%、Randomized48.72%，综合69.04%。官方明确指向同名发布checkpoint、deploy默认10步、DiT cache开启、seed group0、100条、原生预算。我们未误用Full模型的93.6%成绩。

| 任务 / 域 | 本地 | 官方issue #46 | 差值（百分点） |
|---|---:|---:|---:|
| place_dual_shoes / Clean | 13 | 89 | -76 |
| place_shoe / Clean | 50 | 98 | -48 |
| scan_object / Randomized | 14 | 60 | -46 |
| place_bread_skillet / Randomized | 25 | 61 | -36 |
| handover_block / Randomized | 17 | 52 | -35 |
| open_microwave / Clean | 53 | 82 | -29 |
| place_container_plate / Randomized | 86 | 76 | +10 |

差异集中且部分很大，不能仅以100条抽样波动解释；但没有受控消融，不能把差距因果归给某个依赖或并行服务。完整97项差值见outputs/robotwin_eval/audit_official_20261018/per_task_comparison.csv。

## 已对齐的路径

- 发布权重checkpoint_step_10740.safetensors，本地run记录SHA256为2cab63b550ff45a52fbde4adc40ee2bdb7d4bafe5fa0dd6879e41e22ca2ff95f，沿用自身config和stats、80D统一表示、20D原始EEF。
- 当前configs/deploy.yaml与本次获取的官方main逐字一致：10步、sync denoise及executor、full chunk32动作、DiT cache阈值.99/max_skips3、compile开启。
- 原生50任务、Clean/Randomized两域、100个专家筛过的有效策略episode、seed group0从100000起、unseen指令、原成功判据与每任务预算、RT32spp/depth8/oidn；planner fallback=0、step override空。策略失败计入分母，专家不通过跳过是上游行为。
- 我们保留官方single_eval.sh → eval_policy_wrapper.py → RoboTwin script/eval_policy.py → 官方客户端/部署engine的主体流程。额外standard_eval.py做启动、来源记录及验收；parallel_eval.py做整任务调度。
- ROBOTWIN_TEST_NUM修复只将main和eval_policy的分母同步；当前N=100与上游100相同，不能解释本轮差距。
- 全100个run.json中的wrapper、adapter、deploy、policy及step_limits哈希分别一致；当前wrapper/adapter仍与这些启动记录一致，未观察到队列中途这些入口漂移。未由此宣称所有依赖和live进程模块都已完整封存。

## 提交不同，但不能凭提交号认定任务源码不同

OpenWAM验证RoboTwin为0aeea2d669c0f8516f4d5785f0aa33ba812c14b4，本地248266bee01e4bfa9be38be9055a9cd5b4281821。直接获取验证提交的Git tree，对envs/、task_config/、description/、script/中现存非pycache blob按Git blob SHA1检查：719文件相同，只有envs/_base_task.py、script/test_render.py、script/_install.sh不同，无缺失文件。评测脚本、全部任务/判据、相机、robot/planner、任务配置/预算、指令生成在此核查范围相同。

特别注意GitHub compare A...B返回共同祖先至B的差异，两个分支diverged时不是A与B端点逐字差异；本审计以验证提交tree及raw文件直接比对为依据，不能用compare列表归因。

前两个本地文件的差异为原DiT4DiT历史扩展：RT可关闭开关、可选视频捕获、仅qpos分支可选TOPP限幅。本run清理继承的ROBOTWIN_*并强制SAPIEN_DISABLE_RAY_TRACING=0，使用ee，故这些新增分支关闭/不适用。安装脚本的实质差异是验证版固定cuRobo v0.7.8，而本地旧安装未固定该tag。Git tree没有覆盖外部下载的模型/材质/机器人资产内容，不能据源码一致宣称资产同版。

## 最重要的未对齐项

| 仿真依赖 | OpenWAM官方验证 | 本轮本地 |
|---|---|---|
| SAPIEN | 3.0.0b1 | 3.0.3 |
| cuRobo | v0.7.8 | v0.7.7-9-gd64c4b0，HEAD d64c4b005459db10c5dd867d8b30a87d5bda9bdb |
| warp-lang | 1.13.0 | 1.12.1 |
| Torch（仿真端） | 2.4.1 | 2.7.1+cu128 |
| mplib | 0.2.1 | 0.2.1 |

本地cuRobo五个扩展为当前L20/sm89/GLIBC2.31重编译；编译能运行不意味着与官方优化/轨迹分布等价。Base_Task.take_action的ee分支对左右目标调用left/right_plan_path，实际进入cuRobo MotionGen：依赖差异既可改变专家接受seed集合，也可改变策略实际运动轨迹。SAPIEN差异可影响渲染、接触和运动，尤其是鞋子等任务的明显差距值得用视频和初始观测定位；这些均为待验证机制，不是已证实根因。

资产仅核验独立复制、文件大小/inode、部分配置和URDF来源，没有与官方发布评测所用资产revision/hash全量比对；这同样是复现缺口。

## 我们新增并行层的局限

官方传统入口是独立policy server和simulator；当前官方还提供Labtasker：先固定accepted seeds/instructions manifest，再以小批次分发，worker拥有自己的服务端点。本轮采用自建动态整task队列，16个独立PolicyServer/WAMPolicy/action deque共享一个engine并互斥predict，另1路独立原服务。并行本身不改变原生成功判据/预算，且每次generate重置DiT cache，CPU测试已验证deque/reset隔离及互斥；但尚无同一真实输入/seed在独立服务器与共享池的逐动作等价对照。因此不能无条件排除共享模型可变状态、线程/compile交互等问题，也不能把CPU测试当真实等价证明。

本次进一步查明engine.generate默认传seed=42；BaseWAMArchitecture.generate用新建torch.Generator.manual_seed(seed)生成动作噪声，视频初始化亦显式用seed。早期记录中将并行请求交织直接解释为全局模型噪声游标变化过于宽泛，应以此具体实现为准。仿真端/规划器RNG、硬件数值和共享状态仍须分别验证，不能由固定模型噪声推出闭环比特复现。

额外工程不足：没有跨checkpoint冻结的完整episode manifest，整任务单位导致尾部空闲且中断不能精确恢复；属于公平比较/效率/复现的不足，不是已证实的低分原因。当前所有任务原生专家筛seed与100条协议仍保留。

本轮官方main快照openwam=1e103e1c61905c230dab6b0f99980370cf2f9e3c、robotwin=ea8b21121ebb3cd201ff5b3fe361944ac94eda3f。当前RoboTwin main已使用scripts/与XPolicyLab，不可直接替换正在运行的script/适配链并称为同协议复现；复现该OpenWAM成绩应先以其验证版本为准。OpenWAM相关模型/部署/工具/benchmark文件比对165文件相同，6处本地不同：wrapper的分母修复、adapter及engine/policy的默认关闭EA分支、architecture注册和base训练forward_fn；发布配置走原joint_self_attn，未启用EA/Sidecar。以上是静态路径审计，不代替实际输入输出对照。

## 下一步验证顺序（尚未执行）

1. 保留当前全部结果和录像，让最后3项按原协议完成；最终导出50×2逐项差值，停止使用部分集合均值作最终排名。
2. 在OpenWAM内独立目录准备官方验证RoboTwin提交及依赖栈，不覆盖现有run或DiT4DiT；优先核对cuRobo v0.7.8、SAPIEN3.0.0b1和资产来源。
3. 用place_dual_shoes、place_shoe、scan_object和一个接近官方的click_bell作对照：先比较同一输入的独立官方server与当前共享pool动作；再固定seed/指令、比初始三视图/EEF、专家接受列表、轨迹及失败录像。
4. 每次只改变一层（server并行、仿真依赖、资产），记录两域结果和接受seed变化，不调cache/预算或换较容易seed去逼近官方数字。涉及依赖/核心逻辑改变前按AGENTS单独确认。

## 官方来源与证据

- https://github.com/OpenWAM-Official/OpenWAM/issues/46 （包含50任务逐项结果和明确协议）
- https://github.com/OpenWAM-Official/OpenWAM/blob/main/benchmarks/robotwin/README.md
- https://github.com/OpenWAM-Official/OpenWAM/blob/main/benchmarks/robotwin/LABTASKER.md
- https://github.com/OpenWAM-Official/OpenWAM/blob/main/configs/deploy.yaml
- https://github.com/RoboTwin-Platform/RoboTwin/blob/0aeea2d669c0f8516f4d5785f0aa33ba812c14b4/script/eval_policy.py
- https://github.com/RoboTwin-Platform/RoboTwin/blob/0aeea2d669c0f8516f4d5785f0aa33ba812c14b4/script/_install.sh
- https://github.com/RoboTwin-Platform/RoboTwin （当前main用于识别入口迁移，不用来替代历史版本）

官方API响应、raw源码、逐项差值CSV存于outputs/robotwin_eval/audit_official_20261018/。大权重本次未重新下载，没有访问HF官方下载端点。


### 2026-10-18T06:43:58.768972+00:00 cuRobo源码版本表述修正

重新clone官方v0.7.8后发现其tag指向d64c4b005459db10c5dd867d8b30a87d5bda9bdb，旧cuRobo HEAD同为此提交且两者tracked diff均空。旧describe=v0.7.7-9-gd64c4b0是旧checkout缺少v0.7.8 tag造成的版本描述，不代表算法源码不同；不能将前文版本字符串差异当作cuRobo源码升级/已定位的根因。新环境已实际编译安装nvidia_curobo0.7.8（Torch2.4.1），旧扩展绑定Torch2.7.1；SAPIEN/Torch/Warp及二进制运行栈仍有差异。证据outputs/robotwin_eval/rebuild_official_20261018/curobo_source_equivalence.json及git ls-remote tag。当前正式Sidecar队列为balanced_queue7/balanced_queue5共10sim，旧12sim pilot因实测显存余量仅3–4GiB而停止并保留，不混入正式SR。


### 2026-10-18T06:47:38.838199+00:00：真实独立服务/共享池动作对照通过

fresh官方OpenWAM服务19400（GPU2）对原共享pool19000/19001（GPU0），发布同一权重、官方默认cache/compile/10步，使用两个真实HDF5首观测的三视图/EEF/固定指令，各32动作；池两端口交错请求，再reset并交换观测。四组序列均bitwise_equal=true、max_abs_error=0.0。只证明此真实fixture范围，不外推所有输入/长期共享状态或Sidecar模型；本次已明显降低“共享池本身改变原模型action”的嫌疑。证据outputs/robotwin_eval/rebuild_official_20261018/action_parity/result.json，原始请求inputs.json保留。Sidecar原20k与独立副本SHA256均0d67d651577ef53d5a86b0e1e03dd93f6cc7bf652c3c7af08734b7712042ea6e，size24862363126，copy_verification.match=true。新物体包9368文件与旧资产全同；背景纹理还在逐文件核验，官方native诊断仍等新环境完整编译验收。
