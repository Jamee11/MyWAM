# XA action alignment checkpoint21480：接续快速评测

用户2026-11-01授权当前21480补70完成后启动。权重outputs/1101-XA-10epoch/checkpoint_step_21480.safetensors，归档SHA256 f55e7641237f28bb14f004237b2cafaaad8dd7fc55d9c9d43537340075ede1d7，2233张量。action alignment为训练监督；部署保留checkpoint配置和projector，推理generate沿用正常策略，无EMA teacher。不得关闭Sidecar或改动权重来适配。

新run：outputs/robotwin_eval/xa21480_4090_all100_30ep_20261101。100条件×30条；两L20副本×6独立会话，4090按实时显存准入。与原21480推理配置除端口外一致：eager/compile-off、cache-on(.99/max3)、10 denoise、horizon32、无视频、相同原生环境/专家筛seed/RT/预算/成功判据。保留固定fixture重复、跨副本和12session动作验收。XA canonical独立建立，不要求不同权重输出与原21480相等。

launch.py后台每30秒检查full21480_supplement70_4090_20261031的100条件完成及final_complete100_independent_audit.json passed/10000条，之后仅退休该实验自有两模型与394xx隧道，等待两卡足够容量，再hash复制XA、部署/验收、启动仿真。排队期间不占额外GPU、不触碰在途补测。launch_status.json为实际阶段；当前等待不是已启动XA rollout。

部署依赖包含observe_queue；远端launch先等scheduler写running再启report，避免旧终态启动竞态。原始30条件、指令池30、未冻结语言RNG沿用；最终验收原生日志/退出码/实际分母及checkpoint SHA。输出比较显示原21480与full-backward32k相同条件；只完整条件计入SR。

## 2026-11-01 用户改为立即切换与积极并发

用户明确在旧21480补70封存68完整条件后停止旧实验，立即切XA；不再等旧100条件。handoff_record.json明确记录68条件停测，不伪造全100验收。launch_now.py执行当前切换；launch.py旧等待器已退出。

用户要求避免保守并发/显存限制、允许失败重排。XA调度配置为12独立session、每4090最多4路、起始准入6144MiB、低水位256MiB、启动预留6144MiB。容量失败仅短暂退让，120秒后按最新余量恢复配置上限，每条件最多20次infra尝试；partial保留但不计SR。30分钟无进展守卫保留，未改动作、推理或评分。L20模型准入改28000MiB：现有同架构服务实际新增约24GiB，两卡28GiB可试；加载/fixture失败仍阻止正式仿真。XA原权重hash复制通过，未来模型/真实动作/峰值待运行。

## XA实际启动

2026-11-01T18:33:38Z：两份双卡放置模型（主体0/T5 3；主体6/T5 1）已通过固定fixture重复/跨副本精确一致与12端口在线重复chunk验收；同XA hash f55e764...ede1d7。4090已实际启动12个独立仿真，未存视频，处于初始化/专家阶段，没有正式SR。初始分布GPU0/2/3/5各1、GPU1/6各3、GPU4两路；12路启动不是峰值长期稳定证明，若基础设施失败按新策略保留日志并重排。证据deployment_acceptance.json/startup_snapshot.json/launch_split.log/state.json。
