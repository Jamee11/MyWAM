# 三阶段完整监督：原始warmup2K→完整PatchWAM 10epoch

2026-10-31 用户明确批准本地实现并部署，接在当前XA完成之后同一aigc24八卡串行启动。

## 训练定义

- 起点为原始pretrain-origin warmup2K：checkpoint_step_4000.safetensors（acc2=2000更新），不是新C2R warmup，不是XA最终权重。2026-10-31本轮远端重新读取全部24.86GB确认SHA256 `49b6c013eb910b9d0bc3071ec40a38915631d8a54dd6b73b1942d6968383835b`。
- 原完整PatchWAM增强：p .8、Hue .1、AdaIN/FFT保持开启；不减弱为当前XA的中等增强。颜色指代语义风险仍保留，需要分任务看结果。
- 3tap仍为零起算7/15/23，3组Read/Refine/Write和共享EAHeads不变，模型参数/state_dict/推理结构不变。实际自包含Wan主干是30block，3个tap不等于主干只有24层。
- 新增默认关闭的`model.architecture.relational_sidecar.stage_supervision_weights`。本次[.2,.3,.5]，三处都监督Ground/Relation/Plan。每任务跨层系数总预算保持：Ground/Relation乘1+deep_supervision（1.1），Plan1；不重复叠旧deep项。
- 新loss：`.5 * Σ_s a_s * [.025*1.1*Ground_s + .01*1.1*Relation_s + .05*Plan_s]`，a=.2/.3/.5。系数总预算相等不代表实际梯度范数相等。
- `loss_ea_stage_0/1/2`是实际加权贡献，相加等于`loss_ea`；`loss_ea_stage_i_ground/relation/plan`是未加权分项。原`loss_ea_ground/relation/plan`继续表示最终stage；`loss_ea_deep`保留旧浅Ground+中Relation的诊断值，新模式不再将它额外加到loss。
- 8×batch32/acc1/global256；10epoch，预期21480 optimizer updates；fresh optimizer/scheduler/counter，无重复2K冻结/额外WAM ramp；主干/Action LR1e-4、Sidecar5e-5，标准5%warmup+cosine；XA关闭、gate不放大。
- 5000/10000/15000/20000/final保存weights-only，保留5份；无optimizer精确恢复状态，不可称为精确resume。

## 本地文件与验证

- 模型增量：openwam/model/architectures/dual_system/relational_sidecar.py，仅配置验证、loss聚合和额外训练日志。
- 新配置：configs/train_sidecar_allstage_warmup2k_8gpu_10epochs.yaml。
- 新入口：scripts/train_sidecar_allstage_warmup2k_8gpu.sh，复用已运行成功的XA standalone分布式入口但覆盖为本次无XA配置。
- 复用串行控制器scripts/supervise_sidecar_xa.py，新增可选launcher、wait-for-updates、lock-port；旧参数默认值保持，当前XA独立旧快照不变。
- CPU：tests/test_sidecar_allstage.py、test_relational_sidecar.py、test_cross_appearance.py、test_sidecar_xa_supervisor.py，共52项通过（17.60秒）。包含分项系数/直接梯度、严格state_dict/前向兼容、错误权重、完整增强配置、前序未完等待/完毕仅启动一次/重复启动拒绝，以及旧Sidecar和XA回归。
- shell语法、Python编译、指定改动git diff --check通过。远端真实launcher `CHECK_ONLY=1`解析配置通过，断言batch/LR/增强/起点/无XA/保存间隔通过。不另跑10步训练验收；未占用额外H100 GPU。
- 初次标准venv测试导入受文件系统等待影响，改用既有共享内存runtime得到上述结果，并停止本轮多余测试进程；未修改依赖或训练环境。

本地证据：outputs/sidecar/allstage_setup_20261031/{tests.log,resolved_remote.yaml,config_acceptance.json,remote_initial_sha256.txt,source_manifest.json,deployment.json}。

## 部署与串行条件

远端项目根：`/mnt/100T/lixiangyu/cuigm/bitdistill/OpenWAM`。

独立代码：`.experiments/allstage_original_warmup2k_20261031T1842Z`。

输出根：`outputs/sidecar/allstage_original_warmup2k_10epoch_20261031T1842Z_after_xa`。

控制器PID **2951993**，2026-10-31T18:46:27Z启动；控制端口29545，与现XA控制器29544隔离。当前状态实查`waiting_for_predecessor_final_checkpoint`。

前序为XA的`.../xa_original_warmup2k_10epoch_20261030T135522Z_after_c2r/production`：8rank均21480、final文件唯一、日志确认final保存且safetensors头可读后，继续等8卡全部空闲（每卡>75000MiB free、util<5），然后只启动一次正式10epoch。不会从前序权重初始化，不会因前序异常中止就启动。

控制器使用nohup等价的独立session、stdin=/dev/null、持久日志，不依赖本地SSH持续连接。不会杀其他任务或争抢尚未释放的卡；若前序失败或卡被其他任务占用，会保留等待状态，不声称已开始。

监控（登录aigc24后）：

```bash
run=/mnt/100T/lixiangyu/cuigm/bitdistill/OpenWAM/outputs/sidecar/allstage_original_warmup2k_10epoch_20261031T1842Z_after_xa
cat "$run/state.json"
tail -n 40 "$run/controller.log"
# 正式启动后：
tail -n 40 "$run/production/launch.log"
```

2026-10-31T18:46:48Z前序XA仍正常，8rank14577/21480；本次新实验尚未进行GPU更新。因此当前确认的是实现/CPU回归/配置/原始权重SHA和队列等待健康，不是生产首步或最终效果。

## 评价与未决项

沿用先前决策文档：固定输入/noise的Action FM、Full/WriteOff、前8/全32步和窗口尾部误差、Clean/Randomized同协议闭环。辅助指标改善不等于动作收益，未证明浅层Plan直接监督必然有效；不能拿5K候选与旧21480预算不等的差异当因果结论。当前首次正式GPU更新、实际batch32峰值显存/吞吐、全程保存和闭环收益均待自然串行启动后验证。

本地工作区另有ManiSkill任务改动，本次代码归档以git HEAD 5038443为基础，只覆盖上述本次文件及文档，没有带入未提交的其他数据加载器/训练配置改动。部署归档初始SHA256 `63bc68b82877d2bcfc1a11ab3db80f7763935a0eb9bdebb560b8432c62b387a8`，保存在新输出根setup/code.tar.gz；后补实验说明与监控证据独立保留，不覆盖正在运行的XA代码。

## 2026-11-02：训练完成，共享路径复核

通过已恢复的跳板1122读取共享目录，确认控制器complete、8rank均21480且最终指标有限。完成时间2026-11-02T07:20:24.919142Z（北京时间15:20）。最终checkpoint_step_21480.safetensors为24,862,363,126 bytes，完成标记step21480/bytes一致；现存15K/20K/final。rank0最终total .15210399、Action .0002076603、Video .10438462、EA .04751170、grad_norm .02230829；这只是训练终态，不是闭环结果。

当前H100 SSH内网访问未恢复。共享checkpoint为0600、UID1015，跳板用户UID1004，直接读头PermissionError；本轮未验证文件SHA/头内容、未传回该新模型、未改变文件权限。已有生产首步/持续吞吐/最终保存待验项，在日志与marker范围已解决，闭环收益与可读归档仍未解决。连接排查见docs/H100_ACCESS_RECOVERY_20261102.md。
