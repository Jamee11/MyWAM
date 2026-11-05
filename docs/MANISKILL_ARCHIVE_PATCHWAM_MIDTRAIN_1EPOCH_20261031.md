# Midtrain → ManiSkill archive PatchWAM：1 epoch


### 2026-10-31T18:40:41.937858+00:00：midtrain最终权重→ManiSkill archive完整train split，PatchWAM 1epoch

用户明确授权实现并在本地四卡启动。复用既有PreparedSimDataset与ManiSkill baseline日志/标准Sidecar trainer；新增独立ManiSkillArchiveDataset，仅选maniskill_franka/train，不使用多源混采epoch。16000 episodes/2671589 windows，末批补27窗口至2671616；4×32×acc2/global256，1epoch=10436 optimizer updates/20872 microsteps。载入midtrain checkpoint_step_56892，fresh optimizer/scheduler/counters，不重复WAM延迟更新；正常Action/Video/EA梯度，WAM1e-4、Sidecar5e-5、标准5%LR warmup/cosine，不叠加XA/梯度退火/Write放大。复用完整PatchWAM，禁旧ColorJitter，真实front+hand增强、缺失视角保黑；绝对controller-target EEF、80D与未出现手臂mask、原统计保持。

改动：openwam/dataloader/multisim_sidecar.py仅新增默认关闭的PatchWAM可选支持；新增maniskill_archive.py、独立config/入口/四卡脚本及现有测试中的子集/末批/动作与黑区域回归。原六任务ManiSkill/多源midtrain/Aloha流程保留。每4000micro（2000更新）存完整state，保留最近2份与最终weights。无需zero-warmup screening hook，sidecar_lr_screen=null。真实首批GPU前后向与吞吐、最终保存、闭环收益待启动实测；不把CPU检查当SR。

启动：`PYTHON_BIN=/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv/bin/python CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_maniskill_archive_patchwam_4gpu.sh`。独立输出由MANISKILL_ARCHIVE_OUTPUT指定，current指针记录实际启动目录。

后续：首批有限loss和显存检查后按稳定吞吐修正38–46小时时间估计；最终模型用保留test split做ManiSkill闭环，基础/hard分开统计。训练集仅在当前manifest已筛选的train记录内遍历，不将val/test加回。

## 启动与检查证据

输出：`/remote-home/jinminghao/OpenWAM/outputs/maniskill_sidecar/midtrain_patchwam_1epoch_20261031T184131Z`，torchrun PID 561076，GPU0–3。12项数据测试通过；真实base/hard样本读取通过，全部16000条numeric/media存在，初始化checkpoint与dataset归一化文件逐字节一致。完整证据preflight.json；源文件快照保存在source_snapshot。当前仅处于启动加载阶段，未宣称GPU更新完成。

2026-10-31T18:48:37.706637+00:00：四rank均已完成至少1次参数更新，Action/Video/EA loss与grad_norm有限，未报OOM。actual run：`/remote-home/jinminghao/OpenWAM/outputs/maniskill_sidecar/midtrain_patchwam_1epoch_20261031T184131Z/runs/2026-10-31_18-46-16`。首步含加载/预热，不据此报告稳态吞吐或成功率；完整证据startup_acceptance.json。

早期第5–8 microstep约6.49–7.36秒/micro，acc2约13–15秒/更新，支持原38–46小时数量级估计；仅少量步，尚非长期吞吐。已验证至少5次更新，首更新后峰值reserved约62.25GiB。
