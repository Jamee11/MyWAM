# 一次Piper微调，连续2epochs


### 2026-11-02T10:12:45.432907+00:00：用户授权ManiSkill完成后接一次Piper微调（连续2epochs）

用户明确先暂停评估，Piper用midtrain最终56892权重初始化，不用ManiSkill微调权重；fresh optimizer/scheduler/step0，一次运行2epochs，不是两次微调。复用ManiSkill已验证的绝对EEF archive子集读取、完整PatchWAM、联合梯度、WAM1e-4/Sidecar5e-5、bs32×4×acc2/global256、常规5%LR warmup/cosine，不重复WAM2K延迟更新。

Piper实际train9566 episodes/35tasks，Clean1988+Random7578，1404781窗口/epoch，补147至1404928；每epoch5488更新，共10976更新/21952 micro。val1172/test1262不训练；保留双臂20D→80D、三视角、源数据next-achieved绝对EEF robotwin_endpose_reference，继续使用原midtrain统计，不改动作或标签。每4000micro=2000更新保存，最近2完整state及final weights；独立输出/remote-home/jinminghao/OpenWAM/outputs/piper_sidecar/midtrain_patchwam_2epochs_20261102T101245Z。

最小改动：ManiSkillArchiveDataset新增可覆写source_group默认原ManiSkill；Piper子类只换group；独立配置/入口/启动脚本复用现有formal CPU preload handover与已有completion_errors/same_process保存验收；formal只补单subset PatchWAM实际开启的日志识别。新监督器提前CPU加载，等ManiSkill四rank20872、final保存标记/完整safetensors头长度、原torchrun身份退出，才写release marker；GPU0–3各>=70000MiB后交给DeepSpeed，不结束当前训练。20旧/数据检查+2新接续模拟通过，真实Clean/DR/Fresh各1条已读通，两臂mask/3相机/归一化一致。真实GPU接续及完整Piper训练仍待验。

评估PGID704658及其模型进程已按用户要求SIGSTOP并记录pause.json，未删除1600清单/权重/回放结果，暂停期间不自动恢复。未来恢复须先恢复模型子进程并等端口ready，再恢复协调器，避免其24h readiness计时把暂停时间当超时；不能直接盲目SIGCONT整个组。

进度：输出目录status.json；训练日志train.log；handover/cpu_ready.json表示CPU权重已加载，handover/gpu_release.json表示已允许GPU接续。spec.json固定前序进程身份和最终保存验收，不会终止旧训练。


2026-11-02T11:47:55.599006+00:00：真实接续验收：ManiSkill四rank20872/final完整文件保存、原进程退出后，10:27:51Z写完成标记，10:27:55Z放行四卡GPU初始化。Piper四rank已至少完成360次参数更新且所有指标有限；cpu_contract明确PatchWAM=true、WAM delay=0、fresh counters=0、初始化midtrain56892。近期约13.18秒/更新；解决本次真实GPU接续/首更新未验事项，不代表全程完成或SR。最终Piper保存和效果仍待验证，ManiSkill评估仍按用户要求暂停。证据 /remote-home/jinminghao/OpenWAM/outputs/piper_sidecar/midtrain_patchwam_2epochs_20261102T101245Z 下predecessor_complete.json、handover/gpu_release.json、四rank metrics。
