# ManiSkill archive：八任务 Clean / HardRandom 评估


### 2026-11-02T08:53:05.303482+00:00：archive ManiSkill八任务held-out评估适配（用户授权）

用户要求当前微调最新checkpoint，每任务100 Clean/100 Random，共1600；明确允许Clean从未参与训练的val+test取样。核查midtrain launch/saved contract与finetune source_snapshot/preflight manifest SHA完全一致，均为a96d7f13...4a07aa；训练入口显式split=train，历史/微调loader都过滤split。全部16000 ManiSkill train与4000 val/test的ID/family/numeric/media/physical seed无交集；归一化来源train_only_present_arm_state_and_command，prepare脚本仅split=train进入统计。证据outputs/maniskill_eval/archive_suite_20261102/training_exclusion_audit.json和四份读取代码。

新archive_protocol选Clean test优先62/63+val38/37各100，HardRandom仅test每任务D25/M75；保存1600原生env ID/domain/hard_index/reset_kwargs/seed/600步预算，不复用旧六任务ColorDistractors协议。新archive_eval复用原native_action/robot_state/相机与WebSocket，增加原生delta vs绝对EEF专家回放及独立rollout；新run_archive_eval复用已有共享引擎隔离端口，保留失败日志和单次分片重试、严格覆盖后才汇总SR。

必要部署适配：当前20D原生双臂统一契约只出现一只臂；派生部署副本仅取原stats前10维、scatter/gather槽0:10，让另70维保持inactive，与训练mask一致；80D模型结构/权重不改，原config/stats副本保留。serve_sidecar仅增加archive类型与当前原生fixture支持，旧六任务默认分支保留，必须通过同cache/eager与FFN验证才开放端口。未改训练/数据/标签。

新archive原生无RenderSystem模式在URDF RenderMaterial构建处发生segfault，两次日志保留，不算策略失败；已改为即便纯物理回放也初始化SAPIEN渲染上下文，使用已有gpucomp预加载与单卡可见性，待实测。GPU模型容量/新八任务完整expert gate/1600闭环仍未验，不能称评测已完成或有SR。Clean为held-out val+test不冒称纯test，HardRandom含物理clutter和相机/外观扰动，跨域并非逐scene配对。

## 指标口径

Clean为val+test（test优先），每任务100，共800；HardRandom为test，D25/M75，每任务100，共800。只在1600个固定ID/seed均完成且没有重复后报告总体和每任务SR。val从此用于模型评估，应披露该用途，后续严格独立结论可另留test-only子表。专家回放不是policy SR；保留原生600控制步预算，10去噪、32动作chunk/replan8、CFG1、cache .99/max3、Sidecar局部FFN编译。仅RGB、文本、实测TCP进入模型，不输入object/EA oracle。与旧六任务97.83%不是相同协议，不直接比较。

## 实现入口

- `python -m benchmarks.maniskill.archive_protocol --manifest datasets/multisim_sidecar_archive_v2_20261025/manifest.ready.json --output <new_plan.json>`
- `benchmarks/maniskill/archive_eval.py`：prepare/expert/rollout
- `benchmarks/maniskill/run_archive_eval.py`：checkpoint固定、完整专家gate、模型门槛、4隔离会话和分片汇总。默认GPU0模型、GPU1仿真；模型需要33GiB余量，等待不终止训练/未知作业。

运行示例：
```bash
/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv/bin/python -u benchmarks/maniskill/run_archive_eval.py \
  --plan outputs/maniskill_eval/archive_suite_20261102/plan.json \
  --checkpoint <当前固定checkpoint_step_20000.safetensors完整路径> \
  --output outputs/maniskill_eval/archive_step20000_NEW --model-gpu 0 --sim-gpu 1
```


2026-11-02T08:56:29.205605+00:00: 17项回归通过，含部署原生10D→模型80D归一化往返及缺失手臂inactive mask。新增显式front+hand/None multiview配置，避免原multisim训练手工拼接但部署默认单视角导致的错配。render_backend=sapien_cuda（即便obs_mode=none）解决先前URDF材质无RenderSystem崩溃；Clean/hard D/hard M首三场景原delta与绝对EEF回放均成功，最大TCP差2.21–4.47e-6m。仅是控制桥接，不是模型SR。

固定当前最新checkpoint_step_20000（10000参数更新），完整safetensors头2229张量及数据长度一致。后台协调器PID 704658，输出`/remote-home/jinminghao/OpenWAM/outputs/maniskill_eval/archive_step20000_100clean100hard_20261102T085552Z`，先执行八任务三域共24场景控制回放，再加载完整Sidecar并等GPU0至少33GiB空闲；当前训练/未知GPU作业不停止。模型数值/FFN门槛和1600policy rollout未验；GPU1最多四路，实际首批显存需监测。


2026-11-02T08:59:48.390327+00:00：八任务×Clean/hard D/hard M共24场景原delta与转换绝对EEF回放全部成功，max TCP差0.012531201355159283m；expert.json解决本轮24-scene控制桥接未验事项，不等于policy SR。后台已进入模型加载/容量等待。1600条policy闭环和数值/FFN验证仍待完成；原训练不停止。


迁机操作与统一启动命令见[完整手册](MANISKILL_ARCHIVE_EVAL_PORTABLE_GUIDE_20261102.md)，入口`scripts/eval_maniskill_archive.sh`；支持同GPU或模型/仿真分GPU。原暂停评估仍不自动恢复。
