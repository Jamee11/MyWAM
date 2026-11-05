# Piper21952完整100条评测接续

用户2026-11-04授权：PatchWAM42960完整补测完成后，重新部署Piper，在4090尽量并行执行完整评测。

任务来自归档manifest的robotwin_piper/train，实际35任务（不是37），Clean/Randomized共70条件，每条件100条，共7000条。完整任务英文清单见ROBOTWIN_PIPER21952_EVAL_FRAMEWORK_20261104.md与本run handover_plan.json；此前两条小试不混入本次统计。

Checkpoint：outputs/piper_sidecar/midtrain_patchwam_2epochs_20261102T101245Z/runs/2026-11-02_10-18-00/checkpoint_step_21952.safetensors。部署复用已核验的仅含此权重的硬链接目录和配套配置/统计；三视图显式打开，完整Sidecar、Eager、cache .99/max3、10denoise、sync32、无视频，官方专家/RT/成功判据/动作预算保留。

## 已挂接的自动流程

入口scripts/queue_piper_after_patch100.py；run为outputs/robotwin_eval/piper21952_full100_after_patch_20261104，watcher_process.json记录真实PID。

1. 等Patch补70 state=complete且final_independent_audit.json passed，再退休该实验两份旧模型。当前Aloha评估不提前停止。
2. L20部署两份Piper：主GPU3/T5 GPU4及主GPU2/T5 GPU6，各16独立会话；两真实输入与已通过小试动作逐元素验证后开放。
3. 专用43500–43515/43520–43535端口反向SSH到4090。4090目标24路（八卡各最多三路），按实时空闲显存逐步启动；每路8192MiB准入、启动预留180秒。实际并发取决于余量，不在运行中因余量低而杀任务。
4. 全70条件动态worker队列，每条件原生100条。故障时保留错误前完整episode并只补缺失，指令池仍100；保存每episode指令/Python RNG和来源证据。正常模型失败仍计失败，不择优重试。

原Full21480现有服务与补测保留；原来“Patch结束后追加第二份Full”的等待项已取消，释放的资源用于用户最新指定的Piper。没有停止Full当前仿真或模型。

本次只增量扩展Piper队列的可选显存准入/episode恢复参数；默认Piper入口行为与Aloha默认入口保持。现有70条恢复默认仍70，并增加真实官方函数98+2/连续100一致性测试。

## 状态与结果

status.json区分等待Patch、模型加载、实际评测运行或handover_error；watcher.log保留异常。model_processes.json/tunnel_process.json/remote_process.json保存启动身份。eval/state.json和eval/summary.json为最终任务进度与成功率，逐attempt日志与episode_journal.jsonl在eval/jobs。等待状态不等于已经启动7000条。

仍保留Piper历史采集mplib_RRT与当前official curobo差异；此前只验证click_bell双域小试，不能提前保证全35任务成功率。首次正式100条起即保存RNG，不复用旧小试的legacy恢复记录。


2026-11-04T12:35:40.446865+00:00：两组Aloha各100条件/10000条已完整完成并独立审计通过：Patch42960 Clean88.02/Randomized64.02/综合76.02%，Original21480 Clean88.78/Randomized61.86/综合75.32%。相对各自30条速测为-0.18pp与+0.4867pp；Patch相对Full综合+0.70pp。Piper于10:40 UTC自动接续，两模型/23路仿真实际活跃；当前7/70条件700条完整验收，663成功=94.7143%，仅早期子集不可当35任务最终成功率。在跑批次另约1189条未最终验收。记录见piper21952_full100_after_patch_20261104/progress_snapshot_latest.json。


2026-11-05 07:40 UTC收尾估计：67/70条件保持；stack_bowls_three Clean10/100、Randomized62/100。相对02:49 UTC Clean6/100和Randomized42/100，约4.85小时新增4和20条，即约0.83/4.13条每小时。线性外推剩余Clean约109小时、Randomized约9.2小时，仅为专家seed筛选/重试持续同速假设，不能承诺完成时间。put_bottles_dustbin Randomized20次均初始化seed100000的066_vinegar不稳定，终态failed，不会自行完成。瓶颈是官方专家可解场景筛选及初始化，不是正常7000条主队列吞吐。应定位Piper三碗专家可达性及初始化异常后再定标准协议内恢复方案，不能删除任务或将失败初始化当策略0分。


## 用户要求结束（2026-11-05 07:48 UTC）

停止Piper全量评估的剩余条件，不再等待或自动重试；准确说是两个任务的三个条件。完成结果保留：67/70条件、6700条、5835成功，综合87.0896%；Clean2971/3400=87.3824%，Randomized2864/3300=86.7879%。这是已完成条件平均，不是35任务完整平均；缺失条件不补零，不混入未完成批次。

|未完成条件|停止时模型rollout|成功数|原因|
|---|---:|---:|---|
|stack_bowls_three / Clean|10/100|8|专家抓取规划反复target_pose=None，候选seed接受极慢|
|stack_bowls_three / Randomized|63/100|45|同类专家规划问题，筛seed缓慢|
|put_bottles_dustbin / Randomized|0/100|不适用|seed100000初始化066_vinegar不稳定，20次重试均失败|

未完成73条的结果仅作为部分诊断保留，不并入6700条平均。已终止远端调度器2381728、两路仿真15193/4009107，远端2秒后复核无存活；本机两份Piper模型4165943/4165944也已退出。只操作本实验已核对身份的进程，Sidecar-Off模型/评估不改。所有权重、完整成绩、失败日志、episode恢复记录及36条L20录像诊断均保留。

状态eval/state.json、eval/summary.json及status.json已明确stopped_by_user；终止前快照user_stop_20261105T074759Z/state_before_stop.json，原因与部分结果unfinished_conditions.json。后续不继续补测；如重新研究Piper三碗任务须另立诊断，不宣称当前实验完整7000条完成。
