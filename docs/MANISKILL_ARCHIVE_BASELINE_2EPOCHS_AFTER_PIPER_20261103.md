# Piper后接原版WAM ManiSkill baseline 2epochs


### 2026-11-03T09:58:02.216498+00:00：Piper完成后串接原版OpenWAM→ManiSkill baseline 2epochs（用户授权）

用户明确baseline直接微调、每卡bs64/acc1、普通ColorJitter且不启用PatchWAM、同midtrain ManiSkill train。沿用历史baseline官方Foundation154000初始化，不使用Sidecar midtrain/Piper权重；模型joint_self_attn，真正不创建Sidecar/EA模块。Action+Video FM原权重1，LR1e-4、cosine+5%LR预热，全局256，2epochs共20872参数更新/20872micro，常规2000更新保存完整state、保留最近2与最终权重。数据仍16000train/2671589窗口，补27至2671616/epoch，val/test不训练，归一化/80D左臂mask/绝对controller target/图像布局保持。

PreparedSimDataset新增默认True的auxiliary开关，baseline=False不从npz解码targets/不输出EA字段，旧Sidecar/Piper默认True不变；ManiSkill子集from_config透传。普通ColorJitter brightness/contrast/saturation .2/hue0，PatchWAM显式false。新独立config/entry/四卡shell复用原版模型与已有formal CPU handover；supervise_piper_after_maniskill只新增spec可选launcher/输出变量/label/launch_args/延迟预加载阈值，旧默认与正在运行的Piper进程不改。

独立输出 /remote-home/jinminghao/OpenWAM/outputs/maniskill_baseline/archive_cj_bs64_2epochs_20261103T095701Z，冻结完整解析config，队列在Piper达到10896/10976更新时才提前CPU加载，等四rank21952/final保存标记与safetensors完整长度、原进程退出、GPU0–3各>=70000MiB才放行。按用户要求不自动恢复暂停评估，不结束任何旧训练。23旧/数据/交接回归+2新baseline配置/延迟预加载检查通过；真实base/hard两窗口80D/mask/无EA字段通过，Foundation2089张量头无Sidecar且长度完整，冻结启动解析通过。bs64虽旧六任务baseline验证过，新archive数据实际GPU峰值/首更新与长期吞吐仍待正式接续，不能称已通过。

解释边界：此baseline Foundation直训2epoch+ColorJitter vs 当前Sidecar midtrain后1epoch+PatchWAM有初始化、预算、增强多因素差异，不能单独归因为Sidecar结构；如果需要结构归因，后续另做matched protocol。保留已有near/EEF、真机坐标、held-out评估与渲染完整性未决，不改数据来规避问题。

## 入口与状态

- 配置：configs/train_maniskill_archive_baseline_2epochs.yaml
- 启动：scripts/train_maniskill_archive_baseline_4gpu.sh（由监督器设置handover目录）
- 队列：scripts/supervise_piper_after_maniskill.py --spec /remote-home/jinminghao/OpenWAM/outputs/maniskill_baseline/archive_cj_bs64_2epochs_20261103T095701Z/spec.json
- 指针：outputs/maniskill_baseline/current_archive_2epochs.json
- status.json在waiting_to_preload时仅监督等待，不启动模型或占用GPU；临近结束进入cpu_preloading_waiting_for_piper，完成交接后maniskill_baseline_training。
- 训练日志train.log，四rank metrics在runs/<timestamp>/；实际训练后再核查首更新/显存/速度。
