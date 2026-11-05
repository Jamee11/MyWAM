# 三阶段完整监督最终归档回传（2026-11-02）

用户明确要求回传到 `/remote-home/jinminghao/OpenWAM/outputs/1102-3stageall-10epoch`。不改模型/训练/远端权限，仅复用XA归档脚本，固定本次source与最终权重，使用已验证1122跳板机SSH master直接读取共享/mnt/100T。

源：`/mnt/100T/lixiangyu/cuigm/bitdistill/OpenWAM/outputs/sidecar/allstage_original_warmup2k_10epoch_20261031T1842Z_after_xa/production/runs/2026-11-01_14-25-53`。

严格白名单30个源文件：最终`checkpoint_step_21480.safetensors` + 29个配置/normalization/contract/tokenizer/完整8rank训练日志/launch/控制器与实际源码/配置来源文件。排除中间checkpoint和optimizer/scheduler/EMA/RNG/accel_state大文件。生成本地README、eval模板及transfer状态/校验文件。

09:00 UTC重新从跳板检查8个已知H100节点22端口，仍全部超时。最终权重为24,862,363,126 bytes、mode0600、UID/GID1015；跳板lixiangyu为UID1004/groups1004，读取PermissionError。使用原用户验证sudo权限，被明确拒绝（may not run sudo）；不再尝试提权、不绕过UID限制，不chmod/改属主。仍需恢复原节点访问或由管理员授予跳板UID1004对这一最终文件的读取权限。

独立本地worker PID706938，`outputs/1102-3stageall-10epoch/transfer/fetch_final.py`。先SFTP下载29个可读配套文件（总293,785,815 bytes），随后每30秒检查最终文件可读性；可读后rsync断点续传，完整双端SHA256+实际2229张量+离线tokenizer通过才原子改名并写全量verification.json。若恢复节点SSH而非授予跳板权限，需将worker重指向已验证节点，不能声称当前脚本会自动发现任意新端点。

路径/编译/远端QUERY自检通过；远端29个配套文件SHA256已经取得。已到本地的配置验证为[.2,.3,.5]、global256、XA关闭，normalization成功加载，完整tokenizer离线加载成功；此检查不包括最终ckpt。README明确源`.complete.json`不是本地完整归档的完成证据。

最终是否已回传，必须检查transfer/status.json和verification.json；不能把metadata回传成功说成最终模型已归档。

2026-11-02T09:06Z：29个配套文件（293,785,815 bytes）已全部到本地，逐文件大小和双端SHA256一致；normalization/完整tokenizer离线加载通过，启动参数与三阶段配置存在。证据transfer/metadata_verification.json、metadata_SHA256SUMS、metadata_load_checks.json。worker仍健康，状态waiting_for_checkpoint_permission；最终ckpt未开始下载，目录中源complete标记不能冒充本地权重完成。后续只有管理员授予共享文件读取权限时会按当前跳板路径自动续传，或恢复节点后再调整连接入口。


## 2026-11-05重试与sudo核查

最终21480/24,862,363,126字节文件仍存在，0600、UID1015；跳板1004实际open被拒，所有原计算节点及834组合发现仍未恢复。使用已提供凭据核查sudo授权后明确被拒：lixiangyu may not run sudo on zs。不能靠本地sudo下载，也没有远端sudo读权限；不修改源权限。worker706938仍waiting_for_checkpoint_permission，文件可读后会按既有流程开始断点回传并做双端SHA；当前未下载最终权重，也未验证其内容。管理员仅需处理此最终文件对跳板UID1004的读权限，或恢复原账号计算节点访问。证据outputs/analysis/h100_recheck_20261105，源文件完整路径见本文开头。
