# H100实验与checkpoint清单（2026-11-02现场统计）

范围：/mnt/100T/lixiangyu/cuigm/bitdistill/OpenWAM 的 outputs 与 assets，按checkpoint_step_*.safetensors统计。跳板机只读列目录、文件stat、已保存配置和训练日志；未读取受UID限制的权重内容，不重新做全量SHA校验。未统计optimizer/state大文件、压缩包内文件、软链接目录副本。

正式完成4组训练实验，共16份训练checkpoint，397,804,111,840 bytes（约397.8GB）。C2R的预热与联合属于同组的两个阶段；若按执行阶段算共5个阶段。另有2份从本地传入的初始化模型，49,676,130,590 bytes；包括这2份共有18份checkpoint文件。

|组|训练定义|保留的optimizer checkpoint步数|份数|
|---|---|---|---:|
|原始warmup2K→完整PatchWAM 10epoch|fresh optimizer，无重复冻结；8×32acc1/global256，主LR1e-4，Sidecar5e-5|10000、20000、21480|3|
|官方C2R→新Sidecar warmup2K→低LR联合5epoch|阶段1主干参与反传不更新；阶段2新optimizer，主LR2.5e-5，Sidecar1.25e-5|warmup2000一份；联合2000、4000、6000、8000、10000、10740六份|7|
|原始warmup2K→XA 10epoch|中等增强、Action8→EMA20对齐；8×16acc2/global256|15000、20000、21480|3|
|原始warmup2K→三阶段完整监督10epoch|完整PatchWAM，Ground/Relation/Plan每stage全监督，.2/.3/.5；XA关闭，8×32acc1/global256|15000、20000、21480|3|

初始化模型：assets/openwam_ckpt/openwam_alpha/Sidecar-PatchWAM-Warmup2K/checkpoint_step_4000.safetensors（原acc2对应2000更新）；assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Clean2Random/checkpoint_step_10740.safetensors（官方已有模型）。这两份不是本次在H100新完成的实验。

实验根目录：

- outputs/sidecar/patchwam_warmstart8_aigc24_20261029T013934Z
- outputs/sidecar/c2r_two_stage8_20261030T022635Z
- outputs/sidecar/xa_original_warmup2k_10epoch_20261030T135522Z_after_c2r
- outputs/sidecar/allstage_original_warmup2k_10epoch_20261031T1842Z_after_xa

其余patchwam早期启动目录和XA launch2/loopback/重分配队列目录，没有独立正式checkpoint，不多算实验；XA旧排队路径指向正式run的别名不重复计数。旧运行未写.complete.json标记不自动意味着旧checkpoint不完整，完成状态结合日志/既有归档确认。

逐文件原始清单及当时读取权限：outputs/analysis/h100_checkpoint_inventory_20261102.json。该文档只盘点H100共享项目，不是本地OpenWAM全部历史实验的统计，也没有清理或删除文件。

## 配套文件逐字节一致性（本轮哈希复核）

原10epoch、C2R联合5epoch、XA10epoch、三阶段完整监督10epoch、原始warmup2K，以及服务器C2R新warmup2K的normalization_stats.npy、relational_sidecar_contract.json、tokenizer/google/umt5-xxl下4个文件SHA256逐一完全一致。normalization SHA为5bc16e47680eda1f3279204fe44537f037bba3407509a8fb2821e1d63cf320f8；contract SHA为6e6be5c234a110380a88a76f9cea9275eec7ab285fd7406a03730e7d7b19628d。各自config.yaml哈希不同，必须保留对应模型自身config，不能以XA或其他实验配置直接替代。

用户要手动回传的两个自包含运行目录：

- 新C2R+完整PatchWAM纯预热2K：outputs/sidecar/c2r_two_stage8_20261030T022635Z/stage1/runs/2026-10-30_11-52-00，取checkpoint_step_2000.safetensors。
- 三阶段完整监督10epoch：outputs/sidecar/allstage_original_warmup2k_10epoch_20261031T1842Z_after_xa/production/runs/2026-11-01_14-25-53，取checkpoint_step_21480.safetensors；排除同目录15000和20000大权重。

两目录均含config.yaml、normalization_stats.npy、relational_sidecar_contract.json、完整tokenizer和8rank日志，没有optimizer/accel_state目录。两份目标权重各24,862,363,126 bytes，都是0600/UID1015，跳板UID1004目前均不可读。config和权重足以作为新训练的finetune起点，optimizer状态仅在精确resume时需要；后续联合训练应使用自己的训练配置，不重复执行该预热config中的2000冻结更新。
