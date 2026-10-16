# 本机 EA-RSIG-Lite C：切换到正确的 wrist020 Aloha Clean

2026-10-15 用户要求只准备和验证，由用户亲自结束旧训练并启动新训练。本次没有执行停止旧训练或启动新训练。

**当前环境阻塞（2026-10-15 06:46 UTC）**：nvidia-smi报NVML Unknown Error；新进程打开/dev/nvidiactl、/dev/nvidia0、/dev/nvidia-uvm返回EPERM，cuInit返回100/CUDA_ERROR_NO_DEVICE；旧GPU训练仍推进至893步。当前容器不具备启动新GPU作业的访问权限。先由管理员核对宿主机与容器GPU设备/cgroup权限，或准备权限正常的新环境；不要在此状态下先结束旧run，不执行GPU reset/重载驱动或重启容器来影响其他作业。该阻塞不影响已完成的数据和CPU检查，也不证明GPU硬件损坏。

## 数据与检查

旧训练使用有腕部近裁剪问题的数据：

```text
/remote-home/jinminghao/datasets/robotwin2_ea_rsig_v3_clean50_v6_raw_20260921
```

正确的新数据根目录：

```text
/remote-home/jinminghao/DiT4DiT/datasets/robotwin2_ea_rsig_v3_wrist020_20k_20261001
```

本次实际读取的 Aloha Clean 原始 HDF5 路径：

```text
raw/aloha/EA_RSIG_v3/Clean/<task>/demo_clean/data/episode*.hdf5
```

使用raw HDF5内的绝对EEF和oracle重新生成OpenWAM 8/16/32 Lite标签；没有把旧DiT4DiT转换后的14D/delta动作当作OpenWAM动作，没有复用旧bbox标签。共50任务2500轨迹、557651源帧、555151窗口。新产物独立保存在：

```text
/remote-home/jinminghao/OpenWAM/outputs/ea_lite/wrist020_clean50_20261015/data/
  manifest.json
  manifest.ready.json
  normalization_stats.npy
  labels/<task>/episode*.npz
```

完整数据构造核对全部轨迹的相机/EEF长度、有限值、wxyz转换、130D标签、当前EE推导TCP与oracle几何一致；full preflight核对全部2500份源size/mtime、标签hash和统计hash，通过。当前统计SHA256与旧数据相同（`19411c7e525bd0584013f8d60ae2dd45add530a63bc6894a1f29d8c4f7088007`），符合物理轨迹不变、腕部near裁剪改变的重建方式；该相同hash来自重新拟合，不是引用旧统计文件。

源根 `TRANSFER_AUDIT/ACCEPTED_WRIST020_20000_SCHEMA5.json` 记录20K/schema5验收通过，其中Aloha Clean2500。新腕部near=0.02m、head near=0.1m。抽查adjust_bottle/pick_dual_bottles/place_dual_shoes的episode0：动作/EEF/相机内外参相同，腕部图像中原被近裁剪移除的物体变为可见；新画面仍会存在真实的贴近物体/机械臂遮挡，不能将near修复解释为所有视角都能看到目标。对照图及指标在 `outputs/ea_lite/wrist020_clean50_20261015/inspection/`。

配置已解析验证；CPU真实样本验证结果在 `validation/sample_validation.json`（覆盖50任务每任务首/中/末窗口）。未按此新数据运行GPU训练，因此首步显存/实际更新需用户启动后核对，不把CPU数据检查当作GPU训练通过。

## 新训练参数

新入口：`scripts/train_ea_lite_wrist020_clean50.sh`；配置 `configs/train_ea_lite_wrist020_clean50.yaml`。默认只解析配置，`CHECK_ONLY=0`才启动。

保持C joint、Foundation80D、32actions/9video、BF16/ZeRO2/fused AdamW/等LR三组、bs64/rank×4卡/acc1/global256、5epochs/预计10845更新、video/action/base LR1e-4、ColorJitter0.2/0.2/0.2/0开启、EA系数0.025/0.01/0.05、save2000/full states/最近2组、seed42。日常默认fast预检，不重新构造数据。`EA_DATA_CHECK=none`可跳过独立数据预检，但仍检查GPU空闲；full仅在迁移或数据变化时使用。

从同一Foundation重新开始；不要用旧图像域run的resume状态冒充本实验起点。新输出目录：

```text
/remote-home/jinminghao/OpenWAM/outputs/ea_lite/wrist020_clean50_C_joint_bs64_5epochs/<时间戳>/
```

## 用户手动切换

首先确认新进程GPU访问已恢复（仅读取设备/查询驱动，不分配模型）；只有输出status=passed再执行后面的停止和启动：

```bash
cd /remote-home/jinminghao/OpenWAM
python scripts/check_cuda_access.py
```

先检查新配置（不会停止旧进程或启动训练）：

```bash
cd /remote-home/jinminghao/OpenWAM
bash scripts/train_ea_lite_wrist020_clean50.sh
```

由用户亲自停止旧run；辅助脚本读取launch_record，并核对PID的torchrun入口和唯一输出目录，防止杀错进程。去掉`--stop`仅查看，不发信号。

```bash
python scripts/stop_training_run.py \
  --record outputs/ea_lite/clean50_C_joint_bs64_5epochs_lr1e4_jitter_20261015/launch_record.json \
  --stop
```

旧torchrun当次核对PID为845189；不用`pkill python`或`killall`。停止命令只对指定旧torchrun发SIGTERM，由它结束自己的worker；不会触及其他作业。当前未到2000步时可能没有checkpoint，不会自动保存内存中的未落盘更新；本次更换数据从Foundation新开，不依赖旧checkpoint。

确认前四卡已释放（不要停止他人作业）：

```bash
nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
```

由用户启动新tmux训练：

```bash
tmux new-session -d -s ea_wrist020_clean50_20261015 \
  'env CUDA_VISIBLE_DEVICES=0,1,2,3 CHECK_ONLY=0 bash /remote-home/jinminghao/OpenWAM/scripts/train_ea_lite_wrist020_clean50.sh > /remote-home/jinminghao/OpenWAM/outputs/ea_lite/wrist020_clean50_20261015/train.log 2>&1'
tmux set-option -t ea_wrist020_clean50_20261015 remain-on-exit on
tail -f outputs/ea_lite/wrist020_clean50_20261015/train.log
```

已存在同名tmux会话时先查看 `tmux ls` 和日志，不重复启动两份。新脚本默认端口29875，与旧run端口29874区分，但bs64两份GPU模型无法在同卡共存，必须等旧worker释放后再启动GPU阶段。CPU初始化/权重读取仍需数分钟，不是重复全量数据预检。

## 与集群包的区别

已完成的集群baseline包使用用户指定的官方原始Clean2500：552287源帧/549787窗口/10740更新。它不改成wrist020自采数据。包中的legacy EA标签明确保留旧来源，不与官方RGB配对。本机新EA使用wrist020 Clean555151窗口/10845更新，两者数据源不同，不能将结果差异直接归因于EA结构。


## RGB复核补充

2026-10-15对当前旧数据与新wrist020分别抽查50任务/150轨迹/1350张JPEG，训练解码与官方参考RGB逐像素一致，未发现R/B颠倒。当前训练读取HDF5而非此前导出的MP4。旧inspection临时图片使用PIL直接解legacy JPEG，R/B显示会交换，不代表训练输入；请以 `outputs/ea_lite/rgb_audit_20261015/*_training_rgb_comparison.png` 和 `rgb_order_comparison.png` 为准。旧图保留作过程记录，腕部几何裁剪比较仍成立。
