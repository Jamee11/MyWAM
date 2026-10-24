# Sidecar32k腕部近裁剪面对照评估

启动时间：2026-10-22T14:45:46.966092+00:00。用户授权运行near0.02/0.1评估，并要求保存两种near的腕部图片。

## 固定设置

- 权重：32k，原完整10000条批次的只读pinned checkpoint；SHA256 0f22e7954ac5278e0e2184feea65c698754e4efad6b9fc9e317399e14f898de1。
- Randomized三任务：adjust_bottle、beat_block_hammer、place_shoe；两个低成功率任务加一个对照任务。复用已封存的官方专家可解seed/指令manifest，每任务3个场景，共9对/18条策略rollout。不是100条标准评测，不替换已有官方环境成绩。
- 每对相同task/setting/seed/指令、相同cache .99/max3/FFN compile/10步去噪/32动作/seed42生成；仅腕部near0.1/0.02改变，主相机near0.1、原相机分辨率/姿态/预算/成功判据不变。
- 独立L20模型GPU1 PID2905958，原32k完整FFN数值验收后才开放；GPU转移需free>=33000MiB，不停止其他作业。模型之外按free>=16000MiB、每GPU最多两个本次sim动态入场。空闲旧GPU5服务因该卡其他占用仅余约2.9GiB不用于本次。
- 控制器PID2906368。先单独捕获9场景初始双near图像，不需等待模型；随后两个独立端口并行执行near对照。两组均不保存视频。

## 图片保存方式

run_near.py仅在独立Python进程内包装Camera.load_camera与Base_Task.get_obs，不写官方RoboTwin源文件。设置并读取验证左右腕部实际near。每个场景在动作步0/32/64/128/256（实际达到才保存）对同一物理状态拍摄near0.1、near0.02及near0.1重复参考，不调用scene.step，也不调用会随机化灯光的_update_render。相机姿态/内参和动作计数必须不变，图片成对保存，最后恢复该实验的near。主视角也保存，用于检查RT噪声/背景变化。

每个图片目录包含near010/near002的左右腕部与head原始PNG、两幅H384×W320完整拼图、wrist_near_comparison.png及evidence.json。统计两种near图像绝对差异和同near重复渲染差异，避免把光追噪声全部误认为裁剪差异。PNG来自实际仿真RGB，未人工修改或伪造图像。

拍照会增加RT渲染调用，两个策略分支都保留同样采样规则；并发/RT内部采样仍非逐bit控制。未声称RGB差异必然影响SR，更未声明改near可以提升成功率。固定manifest来自先前同版本官方场景，不为某一near按策略结果筛掉失败。

## 状态与后续

原始状态：outputs/robotwin_eval/nearclip32k_ab_20261022/state.json；图片位于capture/near0.10/<task>/images与eval/near0.10、eval/near0.02；policy rollout结果在各目录result.json，完成后汇总paired_results.json。

当前已启动，仿真/模型加载中。下一步核验首批同状态双nearPNG、实际near读数及完整18条终态；失败保持基础设施诊断，不混算为策略失败。数据量仅9对，结果用于定位问题，后续若有明显差异再扩场景数。官方标准32k/baseline/42960已有结果及其运行进程保持不变。


### 2026-10-22T14:50:02.303854+00:00 首组同状态腕部双near图像已保存并目视核查

adjust_bottle/Randomized episode0 step0已得到真实near0.02/0.1左右腕部原图、384×320整图及并排图。相机姿态/内参及动作计数断言通过，渲染采样不推进physics。该初始状态肉眼尚未见明显额外裁剪，双near左右腕平均绝对像素差约0.975/0.892，同near重复渲染约0.819/0.728（0–255尺度），有相近量级RT采样噪声，不能据此说near已导致低SR或对所有接触状态无影响。模型继续加载；控制器会保存后续rollout动作步0/32/64/128/256实际达到状态的双near图片。首图路径capture/near0.10/adjust_bottle/images/episode0_step0000/wrist_near_comparison.png。


## 2026-10-22T15:11:04.263265+00:00 实验异常

["RuntimeError('eval/adjust_bottle/near0.1: exit1; inspect /remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/nearclip32k_ab_20261022/eval/near0.10/adjust_bottle/client.log')"]。这是基础设施/运行异常，不是策略失败率；完整诊断见state.json与各client.log。

所有原始图片：outputs/robotwin_eval/nearclip32k_ab_20261022/{capture,eval}；同状态双near、重复渲染噪声指标见各images/*/evidence.json；原标准评估成绩保持原记录。
