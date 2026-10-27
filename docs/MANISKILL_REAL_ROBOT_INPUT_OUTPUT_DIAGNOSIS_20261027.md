# Sidecar33880 真机输入/动作离线诊断（2026-10-27）

## 范围和证据

只读取 GPU4 Flask server 的 `outputs/real_robot/maniskill_sidecar33880_gpu4_20261027/inference_records/`，并用其最早真机请求 `20261027T031524Z_*` 的原始 front/wrist 字节、EEF10 状态做固定输入 HTTP 对照；没有连接或驱动机械臂。服务 manifest 确认 `sidecar`、33880 checkpoint SHA256 `dc1645504f06e27983080b4b9181df8dd73ce2b3fcd871d6711c91f5c046b3c1`、GPU4、10步去噪、CFG1、无cache。原始响应均随请求落盘。

32条真机请求的任务文本全为 `push the red cube to the target region`，front/wrist 原始文件各有32种不同 SHA256，排除完全卡帧。初始真机 front 里红/蓝块很小、没有可见的 push 目标靶圈；初始 wrist 只有桌面和两块、看不到夹爪。训练集首条 PushCube 原生样本第20帧的 front 有醒目的红色靶圈及大尺度夹爪，wrist 也能看到夹爪。后期真机 front 画面里夹爪进入蓝块上方附近，并未见对红块的明确趋近。这些是图像观测，不等于已标定的三维物体位置。

## 固定同一真机观测，仅改指令

下表使用训练 manifest 中实际出现的完整英文指令。每项位置对照 red push；相同请求重复2次的输出逐位相同（red/blue push、red stack），排除采样随机性。其余两项各做1次。仅作离线灵敏度分析，灰块与 elevated goal 在该真机画面中均不存在，不可据此执行机械臂。

| 指令 | 首步目标xyz (m) | 相对red push首步差 (mm) | 全32步xyz RMS差 (mm) | 首步夹爪，0闭1开 |
| --- | --- | ---: | ---: | ---: |
| push the red cube to the target region | (0.555170, -0.004915, 0.219923) | 0 | 0 | 0.115 |
| push the blue cube to the target region | (0.555170, -0.004915, 0.219605) | 0.318 | 3.756 | 0.127 |
| stack the red cube on the gray cube | (0.556685, -0.004915, 0.220559) | 1.644 | 3.421 | 1.000 |
| pick up the red cube and move it to the elevated goal | (0.556685, -0.003726, 0.221831) | 2.711 | 4.059 | 1.000 |

同任务/EEF换成最后一条真机请求的 front/wrist（跨时刻、不一致的状态，仅作灵敏度测试），首步xyz改变3.909mm、全chunk RMS改变6.691mm、首步夹爪从0.115变0.722。**文本和视觉均没有完全失效，但当前真机画面的空间动作对目标颜色区分很弱。**不能凭此证明模型对所有训练域输入也坍塌。

## 轨迹和接口核查

- 第一条真机请求当前 TCP xyz=(0.548091, -0.009783, 0.249388)m；red push首步目标=(0.555170, -0.004915, 0.219923)m，即(+7.1,+4.9,-29.5)mm、范数30.7mm。chunk内相邻目标位移中位数1.36mm。
- 32条真机请求的首步 `|action[0].xyz - state.xyz|` 中位数5.7mm、最大30.7mm；各chunk内部相邻步长中位数约1.2mm。真机状态从首条到末条净变化(+92.2,+73.1,-220.1)mm，末条TCP z≈0.029m；所有red push首步夹爪值<0.2。
- 训练 manifest 指明 state 是 `achieved_base_tcp_at_t`，action 是 `absolute_controller_target_at_t_for_transition_t_to_t_plus_1`，20Hz、32步；因此“当前实测位姿→首个控制目标”和“两个相邻未来控制目标”本来不是同一比较口径。抽样100条 push 专家轨迹，全部时刻的 state→target 位移中位数12.8mm、相邻 target 中位数3.7mm。真机第一步30.7mm偏大，但不能单凭它诊断维度/单位错误。
- 当前仓库 `infer_openwam_minimal.py` 配置 `CHUNK_STEPS=1`、`STEP_SECONDS=1.0`，只执行chunk首步然后重采样；如真机副本一致，就会反复执行每轮较大的首步，而不是走完后面的细小32步。训练动作是20Hz，真机此处每步约1秒加推理/交互延迟；与训练时间尺度不一致。
- Server按 front→`front_camera`、wrist→`hand_camera` 组成训练同款384×320 L形画布，文本加训练同款前缀，EEF10经checkpoint normalizer反归一化成物理绝对base-frame TCP。client只将预测xyz直接作为绝对TCP目标，不做delta累加；记录样本的目标link8→TCP往返位置误差0、旋转误差约1e-16。真机URDF的link8→hand→hand_tcp为绕z负45°、z+0.1054m，与脚本常数一致。
- 但 `T_MODEL_FROM_REAL_BASE=np.eye(4)` 只是当前假设，不能从数学往返证明真实桌面/物体与仿真base已对齐；相机外参、TCP实际工具点、桌面高度也未现场验证。当前模型输出确实是先下降、近桌面后作小幅平面移动，**不等于这种运动对于当前红块/无目标区场景是正确策略**。

## 结论与下一步

最强的已观察问题是任务/视觉域差：训练 PushCube 需要可见目标靶圈、特定相机近景与腕视角，真机场景缺目标区、夹爪尺度与腕图完全不同；在该真机观测上，改目标颜色几乎不改首步空间动作，故未表现出可靠选红块。首步大跳还与已达状态/控制目标口径不同、初始TCP高度0.249m高于抽样100条训练 push 轨迹初始高度p99约0.207m，以及只执行首步反复重规划有关；这些是可能机制，不能单独定因。

先不要把 stack 指令或这一近桌面轨迹用于真机试动作。优先做独立相机/桌面/机器人base标定，确保目标区域与训练任务对应；采集更接近训练视角且夹爪可见的 front/wrist，再用同一固定EEF与多任务文本复测首步空间差异。仅HTTP和几何往返通过，不证明真机控制或任务成功率。
