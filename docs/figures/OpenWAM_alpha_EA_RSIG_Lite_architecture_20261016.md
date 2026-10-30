# OpenWAM-alpha + EA-RSIG-Lite 架构图

日期：2026-10-16。图示范围：已实现的 Lite C joint 结构；不表示当前集群 baseline 已启用 EA，也不表示已经验证 C2R 性能提升。

图文件：`OpenWAM_alpha_EA_RSIG_Lite_architecture_20261016.png`（3600×2600）及同名可编辑 `.svg`。

## 阅读顺序

1. 左侧 A：OpenWAM 视频/动作双分支。蓝色视频、紫色动作，共用联合注意力。未来视频 latent 与动作分别加噪，当前首帧 latent 保持干净。
2. 绿色 ①：从当前首帧干净 VAE latent 经共享 patch embedding 取特征，送入 EA；不是从中间 Video DiT 层取 H18，也没有额外执行一遍 observation DiT。
3. 右侧 B：展开 EA 空间定位、关系编码、进展汇聚、运动计划以及门控组合，生成 15×512 tokens。
4. 绿色 ②：block 0 完成后，Action hidden 作为 Q，EA tokens 作为 K/V，经8头交叉注意力与零初始化输出投影，残差加回 Action。Video 在这个边界没有直接新增残差，后续仍按原联合注意力交互。
5. 底部 C：原始 oracle 离线生成130D监督标签；只进入辅助损失。推理输入只有当前图像、文本和可观测当前 EEF/TCP 状态，不含oracle标签。

## 与代码的对应

| 图中模块 | 实现 |
|---|---|
| 当前首帧 latent 与 patch embedding | `openwam/model/architectures/dual_system/ea_rsig_lite.py::_observe` |
| 三视角各14×14、文本通道调制 | `openwam/model/modules/ea_rsig_lite.py::EARSIGLiteBridge.observe` |
| Ground：4个query的空间注意力、null slots | `openwam/model/modules/ea_rsig_v3.py::_ground` |
| EEF MLP 与 Relation MLP | `EARSIGV3Module.effector_encoder / relation_encoder / _relations` |
| Progress：learned query、8头注意力、FFN | `EARSIGV3Module._progress` |
| Plan：EEF/object MLP、方向与运动概率 | `EARSIGV3Module._plan` |
| 4 Ground + 4 Relation + 1 Progress + 6 Plan | `EARSIGV3Module._compose_action_tokens` |
| Action残差 | `EARSIGLiteBridge.condition_action` |
| block0之后接入 | `EARSIGLiteArchitecture.forward`，`adapter_layer=0` |
| 三项辅助监督 | `EARSIGLiteArchitecture.compute_loss` |

文本在EA内采用masked mean后调制视觉通道，未额外添加一个EA文本Transformer。Progress是学习到的表征，不是显式完成百分比；Plan预测8/16/32步方向与是否运动，不直接输出绝对EEF执行动作。最终动作仍由OpenWAM Action分支预测与采样。

默认目标：`L_video + L_action + 0.025 L_ground + 0.01 L_relation + 0.05 L_plan`。先前讨论的整体0.5档未作为此图默认。

图中训练预测头输出flow velocity；干净动作与视频须经流采样恢复，不能把训练头输出直接视为可执行动作或已解码图像。

## 生成与保存

先使用 imagegen 技能的内置 image_gen 工具，参考用户提供的 `docs/image.png` 生成配色与分区初稿。因生成式图形的连线仍有歧义，最终以独立矢量绘图脚本 `draw_openwam_ea_rsig_lite_architecture.py` 重绘并逐条固定数据流；SVG 是可编辑源图，PNG 由 resvg-py 渲染。文字使用 Noto Sans CJK SC，渲染工具仅安装于 `/tmp/openwam-diagram-tools-20261016`，没有修改训练环境。

原图未覆盖，模型、数据、训练代码及运行进程均未修改。原始提示词和连线修订要求保存在同目录 `_prompts.md` 文件。生成式初稿仅作设计参考，不作为代码连接关系的最终依据。
