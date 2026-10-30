# Sidecar42960 跨机评测：按用户要求停止

停止记录：2026-10-23T07:59:09.686126+00:00。根目录：
`outputs/robotwin_eval/split42960_a100_gpu56_20261022T160926Z`。

## 最终状态

- 已停止4090上的 **8路独立case评测**（原6路+增量2路），以及调度器、接续等待进程和自身子进程。
- 已停止A100上的 **4份Sidecar42960模型**（GPU5/6各2份）、其编译worker和两条本批SSH反向隧道。
- 本批remote快照31个进程、local快照142个进程均在按PID+start_ticks复核后结束；
  先TERM并等待退出，本轮没有强制KILL。没有按GPU、用户名或宽泛名称批量清理。
- 停止前冻结5个派发/接续控制器，防止结束当前case后又启动新任务。
- 当时8个运行条件中没有98/99条的近完成条件，最近为move_pillbottle_pad Clean88/100，
  其余79/67/47/40/29/19/0条，因此按用户“再久不等”的要求没有额外等待。
- A100 GPU0–3训练PID3501138的start_ticks前后相同，训练仍在运行；GPU4/7与4090其他作业保留。
- 停止后本批两端均无活跃残留，8个模型/转发端口关闭；A100 GPU5/6各1MiB占用，
  GPU0–3约65GiB/卡且利用率约98–100%。

## 保存的成绩

| 状态 | 条件数 | 口径 |
| --- | ---: | --- |
| 完整完成 | 21 | 2100条，1237成功 |
| 用户中止 | 8 | partial保留，不计正式SR |
| 原有异常 | 2 | open_microwave Randomized、open_laptop Clean，不计正式SR |
| 未启动 | 41 | 包含40项后续长队列，不能当0% |
| 总计划 | 72 | 本次跨机计划，不含旧A10020完整条件 |

正式成功率：**58.90%（1237/2100）**。
官方相同21条件1220/2100=**58.10%**，
差0.81个百分点。
两批覆盖与训练/仿真/随机性不同，不能从该选择集推断全50任务或结构因果。

| 任务 | 条件 | 成功/100 |
| --- | --- | ---: |
| handover_mic | Randomized | 27 |
| lift_pot | Clean | 99 |
| lift_pot | Randomized | 21 |
| move_pillbottle_pad | Randomized | 27 |
| move_playingcard_away | Randomized | 54 |
| move_stapler_pad | Randomized | 16 |
| open_laptop | Randomized | 61 |
| place_bread_skillet | Clean | 94 |
| place_bread_skillet | Randomized | 45 |
| place_burger_fries | Clean | 96 |
| place_burger_fries | Randomized | 71 |
| place_cans_plasticbox | Clean | 98 |
| place_cans_plasticbox | Randomized | 91 |
| place_empty_cup | Randomized | 78 |
| place_fan | Randomized | 45 |
| press_stapler | Clean | 96 |
| press_stapler | Randomized | 43 |
| rotate_qrcode | Clean | 90 |
| rotate_qrcode | Randomized | 15 |
| stamp_seal | Randomized | 27 |
| turn_switch | Randomized | 43 |

## 证据与后续

证据目录：原根下`user_stop_20261023T0755Z/`。

- `final_record.json`：最终完整/partial/异常清单、训练身份、汇总分子分母。
- `sim_before.json`、`model_before.json`：停止前进程身份与成绩快照。
- `sim_termination.json`、`model_termination.json`：信号目标及强制清理记录。
- `sim_after.json`、`model_after.json`：残留与显存、训练身份验收。
- `raw_runs_after_termination_before_annotation.json`：终止产生的原始状态，保留原始错误。
- `queue_states_before_stop_annotation.json`：终止后、用户停止标注前的队列状态。

根目录USER_STOPPED.json及各队列state明确stopped_by_user；原8个运行case标注
interrupted_by_user，并保留user_stop.raw_status。用户中止不是模型策略失败。
未删除或覆盖任何权重、视频、原始日志、既有完整成功记录或旧评测；注释前状态另有快照。

下一步：若重新测试，先决定30/50条预评测或100条正式口径，使用新目录和固定场景/指令清单。
现有渲染缓冲区异常和handover_mic Clean停滞未在本次停止中修复；
不要直接拼接partial或将恢复作业与旧失败片段合成100条。本次没有启动新的评测。

