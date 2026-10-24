> 2026-10-22新增：已准备A100 GPU5/6模型服务、4090六路仿真的24项续测脚本与完整命令，见本文后半部分“4090六路续测24项”。该批次尚未由本助手启动；用户已确认跨机链路验证通过。
>
> 原运行状态：2026-10-22T16:05:41.917766+00:00 已按用户要求停止旧42960评测。完整成绩和未完成进度见 [停止记录](ROBOTWIN_SIDECAR42960_STOPPED_20261022.md)。下文保留停止前快照。

# Sidecar最终42960权重：标准RoboTwin评测

更新时间：2026-10-22T16:01:59.666692+00:00；状态：running

运行目录：/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z

完整验收：20/100项；日志观察到完成：2401/10000条。

完整Sidecar；10步；cache .99/max3；FFN局部编译；官方环境/native专家筛选、RT、原预算不变。
部分或异常退出结果仅作进度，不能当最终SR；不与32k或其他机器样本拼接。

|任务|条件|状态|成功/观察完成|完整百条验收|
|---|---|---|---:|---|
|adjust_bottle|demo_clean|complete|99/100|True|
|adjust_bottle|demo_randomized|complete|50/100|True|
|place_shoe|demo_clean|complete|94/100|True|
|place_shoe|demo_randomized|complete|52/100|True|
|place_dual_shoes|demo_clean|complete|87/100|True|
|place_dual_shoes|demo_randomized|complete|7/100|True|
|handover_block|demo_clean|complete|98/100|True|
|handover_block|demo_randomized|complete|52/100|True|
|blocks_ranking_rgb|demo_clean|complete|95/100|True|
|blocks_ranking_rgb|demo_randomized|running|40/66|False|
|click_bell|demo_clean|complete|100/100|True|
|click_bell|demo_randomized|complete|73/100|True|
|grab_roller|demo_clean|complete|100/100|True|
|grab_roller|demo_randomized|running|45/91|False|
|hanging_mug|demo_clean|running|7/26|False|
|hanging_mug|demo_randomized|running|4/16|False|
|move_can_pot|demo_clean|running|38/40|False|
|beat_block_hammer|demo_clean|complete|98/100|True|
|beat_block_hammer|demo_randomized|complete|44/100|True|
|scan_object|demo_clean|complete|91/100|True|
|scan_object|demo_randomized|complete|58/100|True|
|place_container_plate|demo_clean|complete|96/100|True|
|place_container_plate|demo_randomized|complete|79/100|True|
|click_alarmclock|demo_clean|complete|99/100|True|
|click_alarmclock|demo_randomized|complete|66/100|True|
|blocks_ranking_size|demo_clean|running|28/52|False|
|blocks_ranking_size|demo_randomized|running|3/28|False|
|dump_bin_bigbin|demo_clean|running|76/82|False|

## 4090六路续测24项（2026-10-22准备，未自动启动）

### 口径与选择

50个task × Clean/Randomized共100个条件。旧A100仿真实验完整完成20条件，
剩余80条件中包含8个曾中止的partial和72个未启动条件。
这次从未启动条件中选12个task、各测两域，共24条件 × 100条=2400条。
不是24个task各测两域；与旧20完整条件、8个partial均无重叠。
旧partial保留，不拼入这次100条；旧成绩及停止记录不覆盖。
本批全部完成后，将拥有44个完整条件的独立记录，仍不是完整50任务评测。

选择覆盖handover、搬运、按压、开合、放置，并包含open_microwave、
place_bread_skillet、place_burger_fries、place_cans_plasticbox、stamp_seal五个
与复现baseline10740的旧十任务集可比较的task。选集不代表随机抽样或完整总体。

|task|本次域|episode数|旧42960记录|
|---|---|---:|---|
|handover_mic|demo_clean + demo_randomized|100 + 100|两域未启动|
|lift_pot|demo_clean + demo_randomized|100 + 100|两域未启动|
|move_pillbottle_pad|demo_clean + demo_randomized|100 + 100|两域未启动|
|move_playingcard_away|demo_clean + demo_randomized|100 + 100|两域未启动|
|move_stapler_pad|demo_clean + demo_randomized|100 + 100|两域未启动|
|open_laptop|demo_clean + demo_randomized|100 + 100|两域未启动|
|open_microwave|demo_clean + demo_randomized|100 + 100|两域未启动|
|place_bread_skillet|demo_clean + demo_randomized|100 + 100|两域未启动|
|place_burger_fries|demo_clean + demo_randomized|100 + 100|两域未启动|
|place_cans_plasticbox|demo_clean + demo_randomized|100 + 100|两域未启动|
|stamp_seal|demo_clean + demo_randomized|100 + 100|两域未启动|
|turn_switch|demo_clean + demo_randomized|100 + 100|两域未启动|

封存选集：`benchmarks/robotwin/plans/sidecar42960_remaining24_20261022.json`。
脚本会比对旧final_record SHA、确认24条件互异且不与旧complete/partial重叠。
按Randomized优先、域内官方动作预算较大的条件优先领取；不修改动作预算。

### 六路如何调度

每个模型3个slot，独立policy/action buffer、共享模型推理互斥。
六个worker动态领取同一队列；某一路完成后立即取下一项，不固定绑定4项任务。
任何时刻每个slot只由一个worker使用。脚本启动前检查六路session/42960权重SHA、
端口ping与4090显存；每路默认要求至少8000MiB空闲，多路同卡时按数量累加。
该显存门槛是初步容量保护，不能保证所有task峰值；运行前确认旧smoke进程已退出。

|A100模型GPU|模型/隧道端口|session|4090仿真GPU示例|
|---|---:|---|---:|
|5|22700|model5/session_slot0.json|0|
|5|22701|model5/session_slot1.json|1|
|5|22702|model5/session_slot2.json|2|
|6|22710|model6/session_slot0.json|4|
|6|22711|model6/session_slot1.json|5|
|6|22712|model6/session_slot2.json|6|

正式保留：Sidecar42960完整读写、10步去噪、FFN compile、cache .99/max3、
绝对EEF/wxyz、自有stats、官方专家筛seed/unseen/seed0、RT、官方成功判据与预算。
`remote_eval.py sim`在本次独立配置中自动设`eval_video_log: false`；仍正常渲染策略RGB。
相机near仍为官方0.1，不混入训练腕部near0.02的独立诊断。

### 全部操作可在A100终端完成

可以。A100本地运行模型和反向隧道，再通过SSH在4090上启动独立后台dispatcher。
仿真实际执行位置仍是4090，启动后SSH命令立即返回；进度与结果写共享盘。
需要A100→4090免交互SSH登录；首次host-key确认/公钥/agent先配置好。

**1. A100端：模型若已启动，跳过新建，沿用实际RUN。**

以下只用于新建两份模型。当前已有的模型服务不要重复启动。

```bash
cd /remote-home/jinminghao/OpenWAM
RUN="$(pwd)/outputs/robotwin_eval/split42960_a100_gpu56_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN"
CKPT="$(pwd)/outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z/checkpoint_step_42960"
DEPLOY="$(pwd)/outputs/robotwin_eval/sidecar42960_gpu56_20261020T022957Z/deploy.yaml"

nohup .venv/bin/python -u benchmarks/robotwin/remote_eval.py model \
  --kind sidecar --gpu 5 --port 22700 --slots 3 \
  --checkpoint "$CKPT" --deploy-config "$DEPLOY" \
  --output "$RUN/model5" > "$RUN/model5_launcher.log" 2>&1 &

nohup .venv/bin/python -u benchmarks/robotwin/remote_eval.py model \
  --kind sidecar --gpu 6 --port 22710 --slots 3 \
  --checkpoint "$CKPT" --deploy-config "$DEPLOY" \
  --output "$RUN/model6" > "$RUN/model6_launcher.log" 2>&1 &

echo "$RUN"
```

初始化/编译完成后确认六个session文件都存在：

```bash
ls "$RUN"/model{5,6}/session_slot{0,1,2}.json
tail -n 15 "$RUN/model5/server.log" "$RUN/model6/server.log"
```

**2. A100端：若隧道已验证通过，跳过，不重复启动。**

```bash
ssh -n -o BatchMode=yes -p 26199 root@210.45.70.21 'echo KEY_AUTH_OK'

nohup bash benchmarks/robotwin/remote_reverse_tunnel.sh \
  root@210.45.70.21 26199 \
  22700:22700 22701:22701 22702:22702 \
  22710:22710 22711:22711 22712:22712 \
  > "$RUN/tunnel.log" 2>&1 &
```

脚本已禁止后台交互；`Stopped`/认证失败及重复隧道处理见
`docs/ROBOTWIN_SPLIT_HOST_EVAL_20261022.md`第5节。

**3. A100端：仅准备命令时可先dry-run。**

已有服务时，先将RUN设为包含model5/model6的实际共享目录。
下例不启动任务；可核对远程GPU、端口与输出路径。

```bash
.venv/bin/python benchmarks/robotwin/launch_remaining24_from_a100.py \
  --model-run "$RUN" --sim-gpus 0,1,2,4,5,6 --dry-run
```

**4. A100端：正式在4090启动六路、24条件、每条件100条。**

```bash
.venv/bin/python benchmarks/robotwin/launch_remaining24_from_a100.py \
  --model-run "$RUN" \
  --sim-gpus 0,1,2,4,5,6 \
  --ports 22700,22701,22702,22710,22711,22712 \
  --ssh-host root@210.45.70.21 --ssh-port 26199
```

这条命令会返回4090后台dispatcher的PID和hostname，不代表2400条已经完成。
默认输出 `$RUN/remaining24_4090`，启动标记用独占创建阻止重复提交。
不要同时再手动启动另一批使用相同slot的仿真。

如果希望自己登录4090启动，同样只运行一次，二选一：

```bash
cd /remote-home/jinminghao/OpenWAM
# RUN使用与A100相同的实际共享目录
nohup .conda/robotwin-official/bin/python -u benchmarks/robotwin/eval_remaining24.py \
  --model-run "$RUN" --output "$RUN/remaining24_4090" \
  --sim-gpus 0,1,2,4,5,6 \
  --ports 22700,22701,22702,22710,22711,22712 \
  > "$RUN/remaining24_4090_dispatcher.log" 2>&1 < /dev/null &
```

**5. A100端查看进度、日志和最终结果。**

```bash
cat "$RUN/remaining24_4090_launch.json"
tail -f "$RUN/remaining24_4090_dispatcher.log"
cat "$RUN/remaining24_4090/state.json"
cat "$RUN/remaining24_4090/summary.json"
```

`state.json`包含当前各worker/task/PID、完整结果和错误；`summary.json`在队列结束后生成。
每条件原生证据在`remaining24_4090/<mode>/<task>/run.json`与`client.log`。
失败条件保留错误并继续派发其他条件，失败/不完整项不填成0成功率、不计入完整分母。
完整本批验收应为`status=complete`、`complete_conditions=24`、`episodes=2400`。
dispatcher只管理自己启动的sim进程，不停止模型、隧道、其他旧实验或训练。
此脚本不自动拼接旧A100与新4090结果；后续跨硬件汇总需保留来源、条件覆盖和未配对限制。

### 验证记录与范围

静态选集审计通过：24唯一条件，旧complete/partial重叠均0；dry-run输出已封存。
真实dispatcher的CPU替身测试覆盖六worker调度、24条件恰好一次、单条件失败后继续、
完整分母剔除失败项、既有输出拒绝覆盖。测试不调用GPU/真实模型，不是新SR。
证据：`outputs/robotwin_eval/split_host_setup_20261022/remaining24_validation.json`。
用户已报告跨机链路通；本助手没有再次启动实际2400条，以免与现有服务/手动任务重复。
