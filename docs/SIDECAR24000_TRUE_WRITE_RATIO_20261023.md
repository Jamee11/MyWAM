# checkpoint24000：真实写回量级与Action收益核对

## 完成结果：真实量级与Action对照

已完成25窗口×Full/Off=50组生成及32组Action FM配对。50组动作与上一轮逐位相同，
说明新增范数测量没有改变推理。所有Off原生残差/真实变化为零；FM前后Sidecar参数指纹一致。
独立模型PID3654775已退出，GPU5/6各1MiB，GPU0–3原训练保持运行。

下面只汇总24个held-out窗口，不混入训练内控制；每tap为240次真实forward观察，
来自6条专家episode，不是240条独立episode。均值比例是每次先求比再平均。

| Block | hidden L2 | 原生gate×projection L2 | 真正delta L2 | 原生相乘比例 | 实际写入比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 323.499 | 1.3340e-3 | 1.4558e-4 | 4.2316e-6 | 4.6837e-7 |
| 15 | 398.993 | 8.2956e-3 | 2.1787e-3 | 2.1012e-5 | 5.5555e-6 |
| 23 | 369.633 | 5.5939e-3 | 1.2097e-3 | 1.5182e-5 | 3.2897e-6 |

| Block | hidden RMS | gate×projection RMS | 真正delta RMS | 数值变化元素占比 |
| --- | ---: | ---: | ---: | ---: |
| 7 | 1.787094 | 7.3696e-6 | 8.0425e-7 | 0.367% |
| 15 | 2.204143 | 4.5827e-5 | 1.2036e-5 | 2.087% |
| 23 | 2.041952 | 3.0902e-5 | 6.6828e-6 | 1.387% |

**原生相乘残差约为hidden的百万分之4.23/21.01/15.18；经过BF16相加后，实际变化只有
百万分之0.47/5.56/3.29。** FP32乘法参考和原生乘法的范数非常接近，主要进一步削弱发生在加法。
不能把actual/native范数之比叫“信息保留率”；舍入是逐元素非线性，也可能把个别变化量化放大。
本结果既不等于完全没写入，也不支持“beta有1e-4就实际写入1e-4”的说法。

### Action是否改善

使用同一个prepared batch和相同CPU/CUDA RNG状态，Full−Off为正表示Full更差。

| 范围 | 配对数 | Full Action FM loss | Off Action FM loss | 相对变化 | Full更好/更差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部 | 32 | 6.581141e-4 | 6.578709e-4 | +0.0370% | 17/15 |
| Clean | 16 | 4.987002e-4 | 4.973672e-4 | +0.2680% | 9/7 |
| Randomized | 16 | 8.175281e-4 | 8.183746e-4 | -0.1034% | 8/8 |

**没有稳定可观的Action改善：整体FM loss略高0.037%，17/32更好、15/32更差，
Clean平均更差、Randomized略好。** 差异很小，仅六条episode的多个窗口，未作显著性或全分布估计。

相同seed42的10步生成GT对照也无一致收益：Full比Off的全32步XYZ误差，
Clean高0.007602mm，Randomized低0.005690mm。
Full/Off预测的EEF目标本身平均相差Clean0.5119mm、Randomized0.5169mm；
这是两条预测之间的距离，**不是向GT改善了0.51mm**。

### 可审查产物及解释边界

- write_magnitudes_summary.json：按域/三tap范数、RMS、比例及分位数和FM汇总。
- results.json：逐窗口/逐去噪时刻的全部原始测量。
- paired_action_fm.json：逐批Full/Off的原训练Action/Video/EA loss与参数指纹检查。
- measurement_parity.json与complete.json：动作不变及两阶段完成证据。

结论是此checkpoint的真实直接写回非常弱，并未在这些固定样本上带来一致Action收益，
不是已经证明整个Sidecar训练无效。辅助监督对主干的训练作用不能被runtime WriteOff移除。
不因该结果直接增大beta、强制比例或修改BF16路径；下一步若验证控制收益需固定场景的配对闭环，
并保留同训练预算/统计的对照。本轮没有新闭环SR，不把FM loss或GT误差当SR。


对象：最早isolation warmup2K的`outputs/1022-sidecar-warmup2k/checkpoint_step_24000.safetensors`。
SHA256沿用已验证只读快照：c8835677cb59324ce3f2bf8c01e80178e36397ac3379d7bc1ca3d0d6ea23de76。
本轮不训练、不optimizer.step、不改生产模型，仅扩展现有WriteProbe观测字段。

## 测量定义

令h为tap7/15/23实际送入write_action的Action hidden张量，p为Write投影输出，beta为实际gate。

- FP32参考残差：`beta.float() * p.float()`，相对比`||reference||2 / ||h||2`。
- 原生相乘残差：按原表达式dtype执行`beta.to(h.dtype) * p`，再转FP32测范数；
  `r_native_pre = ||native_residual||2 / ||h||2`。
- 真正写入：执行原生产加法后，计算`delta = h_after.float() - h.float()`；
  `r_actual = ||delta||2 / ||h||2`。
- 另记hidden/projection/gated residual/delta的绝对L2与RMS，以及`h_after != h`元素比例。

比例先按每次forward、每个batch样本计算，再汇总分位数；均值比例不强求等于平均范数之比。
这是完整hidden张量（包括实际被写回的token）上的范数，不是80D动作或物理20D EEF的范数。
BF16相乘与加法分开观察，不把beta数值当作真实写回比例，也不把FP32参考当实际hidden变化。

## Action对照

1. 复用前一轮25个GT窗口，Full/Off各一次，共50次10步生成，固定seed42、eager/no-cache。
   断言50个生成动作与上一轮同输入结果逐位一致，排除增加测量改变推理。
2. 复用前一轮32个固定梯度窗口，但本次只做前向Action FM loss配对。
   每对使用同一个prepared输入，并恢复相同CPU/CUDA RNG状态；固定Video ID500、
   Action ID100/500/900与原逐批noise seed。沿用原训练loss、mask和权重，不手写替代MSE。
3. 同时保存Video/EA loss作为参照。FM使用训练定义的带噪GT未来输入，只衡量局部去噪；
   生成GT误差则未来只用于评分。两者都不是闭环成功率。
4. 对比前后Sidecar参数内容指纹，确认没有权重更新；本轮没有重新开始正式RoboTwin队列。

## 执行与证据（启动时历史记录，最终结果见顶部）

输出：`outputs/robotwin_eval/sidecar_isolation2k_step24000_write_magnitudes_20261023`。
复用此前checkpoint_step_24000只读快照与输入，不覆盖前一轮结果。GPU5编码文本、GPU6推理，
GPU0–3训练不操作。初次独立入口在前向前重启一次，修正标量BF16参数指纹reshape，日志保留。

7项原探针测试及1项新增BF16测试通过：明确验证非零gate×projection可以在hidden相加时
完全舍入为0变化；同时验证范数比例和Off的零残差/零变化比例。测试不代表模型真实结果。
现模型PID3654775加载中，完整结果以complete.json及下述产物为准，不提前宣称完成。

- results.json：逐窗口/逐tap/逐去噪时刻的范数、RMS与比例，以及GT误差。
- measurement_parity.json：50个动作输出与旧测量结果逐位一致。
- paired_action_fm.json：32组真实Action FM loss Full/Off配对与参数指纹检查。
- complete.json：GT与FM两阶段均完成。

## 已有先验，不代替本轮绝对范数测量

前一轮GT数据中，FP32相乘参考残差比约4.1e-6–2.2e-5，实际hidden变化约4.4e-7–6.1e-6。
这已经不是单看gate；本轮补全原生dtype相乘、绝对量级与变化元素比例，并确认Action收益。
小样本仅六条held-out专家episode，多个窗口/去噪时刻不视为独立episode。
尚无本轮闭环或普遍泛化收益结论，不因比例小就断言辅助训练无效或直接放大gate。
