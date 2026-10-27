# 24k评测：首批30条结果与官方对照

快照：2026-10-23T08:41:26.985121+00:00。已完整1/100条件、观察到130条终态、9路活跃、基础设施错误0。

仅click_alarmclock/demo_clean完整30/30=100%，官方99%，差+1pp。其余为partial，不纳入完整条件平均，不当成全50任务总体成绩。

|Task|Mode|进度|成功/条数|当前率|官方100条率|差值pp|
|---|---|---|---:|---:|---:|---:|
|click_alarmclock|demo_clean|complete|30/30|100.00%|99.00%|+1.00|
|adjust_bottle|demo_clean|running|17/17|100.00%|90.00%|+10.00|
|adjust_bottle|demo_randomized|running|3/6|50.00%|51.00%|-1.00|
|beat_block_hammer|demo_randomized|running|3/8|37.50%|55.00%|-17.50|
|beat_block_hammer|demo_clean|running|16/18|88.89%|99.00%|-10.11|
|click_alarmclock|demo_randomized|running|12/15|80.00%|77.00%|+3.00|
|place_shoe|demo_randomized|running|6/10|60.00%|51.00%|+9.00|
|place_shoe|demo_clean|running|23/23|100.00%|98.00%|+2.00|
|put_bottles_dustbin|demo_clean|running|3/3|100.00%|83.00%|+17.00|

官方表本次在线复核：[Issue46](https://github.com/OpenWAM-Official/OpenWAM/issues/46#issuecomment-6118279946)；官方每条件100条，本批目标每条件30条。
当前Randomized的hammer偏低、alarmclock/shoe暂高、bottle接近，但各仅6–15条，不能确定最终趋势。
短/高成功任务先完成造成当前覆盖偏差；尚不能给全100条件综合率或判断warmup结构带来提升。
继续当前队列并待更多完整30条，实时进度见ROBOTWIN_WARMUP2K24000_30EP_LIVE_20261023.md。
