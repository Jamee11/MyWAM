# 原21480 Sidecar推理干预：配对闭环评估

更新：2026-11-02T06:59:48.729095+00:00；三组齐全条件：20/20；完成组条件：60/60；运行：0。

|范围|配对条件|每组条数|Full|Write-Off|Relation-Reset|
|---|---:|---:|---:|---:|---:|
|all|20|200|77.00%|79.50%|81.50%|
|demo_clean|10|100|92.00%|91.00%|93.00%|
|demo_randomized|10|100|62.00%|68.00%|70.00%|

只比较三组相同seed/语言清单全部完成的条件。Write-Off为去除全部三tap的直接Action写入；Relation-Reset为每tap后关系token索引4:8置零并进入后续tap与Write。不是只打乱head预测，token置零造成的分布变化不能自动解释为已证明语义因果。原21480历史74.83%来自全50任务/30条，不与此选择集直接排名。

## 逐任务拆分：原10条配对闭环（2026-11-02）

仅原三组600条实验，每task/domain/arm=10条，相同有效seed/语言。不是新18组每条件5条的混合结果。单位%，Δ为相对Full的百分点。

### demo_randomized

|Task|Full|Write-Off|Δpp|Relation-Reset|Δpp|
|---|---:|---:|---:|---:|---:|
|adjust_bottle|40|40|+0|40|+0|
|beat_block_hammer|50|60|+10|60|+10|
|handover_block|80|80|+0|90|+10|
|place_bread_skillet|70|70|+0|70|+0|
|place_burger_fries|80|70|-10|80|+0|
|put_bottles_dustbin|50|80|+30|70|+20|
|place_object_basket|60|80|+20|100|+40|
|place_cans_plasticbox|100|100|+0|100|+0|
|stack_blocks_three|90|90|+0|80|-10|
|open_microwave|0|10|+10|10|+10|
|平均（各组100条）|62|68|+6|70|+8|
### demo_clean

|Task|Full|Write-Off|Δpp|Relation-Reset|Δpp|
|---|---:|---:|---:|---:|---:|
|adjust_bottle|90|90|+0|90|+0|
|beat_block_hammer|100|100|+0|100|+0|
|handover_block|100|100|+0|90|-10|
|place_bread_skillet|100|100|+0|100|+0|
|place_burger_fries|90|90|+0|90|+0|
|put_bottles_dustbin|100|90|-10|90|-10|
|place_object_basket|90|90|+0|100|+10|
|place_cans_plasticbox|100|100|+0|100|+0|
|stack_blocks_three|100|100|+0|100|+0|
|open_microwave|50|50|+0|70|+20|
|平均（各组100条）|92|91|-1|93|+1|

Randomized：Write-Off相对Full，4任务提高/1降低/5持平；Relation-Reset为5提高/1降低/4持平，不是10任务普遍负面。put_bottles_dustbin和place_object_basket贡献Write-Off净增6条中的5条（83.3%），贡献Relation-Reset净增8条中的6条（75%）。剔除两任务后，剩8任务Full51/80=63.75%、Write-Off52/80=65%、Relation-Reset53/80=66.25%；差异缩至+1.25/+2.5pp。
Write-Off在place_burger_fries/Randomized低10pp；Relation-Reset在stack_blocks_three/Randomized低10pp，Full在这些任务反而更好。Clean总体92/91/93%；两域合计Full154/200=77%、Write-Off159/200=79.5%、Relation-Reset163/200=81.5%。每格10条意味着1条改变10pp；相同seed配对降低场景差异，但不足以证明总体或单任务显著性。所有关分支组都是同一联合训练权重的推理干预，不是原OpenWAM baseline。下一步结合18组配对闭环筛查，再扩大两个主要差异任务及反向任务样本。
