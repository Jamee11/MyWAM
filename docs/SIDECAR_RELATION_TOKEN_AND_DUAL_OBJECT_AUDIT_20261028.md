# Sidecar Relation token、near/contact与双物体轨道核查

时间：2026-10-28T08:05:09.565615+00:00。仅分析现有代码和3条Clean轨迹，不改训练或标签。

## 共享token的确切含义

每条手臂轨道有一个512维EEF→Object token，由两个独立Linear(512,1)分别预测near和contact；两项用独立BCE，不做二选一softmax，概率无需相加为1。因此near=1/contact=0、near=1/contact=1和near=0/contact=1均可由同一个特征向量表示。共享表征可能存在梯度竞争，但当前Action–EA梯度诊断不是near–contact的分头梯度检查，不能保证没有竞争或据共享本身认定冲突。

## 两个物体如何表示

左轨道：EEF_L→Object_L→Goal_L；右轨道：EEF_R→Object_R→Goal_R。四个Relation tokens分别对应两条轨道的EO/OG，左/右token独立但预测头共享权重。数据中track_object_pose/track_role_id等均有独立的2轨道维度。不同物体、同一物体双手操作和随阶段切换角色均由GT角色关联定义；推理时模型自行从观测预测，不提供这些GT ID。

|任务 / episode0|左/右有效role ID|双轨有效帧|近接标签四种组合计数|
|---|---|---:|---|
|pick_dual_bottles|[[1], [2]]|125|{'near0_contact0': 90, 'near0_contact1': 160, 'near1_contact0': 0, 'near1_contact1': 0}|
|place_dual_shoes|[[1], [2]]|224|{'near0_contact0': 145, 'near0_contact1': 27, 'near1_contact0': 0, 'near1_contact1': 276}|
|handover_block|[[1], [1]]|172|{'near0_contact0': 132, 'near0_contact1': 16, 'near1_contact0': 1, 'near1_contact1': 318}|

双瓶与双鞋分别得到role1/2、不同物体参考轨迹；handover两条有效轨道可共同引用role1，同物体参考位置一致。只是这3条样本的实现检查，不宣称全数据角色分配均正确。

## 新发现：near与contact标签不等价，须审计物理语义

near当前按EEF参考点到物体参考点距离<0.10m生成，contact直接取oracle的track_contact。pick_dual_bottles/episode0全部250个有效轨道时刻near=0，其中160个contact=1；这些contact时刻两参考点距离为0.12281–0.14646m，中位0.12645m。鞋任务也有27个near=0/contact=1，交接任务有16个。这不构成两个独立BCE目标的逻辑矛盾，但若near想表达“表面接近/可接触”，固定物体参考点阈值可能是较弱代理。物体尺寸、抓取位置、EEF工具点/坐标契约都可能影响；本次未独立物理回放，不断言具体原因或数据有/无错误。应先审计参考点与尺度再考虑改监督或拆token，不静默调阈值/标签。

## 当前表达边界

两条轨道可以表达两手各操作一个物体。没有专门的Object_L↔Object_R、跨手↔另一物体或双手同步监督边；它们可通过15-token SelfAttention与动作目标隐式学习。复杂装配、交接、单手同时涉及多个物体，需进一步审计角色分配与关系覆盖，不能把两条轨道等同于任意多物体关系图。

原始数值证据：`/remote-home/jinminghao/OpenWAM/outputs/sidecar/relation_token_dual_object_audit_20261028/audit.json`。代码：`openwam/model/modules/relational_sidecar.py`（两个head及near_threshold）、`openwam/model/modules/ea_rsig_v3.py`（dual-track losses）、`openwam/dataloader/ea_rsig_lite.py`（oracle转标签）。
