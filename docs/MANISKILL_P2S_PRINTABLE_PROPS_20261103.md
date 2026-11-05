# ManiSkill任务道具打印交付


### 2026-11-03T04:53:10.919000+00:00：用户授权ManiSkill八任务实物打印道具（P2SC按P2S Combo）

只新增独立制造文件/生成器/检查，不改训练模型、动作、仿真或原始数据。按归档Franka20K/Hard10K源码与八任务真实test HDF5交叉核查几何：40mm cube、50×30×36mm cuboid、D40sphere、240×50×50mm双色方peg、50×50×15mm槽（40方口/5底/5壁/10深）；保留灰色stack底块。13封闭STL/14geometry3MF（明确毫米、双色peg部件）、两种平面marker、外观PDF/PNG和打印说明。四种hard障碍从实际seed/index重建并核对HDF5 extents一致。

STL逐个重载单连通/封闭/绕向/正体积/尺寸(.002mm)通过；14个3MF由trimesh重新读入维度通过；4项槽截面/球体体积/3MF单位组装检查通过。生成器assert与证据留在assets/printable_maniskill_p2s_20261103；交付ZIP /remote-home/jinminghao/OpenWAM/assets/ManiSkill_P2S_Props_20261103.zip，SHA256 e108b98f73bfb812420f5f451016f1d82b7166f2fef914ced2a7b9b6668f4ae6。

新实物边界：P2S机型/256成型及0.4默认来自官方规格，用户未确认耗材/是否改喷嘴，PLA仅暂定；未实际Bambu Studio切片或打印，球体支撑/分半胶缝、peg高件/双色与接头强度仍需制造验收。仿真cube64g/cuboid54g/球33.51g/peg600g与PLA填充不同，摩擦.3也不能保证；不声称实物动力学/真机控制成功。Push/Pull目标用薄平面SVG而非厚盘，Pick空中无碰撞goal不可用物理支架/球直接替代，peg立起不添加插孔。暂不改几何做配重孔/装配销，避免叠加未验证接触变化。

详细打印说明：[README](../assets/printable_maniskill_p2s_20261103/README_打印说明.md)。外观总览：[overview.png](../assets/printable_maniskill_p2s_20261103/overview.png)。STL/3MF及所有证据和SHA256SUMS均在上述目录。物理接触与重量需称量/测量，不将制造文件检查当真机验证。


### 2026-11-03T07:27:28.640605+00:00：绿色目标改为可夹持中空口＋PushCube打印标记（用户授权）

用户确认36mm开口候选：D50球面/1.6法向壁厚/内球D46.8，上方截去球冠，口内36mm/外40.072mm，成品H39.952mm；保留球心z25而非包围盒中心。正放与口朝下打印几何STL/3MF、绿色；无凸把手/支架，口背向相机减少显示改变但腕相机/姿态仍不等价。Push/Pull按原radius100mm提供D200H.6齐平红白5环3MF、同尺寸单片STL及内径200/外210/H1.2可移除画圈模板；原SVG/所有旧制造文件保留。未修改仿真/训练。

6项口径/解析壳体体积/薄盘环区分解/模板截面检查通过，逐STL与3MF重载通过，preview已目视；交付/remote-home/jinminghao/OpenWAM/assets/ManiSkill_HollowGreenGoal_PushTarget_v2_20261103.zip。尚未Bambu Studio切片/实物/夹爪适配，名义孔径与真实孔差、薄壳强度、支撑清理未决；原目标无碰撞，实体球不可占机器人最终goal空间；.6mm薄片仍改变推拉台阶/摩擦，优先控制动力学时用画圈后移除模板或薄贴纸，不冒称打印物理等价。
