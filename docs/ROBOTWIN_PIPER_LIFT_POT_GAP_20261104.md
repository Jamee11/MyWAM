# Piper lift_pot Clean/Randomized 差异核查

核查日期：2026-11-04 UTC。权重：piper_sidecar/midtrain_patchwam_2epochs_20261102T101245Z/runs/2026-11-02_10-18-00/checkpoint_step_21952.safetensors。

结果来源：outputs/robotwin_eval/piper21952_full100_after_patch_20261104/eval/jobs/{demo_clean,demo_randomized}/lift_pot/attempt1；均为完整100条，无视频。

|条件|成功|失败|按顺序每25条成功数|
|---|---:|---:|---|
|Clean|59/100|41|18、13、12、16|
|Randomized|83/100|17|23、23、20、17|

Randomized 高24个百分点。两组均无日志 out of memory；日志包含专家筛选异常，不能将这些 Traceback 数目视为模型失败数。

代码证据：envs/lift_pot.py 两域使用同一锅资产060_kitchenpot、model_id随机0或1，同一成功判据：锅中心z>0.82m、两侧TCP各自距对应接触点<0.03m、pot_dir>0.8。模型失败日志只有成功/失败和步数，没有每个判据的最终数值，无法定位漏抓、滑落或倾斜。

两域不是同一场景的纯背景配对：专家筛选最终100个seed仅19个数字重合；_base_task.py在load_actors之前根据random_light等分支消耗NumPy随机数，因此相同数字seed也不保证锅型号/位姿相同。Randomized还改变背景、杂物、光照及桌高；桌高偏置实际np.random.uniform(-0.03,0)，不是对称±3cm。可能改变抓取几何，但尚未验证影响方向。

解释边界：不能由本结果断言随机化提升模型能力、Clean配置错误、某一锅型号导致失败或具体视觉原因。差异跨多个25条区段存在，不是单个短区段造成；仍有型号、位姿、语言及专家筛选分布混杂。

下一步：保留当前主评估。需要归因时另外小规模诊断：固定锅型号/位姿/指令，分别改变背景、光照、杂物、桌高；额外记录最终锅高、双TCP距离和姿态判据，少量保存失败截图。标准主结果保持原官方配置，不为缩小差距修改成功判据。
