# 32k Sidecar推理消融：三个任务、两种场景

2026-10-19 UTC。用户要求以同一32k权重对比完整EA、完全关闭和保留部分功能。当前为实现与待执行方案，**没有本次消融的真实成功率**。本机GPU余量不足一份模型，未启动GPU实验，也没有设置自动抢卡任务。

## 实验回答什么

这是对同一训练后模型的推理干预，回答“当前策略有多依赖EA的在线写入，以及早期/晚期写入各有什么作用”。关闭EA后，主干仍是与EA共同训练后的32k主干，不能把它当作独立训练的原版baseline，也不能据效果下降证明架构天然更优；下降可能同时包含共适应与推理分布变化。

源码确认：`RelationalSidecarArchitecture.forward`在零起始block 7/15/23之后读取当前帧特征、更新15个token、向动作hidden states写入残差。EA heads的ground/relation/plan预测只用于辅助监督，不直接作为动作输入。只屏蔽辅助head不会切断EA对动作的影响；只清零初始token也会被后续read/context/refine重新填充。

## 四组固定干预

| 模式 | 三处关系读取/更新 | 写入动作的block | 含义 |
|---|---|---|---|
| full | 全部保留 | 7、15、23 | 同32k权重的完整对照 |
| off | 全部跳过 | 无 | 完全不执行Sidecar，走现有主干forward |
| first_write | 全部保留 | 仅7 | 保留早期写入，去掉后续两处写入 |
| last_write | 全部保留 | 仅23 | 保留多层关系提取，最后写入一次 |

局部组屏蔽的是write调用，直接返回原动作hidden states；不改beta/其他参数、不写checkpoint。off使用已有`sidecar_enabled=False`路径。局部组不是“只计算第一/最后一层”，更不能称作独立grounding-only或plan-only能力。15个token在refine中有跨角色混合，单独mask plan token的write attention仍可能残留其他token已吸收的plan信息，因此本轮不把这类mask包装成语义功能完全解耦。

新增实现仅在`benchmarks/robotwin/sidecar_ablation.py`、`serve_sidecar_ablation.py`中，由独立评测进程显式调用。训练、模型源码、默认部署入口均不改。一个进程固定一种模式，禁止运行中切换；编译重置时保留局部write屏蔽，避免validator恢复完整分支。

## 任务、样本与公平性

任务固定为adjust_bottle、place_shoe、beat_block_hammer；各跑demo_clean和demo_randomized，共6个条件。四组正式每条件100条，共2400次rollout。预先为每个条件封存100条专家可完成episode的manifest，全部组复用相同manifest hash、seed、unseen指令、logical episode id。先跑统一前20条检查链路并给探索性结果，再扩展到预定100条；不根据早期胜负筛任务/seed或停止差组。

复用现有`eval_policy_wrapper.py labtasker --operation build_manifest/run_eval`。manifest协议包含固定指令与CUDA RNG初始化，和此前native逐seed循环须分开报告；四组的full也重新跑，不能拿此前的native部分成绩直接当配对对照。所有manifest均保留专家筛选、RT32spp/depth8/oidn、无planner fallback和原动作预算；10步联合去噪、32步动作、CFG=1、wxyz、自有normalization stats、同分辨率/相机/仿真版本保持一致。

**主要消融四组统一关闭DiT cache**，隔离分支干预，避免不同模式的中间状态改变自适应cache跳步轨迹。FFN编译在有Sidecar的三组使用同一策略；off分支没有FFN实际执行，不能要求它生成编译kernel或把它算作编译失败。这轮full不等于此前cache开启的部署成绩。若后续需要衡量实际加速部署下的表现，再将全部组统一开启cache=.99/max3，单独报告，不混合两种设置。缓存不会跨generate或跨组复用。

## 验收与结果解释

CPU使用真实tiny MoT/Sidecar测试：非零write gate；各组与对应显式关写入参考逐元素一致；仅调用指定write；off不调用任一Sidecar stage；state_dict全部tensor不变；拒绝未知模式/训练态/重复配置；compile设置切换不能恢复屏蔽write。测试入口：

```bash
.venv/bin/python -m pytest tests/benchmarks/test_sidecar_ablation.py -q
```

真实32k每组仍须两真实RGB fixture、eager/FFN与各自cache参考检查，atol=.005/rtol=.01不放宽，有Sidecar者须真实编译kernel。off须四个对应数值检查通过且无分支FFN kernel，原checker的“无kernel”记录保留，由专用ablation_validation.json明确验收原因；不能将off称作FFN编译已执行。部署恢复配置中的cache关闭，不能继承validator临时打开的cache。

正式报告各条件的成功数/实际完成数、95%区间、逐seed配对的full成功而消融失败/反向计数及精确McNemar检验，同时给动作生成warm延迟与整episode墙钟时间。多组/多条件比较应报告多重比较处理；前20条仅探索，不作显著性结论。OOM、渲染、进程被终止归为基础设施不完整，不能算模型任务失败；完整清单覆盖且exit0、无异常后才形成正式SR。

若full明显优于off，说明当前模型需要在线Sidecar；若last_write接近full，提示三层读取后的末端注入可能足够，下一步才考虑独立训练/结构精简；若off不差或更好，应检查write gate、残差幅度与训练共适应，不能直接判定全部EA监督无用。first与last的比较反映位置/后续主干传播差异，不等同于早期grounding对晚期plan的纯能力比较。

## 固定产物与启动接口

准备目录：`outputs/robotwin_eval/sidecar32000_ablation_20261019`，包含plan.json及四组deploy.yaml。权重直接只读引用已固定32k快照，不再复制24GB。SHA256：`0f22e7954ac5278e0e2184feea65c698754e4efad6b9fc9e317399e14f898de1`。任务runtime和结果须放本目录，不能覆盖旧32k或另一台L20评测。

卡至少有42GB空闲时可启动一组模型并给仿真保留空间；每次启动前重新查询容量，不停止其他作业。下面是full服务的明确接口，其他组换mode目录并使用不同空闲端口：

```bash
CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
TORCHINDUCTOR_CACHE_DIR=/remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_ablation_20261019/full/inductor \
.venv/bin/python -u benchmarks/robotwin/serve_sidecar_ablation.py \
  --ckpt-dir /remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_gpu56_20261019T021700Z/checkpoint_step_32000 \
  --config /remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_ablation_20261019/full/deploy.yaml \
  --port 21700 --slots 2 \
  --manifest /remote-home/jinminghao/OpenWAM/outputs/robotwin_eval/sidecar32000_ablation_20261019/full/pool.json \
  --readiness-token sidecar32k-ablation-full
```

上面仅启动服务，不自动启动仿真。A100仿真复用此前私有sm80 RoboTwin及process-only gpucomp预加载修复；通用TORCH_EXTENSIONS_DIR另设本实验/sm80私有路径，避免与L20共用。manifest builder/evaluator各自独立runtime，ffmpeg选系统已验收版本。执行前还需确认当前显存、每组真实GPU数值门槛及manifest闭环通过。


## 32k权重的只读门值检查

直接读取固定safetensors的六个标量（未加载完整模型、未修改文件），block7/15/23的read alpha分别为0.0327148/0.0996094/0.0410156，write beta分别为0.000143051/0.000337601/0.0000491142，均非零。完整值见准备目录checkpoint_gates.json。write gate很小，但残差还取决于attention/projection输出及hidden state尺度，不能只根据beta断定EA无作用或推断各层贡献。正式离线/闭环需检查动作差异与实际成功率。


### 2026-10-19T08:28:57.819433+00:00：消融CPU验收完成

`tests/benchmarks/test_sidecar_ablation.py`最终6项通过（150.22秒，包含共享盘导入等待），实际tiny模型覆盖四组输出、write/stage调用、权重不变、配置guard及编译重置不恢复write。两个运行时入口语法检查、git diff --check通过。四组cache关闭配置检查通过。尚未加载真实32k模型做本次消融GPU数值检查、未构建本次固定专家manifest、没有闭环成功率；需可用GPU资源后执行。
