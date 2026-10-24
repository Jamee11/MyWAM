# MyWAM 初始备份的 GitHub Actions 失败分析

2026-10-16 UTC。推送成功、Checkout成功；失败发生在代码检查阶段，不能将备份成功说成CI已通过。

原始备份 `cb37290` 及补齐上游Cosmos gitlink的 `364a95f` 在main触发CI。
`38876634167` 的lint job `116893145104` 报107项：48个I001、26个E702、18个E402、10个F401、4个E701、1个E731。
这些属于此次首次上传的既有研究扩展代码，并非新Sidecar造成。修复仅整理imports、无用import和等价排版，bootstrap后的project import标注具体E402理由；测试中的局部lambda改为等价def。不修改模型/动作/标签数学定义，不跳过lint。

tests job `116893145330` 在收集5个Cosmos3测试时失败：未锁定的Transformers最新版引入 `torch.distributed.checkpoint.hf_storage`，但CI固定的PyTorch2.7.1没有该模块，随后报PreTrainedModel/PEFT导入失败。不能当作某个模型训练数值失败。

新增 `requirements-ci.txt`，仅为CPU CI约束Transformers5.19.0、Diffusers0.41.0和Ruff0.11.9。前两者与本地已用的环境锁一致，Ruff与现有pre-commit一致。训练.venv和生产锁不安装、不升级、不改写；CPU Torch2.7.1仍由原workflow安装。测试照常运行全部make check，不屏蔽Cosmos3。feature/fix分支push也运行CI，避免只检查main却误认为开发分支已验收。

原始备份分支保持；修复在独立fix/ci-portability-20261016工作树/分支。排除import语句后，模型/训练/数据等Python AST全部相等，仅tests/test_ea_lite_startup.py存在上述lambda到def差异。Ruff本地检查已通过，远端测试结果须以新run为准。Docker历史run因新main提交触发concurrency取消与CI失败是两类状态，不能混记。


## 2026-10-16：第二轮远端检查与可迁移性修复

CI run 38878693936：lint通过；标准测试2149 passed / 15 skipped / 21 deselected / 2 failed。两个真实失败为：旧policy_config_ea_rsig_v3.yml中的本服务器绝对FK资产路径，以及旧BF16 fixture对bool调用is_floating_point。前者改为仓库相对路径并由AlohaEEF统一基于__file__定位，绝对用户覆盖仍有效；测试从临时工作目录启动客户端。后者仅在Tensor上转换dtype，不改变模型计算。

Docker run 38876633968的实际CUDA镜像构建通过，在make docker-check的lint阶段失败；与旧107项lint债务相同，不能归因为上传失败或Docker构建失败。feature首次提交未触发原workflow，因此无红叉不等于测试通过。现已让feature/fix也运行标准CI。

原始backup分支不改动；修复提交到fix分支后继续查新run，历史失败记录不会自动变绿。未删除检查或跳过失败测试。
