# 多仿真midtrain：GitHub CI失败与修复

时间：2026-10-24T20:50:14.604681+00:00。代码上传成功：远端分支feature/multisim-sidecar-midtrain已回读确认a9e282c6；[失败的是上传后CI](https://github.com/Jamee11/MyWAM/actions/runs/40118208693)。原版本备份分支保持不变。

| 原因 | 修复 |
|---|---|
| 3个HTTP测试：No module named flask | Flask加入dev依赖，CI安装.[dev]即可获得 |
| 2个队列测试：历史outputs/final_record.json不存在 | 测试自行生成临时封存历史文件与正确SHA，继续使用真实load_plan校验 |
| 375个lint问题 | 格式/导入整理；pytest fixture显式注册；保留ManiSkill环境注册的side-effect import；未全局忽略lint |

修改前37个文件完整备份及哈希、计算AST比较证据：outputs/midtrain_audit_20261024/ci_repair。未改模型、梯度、损失或数据构造计算。转换脚本在全量构建结束后才格式整理，原始构造版本保存在不可变a9e282c6代码树，数据BUILD_SOURCE.json可核验。

本地全仓CPU：2221 passed、18 skipped、21 deselected；同CI版本ruff0.11.9全仓通过；Python3.10语法检查通过。远端修复后CI结果待更新，不将本地通过当作远端已通过。日志：ci_repair/tests_full.log、lint_ci_pinned.log。

## 远端完成结果：2026-10-24T20:57:51.687499+00:00

修复提交`7a8022eeb3d894f183cca1d0cef3433eb8e84e34`的[GitHub Actions](https://github.com/Jamee11/MyWAM/actions/runs/40131709993)中lint/tests均success；远端2265 passed、16 skipped、21 deselected。代码上传与CI均已确认成功，未通过跳过失败测试或放宽全局检查来获得绿灯。远端代码树7b75d08a与本地提交e9d3dd9d的树完全一致。
