# 集群Python3.10缺少ensurepip：无sudo修复

报错原因：目标系统Python没有python3.10-venv/ensurepip，旧安装入口调用python -m venv失败，可能留下缺少pip和activate的不完整.venv。不是Conda安装，也不是模型/数据问题。

在目标服务器执行以下命令，现有旧安装脚本也可继续使用，无需重新传输模型和数据：

```bash
cd /home/mingfeihangroup/xiaoming/projects/OpenWAM
# 在当前(base)中只安装uv启动工具；训练依赖仍进入项目.venv。
python -m pip install uv
# 保留本次失败产生的环境，不删除。
mv .venv ".venv.ensurepip_failed_$(date +%Y%m%d_%H%M%S)"
python -m uv venv --python "$(command -v python3.10)" --no-python-downloads --seed .venv
PYTHON_BIN=python3.10 bash scripts/cluster/install_environment.sh
```

上述mv针对用户刚刚失败留下的.venv；不要用来移动正在使用的健康训练环境。

修正版scripts/cluster/install_environment.sh改为先寻找UV_BIN/uv；缺uv时给出明确安装指令；对缺python或activate的不完整环境改名备份，用uv venv --seed创建并安装pip，不依赖系统ensurepip。健康环境直接复用；缺pip但有正常activate时只用uv补pip。保留原Torch2.7.1/cu128和requirements-ea-lock.txt安装路径、约束和pip check。不会自动创建Conda环境，不修改模型/数据，也不安装到Conda base的训练依赖。

2026-10-15验证：/usr/bin/python3.10的ensurepip_available=False，隔离临时目录模拟失败环境，真实uv创建Python3.10环境并seed pip成功；原失败目录内容保留、activate/pip可用、UV_BIN交接通过。依赖安装主体使用stub避免再次下载整套Torch；未在目标集群执行实际安装，后续完整依赖/driver兼容仍需目标机器确认。

官方说明：https://docs.astral.sh/uv/pip/environments/


## 2026-10-15：UV_DEFAULT_INDEX 与旧 --index-url 混用修正

UV_DEFAULT_INDEX=清华镜像时，旧安装主体的 --index-url PyTorch参数会被覆盖。保留UV_DEFAULT_INDEX供通用依赖使用，但必须把scripts/install_ea_environment.sh的第一阶段改为 --default-index https://download.pytorch.org/whl/cu128，并显式指定torch==2.7.1+cu128、torchvision==0.22.1+cu128、torchaudio==2.7.1+cu128。旧环境不删除、不清空uv缓存，修改后重跑安装器，安装器替换不匹配的构建。新源脚本已经增加三包metadata版本门槛。此前交付的压缩包未原地改写，目标端须同步此脚本修复。


## 2026-10-15：目标集群 PyTorch 源不可达时使用本地 wheel

目标 download.pytorch.org 与用户测试的交大镜像均连接超时。用户已有 torch2.7.0+cu128 / torchvision0.22.0+cu128 cp310 wheels，与本项目固定2.7.1/0.22.1版本不一致，不直接替换锁文件。scripts/install_ea_environment.sh 新增 OPENWAM_TORCH_WHEEL_DIR：必须提供精确的 torch2.7.1、torchvision0.22.1、torchaudio2.7.1 三个 cu128 cp310/manylinux_2_28_x86_64 wheel。第一阶段直接指定本地wheel，依赖走UV_DEFAULT_INDEX（或UV_INDEX_URL，默认清华镜像）；第二阶段通过--find-links复用这些wheel。三包metadata精确校验与原锁文件保留，缺wheel明确退出，不回退到其他Torch版本。

交付目录 /remote-home/jinminghao/transfer_packages/OpenWAM_torch_cu128_cp310_20261015，完成状态以 WHEEL_VERIFICATION.json status=passed 和最终tar的sha256为准；安装及传输命令见包内README_INSTALL.md。本包只是Torch三件套本地安装补丁，不是全依赖离线包；nvidia运行库和其他Python依赖仍需镜像可达。目标现有.venv与uv缓存均保留，安装器替换版本不匹配的构建。


## NVIDIA/CUDA 大包补充交付

`OpenWAM_cuda_runtime_cu128_cp310_20261015.tar`（2,800,414,720 bytes）位于transfer_packages，包含14个nvidia-*、triton3.3.1及setuptools79.0.1，共16个已验证wheel。与之前的Torch三件套包分别上传、解压，使用本次runtime包内最新版installer；设置OPENWAM_TORCH_WHEEL_DIR为旧Torch包的wheels目录、OPENWAM_CUDA_WHEEL_DIR为新runtime包的wheels目录。installer先--no-index安装完整runtime，再继续本地Torch和普通依赖阶段。独立空Python3.10环境中真实uv --no-index dry-run完整解析16包通过；目标实际安装/训练待用户执行。具体命令见runtime包内README_INSTALL.md。
