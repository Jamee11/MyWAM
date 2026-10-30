# H100 集群使用手册 (test1208)

> 8 节点 × 8×H100-80GB SXM。本手册为共享地基说明,面向所有使用者。
> 凭据(跳板机/节点密码)向管理员索取,勿写入任何共享文件。

---

## 1. 集群概况

| 主机名 | 内网 IP | GPU | 外网 | 备注 |
|--------|---------|-----|------|------|
| aigc01 | 172.16.80.135 | 8×H100 | 否 | |
| aigc14 | 172.16.80.148 | 8×H100 | 否 | |
| aigc15 | 172.16.80.149 | 8×H100 | **是** | 构建节点 |
| aigc24 | 172.16.80.158 | 8×H100 | 否 | |
| aigc28 | 172.16.80.162 | 8×H100 | 否 | |
| aigc29 | 172.16.80.163 | 8×H100 | 否 | |
| aigc31 | 172.16.80.165 | 8×H100 | **是** | 构建节点 |
| aigc32 | 172.16.80.166 | 8×H100 | 否 | |

- 节点内:8 卡 NVSwitch 全互联(NVLink)。节点间:NDR InfiniBand 400G,每 GPU 一张 rail。
- 所有节点驱动 `580.159.03`,Fabric Manager 已启用,单机 8 卡开箱即用。

## 2. 接入

经跳板机 `36.189.234.168:33` 跳转至内网节点。`~/.ssh/config` 示例:

```
Host h100-jump
    HostName 36.189.234.168
    Port 33
    User root
Host aigc15
    HostName 172.16.80.149
    User <你的用户名>
    ProxyJump h100-jump
```

`ssh aigc15` 即可。其余节点同理替换 IP。

## 3. 存储

| 路径 | 类型 | 容量 | 用途 |
|------|------|------|------|
| `/data1`–`/data4` | 本地 NVMe(每盘 3.5T) | ~14T/节点 | 高速 scratch、数据本地缓存、uv cache |
| `/mnt/100T` | 共享 NFS(全节点一致) | 100T | 共享环境、数据集、checkpoint |
| `/test1208` | 共享 NFS | 12T | 既有数据 |

- **NFS 不支持文件锁**:`uv`/HuggingFace 的**缓存必须放本地**(`/data1`);共享盘只放**只读产物**(已建好的 `.venv`、数据集、ckpt)。
- 多用户共享目录请保持 `umask 0002`。

## 4. 初始化环境(一次性配置)

将共享 env 写入你的 shell 启动文件,之后**每次登录自动生效**,无需手动 source:

```bash
echo 'source /mnt/100T/etc/env.sh' >> ~/.bashrc   # 只需执行一次
```

`/mnt/100T/etc/env.sh` 仅设两项,其余沿用 uv 默认:

| 变量 | 值 | 作用 |
|------|-----|------|
| `PATH` | 追加 `/mnt/100T/bin` | 直接使用共享 `uv` |
| `UV_LINK_MODE` | `copy` | 让 uv 复制装包而非硬链(原因见下) |

> **为何必须 `copy`**:uv 默认把包文件从缓存**硬链接(hardlink)**进 venv——又快又省空间。但硬链接只能在**同一个文件系统**内建立。我们的场景里,缓存在本地盘(`~/.cache/uv`,位于 `/`),而 venv 建在共享盘 `/mnt/100T`(NFS),两者**不是同一文件系统**;跨文件系统硬链会报 `EXDEV`(Invalid cross-device link),导致装包失败。设 `UV_LINK_MODE=copy` 改用复制,就能把包从本地缓存正常装进 NFS 上的 venv。
>
> uv 缓存沿用默认 `~/.cache/uv`(本地盘,**勿改到 NFS**,否则文件锁报错);包源默认 pypi(构建节点可达)。

## 5. Python 环境(uv)

取包源(aliyun/pypi)**仅 aigc15 / aigc31 可达**。因此:

**在构建节点(aigc15 或 aigc31)创建环境 → 放共享盘 → 全节点共用。**

```bash
# 在 aigc15 上
source /mnt/100T/etc/env.sh
cd /mnt/100T/users/<你的名字>/<项目>      # 代码与环境放各自目录
uv venv .venv                              # 或 uv sync(若有 pyproject.toml)
uv pip install --python .venv/bin/python <包>
```

其余无外网节点**直接运行同一份 `.venv`,无需联网**:

```bash
/mnt/100T/users/<你的名字>/<项目>/.venv/bin/python train.py
```

> 多机训练时各 rank 必须使用**同一个 `.venv`**(torch/NCCL 版本一致是组队前提)。

## 6. 多机训练(NCCL over InfiniBand)

启动前每个进程设置(已是真 IB,**勿设 `NCCL_IB_GID_INDEX`**):

```bash
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10
export NCCL_SOCKET_IFNAME=enp86s0f0np0   # 控制面走管理网
export NCCL_IB_DISABLE=0
export PYTHONHASHSEED=0                   # 跨机一致(JAX/部分框架必需)
```

torchrun 示例(N 节点,每节点 8 卡,master 取其一的 172.16.80.x):

```bash
torchrun --nnodes=N --node_rank=R --nproc_per_node=8 \
         --master_addr=172.16.80.149 --master_port=29501 train.py
```

参考性能:4 节点 32 卡 all-reduce busbw ≈ 443 Gb/s(默认配置,可调优)。

## 7. 用户使用规范

- **实验流程**:在构建节点(aigc15/aigc31)建好或同步 `.venv`(见 §5)→ 任意节点用同一 `.venv` 运行;多机训练见 §6。
- **代码与环境放置**:代码库 + 对应 `.venv` 放共享盘 `/mnt/100T/users/<名字>/<项目>/`,各节点共用;uv 缓存留本地默认 `~/.cache/uv`(**勿放 NFS**);`HF_HOME` 可指共享盘**仅供离线读**,下载模型需在构建节点本地下好再拷入。
- **目录与文件**:各自工作在 `/mnt/100T/users/<名字>/` 下,ckpt、日志归位,**不读写他人目录**。checkpoint 写共享盘:同一任务由框架协调(通常仅 rank0 写)、不同实验各写各目录,无需文件锁。
- **清理与配额**:共享盘 100T 与本地 `/data` 均为公用,**及时清理旧 ckpt/数据**,勿占满。
- **取包**:仅 aigc15 / aigc31 有外网,装包、建环境在这两台进行。
- **系统层勿动**:不随意 `apt`、改驱动、动 `/etc`,不动他人或其他租户资源。

> ⚠️ **重启**:节点无带外控制台(IPMI)。改动驱动相关包后,先确认 `modinfo -n nvidia` 有输出(内核模块已落盘)再重启,否则可能起不回来。
> ⚠️ **跨节点文件锁**:本集群 NFS 不支持文件锁,需要锁的工具(如 uv/HF 缓存)放本地;确需跨节点锁请联系管理员开 NFSv4。

## 8. 离线资源

`/mnt/100T/` 下备有官方安装包:`NVIDIA-Linux-x86_64-580.159.03.run`(驱动)、`cuda_12.8.1_570.124.06_linux.run`。驱动损坏时可离线重装。
