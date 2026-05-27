# uc01/02/03 集群当前数据共享方式分析

> 2026-05-27 整理. 综合 `submission/uc_cluster_jobs.md` + `storage_and_env.md` + `overview.md` + memory `reference_uc_cluster_nfs_layout.md` (2026-05-18 后重装架构) 推导而出。
>
> **结论速览**: uc 集群是"**双 NFS 平面 + 单机本地高速盘**"混合架构 — 代码/venv 走管理网 NFS, 集群训练 ckpt 走 RDMA 网 NFS, 单机 ckpt + 数据集走本地 ext4(+ symlink trick 绕过 NFS 同步)。

---

## 1. 硬件 / 网络拓扑前置

| 维度 | 值 |
|---|---|
| 节点 | uc01, uc02, uc03 (各 8× A800-80GB) |
| 本地系统盘 | `/dev/vda2` 492 GB ext4 (装 OS + venv + 本机 ckpt) |
| 本地大盘 | `/data/shared` 4 TB ext4 (代码 + 数据集 + 共享 NFS 导出根) |
| 额外 NVMe | uc03 多一块 `/nix` 3.5 TB NVMe |
| 管理 / 公网 | `eth0` 10.60.x.x/16 (virtio_net, MTU 1452, 慢) |
| RDMA 训练网 | `eth1-4` 192.168.{1-4}.x/24 (mlx5_core, 200 Gbps × 4, RoCEv2, MTU 4200) |

主机间 RDMA 平面 IP:
```
uc01: 192.168.{1,2,3,4}.2
uc02: 192.168.{1,2,3,4}.3
uc03: 192.168.{1,2,3,4}.4
```

---

## 2. 数据共享平面总览 (3 类共存)

uc 集群同时存在 **3 套独立的数据放置策略**, 各自服务不同场景:

| # | 共享方式 | server / mount | 内容 | 谁写 / 谁读 | 网卡 |
|---|---|---|---|---|---|
| **A** | **NFS over 管理网** | `uc01:/data/shared/ubuntu/workspace` → uc02/03 同路径 | **代码 + `kai0/.venv` + `base_init_ckpts/`** | uc01 改, uc02/03 自动可见 | eth0 (10.60.0.0/16) |
| **B** | **NFS over RDMA 网** | `uc01:/data/cluster_ckpt` (`192.168.1.2`) → uc02/03 `/cluster_ckpt` | **3-host 集群训练时的 Orbax ckpt + (可选) 大数据集** | 3 host 同时写/读 (Orbax) | eth1 (200 Gbps RoCE) |
| **C** | **不共享 / 本地独立** | 各机本地 `/data/shared/*` 真实目录 (NFS 导出范围之外的部分) | **数据集 (KAI0 + Kai0_official) + 单机训练 ckpt + 单机 `/home/tim/local_ckpts/`** | 各机自管, 跨机不可见 | (本地 ext4) |

> ⚠️ 一个易混点: A 的 NFS export 只是 `/data/shared/ubuntu/workspace` **这一个子目录**, **不是整个 `/data/shared/`**. 这是 (C) 能与 (A) 共存的关键 — 同一物理盘上, 一部分被 export, 一部分仅本机可见。

---

## 3. NFS 平面 A: 代码 + venv 共享 (管理网)

### 3.1 配置

**uc01 (server)**:
```
/etc/exports:
  /data/shared/ubuntu/workspace  10.60.0.0/16(rw,sync,no_subtree_check,no_root_squash)
```

**uc02 / uc03 (client) `/etc/fstab`** (单行):
```
10.60.135.47:/data/shared/ubuntu/workspace  /data/shared/ubuntu/workspace  nfs  defaults  0 0
```

> uc01 内网 IP = `10.60.135.47` (走 eth0 virtio).

### 3.2 共享的内容

```
/data/shared/ubuntu/workspace/
├── deepdive_kai0/                     ← 主代码仓 (git)
│   └── kai0/
│       ├── .venv/                     ← Python 3.12 (uv 管理) - 8.2 GB
│       ├── src/, scripts/, ...
│       └── checkpoints/ → symlink     ← 见 §5
└── base_init_ckpts/                   ← 共享 init ckpt
    ├── pi05_base/                     ← 13 GB, 官方 pi05
    └── Task_A_mixed_1/                ← 22 GB, 从 TOS shared_ckpt 拉回
```

### 3.3 工作目录路径与软链

各机 `/home/ubuntu/workspace/deepdive_kai0/` 是软链 → `/data/shared/ubuntu/workspace/deepdive_kai0/` (2026-05-18 重装后). `~/workspace` 也类似指向同一处。

> 在 uc01 上 `uv add` / `uv sync` 改 `.venv` **一次**, uc02/03 立即可用, 无需逐机重装。但 `uv python install 3.12` 仍需每机一次 (Python interpreter 在 `~/.local/share/uv/` 本地路径)。

### 3.4 性能 / 瓶颈

- 管理网 eth0 是 virtio + MTU 1452, 慢 — 但代码/venv 是冷读, NFS cache 命中后即本地速度, 不构成训练瓶颈。
- 不适合高频 I/O (训练 ckpt write / dataloader 大批读), 这些走平面 B 或 C。

---

## 4. NFS 平面 B: 集群训练 ckpt 共享 (RDMA 网)

### 4.1 配置

**uc01 (server)**:
```
/etc/exports:
  /data/cluster_ckpt  192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash)
```

**uc02 / uc03 (client) `/etc/fstab`**:
```
192.168.1.2:/data/cluster_ckpt  /cluster_ckpt  nfs  vers=4,hard,intr,timeo=600,rsize=1048576,wsize=1048576
```

### 4.2 用途 (仅 24-GPU HSDP/FSDP 集群训练)

来源: `submission/uc_cluster_jobs.md §12.5`

- **Orbax CheckpointManager 跨主机一致性**: Orbax 用 POSIX 元数据 (file create / rename / fsync) 做 3-host barrier, 必须共享文件系统。
- **(可选) 大数据集**: 测过 ~115 GB 训练数据集放 NFS, GPU 99% util, NFS 不是瓶颈。
- **`config.py` 同步**: ⚠️ 但是 `config.py` 本身 **不** 走 NFS — 3 host 各自从本地 venv 内 import, 改完要 `scp` 推一遍 (见 §12.7)。

### 4.3 性能基线

| 操作 | 实测 |
|---|---|
| write | ~219 MB/s (单 stream NFSv4 over TCP/RDMA) |
| read | ~2 GB/s |
| 网卡 | 跨 host 直传走 RoCE eth1 (192.168.1.x), 不走 eth0 |

### 4.4 必须知道的陷阱 (uc_cluster_jobs.md §12.8 陷阱 C)

**Orbax `sync_global_devices ... save_root_metadata` 名字不匹配**:
- 症状: `AssertionError: sync_global_devices name mismatch ('CheckpointManager:save_root_metadata') Expected: X; got: Y`
- 原因: NFS 元数据 stale + 上次启动残留 → 3 host `os.listdir()` 看到不同内容
- 修复: 启动前 `rm -rf $checkpoint_dir; sync; sleep 1`

---

## 5. 本地高速盘 + Symlink Trick: 单机训练 ckpt

### 5.1 设计动机

用户原话: "节约 NFS 磁盘空间 + 单机训练写 ckpt 不应走 NFS (太慢, 也无谓占共享空间)"。但 openpi 默认把 ckpt 写到 `$KAI0_DATA_ROOT/checkpoints/<config>/<exp>/`, 而 `$KAI0_DATA_ROOT = $HOME/workspace/deepdive_kai0/kai0` 在 NFS 上 — 直接写会走 NFS。

### 5.2 解法: NFS 上放 symlink 字符串, 各机 client 自己解析

```
NFS 内容 (uc01 server):
  /data/shared/ubuntu/workspace/deepdive_kai0/kai0/checkpoints
    → symlink string "/data/shared/ubuntu/local_ckpts"

各 host 解析这个 symlink:
  uc01: /data/shared/ubuntu/local_ckpts → 本机 /dev/vdb (本地 SSD)
  uc02: /data/shared/ubuntu/local_ckpts → 本机 /dev/vdb (本地 SSD, 互不冲突)
  uc03: /data/shared/ubuntu/local_ckpts → 本机 /dev/vdb (本地 SSD, 互不冲突)
```

> 关键: NFS 只 export 了 `workspace/` 子目录, **没 export 整个 `ubuntu/`**, 所以同名兄弟目录 `local_ckpts/` 不被 NFS 覆盖, 各机看到自己的本地版本。这是用 symlink 字符串"穿透 NFS"的 trick。

### 5.3 实际 launcher 用法

```bash
CONFIG=pi05_flatten_fold_<your_config>
EXP=<your_exp_name>
LOCAL_DIR=/home/tim/local_ckpts/$CONFIG/$EXP   # 真实路径在本机 /dev/vda2
WORKSPACE_DIR=$KAI0_DATA_ROOT/checkpoints/$CONFIG/$EXP

mkdir -p "$LOCAL_DIR"
mkdir -p "$(dirname "$WORKSPACE_DIR")"
ln -sfn "$LOCAL_DIR" "$WORKSPACE_DIR"     # per-exp 软链
.venv/bin/python scripts/train.py $CONFIG --exp_name=$EXP
```

### 5.4 `/home/tim/local_ckpts` 实现 (各机)

| 机器 | 实现 | 物理后端 | 可用 |
|---|---|---|---|
| uc01 | 真实 dir | `/dev/vda2` (492G ext4) | ~290 G |
| uc02 | 真实 dir | `/dev/vda2` | ~410 G |
| uc03 | 真实 dir | `/dev/vda2` | (待测) |

---

## 6. 数据集存储 (按机器各自一份, 不共享)

```
各机本地真实目录 (不在 NFS export 范围内):
  /data/shared/dataset/KAI0/Task_<X>/base/        ← 自建, rsync from vePFS
  /data/shared/dataset/Kai0_official/Task_A/      ← HF 官方 base/dagger/advantage
~/workspace/deepdive_kai0/kai0/data/Task_<X>/    ← symlinks 指向上面 (在 NFS 上, 但解析到本地)
```

- 3 机各保一份, 跨机不可见。
- 大数据集 (>100 GB) 合并/构建: `hardlink mp4` (同分区秒级) + `rewrite parquet` (规整 schema)。
- `/data/shared/ubuntu_old/workspace` 是 2026-05-18 重装前的老数据 — uc02 770 G (含 pure_1800_mixed1 SOTA), uc03 1.7 T (含 smooth_800 等)。

---

## 7. 跨集群 (uc ↔ 火山 ↔ 本地) 数据流

uc 与 vePFS / Robot-North-H20 / robot-task 队列 **不直接共享 FS**, 跨集群同步走两条路:

```
            公网 SSH / rsync                  TOS 跨 region 骨干
[uc02 数据] ──────────────► [gf0 vePFS] ◄─────────────► [gf3 vePFS-North-E]
                              │                              │
                              │ 同 vePFS 自动可见            │ 同 vePFS-North-E 自动可见
                              ▼                              ▼
                       robot-task (cnsh)              Robot-North-H20 (cnbj)
                       火山 28 A100                   火山 56 H20
```

- **uc → 火山-cnsh**: `rsync uc02:/data/shared/... gf0:/vePFS/.../`
- **uc → 火山-cnbj**: 中转 TOS — `tosutil cp /vePFS → tos://transfer-shanghai/` 再 `gf3 tosutil cp → /vePFS-North-E/`
- **火山 ckpt → 真机部署**: 反向, 同样经 TOS

---

## 8. 共享方式选择决策树

```
要存什么?
├── 代码 / venv / 共享 init ckpt
│   └── 改 uc01 即可 → NFS 平面 A 自动同步 (eth0 管理网)
│
├── 3-host HSDP/FSDP 集群训练的 Orbax ckpt
│   └── 写到 /cluster_ckpt/... → NFS 平面 B (eth1 RDMA, ~2 GB/s read)
│
├── 单机训练 ckpt
│   └── 写到 /home/tim/local_ckpts/... + per-exp symlink trick
│      → 走本机 /dev/vda2 (不占 NFS, 互不冲突)
│
├── 数据集 (>5 GB, 持续读)
│   └── /data/shared/dataset/... (本地, 各机各一份)
│      或 (集群训练时) /cluster_ckpt/dataset/... (NFS-B, 实测 GPU 99% util)
│
└── 跨集群 (uc → 火山)
    └── rsync 公网 → gf0:/vePFS (cnsh) 或 TOS 中转 → gf3:/vePFS-North-E (cnbj)
```

---

## 9. 重要注意事项 / 历史"踩坑"

1. **NFS export 范围限定**: 只 `workspace/` 一个子目录, 不是 `ubuntu/` 整体 — 这是 §5 symlink trick 能工作的前提。改 export 范围前请评估对单机 ckpt 的影响。
2. **`config.py` 改完必 scp**: 集群训练时 3 host 各自从 venv 内 import, 不走 NFS 自动同步 (因 `config.py` 在 `.venv` 之外的 `src/openpi/` 路径但 worker 是从本机 NFS-mounted 路径读, 实际**是会走 NFS 的** — 但 NFS 缓存可能延迟, 仍建议显式 scp + verify; 见 `uc_cluster_jobs.md §12.7`)。
3. **千万不要直接写 ckpt 到 `<kai0>/checkpoints/<config>/<exp>` 真实目录**: 那是 NFS 路径, 旧版有 lsyncd 双向 mirror 时会导致损坏, 现在虽然 lsyncd 已停, 但占 NFS 空间 + 慢。一律走 symlink trick。
4. **SSH 密码登录已禁用** (cloud-init 50-cloud-init.conf), 唯一登录 = pubkey。运维加新人要先放 pubkey。
5. **uc01 是 SPOF**: 平面 A + 平面 B 都以 uc01 为 server, uc01 down 会同时影响 uc02/03 的 NFS — 重启 uc01 前请先停 uc02/03 上的训练。

---

## 10. 验证当前架构是否仍生效 (建议运行)

文档与 memory 引用最新事件是 2026-05-18 (重装). 本分析基于这些静态信息推导, 建议在使用前在 uc01 上跑一遍快速检查:

```bash
# 在 uc01 上:
cat /etc/exports                                  # 验 NFS export 列表
showmount -e localhost                            # 验当前活跃 export
ssh uc02 'mount | grep -E "nfs|cluster_ckpt"'     # 验 client mount 是否仍生效
ssh uc02 'ls -ld /data/shared/ubuntu/local_ckpts' # 验 symlink trick 仍指向本地

# 在所有 3 机上对比同一文件 (验代码 NFS 同步):
for h in uc01 uc02 uc03; do
  ssh $h "stat -c '%i %s %y' /data/shared/ubuntu/workspace/deepdive_kai0/kai0/scripts/train.py"
done
# inode 应该一样 (同一 NFS 文件)

# 验 RDMA 网 NFS:
ssh uc02 "df -h /cluster_ckpt 2>&1 | head"
```

如有不符, 以现状为准并更新本文件。
