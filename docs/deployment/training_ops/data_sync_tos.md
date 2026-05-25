# 数据同步与 Ckpt 部署 (TOS 枢纽)

> TOS 中心枢纽 / sim01 上传 / 训练服务器拉取 / 跨服务器 sync / ckpt 训练→sim01 部署工作流。
>
> **同 series**: `overview.md` / `storage_and_env.md` / `ssh_and_credentials.md` / `submission/`

---

## 6. 机器间数据同步 ⭐ (2026-05-21 重构: TOS 为中心枢纽)

### 6.0 总架构: sim01 是源, TOS 是枢纽, 训练服务器都从 TOS 拉

```
                 ┌─────────────────────────┐
                 │   Real Robot 真机采集     │
                 │   (摇操 / DAgger / 部署)   │
                 └─────────────┬───────────┘
                               │ direct write
                               ▼
                 ┌─────────────────────────┐
                 │   sim01 (数据中心)         │
                 │   /data1/DATA_IMP/KAI0/   │
                 │   (raw → cleaned → ready) │
                 └─────────────┬───────────┘
                               │ tosutil upload (~85 MB/s 公网)
                               ▼
        ┌─────────────────────────────────────────────────┐
        │  TOS bucket (cn-shanghai, 中心枢纽 ⭐)            │
        │  tos://transfer-shanghai/KAI0/                  │
        │    ├── dataset/  (训练数据集 canonical 副本)       │
        │    ├── checkpoints/  (各 exp ckpt tar)           │
        │    └── base_init_ckpts/  (pi05_base 等基础模型)   │
        └────────┬────────┬────────┬────────┬─────────────┘
                 │        │        │        │
       pull ▼   ▼   pull ▼   pull ▼   pull ▼  (tosutil download)
              gf0      gf3     uc01    uc02/uc03
              │        │        │       │
        /vePFS/   /vePFS-North-E/  /data/shared/.../kai0/data/
        (cnsh)   /vis_robot/      (各 uc 本地 4TB ext4)
                 (cnbj)
```

**核心原则**:
1. **sim01 = 数据 single source of truth** — 真机数据先到 sim01, 清洗/打包/审核后上传 TOS
2. **TOS = 唯一权威副本** — 所有训练服务器都**从 TOS 拉**, 不互相 P2P 传输
3. **训练服务器是 TOS 的镜像消费者** — 本地 path 与 TOS 路径**保持一致**, 通过定期 resync 保证一致
4. **gf0/gf3 的 vePFS 与火山队列绑定** — 数据落地后 volc job 自动可见 (无需再次传输)
5. **uc 集群本地 SSD** — 拉 TOS 后保留本地, 用于训练 IO 加速

**反模式 (禁止)**:
- ❌ uc02 → uc03 直接 rsync 数据 (应都从 TOS 拉, 保持权威副本)
- ❌ gf0 → gf3 直接传 (走 TOS, 否则破坏中心化)
- ❌ 真机数据先到 gf0 不到 sim01 (绕开 single source)

### 6.1 sim01 数据上传 TOS (数据进入枢纽)

**前置**: sim01 上有 `tosutil` CLI + AK/SK (data_manager 后端常驻配置)。

**单文件上传**:
```bash
ssh sim01
cd /data1/DATA_IMP/KAI0/dataset/Task_A/cleaned/A_new_smooth_800/
tosutil cp -r ./base tos://transfer-shanghai/KAI0/dataset/Task_A/A_new_smooth_800/base
tosutil cp -r ./val  tos://transfer-shanghai/KAI0/dataset/Task_A/A_new_smooth_800/val
```

**整集打包上传 (推荐, 减少小文件 overhead)**:
```bash
# tar 后上传 (大块串行更快)
cd /data1/DATA_IMP/KAI0/dataset/Task_A/
tar -cf - A_new_smooth_800/ | tosutil cp - tos://transfer-shanghai/KAI0/dataset/Task_A/A_new_smooth_800.tar
```

**数据集 manifest** (`sim01:/data1/DATA_IMP/KAI0/MANIFEST.md`):
- 每次上传后更新 manifest 记录: dataset name / 上传日期 / 总大小 / ep 数 / 校验 hash
- TOS 上同名旧版本被覆盖 → manifest 标记版本递进 (`v1`, `v2`, ...)

### 6.2 训练服务器从 TOS 拉数据 (镜像消费)

各服务器 KAI0 数据路径**与 TOS 路径一一对应**, 拉取流程:

| 训练服务器 | KAI0 本地路径 (mirror) | 拉取命令 |
|---|---|---|
| **gf0** | `/vePFS/tim/data/KAI0/...` | `cd /vePFS/tim/data/KAI0 && tosutil cp -r tos://transfer-shanghai/KAI0/dataset/Task_A/<dataset>/ ./dataset/Task_A/` |
| **gf3** | `/vePFS-North-E/vis_robot/dataset/KAI0/...` | `cd /vePFS-North-E/vis_robot/dataset/KAI0 && tosutil cp -r tos://transfer-shanghai/KAI0/dataset/Task_A/<dataset>/ ./dataset/Task_A/` |
| **uc01/02/03** | `/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/<dataset>/` | `cd /data/shared/.../kai0/data && tosutil cp -r tos://transfer-shanghai/KAI0/dataset/Task_A/<dataset>/ ./Task_A/` |

**关键: 路径前缀对齐**:
```
TOS:  tos://transfer-shanghai/KAI0/dataset/Task_A/A_new_smooth_800/
            │                       │
            └── 子路径与服务器本地  ─┴── 完全一致
            
gf0:  /vePFS/tim/data/KAI0/dataset/Task_A/A_new_smooth_800/
gf3:  /vePFS-North-E/vis_robot/dataset/KAI0/dataset/Task_A/A_new_smooth_800/
ucNN: /data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/A_new_smooth_800/
```

### 6.3 跨服务器 sync 工作流 (经 gf0 统一发起)

由于 gf0 是统一控制平面 (§5.6.c), TOS sync 命令也从 gf0 发起:

```bash
# gf0 → 火山 / uc 上的数据 sync 都通过 gf0 ssh + 本机 tosutil
ssh gf0 'bash -s' <<'EOF'
# 1. 通知 sim01 准备好数据后, gf0 触发各服务器拉取
DATASET=A_new_smooth_800

# gf0 自己拉 (cnsh vepfs)
cd /vePFS/tim/data/KAI0/dataset/Task_A
tosutil cp -r tos://transfer-shanghai/KAI0/dataset/Task_A/$DATASET ./

# gf3 拉 (cnbj vepfs)
ssh gf3 "cd /vePFS-North-E/vis_robot/dataset/KAI0/dataset/Task_A && \
  tosutil cp -r tos://transfer-shanghai/KAI0/dataset/Task_A/$DATASET ./"

# 各 uc 拉 (uc 本地 SSD)
for u in uc01 uc02 uc03; do
  ssh $u "cd /data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A && \
    tosutil cp -r tos://transfer-shanghai/KAI0/dataset/Task_A/$DATASET ./"
done

# 验证 (各服务器抽样查目录大小 + ep 数)
for h in gf0 gf3 uc01 uc02 uc03; do
  P=$(case $h in
    gf0) echo /vePFS/tim/data/KAI0/dataset/Task_A/$DATASET;;
    gf3) echo /vePFS-North-E/vis_robot/dataset/KAI0/dataset/Task_A/$DATASET;;
    *)   echo /data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/$DATASET;;
  esac)
  echo -n "$h: "; [ "$h" = "gf0" ] && du -sh "$P" 2>/dev/null || \
    ssh $h "du -sh $P 2>/dev/null"
done
EOF
```

### 6.4 Ckpt 回流 (训练完成 → 经 TOS → sim01 部署)

反向链路 (训练产物回到 sim01 真机部署):

```
[训练服务器 ckpt] ──tosutil──→ TOS  ──tosutil──→ sim01 [部署]
   gf0 (vepfs-cnsh)
   gf3 (vepfs-cnbj)
   ucNN (本地)
```

**统一通过 gf0 调度**:
```bash
# 步骤 1: 训练完成后, 在训练所在服务器打包 + 上传 TOS
# 例如 Robot-North-H20 ckpt 在 gf3 vePFS:
ssh gf0 'ssh gf3 "
  cd /vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/checkpoints/<config>/<exp> && \
  tar -cf - <best_step>/ | tosutil cp - tos://transfer-shanghai/KAI0/checkpoints/<exp>_step<N>.tar
"'

# 步骤 2: sim01 从 TOS 拉
ssh sim01 'cd /data1/DATA_IMP/checkpoints && \
  tosutil cp tos://transfer-shanghai/KAI0/checkpoints/<exp>_step<N>.tar ./ && \
  tar -xf <exp>_step<N>.tar'
```

### 6.5 TOS bucket 目录约定 ⭐

#### 6.5.A 顶层 prefix (按用途分)

```
tos://transfer-shanghai/
├── KAI0/                         ⭐ 主数据 prefix (大写, canonical)
├── kai0/checkpoints/             ckpt 落档 (小写)
├── ckpt/                         早期 ckpt 中转
├── shared_ckpt/                  跨用户共享 ckpt (XVLA / mixed_1_clean 等)
├── shared_sz/                    深圳团队共享
├── kai_official_relay/           kai0_base / kai0_dagger 官方数据中转
├── from_uc01/                    uc01 → 任意机器中转 (含 gf3 venv tar)
├── tim/                          用户私有 (临时打包 / 中转)
├── migrate_js/                   js 集群停用前迁移备份
├── migrate_pure200/              early pure200 迁移备份
├── backup_uc_reinstall_20260516/ uc 重装前完整备份
├── graspforge/                   另一项目 (GraspForge)
├── sam3d/                        另一项目 (SAM3D)
└── test_xvla_speed/              性能测试
```

#### 6.5.B KAI0/ 主数据结构 (canonical)

```
tos://transfer-shanghai/KAI0/
├── Task_A/                       ⭐ 主任务 (双臂叠衣)
│   ├── base/
│   │   ├── 2026-04-23-v2/        每个 date dir 结构:
│   │   ├── 2026-04-24-v2/          ├── data/chunk-000/episode_*.parquet
│   │   ├── 2026-04-25-v2/          ├── videos/chunk-000/
│   │   ├── 2026-04-28-v2/          │   ├── hand_left/episode_*.mp4
│   │   ├── 2026-04-29-v2/          │   ├── hand_right/episode_*.mp4
│   │   ├── 2026-04-30-v2/          │   ├── top_head/episode_*.mp4
│   │   ├── 2026-05-06-v2/          │   └── top_head_depth/episode_*.zarr/
│   │   ├── 2026-05-07-v2/          └── meta/{info.json,episodes.jsonl,tasks.jsonl}
│   │   ├── 2026-05-08-v2/
│   │   ├── 2026-05-09-v2/        注: 4 月之前的 date 无 top_head_depth (那时未录 depth);
│   │   ├── 2026-05-16-v2/             5-06 起所有 date 含完整 depth zarr.
│   │   ├── 2026-05-18-v2/
│   │   ├── 2026-05-19-v2/
│   │   ├── 2026-05-20-v2/
│   │   ├── 2026-05-21-v2/
│   │   ├── 2026-05-22-v2/
│   │   └── kai0_official_base/   官方 base 副本 (3055 ep)
│   ├── autonomy/
│   │   └── <date-v2>/            自主推理录制数据 (同 base 结构)
│   └── dagger/
│       └── <date-v2>/            DAgger 接管数据 (同 base 结构)
│
├── Task_E/  base/<date-v2>/      扶起倒箱
├── Task_H/  base/<date-v2>/      
├── Task_HP/ base/<date-v2>/      
├── Task_P/  base/<date-v2>/      抓放盒子
├── Task_PP/ base/<date-v2>/      
└── Task_PS/ base/<date-v2>/      
```

#### 6.5.C "-v2" 后缀约定 (2026-05-11 起)

- `<YYYY-MM-DD>-v2` = canonical 命名 (本地权威格式, 全 TOS 路径只用此格式)
- 没有 -v2 后缀的 legacy 目录均为 deprecated, 不应作为新数据落点
- 一个 date 在 TOS 同时只能以 `-v2` 形式存在; 数据修改 → 升 -v3 / -v4, 同名永不内容变更 (append-only)

### 6.6 同步频率 + 一致性约定

| 数据类别 | 上传时机 | 拉取时机 | 一致性级别 |
|---|---|---|---|
| **训练数据集** (`dataset/`) | sim01 清洗完成 + manifest 更新 | 训练任务启动前 (各服务器手动拉) | 永久 — 同名版本永远相同 |
| **Ckpt** (`checkpoints/`) | 训练完成 + 验证 MAE 合格 | sim01 部署时 / 其他服务器 init 时 | 永久 + 不变更 |
| **base_init_ckpts/** | 极少更新 (新 pi05_base 出版本时) | 各服务器初始化时一次 | 长期 — 当前版本固定 |
| **external/** | 一次性 (例如 XVLA-Soft-Fold) | 各服务器需要时 | 永久 |

**KEY**: TOS 副本是只追加 (append-only), 不修改不删除。要修改 → 上新版本 (`v2`, `v3`...) + 更新 manifest。

### 6.7 历史方法 (已 deprecated)

- ❌ ~~uc01 ↔ uc02 lsyncd~~ (跨集群一致性靠 TOS 保证, 不再依赖 lsyncd 实时镜像)
- ❌ ~~js01-04 JuiceFS 共享~~ (js 集群已停用)
- ❌ ~~gf0 → sim01 SSH 反向隧道传 ckpt~~ (改走 TOS)
- ⚠️ rsync 直连仅用于**临时小文件** (代码 patch / 配置 / 日志), 数据/ckpt **必须走 TOS**

---


---

## 10. 部署 Ckpt 工作流 (训练 → sim01 推理)

```
训练完成 (gf*)
  ↓
auto_pack_on_end.sh: 选 best step, 打包 params+assets+METADATA
  ↓
TOS upload (gf*) → /transfer-shanghai/KAI0/<name>.tar
  ↓
sim01 from_tos_file.py download
  ↓
sim01 解压 + symlink 到 kai0/checkpoints
  ↓
serve_policy.py 启动推理服务
```

详见 [`sim01_deployment.md`](./sim01_deployment.md) 与 [`checkpoints_layout.md`](./checkpoints_layout.md)。

---

