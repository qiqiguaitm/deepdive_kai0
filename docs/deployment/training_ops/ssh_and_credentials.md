# SSH / 用户 / 凭据 / TOS

> SSH 速查 (各机连接命令) / 用户体系 / TOS 凭据与 bucket / uc 集群 SSH 互信拓扑。
>
> **同 series**: `overview.md` / `storage_and_env.md` / `data_sync_tos.md` / `submission/`

---

## 4. 连接方式 / 用户信息

### 4.1 SSH 速查

```bash
# gf0 (从 sim01 / 任意公网机)
ssh -p 55555 tim@14.103.44.161   # gf0 (反向隧道经 14.103.44.161 跳板)

# gsy (火山北京 volc 提交节点, root 直连) — 北京 Robot-North-H20 队列数据同步/环境准备/任务提交入口
ssh -p 16370 root@124.174.16.237  # gsy, 凭据在 /root/.bashrc (VOLC_AK/SK); mlp CLI 在 /root/.volc/bin
# ⚠️ gf3 (:7888) 单卡 H20 dev 机已于 2026-07 关闭, 勿再用

# uc01 / uc02 / uc03 (2026-05-18 重装后, 直连, ubuntu 账户 key-based)
ssh ubuntu@117.50.196.104   # uc01
ssh ubuntu@106.75.68.254    # uc02
ssh ubuntu@117.50.217.231   # uc03
# (旧: sshpass -p tim ssh tim@... — 已废弃, tim 用户在 uc 上不存在)

# 也可在 ~/.bashrc 设别名:
alias gsy='ssh -p 16370 root@124.174.16.237'   # 北京 volc 提交口;gf3 (:7888) 已关闭
alias uc01='ssh ubuntu@117.50.196.104'   # 2026-05-18 后, key-based, 无需密码
alias uc02='ssh ubuntu@106.75.68.254'
alias uc03='ssh ubuntu@117.50.217.231'
```

### 4.2 用户

- **gf0/sim01**: 用户名 `tim`, 密码 `tim` (有密码 sudo)
- **gsy** (火山北京 volc 提交节点): 用户名 **`root`**, 端口 `16370`。凭据(VOLC_AK/SK)在 `/root/.bashrc`,`mlp` CLI 在 `/root/.volc/bin`;项目在 `/vePFS-North-E/vis_robot/workspace/deepdive_kai0` 下(与 volc 北京队列共享盘)。⚠️ **gf3 (:7888, 火山华北 H20 单卡机) 已于 2026-07 关闭**,原经 gf3 直跑的 dev/smoke 改为经 gsy 提交 volc 任务
- **uc01/02/03** (2026-05-18 重装后): 用户名 **`ubuntu`** (不再创建 tim), key-based 登录, 强密码已设
  - cloud-init pre-seed 了本地 dev pubkey + 团队 key (yihaochen / qiqiguaitm / tim@ipc01 等) 到 `/home/ubuntu/.ssh/authorized_keys`
  - 3 台 uc 间 ubuntu 用户 ed25519 互信已配 (详见 §4.4)
  - **⚠️ 重要安全**: 重装后应立刻**禁 SSH 密码登录** (`PasswordAuthentication no` in `/etc/ssh/sshd_config`) 避免被爆破 (上次事件 2026-05-15 即由此引发, 见 `docs/backup/2026-05-16_uc_security_incident_and_backup.md`)
- gf0: 反向隧道无密码 key-based

### 4.3 TOS 凭据 / Bucket

- Bucket: `transfer-shanghai` @ `tos-cn-shanghai.volces.com` (region `cn-shanghai`)
- 读凭据: hardcoded 在 `train_scripts/kai/data/from_tos_file.py` (公开)
- 写凭据: `VOLC_TOS_AK` / `VOLC_TOS_SK` env vars 或 `tosutil` 配置

> **完整 TOS 数据同步架构 (sim01 是源 → TOS 枢纽 → 各训练服务器) 见 §6**。本节仅记录凭据/bucket 信息。

### 4.4 uc 集群 SSH 互信拓扑  ⚠️ 已停用

> uc01/02/03 已彻底停用 (2026-05-18 退役)。SSH 互信拓扑/pubkey 归档见 [`../../backup/uc_cluster_reference.md`](../../backup/uc_cluster_reference.md)。
