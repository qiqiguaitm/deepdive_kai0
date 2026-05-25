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

# gf3 (火山华北 H20 单卡机, root 直连)
ssh -p 7888 root@124.174.16.237  # gf3, 密码 tim (建议改 key-based)

# uc01 / uc02 / uc03 (2026-05-18 重装后, 直连, ubuntu 账户 key-based)
ssh ubuntu@117.50.196.104   # uc01
ssh ubuntu@106.75.68.254    # uc02
ssh ubuntu@117.50.217.231   # uc03
# (旧: sshpass -p tim ssh tim@... — 已废弃, tim 用户在 uc 上不存在)

# 也可在 ~/.bashrc 设别名:
alias gf3='ssh -p 7888 root@124.174.16.237'
alias uc01='ssh ubuntu@117.50.196.104'   # 2026-05-18 后, key-based, 无需密码
alias uc02='ssh ubuntu@106.75.68.254'
alias uc03='ssh ubuntu@117.50.217.231'
```

### 4.2 用户

- **gf0/sim01**: 用户名 `tim`, 密码 `tim` (有密码 sudo)
- **gf3** (火山华北 H20): 用户名 **`root`**, 密码 `tim`。`/root/code/{README*,demo_project}` 是火山初始 demo, 我们的项目在 `/vePFS-North-E/vis_robot/` 下
- **uc01/02/03** (2026-05-18 重装后): 用户名 **`ubuntu`** (不再创建 tim), key-based 登录, 强密码已设
  - cloud-init pre-seed 了本地 dev pubkey + 团队 key (yihaochen / qiqiguaitm / tim@ipc01 等) 到 `/home/ubuntu/.ssh/authorized_keys`
  - 3 台 uc 间 ubuntu 用户 ed25519 互信已配 (详见 §4.4)
  - **⚠️ 重要安全**: 重装后应立刻**禁 SSH 密码登录** (`PasswordAuthentication no` in `/etc/ssh/sshd_config`) 避免被爆破 (上次事件 2026-05-15 即由此引发, 见 `docs/deployment/incidents/2026-05-16_uc_security_incident_and_backup.md`)
- gf0: 反向隧道无密码 key-based

### 4.3 TOS 凭据 / Bucket

- Bucket: `transfer-shanghai` @ `tos-cn-shanghai.volces.com` (region `cn-shanghai`)
- 读凭据: hardcoded 在 `train_scripts/data/from_tos_file.py` (公开)
- 写凭据: `VOLC_TOS_AK` / `VOLC_TOS_SK` env vars 或 `tosutil` 配置

> **完整 TOS 数据同步架构 (sim01 是源 → TOS 枢纽 → 各训练服务器) 见 §6**。本节仅记录凭据/bucket 信息。

### 4.4 uc 集群 SSH 互信拓扑 (2026-05-18 重装后)

**3 台 uc 间 ubuntu 用户 ed25519 互信** (cloud-init 已 pre-seed):

```
                  ┌────────────┐
                  │  本地 dev   │  (id_rsa, qiqiguaitm@sina.com 等)
                  │   tim@*     │
                  └─────┬──────┘
                        │ pubkey 进 3 server ubuntu authorized_keys
                        ▼
        ┌───────────────────────────────┐
        │  uc01 ubuntu@10-60-135-47     │
        │  uc02 ubuntu@10-60-204-66     │  彼此 ed25519 互信 (6 个方向已通)
        │  uc03 ubuntu@10-60-253-225    │
        └───────────────────────────────┘
```

**各 host ubuntu ed25519 pubkey** (2026-05-18 验证):

| Host | Pubkey (前缀 + comment) |
|---|---|
| uc01 | `AAAAC3NzaC1lZDI1NTE5AAAAIF+mEiKsU8Q2fiXWl9fG/6J+THe9+vMZKjvICm0srfLb ubuntu@10-60-135-47` |
| uc02 | `AAAAC3NzaC1lZDI1NTE5AAAAIPOYAi7KHrboT1M1AVXiulnVlyzAmJAa3HKzXaNDfc0n ubuntu@10-60-204-66` |
| uc03 | `AAAAC3NzaC1lZDI1NTE5AAAAILQdFOvow28O9HalNIPUCElD/im+FHxQCiP9N2yVtWYD ubuntu@10-60-253-225` |

**测试命令**:
```bash
ssh uc01 hostname                              # 本地 → uc01 (key-based)
ssh uc01 'ssh ubuntu@10.60.204.66 hostname'    # uc01 → uc02 (内网)
```

**与 gf / sim 集群隔离**: uc 的 ubuntu 用户 SSH key **未推到** gf/sim。跨集群 SSH 仍走 tim 用户旧互信。本地 dev 是唯一同时拥有 tim (gf/sim) + ubuntu (uc) 访问的入口。

---

