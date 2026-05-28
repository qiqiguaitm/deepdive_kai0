# TOS 数据拉取指南 (Volcano Cloud Object Storage)

KAI0 项目使用火山引擎 TOS (Tinder Object Storage) 作为大文件中转 / 数据备份层. 本文档说明如何在**新服务器**上配置 TOS 客户端、列出 / 下载 / 同步 bucket 内容.

---

## 1. 凭证 & 端点

```ini
endpoint = tos-cn-shanghai.volces.com
region   = cn-shanghai
bucket   = transfer-shanghai
AK       = ${KAI0_TOS_AK}   # 真实值见 web/data_manager/.env (gitignored)
SK       = ${KAI0_TOS_SK}   # 真实值见 web/data_manager/.env (gitignored)
```

> ⚠️ **AK/SK 是生产凭证, 严禁写进 git-tracked 文件 / 公共聊天 / 截图.** source-of-truth:
> `web/data_manager/.env`(`.gitignore` 排除). 本文档全程用 `${KAI0_TOS_AK}` / `${KAI0_TOS_SK}`
> 占位, **跑下面任何命令前先把真实值导入环境**:
> ```bash
> set -a; source /data1/tim/workspace/deepdive_kai0/web/data_manager/.env; set +a
> # .env 里需有 KAI0_TOS_AK=... / KAI0_TOS_SK=...
> ```
>
> 如要轮换 AK/SK: 火山引擎控制台 → 访问控制 → 子账号 → 重新生成密钥, 然后同步更新:
> 1. `.env` 里的 `KAI0_TOS_AK` / `KAI0_TOS_SK`
> 2. 各机器的 `~/.tosutilconfig` (`tosutil config -ak "$KAI0_TOS_AK" -sk "$KAI0_TOS_SK"`)

---

## 2. 工具选择

| 工具 | 适合场景 | 推荐度 |
|---|---|---|
| **tosutil** (官方 CLI) | 火山原生, 支持 multi-part / 断点续传 / 并发, 单线程峰值 ~85 MB/s | ⭐⭐⭐⭐⭐ |
| **AWS CLI** (S3-兼容) | 已习惯 `aws s3` 语法, 不想装新工具 | ⭐⭐⭐ |
| **rclone** | 跨云迁移 / 复杂过滤, 集群挂载 | ⭐⭐⭐ |
| **TOS Python SDK** | 嵌进应用代码里 (本仓库 `web/data_manager/backend/app/sync.py` 用的) | ⭐⭐⭐⭐ (代码集成) |

下面以 **tosutil** 为主线介绍, 因为生产 pipeline 都用它.

---

## 3. tosutil 安装

### 方法 A: 火山官方 release (推荐)

```bash
# Linux x86_64
mkdir -p ~/.local/bin
cd /tmp
wget https://tos-tools.tos-cn-beijing.volces.com/linux/amd64/tosutil
chmod +x tosutil
mv tosutil ~/.local/bin/
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
tosutil version
```

ARM64 把 URL 里 `amd64` 换 `arm64`. macOS / Windows 见 [火山引擎文档](https://www.volcengine.com/docs/6349/148777).

### 方法 B: 拷贝现有机器的 binary

```bash
scp tim@<existing-host>:~/.local/bin/tosutil ~/.local/bin/
chmod +x ~/.local/bin/tosutil
```

---

## 4. 配置凭证

```bash
tosutil config \
  -e tos-cn-shanghai.volces.com \
  -re cn-shanghai \
  -ak "$KAI0_TOS_AK" \
  -sk "$KAI0_TOS_SK"
```

保存到 `~/.tosutilconfig` (AK/SK 会被本地加密存放, 看到的不是明文).

**校验**:
```bash
tosutil ls tos://transfer-shanghai/ -d -limit 5
# 列出 bucket 顶层目录, 不递归.
# 如果报 403/AccessDenied → AK/SK 错.
# 如果报 NoSuchBucket → endpoint/region 错.
```

---

## 5. bucket 布局 (你能看到什么)

```
tos://transfer-shanghai/
├── KAI0/                              # 主数据根
│   ├── Task_A/                        # 任务级目录 (v2 layout)
│   │   ├── base/                      # 远程操作采集数据 subset
│   │   │   └── 2026-05-09-v2/         # 日期分桶
│   │   │       ├── data/chunk-000/
│   │   │       │   └── episode_*.parquet
│   │   │       ├── videos/chunk-000/
│   │   │       │   ├── top_head/episode_*.mp4
│   │   │       │   ├── hand_left/episode_*.mp4
│   │   │       │   ├── hand_right/episode_*.mp4
│   │   │       │   └── top_head_depth/episode_*.zarr/
│   │   │       └── meta/{episodes.jsonl, tasks.jsonl, info.json}
│   │   ├── dagger/                    # DAgger 增强采集
│   │   └── autonomy/                  # 自主部署回放数据 (Task #50 输出)
│   ├── Task_B/, Task_C/...
│   └── *.tar                          # nightly_tos_batch.py 临时 tar (会自动删)
└── (其他项目无关目录)
```

- **subset 命名约定**:
  - `base/` — 主从遥操作 (master+puppet)
  - `dagger/` — 人 + 模型混合 (DAgger 修正)
  - `autonomy/` — 自主部署 (policy + puppet)
- **路径规范**: `<Task>/<subset>/<YYYY-MM-DD>-v2/{data,videos,meta}/...` (v2 layout, 见 `web/data_manager/backend/app/layout.py`)
- **chunk-000**: LeRobot v2.1 标准命名, 现阶段所有数据都进 chunk-000.

---

## 6. 常用命令

### 列内容

```bash
# 列任务下所有 subset
tosutil ls tos://transfer-shanghai/KAI0/Task_A/ -d

# 看具体 subset 有哪些日期
tosutil ls tos://transfer-shanghai/KAI0/Task_A/base/ -d

# 看某天具体 episode 列表
tosutil ls tos://transfer-shanghai/KAI0/Task_A/base/2026-05-09-v2/data/chunk-000/

# 递归 + 显示大小
tosutil ls tos://transfer-shanghai/KAI0/Task_A/base/2026-05-09-v2/ -r -size
```

### 下载

```bash
# 下载单个 episode (parquet + 3 个 mp4 + 1 个 zarr)
EP=000024
DATE=2026-05-09-v2
LOCAL_ROOT=/data/KAI0
mkdir -p $LOCAL_ROOT/Task_A/base/$DATE/{data/chunk-000,videos/chunk-000,meta}

tosutil cp tos://transfer-shanghai/KAI0/Task_A/base/$DATE/data/chunk-000/episode_$EP.parquet \
           $LOCAL_ROOT/Task_A/base/$DATE/data/chunk-000/

for cam in top_head hand_left hand_right; do
  tosutil cp tos://transfer-shanghai/KAI0/Task_A/base/$DATE/videos/chunk-000/$cam/episode_$EP.mp4 \
             $LOCAL_ROOT/Task_A/base/$DATE/videos/chunk-000/$cam/
done

tosutil cp tos://transfer-shanghai/KAI0/Task_A/base/$DATE/videos/chunk-000/top_head_depth/episode_$EP.zarr/ \
           $LOCAL_ROOT/Task_A/base/$DATE/videos/chunk-000/top_head_depth/episode_$EP.zarr/ \
           -r -flat

# 下载整个 subset 当天 (建议加 --jobs 提速)
tosutil cp tos://transfer-shanghai/KAI0/Task_A/base/$DATE/ \
           $LOCAL_ROOT/Task_A/base/$DATE/ \
           -r -flat -j 16 -p 8

# 下载整个 Task_A (慎用, 可能 100s GB)
tosutil cp tos://transfer-shanghai/KAI0/Task_A/ \
           $LOCAL_ROOT/Task_A/ \
           -r -flat -j 16 -p 8
```

关键参数:
- `-r` recursive
- `-flat` 保持目录结构 (默认会丢中间层级)
- `-j 16` 16 个并发 job (多文件)
- `-p 8` 单文件 8 partition (大文件 multi-part)
- `-u` 增量更新 (skip 已存在且 size 一致的)
- `--exclude '*.tar'` 跳过 nightly batch 的临时 tar

### 同步 (镜像)

```bash
# 把 TOS 上的 Task_A/base/ 完整镜像到本地, 增量
tosutil sync tos://transfer-shanghai/KAI0/Task_A/base/ \
             $LOCAL_ROOT/Task_A/base/ \
             -j 16 -p 8 -u

# 只镜像最近一周
tosutil sync tos://transfer-shanghai/KAI0/Task_A/base/ \
             $LOCAL_ROOT/Task_A/base/ \
             --since-time '2026-05-07 00:00:00' \
             -j 16 -u
```

### 上传 (一般不需要; 项目自己 nightly 批量传)

```bash
# 单文件
tosutil cp /path/to/episode_000000.parquet \
           tos://transfer-shanghai/KAI0/Task_A/dagger/2026-05-14-v2/data/chunk-000/

# 目录递归
tosutil cp /local/dir/ \
           tos://transfer-shanghai/KAI0/Task_X/autonomy/ \
           -r -flat -j 16 -p 8
```

---

## 7. AWS CLI (S3 兼容) 备选

如果机器已经装了 AWS CLI:

```bash
aws configure set aws_access_key_id "$KAI0_TOS_AK" --profile tos
aws configure set aws_secret_access_key "$KAI0_TOS_SK" --profile tos
aws configure set region cn-shanghai --profile tos

# 列
aws s3 --endpoint-url https://tos-cn-shanghai.volces.com \
       --profile tos \
       ls s3://transfer-shanghai/KAI0/Task_A/base/

# 下载 (multi-part 自动)
aws s3 --endpoint-url https://tos-cn-shanghai.volces.com \
       --profile tos \
       cp s3://transfer-shanghai/KAI0/Task_A/base/2026-05-09-v2/ \
          /data/KAI0/Task_A/base/2026-05-09-v2/ \
          --recursive
```

AWS CLI 比 tosutil 慢 ~30% (单线程多, 没有 tosutil 的 fault-tolerant mode), 但跨厂商语法一致.

---

## 8. Python SDK (代码集成)

```python
import os
import tos

cli = tos.TosClientV2(
    ak=os.environ["KAI0_TOS_AK"],   # 从 .env / 环境变量读, 不要硬编码
    sk=os.environ["KAI0_TOS_SK"],
    endpoint='tos-cn-shanghai.volces.com',
    region='cn-shanghai',
)

# 下载到本地文件
cli.get_object_to_file('transfer-shanghai',
                       'KAI0/Task_A/base/2026-05-09-v2/data/chunk-000/episode_000024.parquet',
                       '/tmp/episode_000024.parquet')

# 列对象
for obj in tos.utils.paginate_iter(
    cli.list_objects_type2, 'transfer-shanghai', prefix='KAI0/Task_A/base/',
):
    print(obj.key, obj.size)
```

参考实现: `web/data_manager/backend/app/sync.py` (上传) 和 `tools/nightly_tos_batch.py` (tar 批量).

---

## 9. 性能 & 调参

| 场景 | 建议 |
|---|---|
| 单大文件 (>1 GB, e.g. .tar / .mp4) | `-p 8` 或 `-p 16` multi-part |
| 多小文件 (e.g. 整 subset 的 100+ parquet/mp4) | `-j 16` 并发 |
| 跨区域 (sim01 在浦东 → bucket 在浦东) | 不需要 proxy; **必须** `unset http_proxy https_proxy` |
| 大批量 (>50 GB) | 用 `-fast-fail-threshold 5` 防卡死, `tosutil cp -u` 重跑断点续传 |
| 限速 (avoid 占满带宽) | `tosutil cp ... -rate-limit-threshold 100000` (100 MB/s 上限) |

**实测性能** (sim01 ↔ TOS): 单文件 multi-part ~85 MB/s, 并发 16 个文件 ~300-400 MB/s 聚合.

---

## 10. 故障排查

| 现象 | 排查 |
|---|---|
| `AccessDenied` (403) | AK/SK 写错; 或者 AK 被禁用; 重 `tosutil config` |
| `NoSuchBucket` (404) | endpoint/region 写错, 或 bucket 名拼错 |
| `Could not resolve host` | 这台机有 HTTP 代理污染; `unset http_proxy https_proxy` 重试 |
| `Connection reset` | 网络抖动; `tosutil cp` 加 `-u` 重试会续传 |
| 上传慢 (单线程) | 增大 `-p` (single file) 或 `-j` (multi file). 单线程 limit ~30 MB/s |
| `signature mismatch` | 系统时钟跟 NTP 差超 15 min, `sudo ntpdate ntp.aliyun.com` |
| 大量小文件 LIST 慢 | TOS 单页 1000 对象, 上 100k+ 文件用 `list-objects-type2` paginator (Python SDK) |

---

## 11. 数据访问范围 (权限边界)

当前 bucket 为**单租户**, AK/SK 持有者拥有 `transfer-shanghai` 的完整读 / 写 / 删权限. 含义:

- ✅ 可以列 / 下载 `KAI0/` 下任何对象
- ⚠️ 可以**误删**对象 — 建议只用 `cp` / `sync`, 不要随便 `tosutil rm`
- ⚠️ 可以**覆盖**对象 — 不要往 `KAI0/Task_X/<subset>/<date>/` 写, 除非走 `web/data_manager/backend/app/sync.py` 的规范流程
- 想给单纯下载用户**只读权限** → 火山控制台 → 创建子账号 → 只授 `tos:GetObject` + `tos:ListBucket`

---

## 12. Quick reference

```bash
# Bootstrap (新机器一次性)
wget -O ~/.local/bin/tosutil https://tos-tools.tos-cn-beijing.volces.com/linux/amd64/tosutil
chmod +x ~/.local/bin/tosutil
tosutil config \
  -e tos-cn-shanghai.volces.com \
  -re cn-shanghai \
  -ak "$KAI0_TOS_AK" \
  -sk "$KAI0_TOS_SK"

# 增量同步 Task_A/base 到本地 (常用)
tosutil sync tos://transfer-shanghai/KAI0/Task_A/base/ /data/KAI0/Task_A/base/ -j 16 -p 8 -u

# 列内容
tosutil ls tos://transfer-shanghai/KAI0/ -d
```
