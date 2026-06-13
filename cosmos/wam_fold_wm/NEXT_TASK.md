# 进行中的任务 — 接续点（2026-06-13）

> 新会话先读 `wam_fold_wm/HANDOFF.md`（全局交接）再读本文件（当前正在做的具体任务）。
> ⚠️ 本环境工具输出会污染：只信单值命令 `du -sm <单一路径>` / `find|wc -l` / `pgrep -fc` / Read 读 json / Write 后 `ls` 目录看文件名复验。复合命令、glob、heredoc、Write 成功返回、甚至 wc -l 都可能是假的。

---

## 当前任务

**下载 `amap_cvlab/AgiBotWorld-Beta_Lerobot_v2` 仓库里所有「折叠衣物」任务，排除「挂衣」场景和非衣物任务。**

### 背景（已确认的事实）
- 该仓库是**完整 AgiBot World Beta**：290 文件 / 183 任务 / **总计 9.7 TB**（不是叠衣专用！绝大多数是厨房/零售/家务等非衣物任务）
- 格式：LeRobot v2，双臂人形，每任务一个 `task_NNN.tar.gz`（大的拆成 `.part.0000` 等分片），许可 CC-BY-NC-SA
- 用户决定：**只要折叠衣物任务，去掉挂衣**（不是全 9.7TB）
- **task_570（叠 T 恤）已下载完成**（188.4G，4 分片，在 `kai0/data/external_cloth/agibot_lerobot_v2/`）

### 难点
amap 仓库的文件只有 `task_NNN` 编号，**没有任务描述**。要靠 **BAAI_DataCube** 的仓库名（格式 `AgiBotWorld-Beta_G1_task_NNN_<英文描述>`）来确认每个编号是什么任务。

### 已确认的折叠衣物任务号（来自 BAAI_DataCube 仓库名，确凿）
| task | 描述 | 状态 |
|---|---|---|
| task_362 | Folded shorts（折短裤） | 待下 |
| task_477 | Fold the towel on the table（折毛巾） | 待下 |
| task_509 | Folding towel（折毛巾） | 待下 |
| task_520 | Fold the shorts on the bed（折短裤） | 待下 |
| task_561 | Fold the shorts flat（折短裤） | 待下 |
| task_570 | Fold the T-shirt（叠T恤） | 已下 OK |

### 需要进一步确认描述的任务号（调研提到是衣物折叠，但未用 BAAI_DataCube 仓库名确认）
- task_599、task_555、task_444（疑似 Folding short sleeves 折短袖）、task_658、task_681
- **下一步动作**：用 ModelScope 搜索 API 确认这些编号的描述，再决定是否纳入。

### 必须排除（挂衣，不要）
- task_414 = Hang clothes on hangers（挂衣）
- task_351 = wardrobe + hang（挂衣）

---

## 怎么确认任务描述（下一步该做的）

ModelScope 搜索 API（结果存文件再 Read，别直接看 stdout）：
```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
curl -s --max-time 30 "https://www.modelscope.cn/api/v1/dolphin/datasets?PageSize=100&PageNumber=1&Query=AgiBotWorld-Beta_G1_task" -o /tmp/s.json
wc -c /tmp/s.json
# 然后 python -c 解析 /tmp/s.json 的 Namespace/Name，grep task_NNN + 描述，写文件，Read
```
查 amap 仓库完整文件树（确认哪些 task_NNN 存在 + 分片名 + 大小）：
```bash
curl -s --max-time 30 "https://www.modelscope.cn/api/v1/datasets/amap_cvlab/AgiBotWorld-Beta_Lerobot_v2/repo/tree?Revision=master&Root=&PageSize=2000" -o /tmp/agibot_tree.json
# python -c 读 /tmp/agibot_tree.json，d['Data']['Files']，每个 f['Path'] f['Size']
```

---

## 确认任务清单后，怎么下载

每个折叠任务用现有脚本（setsid 脱离 session，断点续传）：
```bash
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm/train
# 例：下 task_362。include 必须用显式后缀 ".part.*"（"task_362*" 匹配不到分片且静默成功0文件！）
setsid bash ms_download.sh amap_cvlab/AgiBotWorld-Beta_Lerobot_v2 agibot_task362 "task_362.tar.gz.part.*" > /tmp/dl_362.log 2>&1 < /dev/null &
# 落盘到 kai0/data/external_cloth/agibot_task362/
```
注意：
- **不要用守护**（无条件 pkill 模式对慢大文件有害，已弃）。卡了手动重跑上面命令续传。
- 查进度：`du -sm kai0/data/external_cloth/agibot_taskNNN`（单一路径）、`pgrep -fc "modelscope download"`
- 分片下到 `._____temp/`，下完移到任务目录顶层；字节级 `ls -la .../._____temp/` 看分片是否达预期大小

---

## 下载完成后（用 AgiBot tar 前）
- tar 分片要先合并：`cat task_NNN.tar.gz.part.* > task_NNN.tar.gz`
- 先 `file task_NNN.tar.gz` 确认是 gzip 还是未压缩 POSIX tar（galaxea 的 .tar.gz 实际是未压缩 tar！agibot 也要确认），再决定 `tar xzf`（gzip）还是 `tar xf`（纯tar）
- 解包后是 LeRobot v2，接入需建 domain_id + per-rig norm stats，相机 key 可能要映射

---

## 已完成的下载（16 个，勿重复，详见 HANDOFF.md §5）
robocoin_fold_clothes(584ep) / r1lite(111) / towel×4 + short_sleeve / unitree g1·z1·z1_dex1·h1 / xvla_soft_fold / robomind / full_folding(153G) / galaxea(2 tar) / agibot task_570(188G)。总 ~622G，都在 kai0/data/external_cloth/。
