# CLAUDE.md — `train_scripts/` 训练脚本工作流

> **作用**：本目录下的所有训练脚本（`launch/*.sh` 等）每次运行后，Claude **必须**按本规范产出实验记录。
> **目标**：保证每个训练实验都有可追溯的 markdown 记录与 `00_action_only_finetune_history.md` 中的摘要条目，避免数据散落在 wandb / log / checkpoint 中无人归档。

---

## 1. 触发时机

下列任一情景发生时，本规范生效：

1. 用户运行了 `train_scripts/` 下任何 `*.sh` / `*.py` 训练脚本（包括 `launch/start_train.sh`、`launch/run_*.sh`、`run_kai0_mixed_1_*.sh` 等）
2. 用户让 Claude 跑训练或在远程（gf0/gf1/gf2/sim01）跑了训练
3. 训练在后台跑完，用户来询问"训练好了吗 / 结果如何"

**关键判据**：只要训练**已开始或已完成**，且产出了 inline-eval 数据或 checkpoint，就要按下面流程归档。

---

## 2. 必做流程

### 2.1 收集本次实验的核心数据

需要从训练日志 / wandb / checkpoint dir 中抽出以下字段：

| 字段 | 来源 | 必填 |
|---|---|---|
| `config_name` | 启动命令 / config.py | ✅ |
| `exp_name` | 启动命令 `--exp_name=...` | ✅ |
| `init_ckpt` | config.py 里的 init / `--resume` 参数 | ✅ |
| `dataset_repo_id` | config.py 中的 `repo_id` | ✅ |
| `freeze_filter` 是否启用 | config.py | ✅ |
| `peak_lr / decay_lr / warmup / steps / batch_size / fsdp_devices` | config.py | ✅ |
| `ema_decay` | config.py | ✅ |
| **每 save_interval 的 `MAE@{1,10,25,50}`** | `logs/train_<exp>.log` 中的 `[inline-eval] step=N MAE@1=...` 行 | ✅ |
| **best step 与对应 MAE** | 从上述曲线找 MAE@1 最低 | ✅ |
| **final step 与对应 MAE** | 训练结束最后一行 inline-eval | ✅ |
| 训练耗时 / 每 step 时间 | wandb / log 时间戳 | 可选 |
| 真机部署或 offline archive eval | `kai0/checkpoints/.../eval_val*.json` | 可选 |

提取命令示例：

```bash
# 提取 inline-eval 曲线（标准格式）
grep "inline-eval" /home/tim/workspace/deepdive_kai0/logs/train_<exp>.log \
  | sed 's/.*\[inline-eval\] //' | sed 's/  *(.*$//'

# 找 best step（MAE@1 最低）
grep "inline-eval" logs/train_<exp>.log \
  | sed 's/.*step=\([0-9]*\).*MAE@1=\([0-9.]*\).*/\2 \1/' \
  | sort -n | head -1
```

### 2.2 写单实验记录文件

**路径**：`/home/tim/workspace/deepdive_kai0/docs/training/<exp_name>_results.md`
（如果是一组对照实验，可合并为一个文件，命名 `<series_name>_results.md`，例如 `task_p_unfreeze_8k_20k_analysis.md`）

**模板**：

```markdown
# <实验名> 训练结果

> 时间：YYYY-MM-DD ~ YYYY-MM-DD
> 硬件：sim01 / gf0 / gf1 / gf2 (描述 GPU 型号 + 数量)
> 启动命令：
> ```bash
> ./train_scripts/launch/start_train.sh <config> <exp_name> <gpu_id>
> ```

## 1. 实验设定

| 参数 | 值 |
|---|---|
| config_name | `<config>` |
| exp_name | `<exp_name>` |
| init | `<init_ckpt 路径或名称>` |
| dataset | `<repo_id>` （X ep / Y frames） |
| freeze | action-only / 全解冻 / LoRA r=N |
| steps / bs / fsdp | N / N / N |
| peak_lr / warmup / decay | 1.25e-5 / 500 / cosine to 1.25e-6 |
| ema_decay | None / 0.999 / 0.9999 |
| 其它 | wd=..., dropout=..., 等 |

## 2. Per-step inline-eval 曲线

| step | MAE@1 | @10 | @25 | @50 | 备注 |
|---:|---:|---:|---:|---:|---|
| 2000 | ... | ... | ... | ... | |
| ... |
| **best** | **...** | ... | ... | ... | step=N |
| final | ... | ... | ... | ... | step=N |

## 3. 关键观察

- 趋势：what happens (overfit / converge / oscillation / plateau)
- best step 之后是过拟合还是平台
- train loss vs val MAE gap（如有）
- 与最近的 baseline 对比（去 `00_action_only_finetune_history.md` 排行榜找）

## 4. Checkpoint 路径

```
kai0/checkpoints/<config>/<exp_name>/
  ├── 2000/
  ├── 4000/
  ├── ... (best step 标注)
  └── norm_stats.json
```

## 5. 与上一版的差异 / 结论

- 相对前一个相关实验（明确点名 v3 / E2 / T10 等）的 Δ MAE@1
- 是否值得继续这个方向，还是该 stop
- 是否需要追加真机测试 / archive eval
```

### 2.3 把关键摘要追加到主排行榜

**编辑 `/home/tim/workspace/deepdive_kai0/docs/training/00_action_only_finetune_history.md`**，做两处更新：

1. **第 1 节"TL;DR 全实验 best MAE 排行榜"**：在表格中按 best MAE@1 升序插入新行：
   ```
   | <插入位置> | <exp_name> | <Task> | <数据> | <步数> | <best step> | <best @1> | <@10> | <@25> | <@50> |
   ```

2. **第 3 节"完整实验矩阵"**：在合适的子章节（3.1~3.5）下添加完整 per-step 表，格式与已有实验一致。如果是新类别（既不属于 Task E 也不属于 Task P），新增一个子章节 3.6 / 3.7。

3. **第 5 节"Checkpoint 路径索引"**：在 5.2 / 5.3 / 5.4 中加入新 checkpoint 行。

4. 如果发现 **新的工程经验**（例如某个 hparam 失败模式、新的硬件墙），同步更新第 4 节"关键工程经验"和第 7 节"教训与决策原则"。

5. **更新文件首行的"最近更新"日期**。

### 2.4 提交

完成上述记录后，执行：

```bash
cd /home/tim/workspace/deepdive_kai0
git add docs/training/<exp_name>_results.md docs/training/00_action_only_finetune_history.md
git commit -m "docs(training): record <exp_name> results (best MAE@1=<value>)"
# 默认推到 upstream（mygithub）：
git push
```

如果同时希望同步到 `origin` 远端（备份），追加 `git push origin main`。

---

## 3. 何时**不**做

- **smoke / debug 跑** （steps < 2000，或脚本注释里标记 `# smoke`）：不归档，但日志要保留以供后续查询。
- **已有实验的简单 resume**（例如 NUMA 崩溃后续训）：合并到原 results.md，不另开新文件。
- **失败实验**（OOM / 启动崩 / 0 步）：在 `00_action_only_finetune_history.md` 第 3 节末尾的"失败实验记录"小节加一行（exp_name / 失败原因 / 启动命令），不必单写 results.md。

---

## 4. 数据完整性硬性要求

写记录时必须诚实，不允许：

- ❌ 编造没有的 MAE 数据；找不到 inline-eval 数据要明说"未启用 inline_eval"或"日志已清"
- ❌ 把 inline-eval（9 ep × 20 frames）和 archive eval（9 ep × 50 queries）当成同一数列对比 —— 必须分别标注
- ❌ 把 train loss 当成 val MAE 的代理（实验已多次证明二者背离，见 `task_p_unfreeze_8k_20k_analysis.md`）
- ❌ 用"best @1 ~ 0.025"这种约值；要写精确到 4 位小数（如 `0.0257`）
- ❌ 跳过 best step 的核实，直接报 final step 数（很多实验 final 比 best 差，例如 v8 / v3e_long）

---

## 5. 模板速查

新实验完成后最少要做的 3 件事：

1. **创建** `docs/training/<exp_name>_results.md`
2. **更新** `docs/training/00_action_only_finetune_history.md` 第 1 / 3 / 5 节
3. **提交** 上述两个文件并 push 到云端

不做这 3 件事 = 实验数据丢失 / 后续 Claude 无法在新会话中正确回忆历史。
