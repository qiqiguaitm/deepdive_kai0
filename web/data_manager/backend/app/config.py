import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent

# 默认放在 /data1/DATA_IMP/KAI0 而不是 PROJECT_DIR 下:
# (1) 项目目录里的东西容易被 `git clean -fdx` / 重装 venv / 删 data_mock 误清, 而 DATA_IMP
#     在项目外, 单独一个 7T 大盘, 不会被项目维护脚本误伤。
# (2) 多个 worktree / 不同 checkout 共享同一份采集数据。
# 真正的 per-task / per-date 子目录由 recorder._task_subset_root() 在写盘时拼:
#   /data1/DATA_IMP/KAI0/<task>_<YYYY-MM-DD>/<subset>/...
# 想换位置就 export KAI0_DATA_ROOT=/...
DATA_ROOT = Path(os.environ.get("KAI0_DATA_ROOT", "/data1/DATA_IMP/KAI0")).resolve()
TEMPLATES_PATH = Path(
    os.environ.get("KAI0_TEMPLATES", PROJECT_DIR / "config" / "collection_templates.yml")
).resolve()
STATS_DB_PATH = Path(os.environ.get("KAI0_STATS_DB", BACKEND_DIR / "stats.sqlite3")).resolve()

STATUS_BROADCAST_HZ = 2.0
