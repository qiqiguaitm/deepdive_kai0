#!/bin/bash
# v0_to_v1_ckpt.sh — 一键把 v0 (JAX orbax) ckpt 目录转成自包含的 v1 (Triton) ckpt 目录,
# 产物可直接喂给 start_scripts/kai/start_autonomy_from_ckpt_v1.sh <v1_dir> 做真机测试。
#
# 用法:
#   ./optimize/v1_triton/v0_to_v1_ckpt.sh <v0_ckpt_dir> [v1_out_dir] [--prompt "..."] [--keep-intermediate]
#
# 例:
#   ./optimize/v1_triton/v0_to_v1_ckpt.sh \
#       /data1/DATA_IMP/checkpoints/ckpt_v0/task_a_new_smooth_800_step49999
#   # → /data1/DATA_IMP/checkpoints/ckpt_v1/task_a_new_smooth_800_step49999/{v1_p200.pkl, train_config.json,
#   #    _CHECKPOINT_METADATA, assets/<asset_id>/norm_stats.json}
#
# 输入要求 (pack_inference_ckpt.py / 拉取脚本 的标准产物):
#   <v0_dir>/params/                          (JAX orbax 参数)
#   <v0_dir>/train_config.json                ({"base_config_name": "...", "override_asset_id": "..."})
#   <v0_dir>/_CHECKPOINT_METADATA             (orbax meta)
#   <v0_dir>/assets/<override_asset_id>/norm_stats.json
#
# 自动处理:
#   - prompt: 默认从 base_config 的 data.default_prompt 解析 (大小写/句点原样, 不 lowercase —
#     窄分布 ckpt 大小写不符会静默退化, 见 docs dagger_collection_guide §Prompt 大小写)。可用 --prompt 覆盖。
#   - delta: 由 start_autonomy_from_ckpt_v1.sh 在启动时按 config 自动判定, 转换本身与 action-space 无关。
#
# 依赖 venv:
#   - prompt 解析: kai0/.venv          (能 import openpi.training.config / etils)
#   - 权重转换:    kai0/.venv_5090_trt (torch + sentencepiece)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_CFG="$REPO/kai0/.venv/bin/python"
VENV_TRT="$REPO/kai0/.venv_5090_trt/bin/python"
TOKENIZER="$REPO/openpi_cache/big_vision/paligemma_tokenizer.model"
CONVERT="$REPO/optimize/v1_triton/convert_kai0_to_v1.py"
EXPAND="$REPO/optimize/v1_triton/expand_v1_pkl_for_phase2.py"

# ---- 解析参数 ----
V0_DIR=""
V1_DIR=""
PROMPT_OVERRIDE=""
KEEP_INTER=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prompt) PROMPT_OVERRIDE="$2"; shift 2 ;;
    --keep-intermediate) KEEP_INTER=1; shift ;;
    -*) echo "[FAIL] 未知参数: $1" >&2; exit 2 ;;
    *)
      if [ -z "$V0_DIR" ]; then V0_DIR="$1";
      elif [ -z "$V1_DIR" ]; then V1_DIR="$1";
      else echo "[FAIL] 多余位置参数: $1" >&2; exit 2; fi
      shift ;;
  esac
done

if [ -z "$V0_DIR" ]; then
  echo "用法: $0 <v0_ckpt_dir> [v1_out_dir] [--prompt \"...\"] [--keep-intermediate]" >&2
  exit 2
fi
V0_DIR="$(cd "$V0_DIR" && pwd)"   # 绝对化
BASENAME="$(basename "$V0_DIR")"
V1_DIR="${V1_DIR:-/data1/DATA_IMP/checkpoints/ckpt_v1/$BASENAME}"

# ---- 校验输入 ----
for f in params train_config.json _CHECKPOINT_METADATA; do
  if [ ! -e "$V0_DIR/$f" ]; then
    echo "[FAIL] 缺少 $V0_DIR/$f — 不是标准 v0 ckpt?" >&2; exit 1
  fi
done
for bin in "$VENV_CFG" "$VENV_TRT" "$TOKENIZER" "$CONVERT" "$EXPAND"; do
  if [ ! -e "$bin" ]; then echo "[FAIL] 依赖缺失: $bin" >&2; exit 1; fi
done

BASE_CONFIG=$("$VENV_CFG" -c "import json;print(json.load(open('$V0_DIR/train_config.json'))['base_config_name'])")
ASSET_ID=$("$VENV_CFG" -c "import json;d=json.load(open('$V0_DIR/train_config.json'));print(d.get('override_asset_id',''))")
NORM_STATS="$V0_DIR/assets/$ASSET_ID/norm_stats.json"
if [ ! -f "$NORM_STATS" ]; then
  echo "[FAIL] norm_stats 缺失: $NORM_STATS (override_asset_id=$ASSET_ID 是否匹配 assets/?)" >&2; exit 1
fi

# ---- 解析 prompt ----
if [ -n "$PROMPT_OVERRIDE" ]; then
  PROMPT="$PROMPT_OVERRIDE"
  PROMPT_SRC="--prompt 覆盖"
else
  PROMPT=$("$VENV_CFG" -c "
import sys; sys.path.insert(0,'$REPO/kai0/src')
from openpi.training import config as c
p = getattr(c.get_config('$BASE_CONFIG').data, 'default_prompt', None)
assert p, 'base_config 无 default_prompt, 请用 --prompt 显式指定'
print(p)
")
  PROMPT_SRC="config[$BASE_CONFIG].data.default_prompt"
fi

echo "============================================================"
echo "  v0 → v1 ckpt 转换"
echo "    v0_dir:      $V0_DIR"
echo "    v1_dir:      $V1_DIR"
echo "    base_config: $BASE_CONFIG"
echo "    asset_id:    $ASSET_ID"
echo "    prompt:      '$PROMPT'   ($PROMPT_SRC)"
echo "============================================================"

mkdir -p "$V1_DIR"
INTER="$REPO/optimize/results/${BASENAME}_v1.pkl"
FINAL="$V1_DIR/v1_p200.pkl"
mkdir -p "$REPO/optimize/results"

# ---- 1) convert (JAX orbax → V1 pkl, 烘焙 sentencepiece prompt embeds) ----
# convert 在 CPU 上加载 ~12GB orbax 参数 (sim01 .venv_5090_trt 无 CUDA jaxlib), 峰值
# 内存 ~18-20GB; 机器内存吃紧时 (并发 tosutil / ROS2 / 训练) 可能被 OOM-killer 杀掉
# (SIGKILL → 无 traceback, 退出码 137)。该步幂等, 失败自动重试至多 3 次。
echo "[1/4] convert_kai0_to_v1 → $INTER"
CONV_OK=0
for attempt in 1 2 3; do
  if "$VENV_TRT" "$CONVERT" \
      --jax_path "$V0_DIR" \
      --output "$INTER" \
      --prompt "$PROMPT" \
      --tokenizer_model "$TOKENIZER"; then
    CONV_OK=1; break
  fi
  echo "[warn] convert 第 $attempt 次失败 (可能 OOM, 当前 $(free -m|awk '/Mem:/{print "free="$4"M"} /Swap:/{print " swapfree="$4"M"}'|tr -d '\n'))。10s 后重试..." >&2
  rm -f "$INTER"
  sleep 10
done
if [ "$CONV_OK" -ne 1 ]; then
  echo "[FAIL] convert 连续 3 次失败。多半是内存不足: 暂停并发任务 (tosutil/训练) 或释放内存后重跑。" >&2
  exit 1
fi
# 完整性: 中间 pkl 应 ~6.7GB (pi05). 过小说明被截断。
INTER_SZ=$(stat -c%s "$INTER" 2>/dev/null || echo 0)
if [ "$INTER_SZ" -lt 6000000000 ]; then
  echo "[FAIL] 中间 pkl 仅 $INTER_SZ 字节 (<6GB), 疑似截断/未写完。" >&2; exit 1
fi

# ---- 2) expand language_embeds → 200 行 (Phase2 state encoding) ----
echo "[2/4] expand_v1_pkl_for_phase2 → $FINAL"
"$VENV_TRT" "$EXPAND" --in "$INTER" --out "$FINAL"

# ---- 3) 复制 sidecar / metadata / assets 使 v1_dir 自包含 ----
echo "[3/4] 复制 train_config.json / _CHECKPOINT_METADATA / assets"
cp -f "$V0_DIR/train_config.json"  "$V1_DIR/train_config.json"
cp -f "$V0_DIR/_CHECKPOINT_METADATA" "$V1_DIR/_CHECKPOINT_METADATA"
mkdir -p "$V1_DIR/assets"
cp -rf "$V0_DIR/assets/$ASSET_ID" "$V1_DIR/assets/$ASSET_ID"

# ---- 4) 清理中间产物 ----
if [ "$KEEP_INTER" -eq 0 ]; then
  echo "[4/4] 删除中间 pkl $INTER (--keep-intermediate 可保留)"
  rm -f "$INTER"
else
  echo "[4/4] 保留中间 pkl $INTER"
fi

# ---- 校验产物 ----
FINAL_SZ=$(stat -c%s "$FINAL" 2>/dev/null || echo 0)
if [ "$FINAL_SZ" -lt 6000000000 ]; then
  echo "[FAIL] 最终 v1_p200.pkl 仅 $FINAL_SZ 字节 (<6GB), expand 可能失败。" >&2; exit 1
fi
for f in train_config.json _CHECKPOINT_METADATA "assets/$ASSET_ID/norm_stats.json"; do
  [ -f "$V1_DIR/$f" ] || { echo "[FAIL] 产物缺 $V1_DIR/$f" >&2; exit 1; }
done
echo "------------------------------------------------------------"
echo "  v1 ckpt 就绪 (自包含, v1_p200.pkl=$FINAL_SZ 字节):"
ls -la "$V1_DIR"
echo "  norm_stats: $V1_DIR/assets/$ASSET_ID/norm_stats.json"
echo "------------------------------------------------------------"
echo "  直接测试:"
echo "    ./start_scripts/kai/start_autonomy_from_ckpt_v1.sh $V1_DIR"
echo "============================================================"
