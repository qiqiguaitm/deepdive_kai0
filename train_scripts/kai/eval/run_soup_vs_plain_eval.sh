#!/usr/bin/env bash
# Model-soup (≈EMA) vs plain-50k eval — 验证 "PyTorch 比 JAX 差 = EMA 缺失" (§8.4).
#
# 关键修正 (踩过的坑):
#  - eval 脚本真实路径 = train_scripts/kai/eval/eval_val_action_mse.py (非 kai0/scripts/)
#  - flag = --config / --ckpt / --val / --n-sample-frames / --prompt / --out
#  - 容器只暴露 1 张 GPU = index 0; 不要设 CUDA_VISIBLE_DEVICES=1 (会回落 CPU 极慢)
#  - create_trained_policy 默认自动 cuda:0; 用绝对路径; 结果写 repo 根 (tim 可写)
#  - 50000/soup 均无 assets/, norm_stats 由 config repo_id 自动解析 (OK)
#
# 用法 (稳定 shell): bash train_scripts/kai/eval/run_soup_vs_plain_eval.sh 2>&1 | tee _eval_combined.log

set -uo pipefail
REPO=/vePFS/tim/workspace/deepdive_kai0
PY=$REPO/kai0/.venv/bin/python
EVAL=$REPO/train_scripts/kai/eval/eval_val_action_mse.py
ROOT=$REPO/kai0/checkpoints/pi05_pytorch_a_new_pure_200/A_mirror200_pi05_pytorch
VAL=$REPO/kai0/data/Task_A/self_built/A_new_pure_200_val
PROMPT="Flatten and fold the cloth."
export OPENPI_DATA_HOME=$REPO/openpi_cache
export CUDA_VISIBLE_DEVICES=0
cd "$REPO"

echo "==================== [1/2] PLAIN 50000 (no-EMA control) ===================="
"$PY" "$EVAL" --config pi05_pytorch_a_new_pure_200 \
  --ckpt "$ROOT/50000" --val "$VAL" --n-sample-frames 200 \
  --prompt "$PROMPT" --out "$REPO/_eval_plain50k.json"
echo "plain_rc=$?"

echo "==================== [2/2] SOUP 40k-50k (~EMA) ===================="
"$PY" "$EVAL" --config pi05_pytorch_a_new_pure_200 \
  --ckpt "$REPO/_soup_40k_50k" --val "$VAL" --n-sample-frames 200 \
  --prompt "$PROMPT" --out "$REPO/_eval_soup.json"
echo "soup_rc=$?"

echo "==================== RESULTS ===================="
"$PY" - <<'PYEOF'
import json, os
REPO="/vePFS/tim/workspace/deepdive_kai0"
for tag, p in [("PLAIN 50k (no EMA)", f"{REPO}/_eval_plain50k.json"),
               ("SOUP 40k-50k (~EMA)", f"{REPO}/_eval_soup.json")]:
    if os.path.exists(p):
        d = json.load(open(p))
        m = d["mae"]
        print(f"{tag:22} @1={m['1']:.4f} @10={m['10']:.4f} @25={m['25']:.4f} @50={m['50']:.4f}  (n_ep={d['n_episodes']})")
    else:
        print(f"{tag:22} MISSING ({p})")
print()
print("JAX ref (EMA, §7.1) : @1=0.0065 @10=0.0072 @25=0.0075 @50=0.0087")
print("判读: plain 应≈§8.2 (0.0121/0.0646); soup @50 若 << plain → EMA 缺失主因确认")
PYEOF
