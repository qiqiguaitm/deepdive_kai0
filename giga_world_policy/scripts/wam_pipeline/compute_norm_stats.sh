#!/bin/bash
# 为单个 embodiment 数据集计算 WAM 归一化统计(norm_stats_delta.json)。
# 用法: bash scripts/wam_pipeline/compute_norm_stats.sh <emb_dir> <embodiment_id> <out_short> [sample_rate]
#   bash scripts/wam_pipeline/compute_norm_stats.sh visrobot01 0 vis       # 默认全量 1.0
#   bash scripts/wam_pipeline/compute_norm_stats.sh kairobot01 1 kai 0.2   # 抽 20%
# 第 4 个参数(可选)= sample_rate,默认 1.0(全量,一劳永逸消除 q01/q99 抽样抖动疑虑)。
# 输出 -> assets_visrobot01/norm_stats_<out_short>.json(顺序对应 config 的 norm_path 索引=embodiment_id)
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gigaworld-policy
cd "$(dirname "$0")/../.."
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
# 本地数据集 meta 齐全,强制离线避免 lerobot 联网 ping HF Hub(本机外网需代理,会 ConnectError)
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EMB=${1:?emb dir name e.g. visrobot01}
EID=${2:?embodiment id e.g. 0}
SHORT=${3:?out short e.g. vis}
SR=${4:-1.0}
NW=${WAM_NORM_WORKERS:-32}   # DataLoader worker 数,可用 env 覆盖(56 核机器可设 48)
OUT=./assets_visrobot01/norm_stats_${SHORT}.json
mkdir -p ./assets_visrobot01
# delta_mask: 14维 piper,关节 delta、夹爪(index 6/13)绝对
python -m scripts.compute_norm_stats \
  --data-paths "../kai0/data/wam_fold_v1/$EMB" \
  --output-path "$OUT" \
  --embodiment-id "$EID" \
  --delta-mask True True True True True True False True True True True True True False \
  --sample-rate "$SR" --action-chunk 48 --action-dim 14 --num-workers "$NW"
echo "NORM_DONE $EMB -> $OUT (sample_rate=$SR)"
