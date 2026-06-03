#!/bin/bash
# GigaWorld-Policy(WAM)环境安装 —— 已固化踩坑修正,可一键复现。
#
# 教训(为何不能照搬 CLAUDE.md 的朴素三条 pip install):
#   1. giga-datasets/requirements.txt 硬性要求 torch==2.6.0,但 giga-train/giga-models
#      不钉版本会装最新 torch(2.12)+ torchvision(0.27),三者冲突 → pip 回溯下载
#      torch-2.6.0(766MB)时从 PyPI stall 卡死。故先钉死 torch==2.6.0 + torchvision==0.21.0。
#   2. 用清华镜像避免大包下载 stall。
#   3. giga-datasets 有未在 requirements 声明的运行时依赖(lmdb / lpips / terminaltables /
#      numpydantic 等),用 constraints + 正常 deps 装齐;numpydantic 单独补。
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n gigaworld-policy python=3.11 -y || true
conda activate gigaworld-policy
cd "$(dirname "$0")/../.."   # -> repo root (giga_world_policy/)
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

M="-i https://pypi.tuna.tsinghua.edu.cn/simple --progress-bar off"
CONS=/tmp/wam_constraints.txt
printf 'torch==2.6.0\ntorchvision==0.21.0\n' > "$CONS"

echo "=== [1/5] pin torch stack first (避免后续被升级触发回溯) ==="
pip install $M -c "$CONS" torch==2.6.0 torchvision==0.21.0
echo "=== [2/5] giga-train ==="
pip install $M -c "$CONS" ./third_party/giga-train
echo "=== [3/5] giga-models ==="
pip install $M -c "$CONS" ./third_party/giga-models
echo "=== [4/5] giga-datasets (editable, full deps under constraints) ==="
pip install $M -c "$CONS" -e ./third_party/giga-datasets
echo "=== [5/5] 补未声明依赖 ==="
pip install $M numpydantic

python -c "import torch,torchvision,av,diffusers,giga_train,giga_datasets,giga_models; \
print('SELFCHECK OK | torch',torch.__version__,'| tv',torchvision.__version__,'| av',av.__version__, \
'| av1dec:', [c for c in av.codecs_available if c in ('av1','libdav1d')])"
echo ENV_SETUP_DONE
