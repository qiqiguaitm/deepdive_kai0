#!/bin/bash
# 集群自检 bootstrap —— 在每个 volc entrypoint 顶部 `source` 它, 之后一律用变量而非硬编码路径。
#
# 解决的问题(2026-07-20 清点): entrypoint 里有 684 处集群相关硬编码
#   (/vePFS/tim × 334 · /vePFS-North-E/vis_robot × 182 · /home/tim × 168 · /vePFS/HuanQian × 1),
#   导致 cnsh 的 yaml 搬到北京队列必然失败, 反之亦然。
#
# 用法:
#   source "$(dirname "${BASH_SOURCE[0]}")/_cluster_env.sh"   # 若随 yaml 一起下发
#   或在 entrypoint 里:  source $REPO_GUESS/train_scripts/kai/volc/_cluster_env.sh
#
# 导出: CLUSTER REPO PYTHON CRAVE_REPO LMVLA_LIBERO_ROOT ROBOTWIN_PATH ROBOTWIN_PYTHON
set -uo pipefail

# ---- 1. 按挂载点判定集群(两边挂载路径不同, 这是最可靠的判据) ----
if   [ -d /vePFS-North-E/vis_robot/workspace/deepdive_kai0 ]; then
  export CLUSTER=northe
  export REPO=/vePFS-North-E/vis_robot/workspace/deepdive_kai0
  export ROBOTWIN_PATH=/vePFS-North-E/vis_robot/huanqian/RoboTwin
  _RT_WRAPPER=$REPO/lmvla/lawam/robotwin_python_wrapper_northe.sh
elif [ -d /vePFS/tim/workspace/deepdive_kai0 ]; then
  export CLUSTER=cnsh
  export REPO=/vePFS/tim/workspace/deepdive_kai0
  export ROBOTWIN_PATH=/vePFS/HuanQian/RoboTwin
  _RT_WRAPPER=$REPO/lmvla/lawam/robotwin_python_wrapper.sh
else
  echo "FATAL[_cluster_env]: 两个集群的 repo 都不存在, 无法判定集群" >&2; exit 13
fi

# ---- 2. 仓库内符号链接层(部分脚本硬编码 /home/tim/... , 容器内无该路径) ----
mkdir -p /home/tim/workspace 2>/dev/null || true
ln -sfn "$REPO" /home/tim/workspace/deepdive_kai0 2>/dev/null || true

# ---- 3. 解释器与链路变量 ----
export PYTHON=$REPO/kai0/.venv/bin/python
export STAR_VLA_PYTHON=$PYTHON
export PATH=$REPO/kai0/.venv/bin:$PATH

# CRAVE/LMWM: crave/config/paths.py 默认写死 cnsh 路径, 必须覆盖
export CRAVE_REPO=$REPO
export PYTHONPATH=$REPO/lmvla/crave/src${PYTHONPATH:+:$PYTHONPATH}
# p1_libero_rvalley_pairs.py 的 ROOT(原亦为 cnsh 硬编码)
export LMVLA_LIBERO_ROOT=$REPO/lmvla/lawam/dataset/libero_merged_no_noops_20hz

# RoboTwin 仿真侧独立解释器(sapien/mplib/curobo 不在任何 venv 内)
if [ -x "$_RT_WRAPPER" ]; then export ROBOTWIN_PYTHON=$_RT_WRAPPER; fi

# ---- 4. 自检(fail-fast 好过跑到一半才炸) ----
[ -x "$PYTHON" ] || { echo "FATAL[_cluster_env]: 无解释器 $PYTHON" >&2; exit 13; }
echo "[_cluster_env] CLUSTER=$CLUSTER REPO=$REPO"
echo "[_cluster_env] PYTHON=$PYTHON"
echo "[_cluster_env] CRAVE_REPO=$CRAVE_REPO  LMVLA_LIBERO_ROOT=$LMVLA_LIBERO_ROOT"
echo "[_cluster_env] ROBOTWIN_PATH=$ROBOTWIN_PATH  ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-<未设>}"
