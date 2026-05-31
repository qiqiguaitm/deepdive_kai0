#!/bin/bash
###############################################################################
# kai0 全环境一键安装脚本
#
# 覆盖:
#   1. 系统依赖 (build-essential, ffmpeg, can-utils 等)
#   2. ROS2 Jazzy (desktop + realsense + cv_bridge + tf_transformations)
#   3. 机器人 SDK (piper_sdk, python-can, transforms3d)
#   4. ROS2 工作空间编译 (piper + piper_msgs)
#   5. openpi Python 环境 (uv venv, JAX+CUDA12, PyTorch, Flax 等)
#   6. Checkpoint 下载 (可选)
#
# 用法:
#   chmod +x install_kai0.sh
#   ./install_kai0.sh              # 完整安装
#   ./install_kai0.sh --skip-ros   # 跳过 ROS2 安装 (已装好的情况)
#   ./install_kai0.sh --skip-venv  # 跳过 Python 环境构建
#   ./install_kai0.sh --skip-ckpt  # 跳过 checkpoint 下载
#
# 适用平台:
#   - sim01 (推理机/IPC): Ubuntu 24.04, 2x RTX 5090 — 全部组件
#   - gf0/gf1 (训练机): 8x A100 — 仅 Python 环境 (--skip-ros)
#
# 注意:
#   - 需要 sudo 权限
#   - sim01 可直连外网, 不走代理更快
###############################################################################

set -eo pipefail

# ─── 配置 ────────────────────────────────────────────────────────────────────
WORKSPACE=${WORKSPACE:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
KAI0_DIR="${WORKSPACE}/kai0"
ROS2_WS="${WORKSPACE}/ros2_ws"
ROS_DISTRO="jazzy"

# Python 版本: sim01 使用 3.12 (与 ROS2 Jazzy 的 rclpy 兼容)
# 训练机可用 3.11 或 3.12
PYTHON_VER="3.12"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
step()  { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ─── 参数解析 ─────────────────────────────────────────────────────────────────
SKIP_ROS=false
SKIP_VENV=false
SKIP_CKPT=false
WITH_XVLA=false   # X-VLA 推理 venv (opt-in, 默认不装; 见 docs/deployment/inference/sim01_deployment.md §3.6)
XVLA_ONLY=false   # 只建 X-VLA venv, 完全不碰系统/ROS2/openpi venv/ckpt (旧环境零影响)

# X-VLA 推理环境参数 (可 export 覆盖)
XVLA_VENV="${XVLA_VENV:-${KAI0_DIR}/.venv_xvla}"
XVLA_LEROBOT_UC_SRC="${XVLA_LEROBOT_UC_SRC:-ubuntu@117.50.196.104:/data/shared/ubuntu/workspace/X-VLA-env/.venv/lib/python3.10/site-packages}"
XVLA_TORCH_INDEX="${XVLA_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"   # sim01 5090=cu128

for arg in "$@"; do
  case $arg in
    --skip-ros)  SKIP_ROS=true ;;
    --skip-venv) SKIP_VENV=true ;;
    --skip-ckpt) SKIP_CKPT=true ;;
    --xvla)      WITH_XVLA=true ;;
    --xvla-only) WITH_XVLA=true; XVLA_ONLY=true; SKIP_ROS=true; SKIP_VENV=true; SKIP_CKPT=true ;;
    --help|-h)
      echo "用法: $0 [--skip-ros] [--skip-venv] [--skip-ckpt] [--xvla|--xvla-only]"
      echo "  --xvla       额外搭建 X-VLA 推理 venv (与 uc 训练环境对齐, 5090=cu128)"
      echo "  --xvla-only  只建 X-VLA venv, 不碰系统/ROS2/openpi venv/ckpt (旧环境零影响)"
      exit 0 ;;
    *) warn "未知参数: $arg" ;;
  esac
done

echo ""
echo "============================================================"
echo "  kai0 环境安装"
echo "  工作目录: ${WORKSPACE}"
echo "  跳过 ROS2: ${SKIP_ROS}"
echo "  跳过 venv: ${SKIP_VENV}"
echo "  跳过 ckpt: ${SKIP_CKPT}"
echo "============================================================"

###############################################################################
# Step 1: 系统依赖
###############################################################################
if [ "$XVLA_ONLY" = true ]; then
  step "Step 1/6: 系统依赖 [SKIPPED] (--xvla-only: 不碰 apt/系统)"
else
  step "Step 1/6: 系统依赖"

  info "安装基础构建工具和库..."
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    git-lfs \
    curl \
    wget \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    can-utils \
    ethtool \
    net-tools \
    tmux \
    python3-pip \
    python3-dev \
    libssl-dev \
    libffi-dev \
    software-properties-common

  ok "系统依赖安装完成"
fi

# uv (Python 包管理器)
if ! command -v uv &>/dev/null; then
  info "安装 uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  ok "uv 安装完成: $(uv --version)"
else
  ok "uv 已安装: $(uv --version)"
fi

###############################################################################
# Step 2: ROS2 Jazzy
###############################################################################
if [ "$SKIP_ROS" = false ]; then
  step "Step 2/6: ROS2 Jazzy"

  # 检查是否已安装
  if dpkg -l 2>/dev/null | grep -q "ros-${ROS_DISTRO}-desktop"; then
    ok "ROS2 ${ROS_DISTRO} desktop 已安装"
  else
    info "添加 ROS2 ${ROS_DISTRO} 仓库..."

    sudo apt-get install -y --no-install-recommends \
      locales \
      gnupg2 \
      lsb-release

    # locale
    sudo locale-gen en_US en_US.UTF-8
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    export LANG=en_US.UTF-8

    # GPG key
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg

    # 添加源
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

    sudo apt-get update
    info "安装 ros-${ROS_DISTRO}-desktop (较大, 需要几分钟)..."
    sudo apt-get install -y ros-${ROS_DISTRO}-desktop

    ok "ROS2 ${ROS_DISTRO} desktop 安装完成"
  fi

  # ROS2 附加包
  info "安装 ROS2 附加包 (realsense, cv_bridge, tf_transformations)..."
  sudo apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-realsense2-camera \
    ros-${ROS_DISTRO}-realsense2-description \
    ros-${ROS_DISTRO}-tf-transformations \
    python3-colcon-common-extensions

  ok "ROS2 附加包安装完成"
else
  step "Step 2/6: ROS2 Jazzy [SKIPPED]"
fi

###############################################################################
# Step 3: 机器人 SDK (piper_sdk, python-can, transforms3d)
###############################################################################
if [ "$SKIP_ROS" = false ]; then
  step "Step 3/6: 机器人 SDK"

  info "安装 piper_sdk, python-can, transforms3d (system python)..."
  # 使用系统 pip 安装 ROS2 节点需要的 Python 包
  # piper_sdk: Agilex Piper 双臂控制 SDK (通过 CAN bus 通信)
  # python-can: CAN bus 通信库
  # transforms3d: 3D 变换 (欧拉角/四元数转换)
  pip3 install --break-system-packages \
    piper_sdk \
    python-can \
    transforms3d \
    2>/dev/null || \
  pip3 install \
    piper_sdk \
    python-can \
    transforms3d

  ok "机器人 SDK 安装完成"
else
  step "Step 3/6: 机器人 SDK [SKIPPED]"
fi

###############################################################################
# Step 4: ROS2 工作空间编译
###############################################################################
if [ "$SKIP_ROS" = false ]; then
  step "Step 4/6: ROS2 工作空间编译"

  if [ ! -d "$ROS2_WS/src" ]; then
    warn "ROS2 工作空间不存在: $ROS2_WS/src"
    warn "跳过编译。请先准备 piper/piper_msgs 源码到 $ROS2_WS/src/"
  else
    info "编译 ROS2 工作空间 (piper + piper_msgs)..."

    # 退出 conda 环境 (conda 的 python 会干扰 colcon)
    eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null && conda deactivate 2>/dev/null || true

    source /opt/ros/${ROS_DISTRO}/setup.bash

    cd "$ROS2_WS"

    # 查找系统 python3.12 路径 (ROS2 Jazzy 绑定 Python 3.12)
    SYS_PYTHON=$(which python3.12 2>/dev/null || which python3 2>/dev/null)
    info "使用系统 Python: $SYS_PYTHON"

    colcon build \
      --cmake-args -DPython3_EXECUTABLE="$SYS_PYTHON" \
      --symlink-install \
      2>&1 | tail -10

    cd "$WORKSPACE"

    if [ -f "$ROS2_WS/install/setup.bash" ]; then
      ok "ROS2 工作空间编译完成"
      info "  piper_msgs: 自定义消息 (PosCmd, PiperStatusMsg)"
      info "  piper: 驱动节点 + launch 文件"
    else
      fail "ROS2 工作空间编译失败"
    fi
  fi
else
  step "Step 4/6: ROS2 工作空间编译 [SKIPPED]"
fi

###############################################################################
# Step 5: openpi Python 环境 (uv venv)
###############################################################################
if [ "$SKIP_VENV" = false ]; then
  step "Step 5/6: openpi Python 环境 (uv)"

  if [ ! -d "$KAI0_DIR" ]; then
    fail "kai0 目录不存在: $KAI0_DIR"
  fi

  cd "$KAI0_DIR"

  # 确保 git submodules 已初始化
  if [ -f ".gitmodules" ] && [ -d ".git" ]; then
    info "更新 git submodules..."
    git submodule update --init --recursive 2>/dev/null || warn "git submodule 更新失败 (可忽略)"
  fi

  info "构建 uv venv (Python ${PYTHON_VER})..."
  info "  核心: JAX 0.5.3 (CUDA12), PyTorch 2.7.1, Flax 0.10.2, transformers 4.53.2"
  info "  override: ml-dtypes==0.4.1, tensorstore==0.1.74, av==13.1.0, mujoco>=3.0.0"
  info "  (首次构建需要 10-20 分钟...)"

  # 不走代理, sim01 直连 PyPI 更快
  http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= \
    GIT_LFS_SKIP_SMUDGE=1 \
    uv sync --python "${PYTHON_VER}"

  # 安装 openpi 包本身 (editable mode)
  http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= \
    GIT_LFS_SKIP_SMUDGE=1 \
    uv pip install -e .

  # ─── 验证关键依赖 ───
  VENV_PYTHON="${KAI0_DIR}/.venv/bin/python"

  info "验证关键依赖..."

  # JAX + CUDA
  JAX_VER=$($VENV_PYTHON -c "import jax; print(jax.__version__)" 2>/dev/null) && \
    ok "JAX: $JAX_VER" || warn "JAX 导入失败"

  JAX_DEVICES=$($VENV_PYTHON -c "import jax; print(f'{len(jax.devices())} GPU(s): {[d.device_kind for d in jax.devices()]}')" 2>/dev/null) && \
    ok "JAX devices: $JAX_DEVICES" || warn "JAX GPU 检测失败 (可能需要设置 LD_LIBRARY_PATH)"

  # PyTorch
  TORCH_VER=$($VENV_PYTHON -c "import torch; print(f'{torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null) && \
    ok "PyTorch: $TORCH_VER" || warn "PyTorch 导入失败"

  # Flax + orbax
  FLAX_VER=$($VENV_PYTHON -c "import flax; print(flax.__version__)" 2>/dev/null) && \
    ok "Flax: $FLAX_VER" || warn "Flax 导入失败"

  # transformers
  TF_VER=$($VENV_PYTHON -c "import transformers; print(transformers.__version__)" 2>/dev/null) && \
    ok "Transformers: $TF_VER" || warn "Transformers 导入失败"

  # openpi
  $VENV_PYTHON -c "from openpi.models import pi0" 2>/dev/null && \
    ok "openpi 模块可导入" || warn "openpi 导入失败 (可能需要 PYTHONPATH)"

  # ROS2 集成验证 (仅当 ROS2 已安装)
  if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    (
      source /opt/ros/${ROS_DISTRO}/setup.bash
      $VENV_PYTHON -c "import rclpy" 2>/dev/null && \
        ok "rclpy 可导入 (ROS2 Python 集成 OK)" || warn "rclpy 导入失败 (需 source ROS2 环境后使用)"
    )
  fi

  cd "$WORKSPACE"
  ok "openpi Python 环境构建完成"
else
  step "Step 5/6: openpi Python 环境 [SKIPPED]"
fi

###############################################################################
# Step 6: Checkpoint 下载 (可选)
###############################################################################
if [ "$SKIP_CKPT" = false ]; then
  step "Step 6/6: Checkpoint 下载"

  CKPT_DIR="${KAI0_DIR}/checkpoints"

  if [ -d "${CKPT_DIR}/Task_A" ] && [ "$(ls -A ${CKPT_DIR}/Task_A 2>/dev/null)" ]; then
    ok "Task_A checkpoint 已存在: ${CKPT_DIR}/Task_A"
  else
    info "从 ModelScope 下载 Task_A checkpoint (~22GB)..."

    # 安装 modelscope
    pip3 install --break-system-packages modelscope 2>/dev/null || pip3 install modelscope

    mkdir -p "$CKPT_DIR"
    python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'OpenDriveLab/Kai0',
    local_dir='${CKPT_DIR}',
    allow_patterns=['Task_A/*']
)
print('Download complete')
" && ok "Checkpoint 下载完成" || warn "Checkpoint 下载失败 (可稍后手动下载)"

    # 拷贝 norm_stats
    if [ -f "${CKPT_DIR}/Task_A/mixed_1/norm_stats.json" ]; then
      DATA_DIR="${KAI0_DIR}/data/Task_A/kai0_base"
      if [ -d "$DATA_DIR" ]; then
        cp "${CKPT_DIR}/Task_A/mixed_1/norm_stats.json" "$DATA_DIR/"
        ok "norm_stats.json 已拷贝到 $DATA_DIR/"
      fi
    fi
  fi
else
  step "Step 6/6: Checkpoint 下载 [SKIPPED]"
fi

###############################################################################
# Step 7: X-VLA 推理环境 (可选, --xvla) — 与 uc 训练环境对齐
#   详见 docs/deployment/inference/sim01_deployment.md §3.6
#   关键: lerobot 0.4.4 fork (含 policies/xvla) 从 uc 同步; torch 用 cu128 (5090),
#         不能照搬 uc 的 cu121 (sm_120 无 kernel)。
###############################################################################
if [ "$WITH_XVLA" = true ]; then
  step "Step 7: X-VLA 推理环境 (.venv_xvla, 对齐 uc)"

  if [ -x "${XVLA_VENV}/bin/python" ] && \
     "${XVLA_VENV}/bin/python" -c "from lerobot.policies.xvla.modeling_xvla import XVLAPolicy" 2>/dev/null; then
    ok "X-VLA venv 已存在且可导入 XVLAPolicy: ${XVLA_VENV}"
  else
    command -v uv >/dev/null 2>&1 || fail "需要 uv (curl -LsSf https://astral.sh/uv/install.sh | sh)"

    info "1/4 创建 py3.10 venv: ${XVLA_VENV}"
    uv venv --clear --python 3.10 "${XVLA_VENV}" || fail "uv venv 失败"
    # uv venv 默认不带 pip → 用 uv 原生 `uv pip install --python <venv>` (无需 ensurepip)
    XPY="${XVLA_VENV}/bin/python"

    info "2/4 torch cu128 (5090; 不用 uc 的 cu121)"
    uv pip install --python "$XPY" --index-url "${XVLA_TORCH_INDEX}" torch torchvision \
      || fail "torch cu128 安装失败 (检查 ${XVLA_TORCH_INDEX})"

    info "3a/4 关键依赖 pin 到 uc 版本 + server 依赖"
    uv pip install --python "$XPY" \
      transformers==4.51.3 accelerate==1.13.0 einops==0.8.2 timm==1.0.27 \
      safetensors==0.7.0 tokenizers==0.21.4 huggingface-hub==0.36.2 \
      'numpy>=2.2' pillow opencv-python==4.13.0.92 scipy==1.15.3 av==17.0.1 \
      draccus==0.11.5 gymnasium==1.3.0 \
      msgpack msgpack-numpy websockets pyyaml \
      || warn "关键依赖安装失败, 见 sim01_deployment.md §3.6"

    # 3b: lerobot core 传递依赖较多 (datasets/diffusers/rerun/pandas/...). 不手动维护
    #     pin 列表 (易漂移 + rerun/pyarrow 版本冲突), 而是从 uc freeze 取包名 unpinned
    #     补齐 (uv 自解析), 去掉 torch/torchvision/lerobot/nvidia-*(cu13, 与 cu128 冲突)。
    info "3b/4 从 uc freeze 自动补齐 lerobot 传递依赖 (unpinned, uv 解析)"
    UC_HOST="${XVLA_LEROBOT_UC_SRC%%:*}"
    UC_SITE="${XVLA_LEROBOT_UC_SRC#*:}"
    UC_PYBIN="${UC_SITE%/lib/python3.10/site-packages}/bin/python"
    REQ_TMP="$(mktemp)"
    if ssh "$UC_HOST" "$UC_PYBIN -c 'import importlib.metadata as m; [print(d.metadata[\"Name\"]) for d in m.distributions()]'" 2>/dev/null \
         | grep -viE '^(torch|torchvision|lerobot|pip|setuptools|wheel|transformers|nvidia-)$' > "$REQ_TMP" && [ -s "$REQ_TMP" ]; then
      uv pip install --python "$XPY" -r "$REQ_TMP" \
        || warn "uc 传递依赖补齐部分失败, 见 sim01_deployment.md §3.6"
    else
      warn "无法从 uc 取 freeze ($UC_HOST) — 跳过传递依赖补齐, lerobot 可能 import 失败"
    fi
    rm -f "$REQ_TMP"

    info "4/4 lerobot fork 本体 ← rsync from uc (${XVLA_LEROBOT_UC_SRC})"
    XSITE="$(${XVLA_VENV}/bin/python -c 'import site;print(site.getsitepackages()[0])')"
    if rsync -a "${XVLA_LEROBOT_UC_SRC}/lerobot" \
               "${XVLA_LEROBOT_UC_SRC}/lerobot-0.4.4.dist-info" \
               "${XSITE}/" 2>/dev/null; then
      ok "lerobot fork 已同步到 ${XSITE}"
    else
      warn "rsync uc 失败 — 检查到 uc01 的 ssh, 或按 sim01_deployment.md §3.6 登记的正源安装"
    fi

    if "${XVLA_VENV}/bin/python" -c "from lerobot.policies.xvla.modeling_xvla import XVLAPolicy; import torch; assert torch.cuda.is_available()" 2>/dev/null; then
      ok "X-VLA venv 就绪 (XVLAPolicy 可导入 + CUDA 可用)"
    else
      warn "X-VLA venv 验证未通过 — 见 sim01_deployment.md §3.6 验证 gate 排查"
    fi
  fi
else
  step "Step 7: X-VLA 推理环境 [SKIPPED] (加 --xvla 启用)"
fi

###############################################################################
# 安装总结
###############################################################################
echo ""
echo "============================================================"
echo -e "${GREEN}  kai0 环境安装完成!${NC}"
echo "============================================================"
echo ""
echo "环境依赖总览:"
echo ""
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │  系统层                                                 │"
echo "  │    Ubuntu 24.04 · NVIDIA Driver 580+ · CUDA 12 (pip)   │"
echo "  │    build-essential · ffmpeg · can-utils · ethtool       │"
echo "  ├─────────────────────────────────────────────────────────┤"
echo "  │  ROS2 层 (仅推理机)                                     │"
echo "  │    ROS2 Jazzy Desktop                                   │"
echo "  │    ros-jazzy-realsense2-camera (D435 + 2×D405)          │"
echo "  │    ros-jazzy-cv-bridge · ros-jazzy-tf-transformations   │"
echo "  │    piper_sdk · python-can (CAN bus → Agilex Piper)      │"
echo "  ├─────────────────────────────────────────────────────────┤"
echo "  │  ROS2 WS (colcon build)                                 │"
echo "  │    piper_msgs: PosCmd, PiperStatusMsg                   │"
echo "  │    piper: 驱动节点 + launch 文件                         │"
echo "  ├─────────────────────────────────────────────────────────┤"
echo "  │  Python 环境 (uv venv, Python 3.12)                     │"
echo "  │    JAX 0.5.3+cuda12 · PyTorch 2.7.1+cu126              │"
echo "  │    Flax 0.10.2 · orbax-checkpoint 0.11.13               │"
echo "  │    transformers 4.53.2 · sentencepiece                  │"
echo "  │    openpi (editable) · openpi-client · lerobot          │"
echo "  │    wandb · tyro · rich · polars · einops                │"
echo "  └─────────────────────────────────────────────────────────┘"
echo ""

if [ "$SKIP_ROS" = false ]; then
cat << 'USAGE'
─── 推理机 (sim01) 快速启动 ────────────────────────────────────────

  1. CAN 激活 (每次开机后):
     for iface in can0 can1 can2; do
       sudo ip link set "$iface" down
       sudo ip link set "$iface" type can bitrate 1000000
       sudo ip link set "$iface" up
     done

  2. Policy Server (终端 1):
     cd /data1/tim/workspace/deepdive_kai0/kai0
     JAX_COMPILATION_CACHE_DIR=/tmp/xla_cache CUDA_VISIBLE_DEVICES=0 \
       .venv/bin/python scripts/serve_policy.py --port 8000 \
       policy:checkpoint --policy.config=pi05_flatten_fold_normal \
       --policy.dir=checkpoints/Task_A/mixed_1

  3. ROS2 推理栈 (终端 2):
     ./start_scripts/kai/start_autonomy.sh

USAGE
fi

if [ "$SKIP_VENV" = false ]; then
cat << 'USAGE'
─── 训练机 (gf0/gf1) 快速启动 ──────────────────────────────────────

  1. Full fine-tuning (JAX):
     cd kai0
     uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_normal
     XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_normal --exp_name=run1

  2. Advantage Estimator (PyTorch DDP):
     uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py \
       ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD --exp_name=run1

USAGE
fi

echo "────────────────────────────────────────────────────────────"
