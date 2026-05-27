#!/usr/bin/env bash
# Pull uc01-built .venv + uv-managed Python from TOS, install on gf3 with path rewrites.
# Source layout (on uc01):
#   .venv at /data/shared/ubuntu/workspace/deepdive_kai0/kai0/.venv
#   python at /home/ubuntu/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu
# Target layout (on gf3):
#   .venv at /vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/.venv
#   python at /root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu
set -euo pipefail

KAI0_ROOT="/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0"
UV_PYDIR="/root/.local/share/uv/python"
UC01_VENV_PATH="/data/shared/ubuntu/workspace/deepdive_kai0/kai0/.venv"
UC01_PY_PATH="/home/ubuntu/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu"
GF3_PY_PATH="$UV_PYDIR/cpython-3.12.13-linux-x86_64-gnu"

echo "[$(date -u +%FT%TZ)] === step 1: download tars ==="
mkdir -p /vePFS-North-E/vis_robot/venv "$UV_PYDIR"
cd /vePFS-North-E/vis_robot/venv
tosutil cp tos://transfer-shanghai/from_uc01/gf3/venv.tar     ./venv.tar     -j 16 -p 16 2>&1 | tail -3
tosutil cp tos://transfer-shanghai/from_uc01/gf3/uvpython.tar ./uvpython.tar -j 8  -p 8  2>&1 | tail -3
ls -lh venv.tar uvpython.tar

echo "[$(date -u +%FT%TZ)] === step 2: extract uv-managed python ==="
tar -xf uvpython.tar -C "$UV_PYDIR/"
# uc01 used short symlink name cpython-3.12-linux-x86_64-gnu → cpython-3.12.13-linux-x86_64-gnu;
# we extract only the 3.12.13 dir, then symlink 3.12 for compat with pyvenv.cfg references.
ln -sfn cpython-3.12.13-linux-x86_64-gnu "$UV_PYDIR/cpython-3.12-linux-x86_64-gnu"
ls -la "$UV_PYDIR/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12" | head

echo "[$(date -u +%FT%TZ)] === step 3: extract .venv ==="
rm -rf "$KAI0_ROOT/.venv"
mkdir -p "$KAI0_ROOT"
tar -xf venv.tar -C "$KAI0_ROOT/" .venv
ls "$KAI0_ROOT/.venv" | head

echo "[$(date -u +%FT%TZ)] === step 4: path rewrite ==="
# 4a. pyvenv.cfg
sed -i "s|/home/ubuntu/.local/share/uv|/root/.local/share/uv|g" "$KAI0_ROOT/.venv/pyvenv.cfg"
# 4b. activate scripts (VIRTUAL_ENV path)
find "$KAI0_ROOT/.venv/bin" -maxdepth 1 -type f \( -name "activate*" -o -name "*.csh" -o -name "*.fish" \) \
  -exec sed -i \
    -e "s|/data/shared/ubuntu/workspace/deepdive_kai0/kai0/.venv|$KAI0_ROOT/.venv|g" \
    -e "s|/home/ubuntu/.local/share/uv|/root/.local/share/uv|g" \
    {} +
# 4c. shebangs in scripts (#!/.venv/bin/python or similar) — bin/* with text shebang
for f in "$KAI0_ROOT/.venv/bin"/*; do
  [ -f "$f" ] || continue
  if head -c 100 "$f" 2>/dev/null | grep -qE "^#!.*/data/shared/ubuntu/workspace"; then
    sed -i "1s|/data/shared/ubuntu/workspace/deepdive_kai0/kai0/.venv|$KAI0_ROOT/.venv|" "$f"
  fi
done
# 4d. .pth files (editable install paths)
find "$KAI0_ROOT/.venv/lib/python3.12/site-packages" -maxdepth 2 -name "*.pth" \
  -exec sed -i "s|/data/shared/ubuntu/workspace/deepdive_kai0/kai0|$KAI0_ROOT|g" {} +
# 4e. fix the broken python symlinks in bin/
ln -sfn "$GF3_PY_PATH/bin/python3.12" "$KAI0_ROOT/.venv/bin/python"
ln -sfn python "$KAI0_ROOT/.venv/bin/python3"
ln -sfn python "$KAI0_ROOT/.venv/bin/python3.12"

echo "[$(date -u +%FT%TZ)] === step 5: validate ==="
"$KAI0_ROOT/.venv/bin/python" -c "
import sys
print('sys.prefix', sys.prefix)
print('sys.executable', sys.executable)
print('sys.version', sys.version)
"
"$KAI0_ROOT/.venv/bin/python" -c "
import jax, torch, flax, transformers
print('JAX', jax.__version__, jax.devices())
print('Torch', torch.__version__, 'CUDA?', torch.cuda.is_available(), 'n_dev', torch.cuda.device_count())
print('Flax', flax.__version__)
print('Transformers', transformers.__version__)
"
"$KAI0_ROOT/.venv/bin/python" -c "import openpi; print('openpi OK from', openpi.__file__)"
"$KAI0_ROOT/.venv/bin/python" -c "import lerobot; print('lerobot OK')"

echo "[$(date -u +%FT%TZ)] === DONE ==="
touch /root/gf3_venv_install.done
