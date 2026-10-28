#!/usr/bin/env bash
# Подготовка арендованного GPU-сервера (Ubuntu/Debian + драйвер NVIDIA). Повторный запуск безопасен.
#   bash training/gpu/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null || { echo "нет nvidia-smi — на сервере не установлен драйвер NVIDIA"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if ! command -v ffmpeg >/dev/null || ! python3 -c "import venv" 2>/dev/null; then
  SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq ffmpeg python3-venv python3-pip tmux >/dev/null
fi

python3 -m venv .venv
. .venv/bin/activate
pip install -q -U pip

# Сборка PyTorch под CUDA, которую поддерживает драйвер (строка «CUDA Version» в nvidia-smi).
CUDA=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
ver() { echo "$1" | awk -F. '{print $1*100+$2}'; }
V=$(ver "$CUDA")
if   [ "$V" -ge 1300 ]; then IDX=cu130
elif [ "$V" -ge 1208 ]; then IDX=cu128
elif [ "$V" -ge 1206 ]; then IDX=cu126
elif [ "$V" -ge 1204 ]; then IDX=cu124
elif [ "$V" -ge 1201 ]; then IDX=cu121
else IDX=cu118; fi
echo "драйвер поддерживает CUDA $CUDA → torch $IDX"
pip install -q torch --index-url "https://download.pytorch.org/whl/$IDX"
pip install -q numpy pillow onnx onnxconverter-common "onnxruntime-gpu[cuda,cudnn]"

python - <<'PY'
import torch, onnxruntime as ort
assert torch.cuda.is_available(), "torch не видит GPU"
print("torch", torch.__version__, "→", torch.cuda.get_device_name(0))
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
print("onnxruntime:", ort.get_available_providers())
PY
echo "готово. дальше: bash training/gpu/run.sh"
