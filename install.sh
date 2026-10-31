#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEVICE=CU129
if [[ "${1:-}" == --device ]]; then DEVICE="${2:-}"; shift 2; fi
if [[ $# -ne 0 ]] || [[ "$DEVICE" != CU129 && "$DEVICE" != CPU ]]; then
  echo 'Usage: bash install.sh [--device CU129|CPU]' >&2
  exit 2
fi
command -v uv >/dev/null || { echo 'Install uv first.' >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo 'Install FFmpeg first.' >&2; exit 1; }
command -v ffprobe >/dev/null || { echo 'Install ffprobe first.' >&2; exit 1; }
export UV_CACHE_DIR="$ROOT/.codex/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/.codex/python"
export NLTK_DATA="$ROOT/GPT-SoVITS/.codex/runtime-cache/nltk_data"
PYTHON="$ROOT/.codex/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then uv venv --python 3.12 --seed "$ROOT/.codex/venv"; fi
"$PYTHON" -c "import sys; assert sys.version_info[:2] == (3,12), 'Python 3.12 required'"
BACKEND=cu129
if [[ "$DEVICE" == CPU ]]; then BACKEND=cpu; fi
uv pip install --python "$PYTHON" "torch==2.8.0+$BACKEND" "torchaudio==2.8.0+$BACKEND" --index-url "https://download.pytorch.org/whl/$BACKEND"
# The Windows lock includes platform-specific wheels; resolve native Linux packages here.
uv pip install --python "$PYTHON" -r "$ROOT/requirements.txt" -r "$ROOT/requirements-dev.txt" -c "$ROOT/constraints.txt" --index-url https://pypi.org/simple
uv pip install --python "$PYTHON" --no-deps -e "$ROOT/RVC"
"$PYTHON" -m pip check
mkdir -p "$NLTK_DATA"
"$PYTHON" "$ROOT/scripts/prepare_language_resources.py"
if [[ "$DEVICE" == CU129 ]]; then
  "$PYTHON" -c "import torch; assert torch.version.cuda == '12.9'; assert torch.cuda.is_available(); a=torch.ones(2,2,device='cuda'); assert (a@a).sum().item()==8; print(torch.cuda.get_device_name())"
fi
echo "Shared environment ready. Activate with: source '$ROOT/.codex/venv/bin/activate'"
