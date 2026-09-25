#!/bin/bash
# Start Flybrain on macOS or Linux. Everything it installs stays inside this folder
# (hidden .tools/, .cache/ and .venv-<machine>/), separately for each kind of machine,
# so the same folder also works on a Windows PC (see run.bat).
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)              P=macos-arm64; T=aarch64-apple-darwin ;;
  Darwin-*)                  P=macos-x64;   T=x86_64-apple-darwin ;;
  Linux-aarch64|Linux-arm64) P=linux-arm64; T=aarch64-unknown-linux-gnu ;;
  *)                         P=linux-x64;   T=x86_64-unknown-linux-gnu ;;
esac
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/python-$P"
export UV_PYTHON_PREFERENCE=only-managed
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv-$P"
export HF_HOME="$ROOT/.cache/huggingface"
export TORCH_HOME="$ROOT/.cache/torch"
export NUMBA_CACHE_DIR="$ROOT/.cache/numba-$P"
export PYTHONPYCACHEPREFIX="$ROOT/.cache/pycache-$P"
UV="$ROOT/.tools/$P/uv"
if [ ! -x "$UV" ]; then
  echo "Setting up uv (a small Python manager) in .tools/$P ..."
  mkdir -p "$ROOT/.tools/$P"
  curl -fL --progress-bar "https://github.com/astral-sh/uv/releases/latest/download/uv-$T.tar.gz" \
    | tar -xz -C "$ROOT/.tools/$P" --strip-components=1
  if [ ! -x "$UV" ]; then
    echo "Couldn't download uv. Check the internet connection and run this again."
    exit 1
  fi
fi
echo "Starting Flybrain ($P). The first run installs Python and packages; this takes a few minutes."
exec "$UV" run --frozen --python 3.12 server/app.py
