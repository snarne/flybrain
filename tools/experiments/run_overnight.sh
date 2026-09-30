#!/bin/bash
# Run one of the experiments in tools/experiments with the app's own Python environment (set up by
# run.sh on its first start), keeping the computer awake until it finishes. Results go to results/.
#
#   tools/experiments/run_overnight.sh learned_walking --hours 8
#   tools/experiments/run_overnight.sh learned_walking --quick      # a few minutes, to check it runs
#
# Quit the Flybrain app first: the experiment uses every core but one. Stop it any time with Ctrl-C;
# running it again continues where it left off.
cd "$(dirname "$0")/../.." || exit 1
ROOT="$(pwd)"
EXP="${1:-learned_walking}"
shift
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)              P=macos-arm64 ;;
  Darwin-*)                  P=macos-x64 ;;
  Linux-aarch64|Linux-arm64) P=linux-arm64 ;;
  *)                         P=linux-x64 ;;
esac
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/python-$P"
export UV_PYTHON_PREFERENCE=only-managed
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv-$P"
export NUMBA_CACHE_DIR="$ROOT/.cache/numba-$P"
export PYTHONPYCACHEPREFIX="$ROOT/.cache/pycache-$P"
UV="$ROOT/.tools/$P/uv"
if [ ! -x "$UV" ]; then
  echo "Start Flybrain once first (run.sh or Start Flybrain.command): it sets up Python and the packages."
  exit 1
fi
if [ ! -f "tools/experiments/$EXP.py" ]; then
  echo "No experiment called $EXP. There are: $(ls tools/experiments/*.py | xargs -n1 basename | sed 's/\.py$//' | tr '\n' ' ')"
  exit 1
fi
CMD=("$UV" run --frozen --python 3.12 "tools/experiments/$EXP.py" "$@")
if command -v caffeinate >/dev/null 2>&1; then
  # macOS: don't sleep while it runs (plugged in, the lid may stay open with the screen off)
  exec caffeinate -i -s "${CMD[@]}"
fi
exec "${CMD[@]}"
