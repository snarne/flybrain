#!/bin/bash
# The robotics failure finder, with the app's own Python environment (set up by run.sh on its first
# start). Keeps the computer awake while it runs. Results go to results/failure_finder/.
#
#   tools/run_robotics.sh find --hours 8                 # both robots (Go2 and Panda), half the time each
#   tools/run_robotics.sh find --robots go2 --quick      # a few minutes, to check it runs
#   tools/run_robotics.sh report                         # rebuild the report from saved results
#   tools/run_robotics.sh replay go2 3 [--recovery]      # watch failure #3 in the MuJoCo viewer
#
# The robot models are downloaded once from MuJoCo Menagerie into .cache/ (not committed).
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
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
export PYTHONPYCACHEPREFIX="$ROOT/.cache/pycache-$P"
UV="$ROOT/.tools/$P/uv"
if [ ! -x "$UV" ]; then
  echo "Start Flybrain once first (run.sh or Start Flybrain.command): it sets up Python and the packages."
  exit 1
fi
WHAT="${1:-find}"
shift
case "$WHAT" in
  find)   CMD=("$UV" run --frozen --python 3.12 python -m robotics.run "$@") ;;
  report) CMD=("$UV" run --frozen --python 3.12 python -m robotics.run --report) ;;
  replay)
    PY=python
    [ "$(uname -s)" = "Darwin" ] && PY=mjpython        # the MuJoCo viewer needs mjpython on macOS
    CMD=("$UV" run --frozen --python 3.12 "$PY" -m robotics.replay "$@") ;;
  *) echo "usage: tools/run_robotics.sh find|report|replay ..."; exit 1 ;;
esac
if [ "$WHAT" = "find" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i -s "${CMD[@]}"
fi
exec "${CMD[@]}"
