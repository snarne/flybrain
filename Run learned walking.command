#!/bin/bash
# Double-click me in Finder: the learned-walking experiment for 8 hours (quit the Flybrain app first).
# Results: results/learned_walking/report.md. Stop with Ctrl-C or by closing the window; double-click again to
# continue where it left off.
cd "$(dirname "$0")"
./tools/experiments/run_overnight.sh learned_walking --hours 8
echo
echo "Done. The report is in results/learned_walking/report.md"
read -r -p "Press Return to close."
