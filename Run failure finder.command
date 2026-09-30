#!/bin/bash
# Double-click me in Finder: the robotics failure finder for 8 hours (Go2 quadruped and Panda arm, half
# the time each; quit the Flybrain app first). Results: results/failure_finder/report.md.
cd "$(dirname "$0")"
./tools/run_robotics.sh find --hours 8
echo
echo "Done. The report is in results/failure_finder/report.md"
read -r -p "Press Return to close."
