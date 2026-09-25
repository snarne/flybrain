#!/bin/bash
# Download the public BANC files that tools/build_banc.py needs into .cache/banc
# (the app itself only needs the edge list, which it downloads on first run).
set -e
cd "$(dirname "$0")/.."
mkdir -p .cache/banc && cd .cache/banc
B=https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888
for f in banc_888_meta.feather banc_888_neurotransmitter_prediction_v2.csv banc_888_edgelist_simple_v2.feather; do
  [ -f "$f" ] || curl -fL --retry 3 -o "$f" "$B/$f"
done
ls -la
