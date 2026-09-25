"""
Check the simulator reproduces known fly behaviours, in both modes.

    uv run tools/validate.py

Expected (as in Shiu et al. 2024, now on the BANC brain + nerve cord): sugar drives MN9 (feeding),
adding bitter suppresses it, antenna touch drives grooming neurons, looming drives the giant fibre
(escape) and, through its gap junctions, the jump motor neuron. Stable mode should keep these, and
every input should go quiet within ~0.5 s of being switched off. Also prints which body parts' motor
neurons each input recruits.
"""
import json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
import connectome, sim  # noqa: E402

ip, ix, w = connectome.load()
loop = np.load(ROOT / "server/data/loop_neurons.npy")
P = json.load(open(ROOT / "server/data/presets.json"))
S = {s["key"]: s for s in P["stimuli"]}
R = {r["key"]: r for r in P["readouts"]}
PARTS = {}
for p in P["pools"]:
    part = p["key"][:2] if p["key"][:2] in ("LF", "LM", "LH", "RF", "RM", "RH") else p["part"]
    PARTS.setdefault(part, []).extend(p["neurons"])
MODES = {
    "paper": sim.Brain(ip, ix, w, mode="paper"),
    "stable": sim.Brain(ip, ix, w, mode="stable", adapt_mask=loop),
}

for mode, b in MODES.items():
    b.step(1)
    print(f"== {mode}")
    for keys in [["sugar"], ["sugar", "bitter"], ["water"], ["touch"], ["looming"], ["fruit"], ["geosmin"], ["co2"],
                 ["heat"], ["cold"], ["light"]]:
        b.reset()
        b.set_stim({i: S[k]["rate"] for k in keys for i in S[k]["neurons"]})
        t = time.time()
        spk = np.concatenate([b.step(20).copy() for _ in range(50)])
        el = time.time() - t
        c = np.bincount(spk, minlength=b.N)
        b.set_stim({})
        for _ in range(25):
            b.step(20)
        after = sum(len(b.step(20)) for _ in range(5))
        beh = {r: round(float(c[R[r]["neurons"]].mean()), 1) for r in R if c[R[r]["neurons"]].mean() >= 1}
        parts = {k: round(float(c[v].mean()), 1) for k, v in PARTS.items() if c[v].mean() >= 2}
        print(f"  {'+'.join(keys):13s} {el:5.2f}s per sim-second | neurons fired {int((c > 0).sum()):6d} | "
              f"spikes 0.5 s after off {after:6d} | behaviour Hz {beh} | motor Hz {parts}", flush=True)
