"""
Find neurons that sustain runaway firing and give them spike-frequency adaptation in "stable" mode.

Strong odour (and other) input pushes a set of cholinergic antennal-lobe local neurons (lLN1_bc) and
Kenyon cells into firing that never stops once the input is removed. We stimulate every preset at
two intensities, record which neurons are still firing 200-800 ms after the input is switched off,
give those neurons adaptation, and repeat until nothing persists.

Writes server/data/loop_neurons.npy (bool mask).
"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
import connectome, sim  # noqa: E402

ip, ix, w = connectome.load()
presets = json.load(open(ROOT / "server/data/presets.json"))
tables = json.load(open(ROOT / "server/data/tables.json"))
nz = np.load(ROOT / "server/data/neurons.npz")
sup = np.array(tables["super_class"])[nz["sup"]]
# Only interneurons can be loop neurons. Sensory, descending, ascending and motor neurons carry the
# signals in and out; slowing them would blunt the behaviours themselves (e.g. the giant fibre's
# escape), and they only kept firing in the probes because they were driven directly or sit downstream.
eligible = ~np.isin(sup, ["sensory", "descending", "ascending", "motor", "sensory_ascending", "sensory_descending",
                          "visceral_circulatory", "ascending_visceral_circulatory"])
# inputs to probe: every sense, plus the descending neurons behind each behaviour readout (the nerve
# cord has its own self-exciting loops that only show up when it is driven from the brain)
probes = [(s["key"], s["neurons"]) for s in presets["stimuli"]] + [(r["key"], r["neurons"]) for r in presets["readouts"]]
b = sim.Brain(ip, ix, w, mode="stable")
prev = ROOT / "server/data/loop_neurons.npy"
loop = (np.load(prev) & eligible) if prev.exists() else np.zeros(b.N, bool)   # resume from the last run
for it in range(10):
    b.ad_scale[:] = loop
    b.dep_mask[:] = loop
    new = np.zeros(b.N, bool)
    for key, neurons in probes:
        for rate in (250, 100):
            b.reset()
            b.set_stim({i: rate for i in neurons})
            for _ in range(20):
                b.step(25)
            b.set_stim({})
            for _ in range(8):
                b.step(25)
            c = np.bincount(np.concatenate([b.step(25).copy() for _ in range(24)]), minlength=b.N)
            if c.sum() > 50:
                new |= (c > 0) & eligible
    added = new & ~loop
    print(f"pass {it}: persistent {new.sum()}, new {added.sum()}", flush=True)
    if not added.any():
        break
    loop |= new
np.save(ROOT / "server/data/loop_neurons.npy", loop)
print("neurons given adaptation:", int(loop.sum()))
