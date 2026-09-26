"""
Find the sensory neurons the arena's stimuli reach, by side and by leg, and add them to
server/data/presets.json ("senses"). server/arena.py fires them from where things are relative to
the fly's body: each antenna smells and feels its own side, each eye sees its own side, each foot
tastes what it stands on.

    python tools/build_senses.py [path/to/banc_888_meta.feather]

  odour     olfactory receptor neurons, per antenna: fruit (Or42b DM1, Or22a DM2, Or59b DM4),
            CO2 (Gr21a/Gr63a V), geosmin (Or56a DA2), cVA (Or67d DA1)
  heat/cold thermosensory neurons in the arista, per side (hot VP2; cool VP3, VP1m)
  humid/dry hygrosensory neurons, per side (moist VP1d, VP1l; dry VP4)
  loom      looming detectors per eye (LPLC2, LC4)
  motion    direction-selective T4/T5 neurons per eye: front-to-back (a) and back-to-front (b)
  target    small moving object detectors per eye (LC10a; they steer courtship following through
            AOTU to DNa02, Ribeiro et al. 2018; Hindmarsh Sten et al. 2021)
  leg taste sugar and bitter taste neurons in each leg's tarsal bristles
Neurons whose side isn't annotated are left out.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRV = ROOT / "server" / "data"
META = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".cache" / "banc" / "banc_888_meta.feather"

nz = np.load(SRV / "neurons.npz")
meta = pd.read_feather(META)
meta["rid"] = meta["banc_888_id"].astype("int64")
meta = meta.set_index("rid").reindex(nz["root_ids"])
CT = meta["cell_type"].fillna("").values
SUB = meta["cell_sub_class"].fillna("").values
SIDE = meta["side"].fillna("").values
FUN = meta["cell_function_detailed"].fillna("").values


def by_side(types):
    m = np.isin(CT, types)
    return {s: np.where(m & (SIDE == s))[0].tolist() for s in ("left", "right")}


senses = {
    "odour": {
        "fruit": by_side(["ORN_DM1", "ORN_DM2", "ORN_DM4"]),
        "co2": by_side(["ORN_V"]),
        "geosmin": by_side(["ORN_DA2"]),
        "cva": by_side(["ORN_DA1"]),
    },
    "heat": by_side(["TRN_VP2"]),
    "cold": by_side(["TRN_VP3", "TRN_VP1m"]),
    "humid": by_side(["HRN_VP1d", "HRN_VP1l"]),
    "dry": by_side(["HRN_VP4"]),
    "loom": by_side(["LPLC2", "LC4"]),
    "motion": {"front_to_back": by_side(["T4a", "T5a"]), "back_to_front": by_side(["T4b", "T5b"])},
    "target": by_side(["LC10a"]),
    "leg_taste": {},
}
LEG = {"front_leg": "F", "middle_leg": "M", "hind_leg": "H"}
for part, L in LEG.items():
    for side, S in (("left", "L"), ("right", "R")):
        on = (SUB == f"{part}_taste_bristle_gustatory_neuron") & (SIDE == side)
        senses["leg_taste"][S + L] = {
            "sugar": np.where(on & pd.Series(FUN).str.contains("sugar").values)[0].tolist(),
            "bitter": np.where(on & pd.Series(FUN).str.contains("bitter").values)[0].tolist(),
        }

presets = json.load(open(SRV / "presets.json"))
presets["senses"] = senses
json.dump(presets, open(SRV / "presets.json", "w"))


def count(x):
    return sum(count(v) for v in x.values()) if isinstance(x, dict) else len(x)


for k, v in senses.items():
    print(f"{k:10s} {count(v):6d}  " + (", ".join(f"{a}: {count(b)}" for a, b in v.items()) if isinstance(v, dict) else ""))
