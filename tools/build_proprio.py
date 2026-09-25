"""
Find each leg's sense organs in BANC and add them to server/data/presets.json ("proprio").

    python tools/build_proprio.py [path/to/banc_888_meta.feather]

The physics body (server/physics.py) reports joint angles, joint speeds and foot load every frame;
server/feedback.py turns those into firing rates for these neurons, closing the loop between body and
nerve cord. Organs, per leg (Tuthill & Wilson 2016; Mamiya et al. 2018, 2023; Lesser et al. 2024):

  claw  femoral chordotonal organ, claw neurons: tonic, femur-tibia joint angle (flexion or extension)
  hook  femoral chordotonal organ, hook neurons: phasic, direction of femur-tibia movement
  club  femoral chordotonal organ, club neurons: movement and vibration, both directions
  hp    hair plates: the coxa (or trochanter) joint near the end of its range
  cs    campaniform sensilla: strain in the cuticle, i.e. load on the leg
  touch tarsal tactile bristles: the foot touching something

The claw and hook neurons each come in two types. Which type signals flexion and which extension
isn't in the annotations, so it is read from the wiring: a leg's resistance reflex opposes the
movement it senses, so the type whose output (through up to two synapses, summed over all six legs)
favours the tibia extensor motor neurons over the flexors is taken to report flexion, and the other
extension. It is an inference, and printed when the script runs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRV = ROOT / "server" / "data"
sys.path.insert(0, str(ROOT / "server"))
import connectome  # noqa: E402

META = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".cache" / "banc" / "banc_888_meta.feather"
LEG = {"front_leg": "F", "middle_leg": "M", "hind_leg": "H"}

nz = np.load(SRV / "neurons.npz")
rid = nz["root_ids"]
meta = pd.read_feather(META)
meta["rid"] = meta["banc_888_id"].astype("int64")
meta = meta.set_index("rid").reindex(rid)          # same order as the simulator's neurons
SUB = meta["cell_sub_class"].fillna("").values
CT = meta["cell_type"].fillna("").values
SIDE = meta["side"].fillna("").values
PART = meta["body_part_sensory"].fillna("").values

presets = json.load(open(SRV / "presets.json"))
pools = {p["key"]: p["neurons"] for p in presets["pools"]}

# signed, input-normalised wiring for the two-synapse reflex test
indptr, indices, w = connectome.load()
N = len(indptr) - 1
rows = np.repeat(np.arange(N, dtype=np.int32), np.diff(indptr))
inp = np.bincount(indices, weights=np.abs(w), minlength=N)
A = sparse.csr_matrix((w / np.maximum(inp[indices], 1.0), (rows, indices)), shape=(N, N))


def reach(src, tgt):
    """Mean signed influence of neurons src on neurons tgt through one or two synapses."""
    if not src or not tgt:
        return 0.0
    v = np.zeros(N)
    v[tgt] = 1.0 / len(tgt)
    e1 = A @ v
    e2 = A @ e1
    return float((e1[src] + 0.7 * e2[src]).mean())


def margins():
    """Sum over all six legs of (drive to extensor - drive to flexor), per organ and type. One leg's wiring
    is too noisy to call; all six together decide one assignment, used for every leg."""
    tot = {}
    for part, L in LEG.items():
        for side, S in (("left", "L"), ("right", "R")):
            leg = S + L
            ext = pools.get(f"{leg}.tibia_extensor", [])
            flex = pools.get(f"{leg}.tibia_flexor", []) + pools.get(f"{leg}.accessory_tibia_flexor", [])
            for organ, sub, types in ORGANS:
                for t in types:
                    src = np.where((SIDE == side) & (SUB == f"{part}_{sub}") & (CT == t))[0].tolist()
                    if src:
                        tot[(organ, t)] = tot.get((organ, t), 0.0) + (reach(src, ext) - reach(src, flex)) * len(src)
    return tot


ORGANS = (("claw", "claw_chordotonal_organ_neuron", ("SNpp50", "SNpp51")),
          ("hook", "hook_chordotonal_organ_neuron", ("SNpp39", "SNpp41")))
M = margins()
FLEXION = {}
for organ, _, (ta, tb) in ORGANS:
    FLEXION[organ] = ta if M.get((organ, ta), 0) >= M.get((organ, tb), 0) else tb
    print(f"{organ}: {ta} {M.get((organ, ta), 0):+.4f}  {tb} {M.get((organ, tb), 0):+.4f}  -> flexion = {FLEXION[organ]}")

groups = []
for part, L in LEG.items():
    for side, S in (("left", "L"), ("right", "R")):
        leg = S + L
        on_leg = SIDE == side

        def sel(sub, types=None):
            m = on_leg & (SUB == f"{part}_{sub}")
            if types is not None:
                m &= np.isin(CT, types)
            return np.where(m)[0].tolist()

        for organ, sub, types in ORGANS:
            # the type that drives the extensor more reports flexion (it resists it)
            tf = FLEXION[organ]
            te = types[1] if tf == types[0] else types[0]
            fl, ex = sel(sub, [tf]), sel(sub, [te])
            groups.append({"leg": leg, "organ": organ, "joint": "Tibia", "sense": "flexion", "type": tf, "neurons": fl})
            groups.append({"leg": leg, "organ": organ, "joint": "Tibia", "sense": "extension", "type": te, "neurons": ex})
            rest = [i for i in sel(sub) if i not in fl and i not in ex]
            if rest:     # odd types filed under the organ: both directions
                groups.append({"leg": leg, "organ": organ, "joint": "Tibia", "sense": "both", "type": "other", "neurons": rest})
        groups.append({"leg": leg, "organ": "club", "joint": "Tibia", "sense": "both", "neurons": sel("club_chordotonal_organ_neuron")})
        hp = sel("hair_plate_neuron")
        troch = [i for i in hp if "trochanter" in PART[i]]
        groups.append({"leg": leg, "organ": "hp", "joint": "Femur", "sense": "both", "neurons": troch})
        groups.append({"leg": leg, "organ": "hp", "joint": "Coxa", "sense": "both", "neurons": [i for i in hp if i not in troch]})
        groups.append({"leg": leg, "organ": "cs", "joint": None, "sense": "load",
                       "neurons": sel("campaniform_sensillum_neuron") + sel("bilateral_campaniform_sensillum_neuron")})
        groups.append({"leg": leg, "organ": "touch", "joint": None, "sense": "contact", "neurons": sel("taste_bristle_tactile_neuron")})

groups = [g for g in groups if g["neurons"]]
presets["proprio"] = groups
json.dump(presets, open(SRV / "presets.json", "w"))
for g in groups:
    print(f"{g['leg']} {g['organ']:5s} {str(g['joint']):6s} {g['sense']:9s} {g.get('type', ''):7s} {len(g['neurons'])}")
print(len(groups), "groups,", sum(len(g["neurons"]) for g in groups), "neurons")
