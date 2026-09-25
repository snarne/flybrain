"""
Build the app's static assets from BANC, the public brain-and-nerve-cord connectome of one adult
female fruit fly (Bates, Phelps, Kim, Yang et al. 2026; materialisation 888).

    python tools/build_banc.py path/to/banc_888      # folder with the files below

Inputs (public, gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/):
  banc_888_meta.feather                         per-neuron annotations
  banc_888_neurotransmitter_prediction_v2.csv   per-neuron transmitter prediction (fills gaps)
  banc_888_edgelist_simple_v2.feather           only used to decide which neurons have connections

Outputs:
  web/data/neurons.bin, meta.bin, meta.json     positions and labels for the 3D view
  web/data/meshes.bin, meshes.json              (empty: the point cloud itself shows the shape)
  server/data/neurons.npz, tables.json          ids, labels, transmitter signs
  server/data/presets.json                      senses, behaviour readouts, motor-neuron pools, agent channels

The connectivity itself is downloaded and packed on each machine at first run (server/connectome.py).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".cache" / "banc"
WEB = ROOT / "web/data"
SRV = ROOT / "server/data"
WEB.mkdir(parents=True, exist_ok=True)
SRV.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- regions (by the neuropil each soma sits in)
REGIONS = [
    # key, label, what it does, neuropil names (BANC root_region without prefix/side), colour
    ("optic", "Optic lobes", "Vision: motion, colour, looming", ["LA", "ME", "AME", "LO", "LOP"], "#4f8fd6"),
    ("mb", "Mushroom body", "Learning and memory", ["MB_CA", "MB_PED", "MB_VL", "MB_ML"], "#9b7be0"),
    ("cx", "Central complex", "Navigation, heading, action choice", ["EB", "FB", "PB", "NO"], "#e0795a"),
    ("lx", "Lateral complex", "Steering, links central complex to motor", ["LAL", "BU", "GA"], "#e6a04a"),
    ("al", "Antennal lobe", "Smell (first relay)", ["AL"], "#5dbb7a"),
    ("lh", "Lateral horn", "Innate smell responses", ["LH"], "#8fcf5a"),
    ("sp", "Superior protocerebrum", "Integration, internal state", ["SMP", "SIP", "SLP"], "#d4679a"),
    ("vlp", "Ventrolateral protocerebrum", "Visual and auditory features", ["AVLP", "PVLP", "PLP", "WED", "AOTU"], "#3fb3b3"),
    ("inp", "Inferior and ventromedial", "Premotor, sensorimotor hub", ["IB", "ICL", "SCL", "ATL", "CRE", "VES", "EPA", "GOR", "SPS", "IPS"], "#c2a45a"),
    ("sez", "Subesophageal zone", "Taste, feeding, grooming, motor", ["GNG", "SAD", "AMMC", "FLA", "PRW", "CAN"], "#e05a6a"),
    ("tct", "Nerve cord: flight and neck", "Wing, haltere and neck motor control (tectulum)", ["NTct", "WTct", "HTct", "UTct", "IntTct", "LTct"], "#6fb6ff"),
    ("t1", "Nerve cord: front legs", "Front-leg motor circuits (T1)", ["ProNM", "LNp_T1"], "#f2b84b"),
    ("t2", "Nerve cord: middle legs", "Middle-leg motor circuits (T2)", ["MesoNM", "LNp_T2"], "#e8904a"),
    ("t3", "Nerve cord: hind legs", "Hind-leg motor circuits (T3)", ["MetaNM", "LNp_T3"], "#d2694a"),
    ("abd", "Nerve cord: abdomen", "Abdominal ganglion", ["ANm", "ABDNM"], "#a58a6c"),
    ("other", "Other", "Nerves, sensory entry zones and cell-body layers", [], "#7d7d7d"),
]
REGION_KEYS = [r[0] for r in REGIONS]


def neuropil_base(rr):
    if not isinstance(rr, str) or not rr:
        return ""
    s = re.sub(r"^(ITO_midbrain_|ITO_optic_|COURT_vnc_|MANC_vnc_)", "", rr)
    s = re.sub(r"-T[123]$", "", s)
    s = re.sub(r"_(L|R)$", "", s)
    return s


def region_of(base):
    for i, r in enumerate(REGIONS):
        if base in r[3]:
            return i
    if base.startswith("LNp_T1"):
        return REGION_KEYS.index("t1")
    return len(REGIONS) - 1


# ---------------------------------------------------------------- neurons
meta = pd.read_feather(SRC / "banc_888_meta.feather")
meta["rid"] = meta["banc_888_id"].astype("int64")
e = pd.read_feather(SRC / "banc_888_edgelist_simple_v2.feather", columns=["pre", "post"])
connected = np.unique(np.concatenate([e["pre"].astype("int64").values, e["post"].astype("int64").values]))
del e
keep = meta["rid"].isin(connected) & ~meta["super_class"].isin(["glia", "not_a_neuron", "trachea"])
m = meta[keep].reset_index(drop=True)
N = len(m)
print("neurons", N)

# soma positions in nm (position is in 4x4x45 nm voxels; a handful of rows are already in nm)
p = m["position"].str.split(",", expand=True).astype(float).values
in_nm = p[:, 2] > 8000
pos_nm = p * np.array([4.0, 4.0, 45.0])
pos_nm[in_nm] = p[in_nm]

brain_mask = m["region"].isin(["central_brain", "optic_lobe"]).values
centre = np.array([np.median(pos_nm[brain_mask, 0]), np.median(pos_nm[brain_mask, 1]), np.median(pos_nm[:, 2])])


def to_view(q_nm):
    q = (q_nm - centre) / 1000.0
    # fly's right -> viewer's left in a frontal view; brain up, nerve cord down; front towards the camera
    return np.stack([-q[:, 0], -q[:, 1], -q[:, 2]], 1).astype(np.float32)


pos = to_view(pos_nm)
pos.tofile(WEB / "neurons.bin")
print("extent (um)", pos.min(0).round(), pos.max(0).round())

base = m["root_region"].map(neuropil_base).fillna("")
region = np.array([region_of(b) for b in base], np.int64)
# neurons with no root region: use the coarse region
unk = region == len(REGIONS) - 1
region[unk & (m["region"] == "optic_lobe").values] = REGION_KEYS.index("optic")


# ---------------------------------------------------------------- labels
def codes(values):
    s = pd.Series(values).fillna("").astype(str)
    cats = sorted(set(s))
    lut = {c: i for i, c in enumerate(cats)}
    return np.array([lut[v] for v in s], np.uint16), cats


# the most useful name: BANC cell type, else the matched FlyWire (FAFB) or MANC type, else sub-class
name = m["cell_type"].copy()
for alt in ("fafb_cell_type", "manc_cell_type", "cell_sub_class"):
    name = name.fillna(m[alt])
name = name.str.replace(r"^auto:", "", regex=True)
type_c, type_t = codes(name)
cls_c, cls_t = codes(m["cell_class"].fillna(m["cell_sub_class"]))
sup_c, sup_t = codes(m["super_class"])
side_c, side_t = codes(m["side"])
np_c, np_t = codes(base.replace("", "none"))

# transmitter: verified where known, else the per-neuron prediction
ntp = pd.read_csv(SRC / "banc_888_neurotransmitter_prediction_v2.csv", usecols=["root_id", "neurotransmitter_predicted"])
ntp["root_id"] = ntp["root_id"].astype("int64")
nt_pred = m["rid"].map(ntp.drop_duplicates("root_id").set_index("root_id")["neurotransmitter_predicted"])
nt = m["neurotransmitter_verified"].fillna(m["neurotransmitter_predicted"]).fillna(nt_pred).fillna("")
nt = nt.astype(str).str.split(r"[,;]").str[0].str.strip().str.lower()
nt_c, nt_t = codes(nt)
# GABA and glutamate (GluCl in flies) and histamine (photoreceptors; chloride channels) inhibit; the rest excite
sign = np.where(nt.isin(["gaba", "glutamate", "histamine"]).values, -1, 1).astype(np.int8)
# motor neurons are glutamatergic but excite muscles; their few central synapses are left as predicted
print("inhibitory fraction", (sign < 0).mean().round(3), "no transmitter", (nt == "").sum())

arrs = [type_c, cls_c, sup_c, np_c, region.astype(np.uint16), side_c, nt_c]
np.concatenate(arrs).astype(np.uint16).tofile(WEB / "meta.bin")
counts = {r[0]: int((region == i).sum()) for i, r in enumerate(REGIONS)}
web_meta = {
    "n": int(N), "dataset": "BANC 888",
    "order": ["type", "class", "super_class", "neuropil", "region", "side", "nt"],
    "type": type_t, "class": cls_t, "super_class": sup_t, "neuropil": np_t, "side": side_t, "nt": nt_t,
    "regions": [{"key": r[0], "label": r[1], "about": r[2], "color": r[4]} for r in REGIONS],
    "counts": counts,
}
json.dump(web_meta, open(WEB / "meta.json", "w"))
with open(WEB / "meshes.bin", "wb") as fh:
    fh.write(b"")
json.dump({"nv": 0, "nf": 0, "meshes": []}, open(WEB / "meshes.json", "w"))

np.savez_compressed(SRV / "neurons.npz", root_ids=m["rid"].values, sign=sign, type=type_c, cls=cls_c, sup=sup_c,
                    neuropil=np_c, region=region.astype(np.uint16), side=side_c, nt=nt_c)
json.dump({k: web_meta[k] for k in ["type", "class", "super_class", "neuropil", "side", "nt", "regions", "dataset"]},
          open(SRV / "tables.json", "w"))

# ---------------------------------------------------------------- neuron groups
T = name.fillna("").values
CT = m["cell_type"].fillna("").values
SUB = m["cell_sub_class"].fillna("").values
CLS = m["cell_class"].fillna("").values
FUN = m["cell_function_detailed"].fillna("").values
SIDE = m["side"].fillna("").values
SUP = m["super_class"].fillna("").values


def idx(mask):
    return sorted(np.where(mask)[0].tolist())


def by_type(*types, side=None):
    mk = np.isin(T, types) | np.isin(CT, types)
    if side:
        mk &= SIDE == side
    return idx(mk)


labellum = pd.Series(SUB).str.startswith("labellum_taste").values
stimuli = [
    ("sugar", "Sugar", "Taste", "Sugar-sensing taste neurons on the labellum (Gr64f / Gr5a)",
     idx(labellum & pd.Series(FUN).str.contains("sugar").values), 150),
    ("water", "Water", "Taste", "Water-sensing taste neurons on the labellum (ppk28)",
     idx(labellum & pd.Series(FUN).str.contains("water").values), 150),
    ("bitter", "Bitter", "Taste", "Bitter-sensing taste neurons on the labellum (Gr33a)",
     idx(labellum & pd.Series(FUN).str.contains("bitter").values), 150),
    ("touch", "Antenna touch", "Touch", "Johnston's organ touch neurons in the antenna (JO-C, D, E, F)",
     idx(np.isin(SUB, ["johnstons_organ_C_neuron", "johnstons_organ_D_neuron", "johnstons_organ_E_neuron", "johnstons_organ_F_neuron"])), 150),
    ("looming", "Looming shadow", "Vision", "LPLC2 looming detectors (an approaching predator)", by_type("LPLC2"), 100),
    ("light", "Light on", "Vision", "ON-pathway cells Mi1 and Tm3 (photoreceptors inhibit, so light enters here)", by_type("Mi1", "Tm3"), 30),
    ("motion", "Moving pattern", "Vision", "T4/T5 motion-detector neurons (all directions)",
     idx(pd.Series(CT).str.match(r"^T[45][abcd]$").values), 20),
    ("fruit", "Fruity odour", "Smell", "Or42b olfactory neurons (glomerulus DM1), vinegar and fruit esters", by_type("ORN_DM1"), 150),
    ("geosmin", "Geosmin", "Smell", "Or56a neurons (DA2), smell of harmful mould", by_type("ORN_DA2"), 150),
    ("cva", "cVA pheromone", "Smell", "Or67d neurons (DA1), male sex pheromone", by_type("ORN_DA1"), 150),
    ("co2", "Carbon dioxide", "Smell", "Gr21a/Gr63a neurons (V glomerulus), a stress odour", by_type("ORN_V"), 150),
    ("heat", "Heat", "Temperature", "Hot-sensing thermosensory neurons (VP2)", by_type("TRN_VP2"), 150),
    ("cold", "Cold", "Temperature", "Cool-sensing thermosensory neurons (VP3, VP1m)", by_type("TRN_VP3", "TRN_VP1m"), 150),
    ("humid", "Humid air", "Temperature", "Moist-air hygrosensory neurons (VP1d, VP1l)", by_type("HRN_VP1d", "HRN_VP1l"), 150),
    ("dry", "Dry air", "Temperature", "Dry-air hygrosensory neurons (VP4)", by_type("HRN_VP4"), 150),
    ("leg_touch", "Leg touch", "Touch", "Bristle touch neurons on the front legs", idx(SUB == "front_leg_bristle_neuron"), 60),
]

readouts = [
    ("feed", "Feeding", "MN9 motor neurons: proboscis extension", by_type("MN9"), [80, 80]),
    ("groom", "Antennal grooming", "aDN grooming descending neurons (DNg62, DNge078)", by_type("DNg62", "DNge078"), [20, 20]),
    ("forward", "Walk forward", "DNp09 (P9) and DNg97 descending neurons", by_type("DNp09", "DNg97"), [30, 30]),
    ("backward", "Walk backward", "Moonwalker descending neurons (MDN)", by_type("MDN"), [30, 30]),
    ("turn_l", "Turn left", "DNa01/DNa02 steering neurons, left side", by_type("DNa01", "DNa02", side="left"), [40, 40]),
    ("turn_r", "Turn right", "DNa01/DNa02 steering neurons, right side", by_type("DNa01", "DNa02", side="right"), [40, 40]),
    ("escape", "Escape takeoff", "Giant fibre descending neuron (DNp01)", by_type("DNp01"), [200, 200]),
]

kc = idx(CLS == "kenyon_cell")
channels = {
    "hear": {"label": "Hearing", "about": "Johnston's organ auditory neurons (the fly's ear) while it reads your message",
             "neurons": idx(np.isin(SUB, ["johnstons_organ_A_neuron", "johnstons_organ_B_neuron"])), "rate": 25},
    "think": {"label": "Deliberating", "about": "PFN columnar neurons feeding the central complex, the fly's action-selection hub, while the LLM reasons",
              "neurons": idx(pd.Series(CT).str.match(r"^PFN").values), "rate": 30},
    "memory": {"label": "Remembering", "about": "a sparse set of Kenyon cells (about 5%) in the mushroom body; each memory gets its own set, like an odour",
               "neurons": kc, "rate": 40, "fraction": 0.05},
}

# ---------------------------------------------------------------- motor neurons, pooled by the muscle they drive
LEG = {"front_leg": "F", "middle_leg": "M", "hind_leg": "H"}
mn = m[SUP == "motor"]
pools = []
mn = mn.assign(muscle=mn["peripheral_target_type"].fillna("?"))
# the jump muscle's motor neuron (TTMn) gets its own pool: it is the giant fibre's escape output
mn.loc[mn["manc_cell_type"].fillna("") == "TTMn", "muscle"] = "jump_ttm_muscle"
for (part, muscle, side), g in mn.groupby([mn["body_part_effector"].fillna("?"), mn["muscle"], mn["side"].fillna("center")]):
    musc = muscle.replace("_muscle", "") if muscle != "?" else "unknown"
    s = {"left": "L", "right": "R"}.get(side, "")
    if part in LEG:
        key = f"{s}{LEG[part]}.{musc}"
        label = f"{s}{LEG[part]} leg: {musc.replace('_', ' ')}"
    else:
        subs = g["cell_sub_class"].dropna()
        sub = subs.iloc[0] if part == "neck" and len(subs) else ""
        if part == "neck" and musc in ("neck", "unknown"):
            musc = sub.replace("_motor_neuron", "") if sub else "neck"
        key = f"{part}.{musc}{'.' + s if s else ''}"
        label = f"{part.replace('_', ' ')}: {musc.replace('_', ' ')}{' (' + side + ')' if s else ''}"
    pools.append({"key": key, "part": part, "muscle": musc, "side": side, "label": label,
                  "neurons": sorted(g.index.tolist())})
# merge duplicate keys
merged = {}
for p_ in pools:
    if p_["key"] in merged:
        merged[p_["key"]]["neurons"] = sorted(set(merged[p_["key"]]["neurons"]) | set(p_["neurons"]))
    else:
        merged[p_["key"]] = p_
pools = sorted(merged.values(), key=lambda x: x["key"])
print("motor pools", len(pools), "motor neurons", sum(len(p_["neurons"]) for p_ in pools))

# Known electrical synapses (gap junctions), which an electron-microscopy connectome cannot see.
# The giant fibre drives the jump motor neuron (TTMn) and the PSI interneuron (-> wing depressor
# motor neurons) mainly through gap junctions (King & Wyman 1980; Allen et al. 2006).
MANC = m["manc_cell_type"].fillna("").values
gf = by_type("DNp01")
electrical = [[g, t, 80] for g in gf for t in idx(MANC == "TTMn") + idx(MANC == "PSI")]
print("electrical synapses added", len(electrical))

presets = {
    "dataset": "BANC 888",
    "electrical": electrical,
    "pools": pools,
    "channels": channels,
    "stimuli": [dict(key=k, label=l, group=g, about=a, neurons=n, rate=r) for k, l, g, a, n, r in stimuli],
    "readouts": [dict(key=k, label=l, about=a, neurons=n, full_rate={"stable": r[0], "paper": r[1]}) for k, l, a, n, r in readouts],
}
json.dump(presets, open(SRV / "presets.json", "w"))
for s_ in presets["stimuli"]:
    print("stim", s_["key"], len(s_["neurons"]))
for s_ in presets["readouts"]:
    print("read", s_["key"], len(s_["neurons"]))
print("regions", counts)
