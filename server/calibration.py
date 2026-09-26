"""
Calibration of the olfactory pathway, so that different odours reach different Kenyon cells.

Run as wired, with every connection a fast chemical synapse, the antennal lobe loses odour identity
within 20 ms: its cholinergic local neurons (108 of the 429, about 312,000 synapses) pass excitation
from the few glomeruli an odour activates to nearly all the others. Fruit odour (3 glomeruli), CO2 (1)
and geosmin (1) then all end up driving the same ~290 projection neurons and the same ~90 Kenyon cells
(85 shared by fruit and geosmin), and the mushroom body can't learn one odour without the others.
In real flies each odour activates its own glomeruli and a sparse, odour-specific 5-10% of the KCs.

Three things, each known from real flies, keep that separation in the "stable" model ("paper" mode
keeps the plain wiring):

  - excitatory local neurons couple glomeruli mostly through gap junctions and weakly (Yaksi & Wilson
    2010; Huang et al. 2010): their chemical synapses count EXC_LN of their connectome strength;
  - Kenyon cell outputs onto other KCs are axo-axonal and not known to excite them: set to 0 here (as
    with the loop neurons elsewhere, the model would otherwise keep them firing each other);
  - KCs are kept sparse by one giant GABAergic neuron, APL, which in real flies inhibits them through
    graded release, not spikes (Lin et al. 2014): its synapses onto KCs count APL_GAIN times, and
    projection neurons, KCs and APL are exempt from the model's short-term depression (the fast,
    faithful PN -> KC synapse), so KCs keep firing for as long as the odour is there. PNs keep their
    adaptation (they fire hard at odour onset, then settle, as real ones do).

With these, at the source of each odour (1 s, 100 Hz receptor input): fruit drives ~500 KCs (11%),
CO2 ~40 (6 of them shared with fruit), cVA ~20; geosmin, whose DA2 projection neurons go mostly to
the lateral horn, drives almost none. Nothing fires spontaneously.
"""
import numpy as np

EXC_LN = 0.2
APL_GAIN = 3.0


def groups(world):
    cls = np.array(world.tables["class"])[world.cls].astype(str)
    typ = np.array(world.tables["type"])[world.type].astype(str)
    return {
        "ln": np.where(cls == "antennal_lobe_local_neuron")[0],
        "pn": np.where(cls == "antennal_lobe_projection_neuron")[0],
        "kc": np.where(cls == "kenyon_cell")[0],
        "apl": np.where(typ == "APL")[0],
    }


def weights(world, w_signed):
    """The connectome's signed synapse counts, with the olfactory calibration applied."""
    g = groups(world)
    N = len(world.indptr) - 1
    rows = np.repeat(np.arange(N, dtype=np.int64), np.diff(world.indptr))
    post = world.indices
    mark = lambda idx: np.isin(np.arange(N), idx)
    ln, kc, apl = mark(g["ln"]), mark(g["kc"]), mark(g["apl"])
    w = w_signed.astype(np.float32).copy()
    e = ln[rows] & (w > 0)
    w[e] *= EXC_LN
    w[kc[rows] & kc[post]] = 0.0
    w[apl[rows] & kc[post]] *= APL_GAIN
    return w


def masks(world, brain):
    """Exempt the PN -> KC -> APL circuit from short-term depression, and KCs and APL from adaptation."""
    g = groups(world)
    for k in ("pn", "kc", "apl"):
        brain.dep_mask[g[k]] = False
    for k in ("kc", "apl"):
        brain.ad_scale[g[k]] = 0.0
