"""
Motor neurons -> muscles -> joints.

Every motor neuron in BANC is annotated with the muscle it innervates. Neurons driving the same
muscle on the same side form a pool. Each frame:

  pool firing rate  ->  muscle activation a in [0, 1]   (saturating at SAT_HZ, low-pass TAU_S like a twitch)
  muscle activations ->  joint angles                   (agonist minus antagonist around a resting pose)

The joint angles drive the NeuroMechFly body in the viewer. Nothing here is scripted: if no motor
neuron fires, nothing moves.

"Controls" are the handful of named movements the skill engine and the LLM talk about (per leg:
swing, lift, reach, grip, spread; wings: extend, power, stroke per side; head yaw/pitch; proboscis).
Each control has the muscles that push it one way (+) and the other way (-).

Mapping notes (approximate where the literature is thin, and said so in the UI):
- Legs (Azevedo et al. 2024, Soler et al. 2004): thorax-coxa promotor/remotor and rotators swing the
  leg forward/back; trochanter flexors lift the femur, trochanter extensors (incl. the jump muscle,
  tergotrochanter) push it down. The jump muscle's own motor neuron (TTMn) is kept out of the controls:
  only the giant fibre fires it, for the escape jump; tibia flexors fold, the tibia extensor reaches; tarsus depressor and
  long tendon grip, the levator raises the tarsus.
- Wings: indirect power muscles (DLM, DVM) set flight power; direct steering muscles (b1, b2 increase
  stroke amplitude; i1, i2, iii1-4 decrease it; ps, tp and b3 hold the wing in flight posture)
  (Lindsay et al. 2017; Muijres et al. 2014).
- Proboscis (Schwarz et al. 2017; McKellar et al. 2020): m9 (MN9) rostrum protraction; m8/m4
  haustellum extension; m6/m7 labellum spreading; m1/m2 retraction; pharynx m10-12 pumping.
"""
import json
from pathlib import Path

import numpy as np

BODY_FILE = Path(__file__).resolve().parents[1] / "web" / "data" / "fly_body.json"

SAT_HZ = 80.0     # pool mean rate at which a muscle counts as fully active
# ...except muscles that need only a few spikes: the jump muscle twitches fully on one or two TTMn
# spikes, and the asynchronous flight power muscles are kept going by motor neurons firing at only
# ~5-20 Hz (the 200 Hz wingbeat comes from the muscles' own stretch activation)
SAT_LOW = {"jump_ttm": 20.0, "dorsal_longitudinal": 15.0, "dorsoventral": 15.0}
TAU_S = 0.03      # muscle activation time constant

LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]

# per-leg controls: (muscles for +, muscles for -)
LEG_CONTROLS = {
    "swing": (["tergopleural_promotor", "sternal_anterior_rotator"], ["pleural_remotor_and_abductor", "sternal_posterior_rotator"]),
    "lift": (["trochanter_flexor", "accessory_trochanter_flexor"], ["trochanter_extensor", "sternotrochanter_extensor", "tergotrochanter_extensor"]),
    "reach": (["tibia_extensor"], ["tibia_flexor", "accessory_tibia_flexor"]),
    "grip": (["tarsus_depressor", "long_tendon"], ["tarsus_levator"]),
    # the pleural remotor/abductor both swings the leg back and spreads it; it's counted once, as swing,
    # so "spread" is only the adductor pulling the leg in (and the leg relaxing back out)
    "spread": ([], ["sternal_adductor"]),
    "twist": (["femur_reductor"], []),
}
WING_CONTROLS = {
    "power": (["dorsal_longitudinal", "dorsoventral"], []),                       # both sides' indirect muscles
    "extend": (["ps1", "ps2", "pleurosternal", "b3", "tp1", "tp2", "tpn"], []),
    "stroke": (["b1", "b2"], ["i1", "i2", "iii1", "iii3", "iii4"]),
}
HEAD_CONTROLS = {
    # neck muscles: yaw and roll are left vs right; pitch is raise (levators) vs lower (depressors,
    # ventral longitudinal)
    "yaw": ("transverse_horizontal", "oblique_horizontal", "neck_yaw"),
    "roll": ("adductor", "sclerite_rotator", "dorsoventral", "neck_roll"),
    "pitch_up": ("levator",),
    "pitch_down": ("depressor", "ventral_longitudinal", "neck_pitch"),
}
PROBOSCIS = {
    "rostrum": ["proboscis_m9"],
    "haustellum": ["proboscis_m8", "proboscis_m4a", "proboscis_m4b"],
    "labellum": ["proboscis_m6", "proboscis_m7"],
    "retract": ["proboscis_m1", "proboscis_m2da", "proboscis_m2db", "proboscis_m2v"],
    "pump": ["pharynx_m10", "pharynx_m11d", "pharynx_m11v", "pharynx_m12d"],
}

# How controls move the leg joints. Each control drives one NeuroMechFly degree of freedom; full
# activation takes the joint to the edge of the range a real fly uses when walking (recorded
# kinematics), times SPAN_SCALE. Resting pose = the middle of that range. So a fully driven muscle
# gives a natural, fly-sized movement, never a flail.
LEG_JOINTS = {
    # dof: [(control, direction)]  direction: which way (+1 / -1) the angle moves for a positive control
    "Coxa": [("swing", -1)],         # + swing = leg forward (promotion)
    "Femur": [("lift", -1)],         # + lift = femur raised (trochanter flexion)
    "Tibia": [("reach", -1)],        # + reach = tibia extended
    "Tarsus1": [("grip", -1)],       # + grip = tarsus pressed down
    "Coxa_yaw": [("spread", 1)],     # + spread = leg out to the side (mirrored for right legs)
    "Femur_roll": [("twist", 1)],    # mirrored for right legs
}
MIRRORED = {"Coxa_yaw", "Femur_roll"}
SPAN_SCALE = 1.2                     # a little beyond walking...
SPAN_SCALE_FRONT = 1.7               # ...and more for the front legs, which also groom and reach
# Hard joint limits: the walking range widened by this fraction of itself (+ a fixed margin, rad).
# No drive can bend a joint past them.
LIMIT_FRAC, LIMIT_ABS = 0.35, 0.1
LIMIT_FRONT_EXTRA = 0.35


NATURAL = 0.8   # activations up to this move a joint within its walking range; beyond, towards the full span


def _shape(value, r):
    """Control value -> joint offset. Everyday activation (|value| <= NATURAL) stays inside the range a
    fly uses when walking; only near-full activation reaches past it, up to the span (grooming, reaching)."""
    a = abs(value)
    half = r.get("half", r["span"])
    off = half * min(a, NATURAL) / NATURAL + max(0.0, a - NATURAL) / (1 - NATURAL) * max(0.0, r["span"] - half)
    return off if value >= 0 else -off


def _leg_ranges():
    """Per leg and DOF from recorded walking: centre (resting angle), the offset reached at full
    drive, and the hard limit, both relative to the centre."""
    try:
        steps = json.loads(BODY_FILE.read_text())["steps"]["legs"]
    except Exception:
        return {}
    out = {}
    for leg, dofs in steps.items():
        out[leg] = {}
        front = leg[1] == "F"
        for dof, series in dofs.items():
            a = np.asarray(series, float)
            lo, hi = float(a.min()), float(a.max())
            half = max((hi - lo) / 2, 0.05)
            sc = SPAN_SCALE_FRONT if front else SPAN_SCALE
            lim = half * (1 + LIMIT_FRAC) + LIMIT_ABS + (LIMIT_FRONT_EXTRA if front else 0)
            out[leg][dof] = {"centre": (lo + hi) / 2, "half": half, "span": half * sc, "limit": max(lim, half * sc)}
    return out


class Motor:
    def __init__(self, world):
        self.w = world
        pools = world.presets["pools"]
        self.pools = pools
        self.P = len(pools)
        self.pool_of = np.full(world.N, -1, np.int64)
        for k, p in enumerate(pools):
            self.pool_of[p["neurons"]] = k
        self.size = np.array([max(1, len(p["neurons"])) for p in pools], float)
        self.sat = np.array([SAT_LOW.get(p["muscle"], SAT_HZ) for p in pools], float)
        self.key = [p["key"] for p in pools]
        self.rate = np.zeros(self.P)          # smoothed pool rate (Hz)
        self.act = np.zeros(self.P)           # muscle activation 0..1
        self.index = {k: i for i, k in enumerate(self.key)}

        def find(leg=None, muscles=(), part=None, side=None, contains=()):
            out = []
            for i, p in enumerate(pools):
                if leg is not None:
                    if not p["key"].startswith(leg + "."):
                        continue
                    if p["muscle"] in muscles:
                        out.append(i)
                    continue
                if part and p["part"] != part:
                    continue
                if side and p["side"] != side:
                    continue
                m = p["muscle"]
                if (not muscles and not contains) or (muscles and m in muscles) or (contains and any(c in m for c in contains)):
                    out.append(i)
            return out

        # control -> (plus pools, minus pools)
        self.controls = {}
        for leg in LEGS:
            for c, (pos, neg) in LEG_CONTROLS.items():
                self.controls[f"{leg}.{c}"] = (find(leg, pos), find(leg, neg))
        for s, side in (("L", "left"), ("R", "right")):
            for c, (pos, neg) in WING_CONTROLS.items():
                if c == "power":
                    self.controls[f"wing{s}.power"] = (find(part="wing", muscles=pos, side=side), [])
                else:
                    self.controls[f"wing{s}.{c}"] = (find(part="wing", muscles=pos, side=side), find(part="wing", muscles=neg, side=side))
            self.controls[f"antenna{s}"] = (find(part="antenna", side=side) + find(part="antenna, scape", side=side), [])
            self.controls[f"haltere{s}"] = (find(part="haltere", side=side), [])
        # abdomen: segmental motor neurons of both sides curl it; one side more than the other bends it sideways
        abd_l, abd_r = find(part="abdomen", side="left"), find(part="abdomen", side="right")
        self.controls["abdomen.curl"] = (abd_l + abd_r, [])
        self.controls["abdomen.bend"] = (abd_l, abd_r)
        for c in ("yaw", "roll"):
            words = HEAD_CONTROLS[c]
            self.controls[f"head.{c}"] = (find(part="neck", side="left", contains=words),
                                          find(part="neck", side="right", contains=words))   # + = left side wins
        self.controls["head.pitch"] = (find(part="neck", contains=HEAD_CONTROLS["pitch_up"]),
                                       find(part="neck", contains=HEAD_CONTROLS["pitch_down"]))
        for c, muscles in PROBOSCIS.items():
            part = "pharynx" if c == "pump" else "proboscis"
            self.controls[f"proboscis.{c}"] = (find(part=part, muscles=muscles), [])
        self.controls = {k: v for k, v in self.controls.items() if v[0] or v[1]}
        self.ranges = _leg_ranges()
        # spans stay inside the hard limits, so every control's full range is reachable (reach = 1);
        # kept as a hook for controls whose joints would saturate
        self.reach = {}
        self.value = {k: 0.0 for k in self.controls}

    # ------------------------------------------------------------------ per frame
    def update(self, spikes, dt):
        """spikes: neuron indices that fired this frame; dt: seconds of brain time."""
        if dt <= 0:
            return
        pk = self.pool_of[spikes]
        pk = pk[pk >= 0]
        counts = np.bincount(pk, minlength=self.P).astype(float)
        inst = counts / (self.size * dt)
        a = 1.0 - np.exp(-dt / 0.06)
        self.rate += a * (inst - self.rate)
        target = np.minimum(1.0, self.rate / self.sat)
        b = 1.0 - np.exp(-dt / TAU_S)
        self.act += b * (target - self.act)
        for k, (pos, neg) in self.controls.items():
            vp = float(self.act[pos].max()) if pos else 0.0
            vn = float(self.act[neg].max()) if neg else 0.0
            self.value[k] = vp - vn

    def reset(self):
        self.rate[:] = 0
        self.act[:] = 0
        for k in self.value:
            self.value[k] = 0.0

    # ------------------------------------------------------------------ for the viewer
    def joints(self):
        """Offsets (radians) from the resting pose for each leg DOF, plus wing/head/proboscis values."""
        v = self.value
        legs = {}
        for leg in LEGS:
            rng = self.ranges.get(leg, {})
            d = {}
            for dof, terms in LEG_JOINTS.items():
                r = rng.get(dof, {"span": 0.4, "limit": 0.6})
                x = sum(direction * _shape(v.get(f"{leg}.{c}", 0.0), r) for c, direction in terms)
                if dof in MIRRORED and leg[0] == "R":
                    x = -x
                d[dof] = x
            # the jump muscle (tergotrochanter, TTMn) kicks the middle legs straight down
            j = self.index.get(f"{leg}.jump_ttm")
            if j is not None:
                a = float(self.act[j])
                d["Femur"] += 2.0 * a * rng.get("Femur", {"span": 0.4})["span"]
                d["Tibia"] -= 1.5 * a * rng.get("Tibia", {"span": 0.4})["span"]
            # the skeleton only goes so far
            for dof in d:
                lim = rng.get(dof, {}).get("limit")
                if lim is not None:
                    d[dof] = min(max(d[dof], -lim), lim)
            legs[leg] = {k: round(x, 4) for k, x in d.items()}
        wings = {s: {"power": round(max(v.get("wingL.power", 0), v.get("wingR.power", 0)), 3),
                     "extend": round(max(0.0, v.get(f"wing{s}.extend", 0)), 3),
                     "stroke": round(v.get(f"wing{s}.stroke", 0), 3)} for s in "LR"}
        return {
            "legs": legs,
            "wings": wings,
            "head": {c: round(v.get(f"head.{c}", 0), 3) for c in ("yaw", "pitch", "roll")},
            "proboscis": {c: round(max(0.0, v.get(f"proboscis.{c}", 0)), 3) for c in PROBOSCIS},
            "antenna": {s: round(v.get(f"antenna{s}", 0), 3) for s in "LR"},
            "haltere": {s: round(v.get(f"haltere{s}", 0), 3) for s in "LR"},
            "abdomen": {"curl": round(v.get("abdomen.curl", 0), 3), "bend": round(v.get("abdomen.bend", 0), 3)},
        }

    def summary(self):
        """Activation per body part (for glow and the panel)."""
        v = self.value
        out = {}
        for leg in LEGS:
            out[leg] = max(abs(v.get(f"{leg}.{c}", 0)) for c in LEG_CONTROLS)
        for s in "LR":
            out["wing" + s] = max(abs(v.get(f"wing{s}.{c}", 0)) for c in ("power", "extend", "stroke"))
            out["antenna" + s] = abs(v.get(f"antenna{s}", 0))
            out["haltere" + s] = abs(v.get(f"haltere{s}", 0))
        out["abdomen"] = max(abs(v.get("abdomen.curl", 0)), abs(v.get("abdomen.bend", 0)))
        internal = [i for i, p in enumerate(self.pools) if p["part"] in ("crop", "spiracle", "salivary_gland", "uterus", "eye", "unknown")]
        out["internal"] = float(self.act[internal].max()) if internal else 0.0
        out["head"] = max(abs(v.get(f"head.{c}", 0)) for c in ("yaw", "pitch", "roll"))
        out["proboscis"] = max(v.get(f"proboscis.{c}", 0) for c in PROBOSCIS)
        return {k: round(float(x), 3) for k, x in out.items()}

    def control_neurons(self, control):
        pos, neg = self.controls[control]
        return sorted({n for i in pos + neg for n in self.pools[i]["neurons"]})
