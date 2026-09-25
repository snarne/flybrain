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
import numpy as np

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
    "spread": (["pleural_remotor_and_abductor"], ["sternal_adductor"]),
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

# joint gains (radians at full activation) for NeuroMechFly DOFs, relative to the resting pose
LEG_JOINTS = {
    # dof: [(control, gain)]
    "Coxa": [("swing", -0.55)],
    "Femur": [("lift", -0.95)],
    "Tibia": [("reach", -1.1)],
    "Tarsus1": [("grip", -0.55)],
    "Coxa_yaw": [("spread", 0.35)],       # mirrored for right legs
    "Femur_roll": [("twist", 0.35)],      # mirrored for right legs
}
MIRRORED = {"Coxa_yaw", "Femur_roll"}


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
                if (muscles and m in muscles) or (contains and any(c in m for c in contains)):
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
            d = {}
            for dof, terms in LEG_JOINTS.items():
                x = sum(g * v.get(f"{leg}.{c}", 0.0) for c, g in terms)
                if dof in MIRRORED and leg[0] == "R":
                    x = -x
                d[dof] = x
            # the jump muscle (tergotrochanter, TTMn) kicks the middle legs straight down
            j = self.index.get(f"{leg}.jump_ttm")
            if j is not None:
                d["Femur"] += 1.3 * float(self.act[j])
                d["Tibia"] -= 0.8 * float(self.act[j])
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
        out["head"] = max(abs(v.get(f"head.{c}", 0)) for c in ("yaw", "pitch", "roll"))
        out["proboscis"] = max(v.get(f"proboscis.{c}", 0) for c in PROBOSCIS)
        return {k: round(float(x), 3) for k, x in out.items()}

    def control_neurons(self, control):
        pos, neg = self.controls[control]
        return sorted({n for i in pos + neg for n in self.pools[i]["neurons"]})
