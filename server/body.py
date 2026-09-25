"""
A tethered fly whose every movement comes from its motor neurons.

Like a real neuroscience rig, the fly is held over a floating ball. Every frame, in simulated time:

  1. Stimuli -> sensory neurons. A dark object looming at one eye fires that eye's LPLC2 looming
     detectors (rate follows the shadow's angular expansion); a sugar or bitter drop touched to the
     proboscis fires taste neurons; dust on the antennae fires Johnston's organ neurons. "Activate"
     buttons drive chosen descending neurons directly, like optogenetics.
  2. The whole brain and nerve cord connectome runs (BANC).
  3. Motor neurons -> muscles -> joints (motor.py). Every motor neuron is annotated with the muscle
     it drives, so legs, wings, neck, proboscis and antennae move only as their motor neurons fire.
     Nothing is animated by hand.

The label on screen ("Feeding", "Escape jump", ...) is read from what the muscles are doing, with
some hysteresis so a brief twitch doesn't flip it back and forth.

Two rules from fly physiology the connectome alone doesn't give:
- Flight needs the feet off the ground. Tarsal contact inhibits flight, so a fly standing on the ball
  doesn't beat its wings even if the flight power muscles' motor neurons fire; the giant-fibre jump
  lifts the legs off and flight can start.
- Rhythms: walking and grooming need nerve-cord rhythm generators the model lacks. When the brain's
  command neurons for them fire, the matching built-in skill supplies the rhythm (skills.py).
"""
import math

import numpy as np

# Descending neurons that can be activated directly (like optogenetics), and what's known about them
ACTIVATE = {
    "p9": ("DNp09", "P9 / DNp09: forward walking (Bidaye et al. 2020)"),
    "mdn": ("MDN", "Moonwalker descending neurons: walking backwards (Bidaye et al. 2014)"),
    "dna02_l": ("DNa02:left", "DNa02, left: steering left (Rayshubskiy et al. 2020)"),
    "dna02_r": ("DNa02:right", "DNa02, right: steering right"),
    "gf": ("DNp01", "Giant fibre DNp01: escape takeoff (von Reyn et al. 2014)"),
    "adn": ("DNg62", "aDN1 / DNg62: antennal grooming (Hampel et al. 2015)"),
}


class Body:
    def __init__(self, world):
        self.w = world
        T = world.tables
        self.types = np.array(T["type"])[world.type]
        self.sides = np.array(T["side"])[world.side]
        self.lplc2 = {s: np.where((self.types == "LPLC2") & (self.sides == s))[0].tolist() for s in ("left", "right")}
        self.taste = {k: world.stim[k]["neurons"] for k in ("sugar", "bitter", "touch")}
        self.events = []
        self.decisions = []       # traced "why" chains, newest last (filled by the app)
        self.reset()

    def neurons_for(self, spec):
        t, _, side = spec.partition(":")
        m = self.types == t
        if side:
            m &= self.sides == side
        return np.where(m)[0].tolist()

    # ------------------------------------------------------------ experiments
    def reset(self):
        self.mode, self.mode_t = "rest", 0.0
        self._pending, self._pending_t = None, 0.0
        self._swing = {}
        self.joints, self.parts = {}, {}
        self.airborne, self.air_t = False, 0.0
        self._reflex = None
        self.loom = None          # {"side", "d"} distance in mm
        self.probe = None         # {"kind", "t"}
        self.dust_t = 0.0
        self.active = {}          # activation key -> seconds left
        self.ball = [0.0, 0.0]    # forward (mm/s), yaw (rad/s) of the ball under the fly
        self.rates, self.senses = {}, {}
        self._sent = {}
        for name in list(self.w.channel_stim):
            if name.startswith(("body_", "act_")):
                self.w.pending.append({"cmd": "channel", "name": name, "rate": 0})

    def start(self, action, arg=None):
        if action == "loom":
            self.loom = {"side": arg if arg in ("left", "right") else "left", "d": 32.0, "theta": None}
        elif action in ("sugar", "bitter"):
            self.probe = {"kind": action, "t": 0.0}
        elif action == "dust":
            self.dust_t = 1.6
        elif action in ACTIVATE:
            self.active[action] = 2.5 if action != "gf" else 0.3
        elif action == "stop":
            self.loom, self.probe, self.dust_t, self.active = None, None, 0.0, {}

    # ------------------------------------------------------------ 1. stimuli -> sensory neurons
    def sense(self, dt):
        s = {}
        loom_rate = {"left": 0.0, "right": 0.0}
        L = self.loom
        if L:
            L["d"] -= 34.0 * dt                      # approaching at 34 mm/s
            theta = 2 * math.atan(3.0 / max(L["d"], 0.5))
            if L["theta"] is not None:
                dtheta = (theta - L["theta"]) / max(dt, 1e-6)
                if theta > math.radians(8):
                    loom_rate[L["side"]] = min(200.0, 32.0 * max(dtheta, 0))
            L["theta"] = theta
            L["size_deg"] = math.degrees(theta)
            if L["d"] < 1.0:
                self.loom = None
        s["loom_left"], s["loom_right"] = loom_rate["left"], loom_rate["right"]
        P = self.probe
        if P:
            P["t"] += dt
            if 0.35 < P["t"] < 4.2:                  # the drop touches the proboscis tip
                s[P["kind"]] = 150.0
            if P["t"] > 4.6:
                self.probe = None
        if self.dust_t > 0:
            s["touch"] = 150.0
            self.dust_t -= dt
        for k in list(self.active):
            self.active[k] -= dt
            if self.active[k] <= 0:
                del self.active[k]
        self.senses = s

        want = {f"body_{k}": (self.taste[k], s.get(k, 0.0)) for k in ("sugar", "bitter", "touch")}
        for side in ("left", "right"):
            want[f"body_loom_{side}"] = (self.lplc2[side], round(loom_rate[side], -1))
        for key, (spec, _) in ACTIVATE.items():
            want[f"act_{key}"] = (self.neurons_for(spec), 150.0 if key in self.active else 0.0)
        for name, (neurons, rate) in want.items():
            if self._sent.get(name) != rate:
                self._sent[name] = rate
                self.w.pending.append({"cmd": "channel", "name": name, "neurons": neurons, "rate": rate})

    # ------------------------------------------------------------ 3. what is the body doing?
    REFLEX = {  # built-in skill played while these command neurons fire: (readout, threshold Hz)
        "walk_forward": "Walking forwards", "walk_backward": "Walking backwards", "turn_left": "Turning left",
        "turn_right": "Turning right", "groom_antennae": "Grooming its antennae"}

    def _reflex_for(self, r):
        fwd, back, groom = r.get("forward", 0), r.get("backward", 0), r.get("groom", 0)
        steer = r.get("turn_l", 0) - r.get("turn_r", 0)
        if groom > 6:
            return "groom_antennae", "groom", f"aDN grooming command neurons at {groom:.0f} Hz"
        if abs(steer) > 20 and abs(steer) >= max(fwd, back):
            side = "left" if steer > 0 else "right"
            return f"turn_{side}", "turn_l" if steer > 0 else "turn_r", f"DNa01/DNa02 {side} steering neurons ahead by {abs(steer):.0f} Hz"
        if max(fwd, back) > 10:
            if fwd >= back:
                return "walk_forward", "forward", f"P9/DNg97 walking command neurons at {fwd:.0f} Hz"
            return "walk_backward", "backward", f"Moonwalker neurons (MDN) at {back:.0f} Hz"
        return None

    def act(self, dt, motor, r, skill=None):
        """motor: motor.Motor (updated this frame); r: behaviour readouts (Hz, ~60 ms smoothing)."""
        self.rates = r
        v = motor.value
        eng = getattr(self.w, "skills", None)
        rx = self._reflex_for(r)
        if rx and eng is not None and not eng.state and not eng.busy and eng.get(rx[0]):
            if not (self.mode == "reflex" and self._reflex == rx[0]):
                self.events.append({"event": rx[0], "key": rx[1], "why": rx[2]})
            self._reflex = rx[0]
            eng.reflex(rx[0])
        legs = ("LF", "LM", "LH", "RF", "RM", "RH")
        leg_act = {leg: max(abs(v.get(f"{leg}.{c}", 0)) for c in ("swing", "lift", "reach", "grip")) for leg in legs}
        power = max(v.get("wingL.power", 0), v.get("wingR.power", 0))
        ttm = [i for i, k in enumerate(motor.key) if k.endswith(".jump_ttm")]
        jump = float(motor.act[ttm].max()) if ttm else 0.0
        cands = [
            # label, key for the "why" trace, condition, text
            ("jump", "escape", jump > 0.3 and r.get("escape", 0) > 20, f"jump muscle motor neuron (TTMn) driven by the giant fibre ({r.get('escape', 0):.0f} Hz)"),
            ("fly", "wing.dorsal_longitudinal.L", power > 0.3, "wing power motor neurons (DLM/DVM) firing"),
            ("feed", "proboscis.proboscis_m9.L", v.get("proboscis.rostrum", 0) > 0.25, "MN9 extends the proboscis"),
            ("groom", "groom", r.get("groom", 0) > 5 and max(leg_act["LF"], leg_act["RF"]) > 0.2, f"aDN grooming neurons at {r.get('groom', 0):.0f} Hz, front legs moving"),
            ("legs", None, max(leg_act.values()) > 0.25, "leg motor neurons firing"),
        ]
        # flight: only once the feet are off the ball (after a jump), and while the power muscles stay on
        if jump > 0.3 and r.get("escape", 0) > 20:     # a giant-fibre jump lifts the feet off the ball
            self.airborne, self.air_t = True, 0.0
        elif self.airborne:
            self.air_t = self.air_t + dt if power < 0.2 else 0.0
            if self.air_t > 0.4:
                self.airborne = False
        cands[1] = ("fly", cands[1][1], self.airborne and power > 0.3, cands[1][3])
        want = next(((lab, key, why) for lab, key, cond, why in cands if cond), ("rest", None, ""))
        if skill:
            want = ("reflex", None, "") if skill.get("kind") == "reflex" else ("skill", None, "")
        # hysteresis: a new behaviour must hold for 0.1 s to show, the old one must be gone for 0.5 s
        if want[0] != self.mode:
            self._pending_t = self._pending_t + dt if self._pending == want[0] else dt
            self._pending = want[0]
            order = [c[0] for c in cands]
            stronger = self.mode == "rest" or (want[0] in order and self.mode in order and order.index(want[0]) < order.index(self.mode))
            need = 0.1 if (stronger or want[0] in ("skill", "reflex")) else 0.5
            if self._pending_t >= need:
                if want[0] == "legs":
                    key = max((f"{leg}.{c}" for leg in legs for c in ("swing", "lift", "reach", "grip")), key=lambda k: abs(v.get(k, 0)))
                    pools = motor.controls[key][0 if v.get(key, 0) >= 0 else 1]
                    want = ("legs", motor.key[pools[0]] if pools else None, f"{key} motor neurons firing")
                self._set(want[0], want[1], want[2])
        else:
            self._pending, self._pending_t = None, 0.0
        self.mode_t += dt
        # the ball turns under legs that push back while gripping
        push = 0.0
        for leg in legs:
            s_now = v.get(f"{leg}.swing", 0)
            ds = s_now - self._swing.get(leg, s_now)
            self._swing[leg] = s_now
            if v.get(f"{leg}.grip", 0) > -0.2 and ds < 0:
                push += -ds
        self.ball = [0.9 * self.ball[0] + 0.1 * (push / max(dt, 1e-3)) * 0.6, 0.0]
        self.joints = motor.joints()
        self.parts = motor.summary()

    def _set(self, mode, key, why):
        self.mode, self.mode_t = mode, 0.0
        if key:
            self.events.append({"event": mode, "key": key, "why": why})

    # ------------------------------------------------------------ to the browser
    def state(self, skill=None):
        L, P = self.loom, self.probe
        return {
            "type": "body", "mode": self.mode, "skill": skill, "airborne": self.airborne,
            "joints": self.joints, "parts": self.parts, "ball": [round(b, 3) for b in self.ball],
            "rates": {k: round(v, 1) for k, v in self.rates.items() if v >= 0.5},
            "senses": {k: round(v) for k, v in self.senses.items() if v},
            "loom": {"side": L["side"], "d": round(L["d"], 2)} if L else None,
            "probe": {"kind": P["kind"], "t": round(P["t"], 2)} if P else None,
            "dust": self.dust_t > 0,
            "active": sorted(self.active),
        }


PART_ROWS = [
    # key, label, which pools (by key prefix / part)
    ("LF", "Front-left leg", lambda p: p["key"].startswith("LF.")),
    ("RF", "Front-right leg", lambda p: p["key"].startswith("RF.")),
    ("LM", "Middle-left leg", lambda p: p["key"].startswith("LM.")),
    ("RM", "Middle-right leg", lambda p: p["key"].startswith("RM.")),
    ("LH", "Hind-left leg", lambda p: p["key"].startswith("LH.")),
    ("RH", "Hind-right leg", lambda p: p["key"].startswith("RH.")),
    ("wingL", "Left wing", lambda p: p["part"] == "wing" and p["side"] == "left"),
    ("wingR", "Right wing", lambda p: p["part"] == "wing" and p["side"] == "right"),
    ("head", "Neck (head turns)", lambda p: p["part"] == "neck"),
    ("proboscis", "Proboscis and pharynx", lambda p: p["part"] in ("proboscis", "pharynx")),
    ("antennaL", "Left antenna", lambda p: p["part"].startswith("antenna") and p["side"] == "left"),
    ("antennaR", "Right antenna", lambda p: p["part"].startswith("antenna") and p["side"] == "right"),
]
DN_ROWS = [
    ("escape", "Giant fibre DNp01", "jump motor neuron + wing power (via PSI)"),
    ("forward", "P9 / DNp09, DNg97", "forward walking command"),
    ("backward", "Moonwalker (MDN)", "backward walking command"),
    ("turn_l", "DNa01/DNa02 left", "steering"),
    ("turn_r", "DNa01/DNa02 right", "steering"),
    ("groom", "aDN (DNg62, DNge078)", "antennal grooming command"),
]


def pathways(world, motor):
    """Rows for the 'Neurons -> muscles' panel: motor neurons per body part (all in the connectome),
    and the brain's descending command neurons."""
    ro = {r["key"]: r["neurons"] for r in world.readouts}
    rows = []
    for key, label, sel in PART_ROWS:
        pools = [p for p in motor.pools if sel(p)]
        neurons = sorted({n for p in pools for n in p["neurons"]})
        if neurons:
            rows.append({"key": key, "kind": "part", "neurons": neurons, "from": label,
                         "to": f"{len(neurons)} motor neurons, {len(pools)} muscles"})
    for key, label, what in DN_ROWS:
        if ro.get(key):
            rows.append({"key": key, "kind": "dn", "neurons": ro[key], "from": label, "to": what})
    return {"rows": rows, "activate": {k: v[1] for k, v in ACTIVATE.items()}}
