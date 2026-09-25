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

The body comes in two kinds:
- physical (default when MuJoCo is installed; physics.py): the NeuroMechFly body with gravity, contacts
  and a real ball; muscles set each joint's target angle and stiffness; the legs' sense organs report
  back into the nerve cord every frame (feedback.py), so the brain feels what its body did.
- kinematic: joint angles straight from the muscles, ball estimated from leg sweeps.

Two rules from fly physiology the connectome alone doesn't give:
- Flight needs the feet off the ground. Tarsal contact inhibits flight, so a fly standing on the ball
  doesn't beat its wings even if the flight power muscles' motor neurons fire; the giant-fibre jump
  lifts the legs off and flight can start.
- Rhythms: walking and grooming need nerve-cord rhythm generators the model lacks. When the brain's
  command neurons for them fire, the matching built-in skill supplies the rhythm (skills.py). This
  can be switched off ("connectome only") to see what the wiring does by itself.
"""
import json
import math
from pathlib import Path

import numpy as np

import feedback as feedback_mod
import physics as physics_mod

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
        self.rhythms = True          # built-in rhythms for walking/grooming when their command neurons fire
        self.phys = None             # physics.PhysicsBody when the physical body is on
        self.phys_error = None
        self.sensed = None
        self.fb_rates, self.fb_summary = {}, {}
        self._cmd = None             # muscle commands for the next physics step
        steps = json.loads(physics_mod.BODY_JSON.read_text())["steps"]["legs"]
        self.neutral = {leg: {d: (min(v) + max(v)) / 2 for d, v in dofs.items()} for leg, dofs in steps.items()}
        self.fb = feedback_mod.Feedback(world.presets.get("proprio", []), world.motor.ranges, self.neutral)
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

    # ------------------------------------------------------------ which body
    def set_physics(self, on):
        """Switch between the physical body and the kinematic one. Returns whether physics is on."""
        if on and self.phys is None:
            try:
                self.phys = physics_mod.PhysicsBody(self.neutral)
                self.phys_error = None
            except Exception as e:           # MuJoCo missing or failed: stay kinematic, say why
                self.phys, self.phys_error = None, str(e)
        elif not on and self.phys is not None:
            self.phys = None
            self.w.pending.append({"cmd": "channel", "name": "proprio", "rate": 0})
        self._cmd, self.sensed = None, None
        return self.phys is not None

    def set_feedback(self, gain):
        self.fb.gain = float(max(0.0, min(2.0, gain)))
        if self.fb.gain == 0:
            self.w.pending.append({"cmd": "channel", "name": "proprio", "rate": 0})

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
        self._untouched = 0.0
        self._cmd = None
        self.sensed = None
        self.fb_rates, self.fb_summary = {}, {}
        if self.phys is not None:
            self.phys.reset()
        for name in list(self.w.channel_stim):
            if name.startswith(("body_", "act_", "proprio")):
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
        rx = self._reflex_for(r) if self.rhythms else None
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
        if self.sensed is not None:
            # physical body: the tarsal reflex reads real contact; with no foot on the ball for 0.2 s the
            # flight motor is released, as when a fly is lifted off the ground
            touching = sum(self.sensed["touch"].values())
            self._untouched = self._untouched + dt if touching == 0 else 0.0
            if self._untouched > 0.2 and power > 0.3:
                self.airborne, self.air_t = True, 0.0
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
        self.joints = motor.joints()
        self.parts = motor.summary()
        if self.phys is not None:
            # physical body: the muscles' commands go to the next physics step; the pose and ball shown
            # are what the physics made of the last ones
            stiff, adh = motor.muscles()
            target = {(leg, d): self.neutral[leg][d] + off for leg, dofs in self.joints["legs"].items() for d, off in dofs.items()}
            self._cmd = (target, stiff, adh)
            if self.sensed is not None:
                A = self.sensed["angle"]
                self.joints["legs"] = {leg: {d: round(A[(leg, d)] - self.neutral[leg][d], 4) for d in physics_mod.DOFS}
                                       for leg in physics_mod.LEGS}
                self.ball = [self.sensed["forward"], -self.sensed["yaw"]]
            return
        # kinematic body: the ball turns under the legs that are on it: a leg in stance (not lifted)
        # sweeping backwards pushes the fly forwards, sweeping forwards pushes it backwards; left vs
        # right turns it
        fwd_l = fwd_r = 0.0
        for leg in legs:
            s_now = v.get(f"{leg}.swing", 0)
            ds = s_now - self._swing.get(leg, s_now)
            self._swing[leg] = s_now
            if v.get(f"{leg}.lift", 0) < 0.15:               # foot down
                push = -ds / max(dt, 1e-3)                    # + = foot moving back = fly moving forward
                if leg[0] == "L":
                    fwd_l += push
                else:
                    fwd_r += push
        k = 0.12
        self.ball = [(1 - k) * self.ball[0] + k * 0.6 * (fwd_l + fwd_r) / 2,
                     (1 - k) * self.ball[1] + k * 0.35 * (fwd_l - fwd_r)]   # + = left legs push harder = turning right

    def physics_step(self, dt):
        """Run the physical body for dt seconds on the last muscle commands, then report what the legs'
        sense organs feel to the nerve cord. Runs alongside the next brain frame (a one-frame, ~33 ms,
        sensorimotor delay, about what a fly has)."""
        ph = self.phys
        if ph is None or self._cmd is None:
            return
        self.sensed = ph.step(dt, *self._cmd)
        eng = getattr(self.w, "skills", None)
        moving = bool(eng is not None and eng.state)     # a rhythm or skill is running: gate claw/hook
        rates = self.fb.rates(self.sensed, moving)
        if rates != self.fb_rates:
            self.fb_rates = rates
            self.w.pending.append({"cmd": "channel", "name": "proprio", "rates": rates})
        self.fb_summary = self.fb.summary(rates)

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
            "physics": self.physics_info(),
        }

    def physics_info(self):
        ph = self.phys
        info = {"on": ph is not None, "available": physics_mod.available(), "error": self.phys_error,
                "rhythms": self.rhythms, "feedback": self.fb.gain}
        if ph is not None:
            info.update({"ball_z": round(ph.ball_z, 4), "ball_q": [round(x, 5) for x in ph.ball_quat()],
                         "cost": round(ph.cost, 2), "touch": self.sensed["touch"] if self.sensed else {},
                         "senses": {k: v for k, v in self.fb_summary.items() if v > 0}})
        return info


INTERNAL = ("crop", "spiracle", "salivary_gland", "uterus", "eye", "unknown")
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
    ("haltereL", "Left haltere", lambda p: p["part"] == "haltere" and p["side"] == "left"),
    ("haltereR", "Right haltere", lambda p: p["part"] == "haltere" and p["side"] == "right"),
    ("abdomen", "Abdomen", lambda p: p["part"] == "abdomen"),
    ("internal", "Internal: crop, spiracles, salivary gland, uterus, eye", lambda p: p["part"] in INTERNAL),
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
