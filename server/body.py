"""
A fly whose every movement comes from its motor neurons, free in an arena.

Every frame, in simulated time:

  1. The world -> sensory neurons (arena.py). Food, odours, warm or cool air, humid or dry air,
     threats and moving stripes reach the sensory neurons they would in a real fly, from where they
     are relative to its antennae, eyes and feet. Dust on the antennae fires Johnston's organ neurons.
     "Activate" buttons drive chosen descending neurons directly, like optogenetics.
  2. The whole brain and nerve cord connectome runs (BANC).
  3. Motor neurons -> muscles (motor.py) -> the body. The physical body (physics.py) walks, jumps
     and flies under gravity; its legs' sense organs report back into the nerve cord (feedback.py).
     Without MuJoCo a kinematic body stands in (joint angles straight from the muscles, movement
     estimated from leg sweeps).

The label on screen ("Feeding", "Escape jump", ...) is read from what the muscles are doing, with
some hysteresis so a brief twitch doesn't flip it back and forth.

Rules from fly physiology the connectome alone doesn't give:
- Feet on the ground stop flight (tarsal reflex): the flight motor only runs once no foot touches the
  ground while the flight power muscles fire.
Hand-built parts (ASSISTS), each switchable; skills practise with them off:
- The walking model: the walking command neurons (P9, MDN, DNa01/02, ...) set six leg oscillators
  stepping with a real fly's recorded steps (locomotion.py), the legs' premotor neurons firing along
  (cpg.py), and the aDN grooming neurons play a recorded grooming movement (a built-in skill). The
  nerve cord's rhythm circuits aren't in the connectome model.
- The flight timer: a jump starts flight at once, and flight power is held at cruise, then eased off.
"""
import json
import math
import threading

import numpy as np

import arena as arena_mod
import feedback as feedback_mod
import locomotion as loco_mod
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
RHYTHMS = ("neurons", "off")
# The parts of the body that are built by hand rather than coming from the connectome. Each can be
# switched off; practice (skills.py) runs with them off by default, so what a skill learns is the
# neurons' and muscles' own work, and the built-in versions become baselines to beat.
ASSISTS = {
    "walking": "Walking model: when the walking command neurons (P9, MDN, DNa01/02) fire, six leg oscillators step "
               "the legs with a recorded fly's steps (locomotion.py), and when the aDN grooming neurons fire a "
               "recorded grooming movement plays. Off: the legs do only what their motor neurons do.",
    "flight_timer": "Flight timer: a jump starts flight at once, and flight power is held at cruise for 1.5 s, then "
                    "eased off so the fly lands. Off: flight power is only what the wing power motor neurons give "
                    "(the flight motor still only runs with the feet off the ground).",
}
FLIGHT_CRUISE = 0.85     # flight power kept up once airborne (a hover needs ~0.8)
FLIGHT_SINK = 0.55       # ... easing off to this, so the fly sinks and lands
FLIGHT_HOLD = 1.5        # s of cruising before easing off
FLIGHT_EASE = 1.0        # s to ease off
# reinforcement, as fly labs give it: activating the mushroom body's dopamine neurons directly
ODOURS = ("fruit", "co2", "geosmin", "cva")
REINFORCE = {"reward": ("pam", "PAM dopamine neurons: reward (Liu et al. 2012; Burke et al. 2012)"),
             "punish": ("ppl1", "PPL1 dopamine neurons: punishment (Claridge-Chang et al. 2009; Aso et al. 2010)")}
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")


class Body:
    def __init__(self, world):
        self.w = world
        self.rhythm = "neurons"      # "neurons" | "off" (see the module docstring): the walking model on or off
        self.assists = {k: True for k in ASSISTS}
        self.assist_override = None  # while a learned skill plays: the switches it was learned with
        self.phys_lock = threading.RLock()
        self.walker = loco_mod.Walker()  # the legs' stepping, commanded by descending neurons (rhythm "neurons")
        self.phys = None             # physics.PhysicsBody when the physical body is on
        self.phys_error = None
        self.sensed = None
        self.fb_rates, self.fb_summary = {}, {}
        self._cmd = None             # muscle commands for the next physics step
        steps = json.loads(physics_mod.BODY_JSON.read_text())["steps"]["legs"]
        self.neutral = {leg: {d: (min(v) + max(v)) / 2 for d, v in dofs.items()} for leg, dofs in steps.items()}
        self.fb = feedback_mod.Feedback(world.presets.get("proprio", []), world.motor.ranges, self.neutral)
        self.arena = arena_mod.Arena(world)
        T = world.tables
        self.types = np.array(T["type"])[world.type]
        self.sides = np.array(T["side"])[world.side]
        self.jo = world.stim["touch"]["neurons"]
        self.events = []
        self.decisions = []       # traced "why" chains, newest last (filled by the app)
        self.pose = {"x": 0.0, "y": 0.0, "heading": 0.0}      # kinematic body's place in the arena
        self.reset()

    def set_rhythm(self, mode):
        if mode not in RHYTHMS:
            return
        self.rhythm = mode
        self.assists["walking"] = mode == "neurons"
        cpg = getattr(self.w, "cpg", None)
        if cpg is not None:
            cpg.enable(mode == "neurons")
        eng = getattr(self.w, "skills", None)
        if mode == "off" and eng is not None and eng.state and eng.state.get("kind") == "reflex":
            eng.stop()

    def set_assist(self, name, on):
        """Switch a hand-built part (ASSISTS) on or off."""
        if name == "walking":
            self.set_rhythm("neurons" if on else "off")
        elif name in ASSISTS:
            self.assists[name] = bool(on)

    def assist(self, name):
        o = self.assist_override
        if o is not None and name in o:
            return bool(o[name])
        return self.assists.get(name, True)

    def neurons_for(self, spec):
        t, _, side = spec.partition(":")
        m = self.types == t
        if side:
            m &= self.sides == side
        return np.where(m)[0].tolist()

    # ------------------------------------------------------------ which body
    def set_physics(self, on):
        """Switch between the physical body and the kinematic one. Returns whether physics is on."""
        with self.phys_lock:
            return self._set_physics(on)

    def _set_physics(self, on):
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
    def reset(self, home=False):
        """Calm the body down. home=True also puts the fly back in the middle of the arena."""
        self.mode, self.mode_t = "rest", 0.0
        self._pending, self._pending_t = None, 0.0
        self._swing = {}
        self.joints, self.parts = {}, {}
        self.airborne, self.air_t, self.flight_t, self.flight_power = False, 0.0, 0.0, 0.0
        self._jump_t = 9.0
        self._reflex = None
        self.dust_t = 0.0
        self.probe = None         # a drop touched to the proboscis by the experimenter: {"kind", "t"}
        self.active = {}          # activation key -> seconds left
        self.ball = [0.0, 0.0]    # the fly's own movement: forward (mm/s), turning (rad/s, + = right)
        self._track = None
        self.rates, self.senses = {}, {}
        self.world_rates = {}
        self._sent = {}
        self._untouched = 0.0
        self._cmd = None
        self.fb_rates, self.fb_summary = {}, {}
        self.walker.reset()
        if home:
            self.pose = {"x": 0.0, "y": 0.0, "heading": 0.0}
        with self.phys_lock:          # never while the physics thread is mid-step (MuJoCo isn't thread-safe)
            self._reset_physics(home)
        for name in list(self.w.channel_stim):
            if name.startswith(("body_", "act_", "proprio", "world")):
                self.w.pending.append({"cmd": "channel", "name": name, "rate": 0})

    def _reset_physics(self, home):
        if self.phys is not None:
            if home:
                self.phys.reset(pos=(0.0, 0.0), heading=0.0)
            elif self.sensed is not None:
                self.phys.reset(pos=(self.sensed["pos"][0], self.sensed["pos"][1]), heading=self.sensed["heading"])
            else:
                self.phys.reset()
            self.sensed = self.phys.sensed()
        else:
            self.sensed = None

    def fly(self):
        """Where the fly is and what its body is doing, for the arena's senses."""
        prob = max(0.0, self.w.motor.value.get("proboscis.rostrum", 0.0))
        if self.sensed is not None:
            return arena_mod.fly_from_physics(self.sensed, prob)
        p = self.pose
        return arena_mod.fly_from_pose(p["x"], p["y"], p["heading"], -self.ball[1], prob)

    def start(self, action, arg=None, **kw):
        """Experiments and changes to the world. action: an ACTIVATE key, "dust", "probe_sugar",
        "probe_bitter" (a drop touched to the proboscis), "stop", "clear",
        or something to put in the world: sugar, bitter, fruit, co2, geosmin, cva, heat, cold, humid, dry,
        threat (arg = side), loom (the same), stripes (arg = "left"/"right")."""
        if action == "dust":
            self.dust_t = 1.6
        elif action in ("probe_sugar", "probe_bitter"):
            self.probe = {"kind": action.split("_")[1], "t": 0.0}
        elif action in ACTIVATE:
            self.active[action] = 2.5 if action != "gf" else 0.3
        elif action in REINFORCE:
            self.active[action] = 1.0
            mb = getattr(self.w, "mb", None)
            if mb is not None:
                odours = sorted({o["odour"] for o in self.arena.objects if o["kind"] == "odour"})
                mb.log.append({"t": round(self.w.brain.t_ms / 1000, 1), "reinforcement": action, "odours": odours})
        elif action == "stop":
            self.dust_t, self.active, self.probe = 0.0, {}, None
            self.arena.objects = [o for o in self.arena.objects if o["kind"] not in ("threat", "target")]
        elif action == "clear":
            self.arena.clear()
        elif action in ("loom", "threat"):
            self.arena.add("threat", self.fly(), side=arg if arg in ("left", "right", "front", "back") else "left")
        elif action == "stripes":
            self.arena.add("stripes", self.fly(), speed=-1.5 if arg == "right" else 1.5)
        elif action == "target":
            self.arena.objects = [o for o in self.arena.objects if o["kind"] != "target"]
            self.arena.add("target", self.fly(), side=arg if arg in ("left", "right") else "left")
        elif action in ("sugar", "bitter", "fruit", "co2", "geosmin", "cva", "heat", "cold", "humid", "dry"):
            self.arena.add(action, self.fly(), **kw)

    # ------------------------------------------------------------ 1. the world -> sensory neurons
    def sense(self, dt):
        rates = self.arena.sense(dt, self.fly())
        if self.dust_t > 0:
            for n in self.jo:
                rates[n] = max(rates.get(n, 0.0), 150.0)
            self.dust_t -= dt
        P = self.probe
        if P:                                        # the drop touches the labellum from 0.35 s to 4.2 s
            P["t"] += dt
            if 0.35 < P["t"] < 4.2:
                for n in self.w.stim[P["kind"]]["neurons"]:
                    rates[n] = max(rates.get(n, 0.0), 150.0)
            if P["t"] > 4.6:
                self.probe = None
        for k in list(self.active):
            self.active[k] -= dt
            if self.active[k] <= 0:
                del self.active[k]
        self.senses = dict(self.arena.felt)
        mb = getattr(self.w, "mb", None)
        if mb is not None:            # which odours the fly smells now (for what the mushroom body learns about each)
            mb.smelling = {k.split(" ")[0] for k, v in self.senses.items() if k.split(" ")[0] in ODOURS and v >= 20}
        if self.dust_t > 0:
            self.senses["dust on the antennae"] = 150
        if self.probe and 0.35 < self.probe["t"] < 4.2:
            self.senses[f"{self.probe['kind']} on the proboscis"] = 150
        if rates != self.world_rates:
            self.world_rates = rates
            self.w.pending.append({"cmd": "channel", "name": "world", "rates": rates} if rates
                                  else {"cmd": "channel", "name": "world", "rate": 0})
        for key, (spec, _) in ACTIVATE.items():
            rate = 150.0 if key in self.active else 0.0
            if self._sent.get(key) != rate:
                self._sent[key] = rate
                self.w.pending.append({"cmd": "channel", "name": f"act_{key}", "neurons": self.neurons_for(spec), "rate": rate})
        mb = getattr(self.w, "mb", None)
        for key, (group, _) in REINFORCE.items():
            rate = 150.0 if key in self.active else 0.0
            if mb is not None and self._sent.get(key) != rate:
                self._sent[key] = rate
                self.w.pending.append({"cmd": "channel", "name": f"act_{key}",
                                       "neurons": [int(n) for n in getattr(mb, group)], "rate": rate})

    # ------------------------------------------------------------ 3. what is the body doing?
    REFLEX = {  # built-in skill played while these command neurons fire
        "walk_forward": "Walking forwards", "walk_backward": "Walking backwards", "turn_left": "Turning left",
        "turn_right": "Turning right", "groom_antennae": "Grooming its antennae"}

    def _reflex_for(self, r):
        """The built-in grooming movement plays while the aDN grooming command neurons fire (walking is
        locomotion.py's)."""
        groom = r.get("groom", 0)
        if groom > 6:
            return "groom_antennae", "groom", f"aDN grooming command neurons at {groom:.0f} Hz"
        return None

    def act(self, dt, motor, r, skill=None):
        """motor: motor.Motor (updated this frame); r: behaviour readouts (Hz, ~60 ms smoothing)."""
        self.rates = r
        v = motor.value
        eng = getattr(self.w, "skills", None)
        walking_model = self.assist("walking")
        rx = self._reflex_for(r) if walking_model else None
        W = self.walker
        moving_skill = eng is not None and eng.state and not eng.state.get("command")
        if walking_model and not moving_skill:
            # only the walking and steering command neurons command walking (other descending neurons also
            # reach the legs' networks: counting them set the fly walking while it tasted sugar)
            r_all = self.w.all_readouts() if hasattr(self.w, "all_readouts") else r
            W.command(r_all, dt, yaw_rate=-self.ball[1])
        else:
            W.command({}, dt)
        if cpg_follow := getattr(self.w, "cpg", None):
            cpg_follow.follow(W)
        if rx and eng is not None and not eng.state and not eng.busy and eng.get(rx[0]):
            if not (self.mode == "reflex" and self._reflex == rx[0]):
                self.events.append({"event": rx[0], "key": rx[1], "why": rx[2]})
            self._reflex = rx[0]
            eng.reflex(rx[0])
        leg_act = {leg: max(abs(v.get(f"{leg}.{c}", 0)) for c in ("swing", "lift", "reach", "grip")) for leg in LEGS}
        power = max(v.get("wingL.power", 0), v.get("wingR.power", 0))
        ttm = [i for i, k in enumerate(motor.key) if k.endswith(".jump_ttm")]
        jump = float(motor.act[ttm].max()) if ttm else 0.0
        fwd, turn = self.ball[0], -self.ball[1]
        walking = self.sensed is not None and abs(fwd) > 1.0 and max(leg_act.values()) > 0.2
        cands = [
            # label, key for the "why" trace, condition, text
            ("jump", "escape", jump > 0.3 and r.get("escape", 0) > 20, f"jump muscle motor neuron (TTMn) driven by the giant fibre ({r.get('escape', 0):.0f} Hz)"),
            ("fly", "wing.dorsal_longitudinal.L", power > 0.3, "wing power motor neurons (DLM/DVM) firing"),
            ("feed", "proboscis.proboscis_m9.L", v.get("proboscis.rostrum", 0) > 0.25, "MN9 extends the proboscis"),
            ("groom", "groom", r.get("groom", 0) > 5 and max(leg_act["LF"], leg_act["RF"]) > 0.2, f"aDN grooming neurons at {r.get('groom', 0):.0f} Hz, front legs moving"),
            ("walk", self._walk_key(r), walking or W.w > 0.3, W.why or "legs stepping"),
            ("legs", None, max(leg_act.values()) > 0.25, "leg motor neurons firing"),
        ]
        # flight: the tarsal reflex. With feet on the ground the flight motor is held off; it starts once no
        # foot has touched the ground for 0.1 s while the flight power muscles fire (after a jump, or when
        # lifted), and losing contact keeps it going (in real flies, a fly without ground under its feet
        # keeps flying): at a cruising power for FLIGHT_HOLD seconds, then easing off so the fly sinks and
        # lands; landing on three or more feet ends it
        if self.sensed is not None:
            touching = sum(self.sensed["touch"].values())
            self._untouched = self._untouched + dt if touching == 0 else 0.0
            # an escape jump starts flight straight away: the giant fibre drives the flight power muscles
            # (via PSI) as well as the jump muscle
            self._jump_t = 0.0 if jump > 0.3 else self._jump_t + dt
            timer = self.assist("flight_timer")
            took_off = timer and self._jump_t < 0.5 and self._untouched > 0.03
            if not self.airborne and (took_off or (self._untouched > (0.1 if timer else 0.03) and power > 0.2)):
                self.airborne, self.air_t, self.flight_t = True, 0.0, 0.0
            elif self.airborne:
                self.flight_t += dt
                self.air_t = self.air_t + dt if (touching >= 3 and self.flight_t > 0.3) else 0.0
                if self.air_t > 0.3:
                    self.airborne = False
        else:
            if jump > 0.3 and r.get("escape", 0) > 20:
                self.airborne, self.air_t, self.flight_t = True, 0.0, 0.0
            elif self.airborne:
                self.flight_t += dt
                if self.flight_t > FLIGHT_HOLD + FLIGHT_EASE:
                    self.airborne = False
        if self.airborne and not self.assist("flight_timer"):
            self.flight_power = power               # only what the wing power motor neurons give
        elif self.airborne:
            ease = min(1.0, max(0.0, (self.flight_t - FLIGHT_HOLD) / FLIGHT_EASE))
            self.flight_power = max(power, FLIGHT_CRUISE + (FLIGHT_SINK - FLIGHT_CRUISE) * ease)
        else:
            self.flight_power = 0.0
        power = max(power, self.flight_power)
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
                if want[0] == "legs" or (want[0] == "walk" and not want[1]):
                    key = max((f"{leg}.{c}" for leg in LEGS for c in ("swing", "lift", "reach", "grip")), key=lambda k: abs(v.get(k, 0)))
                    pools = motor.controls[key][0 if v.get(key, 0) >= 0 else 1]
                    want = (want[0], motor.key[pools[0]] if pools else None, f"{key} motor neurons firing")
                self._set(want[0], want[1], want[2])
        else:
            self._pending, self._pending_t = None, 0.0
        self.mode_t += dt
        self.joints = motor.joints()
        self.parts = motor.summary()
        if self.phys is not None:
            # physical body: the muscles' commands go to the next physics step; the pose shown is what
            # the physics made of the last ones
            stiff, adh = motor.muscles()
            target = {(leg, d): self.neutral[leg][d] + off for leg, dofs in self.joints["legs"].items() for d, off in dofs.items()}
            steer = v.get("wingR.stroke", 0) - v.get("wingL.stroke", 0)      # a bigger right stroke turns left
            flight = {"power": power if self.airborne else 0.0, "steer": steer, "jump": jump}
            self._cmd = (target, stiff, adh, flight, W.advance if (W.walking or W.w > 0) else None)
            if self.sensed is not None:
                A = self.sensed["angle"]
                self.joints["legs"] = {leg: {d: round(A[(leg, d)] - self.neutral[leg][d], 4) for d in physics_mod.DOFS}
                                       for leg in physics_mod.LEGS}
                # how the fly moves, as a tracker would measure it: from its position and heading over
                # time (the instantaneous body velocity jitters with every step), ~0.1 s smoothing
                th, hd = self.sensed["thorax"], self.sensed["heading"]
                if self._track is not None:
                    (x0, y0, h0) = self._track
                    dx, dy = th[0] - x0, th[1] - y0
                    fwd = (dx * math.cos(h0) + dy * math.sin(h0)) / max(dt, 1e-3)
                    turn = math.remainder(hd - h0, 2 * math.pi) / max(dt, 1e-3)       # + = left
                    k = 1.0 - math.exp(-dt / 0.1)
                    self.ball = [self.ball[0] + k * (fwd - self.ball[0]), self.ball[1] + k * (-turn - self.ball[1])]
                self._track = (th[0], th[1], hd)
            return
        # kinematic body: stepping legs follow the walking oscillators; movement is estimated from them,
        # or else from the legs in stance (not lifted): sweeping back pushes the fly forwards, sweeping
        # forwards pushes it backwards, left vs right turns it
        if W.walking or W.w > 0:
            w, tg, _ = W.advance(dt)
            for (leg, d), a in tg.items():
                J = self.joints["legs"].setdefault(leg, {})
                J[d] = round((1 - w) * J.get(d, 0.0) + w * (a - self.neutral[leg][d]), 4)
            f, t = W.estimate()
            k = 0.2
            self.ball = [(1 - k) * self.ball[0] + k * f, (1 - k) * self.ball[1] + k * -t]
            p = self.pose
            p["heading"] -= self.ball[1] * dt
            p["x"] += self.ball[0] * dt * math.cos(p["heading"])
            p["y"] += self.ball[0] * dt * math.sin(p["heading"])
            lim = physics_mod.ARENA - 2
            p["x"], p["y"] = max(-lim, min(lim, p["x"])), max(-lim, min(lim, p["y"]))
            return
        fwd_l = fwd_r = 0.0
        for leg in LEGS:
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
        p = self.pose
        p["heading"] -= self.ball[1] * dt
        p["x"] += self.ball[0] * dt * math.cos(p["heading"])
        p["y"] += self.ball[0] * dt * math.sin(p["heading"])
        lim = physics_mod.ARENA - 2
        p["x"], p["y"] = max(-lim, min(lim, p["x"])), max(-lim, min(lim, p["y"]))

    def physics_step(self, dt):
        """Run the physical body for dt seconds on the last muscle commands, then report what the legs'
        sense organs feel to the nerve cord. Runs alongside the next brain frame (a one-frame, ~33 ms,
        sensorimotor delay, about what a fly has)."""
        with self.phys_lock:
            ph = self.phys
            if ph is None or self._cmd is None:
                return
            self.sensed = ph.step(dt, *self._cmd)
        eng = getattr(self.w, "skills", None)
        moving = bool(eng is not None and eng.state) or self.walker.w > 0.1 or self.assist("walking") and self.mode in ("legs", "groom")
        rates = self.fb.rates(self.sensed, moving, dt)
        if rates != self.fb_rates:
            self.fb_rates = rates
            self.w.pending.append({"cmd": "channel", "name": "proprio", "rates": rates} if rates
                                  else {"cmd": "channel", "name": "proprio", "rate": 0})
        self.fb_summary = self.fb.summary(rates)

    def _walk_key(self, r):
        """Which command readout to trace back through the wiring when the fly walks."""
        keys = {"forward": r.get("forward", 0), "backward": r.get("backward", 0), "turn_l": r.get("turn_l", 0),
                "turn_r": r.get("turn_r", 0)}
        k, v = max(keys.items(), key=lambda kv: kv[1])
        return k if v > 5 else None

    def _set(self, mode, key, why):
        self.mode, self.mode_t = mode, 0.0
        if key:
            self.events.append({"event": mode, "key": key, "why": why})

    # ------------------------------------------------------------ to the browser
    def state(self, skill=None):
        return {
            "type": "body", "mode": self.mode, "skill": skill, "airborne": self.airborne,
            "flight_power": round(self.flight_power, 2),
            "joints": self.joints, "parts": self.parts, "ball": [round(b, 3) for b in self.ball],
            "rates": {k: round(v, 1) for k, v in self.rates.items() if v >= 0.5},
            "senses": self.senses,
            "dust": self.dust_t > 0,
            "probe": {"kind": self.probe["kind"], "t": round(self.probe["t"], 2)} if self.probe else None,
            "active": sorted(self.active),
            "fly": self.fly_pose(),
            "arena": self.arena.view(),
            "physics": self.physics_info(),
            "memory": self.memory(),
        }

    def memory(self):
        mb = getattr(self.w, "mb", None)
        if mb is None:
            return None
        return {"learning": mb.learning, "odours": mb.memories(), "changed": dict(list(mb.summary().items())[:6]),
                "pairings": mb.log[-6:]}

    def fly_pose(self):
        """Where the fly's root body is in the arena (mm) and which way up (quaternion w, x, y, z)."""
        if self.sensed is not None:
            return {"pos": [round(x, 4) for x in self.sensed["pos"]], "quat": [round(x, 5) for x in self.sensed["quat"]],
                    "heading": round(self.sensed["heading"], 4)}
        p = self.pose
        h = p["heading"]
        return {"pos": [round(p["x"], 4), round(p["y"], 4), None], "quat": [math.cos(h / 2), 0, 0, math.sin(h / 2)],
                "heading": round(h, 4)}

    def physics_info(self):
        ph = self.phys
        info = {"on": ph is not None, "available": physics_mod.available(), "error": self.phys_error,
                "rhythm": self.rhythm, "feedback": self.fb.gain,
                "assists": dict(self.assists), "assist_override": self.assist_override}
        if ph is not None:
            info.update({"cost": round(ph.cost, 2), "touch": self.sensed["touch"] if self.sensed else {},
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
