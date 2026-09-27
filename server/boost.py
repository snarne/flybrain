"""
The sense boost: the connectome's own paths from the senses to the walking and steering command
neurons, made strong enough to steer the fly. A hand-built part (body.ASSISTS["sense_boost"]).

What the wiring says. Following the signed connections through up to six synapses (each step weighted
by its share of the next neuron's input), the connectome points the right way for several innate
behaviours: fruit odour at the left antenna excites the left steering neurons (DNa02) far more than
the right ones, and the right antenna the right ones (turn towards the smell); CO2 excites the
moonwalker neurons (MDN: back away); warmth on one side excites the steering neurons of the other side
(turn away from heat); cold inhibits walking. But those paths run through 4-6 layers where each link
carries a small share of the next neuron's input, and in the spiking model the signal dies out on the
way: the command neurons don't fire, and the fly doesn't move.

What the boost does. Every frame, each sense's firing (its sensory neurons' rate, relative to a strong
stimulus) is multiplied by the wiring's influence on each command neuron group, and that is given to
the command neurons as synaptic input (excitatory or inhibitory, as the wiring's sign says). The
direction and the left/right pattern come from the connectome; only the overall strength is set by
hand (each sense scaled so a strong stimulus gives a clear command). The influence is recomputed on
the wiring actually running: silenced neurons carry nothing (silence the antennal lobe's projection
neurons and the fly no longer steers by smell), and the shuffled control wiring gives no consistent
steering.

Learned odours. Mushroom body learning weakens a paired odour's Kenyon cell -> output neuron synapses
(plasticity.py), but the output neurons fire too little in this model to reach the command neurons. So
the boost also reads what was learned from those synapses: an odour whose input to the punishment-
gated output neurons has weakened (an aversive memory) is steered away from, one whose input to the
reward-gated ones has weakened (appetitive), towards. That the punishment compartments' output neurons
promote approach, so weakening them means avoidance, is the mushroom body's valence logic (Aso et al.
2014; Owald & Waddell 2015); how strongly a memory steers is set by hand (LEARN_GAIN).
"""
import threading

import numpy as np
from scipy import sparse

HOPS = 6
COMMANDS = ("forward", "backward", "turn_l", "turn_r")
SENSES = ("fruit", "co2", "geosmin", "cva", "heat", "cold", "humid", "dry")
EVENTS_PER_UNIT = 4000.0   # synaptic events/s (1 mV each) to a command neuron per unit of command
# how strongly each sense drives the commands, relative to its strongest path in the wiring: set by hand
# against the published behaviours (tools/benchmark.py): temperature has to move the fly out of a warm
# patch within seconds (Simoes et al. 2021), an odour to turn it towards a source 6 mm away without
# overshooting it
SENSE_GAIN = {"heat": 1.5, "cold": 1.5}
LEARN_GAIN = 1.5           # units of steering per unit of learned memory (0..1), towards or away
CONTRAST = 30.0            # enhancement of the left-right difference between the antennae (the plume differs by
                           # a few percent across them)
QUANT = 0.05               # command units are rounded to this, so small wobbles don't rebuild the input


class SenseBoost:
    def __init__(self, world):
        self.w = world
        self.groups = []           # (sense, side, neurons)
        S = world.presets.get("senses", {})
        for sense in SENSES:
            src = S.get("odour", {}).get(sense) or S.get(sense)
            if not src:
                continue
            for side in ("left", "right"):
                ns = np.asarray(src.get(side, []), np.int64)
                if len(ns):
                    self.groups.append((sense, side, ns))
        self.typical = {s: float(world.stim[s]["rate"]) if s in world.stim else 150.0 for s in SENSES}
        self.targets = {r["key"]: np.asarray(r["neurons"], np.int64) for r in world.readouts if r["key"] in COMMANDS}
        self.I = None              # (groups x commands) influence on the wiring now running
        self.scale = None          # per sense, from the connectome itself (fixed)
        self._key = None
        self._busy = False
        self._sent = None
        self.command = {k: 0.0 for k in COMMANDS}
        self.parts = {}            # sense -> {command: units}, for the "why" text
        self.compute()

    # ------------------------------------------------------------ the wiring's influence
    def _influence(self, blocked):
        w = self.w
        ip, ix, wt = w.wiring
        N = w.N
        rows = np.repeat(np.arange(N, dtype=np.int32), np.diff(ip))
        inp = np.bincount(ix, weights=np.abs(wt), minlength=N)
        A = sparse.csr_matrix((wt / np.maximum(inp[ix], 1.0), (ix, rows)), shape=(N, N))   # A[post, pre]
        keep = np.ones(N)
        if len(blocked):
            keep[np.asarray(sorted(blocked), np.int64)] = 0.0
        X = np.zeros((N, len(self.groups)))
        for g, (_, _, ns) in enumerate(self.groups):
            X[ns, g] = 1.0 / len(ns)
        X *= keep[:, None]
        tot = np.zeros_like(X)
        for _ in range(HOPS):
            X = (A @ X) * keep[:, None]
            tot += X
        return np.array([[tot[self.targets[c], g].mean() if c in self.targets else 0.0 for c in COMMANDS]
                         for g in range(len(self.groups))])

    def _state_key(self):
        w = self.w
        blocked = set(w.blocked)
        for v in w.ablated.values():
            blocked.update(int(i) for i in v)
        return (w.variant, getattr(w, "variant_seed", 0), tuple(sorted(blocked))), blocked

    def compute(self):
        key, blocked = self._state_key()
        I = self._influence(blocked)
        if self.scale is None:
            # each sense's strongest effect on any command group, on the connectome as it is (no
            # silencing, the real wiring): a strong stimulus of that sense gives one unit of command
            I0 = I if (self.w.variant == "real" and not blocked) else self._influence(set())
            self.scale = {}
            for s in SENSES:
                rows = [g for g, (sense, _, _) in enumerate(self.groups) if sense == s]
                m = float(np.abs(I0[rows]).max()) if rows else 0.0
                self.scale[s] = 1.0 / m if m > 0 else 0.0
        self.I, self._key = I, key

    def _recompute_async(self):
        if self._busy:
            return
        self._busy = True

        def run():
            try:
                self.compute()
            finally:
                self._busy = False
        threading.Thread(target=run, daemon=True).start()

    def _memories(self):
        """What the mushroom body has learned, per odour (re-read twice a second)."""
        t = self.w.brain.t_ms
        if getattr(self, "_mem_t", None) is None or abs(t - self._mem_t) > 500:
            self._mem_t = t
            mb = getattr(self.w, "mb", None)
            try:
                self._mem = mb.memories() if mb is not None and LEARN_GAIN > 0 else {}
            except Exception:
                self._mem = {}
        return self._mem

    # ------------------------------------------------------------ every frame
    def _pattern(self, sense):
        """The wiring's effect of a sense on the commands, left and right averaged (the connectome's left-right
        differences are partly those of one fly's reconstruction): forward, backward, turn towards the side
        that senses it, turn away from it; per unit of input on one side. None if it has no groups."""
        rows = {side: g for g, (s, side, _) in enumerate(self.groups) if s == sense}
        if not rows or self.I is None:
            return None
        k = self.scale[sense] * SENSE_GAIN.get(sense, 1.0)
        L = self.I[rows["left"]] * k if "left" in rows else None
        R = self.I[rows["right"]] * k if "right" in rows else None
        if L is None:
            L = R[[0, 1, 3, 2]]
        if R is None:
            R = L[[0, 1, 3, 2]]
        # commands: forward, backward, turn_l, turn_r
        return {"fwd": (L[0] + R[0]) / 2, "back": (L[1] + R[1]) / 2,
                "toward": (L[2] + R[3]) / 2, "away": (L[3] + R[2]) / 2}

    def update(self, on):
        """After each brain frame: the senses' firing -> synaptic input to the command neurons."""
        w = self.w
        cmd = {k: 0.0 for k in COMMANDS}
        parts = {}
        if on and self.I is not None:
            if self._state_key()[0] != self._key:
                self._recompute_async()          # silenced neurons or the wiring changed
            mem = self._memories()
            stim = w.brain.stim_rates             # what the sensory neurons are firing at (Poisson input)
            for sense in SENSES:
                pat = self._pattern(sense)
                if pat is None:
                    continue
                x = {}
                for g, (s, side, ns) in enumerate(self.groups):
                    if s == sense:
                        x[side] = sum(stim.get(int(n), 0.0) for n in ns) / len(ns) / self.typical[sense]
                xl, xr = x.get("left", 0.0), x.get("right", 0.0)
                mean = (xl + xr) / 2
                if mean < 0.02:
                    continue
                # left-right contrast: the difference between the antennae, enhanced as the antennal lobe's
                # bilateral circuits do, so a source to one side steers the fly
                el = max(0.0, mean + CONTRAST * (xl - mean))
                er = max(0.0, mean + CONTRAST * (xr - mean))
                c = {"forward": pat["fwd"] * 2 * mean, "backward": pat["back"] * 2 * mean,
                     "turn_l": pat["toward"] * el + pat["away"] * er, "turn_r": pat["toward"] * er + pat["away"] * el}
                m = mem.get(sense)
                if m:
                    # what the mushroom body learned: an aversive memory steers away, an appetitive one towards
                    v = m.get("appetitive", 0.0) - m.get("aversive", 0.0)
                    c["turn_l"] += LEARN_GAIN * v * (el - er) / 2 + LEARN_GAIN * abs(v) * mean * 0.5
                    c["turn_r"] += LEARN_GAIN * v * (er - el) / 2 + LEARN_GAIN * abs(v) * mean * 0.5
                    c["forward"] += LEARN_GAIN * v * mean
                for k, v in c.items():
                    cmd[k] += v
                    if abs(v) > 0.1:
                        parts.setdefault(sense, {})[k] = v
        self.command, self.parts = cmd, parts
        drive = {}
        for k, v in cmd.items():
            v = round(v / QUANT) * QUANT
            if abs(v) < QUANT or k not in self.targets:
                continue
            ev = EVENTS_PER_UNIT * min(abs(v), 3.0)
            for n in self.targets[k]:
                drive[int(n)] = (ev, 1.0 if v > 0 else -1.0)
        key = tuple(sorted(drive.items()))
        if key != self._sent:
            self._sent = key
            w.brain.set_syn_drive(drive, "boost")

    def why(self):
        """Which senses are boosting which commands now, for the body's 'why' text."""
        names = {"forward": "forward", "backward": "backward", "turn_l": "left turn", "turn_r": "right turn"}
        out = []
        for sense, d in self.parts.items():
            k, v = max(d.items(), key=lambda kv: abs(kv[1]))
            out.append(f"{sense} {'drives' if v > 0 else 'holds back'} the {names[k]} neurons")
        return ", ".join(out)
