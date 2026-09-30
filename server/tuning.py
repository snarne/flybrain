"""
Tuning the leg circuits inside the real wiring (the learned-walking experiment).

The question: can the connectome's own nerve-cord circuits make the fly step, with the hand-built
walking model off, if practice is allowed to adjust them the way real nervous systems adjust
themselves: the strength of existing connections and how excitable neurons are, not who connects to
whom?

What practice may change, per cell type rather than per synapse (a type is the same neuron repeated
for each leg and side, so a setting applies to all six legs alike, which keeps the search small and
the result symmetric):
  - gain of the connections between two cell types (the strongest pairs among the leg motor neurons,
    their premotor interneurons and the forward-walking command neurons): synapse counts stay as the
    connectome has them, their strength is scaled;
  - tonic drive to a premotor type (a steady background input, like neuromodulation);
  - fast adaptation of a premotor type (a second, ~100 ms adaptation current: sim.py's burst current),
    which lets a group of neurons fire in bursts, the standard ingredient of a rhythm generator.

Everything else (the other ~165,000 neurons and their connections) stays as it is.

What counts as walking (score_walk): with the walking model off, drive the forward command neurons
(P9, DNp09) and measure: how far the fly moves forwards; whether each leg's joints oscillate at a
stepping rhythm (2-12 Hz) with a real step's amplitude; whether the legs are coordinated in a tripod
(front and hind of one side with the middle of the other, alternating with the other three), as in
the recorded steps; and that the legs are quiet before the command. Hopping (the jump muscle),
flapping (flight muscles) and falling over count against it.
"""
import math

import numpy as np

LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]
TRIPOD = {"LF": 0, "RM": 0, "LH": 0, "RF": 1, "LM": 1, "RH": 1}
LEG_PARTS = ("front_leg", "middle_leg", "hind_leg")
COMMAND_TYPES = ("DNp09",)              # forward walking command (P9)
DOFS = ("Coxa", "Femur", "Tibia")       # the joints scored for rhythm

# parameter scales (the search works in normalised units)
GAIN_UNIT, GAIN_LO, GAIN_HI = 0.5, -3.0, 1.5      # log gain: x0.05 .. x4.5
TONIC_MAX = 3000.0                                 # events/s (1 mV each) at full tonic drive
BURST_MAX = 3.0                                    # burst_scale at full fast adaptation (x d_burst mV per spike)


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


class LegCircuits:
    """The adjustable part of the wiring, found from the connectome running now (real or a control)."""

    def __init__(self, world, n_types=48, n_pairs=160, min_syn=30):
        w = world
        ip, ix, W = w.wiring
        N = w.N
        names = np.array(w.tables["type"], dtype=object)
        sup = np.array(w.tables["super_class"])[w.sup]
        typ = w.type
        K = len(names)
        # the leg motor neurons, and the cell types they belong to
        mn = sorted({n for p in w.presets["pools"] if p.get("part") in LEG_PARTS for n in p["neurons"]})
        is_mn = np.zeros(N, bool)
        is_mn[mn] = True
        rows = np.repeat(np.arange(N, dtype=np.int64), np.diff(ip))
        absW = np.abs(W.astype(np.float32))
        # premotor types: nerve-cord interneurons, ranked by synapses onto leg motor neurons
        onto = is_mn[ix] & (sup[rows] == "ventral_nerve_cord_intrinsic")
        per_type = np.bincount(typ[rows[onto]], weights=absW[onto], minlength=K)
        per_type[np.array([not n for n in names])] = 0            # unnamed neurons aren't a type
        self.premotor = [int(k) for k in np.argsort(-per_type)[:n_types] if per_type[k] > 0]
        mn_types = sorted({int(t) for t in typ[mn] if names[t]})
        cmd = [names.tolist().index(c) for c in COMMAND_TYPES if c in names.tolist()]
        nodes = sorted(set(self.premotor) | set(mn_types) | set(cmd))
        in_s = np.zeros(K, bool)
        in_s[nodes] = True
        # connections between those types, strongest first
        m = in_s[typ[rows]] & in_s[typ[ix]]
        pos = np.where(m)[0]
        keys = typ[rows[pos]].astype(np.int64) * K + typ[ix[pos]]
        uk, inv = np.unique(keys, return_inverse=True)
        tot = np.bincount(inv, weights=absW[pos])
        order = [i for i in np.argsort(-tot) if tot[i] >= min_syn][:n_pairs]
        self.pairs = []            # (pre type name, post type name, synapse positions in the CSR arrays)
        for i in order:
            sel = pos[inv == i]
            self.pairs.append((names[uk[i] // K], names[uk[i] % K], sel))
        self.type_neurons = {k: np.where(typ == k)[0] for k in self.premotor}
        self.names = names
        self.n_pairs, self.n_types = len(self.pairs), len(self.premotor)
        self.n = self.n_pairs + 2 * self.n_types
        self.base = None           # the brain's weights at these synapses, before any tuning

    # ------------------------------------------------------------ parameters
    def labels(self):
        return ([f"gain {a} -> {b}" for a, b, _ in self.pairs]
                + [f"tonic {self.names[k]}" for k in self.premotor]
                + [f"fast adaptation {self.names[k]}" for k in self.premotor])

    def x0(self):
        """Where the search starts: the connectome as it is (gain 1), a little tonic drive, a little
        fast adaptation."""
        return np.concatenate([np.zeros(self.n_pairs), np.full(self.n_types, -2.0), np.full(self.n_types, -1.0)])

    def decode(self, x):
        x = np.asarray(x, float)
        P, T = self.n_pairs, self.n_types
        gains = np.exp(np.clip(x[:P] * GAIN_UNIT, GAIN_LO, GAIN_HI))
        tonic = np.array([TONIC_MAX * _sig(v - 2.0) for v in x[P:P + T]])
        burst = np.array([BURST_MAX * _sig(v - 1.0) for v in x[P + T:P + 2 * T]])
        return gains, tonic, burst

    def apply(self, brain, x=None):
        """Set the brain's leg circuits to parameters x (None: back to the connectome as it is)."""
        if self.base is None:
            self.base = [brain.weights[sel].copy() for _, _, sel in self.pairs]
        if x is None:
            for (_, _, sel), b in zip(self.pairs, self.base):
                brain.weights[sel] = b
            brain.set_syn_drive({}, "tune")
            for k, ns in self.type_neurons.items():
                brain.burst_scale[ns] = 0.0
            return
        gains, tonic, burst = self.decode(x)
        for (_, _, sel), b, g in zip(self.pairs, self.base, gains):
            brain.weights[sel] = b * np.float32(g)
        drive = {}
        for (k, ns), r in zip(self.type_neurons.items(), tonic):
            if r > 1.0:
                for n in ns:
                    drive[int(n)] = (float(r), 1.0)
        brain.set_syn_drive(drive, "tune")
        for (k, ns), bs in zip(self.type_neurons.items(), burst):
            brain.burst_scale[ns] = float(bs)

    def describe(self, x, top=12):
        """The biggest changes from the connectome as it is, for the report."""
        gains, tonic, burst = self.decode(x)
        g = sorted(((math.log(v), f"{a} -> {b}", v) for (a, b, _), v in zip(self.pairs, gains)), key=lambda t: -abs(t[0]))
        out = {"gains": [(lab, round(float(v), 3)) for _, lab, v in g[:top]],
               "tonic": sorted(((str(self.names[k]), round(float(r), 1)) for k, r in zip(self.premotor, tonic)), key=lambda t: -t[1])[:top],
               "fast_adaptation": sorted(((str(self.names[k]), round(float(b), 3)) for k, b in zip(self.premotor, burst)), key=lambda t: -t[1])[:top]}
        return out


# ---------------------------------------------------------------- scoring a walk
def _spectrum(sig, dt):
    s = np.asarray(sig, float)
    s = s - np.polyval(np.polyfit(np.arange(len(s)), s, 1), np.arange(len(s))) if len(s) > 3 else s - s.mean()
    win = np.hanning(len(s))
    F = np.fft.rfft(s * win)
    f = np.fft.rfftfreq(len(s), dt)
    return f, F


def score_walk(rec, rest_s, dt, halves):
    """rec: a practice trial's recording with joint angles (rec["ang"][leg][dof]); rest_s: seconds before the
    command; halves: {leg: {dof: half the recorded walking range, rad}}. Returns (score, parts)."""
    t = np.asarray(rec["t"])
    rest = t < rest_s
    cmd = t >= rest_s + 0.2                      # skip the first 200 ms of the command (onset)
    if cmd.sum() < 16 or rest.sum() < 4:
        return -10.0, {}
    band_fr, amps, phases, spectra = [], [], {}, []
    quiet = []
    for leg in LEGS:
        for dof in DOFS:
            a = np.asarray(rec["ang"][leg][dof])
            h = max(halves[leg][dof], 0.05)
            f, F = _spectrum(a[cmd], dt)
            p = np.abs(F) ** 2
            band = (f >= 2.0) & (f <= 12.0)
            tot = p[1:].sum() + 1e-12
            band_fr.append(float(p[band].sum() / tot))
            amps.append(min(1.0, float(np.std(a[cmd])) / (0.35 * h)))
            quiet.append(min(1.0, float(np.std(a[rest])) / (0.15 * h)))
            if dof == "Coxa":
                spectra.append(p)
                phases[leg] = F
    rhythm = float(np.mean([b * m for b, m in zip(band_fr, amps)]))
    # coordination at the dominant stepping frequency (the swing joint)
    P = np.sum(spectra, axis=0)
    band = (f >= 2.0) & (f <= 12.0)
    if band.any() and P[band].max() > 0:
        k = np.where(band)[0][np.argmax(P[band])]
        freq = float(f[k])
        ph = {leg: np.angle(F[k]) for leg, F in phases.items()}
        cs = []
        for i, a in enumerate(LEGS):
            for b in LEGS[i + 1:]:
                target = 0.0 if TRIPOD[a] == TRIPOD[b] else math.pi
                cs.append((math.cos(ph[a] - ph[b] - target) + 1) / 2)
        tripod = float(np.mean(cs))
    else:
        freq, tripod = 0.0, 0.0
    x, y, h0 = np.asarray(rec["x"]), np.asarray(rec["y"]), rec["heading"][0]
    i0 = int(np.argmax(t >= rest_s))
    dx, dy = x[-1] - x[i0], y[-1] - y[i0]
    fwd = float(dx * math.cos(h0) + dy * math.sin(h0))
    val = {k: np.asarray(v) for k, v in rec["val"].items()}
    z = np.zeros(len(t))
    launch = max(0.0, float(np.maximum(val.get("LM.jump", z), val.get("RM.jump", z)).max()) - 0.1)
    wings = float(np.maximum(val.get("wingL.power", z), val.get("wingR.power", z)).mean())
    fell = 1.0 if (rec.get("righted", 0) > 0 or min(rec["upright"]) < 0.2) else 0.0
    rest_quiet = 1.0 - float(np.mean(quiet))
    score = (min(max(fwd, -3.0), 6.0) / 3.0 + rhythm + tripod * rhythm * 2.0 + 0.5 * rest_quiet
             - fell - 5.0 * launch - wings)
    parts = {"forward_mm": round(fwd, 3), "rhythm": round(rhythm, 3), "tripod": round(tripod, 3),
             "step_hz": round(freq, 2), "rest_quiet": round(rest_quiet, 3), "fell": fell,
             "launch": round(launch, 3), "wing_power": round(wings, 3)}
    return float(score), parts


class SepCMA:
    """Separable CMA-ES (diagonal covariance; Ros & Hansen 2008), maximising. For hundreds of parameters,
    where a full covariance learns too slowly."""

    def __init__(self, x0, sigma=0.5, popsize=None, seed=0):
        n = len(x0)
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.lam = popsize or 4 + int(3 * math.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1 / np.sum(self.w ** 2)
        self.cc = 4 / (n + 4)
        self.cs = (self.mueff + 2) / (n + self.mueff + 3)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff) * (n + 2) / 3
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff) * (n + 2) / 3)
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = np.asarray(x0, float).copy()
        self.sigma = float(sigma)
        self.D = np.ones(n)            # diagonal of C
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.gen = 0

    def ask(self):
        self._z = self.rng.standard_normal((self.lam, self.n))
        return self.m + self.sigma * self._z * np.sqrt(self.D)

    def tell(self, xs, fs):
        order = np.argsort(-np.asarray(fs))[: self.mu]
        y = (np.asarray(xs)[order] - self.m) / self.sigma
        z = self._z[order]
        yw = self.w @ y
        zw = self.w @ z
        self.m = self.m + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * zw
        self.gen += 1
        hsig = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chiN < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.D = ((1 - self.c1 - self.cmu) * self.D + self.c1 * (self.pc ** 2 + (1 - hsig) * self.cc * (2 - self.cc) * self.D)
                  + self.cmu * (self.w @ (y ** 2)))
        self.D = np.clip(self.D, 1e-8, 1e4)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(min(self.sigma, 3.0))

    def state(self):
        return {"m": self.m.tolist(), "sigma": self.sigma, "D": self.D.tolist(), "pc": self.pc.tolist(),
                "ps": self.ps.tolist(), "gen": self.gen, "lam": self.lam}

    @classmethod
    def restore(cls, st, seed=0):
        c = cls(np.asarray(st["m"]), st["sigma"], st["lam"], seed)
        c.D, c.pc, c.ps, c.gen = np.asarray(st["D"]), np.asarray(st["pc"]), np.asarray(st["ps"]), st["gen"]
        return c
