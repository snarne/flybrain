"""
Rhythm generators grown from the nerve cord's own neurons.

A walking fly's legs are stepped by rhythm-generating circuits in the ventral nerve cord (central
pattern generators). Which neurons form them in Drosophila isn't known yet. The classic design (Brown's
half-centre, found across animals) is two groups of premotor neurons, one driving swing (lifting the
leg and moving it forward), one driving stance (pressing it down and pushing back), that inhibit each
other and tire during a burst, so they take turns.

The connectome has the pieces. For each leg:

  swing group    excitatory nerve-cord interneurons whose synapses go mostly to that leg's swing motor
                 neurons (trochanter flexors, promotors), e.g. IN21A010, IN19B012, INXXX464
  stance group   the same for the stance motor neurons (trochanter extensors, remotors), e.g. IN21A012,
                 IN01A015, IN19A010, IN03A010
  inhibitory     interneurons driven by one group that inhibit the other (e.g. IN19A002, IN19A003),
  links          in both directions

These neurons, their synapses onto the motor neurons, and the rest of their wiring are the
connectome's. Three things a leaky integrate-and-fire network with synapse-count weights can't do are
added, for these neurons only, while the rhythm generators are on:

- Tiring: a second, faster adaptation current (sim.py: tau_burst, d_burst) ends each burst. (They are
  also exempt from the slow stabilising adaptation that "loop" neurons get.)
- Reciprocal inhibition: the inhibition the wiring shows between the groups runs through interneurons
  that the spiking model doesn't drive hard enough to carry it; it is modelled as direct inhibitory
  synapses between the groups (COUPLE_MV each), switched on with the generators.
- Descending drive: command neurons such as P9 reach these premotor neurons through several layers of
  interneurons, and the signal fades out on the way in the spiking model. Each descending neuron's
  summed influence on each leg's rhythm neurons through up to three synapses (signed, normalised by
  each target's input, as in skills.py) is computed from the wiring and delivered as synaptic input, in
  proportion to how fast that descending neuron is firing right now. Which legs a command drives, and
  how hard, is the wiring's.

Everything else, including how the six legs coordinate, is left to the connectome. Neurons that also
excite the jump muscle's motor neuron (directly or through one more neuron, e.g. IN20A.22A001) are
left out, so stepping can't set off the escape jump.
"""
from pathlib import Path

import numpy as np
from scipy import sparse

LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]
# each phase moves two joints: swing lifts the leg (trochanter flexors) and carries it forward (promotors);
# stance presses it down (trochanter extensors) and pulls it back (remotors). Each group gets its best
# neurons for both.
SWING = {"lift": ["trochanter_flexor", "accessory_trochanter_flexor"], "carry": ["tergopleural_promotor", "sternal_anterior_rotator"]}
STANCE = {"press": ["trochanter_extensor", "sternotrochanter_extensor"], "pull": ["pleural_remotor_and_abductor", "sternal_posterior_rotator"]}
GROUP = 6            # neurons per group
LINKS = 6            # inhibitory link interneurons per direction, per leg (identified, for the record)
TAU_BURST = 200.0    # ms: how long the tiring lasts
D_BURST = 0.5        # mV per spike: how fast a burst tires
COUPLE_MV = 8.0      # mV: each rhythm neuron's inhibition of each neuron of the other group in its leg
DRIVE_MAX = 3500.0   # events/s of descending drive at most, per rhythm neuron (1 mV each)
DRIVE_UNIT = 200000.0  # events/s per unit of (Hz x influence), before saturation
RATE_TAU = 0.05      # s, smoothing of the descending neurons' firing rates


class CPG:
    def __init__(self, world):
        self.w = world
        N = world.N
        sup = np.array(world.tables["super_class"])[world.sup]
        sign = np.load(Path(__file__).parent / "data" / "neurons.npz")["sign"]
        rows = np.repeat(np.arange(N, dtype=np.int32), np.diff(world.indptr))
        W = world.weights["stable"].astype(np.float32)
        A = sparse.csr_matrix((W, (rows, world.indices)), shape=(N, N))
        pools = {p["key"]: p["neurons"] for p in world.presets["pools"]}
        vnc = sup == "ventral_nerve_cord_intrinsic"
        premotor = vnc & (sign > 0)
        inhib = vnc & (sign < 0)
        # nothing that excites the jump motor neuron, directly or through one more neuron
        jump = sorted({n for k, v in pools.items() if k.endswith("jump_ttm") for n in v})
        if jump:
            v = np.zeros(N, np.float32)
            v[jump] = 1.0
            e1 = A @ v
            e2 = A @ np.clip(e1, 0, None)
            premotor &= (e1 <= 0) & (e2 <= 0)
        self.groups = {}
        taken = set()
        def into(leg, muscles):
            mn = sorted({n for m in muscles for n in pools.get(f"{leg}.{m}", [])})
            return np.asarray(A[:, mn].sum(1)).ravel() if mn else np.zeros(N)

        for leg in LEGS:
            parts = {k: into(leg, m) for k, m in {**SWING, **STANCE}.items()}
            for name, mine, other in (("swing", SWING, STANCE), ("stance", STANCE, SWING)):
                against = sum(np.clip(parts[k], 0, None) for k in other)
                pick = []
                for k in mine:                       # half the group for each joint of this phase
                    a = parts[k]
                    spec = np.where(premotor & (a > 10), a - against, -np.inf)
                    pick += [int(i) for i in np.argsort(-spec) if np.isfinite(spec[i]) and int(i) not in taken
                             and int(i) not in pick][:GROUP // 2]
                taken.update(pick)
                self.groups[(leg, name)] = pick
            # the connectome's inhibitory links: driven by one group, inhibiting the other more than the first
            for a, b in (("swing", "stance"), ("stance", "swing")):
                src, dst = self.groups[(leg, a)], self.groups[(leg, b)]
                fin = np.asarray(A[src].sum(0)).ravel()
                to_dst = -np.asarray(A[:, dst].sum(1)).ravel()
                to_src = -np.asarray(A[:, src].sum(1)).ravel()
                score = np.where(inhib & (fin > 0) & (to_dst > 0), fin * (to_dst - np.clip(to_src, 0, None)), 0.0)
                self.groups[(leg, f"{a}>{b}")] = [int(i) for i in np.argsort(-score)[:LINKS] if score[i] > 0]
        self.rhythm = sorted(taken)
        names = world.tables["type"]
        self.types = {k: sorted({names[world.type[n]] for n in v}) for k, v in self.groups.items()}
        # the model's reciprocal inhibition: every rhythm neuron onto every neuron of its leg's other group
        pre, post = [], []
        for leg in LEGS:
            a, b = self.groups[(leg, "swing")], self.groups[(leg, "stance")]
            for x in a:
                for y in b:
                    pre += [x, y]
                    post += [y, x]
        self.couple = (np.array(pre, np.int64), np.array(post, np.int64))
        # descending neurons' net influence on each leg's generator, through up to three synapses
        dn = np.where(sup == "descending")[0]
        inp = np.bincount(world.indices, weights=np.abs(W), minlength=N).astype(np.float32)
        An = sparse.csr_matrix((W / np.maximum(inp[world.indices], 1.0), (rows, world.indices)), shape=(N, N))
        M = np.zeros((N, len(LEGS)), np.float32)
        for c, leg in enumerate(LEGS):
            ns = self.groups[(leg, "swing")] + self.groups[(leg, "stance")]
            M[ns, c] = 1.0 / max(len(ns), 1)
        E1 = An @ M
        E2 = An @ E1
        E3 = An @ E2
        E = np.asarray(E1 + 0.7 * E2 + 0.5 * E3, np.float32)
        keep = np.abs(E[dn]).max(1) > 1e-5
        self.dn = dn[keep]
        self.E = E[self.dn]                       # (descending neurons) x (legs)
        self.dn_of = np.full(N, -1, np.int64)
        self.dn_of[self.dn] = np.arange(len(self.dn))
        self.dn_rate = np.zeros(len(self.dn))
        self.drive = np.zeros(len(LEGS))
        self.on = False
        self._ad = self._dep = None

    def enable(self, on):
        """Switch the rhythm generators on or off (off: the network is exactly the connectome model)."""
        self.on = bool(on)
        b = self.w.brain
        R = self.rhythm
        if self._ad is None:
            self._ad = b.ad_scale[R].copy()
            self._dep = b.dep_mask[R].copy()
        b.burst_scale[:] = 0.0
        b.p["tau_burst"], b.p["d_burst"] = TAU_BURST, D_BURST
        t = np.arange(len(b.Em)) * b.p["dt"]
        b.Eb = np.exp(-t / TAU_BURST)
        b.Kb = (TAU_BURST / (TAU_BURST - b.p["t_mbr"])) * (b.Eb - b.Em)
        if self.on:
            b.burst_scale[R] = 1.0
            b.ad_scale[R] = 0.0
            b.dep_mask[R] = False
            pre, post = self.couple
            b.set_extra_synapses(pre, post, np.full(len(pre), -COUPLE_MV, np.float32))
        else:
            b.ad_scale[R] = self._ad
            b.dep_mask[R] = self._dep
            b.set_extra_synapses([], [], [])
        self.dn_rate[:] = 0
        self.drive[:] = 0
        b.set_syn_drive({}, "cpg")

    def update(self, spikes, dt):
        """After each brain frame: the descending neurons' rates -> drive to each leg's generator."""
        if not self.on:
            return
        k = self.dn_of[spikes]
        k = k[k >= 0]
        inst = np.bincount(k, minlength=len(self.dn)) / max(dt, 1e-3)
        a = 1.0 - np.exp(-dt / RATE_TAU)
        self.dn_rate += a * (inst - self.dn_rate)
        self.drive = self.dn_rate @ self.E
        drive = {}
        for c, leg in enumerate(LEGS):
            ev = DRIVE_MAX * float(np.tanh(max(0.0, self.drive[c]) * DRIVE_UNIT / DRIVE_MAX))
            if ev > 20:
                for n in self.groups[(leg, "swing")] + self.groups[(leg, "stance")]:
                    drive[n] = (ev, 1.0)
        self.w.brain.set_syn_drive(drive, "cpg")

    def activity(self):
        """Mean firing rate of each leg's swing and stance group (Hz), for the viewer."""
        r = self.w.rate
        return {f"{leg}.{g}": round(float(r[self.groups[(leg, g)]].mean()), 1) for leg in LEGS for g in ("swing", "stance")
                if self.groups.get((leg, g))}

    def describe(self):
        return {f"{leg} {g}": self.types[(leg, g)] for (leg, g) in self.groups}
