"""
The legs' premotor rhythm neurons, found in the nerve cord's wiring, and the descending drive to them.

Which nerve-cord neurons generate the stepping rhythm in Drosophila isn't known. For each leg this
picks, from the connectome, the excitatory interneurons whose synapses go mostly to that leg's swing
motor neurons (trochanter flexors, promotors; e.g. IN21A010, IN19B012, INXXX464) and to its stance
motor neurons (trochanter extensors, remotors; e.g. IN21A012, IN19A010, IN03A010), and the inhibitory
interneurons linking the two groups (e.g. IN19A002, IN19A003). Neurons that also excite the jump
muscle's motor neuron (directly or through one more neuron) are left out, so stepping can't set off
the escape jump.

An earlier version made these groups generate the rhythm themselves (mutual inhibition plus a
tiring current). They did alternate, but the six legs didn't coordinate and the steps were weak and
forwards only. Now the rhythm and coordination come from the walking oscillators (locomotion.py) and
these neurons follow them: while a leg is in swing its swing group gets synaptic drive, in stance its
stance group, so the leg's premotor and motor neurons fire in time with its steps (visible in the
brain and the activity view), through the connectome's own synapses.

Also computed here from the wiring: each descending neuron's summed influence on each leg's rhythm
neurons through up to three synapses (signed, normalised by each target's input). Weighted by how
fast the descending neurons fire, this is available as wiring_drive(), but it isn't used to command
walking: it counts feeding and grooming descending neurons too, which also reach the legs' networks.
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
DRIVE_MAX = 3500.0   # events/s: scale of the descending drive (wiring_drive saturates around here)
DRIVE_UNIT = 200000.0  # events/s per unit of (Hz x influence), before saturation
FOLLOW_HZ = 2500.0   # events/s (1 mV each) to a leg's swing or stance group while it is in that phase
RATE_TAU = 0.05      # s, smoothing of the descending neurons' firing rates


class CPG:
    def __init__(self, world):
        self.w = world
        N = world.N
        sup = np.array(world.tables["super_class"])[world.sup]
        sign = np.load(Path(__file__).parent / "data" / "neurons.npz")["sign"]
        # the wiring the simulator runs (the connectome, or a control variant: app.World.set_variant)
        indptr, indices, W = getattr(world, "wiring", (world.indptr, world.indices, world.weights["stable"]))
        rows = np.repeat(np.arange(N, dtype=np.int32), np.diff(indptr))
        W = W.astype(np.float32)
        A = sparse.csr_matrix((W, (rows, indices)), shape=(N, N))
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
        # descending neurons' net influence on each leg's generator, through up to three synapses
        dn = np.where(sup == "descending")[0]
        inp = np.bincount(indices, weights=np.abs(W), minlength=N).astype(np.float32)
        An = sparse.csr_matrix((W / np.maximum(inp[indices], 1.0), (rows, indices)), shape=(N, N))
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
        self._sent = None

    def enable(self, on):
        """On: the rhythm neurons follow the walking oscillators (exempt from the stabilising adaptation,
        which would silence them after a burst). Off: the network is exactly the connectome model."""
        self.on = bool(on)
        b = self.w.brain
        R = self.rhythm
        if self._ad is None:
            self._ad = b.ad_scale[R].copy()
            self._dep = b.dep_mask[R].copy()
        b.burst_scale[:] = 0.0
        if self.on:
            b.ad_scale[R] = 0.0
            b.dep_mask[R] = False
        else:
            b.ad_scale[R] = self._ad
            b.dep_mask[R] = self._dep
        if getattr(b, "_base", None) is not None:
            b.set_extra_synapses([], [], [])
        self.dn_rate[:] = 0
        self.drive[:] = 0
        self._sent = None
        b.set_syn_drive({}, "cpg")

    def update(self, spikes, dt):
        """After each brain frame: the descending neurons' rates -> their wiring influence on each leg."""
        if not self.on:
            return
        k = self.dn_of[spikes]
        k = k[k >= 0]
        inst = np.bincount(k, minlength=len(self.dn)) / max(dt, 1e-3)
        a = 1.0 - np.exp(-dt / RATE_TAU)
        self.dn_rate += a * (inst - self.dn_rate)
        self.drive = self.dn_rate @ self.E

    def wiring_drive(self):
        """0..1: how strongly the descending neurons now drive the legs' rhythm neurons, by the wiring."""
        if not self.on:
            return 0.0
        return float(np.mean(np.tanh(np.clip(self.drive, 0, None) * DRIVE_UNIT / DRIVE_MAX)))

    def follow(self, walker):
        """Drive each leg's swing or stance group, whichever phase its oscillator is in."""
        if not self.on:
            return
        drive = {}
        if walker.w > 0.05:
            ev = FOLLOW_HZ * walker.w
            for leg in LEGS:
                for n in self.groups[(leg, "swing" if walker.in_swing(leg) else "stance")]:
                    drive[n] = (ev, 1.0)
        key = tuple(sorted(drive))
        if key != self._sent or drive:
            self._sent = key
            self.w.brain.set_syn_drive(drive, "cpg")

    def activity(self):
        """Mean firing rate of each leg's swing and stance group (Hz), for the viewer."""
        r = self.w.rate
        return {f"{leg}.{g}": round(float(r[self.groups[(leg, g)]].mean()), 1) for leg in LEGS for g in ("swing", "stance")
                if self.groups.get((leg, g))}

    def describe(self):
        return {f"{leg} {g}": self.types[(leg, g)] for (leg, g) in self.groups}
