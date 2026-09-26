"""
The legs' premotor rhythm neurons, found in the nerve cord's wiring.

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
FOLLOW_HZ = 2500.0   # events/s (1 mV each) to a leg's swing or stance group while it is in that phase


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
        if self.on:
            b.ad_scale[R] = 0.0
            b.dep_mask[R] = False
        else:
            b.ad_scale[R] = self._ad
            b.dep_mask[R] = self._dep
        self._sent = None
        b.set_syn_drive({}, "cpg")

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
