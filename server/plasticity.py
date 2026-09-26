"""
Learning in the fly's own synapses: dopamine-gated plasticity in the mushroom body.

The mushroom body is where a fly learns what odours mean. An odour activates a sparse set of Kenyon
cells (KCs; 5-10% in real flies, 1-11% here, different ones for different odours: calibration.py).
The KCs synapse onto
mushroom body output neurons (MBONs), whose activity biases the fly towards or away from things.
Dopamine neurons (DANs) each innervate a compartment of the KC axons and the MBON dendrites there:
PAM neurons mostly signal reward (sugar), PPL1 neurons punishment (heat, shock, bitter). When a DAN
fires while KCs are active, the KC -> MBON synapses in its compartment weaken (Hige et al. 2015;
Aso & Rubin 2016), so the next time that odour comes, those MBONs respond less, and the fly's
behaviour towards the odour changes. KC activity without dopamine slowly restores them (extinction).

Here that is applied to the real BANC synapses:
  - the 15,534 KC -> MBON connections are the plastic ones (their weights change during the run);
  - which DANs gate which MBON is read from the wiring: each MBON's dopamine is the rates of the DANs
    that synapse onto it, weighted by their share of those synapses;
  - each KC keeps an eligibility trace of its recent spikes (ELIG_TAU); each frame a synapse changes by
        dw = -ETA * trace(KC) * dopamine(MBON) * w        (depression, down to FLOOR x its start)
             + RECOVER * trace(KC) * (w0 - w)              (when that MBON's compartment has no dopamine)
  - dopamine neurons' synapses onto the KC axons carry the teaching signal through slow (GPCR)
    receptors, not fast excitation; the model would otherwise count them as excitatory (dopamine isn't
    an inhibitory transmitter) and every reinforcement would make KCs fire for every odour. They are
    modelled as modulatory: no fast effect;
  - MBONs fire spontaneously in real flies; in the spiking model they are silent until KCs drive them.
    A background input (BASE_HZ events of 1 mV each) can make them fire on their own, but it's off:
    in this model even 5 Hz of MBON firing spreads through descending neurons to the legs, and the
    fly then fidgets and walks with nothing happening.

In this model the taste and temperature pathways don't reach the dopamine neurons strongly enough to
teach, so reinforcement is given the way many labs do it: by activating PAM (reward) or PPL1
(punishment) neurons directly while an odour is on. What was learned stays when the brain is reset;
"forget" restores every synapse.
"""
import numpy as np
from scipy import sparse

ELIG_TAU = 1.0       # s: how long a KC stays eligible after firing
DA_TAU = 0.2         # s: smoothing of the DANs' firing rates
ETA = 1.5e-3         # depression per (KC spike trace x DAN Hz) per second (KCs fire only a spike or two per odour)
RECOVER = 2.5e-3     # recovery per (KC spike trace) per second, without dopamine
DA_QUIET = 5.0       # Hz of dopamine below which a compartment counts as unrewarded
FLOOR = 0.1          # synapses can't go below this fraction of their connectome strength
BASE_HZ = 0.0        # background input to MBONs (events/s, 1 mV each; 1400 gives ~5 Hz); off, see above


class MushroomBody:
    def __init__(self, world):
        self.w = world
        N = world.N
        names = np.array(world.tables["type"])
        cls = np.array(world.tables["class"])[world.cls]
        typ = names[world.type].astype(str)
        self.kc = np.where(cls == "kenyon_cell")[0]
        self.mbon = np.where(np.char.startswith(typ, "MBON"))[0]
        self.pam = np.where(np.char.startswith(typ, "PAM"))[0]
        self.ppl1 = np.where(np.char.startswith(typ, "PPL1"))[0]
        self.dan = np.concatenate([self.pam, self.ppl1])
        self.mbon_type = typ[self.mbon]
        self.learning = True
        self.log = []                                    # what has been paired with what
        self.rewire()

    def rewire(self):
        """Read the plastic synapses and the dopamine compartments from the wiring the simulator runs
        (world.wiring: the connectome, or a control variant). Anything learned is forgotten."""
        world = self.w
        N = world.N
        indptr, indices, W = world.wiring
        rows = np.repeat(np.arange(N, dtype=np.int32), np.diff(indptr))
        W = W.astype(np.float32)
        A = sparse.csr_matrix((np.abs(W), (rows, indices)), shape=(N, N))
        # which DANs gate each MBON: their share of the DAN synapses onto it
        D = np.asarray(A[self.dan][:, self.mbon].todense(), np.float32)
        tot = D.sum(0)
        self.D = D / np.maximum(tot, 1.0)                  # (DANs) x (MBONs)
        self.gated = tot > 0
        # the plastic synapses, in the wiring's own order
        is_kc = np.full(N, -1, np.int64)
        is_kc[self.kc] = np.arange(len(self.kc))
        is_mb = np.full(N, -1, np.int64)
        is_mb[self.mbon] = np.arange(len(self.mbon))
        e = np.where((is_kc[rows] >= 0) & (is_mb[indices] >= 0))[0]
        self.pre = is_kc[rows[e]]
        self.post = is_mb[indices[e]]
        self.pre_n = rows[e].astype(np.int64)
        self.post_n = indices[e].astype(np.int64)
        # dopamine -> KC synapses: modulatory, no fast effect
        is_da = np.zeros(N, bool)
        is_da[self.dan] = True
        m = np.where(is_da[rows] & (is_kc[indices] >= 0))[0]
        self.mod_pre = rows[m].astype(np.int64)
        self.mod_post = indices[m].astype(np.int64)
        self.w0 = W[e] * world.brain.p["w_syn"] * world.brain.p["w_scale"]
        self.factor = np.ones(len(e), np.float32)
        self.kc_of = is_kc
        self.dan_of = np.full(N, -1, np.int64)
        self.dan_of[self.dan] = np.arange(len(self.dan))
        self.trace = np.zeros(len(self.kc))
        self.dan_rate = np.zeros(len(self.dan))
        self.version = None
        self.pos = None
        self.log = []
        self.smelling = set()                            # odours the fly smells right now (body.py)
        self.seen = {}                                   # odour -> its KC spikes so far (its code in the KCs)
        self.set_background(True)

    # ------------------------------------------------------------ plumbing
    def _locate(self):
        """Find the plastic synapses in the simulator's current arrays (they move when extra synapses
        are added) and apply what has been learned."""
        b = self.w.brain
        self.pos = self._find(b, self.pre_n, self.post_n)
        self.version = b.csr_version
        b.weights[self.pos] = self.w0 * self.factor
        b.weights[self._find(b, self.mod_pre, self.mod_post)] = 0.0

    @staticmethod
    def _find(b, pre, post):
        """Positions of the synapses pre[i] -> post[i] in the simulator's arrays."""
        ip, ix = b.indptr, b.indices
        rows = np.repeat(np.arange(len(ip) - 1, dtype=np.int64), np.diff(ip))
        key = rows * (len(ip) + 1) + ix
        want = pre * (len(ip) + 1) + post
        order = np.argsort(key, kind="stable")
        ks = key[order]
        at = np.minimum(np.searchsorted(ks, want), len(order) - 1)
        if not np.array_equal(ks[at], want):
            raise RuntimeError("plastic synapses not found in the simulator's wiring")
        return order[at]

    def set_background(self, on):
        self.w.brain.set_syn_drive({int(n): (BASE_HZ, 1.0) for n in self.mbon} if on else {}, "mb")

    # ------------------------------------------------------------ each frame
    def update(self, spikes, dt):
        b = self.w.brain
        if self.version != b.csr_version:
            self._locate()
        k = self.kc_of[spikes]
        k = k[k >= 0]
        if len(self.smelling) == 1 and len(k):
            od = next(iter(self.smelling))
            if od not in self.seen:
                self.seen[od] = np.zeros(len(self.kc), np.float32)
            np.add.at(self.seen[od], k, 1.0)
        self.trace *= np.exp(-dt / ELIG_TAU)
        if len(k):
            np.add.at(self.trace, k, 1.0)
        d = self.dan_of[spikes]
        d = d[d >= 0]
        inst = np.bincount(d, minlength=len(self.dan)) / max(dt, 1e-3)
        self.dan_rate += (1.0 - np.exp(-dt / DA_TAU)) * (inst - self.dan_rate)
        if not self.learning:
            return
        da = self.dan_rate @ self.D                        # dopamine at each MBON's compartment (Hz)
        tr = self.trace[self.pre]
        active = tr > 0.05
        if not active.any():
            return
        idx = np.where(active)[0]
        f = self.factor[idx]
        dam = da[self.post[idx]]
        df = -ETA * tr[idx] * dam * f * dt
        quiet = dam < DA_QUIET
        df[quiet] += RECOVER * tr[idx][quiet] * (1.0 - f[quiet]) * dt
        f = np.clip(f + df, FLOOR, 1.0)
        self.factor[idx] = f
        b.weights[self.pos[idx]] = self.w0[idx] * f

    # ------------------------------------------------------------ experimenter's controls
    def forget(self):
        self.factor[:] = 1.0
        self.trace[:] = 0.0
        self.log = []
        if self.pos is not None:
            self.w.brain.weights[self.pos] = self.w0

    def summary(self):
        """How much each MBON type's KC input has been weakened (0 = untouched, 0.9 = at the floor)."""
        out = {}
        for m in np.unique(self.post[self.factor < 0.999]):
            t = self.mbon_type[m]
            dep = float(1.0 - self.factor[self.post == m].mean())
            out[t] = max(out.get(t, 0.0), round(dep, 3))
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def gated_by(self, which):
        """MBONs whose compartments get most of their dopamine from reward (PAM) or punishment (PPL1)
        neurons."""
        n = len(self.pam)
        D = self.D[:n] if which == "reward" else self.D[n:]
        return self.gated & (D.sum(0) > 0.2)

    def drive(self, kc_spikes, learned=True):
        """Synaptic input (mV) that a pattern of KC spikes (count per KC) gives each MBON through the
        plastic synapses, as they are now (learned) or as the connectome has them."""
        x = kc_spikes[self.pre] * self.w0 * (self.factor if learned else 1.0)
        return np.bincount(self.post, weights=x, minlength=len(self.mbon))

    def memories(self):
        """For each odour the fly has smelled: how much weaker its Kenyon cells' input to the
        punishment-gated MBONs (an aversive memory) and the reward-gated ones (appetitive) has become."""
        out = {}
        for od, code in self.seen.items():
            if code.sum() < 5:
                continue
            now, then = self.drive(code), self.drive(code, learned=False)
            m = {}
            for which, label in (("punish", "aversive"), ("reward", "appetitive")):
                g = self.gated_by(which)
                m[label] = round(float(1 - now[g].sum() / max(then[g].sum(), 1e-9)), 3)
            out[od] = m
        return out

    def odour_weights(self, kcs):
        """Mean strength (fraction of original) of the KC -> MBON synapses from these KCs, per MBON type."""
        sel = np.isin(self.pre, self.kc_of[np.asarray(kcs, np.int64)])
        out = {}
        for m in np.unique(self.post[sel]):
            out.setdefault(self.mbon_type[m], []).append(float(self.factor[sel & (self.post == m)].mean()))
        return {t: round(float(np.mean(v)), 3) for t, v in out.items()}
