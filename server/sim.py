"""
Whole-brain leaky integrate-and-fire model of the adult fruit fly.

Same equations and constants as Shiu et al. (Nature 2024), which is the model
Eon Systems and others build on (u = v - v_0):

    du/dt = (g - u - a) / t_mbr        (u, g frozen while refractory)
    dg/dt = -g / tau
    da/dt = -a / tau_adapt             (a stays 0 in "paper" mode)
    spike when u > v_th - v_0  ->  u = 0, g = 0, a += d_adapt, refractory t_rfc
    each presynaptic spike adds  w_syn * (signed synapse count)  to g after t_dly
    stimulated neurons get Poisson input that kicks u by w_syn * f_poi

"paper" mode is the published model. "stable" mode (see README) uses
FlyWire-annotated transmitter signs and gives spike-frequency adaptation to the
~6k neurons that otherwise lock into self-sustaining firing after odour input.

Speed trick (exact, not an approximation): between inputs the equations have a
closed-form solution, and u can never rise above max(u, g). So a neuron with both
below threshold cannot fire until another spike reaches it. Such neurons are left
untouched and caught up analytically when their next input arrives; only neurons
that could fire are stepped every 0.1 ms.
"""
import math

import numpy as np
from numba import njit

PARAMS = dict(
    v_0=-52.0,     # mV resting potential
    v_rst=-52.0,   # mV reset
    v_th=-45.0,    # mV threshold
    t_mbr=20.0,    # ms membrane time constant
    tau=5.0,       # ms synaptic time constant
    t_rfc=2.2,     # ms refractory period
    t_dly=1.8,     # ms synaptic delay
    w_syn=0.275,   # mV per synapse
    w_scale=1.5,   # BANC v2 detects about half as many synapses per neuron as FlyWire 783, which
                   # the constants above were fitted to (206 vs 393 input synapses per neuron); 1.5
                   # restores the sugar -> MN9 and looming -> giant fibre responses without making
                   # the whole network more excitable than the original
    f_poi=250.0,   # Poisson kick scale
    dt=0.1,        # ms integration step
    tau_adapt=500.0,  # ms adaptation decay            (stable mode)
    d_adapt=6.0,      # mV adaptation added per spike  (stable mode, loop neurons only)
    dep_u=0.7,        # short-term synaptic depression of loop neurons' outputs (stable mode):
    dep_rec=400.0,    #   fraction used per spike, recovery time (ms). Stops their burst at input onset
                      #   from igniting thousands of neurons before adaptation has built up
)
TABLE = 60000  # steps (6 s) of precomputed decay factors; older state has fully decayed


@njit(inline="always")
def _catch_up(j, step, u, g, ad, last, Em, Es, Ea, Kg, Ka):
    k = step - last[j]
    if k > 0:
        if k >= Em.shape[0]:
            u[j] = 0.0
            g[j] = 0.0
            ad[j] = 0.0
        else:
            u[j] = u[j] * Em[k] + g[j] * Kg[k] - ad[j] * Ka[k]
            g[j] = g[j] * Es[k]
            ad[j] = ad[j] * Ea[k]
        last[j] = step


@njit(nogil=True, cache=True, fastmath=True)
def _run(n_steps, u, g, ad, last, refr, refr_steps, indptr, indices, weights,
         stim_idx, stim_p, blocked, ring, ring_n, step0, D,
         out_idx, out_n, act, n_act_arr, in_act, ad_scale,
         Em, Es, Ea, Kg, Ka, theta, d_ad, kick, xdep, tdep, dep_mask, dep_u, dep_rec):
    n_out = out_n[0]
    n_act = n_act_arr[0]
    for s in range(n_steps):
        step = step0 + s
        slot = step % D
        # 1) deliver spikes emitted D steps ago
        m = ring_n[slot]
        for q in range(m):
            pre = ring[slot, q]
            scale = 1.0
            if dep_mask[pre]:
                # short-term depression: recover since the last spike, use the resource, deplete it
                xr = 1.0 - (1.0 - xdep[pre]) * math.exp(-(step - tdep[pre]) / dep_rec)
                scale = xr
                xdep[pre] = xr * (1.0 - dep_u)
                tdep[pre] = step
            for k in range(indptr[pre], indptr[pre + 1]):
                j = indices[k]
                _catch_up(j, step, u, g, ad, last, Em, Es, Ea, Kg, Ka)
                gj = g[j] + weights[k] * scale
                g[j] = gj
                if not in_act[j] and (gj >= theta or refr[j] > 0):
                    in_act[j] = True
                    act[n_act] = j
                    n_act += 1
        ring_n[slot] = 0
        # 2) Poisson drive on stimulated neurons
        for q in range(stim_idx.shape[0]):
            if np.random.random() < stim_p[q]:
                j = stim_idx[q]
                _catch_up(j, step, u, g, ad, last, Em, Es, Ea, Kg, Ka)
                u[j] += kick
                if not in_act[j]:
                    in_act[j] = True
                    act[n_act] = j
                    n_act += 1
        # 3) step neurons that could fire; park the rest
        c = 0
        keep = 0
        for q in range(n_act):
            i = act[q]
            if refr[i] > 0:
                refr[i] -= 1
                ad[i] = ad[i] * Ea[1]
            else:
                ui = u[i] * Em[1] + g[i] * Kg[1] - ad[i] * Ka[1]
                gi = g[i] * Es[1]
                ai = ad[i] * Ea[1]
                if ui > theta and not blocked[i]:
                    ui = 0.0
                    gi = 0.0
                    ai = ai + d_ad * ad_scale[i]
                    refr[i] = refr_steps[i]
                    ring[slot, c] = i
                    c += 1
                    if n_out < out_idx.shape[0]:
                        out_idx[n_out] = i
                        n_out += 1
                u[i] = ui
                g[i] = gi
                ad[i] = ai
            last[i] = step + 1
            if refr[i] == 0 and u[i] < theta and g[i] < theta:
                in_act[i] = False
            else:
                act[keep] = i
                keep += 1
        n_act = keep
        ring_n[slot] = c
    out_n[0] = n_out
    n_act_arr[0] = n_act


class Brain:
    """Holds network state; call step(ms) to advance."""

    def __init__(self, indptr, indices, syn_counts, params=None, mode="stable", adapt_mask=None):
        self.p = dict(PARAMS, **(params or {}))
        p = self.p
        self.N = len(indptr) - 1
        self.indptr = indptr.astype(np.int64)
        self.indices = indices.astype(np.int32)
        self.set_weights(syn_counts)
        self.D = int(round(p["t_dly"] / p["dt"]))
        self.theta = p["v_th"] - p["v_0"]
        self.kick = p["w_syn"] * p["f_poi"]
        self.rfc = int(round(p["t_rfc"] / p["dt"]))
        self.mode = mode
        # closed-form decay factors for k steps
        t = np.arange(TABLE) * p["dt"]
        tm, ts, ta = p["t_mbr"], p["tau"], p["tau_adapt"]
        self.Em, self.Es, self.Ea = np.exp(-t / tm), np.exp(-t / ts), np.exp(-t / ta)
        self.Kg = (ts / (ts - tm)) * (self.Es - self.Em)
        self.Ka = (ta / (ta - tm)) * (self.Ea - self.Em)
        self.ring = np.zeros((self.D, self.N), np.int32)
        self.ad_scale = np.ones(self.N, np.float32) if adapt_mask is None else adapt_mask.astype(np.float32)
        self.dep_mask = np.zeros(self.N, np.bool_) if adapt_mask is None else adapt_mask.astype(np.bool_)
        self.stim_rates = {}          # neuron index -> Hz
        self.blocked = np.zeros(self.N, np.bool_)
        self.out_idx = np.zeros(4_000_000, np.int32)
        self.out_n = np.zeros(1, np.int64)
        self.reset()

    def set_weights(self, syn_counts):
        self.weights = (syn_counts.astype(np.float32) * self.p["w_syn"] * self.p["w_scale"]).astype(np.float32)

    def reset(self):
        self.u = np.zeros(self.N, np.float64)
        self.g = np.zeros(self.N, np.float64)
        self.ad = np.zeros(self.N, np.float64)
        self.last = np.zeros(self.N, np.int64)
        self.refr = np.zeros(self.N, np.int32)
        self.ring_n = np.zeros(self.D, np.int64)
        self.act = np.zeros(self.N, np.int32)
        self.in_act = np.zeros(self.N, np.bool_)
        self.n_act = np.zeros(1, np.int64)
        self.step_i = 0
        self.xdep = np.ones(self.N, np.float64)
        self.tdep = np.zeros(self.N, np.int64)
        self._no_dep = np.zeros(self.N, np.bool_)
        self._rebuild_stim()

    # -------------------------------------------------------------- inputs
    def set_stim(self, rates):
        """rates: dict neuron_index -> Hz (0 removes)."""
        self.stim_rates = {int(k): float(r) for k, r in rates.items() if r > 0}
        self._rebuild_stim()

    def _rebuild_stim(self):
        idx = np.array(sorted(self.stim_rates), np.int64)
        self.stim_idx = idx
        self.stim_p = np.array([self.stim_rates[i] * self.p["dt"] / 1000.0 for i in idx], np.float64)
        # Poisson-driven neurons have no refractory period (as in the original model)
        self.refr_steps = np.full(self.N, self.rfc, np.int32)
        if len(idx):
            self.refr_steps[idx] = 0

    def set_blocked(self, idx):
        self.blocked[:] = False
        idx = list(idx)
        if idx:
            self.blocked[np.asarray(idx, np.int64)] = True

    # -------------------------------------------------------------- run
    def step(self, ms):
        """Advance by `ms` milliseconds. Returns neuron indices that spiked (with repeats)."""
        n = int(round(ms / self.p["dt"]))
        self.out_n[0] = 0
        d_ad = self.p["d_adapt"] if self.mode == "stable" else 0.0
        dep = self.dep_mask if self.mode == "stable" else self._no_dep
        _run(n, self.u, self.g, self.ad, self.last, self.refr, self.refr_steps, self.indptr, self.indices,
             self.weights, self.stim_idx, self.stim_p, self.blocked, self.ring, self.ring_n, self.step_i, self.D,
             self.out_idx, self.out_n, self.act, self.n_act, self.in_act, self.ad_scale,
             self.Em, self.Es, self.Ea, self.Kg, self.Ka, self.theta, d_ad, self.kick,
             self.xdep, self.tdep, dep, self.p["dep_u"], self.p["dep_rec"] / self.p["dt"])
        self.step_i += n
        return self.out_idx[: self.out_n[0]]

    @property
    def t_ms(self):
        return self.step_i * self.p["dt"]

    @property
    def active_count(self):
        """Neurons currently near enough to threshold to be stepped."""
        return int(self.n_act[0])
