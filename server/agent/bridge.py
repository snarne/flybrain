"""
The bridge between the agent and its fly body.

Agent -> brain. Each part of the agent's work drives a chosen set of real neurons; the network
does the rest. The choice of neurons is a design decision (shown as such in the UI); what the
connectome does with the input is the simulation.

  hearing your message      -> Johnston's organ auditory neurons (the fly's ear): the half that doesn't feed
                               the giant fibre (see hear_neurons)
  LLM deliberation          -> PFN neurons into the central complex (action selection), while thinking tokens stream
  memory recall / KV restore-> a sparse, memory-specific set of Kenyon cells (the mushroom body's code)
  tool success / failure    -> sugar (reward) / bitter (aversive) taste neurons

Brain -> agent. Each turn, a one-line "body state" (behaviour readouts, busiest regions) is
appended to your message, so the LLM knows what its body is doing.
"""
import hashlib
import itertools
import time

import numpy as np


class Bridge:
    def __init__(self, world, emit):
        self.world = world
        self.emit = emit              # callable(dict) -> schedules a broadcast
        self.ch = world.presets.get("channels", {})
        self.stim = world.stim
        self._ids = itertools.count()
        self.active = {}              # channel -> pulse id (so a stale 'off' doesn't cut a newer pulse)
        self.enabled = True
        self.loop = None
        self.timers = []              # (sim t_ms, channel, pulse id): switch-offs, timed in brain time

    def _log(self, what, why, neurons, seconds, channel):
        self.emit({"type": "bridge", "t": round(time.time(), 2), "what": what, "why": why,
                   "neurons": int(neurons), "seconds": round(seconds, 2), "channel": channel})

    def pulse(self, channel, neurons, rate, seconds, what, why):
        if not self.enabled or not len(neurons):
            return
        pid = next(self._ids)
        self.active[channel] = pid
        self.world.pending.append({"cmd": "channel", "name": channel, "neurons": [int(i) for i in neurons], "rate": rate})
        self._log(what, why, len(neurons), seconds, channel)
        self.timers.append((self.world.brain.t_ms + seconds * 1000.0, channel, pid))

    def tick(self, t_ms):
        """Called after every simulation frame: end pulses whose (simulated) time is up."""
        if not self.timers:
            return
        due = [x for x in self.timers if x[0] <= t_ms]
        self.timers = [x for x in self.timers if x[0] > t_ms]
        for _, channel, pid in due:
            self._off(channel, pid)

    def hold(self, channel, neurons, rate, on, what="", why=""):
        """Keep a channel on until switched off (used while thinking tokens stream)."""
        if not self.enabled:
            return
        if on and channel not in self.active:
            self.active[channel] = next(self._ids)
            self.world.pending.append({"cmd": "channel", "name": channel, "neurons": [int(i) for i in neurons], "rate": rate})
            self._log(what, why, len(neurons), 0, channel)
        elif not on and channel in self.active:
            self._off(channel, self.active[channel])

    def _off(self, channel, pid):
        if self.active.get(channel) == pid:
            del self.active[channel]
            self.world.pending.append({"cmd": "channel", "name": channel, "rate": 0})

    # ------------------------------------------------------------ named events
    def hear_neurons(self):
        """The auditory Johnston's organ neurons (JO-A/B) that hear a message: the half with the weakest pull
        on the giant fibres. About 90 of the 500 synapse straight onto the giant fibre (the escape neuron), as
        in real flies, where a loud sound can set off an escape; driving all of them made the fly jump every
        time you pressed Send. Hearing a message is meant to be a quiet sound."""
        if getattr(self, "_hear", None) is None:
            from scipy import sparse
            w = self.world
            jo = np.asarray(self.ch["hear"]["neurons"], np.int64)
            gf = w.neurons_of_type("DNp01")
            if not len(gf):
                self._hear = jo
                return jo
            ip, ix, wt = w.wiring
            N = w.N
            rows = np.repeat(np.arange(N, dtype=np.int32), np.diff(ip))
            inp = np.bincount(ix, weights=np.abs(wt), minlength=N)
            A = sparse.csr_matrix((wt / np.maximum(inp[ix], 1.0), (rows, ix)), shape=(N, N))
            t = np.zeros(N)
            t[gf] = 1.0
            e1 = A @ t
            e2 = A @ e1
            e3 = A @ e2
            pull = (e1 + 0.7 * e2 + 0.5 * e3)[jo]
            self._hear = np.sort(jo[np.argsort(pull)[: len(jo) // 2]])
        return self._hear

    def hear(self, text):
        c = self.ch["hear"]
        secs = min(3.0, 0.4 + len(text) / 250)
        self.pulse("hear", self.hear_neurons(), c["rate"], secs, "Hearing your message",
                   c["about"] + " (the half that doesn't drive the escape neuron: a quiet sound)")

    def thinking(self, on, tokens_per_s=None):
        c = self.ch["think"]
        rate = c["rate"] if tokens_per_s is None else min(60, 10 + tokens_per_s)
        self.hold("think", c["neurons"], rate, on, "Deliberating (the language model is thinking)", c["about"])

    def memory(self, key_text, what="Remembering"):
        c = self.ch["memory"]
        seed = int(hashlib.sha1(key_text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        kcs = np.asarray(c["neurons"])
        pick = rng.choice(kcs, max(1, int(len(kcs) * c.get("fraction", 0.05))), replace=False)
        self.pulse("memory", pick, c["rate"], 0.8, what, c["about"])

    def outcome(self, ok, what):
        s = self.stim["sugar" if ok else "bitter"]
        self.pulse("outcome", s["neurons"], s["rate"], 0.8, what,
                   "success tastes like sugar (reward)" if ok else "failure tastes bitter (aversive)")

    # ------------------------------------------------------------ brain -> agent
    def body_state(self):
        st = self.world.status()
        beh = [f"{b['label'].lower()} {b['hz']:.0f} Hz" for b in st["behaviours"] if b["level"] > 0.1]
        regions = sorted(zip(st["regions"], self.world.tables["regions"]), key=lambda x: -x[0]["hz_total"])
        busy = [r["label"].lower() for x, r in regions[:2] if x["hz_total"] > 50]
        return (f"[Body state from the fly brain: {st['firing']} neurons firing; "
                f"behaviour: {', '.join(beh) or 'resting'}; busiest regions: {', '.join(busy) or 'none'}]")
