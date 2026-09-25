"""
Skills: new movements the fly learns by finding which neurons to drive.

A skill is a named movement, planned (usually by the LLM) as keyframes of motor "controls"
(see motor.py: per leg swing / lift / reach / grip / spread, wing power / extend / stroke, head,
proboscis). Learning it the first time:

  1. Plan -> target: each control's value over time becomes a target activation for the muscle
     pools that push it that way (a "channel", e.g. LF.lift+ = the front-left trochanter flexors).
  2. Search the wiring (the slow part, done once): for every neuron, its influence on every channel
     through up to three synapses (signed, normalised by each target's total input). A good driver
     for a channel pushes it strongly and its antagonist and everything else weakly. Candidates are
     tried from the top of the nervous system down: descending neurons (brain -> nerve cord) first,
     then premotor nerve-cord interneurons, and only as a last resort the motor neurons themselves.
  3. Practise, visibly: drive the chosen neurons on the live connectome, measure what the muscles
     actually did, and correct the drive over the whole movement from the error (iterative learning
     control: where a muscle lagged behind the plan, drive its neurons harder at that moment; where it
     overshot, ease off). If a driver can't move its muscle even at full drive, the engine goes one
     level down (descending -> premotor -> motor neuron); if a driver makes other body parts move, it
     is swapped for a more specific one. A few attempts; the best one is kept.
  4. Save it (data/skills/skills.json, this computer only): the drivers (neuron ids), their timing
     and rates, the plan and the score.

Doing a known skill is then just replaying the saved drive: no search, no practice.
"""
import asyncio
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "skills" / "skills.json"
BIN_S = 0.02            # time resolution of targets and drive schedules
BASE_HZ = 150.0         # drive at full activation, gain 1
MAX_HZ = 400.0
LEAD_S = 0.03           # start driving a little early (conduction + muscle delay)
LEVELS = ["descending", "premotor", "motor"]
LEVEL_TEXT = {"descending": "descending neuron (brain → nerve cord)", "premotor": "nerve-cord premotor interneuron",
              "motor": "motor neuron (direct)"}


def _now():
    return time.strftime("%Y-%m-%d %H:%M")


class SkillEngine:
    def __init__(self, world, motor, emit):
        self.w = world
        self.m = motor
        self.emit = emit
        self.lib = self._load()
        self.E = None                     # influence matrix (N x channels), built on first use
        self.state = None                 # the movement being practised or performed
        self.done = None                  # asyncio.Event for the current run
        self.busy = False
        self.log = []                     # recent events, for the UI
        # channels: every control in each direction
        self.channels = []
        for k, (pos, neg) in motor.controls.items():
            if pos:
                self.channels.append((k, +1))
            if neg:
                self.channels.append((k, -1))
        self.ch_index = {c: i for i, c in enumerate(self.channels)}
        sup_names = world.tables["super_class"]
        sup = np.array(sup_names)[world.sup]
        self.level_mask = {
            "descending": sup == "descending",
            "premotor": np.isin(sup, ["ventral_nerve_cord_intrinsic", "ascending"]),
            "motor": sup == "motor",
        }

    # ------------------------------------------------------------------ library
    def _load(self):
        try:
            return json.loads(STORE.read_text())
        except Exception:
            return {}

    def _save(self):
        STORE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STORE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.lib, indent=1))
        tmp.replace(STORE)

    def list(self):
        return [{"name": k, "description": v.get("description", ""), "score": v.get("score"),
                 "duration": v.get("duration"), "learned": v.get("learned"), "uses": v.get("uses", 0),
                 "levels": sorted({d["level"] for d in v.get("drivers", [])})} for k, v in sorted(self.lib.items())]

    def forget(self, name):
        if self.lib.pop(name, None) is not None:
            self._save()
            return True
        return False

    # ------------------------------------------------------------------ plan -> target
    def controls_help(self):
        return sorted(self.m.controls)

    def parse_plan(self, plan):
        """plan: {"duration": s, "keyframes": [{"t": s, "LF.lift": 0.8, ...} or {"t": s, "LF": {"lift": 0.8}}]}
        Returns (duration, {control: np.array over bins}) or raises ValueError."""
        kfs = plan.get("keyframes") or []
        if not kfs:
            raise ValueError("plan needs keyframes: [{\"t\": 0.0, \"LF.lift\": 0.8}, ...]")
        flat = []
        for kf in kfs:
            t = float(kf.get("t", 0))
            vals = {}
            for k, v in kf.items():
                if k == "t":
                    continue
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        vals[f"{k}.{kk}" if not k.endswith(".") else k + kk] = vv
                else:
                    vals[k] = v
            flat.append((t, vals))
        flat.sort(key=lambda x: x[0])
        unknown = sorted({k for _, vals in flat for k in vals if k not in self.m.controls})
        if unknown:
            raise ValueError(f"unknown controls {unknown}. Valid: {', '.join(self.controls_help())}")
        duration = float(plan.get("duration") or (flat[-1][0] + 0.2))
        duration = min(max(duration, 0.2), 6.0)
        n = int(round(duration / BIN_S)) + 1
        ts = np.arange(n) * BIN_S
        tracks = {}
        for k in sorted({k for _, vals in flat for k in vals}):
            pts = [(t, float(np.clip(vals[k], -1, 1))) for t, vals in flat if k in vals]
            if pts[0][0] > 0:
                pts.insert(0, (0.0, 0.0))
            tt, vv = zip(*pts)
            tracks[k] = np.interp(ts, tt, vv)
        return duration, tracks

    # ------------------------------------------------------------------ the wiring search
    def build_influence(self):
        from scipy import sparse
        w = self.w
        N = w.N
        rows = np.repeat(np.arange(N, dtype=np.int32), np.diff(w.indptr))
        wt = w.weights["stable"].astype(np.float32)
        inp = np.bincount(w.indices, weights=np.abs(wt), minlength=N).astype(np.float32)
        vals = wt / np.maximum(inp[w.indices], 1.0)
        A = sparse.csr_matrix((vals, (rows, w.indices)), shape=(N, N))
        C = len(self.channels)
        M = np.zeros((N, C), np.float32)
        for c, (k, d) in enumerate(self.channels):
            pos, neg = self.m.controls[k]
            pools = pos if d > 0 else neg
            mn = sorted({n for i in pools for n in self.m.pools[i]["neurons"]})
            if mn:
                M[mn, c] = 1.0 / len(mn)
        E1 = A @ M
        E2 = A @ E1
        E3 = A @ E2
        self.E1 = np.asarray(E1, np.float32)
        self.E = np.asarray(E1 + 0.7 * E2 + 0.5 * E3, np.float32)

    def candidates(self, ch, level, k=4, exclude=()):
        """Neurons at `level` that push channel ch hardest and most specifically."""
        E = self.E
        c = self.ch_index[ch]
        key, d = ch
        anti = self.ch_index.get((key, -d))
        tgt = E[:, c]
        mask = self.level_mask[level] & (tgt > 0)
        if level == "premotor":
            mask &= self.E1[:, c] > 0            # synapses directly onto the pool
        if exclude:
            mask[list(exclude)] = False
        idx = np.where(mask)[0]
        if not len(idx):
            return []
        sub = E[idx]
        t = sub[:, c]
        a = sub[:, anti] if anti is not None else 0.0
        others = np.clip(sub, 0, None).sum(1) - t - (np.clip(sub[:, anti], 0, None) if anti is not None else 0.0)
        spec = t - 1.0 * np.clip(a, 0, None) - 0.35 * others
        good = (spec > 0.25 * t) & (t > (0.002 if level == "descending" else 0.01))
        idx, spec, t = idx[good], spec[good], t[good]
        order = np.argsort(-spec)[:k]
        return [(int(idx[i]), float(spec[i]), float(t[i])) for i in order]

    def pick_drivers(self, tracks):
        """For each channel the plan uses: the best available drivers, highest level first."""
        drivers = []
        for key, tr in tracks.items():
            for d in (+1, -1):
                if (key, d) not in self.ch_index:
                    continue
                act = np.clip(d * tr, 0, None)
                if act.max() < 0.05:
                    continue
                ch = (key, d)
                for li, level in enumerate(LEVELS):
                    cands = self.candidates(ch, level, k=2 if level == "descending" else 3)
                    if level == "motor":
                        cands = [(n, 1.0, 1.0) for n in self.m.control_neurons(key)
                                 if n in set(sum((self.m.pools[i]["neurons"] for i in self.m.controls[key][0 if d > 0 else 1]), []))]
                    if cands:
                        drivers.append({"channel": f"{key}{'+' if d > 0 else '-'}", "control": key, "dir": d,
                                        "level": level, "level_i": li, "neurons": [c[0] for c in cands],
                                        "gain": 1.0, "tried": [c[0] for c in cands]})
                        break
        return drivers

    # ------------------------------------------------------------------ running a movement on the live brain
    def _initial_rates(self, dr, tracks):
        """First guess: drive in proportion to the planned activation, a little ahead of it."""
        lead = int(round(LEAD_S / BIN_S))
        tr = np.clip(dr["dir"] * tracks[dr["control"]], 0, None)
        tr = np.concatenate([tr[lead:], np.zeros(lead)]) if lead else tr
        return np.minimum(MAX_HZ, BASE_HZ * dr.get("gain", 1.0) * tr)

    def _schedule(self, drivers, tracks, duration):
        for dr in drivers:
            if dr.get("rates") is None:
                dr["rates"] = self._initial_rates(dr, tracks).round(0).tolist()

    def start(self, kind, name, drivers, tracks, duration, attempt=None):
        self._schedule(drivers, tracks, duration)
        self.state = {"kind": kind, "name": name, "t": 0.0, "duration": duration + 0.35, "drivers": drivers,
                      "tracks": tracks, "rec": {k: [] for k in self.m.controls}, "attempt": attempt, "last": None}
        self.done = asyncio.Event()
        neurons = sorted({n for d in drivers for n in d["neurons"]})
        self.emit({"type": "skill", "event": "run", "kind": kind, "name": name, "attempt": attempt,
                   "duration": duration, "neurons": neurons,
                   "drivers": [self._describe(d) for d in drivers]})

    def _describe(self, d):
        names = self.w.tables["type"]
        return {"channel": d["channel"], "level": d["level"], "level_text": LEVEL_TEXT[d["level"]],
                "types": sorted({names[self.w.type[n]] or "(unnamed)" for n in d["neurons"]}),
                "neurons": d["neurons"], "gain": round(float(np.max(d["rates"]) / BASE_HZ), 2) if d.get("rates") is not None else 1.0}

    def tick(self, dt):
        """Before each brain frame: set the drive for this moment."""
        s = self.state
        if not s:
            return
        i = int(s["t"] / BIN_S)
        rates = {}
        for d in s["drivers"]:
            r = d["rates"][i] if i < len(d["rates"]) else 0.0
            if r > 0:
                for n in d["neurons"]:
                    rates[n] = max(rates.get(n, 0.0), r)
        key = tuple(sorted(rates.items()))
        if key != s["last"]:
            s["last"] = key
            if rates:
                self.w.channel_stim["skill"] = rates
            else:
                self.w.channel_stim.pop("skill", None)
            self.w._push_stim()

    def observe(self, dt):
        """After each brain frame: record what the muscles did."""
        s = self.state
        if not s:
            return
        for k in s["rec"]:
            s["rec"][k].append((s["t"], self.m.value[k]))
        s["t"] += dt
        if s["t"] >= s["duration"]:
            self.w.channel_stim.pop("skill", None)
            self.w._push_stim()
            self.result = s
            self.state = None
            self.done.set()

    def stop(self):
        if self.state:
            self.state["t"] = self.state["duration"]

    # ------------------------------------------------------------------ scoring
    def evaluate(self, s, tracks, duration):
        n = len(next(iter(tracks.values())))
        ts = np.arange(n) * BIN_S
        ach = {}
        for k, rec in s["rec"].items():
            if rec:
                t, v = zip(*rec)
                ach[k] = np.interp(ts, t, v)
            else:
                ach[k] = np.zeros(n)
        per = {}
        for k, tgt in tracks.items():
            # allow up to 80 ms of lag (conduction and muscle delays)
            best = 1e9
            for lag in range(0, 5):
                a = np.concatenate([ach[k][lag:], np.full(lag, ach[k][-1])])
                best = min(best, np.abs(a - tgt).mean())
            per[k] = max(0.0, 1.0 - best / (np.abs(tgt).mean() + 0.1))
        planned = set(tracks)
        leg_like = [k for k in self.m.controls if k.split(".")[0] in ("LF", "LM", "LH", "RF", "RM", "RH") or k.startswith("wing")]
        cross = {k: float(np.abs(ach[k]).mean()) for k in leg_like if k not in planned}
        crosstalk = float(np.mean(sorted(cross.values())[-6:])) if cross else 0.0
        score = float(np.mean(list(per.values()))) * max(0.0, 1.0 - 1.5 * crosstalk)
        return score, per, cross, ach

    def adjust(self, drivers, tracks, ach, cross):
        changed = []
        lag = 2                       # bins (40 ms): the muscle answers the drive this much later
        for d in drivers:
            tgt = np.clip(d["dir"] * tracks[d["control"]], 0, None)
            got = np.clip(d["dir"] * ach[d["control"]], 0, None)
            got = np.concatenate([got[lag:], np.full(lag, got[-1])])
            r = np.asarray(d["rates"], float)
            on = tgt > 0.2
            if not on.any():
                continue
            ratio = got[on].mean() / max(tgt[on].mean(), 1e-6)
            saturated = r[on].mean() > 0.8 * MAX_HZ
            if ratio < 0.3 and (saturated or (d["level"] != "motor" and r[on].mean() > 0.5 * MAX_HZ)):
                # can't move the muscle from here: go one level down the nervous system
                li = d["level_i"] + 1
                while li < len(LEVELS):
                    lv = LEVELS[li]
                    if lv == "motor":
                        pools = self.m.controls[d["control"]][0 if d["dir"] > 0 else 1]
                        c = sorted({n for i in pools for n in self.m.pools[i]["neurons"]})
                    else:
                        c = [x[0] for x in self.candidates((d["control"], d["dir"]), lv, k=3)]
                    if c:
                        d.update(level=lv, level_i=li, neurons=c, gain=1.0, tried=list(c))
                        d["rates"] = self._initial_rates(d, tracks).round(0).tolist()
                        changed.append(f"{d['channel']}: now via {LEVEL_TEXT[lv]}")
                        break
                    li += 1
                continue
            # iterative learning control: correct the drive where the muscle lagged or overshot
            e = tgt - got
            r = np.clip(r + 0.7 * BASE_HZ * e, 0, MAX_HZ)
            # no drive where the plan wants this direction relaxed (lets the antagonist work)
            r[tgt < 0.02] = np.minimum(r[tgt < 0.02], 0.0)
            d["rates"] = r.round(0).tolist()
            if abs(ratio - 1) > 0.15:
                changed.append(f"{d['channel']}: {'stronger' if ratio < 1 else 'gentler'} drive where it {'lagged' if ratio < 1 else 'overshot'}")
        # crosstalk: the worst unplanned movement -> swap the driver that pushes it most
        if cross:
            k, v = max(cross.items(), key=lambda kv: kv[1])
            if v > 0.2:
                chans = [self.ch_index[c] for c in ((k, 1), (k, -1)) if c in self.ch_index]
                worst, wv = None, 0.0
                for d in drivers:
                    if d["level"] == "motor":
                        continue
                    infl = float(self.E[d["neurons"]][:, chans].clip(0).sum())
                    if infl > wv:
                        worst, wv = d, infl
                if worst is not None and len(worst["neurons"]) > 1:
                    bad = max(worst["neurons"], key=lambda n: float(self.E[n, chans].clip(0).sum()))
                    alt = [c for c in self.candidates((worst["control"], worst["dir"]), worst["level"], k=8)
                           if c[0] not in worst["tried"]]
                    worst["neurons"] = [n for n in worst["neurons"] if n != bad] + ([alt[0][0]] if alt else [])
                    if alt:
                        worst["tried"].append(alt[0][0])
                    changed.append(f"{worst['channel']}: {'swapped' if alt else 'dropped'} a driver that also moved {k}")
        return changed

    # ------------------------------------------------------------------ public API (async)
    async def learn(self, name, description, plan, attempts=8):
        if self.busy:
            return {"ok": False, "error": "busy with another movement"}
        self.busy = True
        try:
            duration, tracks = self.parse_plan(plan)
            t0 = time.time()
            self.emit({"type": "skill", "event": "search", "name": name})
            if self.E is None:
                await asyncio.get_running_loop().run_in_executor(None, self.build_influence)
            drivers = self.pick_drivers(tracks)
            if not drivers:
                return {"ok": False, "error": "the plan doesn't move anything"}
            best = None
            history = []
            for a in range(1, attempts + 1):
                # each try starts from a quiet nervous system (inputs you switched on stay on)
                self.w.pending.append({"cmd": "reset"})
                await asyncio.sleep(0.15)
                self._schedule(drivers, tracks, duration)
                self.start("learn", name, [dict(d, rates=list(d["rates"])) for d in drivers], tracks, duration, attempt=a)
                await self.done.wait()
                score, per, cross, ach = self.evaluate(self.result, tracks, duration)
                history.append(round(score, 3))
                if best is None or score > best[0]:
                    best = (score, json.loads(json.dumps(drivers, default=float)), per)
                self.emit({"type": "skill", "event": "attempt", "name": name, "attempt": a, "score": round(score, 3),
                           "per_control": {k: round(v, 2) for k, v in per.items()}})
                if score >= 0.8 or a == attempts:
                    break
                changes = self.adjust(drivers, tracks, ach, cross)
                self.emit({"type": "skill", "event": "adjust", "name": name, "changes": changes})
                if not changes:
                    break
                await asyncio.sleep(0.25)
            score, drv, per = best
            rid = self.w.root_ids
            skill = {
                "description": description, "plan": plan, "duration": duration, "score": round(score, 3),
                "attempts": history, "learned": _now(), "dataset": self.w.tables.get("dataset", ""), "uses": 0,
                "drivers": [{"channel": d["channel"], "control": d["control"], "dir": d["dir"], "level": d["level"],
                             "neurons": d["neurons"], "root_ids": [str(rid[n]) for n in d["neurons"]],
                             "rates": [int(x) for x in d["rates"]]} for d in drv],
            }
            self.lib[name] = skill
            self._save()
            self.emit({"type": "skill", "event": "learned", "name": name, "score": skill["score"],
                       "attempts": history, "seconds": round(time.time() - t0, 1)})
            return {"ok": True, "name": name, "score": skill["score"], "attempts": history,
                    "per_control": {k: round(v, 2) for k, v in per.items()},
                    "drivers": [self._describe(d) for d in drv],
                    "seconds": round(time.time() - t0, 1)}
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        finally:
            self.busy = False

    async def perform(self, names, repeat=1):
        if self.busy:
            return {"ok": False, "error": "busy with another movement"}
        missing = [n for n in names if n not in self.lib]
        if missing:
            return {"ok": False, "error": f"not learned yet: {missing}. Learn them first.", "known": sorted(self.lib)}
        self.busy = True
        try:
            done = []
            for _ in range(max(1, min(int(repeat or 1), 20))):
                for name in names:
                    sk = self.lib[name]
                    duration, tracks = self.parse_plan(sk["plan"])
                    drivers = [dict(d, neurons=list(d["neurons"]), rates=list(d["rates"]), level_i=LEVELS.index(d["level"]))
                               for d in sk["drivers"]]
                    self.start("play", name, drivers, tracks, duration)
                    await self.done.wait()
                    score, *_ = self.evaluate(self.result, tracks, duration)
                    done.append((name, round(score, 2)))
                    sk["uses"] = sk.get("uses", 0) + 1
            self._save()
            return {"ok": True, "performed": done}
        finally:
            self.busy = False
