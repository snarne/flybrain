"""
Skills: new movements the fly learns by finding which neurons to drive, and practising.

A skill is a named movement, planned (usually by the LLM) as keyframes of motor "controls" (see
motor.py: per leg swing / lift / reach / grip / spread / twist, the middle legs' jump muscle, wing
power / extend / stroke, head, proboscis, antennae, halteres, abdomen), and optionally a goal: what
the movement should achieve (move forwards, turn, jump high, stay on its feet, keep the wings still;
practice.py). Learning it:

  1. Plan -> target: each control's value over time becomes a target activation for the muscle
     pools that push it that way (a "channel", e.g. LF.lift+ = the front-left trochanter flexors).
  2. Search the wiring (done once): for every neuron, its influence on every channel through up to
     three synapses (signed, normalised by each target's total input). A good driver for a channel
     pushes it strongly and its antagonist and everything else weakly. Candidates are tried from the
     top of the nervous system down: descending neurons (brain -> nerve cord) first, then premotor
     nerve-cord interneurons, and only as a last resort the motor neurons themselves.
  3. Shape: drive the chosen neurons, measure what the muscles did, and correct the drive from the
     error until the muscles follow the plan (iterative learning control; a driver that can't move its
     muscle is replaced one level down, one that moves other body parts is swapped).
  4. Practise for the goal: many tries in background copies of the fly (practice.py), each scored on
     what the movement achieved, with a numerical search (CMA-ES) tuning how hard and when each kind
     of movement's neurons are driven, and the pace. The best is checked again on fresh noise.
     Hand-built parts of the body (the walking model, the flight timer) are off in practice unless
     asked for, and a built-in skill can be run the same way as a baseline to compare with.
  5. Save it (data/skills/skills.json, this computer only): the drivers (neuron ids), their timing and
     rates, the plan, the goal, what it achieved, and the switches it was learned with; then show it
     once on the live fly.

Doing a known skill is then just replaying the saved drive: no search, no practice. If the practice
copies can't run (e.g. no MuJoCo), shaping happens on the live fly, one try at a time, as it used to.

Built-in skills (server/data/skills_builtin.json): walking forwards and backwards and turning (these
activate the command neurons; the walking model does the stepping) and antennal grooming (a recorded
movement played when the aDN grooming neurons fire). They're the baselines practice compares with.
"""
import asyncio
import json
import math
import time
from pathlib import Path

import numpy as np

import practice

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "skills" / "skills.json"
BUILTIN = ROOT / "server" / "data" / "skills_builtin.json"
BIN_S = 0.02            # time resolution of targets and drive schedules
BASE_HZ = 150.0         # drive at full activation, gain 1
MAX_HZ = 400.0
LEAD_S = 0.03           # start driving a little early (conduction + muscle delay)
LEVELS = ["descending", "premotor", "motor"]
SHAPE_TRIES = 5         # shaping tries (muscles following the plan) before practising for the goal
DEFAULT_TRIALS = 40     # practice tries for a goal
PRACTICE_ASSISTS = {"walking": False, "flight_timer": False}
# where in the nervous system a skill's drive goes in: the brain's own routes first (descending neurons, then
# nerve-cord premotor neurons, motor neurons last), only the nerve cord, or the motor neurons directly (the
# closest to working the body like a puppet: it bypasses the wiring above the motor neurons)
ASSIST_NAME = {"walking": "walking model", "flight_timer": "flight timer"}
DRIVE_AT = {"auto": None, "nerve_cord": ["premotor", "motor"], "motor": ["motor"]}
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
        try:
            self.builtin = json.loads(BUILTIN.read_text())
        except Exception:
            self.builtin = {}
        self.E = None                     # influence matrix (N x channels), built on first use
        self.state = None                 # the movement being practised or performed
        self.done = None                  # asyncio.Event for the current run
        self.busy = False
        self._halt = False
        self.log = []                     # recent events, for the UI
        self.pool = practice.Pool()       # background copies of the fly for practice
        self.allow_jump = False           # whether the plan being learned uses the jump muscle
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

    def get(self, name):
        return self.lib.get(name) or self.builtin.get(name)

    def list(self):
        out = []
        for builtin, lib in ((False, self.lib), (True, self.builtin)):
            out += [{"name": k, "description": v.get("description", ""), "score": v.get("score"),
                     "duration": v.get("duration"), "learned": v.get("learned"), "uses": v.get("uses", 0),
                     "builtin": builtin, "levels": sorted({d["level"] for d in v.get("drivers", [])}),
                     "command": bool(v.get("command")),
                     "goal": v.get("goal") if isinstance(v.get("goal"), str) else v.get("goal_text"), "tracking": v.get("tracking"),
                     "metrics": v.get("metrics"), "assists": v.get("assists")}
                    for k, v in sorted(lib.items()) if not (builtin and k in self.lib)]
        return out

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
        # the skeleton only bends so far: clip each control to what its joints can reach
        self.limit_notes = []
        for k, tr in tracks.items():
            up, down = self.m.reach.get((k, 1), 1.0), self.m.reach.get((k, -1), 1.0)
            clipped = np.clip(tr, -down, up)
            if np.abs(clipped - tr).max() > 0.05:
                self.limit_notes.append(f"{k} can only go to {-down:+.2f}..{up:+.2f} (joint limit); the plan asked for "
                                        f"{tr.min():+.2f}..{tr.max():+.2f}")
            tracks[k] = clipped
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
        # plus one column for what a skill must never set off: the jump muscle's motor neuron (TTMn), the
        # giant fibre's escape output. It isn't a control, so without this a driver that happens to
        # excite it would kick the fly's middle legs mid-movement.
        avoid = sorted({n for p in self.m.pools if p["muscle"].startswith("jump") for n in p["neurons"]})
        Mx = np.zeros((N, C + 1), np.float32)
        Mx[:, :C] = M
        if avoid:
            Mx[avoid, C] = 1.0 / len(avoid)
        E1 = A @ Mx
        E2 = A @ E1
        E3 = A @ E2
        E = np.asarray(E1 + 0.7 * E2 + 0.5 * E3, np.float32)
        self.E1 = np.asarray(E1, np.float32)[:, :C]
        self.E = E[:, :C]
        self.E_avoid = E[:, C]

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
        if not self.allow_jump:
            good &= self.E_avoid[idx] <= 0.05 * t      # nothing that also drives the jump muscle
        idx, spec, t = idx[good], spec[good], t[good]
        order = np.argsort(-spec)[:k]
        return [(int(idx[i]), float(spec[i]), float(t[i])) for i in order]

    def pick_drivers(self, tracks, levels=None):
        """For each channel the plan uses: the best available drivers, highest level first (only from
        `levels` if given, e.g. the nerve cord's own premotor and motor neurons for a rhythm)."""
        drivers = []
        self.allow_jump = any(k.endswith(".jump") and np.abs(tr).max() > 0.05 for k, tr in tracks.items())
        for key, tr in tracks.items():
            for d in (+1, -1):
                if (key, d) not in self.ch_index:
                    continue
                act = np.clip(d * tr, 0, None)
                if act.max() < 0.05:
                    continue
                ch = (key, d)
                for li, level in enumerate(LEVELS):
                    if levels and level not in levels:
                        continue
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

    def start(self, kind, name, drivers, tracks, duration, attempt=None, command=False):
        self._schedule(drivers, tracks, duration)
        self.state = {"kind": kind, "name": name, "t": 0.0, "duration": duration + 0.35, "drivers": drivers, "command": command,
                      "tracks": tracks, "rec": {k: [] for k in self.m.controls}, "attempt": attempt, "last": None, "ball": []}
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
        body = getattr(self.w, "body", None)
        if body is not None:
            s.setdefault("ball", []).append(tuple(body.ball))
        s["t"] += dt
        if s["t"] >= s["duration"]:
            self.w.channel_stim.pop("skill", None)
            self.w._push_stim()
            self.result = s
            self.state = None
            self.done.set()

    def reflex(self, name):
        """Play a skill as the body's own rhythm (e.g. walking when P9 fires). Non-blocking; skipped if a
        movement is already running."""
        if self.busy or self.state:
            return False
        sk = self.get(name)
        if not sk:
            return False
        duration, tracks = self.parse_plan(sk["plan"])
        drivers = [dict(d, neurons=list(d["neurons"]), rates=list(d["rates"]), level_i=LEVELS.index(d["level"]))
                   for d in sk["drivers"]]
        self.start("reflex", name, drivers, tracks, duration, command=bool(sk.get("command")))
        self.state["duration"] = duration          # no settling pause: the next cycle follows straight on
        return True

    def ball_summary(self, s):
        """How the movement moved the fly (the physical result, as a tracker would measure it)."""
        b = (s or {}).get("ball") or []
        if not b:
            return None
        f = np.array([x[0] for x in b])
        t = -np.array([x[1] for x in b])          # + = turning left
        return {"forward_mm_s": round(float(f.mean()), 2), "turn_left_rad_s": round(float(t.mean()), 2),
                "peak_forward_mm_s": round(float(f[np.argmax(np.abs(f))]), 2)}

    def stop(self):
        self._halt = True                 # practice stops after the current round
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

    def adjust(self, drivers, tracks, ach, cross, per=None, attempt=0):
        changed = []
        # fine control: a movement that upstream drivers only get roughly right after a few tries gets
        # its own motor neurons added as a trim, driven by the remaining error (like the fine
        # corrections premotor circuits make); the upstream drivers keep doing the bulk of it
        if per and attempt >= 3:
            for k, sc in per.items():
                if sc >= 0.7:
                    continue
                for d in (+1, -1):
                    if (k, d) not in self.ch_index or any(x["control"] == k and x["dir"] == d and x["level"] == "motor" for x in drivers):
                        continue
                    tgt = np.clip(d * tracks[k], 0, None)
                    if tgt.max() < 0.1:
                        continue
                    got = np.clip(d * ach[k], 0, None)
                    got = np.concatenate([got[2:], np.full(2, got[-1])])
                    pools = self.m.controls[k][0 if d > 0 else 1]
                    mns = sorted({n for i in pools for n in self.m.pools[i]["neurons"]})
                    rates = np.clip(BASE_HZ * np.clip(tgt - got, 0, None), 0, MAX_HZ)
                    drivers.append({"channel": f"{k}{'+' if d > 0 else '-'}", "control": k, "dir": d, "level": "motor",
                                    "level_i": LEVELS.index("motor"), "neurons": mns, "gain": 1.0, "tried": list(mns),
                                    "rates": rates.round(0).tolist(), "trim": True})
                    changed.append(f"{k}{'+' if d > 0 else '-'}: fine trim at its motor neurons")
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
    async def _wait_reflex(self):
        while self.state and self.state["kind"] == "reflex":
            await asyncio.sleep(0.05)

    CONTROL_TEXT = {
        "swing": "+ leg forward (promotion) / - back (remotion); the coxa's roll follows it",
        "lift": "+ femur raised (trochanter flexors) / - pressed down (extensors)",
        "reach": "+ tibia extended / - folded against the femur",
        "grip": "+ tarsus pressed down, claws and pads engaged (depressor, long tendon) / - raised (levator)",
        "spread": "- leg pulled in towards the body (adductor); 0 relaxed",
        "twist": "+ femur rotated (reductor)",
        "jump": "0..1 the jump muscle (tergotrochanter, TTMn), middle legs only (LM, RM): kicks the leg down and "
                "launches the fly while the middle feet are on the ground",
    }

    def reference(self):
        """A compact, factual guide for planning movements (for the motor_reference tool)."""
        m = self.m
        lines = ["MOTOR CONTROLS (plan values -1..1; everyday activation up to 0.8 stays within the range a fly "
                 "uses when walking, 0.8-1.0 reaches further, up to the joint limit):",
                 "Legs: LF LM LH RF RM RH (Left/Right, Front/Middle/Hind). Per leg:"]
        lines += [f"  <leg>.{c}: {txt}" for c, txt in self.CONTROL_TEXT.items()]
        lines += ["Wings (each side, wingL/wingR): .power 0..1 flight muscles (the wings only beat once no foot touches "
                  "the ground, e.g. after a jump, and the lift they give is proportional to power: about 0.8 hovers); "
                  ".extend 0..1 hold the wing out; .stroke + bigger / - smaller wingbeat (left vs right steers).",
                  "Head: head.yaw (+ left), head.pitch (+ up), head.roll. Proboscis: proboscis.rostrum, .haustellum, "
                  ".labellum (0..1 extend), .pump. Antennae: antennaL, antennaR. Halteres: haltereL, haltereR. "
                  "Abdomen: abdomen.curl (0..1), abdomen.bend (+ left / - right).",
                  f"All {len(m.controls)} controls: {', '.join(sorted(m.controls))}",
                  "",
                  "JOINT RANGES (degrees; resting angle, walking range, reach at full drive, hard limit), per leg pair:"]
        names = {"Coxa": "swing (thorax-coxa)", "Femur": "lift (coxa-trochanter)", "Tibia": "reach (femur-tibia)",
                 "Tarsus1": "grip (tibia-tarsus)", "Coxa_yaw": "spread", "Femur_roll": "twist"}
        for leg, label in (("LF", "front"), ("LM", "middle"), ("LH", "hind")):
            rng = m.ranges.get(leg, {})
            parts = []
            for dof, what in names.items():
                r = rng.get(dof)
                if r:
                    parts.append(f"{what}: rest {math.degrees(r['centre']):.0f}, walking ±{math.degrees(r['half']):.0f}, "
                                 f"full ±{math.degrees(r['span']):.0f}, limit ±{math.degrees(r['limit']):.0f}")
            lines.append(f"  {label} legs: " + "; ".join(parts))
        lines += ["",
                  "TIMING (real flies, and this body):",
                  "  Walking: tripod gait, LF+RM+LH alternate with RF+LM+RH. 5-15 steps per second; each leg swings "
                  "(lifted, moving forward) for ~35-40% of a step and pushes back on the ground for the rest.",
                  "  Grooming sweeps: ~5-8 per second. Proboscis extension: ~0.1-0.3 s. Escape jump: the jump muscle "
                  "for a few tens of ms.",
                  "  Nerve + muscle delay ~40 ms; muscles smooth anything faster than ~30 ms. Plan keyframes >= 0.05 s "
                  "apart; the engine samples every 20 ms.",
                  "  Plans are linear between keyframes; a control not mentioned stays 0 (relaxed); a control keeps its "
                  "last value after its last keyframe, so return it to 0 explicitly.",
                  "",
                  "PHYSICS (the body stands and walks on the ground under gravity, with friction and sticky foot pads):",
                  "  A foot pushes the body only while it is down and pressing (lift <= 0); lift it (lift > 0.3) to swing "
                  "it forward without dragging. Claws (grip +) engage the sticky pads: needed to pull, e.g. for walking "
                  "backwards. Hind legs do most of the pushing (reach + during stance drives the fly forward); middle "
                  "legs mostly by twist and swing; front legs little.",
                  "  Agonist + antagonist together stiffen a joint; a relaxed joint is floppy and gives under the body's weight.",
                  "  Body-model simplifications (not switchable): the jump muscle's launch is applied as a push on the body "
                  "(the model's legs can't launch it by themselves); wing lift and thrust are proportional to wing power; "
                  "the halteres keep the body level in the air; a fly on its back for a second is set back on its feet.",
                  "",
                  "HAND-BUILT PARTS (switches; practice runs with them OFF unless you ask):"]
        from body import ASSISTS
        lines += [f"  {k}: {v}" for k, v in ASSISTS.items()]
        lines += ["",
                  "PRACTICE AND GOALS (learn_skill with a goal): after the muscles follow your plan, the engine practises "
                  "in background copies of the fly, trying many variations of how hard and when each kind of movement is "
                  "driven (and the pace), scoring each try on the goal. Goal = a preset or {metric: weight} (+ = more is "
                  "better, - = less) or {metric: {\"target\": value}}. Presets: "
                  + "; ".join(f"{k} = {v}" for k, v in practice.GOAL_PRESETS.items()) + ".",
                  "  Metrics: " + "; ".join(f"{k}: {v[1]} (scored per {v[0]:g})" for k, v in practice.METRICS.items()) + ".",
                  f"  Always added unless your goal sets them: {practice.DEFAULT_PENALTY}; launch x{practice.JUMP_PENALTY:g} "
                  f"if the plan doesn't use the jump muscle; wing_power x{practice.WING_PENALTY:g} if it doesn't use the wings.",
                  "  Baselines: pass baseline=walk_forward (or any skill) to measure it the same way, with the hand-built "
                  "parts it relies on. test_skill measures any skill in the copies without touching the live fly.",
                  "",
                  "BUILT-IN SKILLS: walk_forward, walk_backward, turn_left, turn_right activate the walking command neurons "
                  "(the walking model steps the legs: a hand-built part). groom_antennae is a recorded grooming movement:"]
        for name in ("groom_antennae",):
            sk = self.builtin.get(name)
            if sk:
                lines.append(f"  {name} ({sk['plan'].get('duration')} s): " + json.dumps(sk["plan"]["keyframes"]))
        lines += ["", "WALKING PLANS: learn_skill(gait={\"steps_per_s\": 5, \"duration\": 2}) makes the plan from a real fly's "
                  "recorded steps (every joint of all six legs, tripod gait; turn -1..1, backwards). drive_at=motor follows "
                  "it closely but bypasses the wiring above the motor neurons; auto (descending first) is the brain's own "
                  "route but its drivers also move other muscles, so practice matters more there.",
                  "", "Tips: one skill per reusable unit (one tap, one stride), 0.3-2 s long for movements, up to 6 s for a "
                  "few strides; give a goal whenever the movement has a physical purpose (move, turn, jump, stay put) and "
                  "read which metrics moved; compare with a baseline; refine_skill with a new goal or plan."]
        return "\n".join(lines)

    def gait_plan(self, steps_per_s=5.0, duration=1.0, backwards=False, turn=0.0, per_step=8):
        """A walking plan in motor controls, made from a real fly's recorded steps (the same recording the
        walking model plays): each leg's joint angles over a step, converted to the control values that put
        its muscles there, in a tripod gait. A starting point for learning to walk through the muscles.
        turn: -1..1, + shortens the left legs' strides (turns left)."""
        from locomotion import TRIPOD, load_steps
        ang, _ = load_steps()
        n = len(next(iter(next(iter(ang.values())).values())))
        rng = self.m.ranges
        how = {"Coxa": ("swing", -1), "Femur": ("lift", -1), "Tibia": ("reach", -1), "Tarsus1": ("grip", -1),
               "Coxa_yaw": ("spread", 1), "Femur_roll": ("twist", 1)}
        f = float(max(1.0, min(steps_per_s, 15.0)))
        duration = float(max(0.2, min(duration, 6.0)))
        dt = 1.0 / (f * per_step)
        kfs = []
        for k in range(int(duration / dt) + 1):
            t = k * dt
            kf = {"t": round(t, 4)}
            for leg in ("LF", "LM", "LH", "RF", "RM", "RH"):
                ph = ((-1 if backwards else 1) * t * f + TRIPOD[leg]) % 1.0
                x = ph * n
                i0, fr = int(x) % n, x - int(x)
                amp = 1.0 - 0.9 * abs(turn) if (turn > 0 and leg[0] == "L") or (turn < 0 and leg[0] == "R") else 1.0
                for dof, (ctl, sign) in how.items():
                    key = f"{leg}.{ctl}"
                    if key not in self.m.controls or dof not in ang[leg]:
                        continue
                    a = ang[leg][dof]
                    r = rng[leg][dof]
                    off = amp * ((a[i0] * (1 - fr) + a[(i0 + 1) % n] * fr) - r["centre"])
                    if dof in ("Coxa_yaw", "Femur_roll") and leg[0] == "R":
                        off = -off
                    v = sign * off / max(r["half"], 1e-6) * 0.8
                    pos, neg = self.m.controls[key]
                    v = max(-1.0 if neg else 0.0, min(1.0 if pos else 0.0, v))
                    kf[key] = round(v, 2)
            kfs.append(kf)
        last = {k: 0 for k in kfs[-1] if k != "t"}
        kfs.append(dict(last, t=round(duration + 0.1, 4)))
        return {"duration": round(duration + 0.1, 3), "keyframes": kfs}

    def timeline(self, tracks, ach, plan, n=8):
        """Planned vs achieved for each control at the plan's keyframe times (at most n), for the LLM to
        compare and refine its plan."""
        ts = sorted({float(k.get("t", 0)) for k in plan.get("keyframes", [])})
        if len(ts) > n:
            ts = [ts[int(i * (len(ts) - 1) / (n - 1))] for i in range(n)]
        out = {}
        for k, tr in tracks.items():
            grid = np.arange(len(tr)) * BIN_S
            a = ach.get(k, np.zeros(len(tr)))
            out[k] = [(round(t, 2), round(float(np.interp(t, grid, tr)), 2), round(float(np.interp(t + 0.04, grid, a)), 2))
                      for t in ts]
        return out

    # ------------------------------------------------------------------ learning
    def _warm_drivers(self, drivers, warm, keep_rates):
        """Refining a known skill: keep the neurons already found for each movement (the expensive
        part), and with keep_rates its drive too."""
        old = {d["channel"]: d for d in warm.get("drivers", []) if not d.get("trim") or keep_rates}
        if keep_rates:
            return [dict(d, neurons=list(d["neurons"]), rates=list(d["rates"]), level_i=LEVELS.index(d["level"]),
                         gain=1.0, tried=list(d["neurons"])) for d in warm["drivers"]]
        for d in drivers:
            o = old.get(d["channel"])
            if o:
                d.update(level=o["level"], level_i=LEVELS.index(o["level"]), neurons=list(o["neurons"]),
                         tried=list(o["neurons"]))
        return drivers

    def _world_cfg(self):
        w = self.w
        return {"variant": w.variant, "variant_seed": int(getattr(w, "variant_seed", 0)), "ablated": sorted(w.ablated)}

    def _trial(self, drivers, duration, assists, seed, command=False, tail=0.4):
        return {"op": "trial", "world": self._world_cfg(), "seed": int(seed), "bin_s": BIN_S, "duration": float(duration),
                "tail": tail, "assists": assists, "command": bool(command),
                "drivers": [{"neurons": [int(n) for n in d["neurons"]], "rates": [float(x) for x in d["rates"]]}
                            for d in drivers]}

    @staticmethod
    def _as_state(rec):
        return {"rec": {k: list(zip(rec["t"], v)) for k, v in rec["val"].items()}}

    def _measure(self, res, tracks, duration):
        m = practice.measure(res["rec"], set(tracks), duration, MAX_HZ)
        if tracks:
            m["tracking"] = round(self.evaluate(self._as_state(res["rec"]), tracks, duration)[0], 3)
        return m

    async def _run(self, reqs, on_result=None):
        res = await self.pool.run(reqs, on_result)
        bad = [r for r in res if not r.get("ok")]
        if len(bad) == len(res):
            raise RuntimeError(f"practice failed: {bad[0].get('error')}")
        return res

    async def _shape(self, name, drivers, tracks, duration, assists, tries):
        """Correct the drive until the muscles follow the plan (in a practice copy)."""
        best, history = None, []
        for a in range(1, tries + 1):
            self._schedule(drivers, tracks, duration)
            res = (await self._run([self._trial(drivers, duration, assists, seed=10 + a)]))[0]
            sc, per, cross, ach = self.evaluate(self._as_state(res["rec"]), tracks, duration)
            history.append(round(sc, 3))
            if best is None or sc > best["score"]:
                best = {"score": sc, "drivers": json.loads(json.dumps(drivers, default=float)), "per": per, "ach": ach,
                        "res": res}
            self.emit({"type": "skill", "event": "practice", "stage": "shape", "name": name, "n": a, "of": tries,
                       "score": round(sc, 3), "best": round(best["score"], 3)})
            if sc >= 0.85 or a == tries or self._halt:
                break
            changes = self.adjust(drivers, tracks, ach, cross, per, a)
            self.emit({"type": "skill", "event": "adjust", "name": name, "changes": changes})
            if not changes:
                break
        return best, history

    async def _search(self, name, drivers, duration, tracks, goal, assists, trials, timing, per):
        """Practise for the goal: CMA-ES over how hard and when each kind of movement is driven."""
        space = practice.Space(drivers, timing=timing, per=per)
        nw = max(1, len(self.pool.workers))
        lam = max(6, 4 + int(3 * math.log(max(space.n, 2))))
        lam = int(math.ceil(lam / nw) * nw)
        cma = practice.CMA(np.zeros(space.n), 1.0, popsize=lam, seed=int(time.time()) % 10000)
        evals, best, done, gen = [], None, 0, 0
        by_gen = []
        while (done + lam <= trials or gen == 0) and not self._halt:
            xs = cma.ask()
            if gen == 0:
                xs[0] = 0.0                    # the shaped movement as it is, for reference
            cands = [space.apply(drivers, x, duration, BIN_S, MAX_HZ) for x in xs]
            seed = 100 + gen                   # every try of a round sees the same noise: a fair comparison
            fs = [None] * len(xs)

            def on_result(i, r, gen=gen, xs=xs, cands=cands, fs=fs, done=done):
                nonlocal best
                if r.get("ok"):
                    m = self._measure(r, tracks, cands[i][1])
                    f = practice.score(m, goal)[0]
                else:
                    m, f = None, -1e3
                fs[i] = f
                e = {"score": f, "x": np.asarray(xs[i]), "metrics": m, "gen": gen, "start": gen == 0 and i == 0}
                evals.append(e)
                if m is not None and (best is None or f > best["score"]):
                    best = e
                self.emit({"type": "skill", "event": "practice", "stage": "search", "name": name,
                           "n": done + sum(x is not None for x in fs), "of": max(trials, lam),
                           "score": round(f, 3), "best": round(best["score"], 3) if best else None})
            await self._run([self._trial(dr, dur, assists, seed) for dr, dur in cands], on_result)
            cma.tell(xs, [f if f is not None else -1e3 for f in fs])
            by_gen.append(round(max(f for f in fs if f is not None), 3))
            done += lam
            gen += 1
        return space, evals, by_gen, done

    async def _check(self, name, contenders, tracks, goal, assists, seeds=(9001, 9002)):
        """Try the best few again on fresh noise; the best on average wins (a lucky try doesn't)."""
        reqs, owner = [], []
        for k, c in enumerate(contenders):
            for sd in seeds:
                reqs.append(self._trial(c["drivers"], c["duration"], assists, sd))
                owner.append(k)
        self.emit({"type": "skill", "event": "practice", "stage": "check", "name": name, "n": 0, "of": len(reqs)})
        res = await self._run(reqs)
        for c in contenders:
            c["runs"] = [c["metrics"]] if c.get("metrics") else []
        for k, r in zip(owner, res):
            if r.get("ok"):
                contenders[k]["runs"].append(self._measure(r, tracks, contenders[k]["duration"]))
        for c in contenders:
            c["mean"] = {k: round(float(np.mean([m[k] for m in c["runs"]])), 3) for k in c["runs"][0]} if c["runs"] else {}
            c["mean_score"] = float(np.mean([practice.score(m, goal)[0] for m in c["runs"]])) if c["runs"] else -1e3
        return max(contenders, key=lambda c: c["mean_score"])

    async def _baseline(self, bname, duration, tracks, goal):
        """A skill (usually a built-in one, with the hand-built parts it relies on) measured the same way."""
        sk = self.get(bname)
        if not sk:
            return {"name": bname, "error": f"no skill called {bname}"}
        command = bool(sk.get("command"))
        drivers = [dict(d, rates=list(d["rates"])) for d in sk["drivers"]]
        if command:
            # a command skill drives the command neurons for its own length: keep it going for as long as
            # the new skill lasts
            n = int(round(duration / BIN_S)) + 1
            for d in drivers:
                r = d["rates"] or [0]
                d["rates"] = [r[i % len(r)] for i in range(n)]
            dur = duration
            assists = {"walking": True, "flight_timer": True}
        else:
            dur = float(sk.get("duration") or self.parse_plan(sk["plan"])[0])
            assists = dict({"walking": True, "flight_timer": True}, **(sk.get("assists") or {}))
        res = await self._run([self._trial(drivers, dur, assists, sd, command=command) for sd in (9001, 9002)])
        runs = [self._measure(r, tracks, dur) for r in res if r.get("ok")]
        mean = {k: round(float(np.mean([m[k] for m in runs])), 3) for k in runs[0]}
        on = [k for k, v in assists.items() if v]
        return {"name": bname, "metrics": mean, "score": round(float(np.mean([practice.score(m, goal)[0] for m in runs])), 3),
                "assists": assists, "text": f"{bname} ({'with the ' + ' and '.join(ASSIST_NAME.get(k, k) for k in on) if on else 'no hand-built parts'})"}

    async def test(self, name, assists=None, goal=None, runs=3, duration=None):
        """Run a skill in the practice copies (the live fly isn't touched) and measure what it does, e.g.
        with and without a hand-built part, or a built-in skill against a learned one."""
        sk = self.get(name)
        if not sk:
            return {"ok": False, "error": f"no skill called {name}", "known": sorted(set(self.lib) | set(self.builtin))}
        _, tracks = self.parse_plan(sk["plan"])
        command = bool(sk.get("command"))
        drivers = [dict(d, rates=list(d["rates"])) for d in sk["drivers"]]
        dur = float(duration or (2.0 if command else sk.get("duration") or self.parse_plan(sk["plan"])[0]))
        if command:
            n = int(round(dur / BIN_S)) + 1
            for d in drivers:
                r = d["rates"] or [0]
                d["rates"] = [r[i % len(r)] for i in range(n)]
            tracks = {}
        base = {"walking": True, "flight_timer": True} if (command or not sk.get("assists")) else dict(PRACTICE_ASSISTS, **sk["assists"])
        asg = dict(base, **{k: bool(v) for k, v in (assists or {}).items() if k in PRACTICE_ASSISTS})
        g = practice.parse_goal(goal if goal is not None else sk.get("goal"),
                                uses_jump=any(k.endswith(".jump") for k in tracks) or command,
                                uses_wings=any(k.startswith("wing") for k in tracks) or command)
        await self.pool.ensure()
        runs = int(max(1, min(int(runs or 3), 10)))
        res = await self._run([self._trial(drivers, dur, asg, 9100 + k, command=command) for k in range(runs)])
        ms = [self._measure(r, tracks, dur) for r in res if r.get("ok")]
        mean = {k: round(float(np.mean([m[k] for m in ms])), 3) for k in ms[0]}
        spread = {k: round(float(np.std([m[k] for m in ms])), 3) for k in ms[0]}
        out = {"ok": True, "name": name, "duration": dur, "assists": asg, "runs": len(ms), "metrics": mean, "spread": spread}
        if g is not None:
            out["goal"] = practice.goal_text(g)
            out["score"] = round(float(np.mean([practice.score(m, g)[0] for m in ms])), 3)
        return out

    async def learn(self, name, description, plan, goal=None, trials=None, assists=None, baseline=None, warm=None,
                    levels=None, show=True, timing=True, per="kind", keep_rates=False, drive_at=None):
        """Learn (or, with warm, refine) a skill. Returns a report for the agent."""
        if self.busy:
            return {"ok": False, "error": "busy with another movement"}
        if drive_at is not None:
            if drive_at not in DRIVE_AT:
                return {"ok": False, "error": f"drive_at must be one of {', '.join(DRIVE_AT)}"}
            levels = DRIVE_AT[drive_at]
        self.busy = True
        self._halt = False
        await self._wait_reflex()
        try:
            r = await self._learn(name, description, plan, goal, trials, assists, baseline, warm, levels, timing, per,
                                  keep_rates)
        except (ValueError, RuntimeError) as e:
            self.emit({"type": "skill", "event": "error", "name": name, "error": str(e)})
            return {"ok": False, "error": str(e)}
        finally:
            self.busy = False
        if r.get("ok") and show and r.get("where") == "practice":
            p = await self.perform([name])          # show the result on the live fly
            if p.get("ok"):
                r["shown"] = p["performed"][0][1]
        return r

    async def _learn(self, name, description, plan, goal, trials, assists, baseline, warm, levels, timing, per,
                     keep_rates):
        t0 = time.time()
        keep_rates = bool(keep_rates and warm)
        if keep_rates:
            plan = warm["plan"]
        duration, tracks = self.parse_plan(plan)
        notes = list(self.limit_notes)
        g = practice.parse_goal(goal, uses_jump=any(k.endswith(".jump") for k in tracks),
                                uses_wings=any(k.startswith("wing") for k in tracks))
        asg = dict(PRACTICE_ASSISTS, **{k: bool(v) for k, v in (assists or {}).items() if k in PRACTICE_ASSISTS})
        self.emit({"type": "skill", "event": "search", "name": name})
        if self.E is None:
            await asyncio.get_running_loop().run_in_executor(None, self.build_influence)
        drivers = self.pick_drivers(tracks, levels)
        if warm:
            drivers = self._warm_drivers(drivers, warm, keep_rates)
            if keep_rates:
                duration = float(warm.get("duration") or duration)
        if not drivers:
            raise ValueError("the plan doesn't move anything")
        self.emit({"type": "skill", "event": "practice", "stage": "start", "name": name, "copies": self.pool.n,
                   "starting": not self.pool.ready})
        try:
            await self.pool.ensure()
        except Exception as e:
            if g is not None:
                raise RuntimeError(f"the practice copies couldn't start ({e}), so there's no practising for a goal; "
                                   f"learn it without a goal to shape it on the live fly") from None
            r = await self._learn_live(name, description, plan, duration, tracks, drivers, notes, t0)
            r["note"] = f"practised on the live fly: the background copies couldn't start ({e})"
            return r
        # 1. shape: the muscles follow the plan
        if keep_rates:
            res = (await self._run([self._trial(drivers, duration, asg, seed=11)]))[0]
            sc, per_c, _, ach = self.evaluate(self._as_state(res["rec"]), tracks, duration)
            shaped, shape_hist = {"score": sc, "drivers": drivers, "per": per_c, "ach": ach, "res": res}, [round(sc, 3)]
        else:
            shaped, shape_hist = await self._shape(name, drivers, tracks, duration, asg, SHAPE_TRIES)
        drivers = shaped["drivers"]
        timeline = self.timeline(tracks, shaped["ach"], plan)
        final = {"drivers": drivers, "duration": duration, "metrics": self._measure(shaped["res"], tracks, duration)}
        report = {"shape": shape_hist, "tracking": round(shaped["score"], 3)}
        # 2. practise for the goal
        if g is not None:
            n_trials = int(max(6, min(int(trials or DEFAULT_TRIALS), 400)))
            space, evals, by_gen, done = await self._search(name, drivers, duration, tracks, g, asg, n_trials, timing, per)
            ok = sorted([e for e in evals if e["metrics"] is not None], key=lambda e: -e["score"])
            picks, seen = [], []
            for e in ok:
                if len(picks) >= 3:
                    break
                if not any(np.allclose(e["x"], s) for s in seen):
                    picks.append(e)
                    seen.append(e["x"])
            start = next((e for e in evals if e["start"] and e["metrics"] is not None), None)
            if start is not None and not any(np.allclose(start["x"], s) for s in seen):
                picks.append(start)
            contenders = []
            for e in picks:
                dr, dur = space.apply(drivers, e["x"], duration, BIN_S, MAX_HZ)
                contenders.append({"drivers": dr, "duration": dur, "metrics": e["metrics"], "x": e["x"], "start": e["start"],
                                   "what": space.describe(e["x"])})
            if warm and warm.get("drivers") and not keep_rates:
                # the skill as it was: kept unless something beats it
                contenders.append({"drivers": [dict(d) for d in warm["drivers"]], "duration": float(warm["duration"]),
                                   "metrics": None, "old": True, "what": "the skill as it was"})
            win = await self._check(name, contenders, tracks, g, asg)
            st = next((c for c in contenders if c.get("start")), None)
            final = {"drivers": win["drivers"], "duration": win["duration"], "metrics": win["mean"]}
            report.update({"goal": practice.goal_text(g), "trials": done + 2 * len(contenders), "by_round": by_gen,
                           "score": round(win["mean_score"], 3), "parts": practice.score(win["mean"], g)[1],
                           "change": win["what"], "kept_old": bool(win.get("old")),
                           "start": {"score": round(st["mean_score"], 3), "metrics": st["mean"]} if st else None,
                           "search_space": space.names})
            if baseline:
                report["baseline"] = await self._baseline(baseline, final["duration"], tracks, g)
        # 3. save
        rid = self.w.root_ids
        score = report["score"] if g is not None else report["tracking"]
        skill = {
            "description": description, "plan": plan, "duration": round(final["duration"], 3), "score": score,
            "tracking": report["tracking"], "goal": goal, "goal_text": report.get("goal"), "metrics": final["metrics"],
            "assists": asg, "drive_at": levels or ["descending", "premotor", "motor"], "attempts": shape_hist, "practice": {k: report[k] for k in ("trials", "by_round", "change")
                                                                  if k in report},
            "baseline": report.get("baseline"), "limits": notes, "learned": _now(),
            "dataset": self.w.tables.get("dataset", ""), "uses": (warm or {}).get("uses", 0),
            "drivers": [{"channel": d["channel"], "control": d["control"], "dir": d["dir"], "level": d["level"],
                         "neurons": [int(n) for n in d["neurons"]], "root_ids": [str(rid[n]) for n in d["neurons"]],
                         "rates": [int(x) for x in d["rates"]], **({"trim": True} if d.get("trim") else {})}
                        for d in final["drivers"]],
        }
        self.lib[name] = skill
        self._save()
        secs = round(time.time() - t0, 1)
        self.emit({"type": "skill", "event": "learned", "name": name, "score": score, "attempts": shape_hist,
                   "seconds": secs, "goal": report.get("goal"), "metrics": final["metrics"]})
        return {"ok": True, "where": "practice", "name": name, "score": score, "attempts": shape_hist, "limits": notes,
                "timeline": timeline, "per_control": {k: round(float(v), 2) for k, v in shaped["per"].items()},
                "drivers": [self._describe(d) for d in final["drivers"]], "metrics": final["metrics"],
                "assists": asg, "report": report, "seconds": secs, "copies": len(self.pool.workers)}

    async def _learn_live(self, name, description, plan, duration, tracks, drivers, notes, t0, attempts=8):
        """Shaping on the live fly, one try at a time (when the practice copies can't run)."""
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
                best_tl = self.timeline(tracks, ach, plan)
                best_ball = self.ball_summary(self.result)
            self.emit({"type": "skill", "event": "attempt", "name": name, "attempt": a, "score": round(score, 3),
                       "per_control": {k: round(float(v), 2) for k, v in per.items()}})
            if score >= 0.8 or a == attempts or self._halt:
                break
            changes = self.adjust(drivers, tracks, ach, cross, per, a)
            self.emit({"type": "skill", "event": "adjust", "name": name, "changes": changes})
            if not changes:
                break
            await asyncio.sleep(0.25)
        score, drv, per = best
        rid = self.w.root_ids
        skill = {
            "description": description, "plan": plan, "duration": duration, "score": round(score, 3),
            "tracking": round(score, 3), "attempts": history, "limits": notes, "learned": _now(),
            "dataset": self.w.tables.get("dataset", ""), "uses": 0,
            "assists": {"walking": self.w.body.assists.get("walking", True) if getattr(self.w, "body", None) else True},
            "drivers": [{"channel": d["channel"], "control": d["control"], "dir": d["dir"], "level": d["level"],
                         "neurons": d["neurons"], "root_ids": [str(rid[n]) for n in d["neurons"]],
                         "rates": [int(x) for x in d["rates"]]} for d in drv],
        }
        self.lib[name] = skill
        self._save()
        self.emit({"type": "skill", "event": "learned", "name": name, "score": skill["score"],
                   "attempts": history, "seconds": round(time.time() - t0, 1)})
        return {"ok": True, "where": "live", "name": name, "score": skill["score"], "attempts": history, "limits": notes,
                "timeline": best_tl, "ball": best_ball,
                "per_control": {k: round(float(v), 2) for k, v in per.items()},
                "drivers": [self._describe(d) for d in drv], "seconds": round(time.time() - t0, 1)}

    async def perform(self, names, repeat=1):
        if self.busy:
            return {"ok": False, "error": "busy with another movement"}
        missing = [n for n in names if not self.get(n)]
        if missing:
            return {"ok": False, "error": f"not learned yet: {missing}. Learn them first.",
                    "known": sorted(set(self.lib) | set(self.builtin))}
        self.busy = True
        await self._wait_reflex()
        try:
            done = []
            for _ in range(max(1, min(int(repeat or 1), 20))):
                for name in names:
                    sk = self.get(name)
                    duration, tracks = self.parse_plan(sk["plan"])
                    duration = float(sk.get("duration") or duration)
                    drivers = [dict(d, neurons=list(d["neurons"]), rates=list(d["rates"]), level_i=LEVELS.index(d["level"]))
                               for d in sk["drivers"]]
                    body = getattr(self.w, "body", None)
                    if body is not None and not sk.get("command"):
                        # the hand-built parts as they were when the skill was learned (e.g. no flight timer)
                        body.assist_override = sk.get("assists")
                    try:
                        self.start("play", name, drivers, tracks, duration, command=bool(sk.get("command")))
                        await self.done.wait()
                    finally:
                        if body is not None:
                            body.assist_override = None
                    if sk.get("command"):            # a command skill: what counts is how the fly moved
                        b = self.ball_summary(self.result) or {}
                        done.append((name, f"{b.get('forward_mm_s', 0):+.1f} mm/s, turning {b.get('turn_left_rad_s', 0):+.2f} rad/s (+ left)"))
                    else:
                        score, *_ = self.evaluate(self.result, tracks, duration)
                        b = self.ball_summary(self.result) or {}
                        done.append((name, f"muscles followed the plan {score:.2f}; moved {b.get('forward_mm_s', 0):+.1f} mm/s, "
                                           f"turning {b.get('turn_left_rad_s', 0):+.2f} rad/s"))
                    sk["uses"] = sk.get("uses", 0) + 1
            self._save()
            return {"ok": True, "performed": done}
        finally:
            self.busy = False
