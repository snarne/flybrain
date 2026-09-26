"""
Practice: trying a movement many times, fast, in copies of the fly that run in the background.

Learning a skill on the live fly (the fly you watch) is slow: one try at a time, in real time. So the
skill engine (skills.py) practises in separate copies of the whole model (brain, muscles, body and
physics; practice_worker.py), each in its own process, while the live fly keeps running. Each try
("trial") starts from the same resting fly in an empty arena, drives the skill's neurons, and records
what the muscles and the body did. The trials are scored on what the movement achieved (how far the
fly got, which way it turned, how high it jumped, whether it stayed on its feet, which muscles it used
that it shouldn't have), and a numerical search (CMA-ES) tunes the drive to score better.

What is tuned: for each kind of movement in the plan (e.g. all legs' "lift up", or the jump muscle),
how hard its neurons are driven (a gain) and, optionally, when (a time shift), plus the pace of the
whole movement. The neurons themselves are the ones the wiring search found; the search only changes
how they are driven, so what the body can do is still limited by what the connectome lets those
neurons do.

Hand-built parts of the body (body.ASSISTS: the walking model, the flight timer) are off in practice
unless asked for, so a skill learns to move with the neurons and muscles alone; the built-in versions
can be run the same way as a baseline to compare against.

This module: the trial metrics and goals, the search, and the pool of practice processes.
"""
import asyncio
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKER = HERE / "practice_worker.py"
DT = 1.0 / 30                 # the practice copies run the same 33 ms frames as the live fly

# ---------------------------------------------------------------------------- what a trial measured
# metric: (unit it is scored in, what it is)
METRICS = {
    "forward_mm": (5.0, "how far the fly moved forwards (along the way it faced at the start), mm"),
    "side_mm": (5.0, "how far it moved to its left (- = right), mm"),
    "distance_mm": (5.0, "straight-line distance from start to end, mm"),
    "speed_mm_s": (5.0, "forward_mm divided by the movement's duration, mm/s"),
    "turn_deg": (90.0, "how far it turned, degrees (+ = left)"),
    "height_mm": (3.0, "highest the body got above where it stood, mm"),
    "airtime_s": (0.5, "time with no foot on the ground, s"),
    "upright": (1.0, "how upright it ended (1 = on its feet, -1 = on its back)"),
    "fell": (1.0, "1 if it fell over (on its side or back) at any point, else 0"),
    "wing_power": (0.5, "mean activation of the flight power muscles, 0..1"),
    "jump": (1.0, "peak activation of the jump muscle, 0..1"),
    "launch": (1.0, "peak jump muscle activation above 0.1, where the body model starts to launch the fly, 0..0.9"),
    "unwanted": (0.2, "mean activation of leg, wing and jump muscles the plan doesn't use, 0..1"),
    "tracking": (1.0, "how closely the muscles followed the plan, 0..1"),
    "effort": (1.0, "mean drive of the skill's neurons, as a fraction of the maximum"),
}
# unless a goal says otherwise, falling over and moving muscles the plan doesn't use count against a try;
# and so does setting off the jump muscle (above the threshold where the body model launches) when the plan
# doesn't use it (the body model turns the jump
# muscle into a strong launch, which would otherwise be the easiest way to cover ground)
DEFAULT_PENALTY = {"fell": -1.0, "unwanted": -0.5}
JUMP_PENALTY = -10.0
WING_PENALTY = -1.0      # the flight power muscles, when the plan doesn't use the wings
GOAL_PRESETS = {
    # on the ground: hopping or flying there doesn't count
    "forward": {"forward_mm": 1.0, "airtime_s": -1.0},
    "backward": {"forward_mm": -1.0, "airtime_s": -1.0},
    "turn_left": {"turn_deg": 1.0, "airtime_s": -1.0},
    "turn_right": {"turn_deg": -1.0, "airtime_s": -1.0},
    "jump_high": {"height_mm": 1.0, "wing_power": -1.0},
    "jump_far": {"distance_mm": 1.0, "wing_power": -1.0},
    "fly": {"airtime_s": 1.0, "height_mm": 0.3},
    "fly_forward": {"airtime_s": 1.0, "forward_mm": 0.5},
    "still": {"distance_mm": {"target": 0.0}, "turn_deg": {"target": 0.0}},
    "match_plan": {"tracking": 1.0},
}
LEG_OR_WING = ("LF.", "LM.", "LH.", "RF.", "RM.", "RH.", "wingL.", "wingR.")


def parse_goal(goal, uses_jump=True, uses_wings=True):
    """A goal is a preset name, or {metric: weight} (+ = more is better, - = less), or
    {metric: {"target": value, "weight": w}} (closer is better), or a mix; presets can be combined with
    "+", e.g. "forward+still"... Returns {metric: (kind, weight, target)} or raises ValueError."""
    if goal is None or goal == "":
        return None
    if isinstance(goal, str) and goal.strip().startswith("{"):
        try:
            goal = json.loads(goal)
        except ValueError:
            raise ValueError("goal: not valid JSON") from None
    if isinstance(goal, str):
        g = {}
        for part in goal.split("+"):
            p = part.strip()
            if p not in GOAL_PRESETS:
                raise ValueError(f"unknown goal '{p}'. Presets: {', '.join(GOAL_PRESETS)}; or a dict of metrics: "
                                 f"{', '.join(METRICS)}")
            g.update(GOAL_PRESETS[p])
        goal = g
    if not isinstance(goal, dict):
        raise ValueError("goal must be a preset name or a dict {metric: weight}")
    out = {}
    for k, v in goal.items():
        if k in GOAL_PRESETS and isinstance(v, (int, float)) and k not in METRICS:
            for kk, vv in GOAL_PRESETS[k].items():
                out[kk] = ("target", vv.get("weight", 1.0) * v, vv["target"]) if isinstance(vv, dict) else ("weight", vv * v, None)
            continue
        if k not in METRICS:
            raise ValueError(f"unknown metric '{k}'. Metrics: {', '.join(METRICS)}")
        if isinstance(v, dict):
            if "target" not in v:
                raise ValueError(f"{k}: give a number (weight) or {{\"target\": value}}")
            out[k] = ("target", float(v.get("weight", 1.0)), float(v["target"]))
        else:
            out[k] = ("weight", float(v), None)
    for k, w in DEFAULT_PENALTY.items():
        out.setdefault(k, ("weight", w, None))
    if not uses_jump:
        if "jump" not in out:
            out.setdefault("launch", ("weight", JUMP_PENALTY, None))
    if not uses_wings:
        out.setdefault("wing_power", ("weight", WING_PENALTY, None))
    return out


def score(metrics, goal):
    """One number (higher is better) and each term's share of it."""
    total, parts = 0.0, {}
    for k, (kind, w, target) in goal.items():
        unit = METRICS[k][0]
        x = metrics.get(k, 0.0)
        c = -w * abs(x - target) / unit if kind == "target" else w * x / unit
        parts[k] = round(c, 3)
        total += c
    return total, parts


def goal_text(goal):
    parts = []
    for k, (kind, w, target) in goal.items():
        if kind == "target":
            parts.append(f"{k} close to {target:g}" + (f" (x{w:g})" if w != 1 else ""))
        elif w > 0:
            parts.append(f"more {k}" + (f" (x{w:g})" if w != 1 else ""))
        else:
            parts.append(f"less {k}" + (f" (x{-w:g})" if w != -1 else ""))
    return ", ".join(parts)


def measure(rec, planned, duration, max_hz):
    """Metrics of one trial's recording (see practice_worker.Sandbox.trial)."""
    x, y, z = np.asarray(rec["x"]), np.asarray(rec["y"]), np.asarray(rec["z"])
    h = np.unwrap(np.asarray(rec["heading"]))
    if not len(x):
        return {k: 0.0 for k in METRICS}
    h0 = h[0]
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    fwd = dx * math.cos(h0) + dy * math.sin(h0)
    side = -dx * math.sin(h0) + dy * math.cos(h0)
    val = {k: np.asarray(v) for k, v in rec["val"].items()}
    zeros = np.zeros(len(x))
    power = np.maximum(val.get("wingL.power", zeros), val.get("wingR.power", zeros))
    jump = np.maximum(val.get("LM.jump", zeros), val.get("RM.jump", zeros))
    up = np.asarray(rec["upright"])
    others = sorted(float(np.abs(v).mean()) for k, v in val.items() if k.startswith(LEG_OR_WING) and k not in planned)
    unwanted = float(np.mean(others[-6:])) if others else 0.0
    return {
        "forward_mm": round(float(fwd), 3),
        "side_mm": round(float(side), 3),
        "distance_mm": round(float(math.hypot(dx, dy)), 3),
        "speed_mm_s": round(float(fwd / max(duration, 0.05)), 3),
        "turn_deg": round(float(math.degrees(h[-1] - h0)), 2),
        "height_mm": round(float(z.max() - z[0]), 3),
        "airtime_s": round(float(np.sum(np.asarray(rec["air"], float)) * DT), 3),
        "upright": round(float(up[-1]), 3),
        "fell": 1.0 if (rec.get("righted", 0) > 0 or float(up.min()) < 0.2) else 0.0,
        "wing_power": round(float(power.mean()), 3),
        "jump": round(float(jump.max()), 3),
        "launch": round(max(0.0, float(jump.max()) - 0.1), 3),
        "unwanted": round(unwanted, 3),
        "effort": round(float(np.mean(rec.get("drive", [0.0])) / max_hz), 3),
    }


def metrics_text(m, keys=None):
    keys = keys or ["forward_mm", "side_mm", "turn_deg", "height_mm", "airtime_s", "upright", "fell", "wing_power",
                    "jump", "unwanted", "tracking"]
    return ", ".join(f"{k} {m[k]:+.2f}" if isinstance(m.get(k), float) and k not in ("fell",) else f"{k} {m.get(k)}"
                     for k in keys if k in m)


# ---------------------------------------------------------------------------- the search (CMA-ES)
class CMA:
    """Covariance matrix adaptation evolution strategy (Hansen 2016), maximising. Small and plain."""

    def __init__(self, x0, sigma=1.0, popsize=None, seed=0):
        n = len(x0)
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.lam = popsize or 4 + int(3 * math.log(max(n, 1)))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = np.asarray(x0, float).copy()
        self.sigma = float(sigma)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.gen = 0

    def ask(self):
        vals, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(vals, 1e-20))
        self._B, self._D = B, D
        z = self.rng.standard_normal((self.lam, self.n))
        y = (z * D) @ B.T
        return self.m + self.sigma * y

    def tell(self, xs, fs):
        xs = np.asarray(xs)
        order = np.argsort(-np.asarray(fs))[: self.mu]
        old = self.m
        y = (xs[order] - old) / self.sigma
        self.m = old + self.sigma * (self.w @ y)
        yw = self.w @ y
        B, D = self._B, self._D
        Cinv_half = B @ np.diag(1 / D) @ B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (Cinv_half @ yw)
        self.gen += 1
        hsig = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chiN < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc)
                  + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * (y.T * self.w) @ y)
        self.C = (self.C + self.C.T) / 2
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(min(self.sigma, 3.0))


# how the search changes a skill's drive (in units the search steps in)
GAIN_UNIT, GAIN_LO, GAIN_HI = 0.6, -3.0, 1.2        # log of the drive gain: x0.05 .. x3.3
SHIFT_UNIT, SHIFT_MAX = 0.05, 0.15                  # s
PACE_UNIT, PACE_MAX = 0.2, 0.5                      # log of the time scale: x0.6 .. x1.65


def group_of(driver, per="kind"):
    """Which search parameter a driver belongs to: by default one per kind of movement (all legs'
    'lift+' together, both wings' 'power+'), or one per channel."""
    ch = driver["channel"]
    if per == "channel":
        return ch
    part, _, rest = ch.partition(".")
    if part in ("LF", "LM", "LH", "RF", "RM", "RH"):
        return rest
    if part in ("wingL", "wingR"):
        return "wing." + rest
    if part in ("antennaL+", "antennaR+", "haltereL+", "haltereR+"):
        return part[:-2] + part[-1]
    return ch


class Space:
    """The search space over one skill: per group a log gain (and a time shift), plus the pace."""

    def __init__(self, drivers, timing=True, pace=True, per="kind"):
        self.groups = sorted({group_of(d, per) for d in drivers})
        self.per = per
        self.timing = timing
        self.pace = pace
        self.names = [f"gain {g}" for g in self.groups] + ([f"shift {g}" for g in self.groups] if timing else []) \
            + (["pace"] if pace else [])
        self.n = len(self.names)

    def decode(self, x):
        G = len(self.groups)
        gain = {g: float(np.exp(np.clip(x[i] * GAIN_UNIT, GAIN_LO, GAIN_HI))) for i, g in enumerate(self.groups)}
        shift = {g: float(np.clip(x[G + i] * SHIFT_UNIT, -SHIFT_MAX, SHIFT_MAX)) for i, g in enumerate(self.groups)} \
            if self.timing else {g: 0.0 for g in self.groups}
        pace = float(np.exp(np.clip(x[-1] * PACE_UNIT, -PACE_MAX, PACE_MAX))) if self.pace else 1.0
        return gain, shift, pace

    def describe(self, x):
        gain, shift, pace = self.decode(x)
        parts = [f"{g} x{gain[g]:.2f}" + (f" {1000 * shift[g]:+.0f} ms" if self.timing and abs(shift[g]) > 0.005 else "")
                 for g in self.groups]
        if self.pace:
            parts.append(f"pace x{1 / pace:.2f}")
        return ", ".join(parts)

    def apply(self, drivers, x, duration, bin_s, max_hz):
        """New drivers (rates resampled) and duration for parameters x."""
        gain, shift, pace = self.decode(x)
        new_dur = duration * pace
        out = []
        for d in drivers:
            g = group_of(d, self.per)
            r = np.asarray(d["rates"], float)
            n_new = int(round(new_dur / bin_s)) + 1
            t = np.arange(n_new) * bin_s
            src = (t - shift[g]) / pace / bin_s          # position in the old schedule, in bins
            rr = np.interp(src, np.arange(len(r)), r, left=0.0, right=0.0)
            rr = np.clip(rr * gain[g], 0, max_hz)
            out.append(dict(d, rates=rr.round(0).tolist()))
        return out, new_dur


# ---------------------------------------------------------------------------- the practice processes
def default_workers():
    env = os.environ.get("FLYBRAIN_PRACTICE_WORKERS")
    if env:
        return max(1, int(env))
    cpus = os.cpu_count() or 2
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        ram_gb = 8.0
    # each copy needs a core and about 1 GB; leave room for the live fly and the LLM
    return int(max(1, min(4, (cpus - 2) // 2, (ram_gb - 6) // 2)))


class Worker:
    def __init__(self, k):
        self.k = k
        self.proc = None
        self.lock = asyncio.Lock()

    async def start(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1", NO_BROWSER="1", FLYBRAIN_WORKER="1")
        self.proc = await asyncio.create_subprocess_exec(sys.executable, str(WORKER), cwd=str(HERE.parent), env=env,
                                                         stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                         limit=2 ** 27)
        line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=600)
        msg = json.loads(line or b"{}")
        if not msg.get("ready"):
            raise RuntimeError(msg.get("error") or "the practice copy didn't start")
        return msg

    def alive(self):
        return self.proc is not None and self.proc.returncode is None

    async def call(self, req):
        async with self.lock:
            if not self.alive():
                await self.start()
            self.proc.stdin.write((json.dumps(req) + "\n").encode())
            await self.proc.stdin.drain()
            line = await self.proc.stdout.readline()
            if not line:
                raise RuntimeError("the practice copy stopped")
            return json.loads(line)

    def stop(self):
        if self.alive():
            try:
                self.proc.stdin.close()
                self.proc.kill()
            except Exception:
                pass


class Pool:
    """Practice copies of the fly, started on first use and kept for later practice."""

    def __init__(self, n=None):
        self.n = n or default_workers()
        self.workers = []
        self.error = None
        self._starting = None

    @property
    def ready(self):
        return bool(self.workers) and all(w.alive() for w in self.workers)

    async def ensure(self):
        if self.ready:
            return True
        if self._starting is None:
            self._starting = asyncio.ensure_future(self._start())
        try:
            await self._starting
        finally:
            self._starting = None
        return self.ready

    async def _start(self):
        ws = [Worker(k) for k in range(self.n)]
        res = await asyncio.gather(*(w.start() for w in ws), return_exceptions=True)
        ok = [w for w, r in zip(ws, res) if not isinstance(r, Exception)]
        bad = [r for r in res if isinstance(r, Exception)]
        for w in ws:
            if w not in ok:
                w.stop()
        self.workers = ok
        self.error = str(bad[0]) if bad and not ok else None
        if self.error:
            raise RuntimeError(self.error)

    async def run(self, reqs, on_result=None):
        """Run requests on the copies in parallel. Results in the same order."""
        if not await self.ensure():
            raise RuntimeError(self.error or "no practice copies")
        q = asyncio.Queue()
        for i, r in enumerate(reqs):
            q.put_nowait((i, r))
        out = [None] * len(reqs)

        async def drain(w):
            while not q.empty():
                i, r = q.get_nowait()
                try:
                    out[i] = await w.call(r)
                except Exception as e:
                    out[i] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                if on_result:
                    on_result(i, out[i])
        await asyncio.gather(*(drain(w) for w in self.workers))
        return out

    def stop(self):
        for w in self.workers:
            w.stop()
        self.workers = []
