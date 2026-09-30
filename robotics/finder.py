"""
The failure finder: searches the conditions a robot controller will meet (friction, payload, weak
motors, delays, sensor noise, shoves, slopes, perception errors, object size and weight, ...) for the
ones that make it fail, then checks each failure and whether a recovery controller fixes it.

For each robot, within its time budget:
  1. nominal    the controllers under normal conditions (both should succeed)
  1b. limits    each condition alone, from normal towards each end of its range: where it starts to fail
  2. explore    random variation around normal conditions (every condition at once, sd a fifth of its
                range): how often it fails, and which conditions matter
  3. search     adversarial search (CMA-ES in the normalised space), alternating two goals:
                  worst    the lowest success margin (how badly can it fail?)
                  mildest  the smallest change from normal that still fails (the failures most likely
                           to happen in practice)
  4. distil     failures grouped into distinct cases; each one re-run on 5 fresh seeds (is it
                reproducible?) and with each changed condition put back to normal in turn (which
                conditions are needed for it to fail: its causes)
  5. recovery   the recovery controller on each distinct failure (5 seeds: fixed?), and on the explored
                conditions (does it break anything that worked?)

Every run returns a margin: > 0 success, < 0 failure, how far from the edge. Runs where the simulator
itself became unstable are counted apart and never reported as robot failures.
"""
import csv
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from .space import nominal, severity, to_params, to_unit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "failure_finder"

_TASKS = {}


def get_task(name):
    if name not in _TASKS:
        if name == "go2":
            from .go2 import Go2
            _TASKS[name] = Go2()
        elif name == "panda":
            from .panda import Panda
            _TASKS[name] = Panda()
        else:
            raise ValueError(name)
    return _TASKS[name]


def _job(args):
    robot, params, seed, recovery = args
    try:
        m, info = get_task(robot).run(params, seed=seed, recovery=recovery)
    except Exception as e:                      # a crash is reported, never hidden
        m, info = float("nan"), {"why": f"error: {type(e).__name__}: {e}", "unstable": True}
    return m, info


class SepCMA:
    """Separable CMA-ES in the unit cube, minimising."""

    def __init__(self, x0, sigma, lam, rng):
        n = len(x0)
        self.n, self.lam, self.rng = n, lam, rng
        self.mu = lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1 / np.sum(self.w ** 2)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.ds = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = 4 / (n + 4)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff) * (n + 2) / 3
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff) * (n + 2) / 3)
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m, self.sigma = np.asarray(x0, float), sigma
        self.D, self.ps, self.pc = np.ones(n), np.zeros(n), np.zeros(n)

    def ask(self):
        self.z = self.rng.standard_normal((self.lam, self.n))
        return np.clip(self.m + self.sigma * self.z * np.sqrt(self.D), 0, 1)

    def tell(self, xs, fs):
        o = np.argsort(fs)[: self.mu]
        y = (xs[o] - self.m) / self.sigma
        yw, zw = self.w @ y, self.w @ self.z[o]
        self.m = np.clip(self.m + self.sigma * yw, 0, 1)
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * zw
        self.pc = (1 - self.cc) * self.pc + math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.D = np.clip((1 - self.c1 - self.cmu) * self.D + self.c1 * self.pc ** 2 + self.cmu * (self.w @ y ** 2), 1e-6, 1e3)
        self.sigma = float(min(0.5, self.sigma * math.exp(self.cs / self.ds * (np.linalg.norm(self.ps) / self.chiN - 1))))


class Finder:
    def __init__(self, robot, pool, hours, seed=0, log=print):
        self.robot, self.pool, self.log = robot, pool, log
        self.task = get_task(robot)
        self.space = self.task.space
        self.names = [d.name for d in self.space]
        self.u0 = to_unit(self.space, nominal(self.space))
        self.rng = np.random.default_rng(seed)
        self.budget = hours * 3600
        self.t0 = time.time()
        self.dir = OUT / robot
        self.dir.mkdir(parents=True, exist_ok=True)
        self.samples = []            # every run: dict
        self.fails = []              # failing runs of the baseline controller
        self.unstable = 0
        self._csv = (self.dir / "samples.csv").open("w", newline="")
        self._w = csv.writer(self._csv)
        self._w.writerow(["phase", "controller", "seed"] + self.names + ["margin", "why", "unstable"])

    # ------------------------------------------------------------ running
    def left(self):
        return self.budget - (time.time() - self.t0)

    def evaluate(self, phase, params_list, seeds, recovery=False):
        jobs = [(self.robot, p, int(s), recovery) for p, s in zip(params_list, seeds)]
        res = self.pool.map(_job, jobs, chunksize=1)
        out = []
        for (_, p, s, _), (m, info) in zip(jobs, res):
            un = bool(info.get("unstable")) or not np.isfinite(m)
            r = {"phase": phase, "recovery": recovery, "seed": s, "params": p, "margin": float(m) if np.isfinite(m) else None,
                 "why": info.get("why", ""), "unstable": un, "info": info}
            self._w.writerow([phase, "recovery" if recovery else "baseline", s] + [round(p[n], 5) for n in self.names]
                             + [None if r["margin"] is None else round(r["margin"], 4), r["why"], int(un)])
            self.samples.append(r)
            if un:
                self.unstable += 1
            elif not recovery and r["margin"] < 0:
                self.fails.append(r)
            out.append(r)
        self._csv.flush()
        return out

    def seeds(self, n):
        return self.rng.integers(0, 2 ** 31 - 1, n)

    # ------------------------------------------------------------ phases
    def run(self):
        n_w = self.pool._processes
        S = {"robot": self.robot, "space": [d.__dict__ for d in self.space], "hours": self.budget / 3600}
        # 1. nominal
        nom = nominal(self.space)
        base = self.evaluate("nominal", [nom] * 5, range(5))
        rec = self.evaluate("nominal", [nom] * 5, range(5), recovery=True)
        S["nominal"] = {"baseline": [r["margin"] for r in base], "recovery": [r["margin"] for r in rec]}
        self.log(f"[{self.robot}] nominal margin: baseline {np.mean(S['nominal']['baseline']):.2f}, "
                 f"recovery {np.mean(S['nominal']['recovery']):.2f}")
        # 1b. one condition at a time: how far can each go from normal before it fails?
        S["limits"] = self.limits()
        # 2. explore
        explore = []
        t_end = time.time() + 0.25 * self.left()
        while time.time() < t_end:
            # variation around normal (sd a fifth of each range), the conditions a deployed robot meets most
            U = np.clip(self.u0 + self.rng.normal(0, 0.2, (n_w * 4, len(self.space))), 0, 1)
            explore += self.evaluate("explore", [to_params(self.space, u) for u in U], self.seeds(len(U)))
        ok = [r for r in explore if not r["unstable"]]
        S["explore"] = {"runs": len(explore), "fail_rate": float(np.mean([r["margin"] < 0 for r in ok])) if ok else None}
        self.log(f"[{self.robot}] explore: {len(explore)} random variations, {100 * S['explore']['fail_rate']:.1f}% fail")
        # 3. adversarial search
        t_end = time.time() + 0.45 / 0.75 * self.left()
        runs, k = [], 0
        lam = max(8, n_w * 2)
        while time.time() < t_end:
            mode = "worst" if k % 2 == 0 else "mildest"
            x0 = self.rng.random(len(self.space)) if mode == "worst" else self.u0.copy()
            es = SepCMA(x0, 0.3 if mode == "worst" else 0.12, lam, self.rng)
            best = (math.inf, None)
            for g in range(25):
                if time.time() >= t_end:
                    break
                X = es.ask()
                ps = [to_params(self.space, x) for x in X]
                rs = self.evaluate(f"search_{mode}", ps, self.seeds(len(X)))
                fs = []
                for p, r in zip(ps, rs):
                    m = 5.0 if r["margin"] is None else r["margin"]
                    f = m if mode == "worst" else severity(self.space, p) + 3.0 * max(0.0, m + 0.05)
                    fs.append(f)
                es.tell(X, np.array(fs))
                i = int(np.argmin(fs))
                if fs[i] < best[0]:
                    best = (fs[i], rs[i])
            runs.append({"mode": mode, "best_objective": best[0],
                         "best_margin": None if best[1] is None else best[1]["margin"]})
            k += 1
        S["search"] = {"restarts": runs, "runs": sum(1 for r in self.samples if r["phase"].startswith("search"))}
        self.log(f"[{self.robot}] search: {len(runs)} searches, {len(self.fails)} failing runs so far")
        # 4. distil: reproducible? which conditions cause it? the same failure with everything else normal
        cands = self.distinct(max_n=40)
        self.log(f"[{self.robot}] {len(cands)} distinct failures; checking them")
        groups, flaky, checked = {}, 0, 0
        for rep in cands:
            if self.left() < 0.12 * self.budget:
                break
            checked += 1
            p = rep["found"]
            rr = self.evaluate("reproduce", [p] * 5, self.seeds(5))
            ok = [r for r in rr if not r["unstable"]]
            rep["fail_rate"] = float(np.mean([r["margin"] < 0 for r in ok])) if ok else 0.0
            if rep["fail_rate"] < 0.6:
                flaky += 1
                continue
            u = to_unit(self.space, p)
            changed = [d for j, d in enumerate(self.space) if abs(u[j] - self.u0[j]) > 0.02]
            tests = []
            for d in changed:
                tests += [dict(p, **{d.name: d.nominal})] * 2
            rs = self.evaluate("causes", tests, self.seeds(len(tests)))
            causes = []
            for i, d in enumerate(changed):
                ms = [r["margin"] for r in rs[2 * i:2 * i + 2] if r["margin"] is not None]
                if ms and np.mean(ms) >= 0:
                    causes.append(d.name)
            rep["causes"] = causes
            # everything that isn't a cause back to normal: does it still fail?
            q = dict(nominal(self.space), **{c: p[c] for c in causes})
            rs = self.evaluate("minimal", [q] * 3, self.seeds(3))
            ms = [r["margin"] for r in rs if r["margin"] is not None]
            rep["params"] = q if causes and ms and sum(m < 0 for m in ms) >= 2 else p
            rep["minimal"] = rep["params"] is q
            rep["severity"] = round(severity(self.space, rep["params"]), 3)
            key = tuple(sorted(causes)) if rep["minimal"] else ("(several together)",) + tuple(sorted(causes))
            g = groups.setdefault(key, {"causes": list(key), "found": 0, "examples": []})
            g["found"] += 1
            g["examples"].append(rep)
        reps = []
        for key, g in groups.items():
            best = min(g["examples"], key=lambda r: r["severity"])
            reps.append(dict(best, causes=g["causes"], found=g["found"], found_params=best["found"]))
        reps.sort(key=lambda r: (-r["found"], r["severity"]))
        S["distil"] = {"checked": checked, "flaky": flaky, "groups": len(reps)}
        self.log(f"[{self.robot}] {checked} checked: {flaky} not reproducible, {len(reps)} distinct causes")
        # 5. recovery: on each kind of failure, and on the explored conditions (does it break anything?)
        for rep in reps:
            rr = self.evaluate("recovery", [rep["params"]] * 5, self.seeds(5), recovery=True)
            rep["recovery"] = [r["margin"] for r in rr]
            rep["recovery_events"] = sorted({e for r in rr for e in r["info"].get("events", [])})
            ok = [r for r in rr if not r["unstable"]]
            rep["fixed_rate"] = float(np.mean([r["margin"] >= 0 for r in ok])) if ok else 0.0
        sub = [r for r in explore if not r["unstable"]]
        sub = [sub[i] for i in self.rng.permutation(len(sub))[: max(40, int(len(sub) * 0.5))]]
        budget_left = self.left()
        t_per = max(1e-3, (time.time() - self.t0) / max(1, len(self.samples)))
        sub = sub[: max(20, int(budget_left / t_per * 0.9))]
        rr = self.evaluate("regression", [r["params"] for r in sub], [r["seed"] for r in sub], recovery=True)
        pairs = [(a, b) for a, b in zip(sub, rr) if not b["unstable"]]
        S["recovery_on_explored"] = {
            "runs": len(pairs),
            "baseline_fail_rate": float(np.mean([a["margin"] < 0 for a, _ in pairs])) if pairs else None,
            "recovery_fail_rate": float(np.mean([b["margin"] < 0 for _, b in pairs])) if pairs else None,
            "fixed": int(sum(a["margin"] < 0 <= b["margin"] for a, b in pairs)),
            "broken": int(sum(b["margin"] < 0 <= a["margin"] for a, b in pairs)),
            "broken_examples": [{"params": a["params"], "baseline": a["margin"], "recovery": b["margin"], "why": b["why"]}
                                for a, b in pairs if b["margin"] < 0 <= a["margin"]][:5],
        }
        ro = S["recovery_on_explored"]
        self.log(f"[{self.robot}] recovery on explored conditions: fails {100 * (ro['baseline_fail_rate'] or 0):.1f}% -> "
                 f"{100 * (ro['recovery_fail_rate'] or 0):.1f}%, fixed {ro['fixed']}, broke {ro['broken']}")
        S["runs"] = len(self.samples)
        S["unstable_runs"] = self.unstable
        S["minutes"] = round((time.time() - self.t0) / 60, 1)
        S["sensitivity"] = self.sensitivity(explore)
        for rep in reps:
            rep.pop("info", None)
        (self.dir / "failures.json").write_text(json.dumps(reps, indent=1))
        (self.dir / "summary.json").write_text(json.dumps(S, indent=1))
        self._csv.close()
        return S, reps

    def limits(self, steps=8, seeds=2):
        """Each condition alone, from normal towards each end of its range: the first value that fails (mean margin
        < 0 over the seeds), or None if it never does."""
        jobs, keys = [], []
        for d in self.space:
            for end in (d.lo, d.hi):
                if abs(end - d.nominal) < 1e-9:
                    continue
                for k in range(1, steps + 1):
                    v = d.nominal + (end - d.nominal) * k / steps
                    for _ in range(seeds):
                        jobs.append(dict(nominal(self.space), **{d.name: v}))
                        keys.append((d.name, "down" if end < d.nominal else "up", v))
        rs = self.evaluate("limits", jobs, self.seeds(len(jobs)))
        acc = {}
        for key, r in zip(keys, rs):
            if r["margin"] is not None:
                acc.setdefault(key, []).append(r["margin"])
        out = {}
        for d in self.space:
            o = {}
            for side in ("down", "up"):
                pts = sorted((v, float(np.mean(ms))) for (n, sd, v), ms in acc.items() if n == d.name and sd == side)
                if side == "down":
                    pts = pts[::-1]
                first = next((v for v, m in pts if m < 0), None)
                if pts:
                    o[side] = {"fails_from": None if first is None else round(first, 4),
                               "margins": [(round(v, 4), round(m, 3)) for v, m in pts]}
            out[d.name] = o
        fails = [f"{n} {'<=' if sd == 'down' else '>='} {o[sd]['fails_from']:g}" for n, o in out.items() for sd in o
                 if o[sd]["fails_from"] is not None]
        self.log(f"[{self.robot}] one condition at a time, fails at: {', '.join(fails) or 'none'}")
        return out

    def distinct(self, max_n=30):
        """Failures grouped: mildest first, each new one at least 0.25 (normalised) from those kept."""
        fs = sorted(self.fails, key=lambda r: severity(self.space, r["params"]))
        reps = []
        for r in fs:
            u = to_unit(self.space, r["params"])
            if all(np.linalg.norm(u - to_unit(self.space, q["found"])) > 0.25 for q in reps):
                reps.append({"found": r["params"], "params": r["params"], "margin": r["margin"], "why": r["why"],
                             "phase": r["phase"]})
            if len(reps) >= max_n:
                break
        return reps

    def sensitivity(self, explore):
        """How much each condition matters, from the random runs: the difference in its mean (normalised) between
        failing and succeeding runs, and the failure rate in the worst fifth of its range."""
        ok = [r for r in explore if not r["unstable"]]
        if len(ok) < 10:
            return {}
        U = np.array([to_unit(self.space, r["params"]) for r in ok])
        F = np.array([r["margin"] < 0 for r in ok])
        out = {}
        for j, d in enumerate(self.space):
            bins = np.clip((U[:, j] * 5).astype(int), 0, 4)
            rates = [float(F[bins == b].mean()) if (bins == b).any() else None for b in range(5)]
            diff = float(U[F, j].mean() - U[~F, j].mean()) if F.any() and (~F).any() else 0.0
            out[d.name] = {"shift": round(diff, 3), "fail_rate_by_fifth": rates}
        return out


def make_pool(n):
    return mp.get_context("spawn").Pool(n)


def default_workers():
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        ram_gb = 8.0
    return max(1, min((os.cpu_count() or 2) - 1, int(ram_gb - 2)))
