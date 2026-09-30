"""
Learned walking: can the connectome's own leg circuits make the fly step, with the hand-built walking
model off, if practice may adjust them (connection strengths between cell types, tonic drive and fast
adaptation per cell type; never who connects to whom)? See server/tuning.py for what is adjusted and
how a walk is scored.

Two arms run side by side, each with its own practice copies of the fly:
  real      the BANC wiring
  shuffled  the control: the same neurons and synapse counts, partners shuffled (if this learns to
            walk as well as the real wiring, the result says nothing about the connectome)

Each arm: separable CMA-ES over the leg-circuit settings. A trial = 0.5 s standing, then the forward
command neurons (P9, DNp09) driven at 150 Hz for 2 s, every hand-built part off. Every few generations
the current best is checked again on fresh noise (3 seeds); only checked scores count as results.
Before the search, the reference: the same trial with the walking model on (how the app walks now),
and with it off on the untouched wiring.

Stops after --hours and writes results/learned_walking/: report.md with charts, and per arm
progress.csv, validation.csv, best.json, state.json (the search can resume: run it again).

    tools/experiments/run_overnight.sh learned_walking --hours 8
    python tools/experiments/learned_walking.py --quick        # a few minutes, to check it runs
    python tools/experiments/learned_walking.py --report       # redo the report from saved results
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
import practice  # noqa: E402
import tuning  # noqa: E402

OUT = ROOT / "results" / "learned_walking"
REST_S, CMD_S, BIN_S, RATE = 0.5, 2.0, 0.02, 150.0
OFF = {"walking": False, "flight_timer": False, "sense_boost": False}
ARMS = {"real": {"variant": "real", "variant_seed": 0}, "shuffled": {"variant": "shuffled", "variant_seed": 0}}
VAL_SEEDS = (9001, 9002, 9003)
TUNE_CFG = {"n_types": 48, "n_pairs": 160, "min_syn": 30}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def trial_req(world, cmd_neurons, seed, tune=None, assists=None):
    n_rest, n_cmd = int(round(REST_S / BIN_S)), int(round(CMD_S / BIN_S))
    return {"op": "trial", "world": world, "seed": int(seed), "command": True, "angles": True,
            "drivers": [{"neurons": cmd_neurons, "rates": [0.0] * n_rest + [RATE] * n_cmd}],
            "bin_s": BIN_S, "duration": REST_S + CMD_S, "tail": 0.0, "settle": 0.3,
            "assists": dict(OFF, **(assists or {})), "tune": None if tune is None else list(map(float, tune)),
            "tune_cfg": TUNE_CFG}


def small_rec(rec):
    """What the report plots from a trial: time, the swing joint of each leg, position."""
    return {"t": rec["t"], "coxa": {leg: rec["ang"][leg]["Coxa"] for leg in tuning.LEGS},
            "x": rec["x"], "y": rec["y"]}


class Arm:
    def __init__(self, name, world, n_workers, args):
        self.name, self.world, self.args = name, world, args
        self.dir = OUT / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pool = practice.Pool(n_workers)
        self.st = None

    async def evaluate(self, reqs):
        res = await self.pool.run(reqs)
        out = []
        for r in res:
            if not r.get("ok"):
                out.append((-10.0, {"error": r.get("error")}, None))
                continue
            s, parts = tuning.score_walk(r["rec"], REST_S, practice.DT, self.setup["halves"])
            out.append((s, parts, r["rec"]))
        return out

    async def check(self, x, label):
        """A setting's score on fresh noise (the validation seeds): mean score, mean parts, one recording."""
        res = await self.evaluate([trial_req(self.world, self.cmd, s, x) for s in VAL_SEEDS])
        return summarise(res, label)

    async def run(self, deadline):
        a = self.args
        await self.pool.ensure()
        # every copy finds the adjustable circuits in its wiring (the same on all of them)
        setups = await asyncio.gather(*(w.call(dict(op="tune_setup", world=self.world, **TUNE_CFG))
                                        for w in self.pool.workers))
        bad = [s for s in setups if not s.get("ok")]
        if bad:
            raise RuntimeError(bad[0].get("error"))
        self.setup = setups[0]
        self.cmd = self.setup["command_neurons"]
        n = self.setup["n"]
        log(f"[{self.name}] {n} settings: {self.setup['pairs']} connections between cell types, "
            f"{self.setup['types']} premotor types; {len(self.cmd)} command neurons")
        save_json(self.dir / "setup.json", {k: self.setup[k] for k in ("n", "labels", "types", "pairs", "describe_x0")}
                  | {"command_neurons": len(self.cmd), "world": self.world, "trial": trial_req(self.world, [], 0) | {"drivers": "P9 (DNp09) at 150 Hz after 0.5 s"}})
        sp = self.dir / "state.json"
        if sp.exists():
            self.st = json.loads(sp.read_text())
            if self.st.get("n") != n:
                raise RuntimeError(f"{sp} is from a different search ({self.st.get('n')} settings, now {n}); "
                                   "move it away to start again")
            es = tuning.SepCMA.restore(self.st["cma"], seed=self.st["gen"] + 1)
            log(f"[{self.name}] resuming at generation {self.st['gen']}, best checked score {self.st['best']['score']:.3f}")
        else:
            x0 = np.asarray(self.setup["x0"])
            refs = {}
            log(f"[{self.name}] reference trials")
            refs["walking_model"] = summarise(await self.evaluate(
                [trial_req(self.world, self.cmd, s, None, {"walking": True}) for s in VAL_SEEDS]), "walking model on")
            refs["untouched"] = summarise(await self.evaluate(
                [trial_req(self.world, self.cmd, s, None) for s in VAL_SEEDS]), "walking model off, wiring as it is")
            refs["start"] = summarise(await self.evaluate(
                [trial_req(self.world, self.cmd, s, x0) for s in VAL_SEEDS]), "walking model off, search start")
            for k, v in refs.items():
                log(f"[{self.name}] {v['label']}: score {v['score']:.3f} {fmt(v['parts'])}")
            es = tuning.SepCMA(x0, a.sigma, a.popsize, seed=a.seed)
            self.st = {"n": n, "gen": 0, "trials": 0, "hours": 0.0, "refs": refs,
                       "best": dict(refs["start"], x=x0.tolist(), gen=0), "cma": es.state()}
            save_json(sp, self.st)
        t_start, h0 = time.time(), self.st["hours"]
        prog = self.dir / "progress.csv"
        new = not prog.exists()
        pf = prog.open("a", newline="")
        pw = csv.writer(pf)
        if new:
            pw.writerow(["gen", "hours", "trials", "sigma", "gen_mean", "gen_best", "forward_mm", "rhythm", "tripod",
                         "step_hz", "rest_quiet", "fell", "launch", "wing_power"])
        val = self.dir / "validation.csv"
        newv = not val.exists()
        vf = val.open("a", newline="")
        vw = csv.writer(vf)
        if newv:
            vw.writerow(["gen", "hours", "which", "score", "forward_mm", "rhythm", "tripod", "step_hz", "rest_quiet",
                         "fell", "launch", "wing_power", "new_best"])
        gen_best = None
        try:
            while time.time() < deadline:
                g = self.st["gen"]
                xs = es.ask()
                seed = 1000 + g                     # the same noise for the whole generation: a fair comparison
                res = await self.evaluate([trial_req(self.world, self.cmd, seed, x) for x in xs])
                fs = [r[0] for r in res]
                es.tell(xs, fs)
                i = int(np.argmax(fs))
                if gen_best is None or fs[i] > gen_best[0]:
                    gen_best = (fs[i], xs[i])
                self.st["gen"] = g + 1
                self.st["trials"] += len(xs)
                self.st["hours"] = h0 + (time.time() - t_start) / 3600
                p = res[i][1]
                pw.writerow([g + 1, round(self.st["hours"], 4), self.st["trials"], round(es.sigma, 4),
                             round(float(np.mean(fs)), 4), round(fs[i], 4)] + [p.get(k, "") for k in PARTS])
                pf.flush()
                if (g + 1) % a.validate_every == 0 or time.time() >= deadline:
                    # the search's mean and the best single try of the last few generations, on fresh noise
                    for which, x in (("mean", es.m.copy()), ("best_try", gen_best[1])):
                        c = await self.check(x, which)
                        better = c["score"] > self.st["best"]["score"]
                        vw.writerow([g + 1, round(self.st["hours"], 4), which, round(c["score"], 4)]
                                    + [c["parts"].get(k, "") for k in PARTS] + [int(better)])
                        vf.flush()
                        if better:
                            self.st["best"] = dict(c, x=list(map(float, x)), gen=g + 1)
                            log(f"[{self.name}] gen {g + 1}: new best {c['score']:.3f} {fmt(c['parts'])}")
                    gen_best = None
                    self.st["cma"] = es.state()
                    save_json(sp, self.st)
                if (g + 1) % 5 == 0:
                    log(f"[{self.name}] gen {g + 1}  mean {np.mean(fs):.3f}  best {fs[i]:.3f}  sigma {es.sigma:.3f}  "
                        f"best checked {self.st['best']['score']:.3f}")
        finally:
            self.st["cma"] = es.state()
            save_json(sp, self.st)
            pf.close()
            vf.close()
        # what the best setting changes, in words
        w = self.pool.workers[0]
        d = await w.call({"op": "tune_describe", "world": self.world, "tune": self.st["best"]["x"], "top": 15})
        best = {k: v for k, v in self.st["best"].items()}
        best["describe"] = d.get("describe")
        best["labels"] = self.setup["labels"]
        save_json(self.dir / "best.json", best)
        self.pool.stop()


PARTS = ["forward_mm", "rhythm", "tripod", "step_hz", "rest_quiet", "fell", "launch", "wing_power"]


def summarise(res, label):
    ok = [r for r in res if r[2] is not None]
    parts = {k: round(float(np.mean([r[1].get(k, 0.0) for r in ok])), 3) for k in PARTS} if ok else {}
    return {"label": label, "score": float(np.mean([r[0] for r in res])),
            "scores": [round(r[0], 3) for r in res], "parts": parts,
            "rec": small_rec(ok[0][2]) if ok else None}


def fmt(p):
    return " ".join(f"{k}={p[k]}" for k in ("forward_mm", "rhythm", "tripod", "step_hz") if k in p)


# ---------------------------------------------------------------- the report (plain SVG, no packages)
COLORS = {"real": "#1f77b4", "shuffled": "#d62728"}


def svg_lines(series, title, xlabel, ylabel, hlines=(), w=720, h=320, yticks=True, end_labels=False):
    """series: [(name, color, [(x, y)])]; hlines: [(label, color, y)]. end_labels: each name next to the end of
    its line (else a legend on the right)."""
    pts = [p for _, _, s in series for p in s] + [(0, y) for _, _, y in hlines]
    if not pts:
        return ""
    x0, x1 = 0.0, max(max(p[0] for p in pts), 1e-6)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    if y1 - y0 < 1e-6:
        y0, y1 = y0 - 1, y1 + 1
    pad = (y1 - y0) * 0.08
    y0, y1 = y0 - pad, y1 + pad
    L, R, T, B = 60, 150, 30, 40

    def X(v):
        return L + (v - x0) / (x1 - x0) * (w - L - R)

    def Y(v):
        return h - B - (v - y0) / (y1 - y0) * (h - T - B)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="12">',
         f'<rect width="{w}" height="{h}" fill="white"/>',
         f'<text x="{L}" y="18" font-size="14">{title}</text>',
         f'<line x1="{L}" y1="{h - B}" x2="{w - R}" y2="{h - B}" stroke="#333"/>',
         f'<line x1="{L}" y1="{T}" x2="{L}" y2="{h - B}" stroke="#333"/>',
         f'<text x="{(L + w - R) / 2}" y="{h - 8}" text-anchor="middle">{xlabel}</text>',
         f'<text x="14" y="{(T + h - B) / 2}" transform="rotate(-90 14 {(T + h - B) / 2})" text-anchor="middle">{ylabel}</text>']
    for k in range(5):
        v = y0 + (y1 - y0) * k / 4
        if yticks:
            o.append(f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
        v = x0 + (x1 - x0) * k / 4
        o.append(f'<text x="{X(v):.1f}" y="{h - B + 16}" text-anchor="middle">{v:.2g}</text>')
    ly = T + 4
    for label, color, y in hlines:
        o.append(f'<line x1="{L}" y1="{Y(y):.1f}" x2="{w - R}" y2="{Y(y):.1f}" stroke="{color}" stroke-dasharray="5,4"/>')
        o.append(f'<line x1="{w - R + 6}" y1="{ly + 8}" x2="{w - R + 22}" y2="{ly + 8}" stroke="{color}" stroke-dasharray="4,3"/>')
        o.append(f'<text x="{w - R + 26}" y="{ly + 12}" fill="{color}">{label}</text>')
        ly += 16
    for name, color, s in series:
        if not s:
            continue
        d = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in s)
        o.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1.6"/>')
        if end_labels:
            o.append(f'<text x="{w - R + 6}" y="{Y(s[-1][1]) + 4:.1f}" fill="{color}">{name}</text>')
        else:
            o.append(f'<line x1="{w - R + 6}" y1="{ly + 8}" x2="{w - R + 22}" y2="{ly + 8}" stroke="{color}" stroke-width="2"/>')
            o.append(f'<text x="{w - R + 26}" y="{ly + 12}" fill="{color}">{name}</text>')
            ly += 16
    o.append("</svg>")
    return "\n".join(o)


def svg_legs(rec, title, w=720, h=300):
    """The swing joint of each leg over time, tripod groups in two colours."""
    if not rec:
        return ""
    t = rec["t"]
    ser = []
    for i, leg in enumerate(tuning.LEGS):
        a = np.asarray(rec["coxa"][leg], float)
        a = (a - a.mean()) / (np.ptp(a) + 1e-6) * 0.8 + len(tuning.LEGS) - 1 - i
        ser.append((leg, "#1f77b4" if tuning.TRIPOD[leg] == 0 else "#ff7f0e", list(zip(t, a))))
    return svg_lines(ser, title, "time (s), command starts at 0.5 s", "each leg's swing joint", w=w, h=h,
                     yticks=False, end_labels=True)


def load_csv(p):
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def report():
    arms = [a for a in ARMS if (OUT / a / "state.json").exists()]
    if not arms:
        print("no results yet")
        return
    st = {a: json.loads((OUT / a / "state.json").read_text()) for a in arms}
    best = {a: json.loads((OUT / a / "best.json").read_text()) for a in arms if (OUT / a / "best.json").exists()}
    val = {a: load_csv(OUT / a / "validation.csv") for a in arms}
    prog = {a: load_csv(OUT / a / "progress.csv") for a in arms}
    charts = OUT / "charts"
    charts.mkdir(exist_ok=True)
    # checked score over time: the running best
    series = []
    for a in arms:
        pts, b = [], st[a]["refs"]["start"]["score"]
        pts.append((0.0, b))
        for r in val[a]:
            b = max(b, float(r["score"]))
            pts.append((float(r["hours"]), b))
        series.append((f"{a} wiring", COLORS[a], pts))
    ref = st[arms[0]]["refs"]
    hl = [("walking model", "#2ca02c", ref["walking_model"]["score"]), ("untouched", "#777", ref["untouched"]["score"])]
    (charts / "score.svg").write_text(svg_lines(series, "Best checked walking score (3 fresh seeds)", "hours", "score", hl))
    series = []
    for a in arms:
        pts, b, fw = [], st[a]["refs"]["start"]["score"], st[a]["refs"]["start"]["parts"].get("forward_mm", 0.0)
        pts.append((0.0, fw))
        for r in val[a]:
            if float(r["score"]) > b:
                b, fw = float(r["score"]), float(r["forward_mm"])
            pts.append((float(r["hours"]), fw))
        series.append((f"{a} wiring", COLORS[a], pts))
    hl = [("walking model", "#2ca02c", ref["walking_model"]["parts"].get("forward_mm", 0.0))]
    (charts / "forward.svg").write_text(svg_lines(series, "Forward distance of the best setting (mm in 2 s)", "hours", "mm", hl))
    series = [(f"{a}: per-generation best", COLORS[a], [(float(r["hours"]), float(r["gen_best"])) for r in prog[a]])
              for a in arms]
    (charts / "search.svg").write_text(svg_lines(series, "Search progress (one noisy try each, not checked)", "hours", "score"))
    (charts / "legs_walking_model.svg").write_text(svg_legs(ref["walking_model"]["rec"], "Walking model on (reference)"))
    for a in arms:
        (charts / f"legs_{a}_best.svg").write_text(svg_legs(st[a]["best"].get("rec"), f"{a} wiring, best learned setting, walking model off"))

    def row(label, v):
        p = v["parts"]
        return (f"| {label} | {v['score']:.2f} | {p.get('forward_mm', 0):.2f} | {p.get('rhythm', 0):.2f} | "
                f"{p.get('tripod', 0):.2f} | {p.get('step_hz', 0):.1f} | {p.get('fell', 0):.2f} |")
    L = ["# Learned walking: results", "",
         "Can the connectome's own leg circuits make the fly step with the hand-built walking model off, if practice",
         "adjusts connection strengths between cell types, tonic drive and fast adaptation (never who connects to whom)?",
         "The control is the same search on shuffled wiring. Method: `server/tuning.py`, "
         "`tools/experiments/learned_walking.py`.", "",
         "Trial: 0.5 s standing, then P9 (DNp09) at 150 Hz for 2 s; all hand-built parts off except where it says",
         "walking model. Scores below are means over 3 fresh noise seeds.", "",
         "| | score | forward mm | rhythm | tripod | step Hz | fell |", "|---|---|---|---|---|---|---|",
         row("walking model on (reference)", ref["walking_model"]),
         row("walking model off, wiring untouched", ref["untouched"])]
    for a in arms:
        L.append(row(f"**{a} wiring, best learned** (gen {st[a]['best'].get('gen', 0)})", st[a]["best"]))
    L += ["", "| arm | settings | generations | trials | hours |", "|---|---|---|---|---|"]
    for a in arms:
        L.append(f"| {a} | {st[a]['n']} | {st[a]['gen']} | {st[a]['trials']} | {st[a]['hours']:.1f} |")
    L += ["", "![score](charts/score.svg)", "", "![forward](charts/forward.svg)", "", "![search](charts/search.svg)", "",
          "Legs (swing joint of each leg; blue and orange are the two tripods, which should alternate):", "",
          "![walking model](charts/legs_walking_model.svg)", ""]
    for a in arms:
        L += [f"![{a}](charts/legs_{a}_best.svg)", ""]
    # verdict
    r_best = st.get("real", {}).get("best", {})
    s_best = st.get("shuffled", {}).get("best", {})
    wm = ref["walking_model"]
    L += ["## What it means", ""]
    if r_best and s_best:
        rp, sp_ = r_best["parts"], s_best["parts"]
        stepping = rp.get("rhythm", 0) >= 0.3 and rp.get("tripod", 0) >= 0.6 and rp.get("forward_mm", 0) > 0.5
        beats_ctrl = r_best["score"] > s_best["score"] + 0.3
        if stepping and beats_ctrl:
            L.append("The real wiring learned to step: a rhythm in the stepping band with tripod coordination, moving "
                     "forwards, clearly better than the shuffled control. The circuits the search changed most are "
                     "listed below: they are the candidates for the nerve cord's walking rhythm in this model.")
        elif stepping:
            L.append("The real wiring learned something like stepping, but the shuffled control did about as well, so "
                     "the search found a way to make legs oscillate that doesn't depend on the connectome's structure. "
                     "Not evidence for the connectome's walking circuits.")
        elif beats_ctrl:
            L.append("The real wiring scored better than the control but did not reach real stepping (rhythm >= 0.3, "
                     "tripod >= 0.6, moving forwards). Partial result: see which parts improved.")
        else:
            L.append("Neither wiring learned to step within this run. A negative result: with these adjustable settings "
                     "the integrate-and-fire model of the nerve cord doesn't produce a stepping rhythm. A longer run or "
                     "more adjustable types (--types, --pairs) may change that; the hand-built walking model stays "
                     "necessary for now.")
        L.append("")
        L.append(f"(Real {r_best['score']:.2f} vs shuffled {s_best['score']:.2f} vs walking model {wm['score']:.2f}; "
                 f"real rhythm {rp.get('rhythm', 0):.2f}, tripod {rp.get('tripod', 0):.2f}, "
                 f"{rp.get('forward_mm', 0):.2f} mm; shuffled rhythm {sp_.get('rhythm', 0):.2f}, tripod {sp_.get('tripod', 0):.2f}.)")
    L += ["", "Score = forward (mm/3, capped at 2) + rhythm + 2 x tripod x rhythm + 0.5 x quiet before the command "
          "- fell - 5 x jump muscle - wing power. rhythm: share of each leg joint's movement at 2-12 Hz times its size "
          "relative to a real step; tripod: how well the legs keep the tripod phase at the stepping frequency (0.5 = "
          "random)."]
    for a in arms:
        d = best.get(a, {}).get("describe")
        if not d:
            continue
        L += ["", f"## Biggest changes, {a} wiring", "", "Connection gains (x the connectome's strength):", ""]
        L += [f"- {lab}: x{v}" for lab, v in d["gains"][:12]]
        L += ["", "Tonic drive (events/s): " + ", ".join(f"{k} {v}" for k, v in d["tonic"][:8]),
              "", "Fast adaptation: " + ", ".join(f"{k} {v}" for k, v in d["fast_adaptation"][:8])]
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    log(f"report: {OUT / 'report.md'}")


async def main_async(args):
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        ram_gb = 8.0
    # each copy of the fly needs a core and about 1.5 GB
    total = args.workers or max(len(arms), min(8, (os.cpu_count() or 2) - 1, int((ram_gb - 2) / 1.5)))
    per = max(1, total // len(arms))
    deadline = time.time() + args.hours * 3600
    log(f"learned walking: arms {arms}, {per} practice copies each, until {time.strftime('%H:%M', time.localtime(deadline))}")
    runs = [Arm(a, ARMS[a], per, args) for a in arms]
    try:
        res = await asyncio.gather(*(r.run(deadline) for r in runs), return_exceptions=True)
        for r, e in zip(runs, res):
            if isinstance(e, Exception):
                log(f"[{r.name}] stopped: {type(e).__name__}: {e}")
    finally:
        for r in runs:
            r.pool.stop()
    report()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--arms", default="real,shuffled")
    ap.add_argument("--workers", type=int, default=0, help="practice copies in total (default: cores - 1, at most 8)")
    ap.add_argument("--popsize", type=int, default=0, help="tries per generation (default: from the number of settings)")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--validate-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--types", type=int, default=TUNE_CFG["n_types"], help="premotor cell types to adjust")
    ap.add_argument("--pairs", type=int, default=TUNE_CFG["n_pairs"], help="connections between types to adjust")
    ap.add_argument("--quick", action="store_true", help="a few minutes: small generations, checks every 2")
    ap.add_argument("--report", action="store_true", help="only rebuild the report from saved results")
    args = ap.parse_args()
    if args.quick:
        args.hours = min(args.hours, 0.1)
        args.popsize = args.popsize or 4
        args.validate_every = 2
    args.popsize = args.popsize or None
    TUNE_CFG.update(n_types=args.types, n_pairs=args.pairs)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report:
        report()
        return
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        log("stopped; run again to continue where it left off")
        report()


if __name__ == "__main__":
    main()
