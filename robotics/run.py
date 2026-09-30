"""
Run the failure finder overnight and write the report.

    python -m robotics.run --hours 8                  # both robots, half the time each
    python -m robotics.run --robots go2 --quick       # a few minutes, to check it runs
    python -m robotics.run --report                   # rebuild the report from saved results

Results: results/failure_finder/report.md, and per robot summary.json, failures.json (each failure is
replayable: python -m robotics.replay go2 3), samples.csv (every run).
"""
import argparse
import csv
import json
import time

import numpy as np

from .finder import OUT, Finder, default_workers, make_pool
from .svg import bars, scatter


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


ROBOT_TEXT = {
    "go2": ("Unitree Go2 quadruped, trotting forwards for 5 s",
            "a fixed trot tracked by joint PD control (doesn't look at the body)",
            "the same trot + foot placement towards the drift + softening when the joints shake"),
    "panda": ("Franka Panda arm, picking a box off the floor and putting it down 30 cm away",
              "a scripted pick-and-place from one look at the box, never checking anything",
              "the same + grasp check with a closer look and retry, drop check, place check"),
}


def fmt_params(space, p, only=None):
    out = []
    for d in space:
        if only is not None and d["name"] not in only:
            continue
        if abs(p[d["name"]] - d["nominal"]) > 0.05 * (d["hi"] - d["lo"]):
            out.append(f"{d['name']} {p[d['name']]:.3g} (normal {d['nominal']:g})")
    return ", ".join(out) or "normal conditions"


def report():
    robots = [r for r in ("go2", "panda") if (OUT / r / "summary.json").exists()]
    if not robots:
        print("no results yet")
        return
    charts = OUT / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    L = ["# Failure finder: results", "",
         "Each robot's controller was run under conditions searched for failure: each condition alone, random variation",
         "of all of them around normal, then adversarial search for the worst failures and for the mildest conditions that still fail. Each",
         "distinct failure was re-run on fresh seeds (is it reproducible?), with each changed condition put back to",
         "normal in turn (which conditions cause it), and with the recovery controller (does it fix it, and does it",
         "break anything that worked?). Method: `robotics/finder.py`.", ""]
    for r in robots:
        S = json.loads((OUT / r / "summary.json").read_text())
        reps = json.loads((OUT / r / "failures.json").read_text())
        space = S["space"]
        title, base_txt, rec_txt = ROBOT_TEXT[r]
        ro = S.get("recovery_on_explored", {})
        dist = S.get("distil", {})
        fixed = [x for x in reps if x.get("fixed_rate", 0) >= 0.6]
        L += [f"## {title}", "",
              f"- Baseline controller: {base_txt}.", f"- Recovery controller: {rec_txt}.",
              f"- {S['runs']} runs in {S['minutes']:.0f} min"
              + (f"; {S['unstable_runs']} runs where the simulator itself became unstable were left out." if S["unstable_runs"] else "."),
              f"- Normal conditions: margin {np.mean(S['nominal']['baseline']):.2f} (baseline), "
              f"{np.mean(S['nominal']['recovery']):.2f} (recovery); above 0 is success.",
              f"- Random variation of every condition around normal (sd a fifth of each range): "
              f"**{100 * (S['explore']['fail_rate'] or 0):.0f}%** fail ({S['explore']['runs']} runs).",
              f"- Failures checked: {dist.get('checked', 0)} distinct ones, {dist.get('flaky', 0)} not reproducible on fresh "
              f"seeds; the reproducible ones come down to **{len(reps)} distinct cause{'' if len(reps) == 1 else 's'}**; the recovery controller fixes "
              f"**{len(fixed)}** of them.",
              ]
        if ro.get("runs"):
            L.append(f"- Recovery controller on {ro['runs']} of the random conditions: failures "
                     f"{100 * ro['baseline_fail_rate']:.0f}% -> {100 * ro['recovery_fail_rate']:.0f}%; it fixed {ro['fixed']} "
                     f"and **broke {ro['broken']}** that worked before.")
        # one condition at a time
        lim = S.get("limits", {})
        if lim:
            L += ["", "Each condition alone (everything else normal): where it starts to fail.", "",
                  "| condition | normal | range tested | fails from |", "|---|---|---|---|"]
            for d in space:
                o = lim.get(d["name"], {})
                f = [f"{'<=' if sd == 'down' else '>='} {o[sd]['fails_from']:.3g}" for sd in ("down", "up")
                     if sd in o and o[sd]["fails_from"] is not None]
                L.append(f"| {d['name']} ({d['text']}) | {d['nominal']:g} | {d['lo']:g} to {d['hi']:g} | "
                         f"{' and '.join(f) or 'never'} |")
        items = [(", ".join(x["causes"]), x["found"]) for x in reps]
        if items:
            (charts / f"{r}_causes.svg").write_text(bars(items[:15], f"{r}: failures found, by cause"))
            L += ["", f"![causes](charts/{r}_causes.svg)"]
        sens = S.get("sensitivity", {})
        if sens:
            items = sorted(((k, v["fail_rate_by_fifth"]) for k, v in sens.items()), key=lambda kv: -max(x or 0 for x in kv[1]))
            top = [k for k, _ in items[:2]]
            ex = []
            with (OUT / r / "samples.csv").open() as f:
                for row in csv.DictReader(f):
                    if row["phase"] == "explore" and row["unstable"] == "0" and row["margin"] not in ("", "None"):
                        ex.append((float(row[top[0]]), float(row[top[1]]), float(row["margin"]) < 0))
            if ex:
                (charts / f"{r}_explore.svg").write_text(scatter(ex, top[0], top[1], f"{r}: random variations (red = failed)"))
                L += ["", f"![explore](charts/{r}_explore.svg)"]
        L += ["", "Distinct failures, by cause (the mildest example of each; 'found' = how many of the searched failures "
              "came down to it):", "",
              "| # | causes | example (only what differs from normal) | why | found | fails / 5 | recovery fixes | recovery did |",
              "|---|---|---|---|---|---|---|---|"]
        for i, x in enumerate(reps):
            L.append(f"| {i} | {', '.join(x['causes']) or 'several together'} | {fmt_params(space, x['params'])} | "
                     f"{x['why']} | {x['found']} | {round(5 * x['fail_rate'])} | {round(5 * x.get('fixed_rate', 0))}/5 | "
                     f"{'; '.join(x.get('recovery_events', [])) or '-'} |")
        if ro.get("broken_examples"):
            L += ["", "Conditions where the recovery controller fails but the baseline didn't:", ""]
            for b in ro["broken_examples"]:
                L.append(f"- {fmt_params(space, b['params'])}: {b['why']} (margin {b['baseline']:.2f} -> {b['recovery']:.2f})")
        L += ["", f"Replay any of them in the MuJoCo viewer: `tools/run_robotics.sh replay {r} <#>` (add `--recovery`).", ""]
    L += ["## How to read it", "",
          "The margin is how far a run is from the edge of success (Go2: body height, uprightness and distance; Panda: "
          "distance from the target and whether the box stands). 'Causes' are the conditions that, put back to normal "
          "one at a time, make the failure go away; 'only together' means no single one does. Failures found by the "
          "mildest-change search are the likeliest in practice; those from the worst-case search show how badly it can go.",
          ""]
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    log(f"report: {OUT / 'report.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--robots", default="go2,panda")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    robots = [r.strip() for r in a.robots.split(",") if r.strip()]
    hours = min(a.hours, 0.08 * len(robots)) if a.quick else a.hours
    n = a.workers or default_workers()
    log(f"failure finder: {robots}, {n} parallel runs, {hours:.2f} h")
    pool = make_pool(n)
    try:
        for r in robots:
            Finder(r, pool, hours / len(robots), seed=a.seed, log=log).run()
            report()
    finally:
        pool.terminate()
    report()


if __name__ == "__main__":
    main()
