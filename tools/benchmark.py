"""
Score the simulated fly against published experiments.

    uv run tools/benchmark.py                       # the default body (physical if MuJoCo is installed)
    uv run tools/benchmark.py --body kinematic
    uv run tools/benchmark.py --rhythms off         # connectome only: no built-in walking/grooming rhythms
    uv run tools/benchmark.py --feedback 0          # physical body, leg sense organs disconnected
    uv run tools/benchmark.py --only p9_forward,mdn_backward --repeats 1

Each experiment does to the model what was done to real flies (activate a neuron type, show a looming
shadow, touch sugar to the proboscis, ...) and scores what the body does, 0 to 1, from what a tracker
on a real rig would measure: ball speed and turning, joint movement, which muscles fire, and when.
Controls check the fly doesn't do things it shouldn't (take off at a smell, walk with no input).

Each experiment runs three times (--repeats) and scores are averaged: the model is stochastic (Poisson
input, and a rhythm can start at any point of the step), so one run can be misleading. The total is the
mean score. Use it to check that a change to the model helps rather than hurts.
Results are also written to data/benchmark/ (not part of the repository).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
import os  # noqa: E402

os.environ.setdefault("NO_BROWSER", "1")

FPS = 30
DT = 1.0 / FPS


def clip01(x):
    return float(min(1.0, max(0.0, x)))


# ---------------------------------------------------------------------------- experiments
# key, what was done, what real flies do (source), stimulus, seconds recorded, scoring function
def s_rest(r):
    moving = np.mean(np.abs(r["fwd"])) + 2 * np.mean(np.abs(r["turn"]))
    bad = r["frac"]("fly") + r["frac"]("jump")
    return clip01(1 - moving / 3) * (1 - bad), f"ball {np.mean(r['fwd']):+.1f} mm/s, turning {np.mean(r['turn']):+.2f} rad/s"


def s_forward(r):
    v = r["mean_after"]("fwd", 0.4)
    return clip01(v / 5.0), f"ball {v:+.1f} mm/s (walking forwards scores 1 at 5 mm/s)"


def s_backward(r):
    v = r["mean_after"]("fwd", 0.4)
    return clip01(-v / 3.0), f"ball {v:+.1f} mm/s (walking backwards scores 1 at -3 mm/s)"


def s_turn(side):
    def f(r):
        w = r["mean_after"]("turn", 0.4)      # + = turning left
        w = w if side == "left" else -w
        return clip01(w / 0.5), f"turning {w:+.2f} rad/s towards the {side} (1 at 0.5 rad/s)"
    return f


def s_jump(window):
    def f(r):
        t = r["first"](lambda k: r["ttm"][k] > 0.5)
        if t is None:
            return 0.0, "no jump"
        lat = t - r["t_on"]
        return clip01(1 - max(0.0, lat - window[0]) / (window[1] - window[0])), f"jump muscle at {1000 * lat:.0f} ms"
    return f


def s_groom(r):
    front = np.maximum(r["ctl"]("LF.lift"), r["ctl"]("RF.lift"))
    lifted = float(np.mean(front[r["after"](0.3)] > 0.2))
    label = r["frac"]("groom") + r["frac_reflex"]("groom_antennae")
    return 0.5 * clip01(lifted / 0.5) + 0.5 * clip01(label / 0.5), f"front legs raised {100 * lifted:.0f}% of the time, grooming {100 * label:.0f}%"


def s_feed(r):
    ext = float(np.max(r["ctl"]("proboscis.rostrum")))
    return clip01(ext / 0.3), f"proboscis extension {ext:.2f} (1 at 0.3)"


def s_no_feed(r):
    ext = float(np.max(r["ctl"]("proboscis.rostrum")))
    return clip01(1 - ext / 0.3), f"proboscis extension {ext:.2f} (should stay near 0)"


def s_calm(r):
    bad = r["frac"]("fly") + r["frac"]("jump")
    ttm = float(np.max(r["ttm"]))
    return clip01(1 - bad * 3) * (1.0 if ttm < 0.3 else 0.0), f"flying/jumping {100 * bad:.0f}% of the time, jump muscle max {ttm:.2f}"


EXPERIMENTS = [
    ("rest", "No input", "A tethered fly with nothing happening stands still", [], 2.0, s_rest),
    ("p9_forward", "Activate P9 (DNp09)", "Forward walking (Bidaye et al. 2020)", [("act", "p9")], 2.5, s_forward),
    ("mdn_backward", "Activate moonwalker neurons (MDN)", "Backward walking (Bidaye et al. 2014)", [("act", "mdn")], 2.5, s_backward),
    ("dna02_left", "Activate DNa02, left", "Turning left (Rayshubskiy et al. 2020)", [("act", "dna02_l")], 2.5, s_turn("left")),
    ("dna02_right", "Activate DNa02, right", "Turning right", [("act", "dna02_r")], 2.5, s_turn("right")),
    ("gf_jump", "Activate the giant fibre (DNp01)", "Escape jump within a few ms (von Reyn et al. 2014)", [("act", "gf")], 1.0, s_jump((0.035, 0.25))),
    ("loom_left", "Looming shadow from the left", "Escape takeoff before contact (von Reyn et al. 2014)", [("loom", "left")], 1.4, s_jump((0.5, 1.0))),
    ("loom_right", "Looming shadow from the right", "Escape takeoff before contact", [("loom", "right")], 1.4, s_jump((0.5, 1.0))),
    ("adn_groom", "Activate aDN (DNg62)", "Antennal grooming with the front legs (Hampel et al. 2015)", [("act", "adn")], 2.5, s_groom),
    ("dust_groom", "Dust on the antennae", "Antennal grooming (Seeds et al. 2014)", [("dust", None)], 2.0, s_groom),
    ("sugar_feed", "Sugar on the proboscis", "Proboscis extension (Shiu et al. 2024)", [("sugar", None)], 2.0, s_feed),
    ("bitter_no_feed", "Bitter on the proboscis", "No proboscis extension", [("bitter", None)], 2.0, s_no_feed),
    ("fruit_calm", "Fruity odour", "No takeoff or jump (control)", [("preset", "fruit")], 2.0, s_calm),
    ("geosmin_calm", "Geosmin", "No takeoff or jump (control)", [("preset", "geosmin")], 2.0, s_calm),
    ("co2_calm", "Carbon dioxide", "No takeoff or jump (control)", [("preset", "co2")], 2.0, s_calm),
    ("heat_calm", "Heat", "No takeoff or jump (control)", [("preset", "heat")], 2.0, s_calm),
    ("cold_calm", "Cold", "No takeoff or jump (control)", [("preset", "cold")], 2.0, s_calm),
]


# ---------------------------------------------------------------------------- running
class Rig:
    def __init__(self, body_kind, rhythms, feedback):
        t = time.time()
        import app
        self.app = app
        self.world, self.body, self.skills = app.world, app.body, app.skills
        on = self.body.set_physics(body_kind == "physics")
        if body_kind == "physics" and not on:
            sys.exit(f"physics body unavailable: {self.body.phys_error}")
        self.body.rhythms = rhythms
        self.body.set_feedback(feedback)
        self.kind = "physics" if on else "kinematic"
        print(f"loaded in {time.time() - t:.1f} s: {self.kind} body, rhythms {'on' if rhythms else 'off'}"
              + (f", leg feedback x{feedback:g}" if on else ""), flush=True)
        self.ttm = [i for i, k in enumerate(self.world.motor.key) if k.endswith(".jump_ttm")]

    def frame(self):
        w, b, sk = self.world, self.body, self.skills
        b.sense(DT)
        sk.tick(DT)
        w.frame(DT * 1000.0)
        sk.observe(DT)
        b.act(DT, w.motor, w.fast_readouts(), skill=self.app.skill_brief())
        b.physics_step(DT)

    def reset(self):
        w, b = self.world, self.body
        self.skills.stop()
        if self.skills.state:
            self.skills.state = None
            w.channel_stim.pop("skill", None)
        b.start("stop")
        b.reset()
        w.apply_command({"cmd": "clear"})
        w.channel_stim.clear()
        w.apply_command({"cmd": "reset"})
        w.pending.clear()
        w.motor.reset()
        w.ro_fast[:] = 0
        for _ in range(int(0.6 * FPS)):     # settle
            self.frame()

    def run(self, stim, secs):
        self.reset()
        w, b = self.world, self.body
        rec = {"t": [], "mode": [], "fwd": [], "turn": [], "ttm": [], "air": [], "reflex": [], "val": []}
        for kind, arg in stim:
            if kind == "preset":
                w.pending.append({"cmd": "preset", "key": arg, "rate": w.stim[arg]["rate"]})
            else:
                b.start(arg if kind == "act" else kind, arg if kind == "loom" else None)
        t0 = time.time()
        for f in range(int(secs * FPS)):
            self.frame()
            rec["t"].append((f + 1) * DT)
            rec["mode"].append(b.mode)
            rec["fwd"].append(b.ball[0])
            rec["turn"].append(-b.ball[1])            # + = turning left
            rec["ttm"].append(float(w.motor.act[self.ttm].max()) if self.ttm else 0.0)
            rec["air"].append(b.airborne)
            st = self.skills.state
            rec["reflex"].append(st["name"] if st and st.get("kind") == "reflex" else "")
            rec["val"].append(dict(w.motor.value))
        rec["wall"] = time.time() - t0
        return rec


def analyse(rec):
    t = np.array(rec["t"])
    r = {k: np.array(v) for k, v in rec.items() if k in ("fwd", "turn", "ttm")}
    r["t_on"] = 0.0
    r["after"] = lambda s: t >= s
    r["mean_after"] = lambda k, s: float(np.mean(r[k][t >= s])) if np.any(t >= s) else 0.0
    r["frac"] = lambda m: float(np.mean([x == m for x in rec["mode"]]))
    r["frac_reflex"] = lambda n: float(np.mean([x == n for x in rec["reflex"]]))
    r["ctl"] = lambda k: np.array([v.get(k, 0.0) for v in rec["val"]])
    r["first"] = lambda cond: next((float(t[i]) for i in range(len(t)) if cond(i)), None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", choices=["physics", "kinematic"], default="physics")
    ap.add_argument("--rhythms", choices=["on", "off"], default="on")
    ap.add_argument("--feedback", type=float, default=1.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--repeats", type=int, default=3, help="runs per experiment (the model is noisy; scores are averaged)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rig = Rig(a.body, a.rhythms == "on", a.feedback)
    only = set(filter(None, a.only.split(",")))
    rows = []
    for key, what, expect, stim, secs, score in EXPERIMENTS:
        if only and key not in only:
            continue
        runs, wall = [], 0.0
        for _ in range(max(1, a.repeats)):
            rec = rig.run(stim, secs)
            runs.append(score(analyse(rec)))
            wall += rec["wall"]
        scores = [r[0] for r in runs]
        sc = float(np.mean(scores))
        detail = runs[int(np.argsort(scores)[len(scores) // 2])][1]     # the median run's measurement
        rows.append({"key": key, "what": what, "expect": expect, "score": round(sc, 2), "runs": [round(x, 2) for x in scores],
                     "measured": detail, "speed": round(secs * len(runs) / max(wall, 1e-6), 2)})
        spread = f" (runs {', '.join(f'{x:.2f}' for x in scores)})" if len(scores) > 1 else ""
        print(f"  {sc:4.2f}  {key:15s} {detail}{spread}   [{rows[-1]['speed']}x real time]", flush=True)
    total = float(np.mean([r["score"] for r in rows])) if rows else 0.0
    cfg = {"body": rig.kind, "rhythms": a.rhythms, "feedback": a.feedback if rig.kind == "physics" else None}
    print(f"\nTOTAL {total:.2f} over {len(rows)} experiments  ({cfg})")
    out = Path(a.out) if a.out else ROOT / "data" / "benchmark" / f"{rig.kind}-rhythms_{a.rhythms}-fb_{a.feedback:g}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": cfg, "total": round(total, 3), "experiments": rows}, indent=1))
    print("written", out.relative_to(ROOT) if out.is_relative_to(ROOT) else out)


if __name__ == "__main__":
    main()
