"""
Score the simulated fly against published experiments.

    uv run tools/benchmark.py                       # the default body (physical if MuJoCo is installed)
    uv run tools/benchmark.py --body kinematic
    uv run tools/benchmark.py --rhythm off          # connectome only: no rhythm generators at all
    uv run tools/benchmark.py --feedback 0          # physical body, leg sense organs disconnected
    uv run tools/benchmark.py --only p9_forward,mdn_backward --repeats 1
    uv run tools/benchmark.py --variant shuffled    # control: the same neurons, randomly rewired
    uv run tools/benchmark.py --ablate giant_fibre  # silence neurons (groups or cell types) and see what's lost

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
    moved = _dist(r, len(r["xy"]) - 1, r["xy"][0]) if r["xy"] else 0.0
    turned = abs(float(np.sum(r["turn"]))) / 30.0
    bad = r["frac"]("fly") + r["frac"]("jump") + r["frac"]("walk")
    return clip01(1 - moved / 2 - turned / 0.5) * clip01(1 - bad), f"moved {moved:.1f} mm, turned {np.degrees(turned):.0f} deg"


def s_forward(r):
    v = r["mean_after"]("fwd", 0.4)
    return clip01(v / 5.0), f"{v:+.1f} mm/s (walking forwards scores 1 at 5 mm/s)"


def s_backward(r):
    v = r["mean_after"]("fwd", 0.4)
    return clip01(-v / 3.0), f"{v:+.1f} mm/s (walking backwards scores 1 at -3 mm/s)"


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


def _dist(r, i, src):
    x, y = r["xy"][i]
    return float(np.hypot(x - src[0], y - src[1]))


def s_approach(r):
    src = r["source"]
    d0, d1 = _dist(r, 0, src), min(_dist(r, i, src) for i in range(len(r["xy"])))
    return clip01((d0 - d1) / 3.0), f"closest approach {d0 - d1:+.1f} mm towards the source (1 at 3 mm)"


def s_avoid(r):
    """How far the fly moved directly away from the source (walking past it doesn't count)."""
    src = r["source"]
    (x0, y0), (x1, y1) = r["xy"][0], r["xy"][-1]
    ux, uy = x0 - src[0], y0 - src[1]
    n = max(float(np.hypot(ux, uy)), 1e-6)
    away = ((x1 - x0) * ux + (y1 - y0) * uy) / n
    return clip01(away / 3.0), f"moved {away:+.1f} mm directly away from the source (1 at 3 mm)"


def s_leave_zone(r):
    src = r["source"]
    inside = [(_dist(r, i, src) < 5.0) for i in range(len(r["xy"]))]
    frac = float(np.mean(inside[len(inside) // 2:]))
    return clip01(1 - frac), f"in the warm zone {100 * frac:.0f}% of the second half"


def s_follow(side):
    def f(r):
        h = np.unwrap(np.array(r["heading"]))
        dh = float(h[-1] - h[0]) * (1 if side == "left" else -1)
        return clip01(dh / np.radians(30)), f"turned {np.degrees(dh):+.0f} deg towards the target (1 at 30 deg)"
    return f


def s_optomotor(r):
    w = r["mean_after"]("turn", 0.4)
    return clip01(w / 0.3), f"turning {w:+.2f} rad/s with the stripes (1 at 0.3 rad/s)"


# learning: procedures of their own (they return what the score needs)
def _present(rig, odour, secs, reinforce=None, at=0.4):
    """Odour at the fly's antennae for secs; with reinforce ("reward"/"punish"), those dopamine
    neurons are activated from `at` s for 1 s."""
    b = rig.body
    f = b.fly()
    b.arena.add(odour, f, x=f["xy"][0], y=f["xy"][1])
    for i in range(int(secs * FPS)):
        if reinforce and i == int(at * FPS):
            b.start(reinforce)
        rig.frame()
    b.arena.objects = [o for o in b.arena.objects if o.get("odour") != odour]


def _idle(rig, secs):
    for _ in range(int(secs * FPS)):
        rig.frame()


def p_conditioning(us, plus="fruit", minus="co2", trials=3):
    """Smell both odours once (their Kenyon cell codes), pair `plus` with dopamine `trials` times,
    then compare how much each odour's input to the MBONs that dopamine gates has weakened."""
    def run(rig):
        t0 = time.time()
        mb = rig.world.mb
        rig.reset()
        mb.seen = {}
        for od in (plus, minus):
            _present(rig, od, 1.0)
            _idle(rig, 1.0)
        for _ in range(trials):
            _present(rig, plus, 1.5, reinforce=us)
            _idle(rig, 1.0)
        g = mb.gated_by(us)
        drop = {}
        for od in (plus, minus):
            code = mb.seen.get(od, np.zeros(len(mb.kc), np.float32))
            then = mb.drive(code, learned=False)[g].sum()
            drop[od] = float(1 - mb.drive(code)[g].sum() / then) if then > 0 else 0.0
        kcs = {od: int((mb.seen.get(od, np.zeros(1)) > 0).sum()) for od in (plus, minus)}
        mb.forget()
        return {"plus": drop[plus], "minus": drop[minus], "kcs": kcs, "names": (plus, minus), "wall": time.time() - t0}
    return run


def s_conditioning(r):
    spec = r["plus"] - r["minus"]
    (p, m), k = r["names"], r["kcs"]
    return clip01(spec / 0.3), (f"{p} ({k[p]} KCs) input to those MBONs {-100 * r['plus']:+.0f}%, "
                                f"{m} ({k[m]} KCs, unpaired) {-100 * r['minus']:+.0f}% (1 at 30 points apart)")


def p_learned_avoidance(trials=3):
    """Fruit approach, then fruit paired with punishment, then fruit approach again."""
    stim = [("world", ("fruit", 6.0, 5.0, None))]          # ahead-left: the odour sets the fly walking towards it

    def run(rig):
        t0 = time.time()
        a = analyse(rig.run(stim, 4.0))
        before = _approach(a)
        rig.reset()
        for _ in range(trials):
            _present(rig, "fruit", 1.5, reinforce="punish")
            _idle(rig, 1.0)
        a = analyse(rig.run(stim, 4.0, forget=False))
        after = _approach(a)
        rig.world.mb.forget()
        return {"before": before, "after": after, "wall": time.time() - t0}
    return run


def _approach(r):
    src = r["source"]
    return _dist(r, 0, src) - min(_dist(r, i, src) for i in range(len(r["xy"])))


def s_learned_avoidance(r):
    d = r["before"] - r["after"]
    return clip01(d / 2.0), f"approach {r['before']:+.1f} mm before training, {r['after']:+.1f} mm after (1 at 2 mm less)"


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
    ("sugar_mouth", "Sugar touched to the proboscis", "Proboscis extension (Shiu et al. 2024)", [("act", "probe_sugar")], 2.0, s_feed),
    ("bitter_mouth", "Bitter touched to the proboscis", "No proboscis extension", [("act", "probe_bitter")], 2.0, s_no_feed),
    ("sugar_feed", "Sugar under the front feet", "Proboscis extension (Dethier 1976; Shiu et al. 2024)", [("world", ("sugar", 1.3, 0.0, 1.0))], 2.0, s_feed),
    ("bitter_no_feed", "Bitter under the front feet", "No proboscis extension", [("world", ("bitter", 1.3, 0.0, 1.0))], 2.0, s_no_feed),
    ("fruit_approach", "Fruity odour 6 mm to the left", "Turns and walks towards it (odour attraction, osmotropotaxis)",
     [("world", ("fruit", 0.0, 6.0, None))], 4.0, s_approach),
    ("fruit_approach_right", "Fruity odour 6 mm to the right", "Turns and walks towards it (the same, mirrored)",
     [("world", ("fruit", 0.0, -6.0, None))], 4.0, s_approach),
    ("co2_avoid", "CO2 source 4 mm ahead", "Moves away (CO2 avoidance, Suh et al. 2004)", [("world", ("co2", 4.0, 0.0, None))], 4.0, s_avoid),
    ("heat_avoid", "Warm zone around the fly", "Leaves the warm air (Simoes et al. 2021)", [("world", ("heat", 0.5, 0.0, None))], 4.0, s_leave_zone),
    ("optomotor", "Stripes turning left", "Turns with them (optomotor response)", [("stripes", "left")], 3.0, s_optomotor),
    ("learn_punish", "Fruit odour paired 3x with PPL1 dopamine neurons; CO2 unpaired",
     "The paired odour's KC -> MBON synapses weaken, the unpaired one's don't (Hige et al. 2015)",
     p_conditioning("punish"), 0, s_conditioning),
    ("learn_reward", "Fruit odour paired 3x with PAM dopamine neurons; CO2 unpaired",
     "The same in reward compartments (Cohn et al. 2015; Owald et al. 2015)", p_conditioning("reward"), 0, s_conditioning),
    ("learned_avoidance", "Fruit approach, before and after 3 fruit + punishment pairings",
     "Approaches the odour less (aversive conditioning, Tully & Quinn 1985)", p_learned_avoidance(), 0, s_learned_avoidance),
    ("follow_left", "Small target darting ahead-left", "Turns towards it (object tracking; LC10a, Ribeiro et al. 2018)",
     [("target", "left")], 3.0, s_follow("left")),
    ("follow_right", "Small target darting ahead-right", "Turns towards it", [("target", "right")], 3.0, s_follow("right")),
    ("fruit_calm", "Fruity odour", "No takeoff or jump (control)", [("preset", "fruit")], 2.0, s_calm),
    ("geosmin_calm", "Geosmin", "No takeoff or jump (control)", [("preset", "geosmin")], 2.0, s_calm),
    ("co2_calm", "Carbon dioxide", "No takeoff or jump (control)", [("preset", "co2")], 2.0, s_calm),
    ("heat_calm", "Heat", "No takeoff or jump (control)", [("preset", "heat")], 2.0, s_calm),
    ("cold_calm", "Cold", "No takeoff or jump (control)", [("preset", "cold")], 2.0, s_calm),
]


# controls: the fly should NOT do something (the rest are behaviours it should produce)
CONTROLS = {"rest", "bitter_mouth", "bitter_no_feed", "fruit_calm", "geosmin_calm", "co2_calm", "heat_calm", "cold_calm"}


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
        self.body.set_rhythm(rhythms if isinstance(rhythms, str) else ("neurons" if rhythms else "off"))
        self.body.set_feedback(feedback)
        self.kind = "physics" if on else "kinematic"
        print(f"loaded in {time.time() - t:.1f} s: {self.kind} body, rhythm: {self.body.rhythm}"
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

    def reset(self, forget=True):
        w, b = self.world, self.body
        if forget:
            w.mb.forget()
        self.skills.stop()
        if self.skills.state:
            self.skills.state = None
            w.channel_stim.pop("skill", None)
        b.start("stop")
        b.start("clear")
        b.reset(home=True)
        w.apply_command({"cmd": "clear"})
        w.channel_stim.clear()
        w.apply_command({"cmd": "reset"})
        w.pending.clear()
        w.motor.reset()
        w.ro_fast[:] = 0
        for _ in range(int(0.6 * FPS)):     # settle
            self.frame()

    def run(self, stim, secs, forget=True):
        self.reset(forget)
        w, b = self.world, self.body
        rec = {"t": [], "mode": [], "fwd": [], "turn": [], "ttm": [], "air": [], "reflex": [], "val": []}
        rec["xy"] = []
        rec["source"] = None
        for kind, arg in stim:
            if kind == "preset":
                w.pending.append({"cmd": "preset", "key": arg, "rate": w.stim[arg]["rate"]})
            elif kind == "world":
                what, ahead, left, radius = arg
                f = b.fly()
                (x0, y0), h = f["xy"], f["heading"]
                x, y = x0 + ahead * np.cos(h) - left * np.sin(h), y0 + ahead * np.sin(h) + left * np.cos(h)
                o = b.arena.add(what, f, x=x, y=y)
                if radius and o is not None:
                    o["r"] = radius
                rec["source"] = (x, y)
            elif kind in ("stripes", "target"):
                b.start(kind, arg)
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
            f = b.fly()
            rec["xy"].append(tuple(f["xy"]))
            rec.setdefault("heading", []).append(f["heading"])
        rec["wall"] = time.time() - t0
        return rec


def analyse(rec):
    t = np.array(rec["t"])
    r = {k: np.array(v) for k, v in rec.items() if k in ("fwd", "turn", "ttm")}
    r["xy"], r["source"], r["heading"] = rec["xy"], rec["source"], rec.get("heading", [0.0])
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
    ap.add_argument("--rhythm", choices=["neurons", "off"], default="neurons",
                    help="walking commanded by the descending neurons (locomotion.py), or none: the connectome alone")
    ap.add_argument("--feedback", type=float, default=1.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--repeats", type=int, default=3, help="runs per experiment (the model is noisy; scores are averaged)")
    ap.add_argument("--variant", choices=["real", "shuffled", "no_inhibition"], default="real",
                    help="run on the connectome, a degree-preserving shuffle of it, or it without inhibition")
    ap.add_argument("--seed", type=int, default=0, help="for the shuffle")
    ap.add_argument("--ablate", default="", help="silence neuron groups, e.g. vision,giant_fibre (or cell types)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rig = Rig(a.body, a.rhythm, a.feedback)
    if a.variant != "real":
        rig.world.set_variant(a.variant, a.seed)
        print(f"wiring: {a.variant} ({rig.world.VARIANTS[a.variant]})", flush=True)
    ablate = [x for x in a.ablate.split(",") if x]
    for g in ablate:
        n = rig.world.ablate(g)
        if not n:
            sys.exit(f"nothing to ablate called {g}; groups: {', '.join(rig.world.ABLATIONS)}, or a cell type")
        print(f"ablated {g}: {n} neurons silenced", flush=True)
    only = set(filter(None, a.only.split(",")))
    rows = []
    for key, what, expect, stim, secs, score in EXPERIMENTS:
        if only and key not in only:
            continue
        runs, wall = [], 0.0
        for _ in range(max(1, a.repeats)):
            rec = stim(rig) if callable(stim) else rig.run(stim, secs)
            runs.append(score(rec if callable(stim) else analyse(rec)))
            wall += rec["wall"]
        scores = [r[0] for r in runs]
        sc = float(np.mean(scores))
        detail = runs[int(np.argsort(scores)[len(scores) // 2])][1]     # the median run's measurement
        rows.append({"key": key, "what": what, "expect": expect, "score": round(sc, 2), "runs": [round(x, 2) for x in scores],
                     "measured": detail, "speed": round(secs * len(runs) / max(wall, 1e-6), 2) if secs else None})
        spread = f" (runs {', '.join(f'{x:.2f}' for x in scores)})" if len(scores) > 1 else ""
        speed = f"   [{rows[-1]['speed']}x real time]" if secs else ""
        print(f"  {sc:4.2f}  {key:17s} {detail}{spread}{speed}", flush=True)
    total = float(np.mean([r["score"] for r in rows])) if rows else 0.0
    beh = [r["score"] for r in rows if r["key"] not in CONTROLS]
    ctl = [r["score"] for r in rows if r["key"] in CONTROLS]
    behaviours = float(np.mean(beh)) if beh else None
    controls = float(np.mean(ctl)) if ctl else None
    cfg = {"body": rig.kind, "rhythm": a.rhythm, "feedback": a.feedback if rig.kind == "physics" else None,
           "variant": a.variant, "ablate": ablate}
    print(f"\nTOTAL {total:.2f} over {len(rows)} experiments  ({cfg})")
    if beh and ctl:
        print(f"  behaviours {behaviours:.2f} ({len(beh)}), controls {controls:.2f} ({len(ctl)})")
    name = f"{rig.kind}-rhythm_{a.rhythm}-fb_{a.feedback:g}" + (f"-{a.variant}" if a.variant != "real" else "") \
        + "".join(f"-no_{g}" for g in ablate)
    out = Path(a.out) if a.out else ROOT / "data" / "benchmark" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": cfg, "total": round(total, 3),
                               "behaviours": None if behaviours is None else round(behaviours, 3),
                               "controls": None if controls is None else round(controls, 3), "experiments": rows}, indent=1))
    print("written", out.relative_to(ROOT) if out.is_relative_to(ROOT) else out)


if __name__ == "__main__":
    main()
