"""
Walking and turning plans made from a real fly's recorded steps (NeuroMechFly v2, single-step
kinematics of an untethered fly; flygym, Apache-2.0).

The recorded joint angles of each leg over one step are turned back into the motor controls that
would produce them (the inverse of motor.py's control -> joint mapping), and the six legs are put in a
tripod: front and hind legs of one side move with the middle leg of the other. A plan is then a set of
keyframes the skill engine learns to produce by driving neurons (tools/learn_builtin.py).

Only the four controls that move the foot along the ball are used per leg (checked on the physical
body): swing (thorax-coxa), lift (coxa-trochanter), reach (femur-tibia) and twist (femur rotation).

  walk_forward   the recorded step, tripod gait
  walk_backward  the same step played in reverse
  turn_left/right  the outer legs step normally, the inner legs step in reverse with a shorter stride,
                 which spins the ball under the fly (a turn on the spot)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
import motor as M  # noqa: E402

CONTROLS = ("swing", "lift", "reach", "twist", "grip")
DOF_OF = {terms[0][0]: dof for dof, terms in M.LEG_JOINTS.items()}
CYCLE = 0.5            # s per step: a slow walk (flies step at ~2-10 Hz), what the drive can follow
CYCLES = 2             # per plan (a reflex replays the plan back to back)
KEY_DT = 0.04          # s between keyframes
TRIPOD = {"LF": 0.0, "RM": 0.0, "LH": 0.0, "RF": 0.5, "LM": 0.5, "RH": 0.5}
INNER = 0.5            # inner legs' stride when turning (reversed)


def _inverse(offset, r):
    a, half, span = abs(offset), r["half"], r["span"]
    v = M.NATURAL * a / half if a <= half else M.NATURAL + (a - half) / max(span - half, 1e-6) * (1 - M.NATURAL)
    return min(v, 1.0) if offset >= 0 else -min(v, 1.0)


def recorded_controls():
    """{leg: {control: array over one step}} from the recorded joint angles."""
    body = json.loads(M.BODY_FILE.read_text())["steps"]["legs"]
    ranges = M._leg_ranges()
    out = {}
    for leg in M.LEGS:
        out[leg] = {}
        for c in CONTROLS:
            dof = DOF_OF[c]
            direction = dict(M.LEG_JOINTS[dof])[c]
            a = np.asarray(body[leg][dof], float)
            off = a - (a.min() + a.max()) / 2
            if dof in M.MIRRORED and leg[0] == "R":
                off = -off
            v = np.array([_inverse(direction * o, ranges[leg][dof]) for o in off])
            pos, neg = M.LEG_CONTROLS[c]
            if not pos:
                v = np.minimum(v, 0.0)
            if not neg:
                v = np.maximum(v, 0.0)
            out[leg][c] = v
    return out


BACKWARD = "reverse"     # how the backward step is made: "reverse", "sweep" (reverse only the sweeping
                         # controls) or "shift" (lift half a cycle out of phase with the sweep)
LIFTING = ("lift", "grip")
BACK_GRIP = 0.0          # constant tarsus posture added when walking backwards


def plan(kind):
    rec = recorded_controls()
    n = len(next(iter(rec["LF"].values())))
    duration = CYCLE * CYCLES
    frames = []
    for t in np.arange(0.0, duration + 1e-9, KEY_DT):
        kf = {"t": round(float(t), 3)}
        for leg in M.LEGS:
            ph = (t / CYCLE + TRIPOD[leg]) % 1.0
            scale, rev = 1.0, False
            if kind == "walk_backward":
                rev = True
            elif kind in ("turn_left", "turn_right"):
                inner = "L" if kind == "turn_left" else "R"
                if leg[0] == inner:
                    rev, scale = True, INNER
            for c in CONTROLS:
                p = ph
                mode = BACKWARD if kind == "walk_backward" else "reverse"
                if rev and (mode == "reverse" or (mode == "sweep" and c not in LIFTING)):
                    p = 1.0 - ph
                if rev and mode == "shift" and c in LIFTING:
                    p = (ph + 0.5) % 1.0
                x = p * n
                i0 = int(x) % n
                i1 = (i0 + 1) % n
                f = x - int(x)
                v = rec[leg][c][i0] * (1 - f) + rec[leg][c][i1] * f
                kf[f"{leg}.{c}"] = round(float(v * scale), 3)
            if kind == "walk_backward" and BACK_GRIP:
                kf[f"{leg}.grip"] = round(float(np.clip(kf.get(f"{leg}.grip", 0.0) + BACK_GRIP, -1, 1)), 3)
        frames.append(kf)
    return {"duration": duration, "keyframes": frames}


DESCRIPTIONS = {
    "walk_forward": "Walk forwards: a real fly's recorded step in a tripod gait, two steps per second",
    "walk_backward": "Walk backwards: the recorded step played in reverse, tripod gait",
    "turn_left": "Turn left on the spot: right legs step forwards, left legs step backwards with a shorter stride",
    "turn_right": "Turn right on the spot: left legs step forwards, right legs step backwards with a shorter stride",
}

if __name__ == "__main__":
    for k in DESCRIPTIONS:
        p = plan(k)
        print(k, p["duration"], "s,", len(p["keyframes"]), "keyframes,", len(p["keyframes"][0]) - 1, "controls")
