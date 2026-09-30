"""
Watch a found failure in the MuJoCo viewer.

    python -m robotics.replay go2 3                 # failure #3 from the report, baseline controller
    python -m robotics.replay panda 0 --recovery    # the same conditions with the recovery controller
    python -m robotics.replay go2 nominal

On macOS the viewer needs `mjpython` instead of `python` (tools/run_robotics.sh handles that).
"""
import argparse
import json
import time

import mujoco
import mujoco.viewer

from .finder import OUT, get_task
from .space import nominal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("robot", choices=["go2", "panda"])
    ap.add_argument("which", help="failure number from the report, or 'nominal'")
    ap.add_argument("--recovery", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    a = ap.parse_args()
    task = get_task(a.robot)
    if a.which == "nominal":
        p = nominal(task.space)
    else:
        p = json.loads((OUT / a.robot / "failures.json").read_text())[int(a.which)]["params"]
    print("conditions:", {k: round(v, 3) for k, v in p.items()})
    state = {"v": None, "t0": None}

    def view(m, d):
        if state["v"] is None:
            state["v"] = mujoco.viewer.launch_passive(m, d)
            state["t0"] = time.time()
        v = state["v"]
        if not v.is_running():
            raise SystemExit
        if int(round(d.time / m.opt.timestep)) % max(1, int(1 / 60 / m.opt.timestep)) == 0:
            v.sync()
            lag = d.time / a.speed - (time.time() - state["t0"])
            if lag > 0:
                time.sleep(lag)

    margin, info = task.run(p, seed=a.seed, recovery=a.recovery, viewer=view)
    print(f"margin {margin:.2f} ({'success' if margin >= 0 else 'failure'}):", info)
    if state["v"] is not None:
        while state["v"].is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
