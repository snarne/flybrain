"""
(Re)learn the built-in movements: find the neurons to drive for each, on the live connectome, and
store them in server/data/skills_builtin.json (shipped with the app, so nobody has to wait for this).

    uv run tools/learn_builtin.py                     # walk_forward, walk_backward, turn_left, turn_right
    uv run tools/learn_builtin.py groom_antennae      # or name the ones to redo

The gaits come from a real fly's recorded steps (tools/gaits.py). They are the stand-in for the nerve
cord's own rhythm generators (central pattern generators), which the model can't produce yet, so they
are learned at the level those circuits act on: the leg motor neurons. (Driving from descending neurons
spreads to too many other muscles for 30 controls at once.) Learning runs on the default body (physical,
with the legs' sense organs feeding back, as the app uses it) and takes a few minutes. Other built-ins
are relearned from the plan already stored for them, at any level.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("NO_BROWSER", "1")

import gaits  # noqa: E402

ATTEMPTS = 12
DT = 1.0 / 30


async def main(names):
    import app
    w, b, sk = app.world, app.body, app.skills
    print(f"body: {'physical' if b.phys else 'kinematic'}", flush=True)
    b.rhythms = False            # no reflexes interrupting practice
    stop = False

    async def frames():
        while not stop:
            b.sense(DT)
            sk.tick(DT)
            w.frame(DT * 1000)
            sk.observe(DT)
            b.act(DT, w.motor, w.fast_readouts(), skill=app.skill_brief())
            b.physics_step(DT)
            await asyncio.sleep(0)

    task = asyncio.create_task(frames())
    builtin_path = ROOT / "server" / "data" / "skills_builtin.json"
    builtin = json.loads(builtin_path.read_text())
    for name in names:
        if name in gaits.DESCRIPTIONS:
            desc, plan = gaits.DESCRIPTIONS[name], gaits.plan(name)
        else:
            desc, plan = builtin[name]["description"], builtin[name]["plan"]
        t = time.time()
        sk.lib.pop(name, None)
        levels = ("motor",) if name in gaits.DESCRIPTIONS else None
        res = await sk.learn(name, desc, plan, attempts=ATTEMPTS, levels=levels)
        if not res.get("ok"):
            print(name, "failed:", res.get("error"), flush=True)
            continue
        print(f"{name}: score {res['score']} after {res['attempts']} ({time.time() - t:.0f} s)", flush=True)
        skill = sk.lib.pop(name)
        skill["uses"] = 0
        skill["learned"] = "built in"
        builtin[name] = skill
        builtin_path.write_text(json.dumps(builtin, indent=1))
        sk.builtin = builtin
        sk._save()
    stop = True
    await task


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or list(gaits.DESCRIPTIONS)))
