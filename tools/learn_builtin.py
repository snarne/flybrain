"""
(Re)learn the built-in movements that are learned skills (grooming its antennae): find the neurons to
drive for each, on the live connectome, from the plan stored for it, and store them in
server/data/skills_builtin.json (shipped with the app, so nobody has to wait for this).

    uv run tools/learn_builtin.py                     # every learned built-in
    uv run tools/learn_builtin.py groom_antennae      # or name the ones to redo

The walking and turning built-ins aren't learned: they activate the command neurons (P9, MDN, DNa02)
and the legs step as the brain commands (server/locomotion.py). Learning runs on the default body
(physical, with the legs' sense organs feeding back, as the app uses it), in a practice copy of the fly
(server/practice.py), and takes a minute or two.
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
        if builtin.get(name, {}).get("command"):
            print(name, "activates command neurons: nothing to learn", flush=True)
            continue
        desc, plan = builtin[name]["description"], builtin[name]["plan"]
        t = time.time()
        sk.lib.pop(name, None)
        res = await sk.learn(name, desc, plan, show=False)   # shaped in a practice copy (live, if none can start)
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
    stored = json.loads((ROOT / "server" / "data" / "skills_builtin.json").read_text())
    asyncio.run(main(sys.argv[1:] or [k for k, v in stored.items() if not v.get("command")]))
