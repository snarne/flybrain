"""
A practice copy of the fly (see practice.py): the whole model (brain, muscles, body, physics) without
the viewer, run as fast as the computer goes. The skill engine starts a few of these in the background;
each reads one JSON request per line on stdin and answers with one JSON line on stdout.

Requests:
  {"op": "trial", "drivers": [{"neurons": [...], "rates": [Hz per bin]}], "bin_s": 0.02, "duration": s,
   "tail": s, "assists": {"walking": false, "flight_timer": false}, "seed": int, "command": bool,
   "world": {"variant": "real", "variant_seed": 0, "ablated": [...]}}
      -> the recording: every motor control's value, where the body was, which way it faced and up,
         feet on the ground, per 33 ms frame
  {"op": "ping"}
"""
import json
import os
import sys
import time
from pathlib import Path

# answers go out on the real stdout; anything else printed (warnings, logs) goes to stderr
_proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.path.insert(0, str(Path(__file__).resolve().parent))

DT = 1.0 / 30


def _send(msg):
    _proto.write(json.dumps(msg, separators=(",", ":")) + "\n")
    _proto.flush()


class Sandbox:
    def __init__(self):
        import app
        self.app = app
        self.w, self.b, self.sk = app.world, app.body, app.skills
        self.w.mb.learning = False          # practice mustn't change what the mushroom body has learned
        self.w.brain.step(1)                # compile the kernel
        self.cfg = None

    def configure(self, cfg):
        """Match the live fly's wiring (variant) and silenced neurons."""
        key = json.dumps(cfg, sort_keys=True)
        if key == self.cfg:
            return
        w = self.w
        v, s = cfg.get("variant", "real"), int(cfg.get("variant_seed", 0))
        if v != w.variant or s != getattr(self, "_vseed", 0):
            w.set_variant(v, s)
            self._vseed = s
        want = set(cfg.get("ablated", []))
        for g in list(w.ablated):
            if g not in want:
                w.ablate(g, False)
        for g in want:
            if g not in w.ablated:
                w.ablate(g, True)
        self.cfg = key

    def frame(self):
        w, b = self.w, self.b
        b.sense(DT)
        w.frame(DT * 1000.0)
        b.act(DT, w.motor, w.fast_readouts(), skill=None)
        b.physics_step(DT)

    def _reset_body(self):
        """Back to the middle of the arena, standing. The first time the body settles onto its feet; after
        that the settled physical state is restored (the same start for every trial, and faster)."""
        b = self.b
        ph = b.phys
        snap = getattr(self, "_snap", None)
        if ph is None or snap is None:
            b.reset(home=True)
            if ph is not None:
                d = ph.data
                self._snap = (d.qpos.copy(), d.qvel.copy(), d.act.copy(), d.ctrl.copy(), d.qacc_warmstart.copy(),
                              ph._t, ph._k, dict(ph.touch), dict(ph.load))
            return
        import mujoco
        ph._settling = True                 # no settling: the snapshot is already settled
        try:
            b.reset(home=True)
        finally:
            ph._settling = False
        d = ph.data
        qpos, qvel, act, ctrl, warm, t_, k_, touch, load = snap
        d.qpos[:], d.qvel[:], d.act[:], d.ctrl[:], d.qacc_warmstart[:] = qpos, qvel, act, ctrl, warm
        ph._t, ph._k, ph.touch, ph.load = t_, k_, dict(touch), dict(load)
        mujoco.mj_forward(ph.model, d)
        b.sensed = ph.sensed()

    def reset(self, seed, assists, settle):
        w, b, sk = self.w, self.b, self.sk
        sk.state = None
        w.channel_stim.clear()
        b.assist_override = None
        for k, on in (assists or {}).items():
            b.set_assist(k, on)
        b.start("stop")
        b.start("clear")
        self._reset_body()
        w.apply_command({"cmd": "clear"})
        w.channel_stim.clear()
        w.apply_command({"cmd": "reset"})
        w.pending.clear()
        w.motor.reset()
        w.ro_fast[:] = 0
        w.ro_all[:] = 0
        w.brain.seed(seed)
        for _ in range(int(round(settle / DT))):
            self.frame()

    def trial(self, req):
        t0 = time.time()
        w, b = self.w, self.b
        drivers = req["drivers"]
        bin_s = float(req.get("bin_s", 0.02))
        n_frames = int(round((float(req["duration"]) + float(req.get("tail", 0.4))) / DT))
        self.reset(int(req.get("seed", 0)), req.get("assists"), float(req.get("settle", 0.3)))
        # the body acts on a skill's drive the way it does live: the walking model only follows command skills
        self.sk.state = {"kind": "trial", "name": "trial", "command": bool(req.get("command"))}
        righted0 = b.phys.righted if b.phys is not None else 0
        keys = list(w.motor.controls)
        rec = {"t": [], "val": {k: [] for k in keys}, "x": [], "y": [], "z": [], "heading": [], "upright": [],
               "feet": [], "air": [], "drive": []}
        last = None
        try:
            for f in range(n_frames):
                i = int(f * DT / bin_s + 1e-9)
                rates = {}
                for d in drivers:
                    r = d["rates"][i] if i < len(d["rates"]) else 0.0
                    if r > 0:
                        for n in d["neurons"]:
                            rates[n] = max(rates.get(n, 0.0), r)
                key = tuple(sorted(rates.items()))
                if key != last:
                    last = key
                    if rates:
                        w.channel_stim["skill"] = rates
                    else:
                        w.channel_stim.pop("skill", None)
                    w._push_stim()
                self.frame()
                rec["t"].append(round(f * DT, 4))
                for k in keys:
                    rec["val"][k].append(round(float(w.motor.value[k]), 3))
                s = b.sensed
                if s is not None:
                    rec["x"].append(round(s["pos"][0], 4))
                    rec["y"].append(round(s["pos"][1], 4))
                    rec["z"].append(round(s["pos"][2], 4))
                    rec["heading"].append(round(s["heading"], 4))
                    rec["upright"].append(round(s["upright"], 3))
                    feet = int(sum(s["touch"].values()))
                else:
                    p = b.pose
                    rec["x"].append(round(p["x"], 4))
                    rec["y"].append(round(p["y"], 4))
                    rec["z"].append(0.0)
                    rec["heading"].append(round(p["heading"], 4))
                    rec["upright"].append(1.0)
                    feet = 6
                rec["feet"].append(feet)
                rec["air"].append(feet == 0)
                rec["drive"].append(round(sum(rates.values()) / max(len(rates), 1), 1) if rates else 0.0)
        finally:
            self.sk.state = None
            w.channel_stim.pop("skill", None)
            w._push_stim()
        rec["righted"] = (b.phys.righted - righted0) if b.phys is not None else 0
        rec["wall"] = round(time.time() - t0, 2)
        return {"rec": rec}


def main():
    t0 = time.time()
    try:
        sb = Sandbox()
    except Exception as e:
        _send({"ready": False, "error": f"{type(e).__name__}: {e}"})
        return
    _send({"ready": True, "physics": sb.b.phys is not None, "load_s": round(time.time() - t0, 1)})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if req.get("world") is not None:
                sb.configure(req["world"])
            op = req.get("op", "trial")
            if op == "trial":
                res = sb.trial(req)
            elif op == "ping":
                res = {}
            else:
                raise ValueError(f"unknown op {op}")
            res["ok"] = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        _send(res)


if __name__ == "__main__":
    main()
