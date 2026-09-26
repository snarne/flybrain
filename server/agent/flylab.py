"""
flylab: a small Python helper for scripts that work with the running fly (the Flybrain app).

    import flylab as fl
    fl.status()                         # behaviour readouts, active regions and neurons
    fl.body()                           # what the body is doing, where it is, the hand-built parts
    fl.find("DNa")                      # cell types matching a name, class or super class
    fl.info("DNp01")                    # one cell type: inputs, outputs, firing now, what the app uses it for
    fl.path("LC4", "DNp01")             # the strongest chain of cell types from one to the other
    fl.rates(["DNp01", "MDN"])          # firing rates now (Hz)
    fl.skills(); fl.skill("walk_forward")
    fl.controls()                       # every motor control and its muscles, plus the planning reference
    fl.test("walk_forward", assists={"walking": False})    # measured in a background copy of the fly
    fl.learn("hop", "a small jump", keyframes=[...], goal="jump_high", trials=40)
    fl.do(["hop"])                      # on the live fly
    fl.activate("DNp09", seconds=2)     # optogenetics-style, on the live fly
    fl.watch(2.0)                       # sample the body for 2 s: mode, speed, turning over time

It only talks to the app's local HTTP API (http://127.0.0.1:8765, or $FLYBRAIN_URL); nothing else.
Everything returns plain dicts and lists. Needs no packages beyond Python itself.
"""
import json
import os
import time
import urllib.parse
import urllib.request

FLYLAB_VERSION = "2"
URL = os.environ.get("FLYBRAIN_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8765"))


def get(path, **params):
    q = ("?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})) if params else ""
    with urllib.request.urlopen(URL + path + q, timeout=60) as r:
        return json.loads(r.read())


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------- reading (never changes anything)
def status():
    return get("/api/status")


def body():
    return get("/api/body")


def find(query):
    return get("/api/find", q=query)


def info(cell_type):
    return get("/api/typeinfo/" + urllib.parse.quote(cell_type))


def path(src, dst):
    return get("/api/path", src=src, dst=dst)


def neurons(cell_type):
    return get("/api/type/" + urllib.parse.quote(cell_type))["neurons"]


def neuron(index):
    return get(f"/api/neuron/{int(index)}")


def rates(types=(), neurons=()):
    return get("/api/rates", types=",".join(types), neurons=",".join(str(n) for n in neurons))


def skills():
    return get("/api/skills")["skills"]


def skill(name):
    return get("/api/skill/" + urllib.parse.quote(name))


def controls():
    return get("/api/controls")


def practice_info():
    return get("/api/practice")


def watch(seconds=2.0, every=0.1):
    """Sample the live body for a while: time, behaviour, forward speed (mm/s), turning (rad/s, + left), position."""
    out, t0 = [], time.time()
    while time.time() - t0 < seconds:
        b = body()
        out.append({"t": round(b["t_ms"] / 1000, 3), "mode": b["mode"], "forward": b["ball"][0],
                    "turn_left": -b["ball"][1], "pos": b["fly"]["pos"], "airborne": b["airborne"]})
        time.sleep(every)
    return out


# ---------------------------------------------------------------- practice copies (the live fly isn't touched)
def test(name, assists=None, goal=None, runs=3, duration=None):
    """Measure a skill: forward_mm, turn_deg, height_mm, airtime_s, fell, wing_power, ... (mean of runs)."""
    return post("/api/skills/test", {"name": name, "assists": assists, "goal": goal, "runs": runs, "duration": duration})


def learn(name, description, keyframes=None, duration=None, goal=None, trials=None, baseline=None, assists=None,
          gait=None, drive_at=None, show=True):
    """Learn a skill (as the learn_skill tool): keyframes, or gait={"steps_per_s": 5, "duration": 2} for a walking
    plan from recorded steps. drive_at: "auto" | "nerve_cord" | "motor". Takes a minute or more."""
    return post("/api/skills/learn", {"name": name, "description": description, "keyframes": keyframes, "gait": gait,
                                      "duration": duration, "goal": goal, "trials": trials, "baseline": baseline,
                                      "assists": assists, "drive_at": drive_at, "show": show})


def refine(name, goal=None, trials=None, keyframes=None, duration=None, baseline=None, assists=None):
    return post("/api/skills/learn", {"name": name, "refine": True, "goal": goal, "trials": trials,
                                      "keyframes": keyframes, "duration": duration, "baseline": baseline,
                                      "assists": assists})


def combine(name, description, steps):
    """A sequence of skills under one name: steps=[{"skill": "walk_forward", "repeat": 4}, {"skill": "hop"}]."""
    return post("/api/skills/combine", {"name": name, "description": description, "steps": steps})


# ---------------------------------------------------------------- the live fly (changes what it's doing)
def command(c):
    """Any app command, e.g. {"cmd": "preset", "key": "sugar", "rate": 150} or {"cmd": "body", "action": "loom",
    "side": "left"}. See guide('scripts')."""
    return post("/api/command", c, timeout=60)


def do(names, repeat=1):
    return post("/api/skills/do", {"names": list(names), "repeat": repeat})


def activate(cell_type, seconds=2.0, rate=150.0):
    """Drive every neuron of a cell type (Poisson spikes) for a while, then stop."""
    idx = neurons(cell_type)
    command({"cmd": "channel", "name": "flylab", "neurons": idx, "rate": rate})
    time.sleep(seconds)
    command({"cmd": "channel", "name": "flylab", "rate": 0})
    return len(idx)


def set_parts(walking=None, flight_timer=None):
    """Switch the live body's hand-built parts on or off."""
    for k, v in (("walking", walking), ("flight_timer", flight_timer)):
        if v is not None:
            command({"cmd": "body", "action": "assist", "name": k, "on": bool(v)})
    return body()["physics"].get("assists")


def reset(home=True):
    return command({"cmd": "body", "action": "reset", "home": home})
