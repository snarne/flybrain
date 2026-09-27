"""
The fly's world: an open arena with things in it, and the senses they reach.

Every frame the fly's position and posture (from physics.py) decide what its sensory neurons report
(neuron groups from tools/build_senses.py):

  food drop (sugar, bitter)   the leg taste neurons of each foot standing in it; the labellum's taste
                              neurons when the extended proboscis reaches it (and the fly drinks)
  odour source (fruit, CO2,   a plume around the source; each antenna's receptor neurons fire with the
  geosmin, cVA)               concentration at that antenna, so left and right differ near the source
  heat / cold zone            warm or cool air; the arista's hot and cold cells fire with the
                              temperature at that antenna, and more while it's changing
  humid / dry zone            the same for the hygrosensory neurons
  threat                      a dark object swooping straight at the fly; each eye's looming detectors
                              (LPLC2, LC4) fire with how fast it grows on that side of the visual field
  moving target               a small dark fly-sized object darting back and forth ahead of the fly, to one
                              side; each eye's small-object detectors (LC10a) fire while it moves in
                              that eye's view (they steer male flies following a female)
  moving stripes              a striped drum turning around the arena; each eye's T4/T5 motion
                              neurons fire with the slip between the stripes and the fly's own turning
                              (so turning with the stripes quietens them, as in a real optomotor arena)

Nothing here moves the fly. Where it goes is up to its brain and legs.
"""
import math

AIR_T, AIR_RH = 25.0, 50.0       # room temperature (C) and humidity (%)
ZONE_R, EDGE = 6.0, 1.5          # zone radius and the width of its soft edge (mm)
ZONE_DT, ZONE_DRH = 10.0, 40.0   # how much warmer/cooler, wetter/drier a zone is
PLUME = 6.0                      # odour plume width (mm)
DROP_R = 0.6                     # food drop radius (mm)
THREAT_R, THREAT_V, THREAT_D = 3.0, 40.0, 30.0   # radius, speed (mm/s), starting distance (mm)
QUANT = 5.0
QUANT_FINE = 1.0                 # Hz, for odours and air: the sense boost compares the two antennae, which differ by a
                                 # few Hz near a source
ARENA = 40.0                     # half-width of the arena floor (mm), as physics.py
TARGET_D, TARGET_R, TARGET_A, TARGET_F, TARGET_LIFE = 5.0, 0.5, 1.2, 1.0, 8.0   # distance, radius, sway (mm), Hz, s

# where the parts are on a standing fly, in its own frame (mm; x forward, y left): used by the
# kinematic body, which has no physics to say where its feet are
REST = {"LF": (1.31, 0.71), "LM": (-0.31, 1.47), "LH": (-1.72, 0.56), "RF": (1.31, -0.71), "RM": (-0.31, -1.47),
        "RH": (-1.72, -0.56), "LArista": (1.04, 0.24), "RArista": (1.04, -0.24), "Haustellum": (0.76, 0.0),
        "LEye": (0.76, 0.3), "REye": (0.76, -0.3), "Head": (0.75, 0.0)}


def smooth_inside(d, r=ZONE_R, edge=EDGE):
    """1 inside a zone, 0 outside, smooth across its edge."""
    x = (r - d) / edge
    return float(min(1.0, max(0.0, 0.5 + x / 2)))


def sat(x):
    return float(min(1.0, max(0.0, x)))


class Arena:
    def __init__(self, world):
        self.w = world
        self.S = world.presets.get("senses", {})
        self.labellum = {k: world.stim[k]["neurons"] for k in ("sugar", "bitter") if k in world.stim}
        self.objects = []
        self._id = 0
        self.stripes = 0.0             # drum angular velocity, rad/s (+ = counter-clockwise from above)
        self.stripe_angle = 0.0
        self.last_T = {}
        self.last_RH = {}
        self.felt = {}
        self.eating = 0.0

    # ------------------------------------------------------------ what's in the world
    def add(self, kind, fly, **kw):
        """Put something in the world, placed relative to the fly unless x, y are given."""
        x0, y0 = fly["xy"]
        h = fly["heading"]
        c, s = math.cos(h), math.sin(h)

        def ahead(dist, side=0.0):
            return x0 + dist * c - side * s, y0 + dist * s + side * c

        self._id += 1
        o = {"id": self._id, "kind": kind}
        if kind in ("sugar", "bitter"):
            # a drop just in front of the fly's front feet, unless told where
            x, y = (kw["x"], kw["y"]) if "x" in kw else ahead(kw.get("dist", 1.6))
            o.update(x=x, y=y, r=DROP_R, amount=1.0)
        elif kind in ("fruit", "co2", "geosmin", "cva"):
            x, y = (kw["x"], kw["y"]) if "x" in kw else ahead(kw.get("dist", 8.0), kw.get("side", 3.0))
            o.update(kind="odour", odour=kind, x=x, y=y, strength=1.0, spread=PLUME)
        elif kind in ("heat", "cold", "humid", "dry"):
            x, y = (kw["x"], kw["y"]) if "x" in kw else ahead(kw.get("dist", 4.0))
            o.update(x=x, y=y, r=ZONE_R)
        elif kind == "threat":
            side = kw.get("side", "left")
            az = {"left": math.pi / 2, "right": -math.pi / 2, "front": 0.0, "back": math.pi}.get(side, math.pi / 2)
            ex, ey = x0 + THREAT_D * math.cos(h + az), y0 + THREAT_D * math.sin(h + az)
            tx, ty = fly["head"][:2]
            d = math.hypot(tx - ex, ty - ey)
            o.update(x=ex, y=ey, z=1.2, vx=THREAT_V * (tx - ex) / d, vy=THREAT_V * (ty - ey) / d, r=THREAT_R,
                     side=side, theta=None)
        elif kind == "target":
            side = kw.get("side", "left")
            az = math.radians(40.0) * (1 if side == "left" else -1)
            cx, cy = x0 + TARGET_D * math.cos(h + az), y0 + TARGET_D * math.sin(h + az)
            o.update(cx=cx, cy=cy, x=cx, y=cy, z=TARGET_R, r=TARGET_R, side=side, t=0.0,
                     tx=-math.sin(h + az), ty=math.cos(h + az), az=None)
        elif kind == "stripes":
            self.stripes = float(kw.get("speed", 1.5))      # + = turning left (counter-clockwise)
            return None
        else:
            raise ValueError(f"unknown thing: {kind}")
        # everything stays inside the arena (a threat swoops in from outside it, and that's fine)
        if o["kind"] in ("sugar", "bitter", "odour", "heat", "cold", "humid", "dry"):
            lim = ARENA - (1.0 if o["kind"] in ("sugar", "bitter") else 2.0)
            o["x"], o["y"] = max(-lim, min(lim, float(o["x"]))), max(-lim, min(lim, float(o["y"])))
        elif o["kind"] == "target":
            lim = ARENA - 1.0
            o["cx"], o["cy"] = max(-lim, min(lim, o["cx"])), max(-lim, min(lim, o["cy"]))
            o["x"], o["y"] = o["cx"], o["cy"]
        # one of each odour/zone kind at a time keeps the world readable
        if o["kind"] in ("odour", "heat", "cold", "humid", "dry"):
            self.objects = [p for p in self.objects if not (p["kind"] == o["kind"] and p.get("odour") == o.get("odour"))]
        self.objects.append(o)
        return o

    def clear(self):
        self.objects = []
        self.stripes = 0.0

    def view(self):
        """For the viewer."""
        out = []
        for o in self.objects:
            v = {k: (round(x, 3) if isinstance(x, float) else x) for k, x in o.items()
                 if k not in ("theta", "az", "cx", "cy", "tx", "ty")}
            out.append(v)
        return {"objects": out, "stripes": round(self.stripes, 2), "stripe_angle": round(self.stripe_angle % (2 * math.pi), 3),
                "felt": self.felt}

    # ------------------------------------------------------------ the senses, each frame
    def sense(self, dt, fly):
        """fly: xy, heading, yaw_rate, parts {name: (x, y, z)}, feet {leg: (x, y, z)}, touch {leg: 0/1},
        proboscis (0..1 extension). Returns {neuron: Hz}."""
        S = self.S
        rates = {}
        felt = {}

        def fire(neurons, hz, q=QUANT):
            hz = round(min(200.0, hz) / q) * q
            if hz > 0:
                for n in neurons:
                    rates[n] = max(rates.get(n, 0.0), hz)

        P = fly["parts"]
        ant = {"left": P["LArista"], "right": P["RArista"]}
        # odours: concentration at each antenna
        for o in [o for o in self.objects if o["kind"] == "odour"]:
            groups = S.get("odour", {}).get(o["odour"], {})
            for side, a in ant.items():
                d2 = (a[0] - o["x"]) ** 2 + (a[1] - o["y"]) ** 2
                c = o["strength"] * math.exp(-d2 / (2 * o["spread"] ** 2))
                hz = 150.0 * c / (c + 0.2)
                fire(groups.get(side, []), hz, QUANT_FINE)
                if hz >= 5:
                    felt[f"{o['odour']} {side[0].upper()}"] = round(hz)
        # temperature and humidity at each antenna (the arista)
        for side, a in ant.items():
            T, RH = AIR_T, AIR_RH
            for o in self.objects:
                if o["kind"] in ("heat", "cold", "humid", "dry"):
                    inside = smooth_inside(math.hypot(a[0] - o["x"], a[1] - o["y"]), o["r"])
                    T += {"heat": ZONE_DT, "cold": -ZONE_DT}.get(o["kind"], 0.0) * inside
                    RH += {"humid": ZONE_DRH, "dry": -ZONE_DRH}.get(o["kind"], 0.0) * inside
            dT = (T - self.last_T.get(side, T)) / max(dt, 1e-3)
            self.last_T[side] = T
            self.last_RH[side] = RH
            hot = 100.0 * sat((T - AIR_T) / 8.0) + 150.0 * sat(dT / 4.0)
            cold = 100.0 * sat((AIR_T - T) / 8.0) + 150.0 * sat(-dT / 4.0)
            fire(S.get("heat", {}).get(side, []), hot, QUANT_FINE)
            fire(S.get("cold", {}).get(side, []), cold, QUANT_FINE)
            fire(S.get("humid", {}).get(side, []), 150.0 * sat((RH - AIR_RH) / 35.0), QUANT_FINE)
            fire(S.get("dry", {}).get(side, []), 150.0 * sat((AIR_RH - RH) / 35.0), QUANT_FINE)
            if abs(T - AIR_T) > 0.5:
                felt[f"{T:.0f} C {side[0].upper()}"] = round(max(hot, cold))
            if abs(RH - AIR_RH) > 3:
                felt[f"{RH:.0f}% RH {side[0].upper()}"] = 1
        # food: taste through the feet, and through the proboscis when it reaches the drop
        self.eating = 0.0
        for o in [o for o in self.objects if o["kind"] in ("sugar", "bitter")]:
            for leg, f in fly["feet"].items():
                if fly["touch"].get(leg) and math.hypot(f[0] - o["x"], f[1] - o["y"]) < o["r"] + 0.15:
                    fire(S.get("leg_taste", {}).get(leg, {}).get(o["kind"], []), 150.0)
                    felt[f"{o['kind']} ({leg} foot)"] = 150
            m = P["Haustellum"]
            if fly["proboscis"] > 0.25 and math.hypot(m[0] - o["x"], m[1] - o["y"]) < o["r"] + 0.35:
                fire(self.labellum.get(o["kind"], []), 150.0)
                felt[f"{o['kind']} (proboscis)"] = 150
                if o["kind"] == "sugar":
                    o["amount"] -= 0.08 * dt          # drinking: the drop shrinks
                    o["r"] = DROP_R * max(0.2, o["amount"]) ** (1 / 3)
                    self.eating = 1.0
        self.objects = [o for o in self.objects if o.get("amount", 1.0) > 0.02]
        # threats: looming on each eye
        h = fly["heading"]
        c, s = math.cos(h), math.sin(h)
        eye = P["Head"]
        for o in [o for o in self.objects if o["kind"] == "threat"]:
            o["x"] += o["vx"] * dt
            o["y"] += o["vy"] * dt
            dx, dy, dz = o["x"] - eye[0], o["y"] - eye[1], o["z"] - eye[2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            theta = 2 * math.atan(o["r"] / max(d - 0.0, 0.3))
            grow = (theta - o["theta"]) / max(dt, 1e-3) if o["theta"] is not None else 0.0
            o["theta"] = theta
            az = math.atan2(-dx * s + dy * c, dx * c + dy * s)       # in the fly's frame, + = left
            hz = min(200.0, 32.0 * max(grow, 0.0)) if theta > math.radians(8) else 0.0
            sides = [k for k, ok in (("left", az > -math.radians(30)), ("right", az < math.radians(30))) if ok]
            for side in sides:
                fire(S.get("loom", {}).get(side, []), hz)
            if hz > 0:
                felt[f"looming {'/'.join(x[0].upper() for x in sides)}"] = round(hz)
            # it passes over the fly and away
            o["done"] = (dx * o["vx"] + dy * o["vy"]) > 0 and d > 3 * o["r"]
        self.objects = [o for o in self.objects if not o.get("done")]
        # moving target: small-object detectors of the eye(s) that see it, while it moves on the retina
        for o in [o for o in self.objects if o["kind"] == "target"]:
            o["t"] += dt
            sway = TARGET_A * math.sin(2 * math.pi * TARGET_F * o["t"])
            o["x"], o["y"] = o["cx"] + sway * o["tx"], o["cy"] + sway * o["ty"]
            dx, dy = o["x"] - eye[0], o["y"] - eye[1]
            d = math.hypot(dx, dy)
            az = math.atan2(-dx * s + dy * c, dx * c + dy * s)       # + = left
            slip = abs(az - o["az"]) / max(dt, 1e-3) if o["az"] is not None else 0.0
            o["az"] = az
            size = 2 * math.atan(o["r"] / max(d, 0.5))
            hz = 150.0 * sat(size / math.radians(10)) * (0.3 + 0.7 * sat(slip / 0.5))
            if abs(az) > math.radians(160):
                hz = 0.0                                             # behind the fly: out of view
            sides = [k for k, ok in (("left", az > -math.radians(15)), ("right", az < math.radians(15))) if ok]
            for side in sides:
                fire(S.get("target", {}).get(side, []), hz)
            if hz >= 5:
                felt[f"target {'/'.join(x[0].upper() for x in sides)}"] = round(hz)
            o["done"] = o["t"] > TARGET_LIFE
        self.objects = [o for o in self.objects if not o.get("done")]
        # moving stripes: slip between the drum and the fly's own turning
        if self.stripes:
            self.stripe_angle += self.stripes * dt
            slip = self.stripes - fly["yaw_rate"]
            hz = 30.0 * sat(abs(slip) / 1.5)
            M = S.get("motion", {})
            a, b = ("front_to_back", "back_to_front") if slip > 0 else ("back_to_front", "front_to_back")
            fire(M.get(a, {}).get("left", []), hz)
            fire(M.get(b, {}).get("right", []), hz)
            if hz > 0:
                felt[f"stripes turning {'left' if slip > 0 else 'right'}"] = round(hz)
        self.felt = felt
        return rates


def fly_from_physics(sensed, proboscis):
    """The fly as the arena needs it, from physics.PhysicsBody.sensed()."""
    th = sensed["thorax"]
    return {"xy": (th[0], th[1]), "heading": sensed["heading"], "yaw_rate": sensed["yaw"],
            "parts": sensed["parts"], "feet": sensed["feet"], "touch": sensed["touch"], "proboscis": proboscis,
            "head": sensed["parts"]["Head"]}


def fly_from_pose(x, y, heading, yaw_rate, proboscis, z=0.0):
    """The same for the kinematic body: parts at their resting places, all six feet down."""
    c, s = math.cos(heading), math.sin(heading)
    at = lambda p, zz=0.0: (x + p[0] * c - p[1] * s, y + p[0] * s + p[1] * c, zz)
    parts = {k: at(v, 1.17) for k, v in REST.items() if len(k) > 2}
    feet = {k: at(v, 0.0) for k, v in REST.items() if len(k) == 2}
    return {"xy": (x, y), "heading": heading, "yaw_rate": yaw_rate, "parts": parts, "feet": feet,
            "touch": {k: 1.0 for k in feet}, "proboscis": proboscis, "head": parts["Head"]}
