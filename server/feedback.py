"""
The body talking back: leg sense organs -> sensory neurons in the connectome.

Each frame the physics body (physics.py) reports every leg's joint angles, joint speeds, the load on
the leg and whether it touches the ball. This turns them into firing rates for the leg's
proprioceptive and touch neurons (found in BANC by tools/build_proprio.py), which then drive the nerve
cord like any other input. Tuning follows recordings where they exist (Mamiya et al. 2018, 2023;
Tuthill & Wilson 2016; Zill et al. 2004):

  claw  tonic, range-fractionated position: flexion neurons fire as the femur-tibia joint flexes past
        rest, extension neurons as it extends; each claw neuron covers part of the range, so the
        population stays quiet around the standing posture (CLAW_START)
  hook  phasic, direction: flexion or extension movement, scaled by speed
  club  movement in either direction, scaled by speed (the organ also senses vibration)
  hp    hair plates: the joint approaching either end of its range
  cs    campaniform sensilla: changes in load on the leg (strongly), and the load itself (weakly)
  touch tarsal tactile bristles: the foot touching down (they adapt within ~50 ms)

Presynaptic inhibition: during movements the fly makes itself, GABAergic interneurons suppress the
claw (position) and hook (movement) neurons' output, while club and load signals get through
(Dallmann et al. 2023, "Presynaptic inhibition selectively suppresses leg proprioception in behaving
Drosophila"). A leaky integrate-and-fire network has no presynaptic inhibition, so it's applied here:
while a movement is being produced (a rhythm or a skill), claw and hook rates are scaled by GATED.
Without it, the resistance reflexes these neurons drive fight every step.

Directions use the model's joint angles: for the tibia, a larger angle is more flexed (motor.py's
"reach" extends the tibia by making it smaller).
"""
import math

import numpy as np

MAX_HZ = 150.0
TONIC_HZ = {"claw": 150.0, "hook": 150.0, "club": 100.0, "hp": 60.0, "cs": 80.0, "touch": 60.0}
SPEED_FULL = 10.0       # rad/s of joint speed that saturates the movement-sensing neurons
SPEED_MIN = 1.0         # below this (rad/s) they don't respond
HP_START = 0.6          # hair plates start firing past this fraction of the joint's walking range
CLAW_START = 0.35       # claw neurons start firing past this fraction: a standing fly's legs give a little
                        # under its weight (up to ~0.3), and a tonic response to that set off resistance
                        # reflexes that twitched the resting legs
LOAD_FULL = 12.0        # contact force on a leg that saturates the sensilla's tonic response (model units)
LOAD_RATE_FULL = 150.0  # rate of change of load that saturates their phasic response (units/s)
CS_TONIC = 0.25         # share of the campaniform response that is tonic
TOUCH_S = 0.05          # how long the tarsal bristles fire after touch-down (s)
QUANT = 5.0             # rates are rounded to this many Hz, so small wobbles don't rebuild the input
GATED = 0.2             # claw and hook output during self-generated movement (presynaptic inhibition)
GATED_ORGANS = ("claw", "hook")
ORGAN_GAIN = {}          # per-organ multiplier (experiments)


class Feedback:
    def __init__(self, groups, ranges, neutral):
        """groups: presets["proprio"]; ranges: motor.py's per-leg joint ranges; neutral: resting angles."""
        self.groups = [g for g in groups if g["neurons"]]
        self.ranges = ranges
        self.neutral = neutral
        self.gain = 1.0
        self.last = {}
        self.last_load = {}
        self.last_angle = {}
        self.touch_t = {}
        self.neurons = sorted({n for g in self.groups for n in g["neurons"]})

    def rates(self, sensed, moving=False, dt=1 / 30):
        """sensed: physics.PhysicsBody.sensed(); moving: the fly is making a movement itself.
        Returns {neuron: Hz} (only neurons that fire)."""
        out = {}
        if sensed is None or self.gain <= 0:
            return out
        A = sensed["angle"]
        # joint movement over the frame (not the instantaneous speed, which chatters with contact)
        V = {}
        for k, a in A.items():
            v = (a - self.last_angle.get(k, a)) / max(dt, 1e-3)
            V[k] = math.copysign(max(0.0, abs(v) - SPEED_MIN), v)
            self.last_angle[k] = a
        # load changes and touch-downs, per leg
        dload, touched = {}, {}
        for leg, ld in sensed["load"].items():
            dload[leg] = abs(ld - self.last_load.get(leg, ld)) / max(dt, 1e-3)
            self.last_load[leg] = ld
            on = sensed["touch"].get(leg, 0.0) > 0
            t = self.touch_t.get(leg)
            if on and t is None:
                t = 0.0                                  # just touched down
            elif on:
                t += dt
            else:
                t = None
            self.touch_t[leg] = t
            touched[leg] = 1.0 if (t is not None and t < TOUCH_S) else 0.0
        for g in self.groups:
            leg, organ, joint, sense = g["leg"], g["organ"], g["joint"], g["sense"]
            x = 0.0
            if joint is not None:
                a = A.get((leg, joint), 0.0) - self.neutral[leg][joint]
                v = V.get((leg, joint), 0.0)
                half = max(self.ranges.get(leg, {}).get(joint, {}).get("half", 0.3), 0.05)
                if organ == "claw":
                    x = ((a if sense == "flexion" else -a if sense == "extension" else abs(a)) / half - CLAW_START) \
                        / (1.0 - CLAW_START)
                elif organ == "hook":
                    x = (v if sense == "flexion" else -v if sense == "extension" else abs(v)) / SPEED_FULL
                elif organ == "club":
                    x = abs(v) / SPEED_FULL
                elif organ == "hp":
                    x = (abs(a) / half - HP_START) / (1.0 - HP_START)
            elif organ == "cs":
                x = (1 - CS_TONIC) * min(1.0, dload.get(leg, 0.0) / LOAD_RATE_FULL) \
                    + CS_TONIC * min(1.0, sensed["load"].get(leg, 0.0) / LOAD_FULL)
            elif organ == "touch":
                x = touched.get(leg, 0.0)
            hz = TONIC_HZ.get(organ, MAX_HZ) * float(np.clip(x, 0.0, 1.0)) * self.gain
            if moving and organ in GATED_ORGANS:
                hz *= GATED
            hz *= ORGAN_GAIN.get(organ, 1.0)
            hz = round(hz / QUANT) * QUANT
            if hz > 0:
                for n in g["neurons"]:
                    out[n] = max(out.get(n, 0.0), hz)
        return out

    def summary(self, rates):
        """Mean rate per leg and organ, for the panel."""
        s = {}
        for g in self.groups:
            r = [rates.get(n, 0.0) for n in g["neurons"]]
            k = f"{g['leg']}.{g['organ']}"
            s[k] = max(s.get(k, 0.0), round(float(np.mean(r)), 1))
        return s
