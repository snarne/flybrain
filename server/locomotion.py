"""
Walking: the nerve cord's stepping, commanded by the brain's descending neurons.

A walking fly's legs are stepped by circuits in the ventral nerve cord that aren't mapped yet (which
neurons generate the rhythm, and how the legs' timing is coordinated, is still open). What is known is
how the brain commands them: descending neurons such as P9 (DNp09) and DNg97 for walking forwards
(Bidaye et al. 2020), the moonwalkers (MDN) for walking backwards (Bidaye et al. 2014), and DNa01/DNa02
for steering, which shorten the strides of the legs on the side the fly turns to (Rayshubskiy et al.
2020; Yang et al. 2023). So walking is modelled the way NeuroMechFly does it (Lobato-Rios et al. 2022;
Wang-Chen et al. 2024), with the command coming from the simulated brain:

  - six leg oscillators in a tripod (front and hind legs of one side with the middle leg of the other);
  - each leg follows a real fly's recorded step (all seven joints, NeuroMechFly's kinematic recording)
    at its oscillator's phase (the adhesive pads stay off while stepping: see PAD_STANCE);
  - the brain sets the gait every frame from its descending neurons' firing: how fast (P9/DNg97, MDN,
    and each descending neuron's wiring influence on the leg networks, cpg.py), which way (MDN against
    P9/DNg97: backwards is the step cycle run in reverse), and turning (DNa01/DNa02 left against right:
    the legs on the inside of the turn take shorter strides); and, like a real walking fly, it holds
    its course against turns it didn't command;
  - the leg's premotor rhythm neurons (cpg.py) and through them its motor neurons fire in time with
    the steps.

So when and how the fly walks is decided by its neurons; how a single step moves the leg is a real
fly's. With no descending command, the legs are left to the motor neurons alone.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]
TRIPOD = {"LF": 0.0, "RM": 0.0, "LH": 0.0, "RF": 0.5, "LM": 0.5, "RH": 0.5}

START_HZ, STOP_HZ = 14.0, 7.0   # command strength (Hz of the command neurons) to start / keep walking
FULL_HZ = 80.0                  # command strength for the fastest walk
F_MIN, F_MAX = 3.0, 6.0         # steps per second (real flies: ~2-15)
AMP_MIN = 0.6                   # stride at the slowest walk, of the recorded stride
TURN_GAIN = 0.9                 # how much a full steering command shortens the inner legs' strides
PAD_STANCE = 0.0                # pad adhesion in stance while walking: real feet stick, but in this physics
                                # sticky feet make fast stepping chaotic (the fly spins or tips over); friction
                                # alone walks straight
RAMP_S = 0.15                   # s to blend into and out of stepping
SMOOTH_S = 0.12                 # s: smoothing of the command
COURSE_GAIN = 0.8               # course holding: steering per rad/s of unintended turning
TURN_RATE = 1.0                 # rad/s of turning a full steering command aims for


def load_steps():
    s = json.loads((ROOT / "web" / "data" / "fly_body.json").read_text())["steps"]
    ang = {leg: {d: np.asarray(v, float) for d, v in s["legs"][leg].items()} for leg in s["legs"]}
    period = s["n"] * s["dt"]
    stance_from = {leg: s["swing_stance"]["stance"][leg] / period for leg in s["legs"]}   # phase where stance starts
    return ang, stance_from


class Walker:
    def __init__(self):
        self.ang, self.stance_from = load_steps()
        self.n = len(next(iter(next(iter(self.ang.values())).values())))
        self.mean = {leg: {d: float(a.mean()) for d, a in dofs.items()} for leg, dofs in self.ang.items()}
        self.reset()

    def reset(self):
        self.phase = 0.0
        self.w = 0.0                       # blend into stepping, 0..1
        self.walking = False
        self.speed = 0.0                   # 0..1 (smoothed command)
        self.dir = 1.0                     # +1 forwards, -1 backwards (smoothed through 0: the cycle slows and reverses)
        self.steer = 0.0                   # -1..1, + = left
        self.yaw = 0.0                     # the body's own turning, smoothed (rad/s)
        self.hold = 0.0                    # course correction, added to steer
        self.why = ""

    # ------------------------------------------------------------ the brain's command, once a frame
    def command(self, r, wiring_drive, dt, yaw_rate=0.0):
        """r: readouts (Hz); wiring_drive: 0..1 from all descending neurons' influence on the leg networks;
        yaw_rate: the body's own turning (rad/s, + left), for holding a course."""
        fwd, back = r.get("forward", 0.0), r.get("backward", 0.0)
        tl, tr = r.get("turn_l", 0.0), r.get("turn_r", 0.0)
        c = max(fwd, back, 0.7 * max(tl, tr)) + FULL_HZ * wiring_drive
        if self.walking:
            self.walking = c > STOP_HZ
        else:
            self.walking = c > START_HZ
        k = 1.0 - math.exp(-dt / SMOOTH_S)
        speed = min(1.0, max(0.0, (c - STOP_HZ) / (FULL_HZ - STOP_HZ))) if self.walking else 0.0
        self.speed += k * (speed - self.speed)
        want_dir = -1.0 if back > fwd + 3.0 else 1.0
        self.dir += 2 * k * (want_dir - self.dir)
        steer = (tl - tr) / max(tl + tr, 25.0) if self.walking else 0.0
        self.steer += k * (max(-1.0, min(1.0, steer)) - self.steer)
        # holding a course: a walking fly corrects turns it didn't command (from optic flow and its legs'
        # sense organs); a steering command turns the fly about 1 rad/s per unit
        self.yaw += k * (yaw_rate - self.yaw)
        self.hold = max(-1.0, min(1.0, COURSE_GAIN * (self.steer * TURN_RATE - self.yaw))) if self.walking else 0.0
        parts = []
        if fwd > 5 and self.dir > 0:
            parts.append(f"P9/DNg97 forward {fwd:.0f} Hz")
        if back > 5 and self.dir < 0:
            parts.append(f"MDN backward {back:.0f} Hz")
        if abs(tl - tr) > 5:
            parts.append(f"DNa01/02 {'left' if tl > tr else 'right'} ahead by {abs(tl - tr):.0f} Hz")
        if wiring_drive > 0.05:
            parts.append("other descending neurons")
        self.why = ", ".join(parts)

    def amp(self, leg):
        base = AMP_MIN + (1 - AMP_MIN) * self.speed
        st = max(-1.0, min(1.0, self.steer + self.hold))
        inner = (leg[0] == "L" and st > 0) or (leg[0] == "R" and st < 0)
        return base * (1.0 - TURN_GAIN * abs(st)) if inner else base

    @property
    def freq(self):
        return F_MIN + (F_MAX - F_MIN) * self.speed

    # ------------------------------------------------------------ the legs, every few ms
    def advance(self, dt):
        """Move the oscillators on by dt. Returns (weight, {(leg, dof): angle}, {leg: pad adhesion 0..1})."""
        target = 1.0 if self.walking else 0.0
        self.w += max(-dt / RAMP_S, min(dt / RAMP_S, target - self.w))
        if self.w > 0:
            self.phase = (self.phase + self.dir * self.freq * dt) % 1.0
        return self.w, self.targets(), self.pads()

    def leg_phase(self, leg):
        return (self.phase + TRIPOD[leg]) % 1.0

    def targets(self):
        out = {}
        for leg in LEGS:
            x = self.leg_phase(leg) * self.n
            i0 = int(x) % self.n
            f = x - int(x)
            a = self.amp(leg)
            for dof, arr in self.ang[leg].items():
                v = arr[i0] * (1 - f) + arr[(i0 + 1) % self.n] * f
                out[(leg, dof)] = self.mean[leg][dof] + a * (v - self.mean[leg][dof])
        return out

    def in_swing(self, leg):
        return self.leg_phase(leg) < self.stance_from[leg]

    def pads(self):
        return {leg: 0.0 if self.in_swing(leg) else PAD_STANCE for leg in LEGS}

    def estimate(self):
        """Movement for the kinematic body (no physics): mm/s forward, rad/s turning (+ left), measured
        from the physical body walking these steps."""
        if self.w <= 0:
            return 0.0, 0.0
        stride = 0.9 * sum(self.amp(leg) for leg in LEGS) / 6          # mm per step at full stride
        fwd = self.w * self.dir * stride * self.freq
        turn = self.w * self.steer * 0.9 * self.freq / 7.4
        return fwd, turn
