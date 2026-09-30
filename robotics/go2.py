"""
Unitree Go2 quadruped: stand, then trot forwards for 5 s.

Baseline controller: a fixed trot (diagonal legs together, 2.5 steps/s, 10 cm strides): each foot
follows a swing/stance path in the hip's frame, converted to joint angles (leg inverse kinematics)
and tracked by joint PD control on the torque motors. It uses only joint angles; it doesn't look at
the body.

Recovery controller (recovery=True): the same trot plus two reflexes that use what the body senses,
the way an animal's sense organs correct its stepping:
  - placement: a swinging foot lands further in the direction the body is drifting (Raibert);
  - soften: when the joints start shaking (delayed or noisy feedback makes a stiff controller
    oscillate), the joints' stiffness drops to 40% until it calms, the way animals lower co-contraction.
(Tried and left out because they made things worse in the probes: levelling the body by leg length,
shorter strides when the feet slip, pushing harder when the body sinks.)

Success: after the 5 s trot the robot is upright (body up-vector > 0.5, body higher than 12 cm all the
time) and has gone forwards at least half of what the trot should cover.
"""
import copy
import math

import mujoco
import numpy as np

from .assets import model_path
from .space import Dim

LEGS = ("FL", "FR", "RL", "RR")
PHASE = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}
L1 = L2 = 0.213            # thigh and calf length (m)
FREQ, STRIDE, LIFT, HEIGHT = 2.5, 0.10, 0.07, 0.27
KP, KD = 200.0, 4.0
SETTLE_S, WALK_S = 1.0, 5.0
V_DES = 2 * STRIDE * FREQ  # m/s, the trot's speed if the feet don't slip
RECOVERY = ("placement", "soften")
PLACE, SHAKE = 0.12, 300.0     # m of foot offset per m/s of sideways drift; joint shaking (rad/s^2) that softens

SPACE = [
    Dim("friction", 0.1, 1.2, 0.8, "foot-ground friction coefficient"),
    Dim("payload_kg", 0.0, 8.0, 0.0, "extra mass on the body"),
    Dim("payload_x", -0.12, 0.12, 0.0, "payload position, forwards (m)"),
    Dim("motor_strength", 0.3, 1.0, 1.0, "fraction of the motors' torque"),
    Dim("delay_ms", 0.0, 40.0, 0.0, "sensor-to-motor delay"),
    Dim("joint_noise", 0.0, 0.06, 0.0, "joint angle sensor noise (rad)"),
    Dim("push_n", 0.0, 400.0, 0.0, "sideways/forward shove on the body for 0.1 s (N)"),
    Dim("push_dir", -math.pi, math.pi, 0.0, "shove direction (rad, 0 = forwards)"),
    Dim("push_t", 1.0, 4.0, 2.5, "when the shove comes (s into the trot)"),
    Dim("slope_deg", -12.0, 12.0, 0.0, "ground slope along the walking direction (+ = uphill)"),
]

_BASE = None


def _base_model():
    global _BASE
    if _BASE is None:
        spec = mujoco.MjSpec.from_file(str(model_path("go2")))
        spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                                rgba=[0.3, 0.35, 0.4, 1])
        _BASE = spec.compile()
    return _BASE


def leg_ik(x, y, z):
    """Foot position (hip frame, m: forwards, outwards, down negative) -> abduction, thigh, calf angles."""
    a = math.atan2(y, -z)
    zz = -math.hypot(y, z)
    d = min(math.hypot(x, zz), L1 + L2 - 1e-4)
    c = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    knee = -math.acos(max(-1.0, min(1.0, c)))
    alpha = math.atan2(-x, -zz)
    return a, alpha - knee / 2, knee


def foot_path(phi, stride, lift, height, land_dx=0.0, land_dy=0.0):
    """One step cycle: swing (phi < 0.5) then stance. Returns x, y, z in the hip frame."""
    if phi < 0.5:
        s = phi / 0.5
        c = s - math.sin(2 * math.pi * s) / (2 * math.pi)       # cycloid: smooth lift-off and touch-down
        x = -stride / 2 + (stride + land_dx) * c
        return x, land_dy * c, -height + lift * math.sin(math.pi * s)
    s = (phi - 0.5) / 0.5
    return stride / 2 + land_dx - (stride + land_dx) * s, land_dy * (1 - s), -height


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


class Go2:
    name = "go2"
    space = SPACE
    duration = SETTLE_S + WALK_S

    def __init__(self):
        self.m0 = _base_model()
        m = self.m0
        self.jq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{j}_joint")]
                   for leg in LEGS for j in ("hip", "thigh", "calf")]
        self.jv = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{j}_joint")]
                   for leg in LEGS for j in ("hip", "thigh", "calf")]
        self.act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_{j}")
                    for leg in LEGS for j in ("hip", "thigh", "calf")]
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, leg) for leg in LEGS]
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.hips = {leg: m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_hip")].copy()
                     + np.array([0, (0.0955 if leg[1] == "L" else -0.0955), 0]) for leg in LEGS}
        self.home = m.key_qpos[0].copy()

    def build(self, p):
        m = copy.deepcopy(self.m0)
        m.geom_friction[self.feet, 0] = p["friction"]
        m.geom_friction[self.floor, 0] = p["friction"]
        b = self.base
        mass0, c0 = float(m.body_mass[b]), m.body_ipos[b].copy()
        pm = p["payload_kg"]
        if pm > 0:
            off = np.array([p["payload_x"], 0.0, 0.05])
            m.body_mass[b] = mass0 + pm
            m.body_ipos[b] = (mass0 * c0 + pm * off) / (mass0 + pm)
            m.body_inertia[b] = m.body_inertia[b] + pm * 0.01
        s = p["motor_strength"]
        m.actuator_gear[self.act, 0] *= s
        # a slope, as gravity tilted against the walking direction (the same physics as a tilted floor)
        ang = math.radians(p["slope_deg"])
        m.opt.gravity[:] = (-9.81 * math.sin(ang), 0.0, -9.81 * math.cos(ang))
        return m

    def run(self, p, seed=0, recovery=False, record=False, viewer=None):
        rng = np.random.default_rng(seed)
        m = self.build(p)
        d = mujoco.MjData(m)
        d.qpos[:] = self.home
        mujoco.mj_forward(m, d)
        dt = m.opt.timestep
        n_settle, n_walk = int(SETTLE_S / dt), int(WALK_S / dt)
        delay = int(round(p["delay_ms"] / 1000 / dt))
        buf = []
        push_t0 = SETTLE_S + p["push_t"]
        pf = p["push_n"] * np.array([math.cos(p["push_dir"]), math.sin(p["push_dir"]), 0.0])
        min_h, min_up = 10.0, 1.0
        x_start = None
        stride_scale = 1.0
        phase_jitter = rng.uniform(-0.02, 0.02)
        rec = [] if record else None
        rx = set(RECOVERY if recovery is True else (recovery or ()))
        gain, shake, qd_prev = 1.0, 0.0, np.zeros(12)
        lim = m.actuator_ctrlrange[self.act, 1]
        for k in range(n_settle + n_walk):
            t = k * dt
            # what the controller senses (maybe late, maybe noisy)
            obs = (d.qpos[self.jq].copy(), d.qvel[self.jv].copy(), d.qpos[3:7].copy(), d.qvel[0:3].copy(),
                   d.qpos[2], [d.geom_xpos[g].copy() for g in self.feet])
            buf.append(obs)
            if len(buf) > delay + 1:
                buf.pop(0)
            q, qd, quat, vel, bz, feet = buf[0]
            if p["joint_noise"] > 0:
                q = q + rng.normal(0, p["joint_noise"], len(q))
            walking = t >= SETTLE_S
            tw = t - SETTLE_S
            ramp = min(1.0, max(0.0, tw / 0.5)) if walking else 0.0
            R = quat_to_mat(quat)
            vb = R.T @ vel                      # body-frame velocity
            qdes = np.zeros(12)
            for i, leg in enumerate(LEGS):
                phi = (tw * FREQ + PHASE[leg] + phase_jitter) % 1.0 if walking else 0.75
                stride = STRIDE * ramp * stride_scale
                height, dy = HEIGHT, 0.0
                if "placement" in rx and walking:
                    dy = float(np.clip(PLACE * vb[1], -0.06, 0.06))       # step towards the drift
                x, y, z = foot_path(phi, stride, LIFT if walking else 0.0, height, 0.0, dy)
                qdes[3 * i:3 * i + 3] = leg_ik(x, y, z)
            if "soften" in rx:
                # shaking (delayed or noisy feedback): the joints' fast back-and-forth, relative to the trot's
                shake += (float(np.mean(np.abs(qd - qd_prev))) / dt - shake) * 0.02
                gain += ((0.4 if shake > SHAKE else 1.0) - gain) * 0.005
            qd_prev = qd
            tau = gain * (KP * (qdes - q) - KD * qd)
            d.ctrl[self.act] = np.clip(tau, -lim, lim)
            d.xfrc_applied[self.base, :3] = pf if (push_t0 <= t < push_t0 + 0.1) else 0.0
            mujoco.mj_step(m, d)
            if walking:
                if x_start is None:
                    x_start = float(d.qpos[0])
                h = float(d.qpos[2])
                min_h = min(min_h, h)
                up = quat_to_mat(d.qpos[3:7])[2, 2]
                min_up = min(min_up, float(up))
            if record and k % 10 == 0:
                rec.append(d.qpos.copy())
            if viewer is not None:
                viewer(m, d)
        dist = float(d.qpos[0]) - x_start
        expect = V_DES * (WALK_S - 0.25)
        m_h = (min_h - 0.12) / 0.1
        m_u = (min_up - 0.5) / 0.5
        m_p = dist / (0.5 * expect) - 1.0
        margin = float(min(m_h, m_u, m_p, 1.0))
        unstable = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number + d.warning[mujoco.mjtWarning.mjWARN_BADQVEL].number
                       + d.warning[mujoco.mjtWarning.mjWARN_BADQPOS].number) > 0
        info = {"distance_m": round(dist, 3), "min_height_m": round(min_h, 3), "min_upright": round(min_up, 3),
                "fell": bool(m_h < 0 or m_u < 0), "why": "fell" if (m_h < 0 or m_u < 0) else ("too slow" if m_p < 0 else "")}
        info["unstable"] = unstable     # the simulation blew up: a simulator artefact, not a robot failure
        return margin, info
