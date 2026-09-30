"""
Franka Emika Panda arm: pick a box off the floor and put it down 30 cm to the side.

Baseline controller: a scripted pick-and-place. It looks once where the box is (perception with a
possible error), then moves the gripper along straight lines (above the box, down, close, lift, over
the target, down, open, up), converting each point to joint angles with damped least-squares inverse
kinematics; the arm's joint servos track them. It never checks whether anything worked.

Recovery controller (recovery=True): the same script, plus checks that use what the robot can sense,
and a fix for each:
  - grasp check: after lifting, is the box in the hand (the fingers stopped on something and the box
    rose)? If not, take a closer look (a wrist-camera close-up: a third of the perception error) and
    grasp again (twice at most);
  - drop check: if the box falls out while carrying it, look again and pick it up from where it fell;
  - place check: after letting go, is the box where it should be? If not, pick it up and place it again
    (once).

Success: at the end the box stands upright within 3 cm of the target.
"""
import copy
import math

import mujoco
import numpy as np

from .assets import model_path
from .space import Dim

BOX0 = np.array([0.5, 0.0])          # where the box normally is (m)
TARGET = np.array([0.45, 0.30])      # where it goes
HALF_H = 0.03                        # box half height
LIFT_Z, HOVER_Z = 0.20, 0.13         # heights of the gripper centre for carrying and above the box
TOL = 0.03                           # m, success radius
GRIP_K = 10.0                        # gripper stiffness relative to the Menagerie model (see build)
CTRL_EVERY = 10                      # physics steps per control step (200 Hz)
TIMESTEP = 0.0005                    # s; at the default 2 ms a held box heavier than ~0.3 kg makes the contacts blow up

SPACE = [
    Dim("box_dx", -0.10, 0.10, 0.0, "where the box really is, forwards (m)"),
    Dim("box_dy", -0.10, 0.10, 0.0, "where the box really is, sideways (m)"),
    Dim("box_yaw", -0.78, 0.78, 0.0, "the box's rotation (rad)"),
    Dim("percep_dx", -0.025, 0.025, 0.0, "perception error, forwards (m)"),
    Dim("percep_dy", -0.025, 0.025, 0.0, "perception error, sideways (m)"),
    Dim("percep_yaw", -0.3, 0.3, 0.0, "perception error in the box's rotation (rad)"),
    Dim("box_width", 0.025, 0.06, 0.04, "box width (m; the gripper opens 8 cm)"),
    Dim("box_mass", 0.05, 1.0, 0.2, "box mass (kg)"),
    Dim("friction", 0.15, 1.5, 1.0, "finger-box friction coefficient"),
    Dim("grip_strength", 0.1, 1.0, 1.0, "fraction of the gripper's force"),
    Dim("arm_strength", 0.3, 1.0, 1.0, "fraction of the arm servos' stiffness and torque"),
    Dim("push_n", 0.0, 60.0, 0.0, "shove on the hand while carrying (N)"),
    Dim("push_dir", -math.pi, math.pi, 0.0, "shove direction (rad)"),
]

_BASE = None


def _base_model():
    global _BASE
    if _BASE is None:
        spec = mujoco.MjSpec.from_file(str(model_path("panda")))
        spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                                rgba=[0.3, 0.35, 0.4, 1])
        hand = spec.body("hand")
        hand.add_site(name="tcp", pos=[0, 0, 0.1034])
        box = spec.worldbody.add_body(name="box", pos=[BOX0[0], BOX0[1], HALF_H])
        box.add_freejoint()
        box.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, HALF_H], mass=0.2,
                     rgba=[0.1, 0.7, 0.2, 1], condim=3)
        spec.worldbody.add_site(name="target", pos=[TARGET[0], TARGET[1], 0.001], size=[TOL, 0.001, 0],
                                type=mujoco.mjtGeom.mjGEOM_CYLINDER, rgba=[1, 0.2, 0.2, 0.4])
        _BASE = spec.compile()
        _BASE.opt.timestep = TIMESTEP
    return _BASE


def _yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def _down(yaw):
    """Rotation with the gripper pointing down and its x axis at yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


class Panda:
    name = "panda"
    space = SPACE
    duration = 16.0            # at most, with retries

    def __init__(self):
        m = self.m0 = _base_model()
        n = lambda t, s: mujoco.mj_name2id(m, t, s)   # noqa: E731
        self.tcp = n(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.box = n(mujoco.mjtObj.mjOBJ_BODY, "box")
        self.box_geom = n(mujoco.mjtObj.mjOBJ_GEOM, "box")
        self.hand = n(mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.box_q = m.jnt_qposadr[m.body_jntadr[self.box]]
        fingers = {n(mujoco.mjtObj.mjOBJ_BODY, "left_finger"), n(mujoco.mjtObj.mjOBJ_BODY, "right_finger")}
        self.finger_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in fingers and m.geom_contype[g]]
        self.grip_tendon = n(mujoco.mjtObj.mjOBJ_TENDON, "split")
        self.home = m.key_qpos[0].copy()
        self.home_ctrl = m.key_ctrl[0].copy()

    def build(self, p):
        m = copy.deepcopy(self.m0)
        w = p["box_width"] / 2
        m.geom_size[self.box_geom] = (w, w, HALF_H)
        m.body_mass[self.box] = p["box_mass"]
        I = p["box_mass"] / 12 * np.array([2 * (2 * w) ** 2, (2 * w) ** 2 + (2 * HALF_H) ** 2, (2 * w) ** 2 + (2 * HALF_H) ** 2])
        m.body_inertia[self.box] = (I[1], I[1], I[0])
        m.geom_friction[self.box_geom, 0] = p["friction"]
        m.geom_friction[self.finger_geoms, 0] = p["friction"]
        s = p["arm_strength"]
        m.actuator_gainprm[:7, 0] *= s
        m.actuator_biasprm[:7, 1:3] *= s
        m.actuator_forcerange[:7] *= s
        # the gripper: Menagerie's servo squeezes a 4 cm box with only ~2 N; a real Panda hand squeezes with up to
        # 70 N, so its stiffness is raised (GRIP_K) and then scaled by grip_strength
        k = GRIP_K * p["grip_strength"]
        m.actuator_gainprm[7, 0] *= k
        m.actuator_biasprm[7, 1:3] *= k
        return m

    # ------------------------------------------------------------ inverse kinematics
    def _ik(self, mk, dk, q, pos, R, iters=3):
        jp, jr = np.zeros((3, mk.nv)), np.zeros((3, mk.nv))
        for _ in range(iters):
            dk.qpos[:7] = q
            mujoco.mj_kinematics(mk, dk)
            mujoco.mj_comPos(mk, dk)
            x = dk.site_xpos[self.tcp]
            Rc = dk.site_xmat[self.tcp].reshape(3, 3)
            e = np.concatenate([pos - x, 0.5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3))])
            mujoco.mj_jacSite(mk, dk, jp, jr, self.tcp)
            J = np.vstack([jp[:, :7], jr[:, :7]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + dq, mk.jnt_range[:7, 0], mk.jnt_range[:7, 1])
        return q

    def run(self, p, seed=0, recovery=False, record=False, viewer=None):
        rng = np.random.default_rng(seed)
        m = self.build(p)
        d = mujoco.MjData(m)
        mk = self.m0
        dk = mujoco.MjData(mk)
        d.qpos[:] = 0
        d.qpos[:9] = self.home[:9]
        bx = BOX0 + (p["box_dx"], p["box_dy"])
        d.qpos[self.box_q:self.box_q + 3] = (bx[0], bx[1], HALF_H)
        mujoco.mju_axisAngle2Quat(d.qpos[self.box_q + 3:self.box_q + 7], np.array([0, 0, 1.0]), p["box_yaw"])
        d.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(m, d)
        dt = m.opt.timestep
        q_cmd = d.qpos[:7].copy()
        home_yaw = _yaw_of(d.site_xmat[self.tcp].reshape(3, 3))
        state = {"t": 0.0, "grip": 255.0, "retries": 0, "places": 0, "events": []}
        pushed = [False]

        def box_pose():
            b = d.xpos[self.box]
            R = d.xmat[self.box].reshape(3, 3)
            return b.copy(), _yaw_of(R), float(R[2, 2])

        def look(closer=False):
            """Where the robot thinks the box is: the truth plus the perception error (and a little noise)."""
            b, yaw, _ = box_pose()
            k = 0.3 if closer else 1.0
            err = np.array([p["percep_dx"], p["percep_dy"]]) * k + rng.normal(0, 0.002, 2)
            return b[:2] + err, yaw + p["percep_yaw"] * k + rng.normal(0, 0.01)

        def grip_yaw(box_yaw):
            # fingers close along the gripper's y axis: any of the box's four sides, nearest the current yaw
            cands = [box_yaw + k * math.pi / 2 for k in range(-4, 5)]
            return min(cands, key=lambda a: abs((a - home_yaw + math.pi) % (2 * math.pi) - math.pi))

        def step(n=1, push=None):
            for _ in range(n):
                d.ctrl[:7] = q_cmd
                d.ctrl[7] = state["grip"]
                d.xfrc_applied[self.hand, :3] = push if push is not None else 0.0
                mujoco.mj_step(m, d)
                state["t"] += dt
                if viewer is not None:
                    viewer(m, d)

        def move(pos, yaw, secs, push_at=None):
            """Straight line from where the gripper is to pos, over secs."""
            nonlocal q_cmd
            dk.qpos[:7] = q_cmd
            mujoco.mj_kinematics(mk, dk)
            start = dk.site_xpos[self.tcp].copy()
            R = _down(yaw)
            n = max(1, int(secs / (dt * CTRL_EVERY)))
            for i in range(1, n + 1):
                a = i / n
                a = a * a * (3 - 2 * a)                       # smooth start and stop
                q_cmd = self._ik(mk, dk, q_cmd, start + (pos - start) * a, R)
                push = None
                if push_at is not None and not pushed[0] and p["push_n"] > 0 and a >= push_at:
                    push = p["push_n"] * np.array([math.cos(p["push_dir"]), math.sin(p["push_dir"]), 0.0])
                step(CTRL_EVERY, push)
                if push is not None and a >= push_at + 0.1:
                    pushed[0] = True
            if push_at is not None and p["push_n"] > 0:
                pushed[0] = True

        def pick(closer=False):
            xy, yaw = look(closer)
            gy = grip_yaw(yaw)
            state["grip"] = 255.0
            move(np.array([xy[0], xy[1], HOVER_Z]), gy, 1.0)
            move(np.array([xy[0], xy[1], HALF_H + 0.005]), gy, 0.7)
            state["grip"] = 0.0
            step(int(0.5 / dt))
            move(np.array([xy[0], xy[1], LIFT_Z]), gy, 0.7)
            return gy

        def holding():
            b, _, up = box_pose()
            opening = float(d.ten_length[self.grip_tendon])
            return b[2] > HALF_H + 0.05 and opening > 0.004

        def carry_and_place(gy):
            move(np.array([TARGET[0], TARGET[1], LIFT_Z]), gy, 1.2, push_at=0.3 if state["places"] == 0 else None)
            if recovery and not holding():
                return False
            move(np.array([TARGET[0], TARGET[1], HALF_H + 0.008]), gy, 0.7)
            state["grip"] = 255.0
            step(int(0.3 / dt))
            dk.qpos[:7] = q_cmd
            mujoco.mj_kinematics(mk, dk)
            above = dk.site_xpos[self.tcp].copy()
            above[2] = LIFT_Z
            move(above, gy, 0.5)
            return True

        # ---- the task
        gy = pick()
        if recovery:
            while not holding() and state["retries"] < 2:
                state["retries"] += 1
                state["events"].append("grasp missed: closer look, grasp again")
                gy = pick(closer=True)
        done = carry_and_place(gy)
        if recovery:
            while not done and state["retries"] < 3:
                state["retries"] += 1
                state["events"].append("dropped while carrying: pick it up again")
                gy = pick(closer=True)
                done = carry_and_place(gy)
            b, _, _ = box_pose()
            if np.linalg.norm(b[:2] - TARGET) > TOL * 0.7 and state["places"] < 1 and state["t"] < self.duration - 6:
                state["places"] += 1
                state["events"].append("placed off target: pick it up and place again")
                gy = pick(closer=True)
                carry_and_place(gy)
        step(int(0.5 / dt))
        b, _, up = box_pose()
        dist = float(np.linalg.norm(b[:2] - TARGET))
        m_d = max(-3.0, 1.0 - dist / TOL)
        m_u = (up - 0.8) * 5.0
        margin = float(min(m_d, m_u, 1.0))
        why = "" if margin >= 0 else ("box tipped over" if m_u < m_d else
                                      ("never picked up" if np.linalg.norm(b[:2] - bx) < 0.03 else "off target"))
        unstable = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number + d.warning[mujoco.mjtWarning.mjWARN_BADQVEL].number
                       + d.warning[mujoco.mjtWarning.mjWARN_BADQPOS].number) > 0
        info = {"distance_m": round(dist, 4), "upright": round(up, 3), "why": why, "retries": state["retries"],
                "events": state["events"], "time_s": round(state["t"], 2)}
        info["unstable"] = unstable     # the simulation blew up: a simulator artefact, not a robot failure
        return margin, info
