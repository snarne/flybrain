"""
A physical fly body: NeuroMechFly (Lobato-Rios et al. 2022, Wang-Chen et al. 2024; flygym, Apache-2.0)
simulated with MuJoCo, tethered over a floating ball like a real rig.

Nothing here decides what the fly does. Each frame:

  muscles (motor.py)  ->  each leg joint gets a target angle and a stiffness   (the muscle layer)
  MuJoCo              ->  gravity, contacts with the ball, friction, leg inertia
  the body            ->  joint angles, joint speeds, foot load, what touches the ball
                      ->  leg sense organs (feedback.py) -> sensory neurons in the connectome

The muscle layer is an equilibrium-point model: the agonist and antagonist of a joint pull it towards
an angle set by their difference (motor.py's mapping), and together they set how stiffly it's held.
Both muscles relaxed = a floppy joint; both contracted = a stiff one (co-contraction), as in insects.

The ball is real: it has mass and friction, and only turns when feet push it. Its spin is what the
viewer shows and what the scoring suite reads, exactly like the ball tracker on a real rig.

Needs the `mujoco` package. Without it the app uses the kinematic body (joint angles straight from
the muscles, ball estimated from leg sweeps).
"""
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    import mujoco
except Exception:          # not installed: the app falls back to the kinematic body
    mujoco = None

ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = Path(__file__).parent / "data" / "neuromechfly.xml"
BODY_JSON = ROOT / "web" / "data" / "fly_body.json"
BODY_BIN = ROOT / "web" / "data" / "fly_body.bin"

LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]
DOFS = ["Coxa", "Coxa_roll", "Coxa_yaw", "Femur", "Femur_roll", "Tibia", "Tarsus1"]
FOOT = ["Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
TOUCH = ["Femur", "Tibia"] + FOOT          # leg segments that can touch the ball

# the rig (mm), the same as the viewer's
BALL_R = 4.5
BALL_MASS = 0.05456                        # flygym's ball (g-scale units used by the model)
BALL_DAMPING = 0.6                         # air-bearing drag: a spinning ball left alone slows within ~1 s
BALL_POS = (-0.45, 0.0, None)              # z is set so the standing fly's feet rest on it

# joints and muscles (flygym's defaults where they exist)
KP_REST = 12.0          # joint stiffness with both muscles relaxed (passive tissue + tonus)
KP_MAX = 60.0           # with agonist and antagonist fully co-contracted
FORCE_MAX = 65.0
JOINT_DAMPING = 0.06
PASSIVE = 1.0           # stiffness and damping of joints without muscles (head, antennae, wings)
TARSUS_K, TARSUS_D = 7.5, 0.01
ADHESION = 40.0         # tarsal pad adhesion, engaged by the claw muscles (long tendon)
FRICTION = (1.0, 0.005, 0.0001)
SOLREF = "2e-4 1e3"
SOLIMP = "0.999 0.9999 0.001 0.5 2"
TIMESTEP = 3e-4         # s; stable for anything the muscles can do, and ~1.5x faster than 0.2 ms
CHUNK = 0.004           # targets are updated this often within a frame (s)


def available():
    return mujoco is not None and MODEL_XML.exists() and BODY_BIN.exists()


def _meshes():
    """Vertices per body from the decimated meshes already shipped for the viewer (mm)."""
    meta = json.loads(BODY_JSON.read_text())
    raw = BODY_BIN.read_bytes()
    V = np.frombuffer(raw[: meta["nv"] * 12], np.float32).reshape(-1, 3)
    return {b["name"]: V[b["mesh"]["v"][0]: b["mesh"]["v"][0] + b["mesh"]["v"][1]] for b in meta["bodies"] if "mesh" in b}, meta


class PhysicsBody:
    def __init__(self, neutral):
        """neutral: {leg: {dof: angle}} resting angle of each leg DOF (the viewer's and motor.py's centre)."""
        if not available():
            raise RuntimeError("mujoco is not installed")
        self.neutral = neutral
        verts, self.meta = _meshes()
        xml = self._build_xml(verts)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.act_ids = {(leg, d): aid(f"act_{leg}{d}") for leg in LEGS for d in DOFS}
        self.qadr = {(leg, d): m.jnt_qposadr[jid(f"joint_{leg}{d}")] for leg in LEGS for d in DOFS}
        self.dadr = {(leg, d): m.jnt_dofadr[jid(f"joint_{leg}{d}")] for leg in LEGS for d in DOFS}
        self.adh_ids = {leg: aid(f"adh_{leg}") for leg in LEGS}
        self.ball_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        self.ball_dof = m.jnt_dofadr[jid("ball")]
        self.ball_q = m.jnt_qposadr[jid("ball")]
        self.leg_of_geom = {}
        for leg in LEGS:
            for seg in TOUCH:
                g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}{seg}")
                if g >= 0:
                    self.leg_of_geom[g] = (leg, seg)
        self.cost = 0.0            # seconds of compute per simulated second (smoothed)
        self.unstable = 0          # times the simulation blew up and was reset
        self.reset()

    # ------------------------------------------------------------------ model
    def _build_xml(self, verts):
        root = ET.fromstring(MODEL_XML.read_text())
        root.find("compiler").set("meshdir", ".")
        opt = root.find("option")
        opt.attrib.update({"timestep": str(TIMESTEP), "integrator": "implicitfast", "solver": "Newton",
                           "iterations": "50", "tolerance": "1e-8", "noslip_iterations": "0"})
        for k in ("ccd_iterations",):
            opt.attrib.pop(k, None)
        flag = opt.find("flag")
        if flag is not None:
            opt.remove(flag)
        size = root.find("size")
        if size is not None:
            root.remove(size)
        # meshes inline from the viewer's decimated copies (already in mm)
        for mesh in root.find("asset").findall("mesh"):
            name = mesh.get("name").replace("mesh_", "", 1)
            v = verts.get(name)
            for k in ("file", "scale", "class"):
                mesh.attrib.pop(k, None)
            mesh.set("vertex", " ".join(f"{x:.5f}" for x in v.reshape(-1)))
        # joints: muscles act on the leg DOFs; the rest are held by passive tissue
        leg_joints = {f"joint_{leg}{d}" for leg in LEGS for d in DOFS}
        for j in root.iter("joint"):
            n = j.get("name")
            j.attrib.pop("class", None)
            if n in leg_joints:
                j.set("stiffness", "0")
                j.set("damping", str(JOINT_DAMPING))
            elif "Tarsus" in n:
                j.set("stiffness", str(TARSUS_K))
                j.set("damping", str(TARSUS_D))
            else:
                j.set("stiffness", str(PASSIVE))
                j.set("damping", str(PASSIVE))
        for g in root.iter("geom"):
            g.attrib.pop("class", None)
            g.set("friction", " ".join(map(str, FRICTION)))
        wb = root.find("worldbody")
        # the ball, floating on air in its cup: free to spin, not to move
        bz = self._ball_z(root)
        ball = ET.SubElement(wb, "body", name="ball", pos=f"{BALL_POS[0]} {BALL_POS[1]} {bz}")
        ET.SubElement(ball, "joint", name="ball", type="ball", damping=str(BALL_DAMPING))
        ET.SubElement(ball, "geom", name="ball", type="sphere", size=str(BALL_R), mass=str(BALL_MASS),
                      friction="1.3 0.005 0.0001", contype="0", conaffinity="0")
        contact = ET.SubElement(root, "contact")
        for leg in LEGS:
            for seg in TOUCH:
                ET.SubElement(contact, "pair", geom1=f"{leg}{seg}", geom2="ball", solref=SOLREF, solimp=SOLIMP,
                              friction="1.15 1.15 0.005 0.0001 0.0001", margin="0")
        act = ET.SubElement(root, "actuator")
        for leg in LEGS:
            for d in DOFS:
                ET.SubElement(act, "position", name=f"act_{leg}{d}", joint=f"joint_{leg}{d}", kp=str(KP_REST),
                              forcerange=f"{-FORCE_MAX} {FORCE_MAX}", forcelimited="true", ctrlrange="-10 10")
        for leg in LEGS:
            ET.SubElement(act, "adhesion", name=f"adh_{leg}", body=f"{leg}Tarsus5", gain=str(ADHESION), ctrlrange="0 1")
        return ET.tostring(root, encoding="unicode")

    def _ball_z(self, root):
        """Place the ball so the feet of the standing fly just touch it."""
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        d = mujoco.MjData(m)
        for leg in LEGS:
            for dof in DOFS:
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{leg}{dof}")
                d.qpos[m.jnt_qposadr[j]] = self.neutral[leg][dof]
        mujoco.mj_kinematics(m, d)
        # lowest point of each leg's last tarsal segment, relative to the ball's centre line
        tips = []
        for leg in LEGS:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}Tarsus5")
            tips.append(d.xpos[b].copy())
        tips = np.array(tips)
        # z of a sphere centre (x, y fixed) touching the lowest feet: centre = foot_z - sqrt(R^2 - dx^2 - dy^2)
        zs = [p[2] - math.sqrt(max(BALL_R ** 2 - (p[0] - BALL_POS[0]) ** 2 - (p[1] - BALL_POS[1]) ** 2, 0.0)) for p in tips]
        self.feet_neutral = tips
        return float(np.median(zs)) + 0.05

    # ------------------------------------------------------------------ run
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for (leg, d), a in self.qadr.items():
            self.data.qpos[a] = self.neutral[leg][d]
        for (leg, d), i in self.act_ids.items():
            self.data.ctrl[i] = self.neutral[leg][d]
        mujoco.mj_forward(self.model, self.data)
        self._t = self._k = None
        self.ball_w = np.zeros(3)
        self.load = {leg: 0.0 for leg in LEGS}
        self.touch = {leg: 0.0 for leg in LEGS}

    def step(self, dt, target, stiffness, adhesion):
        """Advance dt seconds. target: {(leg, dof): angle}; stiffness: {(leg, dof): 0..1 co-contraction};
        adhesion: {leg: 0..1}. Targets and stiffness move smoothly from the last frame's values to these
        over the frame (muscles don't jump), in CHUNK-long pieces. Returns the sensed state."""
        m, d = self.model, self.data
        keys = list(self.act_ids)
        ids = np.array([self.act_ids[k] for k in keys])
        new_t = np.array([target.get(k, self.neutral[k[0]][k[1]]) for k in keys])
        new_k = np.array([KP_REST + (KP_MAX - KP_REST) * float(stiffness.get(k, 0.0)) for k in keys])
        old_t = self._t if self._t is not None else new_t
        old_k = self._k if self._k is not None else new_k
        for leg, i in self.adh_ids.items():
            d.ctrl[i] = float(adhesion.get(leg, 0.0))
        n = max(1, int(round(dt / m.opt.timestep)))
        chunks = max(1, min(n, int(round(dt / CHUNK))))
        bad0 = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
        per = [n // chunks + (1 if c < n % chunks else 0) for c in range(chunks)]
        t0 = time.perf_counter()
        load = {leg: 0.0 for leg in LEGS}
        touch = {leg: 0.0 for leg in LEGS}
        w = np.zeros(3)
        f6 = np.zeros(6)
        for c, k in enumerate(per):
            a = (c + 1) / chunks
            d.ctrl[ids] = old_t + a * (new_t - old_t)
            kp = old_k + a * (new_k - old_k)
            m.actuator_gainprm[ids, 0] = kp
            m.actuator_biasprm[ids, 1] = -kp
            if k:
                mujoco.mj_step(m, d, nstep=k)
            w += d.qvel[self.ball_dof: self.ball_dof + 3]
            for ci in range(d.ncon):
                con = d.contact[ci]
                g = con.geom1 if con.geom1 in self.leg_of_geom else con.geom2
                lg = self.leg_of_geom.get(g)
                if lg is None:
                    continue
                mujoco.mj_contactForce(m, d, ci, f6)
                load[lg[0]] += abs(f6[0])
                touch[lg[0]] = 1.0
        self._t, self._k = new_t, new_k
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number != bad0 or not np.isfinite(d.qpos).all():
            # a numerical blow-up (MuJoCo would restart from the model's zero pose): stand the fly
            # back up in its resting pose instead
            self.reset()
            self.unstable += 1
        el = time.perf_counter() - t0
        self.cost = 0.8 * self.cost + 0.2 * el / max(dt, 1e-6)
        self.ball_w = w / chunks
        self.load = {leg: v / chunks for leg, v in load.items()}
        self.touch = touch
        return self.sensed()

    def sensed(self):
        d = self.data
        ang = {k: float(d.qpos[a]) for k, a in self.qadr.items()}
        vel = {k: float(d.qvel[a]) for k, a in self.dadr.items()}
        # ball spin (world frame, rad/s) -> what a ball tracker reports: forward speed of the fly (mm/s,
        # + = walking forwards, i.e. the ball's top surface moving backwards) and turning (rad/s, + = left)
        wx, wy, wz = self.ball_w
        return {"angle": ang, "vel": vel, "load": dict(self.load), "touch": dict(self.touch),
                "forward": float(-wy * BALL_R), "side": float(wx * BALL_R), "yaw": float(-wz)}

    @property
    def ball_z(self):
        return float(self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")][2])

    def ball_quat(self):
        return [float(x) for x in self.data.qpos[self.ball_q: self.ball_q + 4]]
