"""
A physical fly: NeuroMechFly (Lobato-Rios et al. 2022, Wang-Chen et al. 2024; flygym, Apache-2.0)
simulated with MuJoCo, free to walk, jump and fly in an open arena.

Nothing here decides what the fly does. Each frame:

  muscles (motor.py)  ->  each leg joint gets a target angle and a stiffness   (the muscle layer)
                          the claw muscles engage the adhesive foot pads
                          the flight muscles set wing lift, thrust and steering
  MuJoCo              ->  gravity, contacts with the ground, friction, inertia
  the body            ->  where the fly is and how it moves; joint angles, speeds, foot load,
                          which feet touch the ground and where
                      ->  leg sense organs (feedback.py) and the world's senses (arena.py)

The muscle layer is an equilibrium-point model: the agonist and antagonist of a joint pull it towards
an angle set by their difference (motor.py's mapping), and together they set how stiffly it's held.
Both muscles relaxed = a floppy joint (the leg only holds some tone); both contracted = a stiff one.

Flight: the wings aren't simulated as airfoils. While the fly is airborne and its indirect flight
muscles (DLM/DVM) are active, the wingbeat produces lift and thrust in proportion to their activation,
steering muscles tilt it into turns (stroke amplitude left vs right), and the halteres keep the body
level (a stabilising torque; halteres aren't in the neural model). Air drag slows everything.

Needs the `mujoco` package. Without it the app uses the kinematic body.
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
TOUCH = ["Tibia"] + FOOT                   # leg segments that can touch the ground
BODY_GEOMS = ["Thorax", "A1A2", "A3", "A4", "A5", "A6", "Head", "LFFemur", "LMFemur", "LHFemur",
              "RFFemur", "RMFemur", "RHFemur", "LWing", "RWing"]

ARENA = 40.0            # half-width of the square arena floor (mm)

# joints and muscles (flygym's defaults where they exist)
KP_REST = 30.0          # joint stiffness with both muscles relaxed (passive tissue + resting tone: enough to stand)
KP_MAX = 80.0           # with agonist and antagonist fully co-contracted
STEP_STIFF = 1.0        # co-contraction of a stepping leg's muscles (locomotion.py)
FORCE_MAX = 65.0
FORCE_JUMP = 65.0       # the middle legs' femur and tibia during the jump kick (the launch itself is JUMP_MAX)
JOINT_DAMPING = 0.5     # joint tissue is viscous: this also stills the contact chatter of a standing leg
PASSIVE = 1.0           # stiffness and damping of joints without muscles (head, antennae, wings)
TARSUS_K, TARSUS_D = 7.5, 0.01
ADHESION = 20.0         # tarsal pad adhesion, engaged by the claw muscles (long tendon); needed to pull (walk backwards)
FRICTION = (1.0, 0.005, 0.0001)
SOLREF = "2e-4 1e3"
SOLIMP = "0.999 0.9999 0.001 0.5 2"
TIMESTEP = 2e-4         # s (adhesion needs it this fine)
CHUNK = 0.004           # targets are updated this often within a frame (s)

# the escape jump: the tergotrochanter (jump) muscle's thrust through the middle legs, up and forward.
# In this model's resting stance the middle legs splay sideways, so extending them alone can't launch
# the fly (real flies bring them under the body before takeoff); the thrust is applied to the body while
# the jump muscle is active: ~3x body weight for ~50 ms, a takeoff speed of about 0.5-1 m/s
JUMP_MAX = 10.0         # x body weight, only while the middle legs touch the ground (a few ms): a 4-6 mm hop
JUMP_TILT = 0.25        # forward component (up is 1): a steep takeoff, as real escape jumps are

# flight (in units of the fly's weight and body lengths)
LIFT_MAX = 1.25         # lift at full flight power, x body weight (a hover needs ~0.8 power)
CEILING = 15.0          # mm: lift fades above this height (the arena's lid)
THRUST_MAX = 0.25       # forward thrust at full power, x body weight (about 5 cm/s in this small arena)
YAW_GAIN = 2.5e-3       # steering torque per unit of left-right stroke difference (x weight x mm)
DRAG = 4.0              # air drag, x weight per (m/s)... per 1000 mm/s
LEVEL_GAIN = 0.6        # haltere-mediated levelling torque (x weight x mm per rad)
ANG_DAMP = 3e-2         # rotation damping in flight (halteres sense it, steering muscles oppose it)
WALL_K = 40.0           # arena walls: push back in, x weight per mm past them
WALL_C = 40.0           # and damping, x weight per (m/s) outwards
WALL_LOOK = 10.0        # mm: how far ahead a flying fly sees a wall coming
AVOID = 4.0             # how hard it turns away (steering units)


def available():
    return mujoco is not None and MODEL_XML.exists() and BODY_BIN.exists()


def _meshes():
    """Vertices per body from the decimated meshes already shipped for the viewer (mm)."""
    meta = json.loads(BODY_JSON.read_text())
    raw = BODY_BIN.read_bytes()
    V = np.frombuffer(raw[: meta["nv"] * 12], np.float32).reshape(-1, 3)
    return {b["name"]: V[b["mesh"]["v"][0]: b["mesh"]["v"][0] + b["mesh"]["v"][1]] for b in meta["bodies"] if "mesh" in b}, meta


def _yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


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
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.act_ids = {(leg, d): aid(f"act_{leg}{d}") for leg in LEGS for d in DOFS}
        self.qadr = {(leg, d): m.jnt_qposadr[jid(f"joint_{leg}{d}")] for leg in LEGS for d in DOFS}
        self.dadr = {(leg, d): m.jnt_dofadr[jid(f"joint_{leg}{d}")] for leg in LEGS for d in DOFS}
        self.adh_ids = {leg: aid(f"adh_{leg}") for leg in LEGS}
        self.root_q = m.jnt_qposadr[jid("root")]
        self.root_v = m.jnt_dofadr[jid("root")]
        # the compiler fuses jointless bodies into their parent: thorax, abdomen, wings and halteres are
        # all part of the root body now, so forces go on it and positions are read from geoms
        self.thorax = bid("FlyBody")
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.thorax_geom = gid("Thorax")
        self.feet = {leg: bid(f"{leg}Tarsus5") for leg in LEGS}
        self.parts = {k: gid(k) for k in ("Head", "Haustellum", "LArista", "RArista", "LEye", "REye", "A6")}
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.leg_of_geom = {}
        for leg in LEGS:
            for seg in TOUCH:
                g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}{seg}")
                if g >= 0:
                    self.leg_of_geom[g] = (leg, seg)
        self.weight = float(mujoco.mj_getTotalmass(m) * -m.opt.gravity[2])
        self.cost = 0.0            # seconds of compute per simulated second (smoothed)
        self.unstable = 0          # times the simulation blew up and was reset
        self.righted = 0           # times it was set back on its feet
        self._settling = False
        self.back_t = 0.0
        self.home = (0.0, 0.0, 0.0)
        self.reset()

    # ------------------------------------------------------------------ model
    def _build_xml(self, verts):
        root = ET.fromstring(MODEL_XML.read_text())
        root.find("compiler").set("meshdir", ".")
        opt = root.find("option")
        opt.attrib.update({"timestep": str(TIMESTEP), "integrator": "implicitfast", "solver": "Newton",
                           "iterations": "50", "tolerance": "1e-8", "noslip_iterations": "0"})
        opt.attrib.pop("ccd_iterations", None)
        flag = opt.find("flag")
        if flag is not None:
            opt.remove(flag)
        size = root.find("size")
        if size is not None:
            root.remove(size)
        # meshes inline from the viewer's decimated copies (already in mm, mirrored where the model mirrors)
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
        fly = wb.find("body")                               # FlyBody: the whole fly, now free
        ET.SubElement(fly, "freejoint", name="root")
        fly.set("pos", f"0 0 {self._stand_height(root):.4f}")
        ET.SubElement(wb, "geom", name="floor", type="plane", size=f"{ARENA} {ARENA} 1", contype="0", conaffinity="0",
                      friction=" ".join(map(str, FRICTION)))
        contact = ET.SubElement(root, "contact")
        pair = dict(solref=SOLREF, solimp=SOLIMP, margin="0")
        for leg in LEGS:
            for seg in TOUCH:
                ET.SubElement(contact, "pair", geom1=f"{leg}{seg}", geom2="floor", friction="1.0 1.0 0.005 0.0001 0.0001", **pair)
        for gname in BODY_GEOMS:
            ET.SubElement(contact, "pair", geom1=gname, geom2="floor", friction="0.6 0.6 0.005 0.0001 0.0001", **pair)
        act = ET.SubElement(root, "actuator")
        for leg in LEGS:
            for d in DOFS:
                fm = FORCE_JUMP if (leg[1] == "M" and d in ("Femur", "Tibia")) else FORCE_MAX
                ET.SubElement(act, "position", name=f"act_{leg}{d}", joint=f"joint_{leg}{d}", kp=str(KP_REST),
                              forcerange=f"{-fm} {fm}", forcelimited="true", ctrlrange="-10 10")
        for leg in LEGS:
            ET.SubElement(act, "adhesion", name=f"adh_{leg}", body=f"{leg}Tarsus5", gain=str(ADHESION), ctrlrange="0 1")
        return ET.tostring(root, encoding="unicode")

    def _stand_height(self, root):
        """Height of the fly's origin at which its feet, in the resting pose, just reach the ground."""
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        d = mujoco.MjData(m)
        for leg in LEGS:
            for dof in DOFS:
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{leg}{dof}")
                d.qpos[m.jnt_qposadr[j]] = self.neutral[leg][dof]
        mujoco.mj_kinematics(m, d)
        low = min(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}Tarsus5")][2] for leg in LEGS)
        return -low + 0.03

    # ------------------------------------------------------------------ run
    def reset(self, pos=None, heading=None, settle=True):
        """Stand the fly up in its resting pose, at pos (x, y) facing heading (rad), or where it was put last.
        settle: let it settle onto its feet for 0.3 s first (not while recovering from a blow-up inside a
        step, which would recurse)."""
        if pos is not None or heading is not None:
            x, y = pos if pos is not None else self.home[:2]
            self.home = (float(x), float(y), float(heading if heading is not None else self.home[2]))
        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        q = self.root_q
        z0 = float(self.model.body_pos[self.model.jnt_bodyid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")]][2])
        x, y, h = self.home
        d.qpos[q:q + 3] = [x, y, z0]
        d.qpos[q + 3:q + 7] = [math.cos(h / 2), 0, 0, math.sin(h / 2)]
        for (leg, dof), a in self.qadr.items():
            d.qpos[a] = self.neutral[leg][dof]
        for (leg, dof), i in self.act_ids.items():
            d.ctrl[i] = self.neutral[leg][dof]
        mujoco.mj_forward(self.model, d)
        self._t = self._k = None
        self.load = {leg: 0.0 for leg in LEGS}
        self.touch = {leg: 0.0 for leg in LEGS}
        self.flight_force = np.zeros(3)
        self.back_t = 0.0
        # let it settle onto its feet before anything reads its senses (no jolt for the nerve cord)
        if settle and not self._settling:
            self._settling = True
            try:
                rest = {k: self.neutral[k[0]][k[1]] for k in self.act_ids}
                for _ in range(int(0.3 / 0.033)):
                    self.step(0.033, rest, {}, {})
            finally:
                self._settling = False
        self.cost = 0.0

    def _flight(self, flight):
        """External force and torque on the thorax from the wings (world frame)."""
        d = self.data
        power = float(flight.get("power", 0.0)) if flight else 0.0
        W = self.weight
        R = d.xmat[self.thorax].reshape(3, 3)          # body -> world
        fwd, up = R[:, 0], R[:, 2]
        v = d.qvel[self.root_v: self.root_v + 3]
        w = d.qvel[self.root_v + 3: self.root_v + 6]
        f = np.zeros(3)
        t = np.zeros(3)
        jump = float(flight.get("jump", 0.0)) if flight else 0.0
        # the legs can only push while they're on the ground: once the middle legs are off, the jump is over
        if not (self.touch.get("LM") or self.touch.get("RM")):
            jump = 0.0
        if jump > 0.1:
            dirn = up + JUMP_TILT * fwd
            f += JUMP_MAX * jump * W * dirn / np.linalg.norm(dirn)
        airborne = sum(self.touch.values()) == 0
        com = d.subtree_com[self.thorax]
        if power > 0.05:
            ceil = min(1.0, max(0.0, (CEILING - float(com[2])) / 3.0))
            f += LIFT_MAX * power * ceil * W * up + THRUST_MAX * power * W * fwd
            steer = float(flight.get("steer", 0.0))
            # flying flies turn away from a wall that looms ahead (visual collision avoidance)
            ahead = com[:2] + WALL_LOOK * fwd[:2] / max(float(np.linalg.norm(fwd[:2])), 1e-6)
            if np.abs(ahead).max() > ARENA - 2.0:
                centre = -com[:2]
                side = fwd[0] * centre[1] - fwd[1] * centre[0]          # + = the centre is to the left
                steer += AVOID if side >= 0 else -AVOID
            t += YAW_GAIN * W * steer * power * np.array([0, 0, 1.0])
        if power > 0.05 or airborne or jump > 0.1:
            # halteres: keep the body level (torque towards up = world z) and damp rotation; they beat with
            # the wings, which start at takeoff
            t += LEVEL_GAIN * W * np.cross(up, np.array([0, 0, 1.0]))
            t -= ANG_DAMP * (R @ w)                  # (the free joint's angular velocity is in the body frame)
        if airborne:
            f -= DRAG * W * v / 1000.0
        # the arena's walls: a stiff, damped push back in (the legs may touch the wall line)
        for i in range(2):
            over = abs(float(com[i])) - (ARENA - 0.8)
            if over > 0:
                sgn = 1.0 if com[i] > 0 else -1.0
                f[i] -= sgn * (WALL_K * over * W + WALL_C * W * max(0.0, sgn * float(v[i])) / 1000.0)
        self.flight_force = f
        # apply at the whole fly's centre of mass (the root body's own centre is off to one side, and a
        # push there would spin the fly)
        arm = d.subtree_com[self.thorax] - d.xipos[self.thorax]
        d.xfrc_applied[self.thorax, :3] = f
        d.xfrc_applied[self.thorax, 3:] = t + np.cross(arm, f)

    def step(self, dt, target, stiffness, adhesion, flight=None, gait=None):
        """Advance dt seconds. target: {(leg, dof): angle}; stiffness: {(leg, dof): 0..1 co-contraction};
        adhesion: {leg: 0..1}; flight: {"power": 0..1, "steer": -1..1 (+ = left), "jump": 0..1}. Targets and stiffness
        move smoothly from the last frame's values to these over the frame (muscles don't jump), in
        CHUNK-long pieces. gait: optional callable(dt) -> (weight, {(leg, dof): angle}, {leg: adhesion}) from
        the walking oscillators (locomotion.py), blended in over the muscles' targets each chunk.
        Returns the sensed state."""
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
        f6 = np.zeros(6)
        adh_base = {leg: float(adhesion.get(leg, 0.0)) for leg in self.adh_ids}
        for c, k in enumerate(per):
            a = (c + 1) / chunks
            tgt = old_t + a * (new_t - old_t)
            kp = old_k + a * (new_k - old_k)
            if gait is not None:
                w, gt, gpad = gait(k * m.opt.timestep)
                if w > 0:
                    g = np.array([gt.get(key, t) for key, t in zip(keys, tgt)])
                    tgt = (1 - w) * tgt + w * g
                    kp = np.maximum(kp, KP_REST + (KP_MAX - KP_REST) * w * STEP_STIFF)
                    for leg, i in self.adh_ids.items():
                        d.ctrl[i] = (1 - w) * adh_base[leg] + w * gpad.get(leg, 0.0)
            d.ctrl[ids] = tgt
            m.actuator_gainprm[ids, 0] = kp
            m.actuator_biasprm[ids, 1] = -kp
            self._flight(flight)
            if k:
                mujoco.mj_step(m, d, nstep=k)
            now = {leg: 0.0 for leg in LEGS}
            for ci in range(d.ncon):
                con = d.contact[ci]
                g = con.geom1 if con.geom1 in self.leg_of_geom else con.geom2
                lg = self.leg_of_geom.get(g)
                if lg is None:
                    continue
                mujoco.mj_contactForce(m, d, ci, f6)
                load[lg[0]] += abs(f6[0])
                now[lg[0]] = 1.0
                touch[lg[0]] = 1.0
            self.touch = now
        self._t, self._k = new_t, new_k
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number != bad0 or not np.isfinite(d.qpos).all():
            # a numerical blow-up (MuJoCo would restart from the model's zero pose): stand the fly
            # back up where it was, in its resting pose
            x, y = float(d.qpos[self.root_q]), float(d.qpos[self.root_q + 1])
            self.reset(pos=(x, y) if np.isfinite([x, y]).all() else None, settle=False)
            self.unstable += 1
            return self.sensed()
        # a fly that lands on its back rights itself with its legs; this model can't make those movements,
        # so after a second on its back on the ground it is set back on its feet where it lies
        R = d.xmat[self.thorax].reshape(3, 3)
        on_back = R[2, 2] < -0.2 and d.qvel[self.root_v + 2] ** 2 < 25.0
        self.back_t = self.back_t + dt if on_back else 0.0
        if self.back_t > 1.0:
            self.reset(pos=(float(d.qpos[self.root_q]), float(d.qpos[self.root_q + 1])),
                       heading=_yaw_of(d.qpos[self.root_q + 3:self.root_q + 7]), settle=False)
            self.righted += 1
            return self.sensed()
        el = time.perf_counter() - t0
        self.cost = 0.8 * self.cost + 0.2 * el / max(dt, 1e-6)
        self.load = {leg: v / chunks for leg, v in load.items()}
        self.touch = touch
        return self.sensed()

    def sensed(self):
        d = self.data
        ang = {k: float(d.qpos[a]) for k, a in self.qadr.items()}
        vel = {k: float(d.qvel[a]) for k, a in self.dadr.items()}
        q = d.qpos[self.root_q: self.root_q + 7]
        heading = _yaw_of(q[3:7])
        v = d.qvel[self.root_v: self.root_v + 3]
        w = d.qvel[self.root_v + 3: self.root_v + 6]
        R = d.xmat[self.thorax].reshape(3, 3)
        c, s = math.cos(heading), math.sin(heading)
        fwd = float(v[0] * c + v[1] * s)            # along the heading, mm/s
        side = float(-v[0] * s + v[1] * c)          # + = to the fly's left
        xy = lambda b: [round(float(d.xpos[b][0]), 3), round(float(d.xpos[b][1]), 3), round(float(d.xpos[b][2]), 3)]
        gp = lambda g: [round(float(d.geom_xpos[g][0]), 3), round(float(d.geom_xpos[g][1]), 3), round(float(d.geom_xpos[g][2]), 3)]
        return {"angle": ang, "vel": vel, "load": dict(self.load), "touch": dict(self.touch),
                "pos": [float(q[0]), float(q[1]), float(q[2])], "quat": [float(x) for x in q[3:7]],
                "thorax": gp(self.thorax_geom), "heading": heading,
                "forward": fwd, "side": side, "yaw": float(w[2]), "climb": float(v[2]),
                "upright": float(R[2, 2]),
                "feet": {leg: xy(b) for leg, b in self.feet.items()},
                "parts": {k: gp(g) for k, g in self.parts.items()}}
