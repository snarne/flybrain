"""
Build the 3D fly body for the viewer from NeuroMechFly (flygym, Apache-2.0,
Lobato-Rios et al. 2022, Wang-Chen et al. 2024).

    python tools/build_body.py path/to/flygym   # the unpacked flygym package folder

Writes web/data/fly_body.bin (decimated meshes, mm) and web/data/fly_body.json
(skeleton: bodies, joints, mesh offsets; recorded single-step leg kinematics; a standing pose).
"""
import json
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import fast_simplification

ROOT = Path(__file__).resolve().parents[1]
FG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("flygym")
DATA = FG / "data" if (FG / "data").exists() else FG / "flygym" / "data"
OUT = ROOT / "web" / "data"

xml = ET.parse(DATA / "mjcf" / "neuromechfly_seqik_kinorder_ypr.xml").getroot()
mesh_files = {m.get("name"): (DATA / "mjcf" / m.get("file")).resolve() for m in xml.find("asset").findall("mesh")}


def floats(s, default):
    return [float(x) for x in s.split()] if s else default


bodies, verts, faces = [], [], []
vo = fo = 0


def add_body(b, parent):
    global vo, fo
    name = b.get("name")
    geoms = [g for g in b.findall("geom") if g.get("mesh")]
    entry = {"name": name, "parent": parent, "pos": floats(b.get("pos"), [0, 0, 0]),
             "quat": floats(b.get("quat"), [1, 0, 0, 0]),
             "joints": [{"name": j.get("name"), "axis": floats(j.get("axis"), [0, 0, 1])} for j in b.findall("joint")]}
    if geoms:
        g = geoms[0]
        m = trimesh.load(mesh_files[g.get("mesh")], force="mesh")
        v = np.asarray(m.vertices, np.float64) * 1000.0  # metres -> mm
        f = np.asarray(m.faces, np.int64)
        n = len(f)
        target = int(min(7000, max(250, n * 0.12)))
        if n > target:
            v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), 1.0 - target / n)
        entry["mesh"] = {"v": [vo, len(v)], "f": [fo, len(f)],
                         "pos": floats(g.get("pos"), [0, 0, 0]), "quat": floats(g.get("quat"), [1, 0, 0, 0])}
        verts.append(np.asarray(v, np.float32))
        faces.append(np.asarray(f, np.uint32))
        vo += len(v)
        fo += len(f)
    bodies.append(entry)
    for c in b.findall("body"):
        add_body(c, name)


for b in xml.find("worldbody").findall("body"):
    add_body(b, None)

V = np.concatenate(verts)
F = np.concatenate(faces)
with open(OUT / "fly_body.bin", "wb") as fh:
    fh.write(V.tobytes())
    fh.write(F.tobytes())

steps = pickle.load(open(DATA / "behavior" / "single_steps_untethered.pkl", "rb"))
legs = [f"{s}{p}" for s in "LR" for p in "FMH"]
dofs = ["Coxa", "Coxa_roll", "Coxa_yaw", "Femur", "Femur_roll", "Tibia", "Tarsus1"]
step_data = {leg: {d: [round(float(x), 5) for x in steps[f"joint_{leg}{d}"]] for d in dofs} for leg in legs}
swing_stance = {k: {leg: float(v) for leg, v in d.items()} for k, d in steps["swing_stance_time"].items()}

json.dump({
    "source": "NeuroMechFly v2 / flygym (Apache-2.0)",
    "nv": int(len(V)), "nf": int(len(F)),
    "bodies": bodies,
    "steps": {"dt": steps["meta"]["timestep"], "n": len(steps["joint_LFCoxa"]), "legs": step_data,
              "swing_stance": swing_stance},
}, open(OUT / "fly_body.json", "w"))
print(f"{len(bodies)} bodies, {len(V)} vertices, {len(F)} faces -> {OUT}")
