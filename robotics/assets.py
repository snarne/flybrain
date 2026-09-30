"""
The robot models, from MuJoCo Menagerie (google-deepmind/mujoco_menagerie, each model under its own
licence, BSD-3 for these two), downloaded on first use into .cache/menagerie/ (kept out of git).
Pinned to one commit so every run uses the same models.
"""
import re
import sys
import urllib.request
from pathlib import Path

COMMIT = "4d038b3feae26ec82b46a4d586379114012a8ac7"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "menagerie" / COMMIT[:12]
MODELS = {"go2": ("unitree_go2", "go2.xml"), "panda": ("franka_emika_panda", "panda.xml")}


def _get(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r:
        tmp.write_bytes(r.read())
    tmp.replace(dest)


def model_path(robot):
    """Path of the robot's MJCF file, downloading it and its meshes the first time."""
    folder, xml = MODELS[robot]
    d = CACHE / folder
    main = d / xml
    done = d / ".complete"
    if done.exists():
        return main
    print(f"Downloading the {robot} model from MuJoCo Menagerie (once) ...", file=sys.stderr, flush=True)
    for f in (xml, "LICENSE"):
        _get(f"{BASE}/{folder}/{f}", d / f)
    text = main.read_text()
    m = re.search(r'meshdir="([^"]+)"', text)
    meshdir = m.group(1) if m else "."
    for f in sorted(set(re.findall(r'<mesh[^>]*file="([^"]+)"', text))):
        dest = d / meshdir / f
        if not dest.exists():
            _get(f"{BASE}/{folder}/{meshdir}/{f}", dest)
    done.write_text("ok\n")
    return main


if __name__ == "__main__":
    for r in MODELS:
        print(model_path(r))
