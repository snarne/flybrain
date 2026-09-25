"""
Tools the LLM can call. File tools are confined to the workspace/ folder; run_command runs there
too and needs your approval in the chat (unless you switch on auto-approve).
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace"
MEMORY_FILE = ROOT / "data" / "memory.json"
MAX_OUT = 8000

SHELL = "PowerShell" if sys.platform == "win32" else ("zsh/bash (macOS)" if sys.platform == "darwin" else "bash")


EXPERIMENTS = {
    "threat_left": ("loom", "left"), "threat_right": ("loom", "right"), "sugar": ("sugar", None),
    "bitter": ("bitter", None), "dust": ("dust", None), "p9": ("p9", None), "mdn": ("mdn", None),
    "dna02_left": ("dna02_l", None), "dna02_right": ("dna02_r", None), "giant_fibre": ("gf", None), "adn": ("adn", None),
}
MODE_TEXT = {"rest": "resting", "legs": "moving its legs", "groom": "grooming its antennae", "feed": "feeding (proboscis out)",
             "jump": "escape jump", "fly": "flying", "skill": "doing a skill"}


def _fn(name, description, props, required=()):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": list(required)}}}


TOOLS = [
    _fn("list_files", "List files and folders in the workspace (or a subfolder of it).",
        {"path": {"type": "string", "description": "folder relative to the workspace, default '.'"}}),
    _fn("read_file", "Read a text file from the workspace. Optionally a line range.",
        {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
    _fn("write_file", "Create or overwrite a file in the workspace with the given content.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _fn("edit_file", "Replace one exact occurrence of old_text with new_text in a workspace file.",
        {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
        ["path", "old_text", "new_text"]),
    _fn("run_command", f"Run a shell command ({SHELL}) in the workspace, e.g. 'python main.py' or 'python -m pytest'. "
        "Returns exit code, stdout and stderr. `python` is the app's own Python 3.12.",
        {"command": {"type": "string"}, "timeout_s": {"type": "integer", "description": "default 60, max 600"}},
        ["command"]),
    _fn("remember", "Save a fact to long-term memory (kept across conversations).",
        {"note": {"type": "string"}}, ["note"]),
    _fn("recall", "Search long-term memory for notes related to a query.",
        {"query": {"type": "string"}}, ["query"]),
    _fn("stimulate_senses", "Use your fly body: switch one of its senses on for a few seconds and see how the "
        "simulated brain responds. Returns the behaviour readouts afterwards.",
        {"sense": {"type": "string", "description": "one of: sugar, water, bitter, touch, looming, light, motion, "
                   "fruit, geosmin, cva, co2, heat, cold, humid"},
         "seconds": {"type": "number", "description": "0.2 to 5, default 1"}}, ["sense"]),
    _fn("body_experiment", "Run an experiment on your tethered fly body (shown to the user in 3D) and find out what "
        "your brain made the body do and why: the behaviours it switched between, which muscles moved, and the "
        "neuron-by-neuron pathway traced through the connectome for each decision.",
        {"experiment": {"type": "string", "enum": list(EXPERIMENTS),
                        "description": "threat_left/threat_right: a dark object looms at one eye; sugar/bitter: a drop "
                        "touched to the proboscis; dust: dust on the antennae; p9, mdn, dna02_left, dna02_right, "
                        "giant_fibre, adn: activate those descending neurons directly (optogenetics)"}},
        ["experiment"]),
    _fn("list_skills", "List the movements your body has already learned (your motor skill cache). Check this "
        "before learning something new: a learned skill can be done instantly with do_skills.", {}),
    _fn("learn_skill", "Teach your fly body a NEW movement it has never done (e.g. tapping a key, waving a leg). "
        "You plan it as keyframes of motor controls; the engine then searches the connectome for neurons to drive "
        "(descending neurons first, then nerve-cord premotor neurons, motor neurons only as a last resort), and "
        "the fly practises a few times while the user watches, keeping the best. Slow the first time (several "
        "seconds); afterwards the skill is saved and replayed instantly. Controls (values -1..1, linear between "
        "keyframes, unlisted controls stay relaxed at 0): per leg LF LM LH RF RM RH (front/middle/hind, left/right): "
        "<leg>.swing (+ forward / - back), <leg>.lift (+ raise the femur / - press down), <leg>.reach (+ extend the "
        "tibia / - fold it), <leg>.grip (+ press the tarsus down / - raise it), <leg>.spread (+ out / - in); "
        "wingL/wingR .power (flight muscles, 0..1), .extend (hold the wing out, 0..1), .stroke (+ bigger / - smaller "
        "wingbeat); head.yaw (+ left), head.pitch (+ up), head.roll; proboscis.rostrum / .haustellum / .labellum "
        "(extend, 0..1), proboscis.pump; antennaL, antennaR. Keep movements short (0.3-2 s) and make one skill per "
        "reusable unit (e.g. one tap), then chain them with do_skills.",
        {"name": {"type": "string", "description": "short snake_case name, e.g. tap_front_left"},
         "description": {"type": "string", "description": "what the movement is, in a few words"},
         "duration": {"type": "number", "description": "seconds, 0.2-6"},
         "keyframes": {"type": "array", "description": "e.g. [{\"t\": 0, \"LF.lift\": 0}, {\"t\": 0.15, \"LF.lift\": 0.9, "
                       "\"LF.reach\": 0.5}, {\"t\": 0.35, \"LF.lift\": -0.6, \"LF.grip\": 0.8}, {\"t\": 0.5, \"LF.lift\": 0, \"LF.grip\": 0}]",
                       "items": {"type": "object"}}},
        ["name", "description", "keyframes"]),
    _fn("do_skills", "Perform learned skills on the fly body, in order (e.g. to type a word: alternate tap skills, one "
        "per letter). Instant: replays the saved neuron drive.",
        {"names": {"type": "array", "items": {"type": "string"}}, "repeat": {"type": "integer", "description": "1-20, default 1"}},
        ["names"]),
    _fn("brain_status", "Read your fly brain right now: behaviour readouts, most active regions and neurons.", {}),
]

NEEDS_APPROVAL = {"run_command"}


def _safe(path):
    p = (WORKSPACE / (path or ".")).resolve()
    if p != WORKSPACE.resolve() and WORKSPACE.resolve() not in p.parents:
        raise ValueError("path must stay inside the workspace folder")
    return p


def _clip(s):
    return s if len(s) <= MAX_OUT else s[:MAX_OUT // 2] + f"\n... [{len(s) - MAX_OUT} characters cut] ...\n" + s[-MAX_OUT // 2:]


def _load_memory():
    try:
        return json.loads(MEMORY_FILE.read_text())
    except Exception:
        return []


class ToolRunner:
    def __init__(self, world, bridge):
        self.world = world
        self.bridge = bridge
        WORKSPACE.mkdir(exist_ok=True)

    async def run(self, name, args):
        """Returns (result_text, ok: bool, extra dict for the UI)."""
        try:
            fn = getattr(self, "t_" + name)
        except AttributeError:
            return f"Unknown tool '{name}'.", False, {}
        try:
            return await fn(**args)
        except TypeError as e:
            return f"Bad arguments for {name}: {e}", False, {}
        except Exception as e:
            return f"{type(e).__name__}: {e}", False, {}

    # ------------------------------------------------------------ files
    async def t_list_files(self, path="."):
        p = _safe(path)
        if not p.exists():
            return f"{path} does not exist.", False, {}
        rows = []
        for f in sorted(p.rglob("*"))[:300]:
            if any(part.startswith(".") or part in ("__pycache__", "node_modules", ".venv") for part in f.relative_to(WORKSPACE).parts):
                continue
            rel = f.relative_to(WORKSPACE).as_posix()
            rows.append(rel + "/" if f.is_dir() else f"{rel}  ({f.stat().st_size} bytes)")
        return ("\n".join(rows) or "(empty)"), True, {}

    async def t_read_file(self, path, start_line=None, end_line=None):
        text = _safe(path).read_text(errors="replace")
        lines = text.splitlines()
        if start_line or end_line:
            s = max(1, start_line or 1)
            e = min(len(lines), end_line or len(lines))
            text = "\n".join(f"{i}: {lines[i - 1]}" for i in range(s, e + 1))
        return _clip(text), True, {}

    async def t_write_file(self, path, content):
        p = _safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        p.write_text(content)
        return f"{'Updated' if existed else 'Created'} {path} ({len(content.splitlines())} lines).", True, {"file": path}

    async def t_edit_file(self, path, old_text, new_text):
        p = _safe(path)
        text = p.read_text()
        n = text.count(old_text)
        if n != 1:
            return f"old_text found {n} times in {path}; it must match exactly once.", False, {}
        p.write_text(text.replace(old_text, new_text))
        return f"Edited {path}.", True, {"file": path}

    async def t_run_command(self, command, timeout_s=60):
        timeout_s = max(1, min(int(timeout_s or 60), 600))
        env = dict(os.environ)
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        t0 = time.time()
        if sys.platform == "win32":
            proc = await asyncio.create_subprocess_exec(
                "powershell", "-NoProfile", "-Command", command, cwd=WORKSPACE, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        else:
            proc = await asyncio.create_subprocess_shell(
                command, cwd=WORKSPACE, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout_s)
            code = proc.returncode
        except asyncio.TimeoutError:
            proc.kill()
            out, err = await proc.communicate()
            code = None
        el = time.time() - t0
        out, err = out.decode(errors="replace"), err.decode(errors="replace")
        ok = code == 0
        text = (f"exit code: {code if code is not None else f'killed after {timeout_s}s timeout'} ({el:.1f}s)\n"
                f"--- stdout ---\n{_clip(out)}\n--- stderr ---\n{_clip(err)}")
        return text, ok, {"exit_code": code, "seconds": round(el, 1)}

    # ------------------------------------------------------------ memory
    async def t_remember(self, note):
        mem = _load_memory()
        mem.append({"note": note, "t": time.time()})
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.write_text(json.dumps(mem, indent=1))
        self.bridge.memory(note, "Stored a memory")
        return f"Remembered ({len(mem)} notes in memory).", True, {}

    async def t_recall(self, query):
        mem = _load_memory()
        words = set(re.findall(r"\w{3,}", query.lower()))
        scored = sorted(((len(words & set(re.findall(r"\w{3,}", m["note"].lower()))), m) for m in mem),
                        key=lambda x: -x[0])
        hits = [m["note"] for s, m in scored if s > 0][:8]
        for h in hits[:3]:
            self.bridge.memory(h, "Recalled a memory")
        return ("\n".join(f"- {h}" for h in hits) or "Nothing related in memory."), True, {}

    # ------------------------------------------------------------ body
    async def t_stimulate_senses(self, sense, seconds=1.0):
        if sense not in self.world.stim:
            return f"Unknown sense '{sense}'. Options: {', '.join(self.world.stim)}", False, {}
        seconds = max(0.2, min(float(seconds or 1.0), 5.0))
        s = self.world.stim[sense]
        t0 = self.world.brain.t_ms
        self.bridge.pulse("tool_" + sense, s["neurons"], s["rate"], seconds, f"Agent used its body: {s['label']}",
                          s["about"])
        # watch the brain (in simulated time) and keep each readout's peak
        peak, busiest, wall0 = {}, {}, time.time()
        while self.world.brain.t_ms < t0 + seconds * 1000 + 300 and time.time() - wall0 < 120:
            await asyncio.sleep(0.1)
            st = self.world.status()
            for b in st["behaviours"]:
                peak[b["label"]] = max(peak.get(b["label"], 0), b["hz"])
            for x, r in zip(st["regions"], self.world.tables["regions"]):
                busiest[r["label"]] = max(busiest.get(r["label"], 0), x["hz_total"])
        beh = ", ".join(f"{k} {v:.0f} Hz" for k, v in peak.items() if v >= 1) or "no behaviour neurons fired"
        reg = ", ".join(f"{k} {v:.0f} spikes/s" for k, v in sorted(busiest.items(), key=lambda x: -x[1])[:3] if v > 0)
        return (f"Sense '{s['label']}' on for {seconds:.1f}s of brain time ({len(s['neurons'])} sensory neurons). "
                f"Peak behaviour readouts: {beh}. Busiest regions: {reg or 'none'}."), True, {}

    async def t_brain_status(self):
        st = self.world.status()
        beh = ", ".join(f"{b['label']} {b['hz']:.0f} Hz" for b in st["behaviours"])
        regions = sorted(zip(st["regions"], self.world.tables["regions"]), key=lambda x: -x[0]["hz_total"])[:5]
        reg = ", ".join(f"{r['label']} {x['hz_total']:.0f}" for x, r in regions)
        top = ", ".join(f"{t['type']} ({t['region']}, {t['hz']:.0f} Hz)" for t in st["top"][:6])
        return (f"{st['firing']} neurons firing. Behaviour: {beh}. Regions (spikes/s): {reg}. "
                f"Most active neurons: {top or 'none'}. Inputs on: {list(st['inputs']['presets']) or 'none'}."), True, {}

    async def t_body_experiment(self, experiment):
        body = getattr(self.world, "body", None)
        if body is None:
            return "The body isn't running.", False, {}
        if experiment not in EXPERIMENTS:
            return f"Unknown experiment '{experiment}'. Options: {', '.join(EXPERIMENTS)}", False, {}
        action, side = EXPERIMENTS[experiment]
        t0, n0, wall0 = self.world.brain.t_ms, len(body.decisions), time.time()
        body.start(action, side)
        modes, peak_rates, peak_muscles = [], {}, {}
        while self.world.brain.t_ms < t0 + 5000 and time.time() - wall0 < 60:
            await asyncio.sleep(0.05)
            if not modes or modes[-1] != body.mode:
                modes.append(body.mode)
            for k, v in body.rates.items():
                peak_rates[k] = max(peak_rates.get(k, 0), v)
            for k, v in body.parts.items():
                peak_muscles[k] = max(peak_muscles.get(k, 0), abs(v))
        seq = " -> ".join(MODE_TEXT.get(m, m) for m in modes) or "resting"
        rates = ", ".join(f"{k} {v:.0f} Hz" for k, v in sorted(peak_rates.items(), key=lambda x: -x[1]) if v >= 3) or "none"
        muscles = ", ".join(k.replace("_", " ") for k, v in sorted(peak_muscles.items(), key=lambda x: -x[1]) if v > 0.15) or "none"
        why = []
        for d in [d for d in body.decisions[n0:] if d.get("chain")]:
            chain = " -> ".join(f"{g['type']} ({g['hz']} Hz)" for g in d["chain"])
            inh = f"; opposed by {d['inhibitor']['type']}" if d.get("inhibitor") else ""
            why.append(f"- {d['title']}: {d['why']}. Pathway: {chain}{inh}")
        return (f"Experiment '{experiment}' on the tethered fly, 5 s of brain time. Body: {seq}. "
                f"Peak command/motor neuron rates: {rates}. Muscles used: {muscles}.\n"
                + ("Decisions, traced through the connectome:\n" + "\n".join(why) if why else "No behaviour change was triggered.")), True, {}

    # ------------------------------------------------------------ skills (motor skill cache)
    async def t_list_skills(self):
        eng = getattr(self.world, "skills", None)
        if eng is None:
            return "The body isn't running.", False, {}
        lst = eng.list()
        if not lst:
            return "No skills learned yet. Use learn_skill to teach one.", True, {}
        return "\n".join(f"- {s['name']}: {s['description']} ({s['duration']:.2f} s, score {s['score']:.2f}, "
                         f"driven via {', '.join(s['levels'])}, used {s['uses']}x)" for s in lst), True, {}

    async def t_learn_skill(self, name, description, keyframes, duration=None):
        eng = getattr(self.world, "skills", None)
        if eng is None:
            return "The body isn't running.", False, {}
        if name in eng.lib:
            return f"'{name}' is already learned (score {eng.lib[name]['score']}). Use do_skills, or pick another name.", True, {}
        r = await eng.learn(name, description, {"duration": duration, "keyframes": keyframes})
        if not r.get("ok"):
            return f"Couldn't learn it: {r.get('error')}", False, {}
        drv = "; ".join(f"{d['channel']} via {d['level_text']}: {', '.join(d['types'][:3])}" for d in r["drivers"])
        return (f"Learned '{name}' in {r['seconds']} s. Practice scores {r['attempts']} (best {r['score']}; 1.0 = the "
                f"muscles did exactly what you planned). Per control: {r['per_control']}. Neurons driven: {drv}. "
                f"Saved: do_skills can now replay it instantly."), True, {}

    async def t_do_skills(self, names, repeat=1):
        eng = getattr(self.world, "skills", None)
        if eng is None:
            return "The body isn't running.", False, {}
        r = await eng.perform(list(names), repeat)
        if not r.get("ok"):
            return f"Couldn't: {r.get('error')}", False, {}
        return f"Done: {', '.join(f'{n} ({s})' for n, s in r['performed'])} (numbers: how closely the muscles followed the plan).", True, {}

