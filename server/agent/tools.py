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
BENCH_DIR = ROOT / "data" / "benchmark" / "agent"
MAX_OUT = 8000

SHELL = "PowerShell" if sys.platform == "win32" else ("zsh/bash (macOS)" if sys.platform == "darwin" else "bash")


EXPERIMENTS = {
    # things in the fly's world (it senses them from where they are; natural behaviour)
    "threat_left": ("loom", "left"), "threat_right": ("loom", "right"), "sugar_drop": ("sugar", None),
    "bitter_drop": ("bitter", None), "fruit_odour": ("fruit", None), "co2": ("co2", None), "geosmin": ("geosmin", None),
    "heat_zone": ("heat", None), "cold_zone": ("cold", None), "humid_zone": ("humid", None), "dry_zone": ("dry", None),
    "stripes_left": ("stripes", "left"), "stripes_right": ("stripes", "right"),
    "target_left": ("target", "left"), "target_right": ("target", "right"),
    # what an experimenter does to the fly
    "sugar": ("probe_sugar", None), "bitter": ("probe_bitter", None), "dust": ("dust", None),
    # activating descending neurons directly (optogenetics)
    "p9": ("p9", None), "mdn": ("mdn", None), "dna02_left": ("dna02_l", None), "dna02_right": ("dna02_r", None),
    "giant_fibre": ("gf", None), "adn": ("adn", None),
}
MODE_TEXT = {"rest": "resting", "legs": "moving its legs", "walk": "walking", "groom": "grooming its antennae",
             "feed": "feeding (proboscis out)", "jump": "escape jump", "fly": "flying", "skill": "doing a skill",
             "reflex": "grooming (built-in movement)"}


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
    _fn("body_experiment", "Run an experiment on your fly body, which walks freely in an arena (shown to the user in "
        "3D), and find out what your brain made the body do and why: the behaviours it switched between, how it moved, "
        "which muscles worked, and the neuron-by-neuron pathway traced through the connectome for each decision.",
        {"experiment": {"type": "string", "enum": list(EXPERIMENTS),
                        "description": "Things put in the world near the fly, sensed from where they are: threat_left/"
                        "threat_right (a dark object swooping from that side), sugar_drop/bitter_drop (under its front "
                        "feet), fruit_odour, co2, geosmin (an odour source ahead), heat_zone, cold_zone, humid_zone, "
                        "dry_zone (air around it), stripes_left/stripes_right (a turning striped drum), target_left/target_right "
                        "(a small dark object darting back and forth ahead to that side). What an "
                        "experimenter does: sugar/bitter (a drop touched to the proboscis), dust (on the antennae). "
                        "Optogenetics: p9, mdn, dna02_left, dna02_right, giant_fibre, adn (activate those descending "
                        "neurons directly)."},
         "record": {"type": "array", "items": {"type": "string"}, "description": "optional: cell types whose firing "
                    "rates to record during the experiment (mean and peak Hz)"}},
        ["experiment"]),
    # ---- neuroscientist tools: read the wiring, intervene, measure, compare
    _fn("find_neurons", "Search the BANC connectome's cell types by name, class or super class (e.g. 'LPLC2', 'DNa', "
        "'kenyon', 'descending', 'gustatory'). Returns each type's count, class, transmitter, regions, sides and how "
        "active it is right now.", {"query": {"type": "string"}}, ["query"]),
    _fn("neuron_info", "Everything about one cell type (or a neuron index / BANC root id): what it is, its transmitter, "
        "where it is, how fast it fires now, its strongest input and output cell types (synapses, excites/inhibits, share), "
        "and what the app uses it for (behaviour readout, sense, motor pool).", {"name": {"type": "string"}}, ["name"]),
    _fn("trace_path", "The strongest chain of cell types in the connectome from one cell type to another (each step "
        "weighted by how much of the next type's input it provides), with synapse counts and signs. Use it to form a "
        "hypothesis about which neurons carry a signal; then test it by silencing them.",
        {"from_type": {"type": "string"}, "to_type": {"type": "string"}}, ["from_type", "to_type"]),
    _fn("activate_neurons", "Activate every neuron of a cell type (optogenetics-style, Poisson spikes) for a few seconds "
        "and watch what the fly does, optionally recording the firing rates of other cell types meanwhile.",
        {"cell_type": {"type": "string"}, "seconds": {"type": "number", "description": "0.2-5, default 2"},
         "rate": {"type": "number", "description": "Hz, default 150"},
         "record": {"type": "array", "items": {"type": "string"}, "description": "cell types to record (mean and peak Hz)"}},
        ["cell_type"]),
    _fn("silence_neurons", "Silence a group of neurons (they can't fire) until restored: an ablation that stays through "
        "experiments and resets. Groups: vision, mushroom_body, descending, giant_fibre, antennal_lobe, lateral_horn; or "
        "any cell type name. restore=true undoes it. With no group, lists what is silenced.",
        {"group": {"type": "string"}, "restore": {"type": "boolean"}}),
    _fn("run_benchmark", "Score the fly against published experiments in a separate copy of the model (takes a minute "
        "or more; the live fly keeps running). Choose experiments (default: all), silence groups (ablate), or run on a "
        "control wiring (variant: shuffled = degree-preserving random rewiring, no_inhibition). Saved under a label so "
        "you can compare runs.",
        {"experiments": {"type": "array", "items": {"type": "string"}, "description": "benchmark keys, e.g. loom_left, "
                         "p9_forward, sugar_mouth, fruit_approach, learn_punish; empty = all"},
         "ablate": {"type": "array", "items": {"type": "string"}},
         "variant": {"type": "string", "enum": ["real", "shuffled", "no_inhibition"]},
         "repeats": {"type": "integer", "description": "1-3, default 1"},
         "label": {"type": "string", "description": "short name for this run"}}),
    _fn("compare_runs", "Compare saved benchmark runs (by label) experiment by experiment. Without labels, lists the "
        "saved runs.", {"labels": {"type": "array", "items": {"type": "string"}}}),
    _fn("list_skills", "List the movements your body has already learned (your motor skill cache). Check this "
        "before learning something new: a learned skill can be done instantly with do_skills.", {}),
    _fn("learn_skill", "Teach your fly body a NEW movement. You plan it as keyframes of motor controls (read "
        "motor_reference first) and, when the movement has a purpose, a goal (what it should achieve). The engine "
        "searches the connectome for neurons to drive (descending neurons first, then nerve-cord premotor neurons, motor "
        "neurons last), shapes the drive until the muscles follow your plan, then practises many variations in background "
        "copies of the fly (the live fly keeps running) and keeps the one that scores best on the goal, checked again on "
        "fresh noise. Hand-built parts (walking model, flight timer) are off in practice, so the result is the neurons' "
        "and muscles' own work; give a baseline (e.g. walk_forward) to compare. Saved afterwards and shown once on the "
        "live fly. Takes about a minute or more (more trials, longer). Controls (values -1..1, linear between keyframes, "
        "unlisted stay relaxed at 0): per leg LF LM LH RF RM RH: <leg>.swing, .lift, .reach, .grip, .spread, .twist, "
        "and LM.jump / RM.jump (the jump muscle); wingL/wingR .power, .extend, .stroke; head.yaw/.pitch/.roll; "
        "proboscis.rostrum/.haustellum/.labellum/.pump; antennaL/R; haltereL/R; abdomen.curl/.bend.",
        {"name": {"type": "string", "description": "short snake_case name, e.g. tap_front_left"},
         "description": {"type": "string", "description": "what the movement is, in a few words"},
         "duration": {"type": "number", "description": "seconds, 0.2-6"},
         "keyframes": {"type": "array", "description": "e.g. [{\"t\": 0, \"LF.lift\": 0}, {\"t\": 0.15, \"LF.lift\": 0.9, "
                       "\"LF.reach\": 0.5}, {\"t\": 0.35, \"LF.lift\": -0.6, \"LF.grip\": 0.8}, {\"t\": 0.5, \"LF.lift\": 0, \"LF.grip\": 0}]",
                       "items": {"type": "object"}},
         "goal": {"anyOf": [{"type": "string"}, {"type": "object"}], "description": "what the movement should achieve: a preset (forward, backward, turn_left, turn_right, "
                  "jump_high, jump_far, fly, still, match_plan) or {metric: weight}, e.g. {\"forward_mm\": 1, "
                  "\"wing_power\": -2} or {\"turn_deg\": {\"target\": 90}}. Metrics: forward_mm, side_mm, distance_mm, "
                  "speed_mm_s, turn_deg, height_mm, airtime_s, upright, fell, wing_power, jump, unwanted, tracking, effort. "
                  "Without a goal the engine only makes the muscles follow the plan."},
         "trials": {"type": "integer", "description": "practice tries for the goal, default 40 (more = better, slower)"},
         "baseline": {"type": "string", "description": "a skill to measure the same way for comparison, e.g. walk_forward"},
         "assists": {"type": "object", "description": "hand-built parts to allow in practice, e.g. {\"flight_timer\": true}; "
                     "default all off"},
         "gait": {"type": "object", "description": "instead of keyframes, a walking plan made from a real fly's recorded "
                  "steps (all joints of all six legs, tripod gait): {\"steps_per_s\": 5, \"duration\": 2, \"backwards\": false, "
                  "\"turn\": 0} (turn -1..1, + = left)"},
         "drive_at": {"type": "string", "enum": ["auto", "nerve_cord", "motor"], "description": "where the drive goes in: "
                      "auto (default: descending neurons first, the brain's own routes), nerve_cord (premotor and motor "
                      "neurons), motor (motor neurons directly: the most precise, but it bypasses the wiring above them, "
                      "so say so when you report it)"}},
        ["name", "description"]),
    _fn("motor_reference", "Everything you need to plan a movement: every motor control and what it moves, joint ranges "
        "(degrees), real fly timing, the physics and the body model's simplifications, the hand-built parts, and the goal "
        "metrics and presets for practice. Read it before planning a new skill.", {}),
    _fn("refine_skill", "Improve one of your learned skills: practise it more for its goal (no keyframes: continue from "
        "the saved drive), or for a new goal, or from a corrected plan (keyframes: keeps the neurons already found). The "
        "skill as it was competes with the new versions; the best is kept.",
        {"name": {"type": "string"}, "keyframes": {"type": "array", "items": {"type": "object"}},
         "duration": {"type": "number"}, "description": {"type": "string"}, "goal": {"anyOf": [{"type": "string"}, {"type": "object"}], "description": "as in learn_skill"},
         "trials": {"type": "integer"}, "baseline": {"type": "string"}, "assists": {"type": "object"},
         "drive_at": {"type": "string", "enum": ["auto", "nerve_cord", "motor"], "description": "with keyframes: where "
                      "the drive goes in (as in learn_skill)"}}, ["name"]),
    _fn("test_skill", "Measure what a skill does, in background copies of the fly (the live fly isn't touched): how far "
        "it moved and turned, height, time in the air, whether it fell, which muscles it used, averaged over a few runs. "
        "Use it to compare skills (a learned one against a built-in one), or one skill with and without a hand-built part "
        "(assists), or under a different goal.",
        {"name": {"type": "string"}, "assists": {"type": "object", "description": "e.g. {\"walking\": false}"},
         "goal": {"anyOf": [{"type": "string"}, {"type": "object"}], "description": "as in learn_skill; default the skill's own"}, "runs": {"type": "integer", "description": "1-10, default 3"},
         "duration": {"type": "number", "description": "seconds, for built-in walking/turning (default 2)"}}, ["name"]),
    _fn("combine_skills", "Save a sequence of skills under one name, e.g. walk forward for 4 s then jump: "
        "steps=[{\"skill\": \"walk_forward\", \"repeat\": 4}, {\"skill\": \"hop\"}] (walk_forward is 1 s; repeats of a "
        "walking skill run as one continuous walk). Nothing new is learned: learn the parts first (a jump with "
        "learn_skill), then combine them; do_skills plays the combination.",
        {"name": {"type": "string"}, "description": {"type": "string"},
         "steps": {"type": "array", "items": {"type": "object"}, "description": "[{\"skill\": name, \"repeat\": n}, ...]"}},
        ["name", "description", "steps"]),
    _fn("do_skills", "Perform learned skills on the live fly body, in order (e.g. to type a word: alternate tap skills, "
        "one per letter). Instant: replays the saved neuron drive, with the hand-built parts as they were when the skill "
        "was learned.",
        {"names": {"type": "array", "items": {"type": "string"}}, "repeat": {"type": "integer", "description": "1-20, default 1"}},
        ["names"]),
    _fn("body_parts", "See or switch the live body's hand-built parts: walking (the walking model that steps the legs "
        "when the walking command neurons fire) and flight_timer (a jump starts flight and power is held at cruise). "
        "With no arguments, says what each is and whether it's on.",
        {"walking": {"type": "boolean"}, "flight_timer": {"type": "boolean"}}),
    _fn("guide", "Read the guide to this system: how the brain, muscles, body and practice work, what is real "
        "connectome and what is hand-built, what each tool is for, and the Python helper (flylab) and HTTP API for "
        "scripts. Topics: overview, body, skills, neuroscience, scripts, limits. Read 'overview' when you start on a task "
        "about the fly, and 'scripts' before writing code that talks to it.",
        {"topic": {"type": "string", "enum": ["overview", "body", "skills", "neuroscience", "scripts", "limits"]}}, ["topic"]),
    _fn("api_get", "Read from the fly's local HTTP API (read-only GET, no approval needed), e.g. /api/status, "
        "/api/body, /api/skills, /api/skill/<name>, /api/controls, /api/rates?types=DNp01,MDN, /api/typeinfo/<type>, "
        "/api/path?src=LC4&dst=DNp01, /api/search?q=<text>. See guide('scripts').",
        {"path": {"type": "string"}}, ["path"]),
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



def _ball_text(b):
    """The physical result of a movement: how the fly moved over the ground."""
    if not b:
        return ""
    return (f"Movement: {b['forward_mm_s']:+.2f} mm/s forward on average (peak {b['peak_forward_mm_s']:+.2f}), "
            f"turning {b['turn_left_rad_s']:+.2f} rad/s (+ = left). ")

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

    async def _watch(self, seconds, record=(), label=""):
        """Let the live fly run for `seconds` of brain time; summarise what the body did (and the recorded
        cell types' firing)."""
        body = self.world.body
        rec = {}
        for t in record or ():
            name, idx = self._atlas().resolve(t)
            if name is not None:
                rec[name] = idx
        t0, n0, wall0 = self.world.brain.t_ms, len(body.decisions), time.time()
        modes, peak_rates, peak_muscles, ball = [], {}, {}, []
        rsum = {k: 0.0 for k in rec}
        rpeak = {k: 0.0 for k in rec}
        nsamp = 0
        while self.world.brain.t_ms < t0 + seconds * 1000 and time.time() - wall0 < 12 * seconds + 10:
            await asyncio.sleep(0.05)
            ball.append(tuple(body.ball))
            if not modes or modes[-1] != body.mode:
                modes.append(body.mode)
            for k, v in body.rates.items():
                peak_rates[k] = max(peak_rates.get(k, 0), v)
            for k, v in body.parts.items():
                peak_muscles[k] = max(peak_muscles.get(k, 0), abs(v))
            nsamp += 1
            for k, idx in rec.items():
                hz = float(self.world.rate[idx].mean())
                rsum[k] += hz
                rpeak[k] = max(rpeak[k], hz)
        seq = " -> ".join(MODE_TEXT.get(m, m) for m in modes) or "resting"
        rates = ", ".join(f"{k} {v:.0f} Hz" for k, v in sorted(peak_rates.items(), key=lambda x: -x[1]) if v >= 3) or "none"
        muscles = ", ".join(k.replace("_", " ") for k, v in sorted(peak_muscles.items(), key=lambda x: -x[1]) if v > 0.15) or "none"
        why = []
        for d in [d for d in body.decisions[n0:] if d.get("chain")]:
            chain = " -> ".join(f"{g['type']} ({g['hz']} Hz)" for g in d["chain"])
            inh = f"; opposed by {d['inhibitor']['type']}" if d.get("inhibitor") else ""
            why.append(f"- {d['title']}: {d['why']}. Pathway: {chain}{inh}")
        fwd = sum(b[0] for b in ball) / max(len(ball), 1)
        turn = -sum(b[1] for b in ball) / max(len(ball), 1)
        kind = "physical body" if getattr(body, "phys", None) is not None else "kinematic body (movement estimated)"
        felt = ", ".join(sorted(getattr(body, "senses", {}) or {})) or "nothing more"
        recorded = ""
        if rec:
            recorded = "Recorded firing (mean / peak over the run): " + ", ".join(
                f"{k} {rsum[k] / max(nsamp, 1):.1f} / {rpeak[k]:.1f} Hz" for k in rec) + ".\n"
        silenced = f" Silenced: {', '.join(self.world.ablated)}." if self.world.ablated else ""
        return (f"{label} ({kind}, leg rhythm: {getattr(body, 'rhythm', '?')}, wiring: {self.world.variant}), "
                f"{seconds:g} s of brain time.{silenced} Body: {seq}. Movement: {fwd:+.1f} mm/s forward on average, "
                f"turning {turn:+.2f} rad/s (+ = left). Still sensing at the end: {felt}. "
                f"Peak command/motor neuron rates: {rates}. Muscles used: {muscles}.\n" + recorded
                + ("Decisions, traced through the connectome:\n" + "\n".join(why) if why else "No behaviour change was triggered."))

    async def t_body_experiment(self, experiment, record=None):
        body = getattr(self.world, "body", None)
        if body is None:
            return "The body isn't running.", False, {}
        if experiment not in EXPERIMENTS:
            return f"Unknown experiment '{experiment}'. Options: {', '.join(EXPERIMENTS)}", False, {}
        action, side = EXPERIMENTS[experiment]
        body.start(action, side)
        return await self._watch(5.0, record, f"Experiment '{experiment}' on the fly"), True, {}

    # ------------------------------------------------------------ neuroscientist tools
    def _atlas(self):
        if getattr(self.world, "atlas", None) is None:
            from neuroscience import Atlas
            self.world.atlas = Atlas(self.world)
        return self.world.atlas

    async def t_find_neurons(self, query):
        hits = await asyncio.to_thread(self._atlas().find, query)
        if not hits:
            return f"No cell types match '{query}'.", True, {}
        return "\n".join(f"- {h['type']}: {h['n']} neurons, {h['class'] or h['super_class'] or 'unclassified'}, "
                         f"{h['transmitter'] or 'transmitter unknown'}, in {', '.join(h['regions'])}; "
                         f"sides {h['sides']}; now {h['hz_now']} Hz (max {h['hz_max']})" for h in hits), True, {}

    async def t_neuron_info(self, name):
        d = await asyncio.to_thread(self._atlas().info, name)
        if d is None:
            return f"No cell type or neuron called '{name}'. Try find_neurons.", False, {}
        part = lambda xs: "; ".join(f"{x['type']} {x['sign']} ({x['synapses']} syn, {100 * x['share']:.0f}%, now {x['hz_now']} Hz)"
                                    for x in xs) or "none"
        return (f"{d['type']}: {d['n']} neurons, class {d['class'] or '-'} ({d['super_class'] or '-'}), transmitter "
                f"{d['transmitter'] or 'unknown'}, in {', '.join(d['regions'])}, sides {d['sides']}. Firing now: mean "
                f"{d['hz_now']} Hz, max {d['hz_max']} Hz.\nInputs: {part(d['inputs'])}\nOutputs: {part(d['outputs'])}\n"
                f"Used by the app as: {', '.join(d['roles']) or 'nothing in particular'}"), True, {}

    async def t_trace_path(self, from_type, to_type):
        p = await asyncio.to_thread(self._atlas().path, from_type, to_type)
        if p.get("error"):
            return p["error"], False, {}
        if not p["path"]:
            return f"No path from {from_type} to {to_type} through named cell types.", True, {}
        steps = "\n".join(f"  {i + 1}. {s['from']} {s['sign']} {s['to']}: {s['synapses']} synapses, "
                          f"{100 * s['share_of_input']:.1f}% of {s['to']}'s input" for i, s in enumerate(p["path"]))
        return (f"Strongest path {p['from']} -> {p['to']}: {p['hops']} steps, net {p['net']}. {p['note']}\n{steps}"), True, {}

    async def t_activate_neurons(self, cell_type, seconds=2.0, rate=150.0, record=None):
        if getattr(self.world, "body", None) is None:
            return "The body isn't running.", False, {}
        name, idx = self._atlas().resolve(cell_type)
        if name is None:
            return f"No cell type called '{cell_type}'. Try find_neurons.", False, {}
        seconds = max(0.2, min(float(seconds or 2.0), 5.0))
        rate = max(1.0, min(float(rate or 150.0), 300.0))
        w = self.world
        w.pending.append({"cmd": "channel", "name": "agent_activate", "neurons": [int(i) for i in idx], "rate": rate})
        t_off = w.brain.t_ms + seconds * 1000.0

        async def switch_off():
            wall0 = time.time()
            while w.brain.t_ms < t_off and time.time() - wall0 < 12 * seconds + 10:
                await asyncio.sleep(0.02)
            w.pending.append({"cmd": "channel", "name": "agent_activate", "rate": 0})
        off = asyncio.create_task(switch_off())
        try:
            text = await self._watch(seconds + 0.5, record, f"Activated {name} ({len(idx)} neurons, {rate:.0f} Hz, {seconds:g} s)")
        finally:
            await off
        return text, True, {}

    async def t_silence_neurons(self, group=None, restore=False):
        w = self.world
        if not group:
            now = ", ".join(f"{k} ({len(v)} neurons)" for k, v in w.ablated.items()) or "nothing"
            return (f"Silenced: {now}. Groups: " + "; ".join(f"{k}: {v}" for k, v in w.ABLATIONS.items())
                    + ". Or any cell type name."), True, {}
        n = w.ablate(group, on=not restore)
        if not n:
            return f"Nothing called '{group}' to {'restore' if restore else 'silence'}.", False, {}
        return f"{'Restored' if restore else 'Silenced'} {group}: {n} neurons.", True, {}

    async def t_run_benchmark(self, experiments=None, ablate=None, variant="real", repeats=1, label=None):
        label = re.sub(r"[^\w-]", "_", label or f"run_{time.strftime('%H%M%S')}")[:40]
        out = BENCH_DIR / f"{label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "tools" / "benchmark.py"), "--repeats", str(max(1, min(int(repeats or 1), 3))),
               "--variant", variant or "real", "--out", str(out)]
        body = getattr(self.world, "body", None)
        if body is not None:
            cmd += ["--body", "physics" if body.phys is not None else "kinematic", "--rhythm", body.rhythm,
                    "--flight-timer", "on" if body.assists.get("flight_timer", True) else "off"]
        if experiments:
            cmd += ["--only", ",".join(experiments)]
        if ablate:
            cmd += ["--ablate", ",".join(ablate)]
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(ROOT), stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT)
        try:
            so, _ = await asyncio.wait_for(proc.communicate(), timeout=1800)
        except asyncio.TimeoutError:
            proc.kill()
            return "The benchmark took over 30 minutes and was stopped. Try fewer experiments.", False, {}
        text = so.decode(errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip() and "Warning" not in ln]
        return _clip(f"Benchmark run '{label}':\n" + "\n".join(lines)), proc.returncode == 0, {}

    async def t_compare_runs(self, labels=None):
        runs = {}
        for f in sorted(BENCH_DIR.glob("*.json")) + sorted(BENCH_DIR.parent.glob("*.json")):
            runs.setdefault(f.stem, f)
        if not labels:
            return ("Saved runs: " + (", ".join(runs) or "none yet")), True, {}
        data = {}
        for lb in labels:
            if lb not in runs:
                return f"No saved run '{lb}'. Saved: {', '.join(runs)}", False, {}
            data[lb] = {e["key"]: e for e in json.loads(runs[lb].read_text())["experiments"]}
        keys = [k for k in dict.fromkeys(k for d in data.values() for k in d)]
        rows = ["experiment: " + " | ".join(labels)]
        for k in keys:
            cells = []
            for lb in labels:
                e = data[lb].get(k)
                cells.append(f"{e['score']:.2f} ({e['measured']})" if e else "-")
            rows.append(f"{k}: " + " | ".join(cells))
        return _clip("\n".join(rows)), True, {}

    # ------------------------------------------------------------ skills (motor skill cache)
    async def t_list_skills(self):
        eng = getattr(self.world, "skills", None)
        if eng is None:
            return "The body isn't running.", False, {}
        lst = eng.list()
        if not lst:
            return "No skills learned yet. Use learn_skill to teach one.", True, {}
        rows = []
        for sk in lst:
            what = ("combination: " + ", ".join(sk["sequence"]) if sk.get("sequence")
                    else f"goal: {sk['goal']}; score {sk['score']}; {_metrics_brief(sk['metrics'])}" if sk.get("goal")
                    else "activates the walking command neurons (the walking model steps)" if sk.get("command")
                    else f"muscles followed the plan {(sk['score'] or 0):.2f}" + (f"; {_metrics_brief(sk['metrics'])}" if sk.get("metrics") else ""))
            on = [k for k, v in (sk.get("assists") or {}).items() if v]
            parts = "" if sk.get("command") or sk.get("assists") is None else f"; hand-built parts: {', '.join(on) or 'none'}"
            rows.append(f"- {sk['name']}{' (built in)' if sk['builtin'] else ''}: {sk['description']} "
                        f"({(sk['duration'] or 0):.2f} s; {what}; driven via {', '.join(sk['levels'])}{parts}; used {sk['uses']}x)")
        return "\n".join(rows), True, {}

    async def t_learn_skill(self, name, description, keyframes=None, duration=None, goal=None, trials=None, baseline=None,
                            assists=None, gait=None, drive_at=None):
        eng = getattr(self.world, "skills", None)
        if eng is None:
            return "The body isn't running.", False, {}
        if eng.get(name):
            return (f"'{name}' is already learned (score {eng.get(name)['score']}). Use do_skills or test_skill, "
                    f"refine_skill to improve it, or pick another name."), True, {}
        base = re.sub(r"(_?v\d+|_\d+)$", "", name)
        kin = [k for k in eng.lib if re.sub(r"(_?v\d+|_\d+)$", "", k) == base]
        if kin:
            return (f"There is already '{kin[0]}'. Don't start a new version from scratch: refine_skill(name='{kin[0]}', "
                    f"goal=..., keyframes=... if the plan should change) continues from what it learned and keeps whichever "
                    f"version does better. (To really start over, forget it first, or pick a name that says what's "
                    f"different.)"), False, {}
        text = f"{name} {description}".lower()
        walking = any(w in text for w in ("walk", "step", "run ", "crawl")) or \
            (isinstance(goal, str) and goal.split("+")[0] in ("forward", "backward", "turn_left", "turn_right"))
        if keyframes and walking and not gait:
            dur = float(duration or max((float(k.get("t", 0)) for k in keyframes), default=0))
            legs = any(k.split(".")[0] in ("LF", "LM", "LH", "RF", "RM", "RH") for kf in keyframes for k in kf)
            if legs and dur >= 0.8 and len(keyframes) < 6 * dur:
                return (f"This plan has {len(keyframes)} keyframes over {dur:g} s: the legs would mostly hold still, which "
                        "can't move the fly. Walking needs every leg to lift, swing forward, press down and sweep back "
                        "several times a second (tripod gait, ~8 keyframes per step). Use gait={\"steps_per_s\": 5, "
                        "\"duration\": ...} for a plan made from a real fly's recorded steps (with goal forward), or the "
                        "built-in walk_forward (the walking model), and combine_skills to chain walking with other "
                        "skills."), False, {}
        if gait and goal is None:
            t = float(gait.get("turn", 0) or 0)
            goal = "backward" if gait.get("backwards") else "turn_left" if t > 0.3 else "turn_right" if t < -0.3 else "forward"
        if keyframes:
            plan = {"duration": duration, "keyframes": keyframes}
        elif gait:
            try:
                plan = eng.gait_plan(**{k: v for k, v in gait.items() if k in ("steps_per_s", "duration", "backwards", "turn")})
            except (TypeError, ValueError) as e:
                return f"Bad gait: {e}", False, {}
        else:
            return "Give keyframes (your plan) or gait (a walking plan from recorded steps).", False, {}
        r = await eng.learn(name, description, plan, goal=goal, trials=trials, baseline=baseline, assists=assists,
                            drive_at=drive_at)
        if not r.get("ok"):
            return f"Couldn't learn it: {r.get('error')}", False, {}
        return _learn_text(f"Learned '{name}'", r), True, {}

    async def t_refine_skill(self, name, keyframes=None, duration=None, description=None, goal=None, trials=None,
                             baseline=None, assists=None, drive_at=None):
        eng = getattr(self.world, "skills", None)
        if eng is None:
            return "The body isn't running.", False, {}
        old = eng.lib.get(name)
        if not old:
            return (f"'{name}' isn't one of your learned skills (built-in ones can't be changed). Learn it first, "
                    f"or learn a new variant under another name."), False, {}
        if old.get("sequence"):
            return (f"'{name}' is a combination of skills: refine its parts "
                    f"({', '.join(x['skill'] for x in old['sequence'])}), or combine_skills again."), False, {}
        goal = goal if goal is not None else old.get("goal")
        if goal is None and not keyframes:
            return ("Give it a goal (what the movement should achieve) to practise for, or corrected keyframes."), False, {}
        plan = {"duration": duration, "keyframes": keyframes} if keyframes else old["plan"]
        r = await eng.learn(name, description or old.get("description", ""), plan, goal=goal, trials=trials,
                            baseline=baseline, assists=assists if assists is not None else old.get("assists"),
                            warm=old if not drive_at else None, keep_rates=not keyframes, drive_at=drive_at)
        if not r.get("ok"):
            return f"Couldn't refine it: {r.get('error')}", False, {}
        before = f" Before: score {old.get('score')}, {_metrics_brief(old.get('metrics'))}." if old.get("metrics") else ""
        return _learn_text(f"Refined '{name}'", r) + before, True, {}

    async def t_test_skill(self, name, assists=None, goal=None, runs=3, duration=None):
        eng = getattr(self.world, "skills", None)
        if eng is None:
            return "The body isn't running.", False, {}
        try:
            r = await eng.test(name, assists=assists, goal=goal, runs=runs, duration=duration)
        except (ValueError, RuntimeError) as e:
            return f"Couldn't test it: {e}", False, {}
        if not r.get("ok"):
            return f"Couldn't test it: {r.get('error')}", False, {}
        on = [k for k, v in r["assists"].items() if v]
        sc = f" Goal ({r['goal']}): score {r['score']}." if "score" in r else ""
        spread = ", ".join(f"{k} ±{v}" for k, v in r["spread"].items() if k in ("forward_mm", "turn_deg", "height_mm") and v)
        return (f"'{name}' in {r['runs']} runs of {r['duration']:.2f} s (+0.4 s after) in a practice copy, hand-built parts "
                f"{'on: ' + ', '.join(on) if on else 'all off'}: {_metrics_brief(r['metrics'])}"
                + (f" (run-to-run spread {spread})" if spread else "") + f".{sc}"), True, {}

    async def t_combine_skills(self, name, description, steps):
        eng = getattr(self.world, "skills", None)
        if eng is None:
            return "The body isn't running.", False, {}
        if eng.builtin.get(name):
            return f"'{name}' is a built-in skill; pick another name.", False, {}
        try:
            sk = eng.combine(name, description, steps)
        except (ValueError, TypeError) as e:
            return f"Couldn't combine: {e}", False, {}
        parts = ", ".join(x["skill"] + (f" x{x['repeat']}" if x["repeat"] > 1 else "") for x in sk["sequence"])
        return f"Saved '{name}': {parts} (about {sk['duration']:g} s). do_skills(['{name}']) plays it.", True, {}

    async def t_do_skills(self, names, repeat=1):
        eng = getattr(self.world, "skills", None)
        if eng is None:
            return "The body isn't running.", False, {}
        r = await eng.perform(list(names), repeat)
        if not r.get("ok"):
            return f"Couldn't: {r.get('error')}", False, {}
        return f"Done on the live fly: {'; '.join(f'{n}: {s}' for n, s in r['performed'])}.", True, {}

    async def t_motor_reference(self):
        eng = getattr(self.world, "skills", None)
        if eng is None:
            return "The body isn't running.", False, {}
        return eng.reference(), True, {}

    async def t_body_parts(self, walking=None, flight_timer=None):
        body = getattr(self.world, "body", None)
        if body is None:
            return "The body isn't running.", False, {}
        from body import ASSISTS
        for k, v in (("walking", walking), ("flight_timer", flight_timer)):
            if v is not None:
                body.set_assist(k, v)
        return "\n".join(f"- {k}: {'ON' if body.assists.get(k) else 'off'}. {t}" for k, t in ASSISTS.items()), True, {}

    async def t_guide(self, topic="overview"):
        return guide_section(topic), True, {}

    async def t_api_get(self, path):
        import urllib.parse
        import urllib.request
        path = "/" + str(path or "").lstrip("/")
        if not path.startswith("/api/") or any(path.startswith(x) for x in ("/api/command", "/api/skills/learn",
                                                                           "/api/skills/test")):
            return "Only the read-only /api/... paths (GET). See guide('scripts').", False, {}
        url = f"http://127.0.0.1:{os.environ.get('PORT', '8765')}{urllib.parse.quote(path, safe='/?=&,:._-')}"

        def get():
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read().decode(errors="replace")
        try:
            text = await asyncio.to_thread(get)
        except Exception as e:
            return f"GET {path} failed: {e}", False, {}
        return _clip(text), True, {}


GUIDE_FILE = Path(__file__).with_name("guide.md")


def guide_section(topic):
    try:
        text = GUIDE_FILE.read_text()
    except OSError:
        return "The guide is missing."
    parts = re.split(r"^## ", text, flags=re.M)
    for p in parts[1:]:
        head, _, body = p.partition("\n")
        if head.strip().lower().startswith(str(topic).lower()):
            return _clip("## " + head + "\n" + body.strip())
    return "Topics: " + ", ".join(p.partition("\n")[0].strip() for p in parts[1:])


def _metrics_brief(m):
    if not m:
        return "no measurements"
    keys = [("forward_mm", "forward {:+.1f} mm"), ("side_mm", "left {:+.1f} mm"), ("turn_deg", "turned {:+.0f} deg"),
            ("height_mm", "height {:.1f} mm"), ("airtime_s", "in the air {:.2f} s"), ("fell", "fell {:.0%} of runs"),
            ("wing_power", "wing power {:.2f}"), ("jump", "jump muscle {:.2f}"), ("unwanted", "unplanned muscles {:.2f}"),
            ("tracking", "muscles followed the plan {:.2f}")]
    return ", ".join(f.format(m[k]) for k, f in keys if k in m and m[k] is not None)


def _learn_text(head, r):
    rep = r.get("report") or {}
    drv = "; ".join(f"{d['channel']} via {d['level_text']}: {', '.join(d['types'][:3])}" for d in r["drivers"][:14])
    if len(r["drivers"]) > 14:
        drv += f"; ... ({len(r['drivers'])} channels)"
    tl = "; ".join(f"{k}: " + ", ".join(f"t={t} plan {p:+.2f} got {g:+.2f}" for t, p, g in pts)
                   for k, pts in list((r.get("timeline") or {}).items())[:8])
    lim = (" Clipped by the skeleton: " + "; ".join(r["limits"]) + ".") if r.get("limits") else ""
    n = r.get("copies")
    where = (f"practised in {n} background {'copy' if n == 1 else 'copies'} of the fly" if r.get("where") == "practice"
             else "practised on the live fly")
    out = [f"{head} in {r['seconds']} s ({where}).{lim}"]
    if r.get("note"):
        out.append(r["note"] + ".")
    out.append(f"Shaping (muscles following the plan, 1 = exactly): tries {r['attempts']}, per control {r['per_control']}.")
    if rep.get("goal"):
        st = rep.get("start") or {}
        out.append(f"Practice for the goal ({rep['goal']}): {rep['trials']} tries; best of each round {rep['by_round']}. "
                   f"Result (mean of 3 runs): score {rep['score']} [{', '.join(f'{k} {v:+.2f}' for k, v in rep['parts'].items())}]; "
                   f"{_metrics_brief(r['metrics'])}. "
                   + (f"Before practice (the shaped plan): score {st.get('score')}, {_metrics_brief(st.get('metrics'))}. " if st else "")
                   + ("The skill as it was stayed best. " if rep.get("kept_old") else f"What practice changed: {rep['change']}. "))
        b = rep.get("baseline")
        if b and not b.get("error"):
            out.append(f"Baseline {b['text']}: score {b['score']}, {_metrics_brief(b['metrics'])}.")
        elif b:
            out.append(f"Baseline: {b['error']}.")
    else:
        out.append(f"Result: {_metrics_brief(r.get('metrics'))}.")
    on = [k for k, v in (r.get("assists") or {}).items() if v]
    out.append(f"Hand-built parts in practice: {', '.join(on) if on else 'none (neurons and muscles only)'}.")
    out.append(f"Neurons driven: {drv}.")
    if tl:
        out.append(f"Planned vs achieved (shaping): {tl}.")
    if r.get("shown"):
        out.append(f"Shown on the live fly: {r['shown']}.")
    out.append("Saved: do_skills replays it; test_skill measures it; refine_skill practises more or for a new goal.")
    return " ".join(out)
