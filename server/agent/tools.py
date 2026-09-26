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
    _fn("learn_skill", "Teach your fly body a NEW movement it has never done (e.g. tapping a key, waving a leg). "
        "You plan it as keyframes of motor controls; the engine then searches the connectome for neurons to drive "
        "(descending neurons first, then nerve-cord premotor neurons, motor neurons only as a last resort), and "
        "the fly practises a few times while the user watches, keeping the best. Slow the first time (several "
        "seconds); afterwards the skill is saved and replayed instantly. Controls (values -1..1, linear between "
        "keyframes, unlisted controls stay relaxed at 0): per leg LF LM LH RF RM RH (front/middle/hind, left/right): "
        "<leg>.swing (+ forward / - back), <leg>.lift (+ raise the femur / - press down), <leg>.reach (+ extend the "
        "tibia / - fold it), <leg>.grip (+ press the tarsus down / - raise it), <leg>.spread (- pull the leg in towards the body, 0 relaxed); "
        "wingL/wingR .power (flight muscles, 0..1), .extend (hold the wing out, 0..1), .stroke (+ bigger / - smaller "
        "wingbeat); head.yaw (+ left), head.pitch (+ up), head.roll; proboscis.rostrum / .haustellum / .labellum "
        "(extend, 0..1), proboscis.pump; antennaL, antennaR; haltereL, haltereR; abdomen.curl (0..1), abdomen.bend "
        "(+ left / - right). The joints have real limits (a fly's leg can't bend backwards or spin), so extreme or "
        "impossible targets get clipped and you'll be told which. Keep movements short (0.3-2 s) and make one skill per "
        "reusable unit (e.g. one tap), then chain them with do_skills.",
        {"name": {"type": "string", "description": "short snake_case name, e.g. tap_front_left"},
         "description": {"type": "string", "description": "what the movement is, in a few words"},
         "duration": {"type": "number", "description": "seconds, 0.2-6"},
         "keyframes": {"type": "array", "description": "e.g. [{\"t\": 0, \"LF.lift\": 0}, {\"t\": 0.15, \"LF.lift\": 0.9, "
                       "\"LF.reach\": 0.5}, {\"t\": 0.35, \"LF.lift\": -0.6, \"LF.grip\": 0.8}, {\"t\": 0.5, \"LF.lift\": 0, \"LF.grip\": 0}]",
                       "items": {"type": "object"}}},
        ["name", "description", "keyframes"]),
    _fn("motor_reference", "Everything you need to plan an accurate movement: every motor control and what it "
        "moves, how far each leg joint can go (degrees), how fast real flies move, and the keyframe timelines of "
        "the built-in grooming skill as a worked example (walking and turning are do_skills walk_forward, walk_backward, turn_left, turn_right: they activate command neurons). Read it before planning a new skill.", {}),
    _fn("refine_skill", "Improve a learned skill with a corrected plan (e.g. after comparing the planned vs achieved "
        "timeline learn_skill returned). Keeps the neurons already found and practises again; the better version is kept.",
        {"name": {"type": "string"}, "keyframes": {"type": "array", "items": {"type": "object"}},
         "duration": {"type": "number"}, "description": {"type": "string"}}, ["name", "keyframes"]),
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
            cmd += ["--body", "physics" if body.phys is not None else "kinematic", "--rhythm", body.rhythm]
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
        return "\n".join(f"- {s['name']}{' (built in)' if s['builtin'] else ''}: {s['description']} ({s['duration']:.2f} s, "
                         f"score {s['score']:.2f}, driven via {', '.join(s['levels'])}, used {s['uses']}x)" for s in lst), True, {}

    async def t_learn_skill(self, name, description, keyframes, duration=None):
        eng = getattr(self.world, "skills", None)
        if eng is None:
            return "The body isn't running.", False, {}
        if eng.get(name):
            return f"'{name}' is already learned (score {eng.get(name)['score']}). Use do_skills, or pick another name.", True, {}
        r = await eng.learn(name, description, {"duration": duration, "keyframes": keyframes})
        if not r.get("ok"):
            return f"Couldn't learn it: {r.get('error')}", False, {}
        drv = "; ".join(f"{d['channel']} via {d['level_text']}: {', '.join(d['types'][:3])}" for d in r["drivers"])
        tl = "; ".join(f"{k}: " + ", ".join(f"t={t} plan {p:+.2f} got {g:+.2f}" for t, p, g in pts)
                       for k, pts in (r.get("timeline") or {}).items())
        lim = (" Beyond what the skeleton allows, so clipped: " + "; ".join(r["limits"]) + ".") if r.get("limits") else ""
        return (f"Learned '{name}' in {r['seconds']} s.{lim} Practice scores {r['attempts']} (best {r['score']}; 1.0 = the "
                f"muscles did exactly what you planned). Per control: {r['per_control']}. Neurons driven: {drv}. "
                f"Saved: do_skills can now replay it instantly. Planned vs achieved (control values, -1..1): {tl}. "
                f"{_ball_text(r.get('ball'))}"
                f"If something lags or overshoots, refine_skill with an adjusted timeline."), True, {}

    async def t_do_skills(self, names, repeat=1):
        eng = getattr(self.world, "skills", None)
        if eng is None:
            return "The body isn't running.", False, {}
        r = await eng.perform(list(names), repeat)
        if not r.get("ok"):
            return f"Couldn't: {r.get('error')}", False, {}
        return (f"Done: {', '.join(f'{n} ({s})' for n, s in r['performed'])} (numbers: how closely the muscles followed "
                "the plan; for walking and turning, which activate command neurons, how the fly moved)."), True, {}

    async def t_motor_reference(self):
        eng = getattr(self.world, "skills", None)
        if eng is None:
            return "The body isn't running.", False, {}
        return eng.reference(), True, {}

    async def t_refine_skill(self, name, keyframes, duration=None, description=None):
        eng = getattr(self.world, "skills", None)
        if eng is None:
            return "The body isn't running.", False, {}
        old = eng.lib.get(name)
        if not old:
            return (f"'{name}' isn't one of your learned skills (built-in ones can't be changed). Learn it first, "
                    f"or learn a new variant under another name."), False, {}
        r = await eng.learn(name, description or old.get("description", ""), {"duration": duration, "keyframes": keyframes},
                            warm=old)
        if not r.get("ok"):
            return f"Couldn't refine it: {r.get('error')}", False, {}
        if r["score"] < old.get("score", 0):
            eng.lib[name] = old
            eng._save()
            return (f"The new version scored {r['score']} (old {old.get('score')}); kept the old one. "
                    f"Planned vs achieved for the new plan: {r.get('timeline')}"), True, {}
        tl = "; ".join(f"{k}: " + ", ".join(f"t={t} plan {p:+.2f} got {g:+.2f}" for t, p, g in pts)
                       for k, pts in (r.get("timeline") or {}).items())
        return (f"Refined '{name}': score {r['score']} (was {old.get('score')}), tries {r['attempts']}. "
                f"Planned vs achieved: {tl}. {_ball_text(r.get('ball'))}"), True, {}

