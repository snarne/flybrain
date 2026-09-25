"""
Flybrain server: runs the whole-brain simulation and streams it to the 3D viewer.

  uv run server/app.py        then open http://localhost:8765

WebSocket /ws
  server -> browser  binary frames: header (float64 t_ms, uint32 n) + uint32 spiking neuron ids
                     JSON "status" messages ~5x a second (behaviour, regions, top neurons)
  browser -> server  JSON commands, same shape as POST /api/command (see apply_command),
                     plus agent commands (chat, stop, approve, ...; see handle_agent_command)

The agent (open LLM + Laya + tools) lives in server/agent/. Its events go out on the same socket.
The body: motor.py (motor neurons -> muscles -> joints), body.py (stimuli, what the fly is doing),
skills.py (learning new movements by finding which neurons to drive).
"""
import asyncio
import json
import os
import struct
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import connectome
import motor as motor_mod
import sim

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"
# Keep downloads (e.g. Laya's checkpoint) inside the project folder even when not started via run.sh
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

FPS = 30
STATUS_EVERY = 6          # frames
WINDOW_MS = 250.0         # window for firing-rate readouts


class World:
    def __init__(self):
        indptr, indices, w_signed = connectome.load()
        nz = np.load(DATA / "neurons.npz")
        self.tables = json.load(open(DATA / "tables.json"))
        self.presets = json.load(open(DATA / "presets.json"))
        self.root_ids = nz["root_ids"]
        self.type = nz["type"]
        self.cls = nz["cls"]
        self.sup = nz["sup"]
        self.neuropil = nz["neuropil"]
        self.region = nz["region"].astype(np.int64)
        self.side = nz["side"]
        self.nt = nz["nt"]
        self.indptr, self.indices = indptr, indices
        # "stable" adds spike-frequency adaptation to self-exciting loop neurons; "paper" is the plain
        # Shiu et al. model on the same wiring
        self.weights = {"paper": w_signed, "stable": w_signed}
        loop_file = DATA / "loop_neurons.npy"
        adapt = np.load(loop_file) if loop_file.exists() else np.ones(len(indptr) - 1, bool)
        self.brain = sim.Brain(indptr, indices, self.weights["stable"], mode="stable", adapt_mask=adapt)
        self.N = self.brain.N
        self.n_regions = len(self.tables["regions"])
        self.stim = {s["key"]: s for s in self.presets["stimuli"]}
        self.readouts = self.presets["readouts"]
        self.readout_of = np.full(self.N, -1, np.int64)
        for k, r in enumerate(self.readouts):
            self.readout_of[r["neurons"]] = k
        self.ro_sizes = np.array([max(len(r["neurons"]), 1) for r in self.readouts], float)
        self.ro_fast = np.zeros(len(self.readouts))
        # motor neurons -> muscles -> joints
        self.motor = motor_mod.Motor(self)
        self.csc = None  # incoming-connection index, built in the background for explain()
        # user-controlled state
        self.active_presets = {}      # key -> Hz
        self.neuron_stim = {}         # idx -> Hz
        self.channel_stim = {}        # agent bridge channel -> {idx: Hz}
        self.blocked = set()
        self.paused = False
        self.speed = 1.0
        self.pending = deque()
        self.hist = deque()           # (t_ms, region_counts, readout_counts)
        self.rate = np.zeros(self.N, np.float32)   # smoothed Hz per neuron
        self.sim_ratio = 1.0
        self.stim_log = deque(maxlen=50)

    # ------------------------------------------------------------ commands
    def apply_command(self, c):
        """All control goes through here (browser, HTTP API, future LLM bridge)."""
        cmd = c.get("cmd")
        b = self.brain
        if cmd == "preset":                      # {"cmd":"preset","key":"sugar","rate":150}  rate 0 = off
            if c["key"] in self.stim:
                r = float(c.get("rate", self.stim[c["key"]]["rate"]))
                if r > 0:
                    self.active_presets[c["key"]] = r
                else:
                    self.active_presets.pop(c["key"], None)
                self._push_stim()
        elif cmd == "neuron":                    # {"cmd":"neuron","idx":123,"rate":150}
            i, r = int(c["idx"]), float(c.get("rate", 150))
            if r > 0:
                self.neuron_stim[i] = r
            else:
                self.neuron_stim.pop(i, None)
            self._push_stim()
        elif cmd == "type":                      # {"cmd":"type","type":"LPLC2","rate":100}
            idx = self.neurons_of_type(c["type"])
            r = float(c.get("rate", 150))
            for i in idx:
                if r > 0:
                    self.neuron_stim[int(i)] = r
                else:
                    self.neuron_stim.pop(int(i), None)
            self._push_stim()
        elif cmd == "channel":                   # agent bridge: {"cmd":"channel","name":"hear","neurons":[...],"rate":120}
            if float(c.get("rate", 0)) > 0:
                self.channel_stim[c["name"]] = {int(i): float(c["rate"]) for i in c["neurons"]}
            else:
                self.channel_stim.pop(c["name"], None)
            self._push_stim()
        elif cmd == "block":                     # {"cmd":"block","idx":123,"on":true}
            i = int(c["idx"])
            (self.blocked.add if c.get("on", True) else self.blocked.discard)(i)
            b.set_blocked(self.blocked)
        elif cmd == "clear":                     # all inputs off
            self.active_presets.clear()
            self.neuron_stim.clear()
            self.blocked.clear()
            b.set_blocked([])
            self._push_stim()
        elif cmd == "reset":                     # quiet the brain, keep inputs
            b.reset()
            b.set_blocked(self.blocked)
            self._push_stim()
            self.hist.clear()
            self.rate[:] = 0
        elif cmd == "pause":
            self.paused = bool(c.get("on", not self.paused))
        elif cmd == "speed":
            self.speed = float(min(max(c.get("value", 1.0), 0.05), 4.0))
        elif cmd == "mode":                      # "stable" | "paper"
            m = c.get("value", "stable")
            if m in self.weights:
                b.set_weights(self.weights[m])
                b.mode = m
                self.apply_command({"cmd": "reset"})

    def _push_stim(self):
        rates = {}
        for k, r in self.active_presets.items():
            for i in self.stim[k]["neurons"]:
                rates[i] = max(rates.get(i, 0.0), r)
        for i, r in self.neuron_stim.items():
            rates[i] = max(rates.get(i, 0.0), r)
        for ch in self.channel_stim.values():
            for i, r in ch.items():
                rates[i] = max(rates.get(i, 0.0), r)
        self.brain.set_stim(rates)

    def neurons_of_type(self, t):
        names = self.tables["type"]
        if t not in names:
            return np.array([], np.int64)
        return np.where(self.type == names.index(t))[0]

    # ------------------------------------------------------------ run one frame
    def frame(self, sim_ms):
        while self.pending:
            self.apply_command(self.pending.popleft())
        if self.paused:
            return None
        t0 = time.perf_counter()
        spk = self.brain.step(sim_ms).copy()
        el = time.perf_counter() - t0
        self.sim_ratio = 0.8 * self.sim_ratio + 0.2 * (sim_ms / 1000.0) / max(el, 1e-6)
        t = self.brain.t_ms
        reg = np.bincount(self.region[spk], minlength=self.n_regions)
        ro = self.readout_of[spk]
        ro = np.bincount(ro[ro >= 0], minlength=len(self.readouts))
        a = np.exp(-sim_ms / 60.0)  # ~60 ms smoothing for the body's motor readout
        self.ro_fast = a * self.ro_fast + (1 - a) * ro / self.ro_sizes / (sim_ms / 1000.0)
        self.motor.update(spk, sim_ms / 1000.0)
        self.hist.append((t, sim_ms, reg, ro))
        while self.hist and self.hist[0][0] < t - WINDOW_MS:
            self.hist.popleft()
        decay = np.exp(-sim_ms / WINDOW_MS)
        self.rate *= decay
        if len(spk):
            np.add.at(self.rate, spk, (1.0 - decay) * 1000.0 / sim_ms)
        return spk

    def status(self):
        span = sum(h[1] for h in self.hist) or 1.0
        reg = sum((h[2] for h in self.hist), np.zeros(self.n_regions)) / (span / 1000.0)
        ro = sum((h[3] for h in self.hist), np.zeros(len(self.readouts))) / (span / 1000.0)
        mode = self.brain.mode
        behaviours = []
        for k, r in enumerate(self.readouts):
            hz = ro[k] / max(len(r["neurons"]), 1)
            behaviours.append({"key": r["key"], "label": r["label"], "hz": round(float(hz), 1),
                               "level": round(float(min(hz / r["full_rate"][mode], 1.0)), 3)})
        counts = [max(v, 1) for v in self.tables_region_counts()]
        top = np.argpartition(-self.rate, 12)[:12]
        top = top[np.argsort(-self.rate[top])]
        return {
            "type": "status",
            "t_ms": round(self.brain.t_ms, 1),
            "mode": mode,
            "paused": self.paused,
            "speed": self.speed,
            "sim_ratio": round(float(self.sim_ratio), 2),
            "near_threshold": self.brain.active_count,
            "firing": int((self.rate > 1.0).sum()),
            "regions": [{"hz_total": round(float(reg[i]), 1), "hz_per_neuron": round(float(reg[i] / counts[i]), 3)} for i in range(self.n_regions)],
            "behaviours": behaviours,
            "top": [self.brief(int(i)) | {"hz": round(float(self.rate[i]), 1)} for i in top if self.rate[i] > 0.5],
            "inputs": {"presets": self.active_presets, "neurons": len(self.neuron_stim), "blocked": sorted(self.blocked)[:200],
                       "channels": sorted(self.channel_stim)},
            "runaway": bool(not self.active_presets and not self.neuron_stim and not self.channel_stim
                            and (self.rate > 1.0).sum() > 2000),
        }

    def fast_readouts(self):
        return {r["key"]: float(self.ro_fast[k]) for k, r in enumerate(self.readouts)}

    # ------------------------------------------------------------ why did it do that?
    def build_csc(self):
        """Incoming connections per neuron (the connectome read backwards), for tracing decisions."""
        order = np.argsort(self.indices, kind="stable").astype(np.int32)
        pre = (np.searchsorted(self.indptr, np.arange(len(self.indices)), side="right") - 1)[order]
        ptr = np.zeros(self.N + 1, np.int64)
        np.add.at(ptr, self.indices[order].astype(np.int64) + 1, 1)
        self.csc = (np.cumsum(ptr), pre.astype(np.int32), order)

    def explain(self, key, hops=8):
        """Trace backwards from a behaviour's descending neurons along the strongest currently-active
        excitatory inputs: at each step, the upstream cell type contributing most drive
        (firing rate x synapse count) right now, until a stimulated sensory neuron is reached."""
        if self.csc is None:
            self.build_csc()
        ptr, pre_all, order = self.csc
        w_all = self.weights[self.brain.mode]
        names = self.tables["type"]
        ro = next(r for r in list(self.readouts) + list(self.motor.pools) if r["key"] == key)
        stim = set(self.brain.stim_rates)
        cur = np.array(ro["neurons"], np.int64)
        chain = [self._group(cur, ro["label"] + " neurons")]
        seen = set(cur.tolist())
        inhibitor = None
        for hop in range(hops):
            idx = np.concatenate([np.arange(ptr[i], ptr[i + 1]) for i in cur]) if len(cur) else np.array([], np.int64)
            if not len(idx):
                break
            pre = pre_all[idx].astype(np.int64)
            w = w_all[order[idx]].astype(np.float64)
            contrib = self.rate[pre] * w
            if hop == 0:  # strongest inhibition acting on the behaviour right now
                neg = contrib < 0
                if neg.any():
                    tneg = {}
                    for p, c in zip(pre[neg], contrib[neg]):
                        tneg[self.type[p]] = tneg.get(self.type[p], 0.0) + c
                    t, c = min(tneg.items(), key=lambda kv: kv[1])
                    total_pos = contrib[contrib > 0].sum()
                    if -c > 0.15 * max(total_pos, 1e-9):
                        inhibitor = {"type": names[t] or "(unnamed)", "strength": round(float(-c), 1),
                                     "vs_excitation": round(float(-c / max(total_pos, 1e-9)), 2)}
            pos = contrib > 0
            if not pos.any():
                break
            by_type = {}
            for p, c in zip(pre[pos], contrib[pos]):
                by_type.setdefault(self.type[p], [0.0, set()])
                by_type[self.type[p]][0] += c
                by_type[self.type[p]][1].add(int(p))
            total = contrib[pos].sum()
            t, (c, members) = max(by_type.items(), key=lambda kv: kv[1][0])
            members = np.array(sorted(m for m in members if m not in seen), np.int64)
            if not len(members):
                break
            syn = int(w[pos][np.isin(pre[pos], members)].sum())
            g = self._group(members, None)
            g.update({"share": round(float(c / total), 2), "synapses": syn})
            chain.append(g)
            seen.update(members.tolist())
            if stim & set(members.tolist()):
                g["sensory"] = True
                break
            cur = members
        chain.reverse()
        return {"behaviour": key, "label": ro["label"], "chain": chain, "inhibitor": inhibitor}

    def _group(self, idx, label):
        names = self.tables["type"]
        tc = np.bincount(self.type[idx], minlength=len(names))
        main = int(np.argmax(tc))
        reg = np.bincount(self.region[idx], minlength=self.n_regions)
        return {"type": label or (names[main] or "(unnamed)"), "n": int(len(idx)),
                "hz": round(float(self.rate[idx].mean()), 1),
                "region": self.tables["regions"][int(np.argmax(reg))]["label"],
                "neurons": [int(i) for i in idx[:60]]}

    def tables_region_counts(self):
        if not hasattr(self, "_rc"):
            self._rc = np.bincount(self.region, minlength=self.n_regions).tolist()
        return self._rc

    def brief(self, i):
        T = self.tables
        return {
            "idx": i,
            "type": T["type"][self.type[i]] or "(unnamed)",
            "class": T["class"][self.cls[i]],
            "region": T["regions"][self.region[i]]["label"],
        }

    def detail(self, i):
        T = self.tables
        d = self.brief(i)

        def partners(idx, w):
            order = np.argsort(-np.abs(w))[:10]
            return [self.brief(int(idx[k])) | {"synapses": int(w[k])} for k in order]

        s, e = self.indptr[i], self.indptr[i + 1]
        out_idx, out_w = self.indices[s:e], self.weights[self.brain.mode][s:e]
        # incoming: search columns (vectorised)
        mask = self.indices == i
        pre = np.searchsorted(self.indptr, np.where(mask)[0], side="right") - 1
        in_w = self.weights[self.brain.mode][mask]
        d.update({
            "root_id": str(self.root_ids[i]),
            "super_class": T["super_class"][self.sup[i]],
            "neuropil": T["neuropil"][self.neuropil[i]],
            "side": T["side"][self.side[i]],
            "transmitter": T["nt"][self.nt[i]],
            "sign": "inhibitory" if (out_w.sum() < 0) else "excitatory",
            "n_out": int(len(out_idx)), "syn_out": int(np.abs(out_w).sum()),
            "n_in": int(len(pre)), "syn_in": int(np.abs(in_w).sum()),
            "outputs": partners(out_idx, out_w),
            "inputs": partners(pre, in_w),
            "hz": round(float(self.rate[i]), 1),
            "stimulated": i in self.neuron_stim,
            "blocked": i in self.blocked,
            "codex_url": f"https://codex.flywire.ai/app/cell_details?root_id={self.root_ids[i]}&dataset=banc",
        })
        return d


world = World()
clients = set()
outbox = None  # asyncio.Queue of JSON strings for all clients (agent events)


def emit(msg):
    """Send a JSON event to every browser. Safe to call from any thread."""
    if outbox is not None and main_loop is not None:
        main_loop.call_soon_threadsafe(outbox.put_nowait, json.dumps(msg))


main_loop = None
agent = runtime = system1 = bridge = None
from body import Body  # noqa: E402
from skills import SkillEngine  # noqa: E402

body = Body(world)
world.body = body  # so the agent's tools can run experiments on it
skills = SkillEngine(world, world.motor, lambda m: emit(m))
world.skills = skills


def handle_body_command(c):
    a = c.get("action")
    if a == "reset":
        skills.stop()
        body.reset()
        world.pending.append({"cmd": "reset"})
    else:
        body.start(a, c.get("side"))


async def explain_events():
    """Turn the body's behaviour changes into 'why did it do that' traces through the connectome."""
    loop = asyncio.get_running_loop()
    text = {"jump": "Escape jump", "fly": "Flight", "feed": "Proboscis extension (feeding)", "groom": "Antennal grooming",
            "legs": "Leg movement"}

    while True:
        await asyncio.sleep(0.05)
        while body.events:
            ev = body.events.pop(0)
            if skills.state:            # a skill is driving the body: its drivers are the explanation
                continue
            try:
                r = await loop.run_in_executor(None, world.explain, ev["key"])
            except Exception as e:  # never let tracing break the simulation
                r = {"chain": [], "error": str(e)}
            r.update({"type": "decision", "event": ev["event"], "title": text.get(ev["event"], ev["event"]), "why": ev["why"],
                      "senses": dict(body.senses), "t_ms": round(world.brain.t_ms)})
            body.decisions = (body.decisions + [r])[-20:]
            emit(r)


def setup_agent():
    global agent, runtime, system1, bridge
    from agent.agent import Agent
    from agent.bridge import Bridge
    from agent.llm import LLM
    from agent.runtime import Runtime
    from agent.system1 import System1

    runtime = Runtime()
    system1 = System1()
    bridge = Bridge(world, emit)
    bridge.loop = main_loop
    agent = Agent(world, runtime, LLM(runtime), system1, bridge, emit)
    # start the model automatically only if it's already downloaded (never start a big download unasked)
    key = runtime.cfg.get("model") or runtime.recommended
    if (runtime.cfg.get("external_url") or runtime.cfg.get("model_path") or runtime.cfg.get("hf_model")
            or (runtime.server_bin and runtime.model_installed(key))):
        runtime.ensure_started()  # None = whatever llm_config.json says (model_path, model, or recommended)
    else:
        runtime.prepare()


def agent_status():
    st = {"type": "agent_status", "runtime": runtime.status(), "system1": system1.status(),
          "busy": agent.busy, "auto_approve": agent.auto_approve, "bridge": bridge.enabled,
          "conversation": {"id": agent.conv.id, "title": agent.conv.title, "kv_tokens": agent.conv.kv_tokens}}
    return st


async def handle_agent_command(c):
    cmd = c.get("cmd")
    if cmd == "chat":
        agent.submit(str(c.get("text", "")))
    elif cmd == "stop":
        agent.stop()
    elif cmd == "approve":
        agent.approve(c["call_id"], c.get("allow", False), c.get("always", False))
    elif cmd == "auto_approve":
        agent.auto_approve = bool(c.get("on"))
    elif cmd == "new_chat":
        await agent.open_conversation(None)
    elif cmd == "open_chat":
        await agent.open_conversation(c["id"])
    elif cmd == "model_setup":                # one of the built-in models
        runtime.choose_builtin(c.get("model"))
    elif cmd == "model_custom":               # any other: {"kind": "hf"|"path"|"server", "value": ..., "model", "api_key"}
        runtime.set_custom(c.get("kind"), c.get("value"), c.get("model"), c.get("api_key"))
    elif cmd == "bridge":
        bridge.enabled = bool(c.get("on"))
    elif cmd == "body":
        handle_body_command(c)
    elif cmd == "skill":                     # {"cmd":"skill","action":"play"|"stop"|"forget","name":...}
        a = c.get("action")
        if a == "play" and c.get("name"):
            asyncio.create_task(_play_skill([c["name"]], int(c.get("repeat", 1))))
        elif a == "stop":
            skills.stop()
        elif a == "forget":
            skills.forget(c.get("name", ""))
            emit({"type": "skills", "skills": skills.list()})
    else:
        return False
    return True


async def _play_skill(names, repeat):
    r = await skills.perform(names, repeat)
    if not r.get("ok"):
        emit({"type": "skill", "event": "error", "error": r.get("error")})
    emit({"type": "skills", "skills": skills.list()})


async def pump_outbox():
    while True:
        m = await outbox.get()
        for c in list(clients):
            try:
                await c.send_text(m)
            except Exception:
                clients.discard(c)


async def watch_runtime():
    """Push model/Laya status when it changes; restore the KV cache once the model is up."""
    last = None
    while True:
        st = agent_status()
        key = json.dumps(st, sort_keys=True)
        if key != last:
            last = key
            emit(st)
            if st["runtime"]["state"] in ("ready", "external"):
                system1.start()
            if st["runtime"]["state"] == "ready" and not agent.busy:
                await agent.restore_kv()
        await asyncio.sleep(0.5)


@asynccontextmanager
async def lifespan(_app):
    global outbox, main_loop
    main_loop = asyncio.get_running_loop()
    outbox = asyncio.Queue()
    world.brain.step(1)  # compile the numba kernel before the first client
    setup_agent()
    tasks = [asyncio.create_task(t) for t in (run_loop(), pump_outbox(), watch_runtime(), explain_events())]
    loop_ = asyncio.get_running_loop()
    loop_.run_in_executor(None, world.build_csc)  # ready before the first decision
    yield
    for t in tasks:
        t.cancel()
    runtime.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/api/presets")
def get_presets():
    return {
        "stimuli": [{k: v for k, v in s.items() if k != "neurons"} | {"n": len(s["neurons"])} for s in world.presets["stimuli"]],
        "readouts": [{k: v for k, v in r.items() if k != "neurons"} | {"neurons": r["neurons"]} for r in world.readouts],
    }


@app.get("/api/status")
def get_status():
    return world.status()


@app.get("/api/neuron/{i}")
def get_neuron(i: int):
    if not 0 <= i < world.N:
        return JSONResponse({"error": "no such neuron"}, 404)
    return world.detail(i)


@app.get("/api/search")
def search(q: str):
    q = q.strip().lower()
    if not q:
        return []
    names = world.tables["type"]
    counts = np.bincount(world.type, minlength=len(names))
    hits = [(n, int(counts[k])) for k, n in enumerate(names) if n and q in n.lower()]
    hits.sort(key=lambda h: (not h[0].lower().startswith(q), len(h[0])))
    if q.startswith("720575"):
        m = np.where(world.root_ids.astype(str) == q)[0]
        if len(m):
            return [{"neuron": world.brief(int(m[0]))}]
    return [{"type": n, "n": c} for n, c in hits[:20]]


@app.get("/api/type/{name}")
def type_neurons(name: str):
    return {"type": name, "neurons": world.neurons_of_type(name).tolist()}


@app.post("/api/command")
async def post_command(c: dict):
    if not await handle_agent_command(c):
        world.pending.append(c)
    return {"ok": True}


@app.get("/api/agent/status")
def get_agent_status():
    return agent_status()


@app.get("/api/agent/conversations")
def get_conversations():
    from agent.agent import Conversation
    return Conversation.list_all()


@app.get("/api/agent/conversation")
def get_conversation():
    return {"id": agent.conv.id, "title": agent.conv.title, "log": agent.conv.log}


@app.get("/api/body/pathways")
def get_pathways():
    from body import pathways
    return pathways(world, world.motor)


@app.get("/api/skills")
def get_skills():
    return {"skills": skills.list(), "controls": skills.controls_help()}


@app.get("/api/agent/bridge")
def get_bridge():
    return {"channels": {k: {kk: v[kk] for kk in ("label", "about", "rate")} | {"n": len(v["neurons"])}
                         for k, v in world.presets.get("channels", {}).items()}}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    clients.add(sock)
    try:
        await sock.send_text(json.dumps(world.status()))
        await sock.send_text(json.dumps(agent_status()))
        for ev in list(agent.pending_approvals.values()):
            await sock.send_text(json.dumps(ev))
        while True:
            c = json.loads(await sock.receive_text())
            if not await handle_agent_command(c):
                world.pending.append(c)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(sock)


def skill_brief():
    s = skills.state
    if not s:
        return None
    return {"kind": s["kind"], "name": s["name"], "t": round(s["t"], 2), "duration": round(s["duration"], 2),
            "attempt": s["attempt"]}


async def run_loop():
    loop = asyncio.get_running_loop()
    frame_s = 1.0 / FPS
    n = 0
    while True:
        t0 = time.perf_counter()
        sim_ms = world.speed * frame_s * 1000.0
        if not world.paused:
            body.sense(sim_ms / 1000.0)
            skills.tick(sim_ms / 1000.0)
        spk = await loop.run_in_executor(None, world.frame, sim_ms)
        if spk is not None:
            skills.observe(sim_ms / 1000.0)
            body.act(sim_ms / 1000.0, world.motor, world.fast_readouts(), skill=skill_brief())
        if bridge is not None:
            bridge.tick(world.brain.t_ms)
        n += 1
        if clients:
            msgs = []
            if spk is not None:
                msgs.append(struct.pack("<dI", world.brain.t_ms, len(spk)) + spk.astype(np.uint32).tobytes())
            if spk is not None:
                msgs.append(json.dumps(body.state(skill_brief())))
            if n % STATUS_EVERY == 0:
                msgs.append(json.dumps(world.status()))
            for c in list(clients):
                try:
                    for m in msgs:
                        if isinstance(m, bytes):
                            await c.send_bytes(m)
                        else:
                            await c.send_text(m)
                except Exception:
                    clients.discard(c)
        await asyncio.sleep(max(0.0, frame_s - (time.perf_counter() - t0)))


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")

if __name__ == "__main__":
    import os
    import webbrowser

    import uvicorn

    port = int(os.environ.get("PORT", 8765))
    if not os.environ.get("NO_BROWSER"):
        import threading

        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    print(f"\n  Flybrain running at http://localhost:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
