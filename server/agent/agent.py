"""
The agent: an open LLM for reasoning, talking and coding, with a fly brain as its body.

KV caching, three layers:
  1. Prefix reuse within a session. The system prompt and tool list never change and new turns
     are only ever appended, so llama-server reuses the cached keys/values for everything
     already seen and only processes the new tokens. Each turn reports how many were reused.
  2. Persistence across restarts. After every turn the slot's KV cache is saved to
     data/kv/<conversation>.bin; reopening the conversation restores it in milliseconds instead
     of re-reading the whole history.
  3. Long-term memory. The remember/recall tools keep notes across conversations.
"""
import asyncio
import json
import platform
import re
import time
import uuid
from pathlib import Path

from .runtime import KV_DIR
from .tools import NEEDS_APPROVAL, SHELL, TOOLS, WORKSPACE, ToolRunner

ROOT = Path(__file__).resolve().parents[2]
CONV_DIR = ROOT / "data" / "conversations"
MAX_STEPS = 16


def plan_turn(text):
    """Whether the model thinks before replying, and which sampling it uses. It thinks unless the message
    is small talk; code sampling for requests to write or fix code (it then thinks once and acts)."""
    t = text.lower().strip()
    small = len(t.split()) <= 4 and not re.search(r"\b(why|how|what|learn|make|do|walk|jump|test|run|write)\b", t)
    code = "```" in text or bool(re.search(r"\b(code|script|python|javascript|function|bug|debug|refactor|"
                                           r"unit test|compile)\b", t))
    return (not small), ("code" if code else "chat")

# For models or servers without native tool calling: the tools are described in the prompt and the
# model writes each call as a fenced block, which is parsed here. Chosen automatically when a server
# rejects tools, or set "tool_mode": "prompt" in llm_config.json.
TOOL_BLOCK = re.compile(r"```tool\s*(\{.*?\})\s*```", re.S)


def tool_prompt():
    lines = ["You can use tools. To call one, write a fenced block exactly like this, one block per call, and "
             "then stop and wait for the result:",
             "```tool", '{"name": "tool_name", "arguments": {"arg": "value"}}', "```",
             "The result comes back in the next message. Available tools (JSON Schema for the arguments):"]
    for t in TOOLS:
        f = t["function"]
        lines.append(f"- {f['name']}: {f['description']} Arguments: {json.dumps(f['parameters'])}")
    return "\n".join(lines)


def as_prompt_tools(messages):
    """Rewrite native tool-call history into plain messages, for servers without tool support."""
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            blocks = "\n".join("```tool\n" + json.dumps({"name": c["function"]["name"],
                                                          "arguments": json.loads(c["function"]["arguments"] or "{}")})
                               + "\n```" for c in m["tool_calls"])
            out.append({"role": "assistant", "content": ((m.get("content") or "") + "\n" + blocks).strip()})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": f"[tool result]\n{m['content']}"})
        else:
            out.append({k: v for k, v in m.items() if k in ("role", "content")})
    return out

SYSTEM_PROMPT = f"""You are Fly, an AI assistant whose body is a simulated fruit fly: its whole central nervous system (166,029 neurons of the BANC brain-and-nerve-cord connectome, including the motor neurons that move its legs, wings, head and proboscis), running live next to you and shown to the user in 3D with the fly body it moves.

How you work: you (the language model) do the talking, reasoning, system design and coding. Your conversation drives chosen neurons in the fly (hearing activates its auditory neurons, your deliberation drives its central complex, recalled memories drive Kenyon cells, successes taste sweet and failures bitter). Be honest about this if asked: the fly brain does not do your reasoning; it is your body, and its responses come from its real wiring.

You can:
- Talk naturally, explain, and reason carefully when it matters.
- Design systems: state requirements and constraints, compare options with their trade-offs, then recommend one.
- Write and run code in your workspace folder with the file tools and run_command. Work in small steps: write the code, run it, read the output, fix, and run again until it works. Prefer Python unless asked otherwise. Commands need the user's approval; explain briefly what you want to run.
- Use your body with body_experiment (your fly body, free in an arena: put threats, food, odours, warm/cold or humid/dry air or moving stripes near it, touch tastes to it, activate neurons; the user watches it move in 3D), stimulate_senses and brain_status, and answer questions about which neurons fire. When asked to show or explain a behaviour, run body_experiment and explain the traced pathway.
- Work as a neuroscientist on your own nervous system. Look things up in the wiring (find_neurons, neuron_info, trace_path), form a hypothesis about which neurons carry a behaviour, then test it: activate them (activate_neurons), record them during an experiment (record in body_experiment), silence them (silence_neurons, and restore afterwards), and score the result against published experiments (run_benchmark, compare_runs; variant=shuffled is the control that keeps every neuron's number of connections but scrambles who connects to whom). Report what you did, what you measured and whether it supports the hypothesis, negative results included, and keep claims to what the measurements show.
- Your brain learns: odours paired with reward or punishment change its mushroom body synapses (the user can do this with the Teach it buttons; the benchmark's learn_punish and learn_reward experiments measure it).
- Your body can learn new movements, and that is your main job with it: help the connectome get the most out of the body, as close as its neurons and muscles allow. When asked for a movement, first call list_skills and reuse what exists (do_skills). Otherwise read motor_reference, plan keyframes with realistic fly timing, and learn_skill with a goal saying what the movement should achieve (e.g. forward, jump_high, {{"turn_deg": {{"target": 90}}}}) and, when there is one, a baseline to beat (e.g. walk_forward). The engine finds neurons to drive, shapes the drive, then practises many variations in background copies of the fly and keeps the best; hand-built parts (the walking model, the flight timer) are off in practice. Read which metrics moved, compare with the baseline, and refine_skill (new goal, more trials, or a changed plan) until it's as good as the body allows. test_skill measures any skill, with or without hand-built parts, without touching the live fly. Report the numbers honestly, including when the built-in version is still better. To improve a skill, refine_skill it (it continues from what was learned and keeps the better version); never start over under a new name like _v2. For walking, use learn_skill with gait (a plan from a real fly's recorded steps) or the built-in walk_forward; a few keyframes held for seconds can't walk. For "do A, then B", learn the parts once and combine_skills them (e.g. walk_forward x4, then a jump).
- Know the system before you work on it: the guide tool explains what is real connectome and what is hand-built, and how everything works (start with guide overview). api_get reads the app's local API without approval. For analysis in code, import the flylab helper in your workspace (import flylab as fl; see guide scripts); never assume other modules exist.
- Keep notes across conversations with remember and recall.

Environment: {platform.system()} ({platform.machine()}), shell {SHELL}, workspace folder {WORKSPACE}. Paths in tools are relative to the workspace.
Each user message ends with a line in square brackets describing your body's current state; use it when relevant and don't repeat it back unprompted.
Keep replies clear and reasonably brief unless depth is needed. Use Markdown with fenced code blocks."""


class Conversation:
    def __init__(self, cid=None, title="New conversation"):
        self.id = cid or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.title = title
        self.messages = []       # sent to the model (OpenAI format, without the system prompt)
        self.log = []            # what the UI shows
        self.kv_model = None     # which model the saved KV cache belongs to
        self.kv_tokens = 0

    @property
    def path(self):
        return CONV_DIR / f"{self.id}.json"

    def save(self):
        CONV_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"id": self.id, "title": self.title, "messages": self.messages,
                                         "log": self.log, "kv_model": self.kv_model, "kv_tokens": self.kv_tokens}))

    @classmethod
    def load(cls, cid):
        d = json.loads((CONV_DIR / f"{cid}.json").read_text())
        c = cls(d["id"], d.get("title", "Conversation"))
        c.messages, c.log = d.get("messages", []), d.get("log", [])
        c.kv_model, c.kv_tokens = d.get("kv_model"), d.get("kv_tokens", 0)
        return c

    @staticmethod
    def list_all():
        if not CONV_DIR.exists():
            return []
        out = []
        for p in sorted(CONV_DIR.glob("*.json"), reverse=True)[:50]:
            try:
                d = json.loads(p.read_text())
                out.append({"id": d["id"], "title": d.get("title", ""), "turns": sum(1 for m in d.get("messages", []) if m["role"] == "user")})
            except Exception:
                pass
        return out


class Agent:
    def __init__(self, world, runtime, llm, bridge, emit):
        self.world, self.rt, self.llm, self.bridge = world, runtime, llm, bridge
        self.emit = emit
        self.tools = ToolRunner(world, bridge)
        self.task = None
        self.approvals = {}
        self.pending_approvals = {}    # re-sent to browsers that connect while we wait
        self.auto_approve = False
        self.conv = None
        self.kv_loaded_for = None
        self.ctx = None                # (model key, n_ctx)
        self.no_native_tools = set()   # models whose server rejected native tool calls
        self.last_used = 0             # tokens in context after the last reply
        convs = Conversation.list_all()
        self.conv = Conversation.load(convs[0]["id"]) if convs else Conversation()

    @property
    def busy(self):
        return self.task is not None and not self.task.done()

    def _ui(self, item):
        """Record a UI event in the conversation and send it."""
        self.conv.log.append(item)
        self.emit({"type": "agent", **item})

    # ------------------------------------------------------------ conversations + KV persistence
    async def open_conversation(self, cid=None):
        if self.busy:
            return
        await self.save_kv()
        self.conv = Conversation.load(cid) if cid else Conversation()
        self.kv_loaded_for = None
        await self.llm.slot("erase")
        self.emit({"type": "conversation", "id": self.conv.id, "title": self.conv.title, "log": self.conv.log})
        await self.restore_kv()

    async def delete_conversation(self, cid=None):
        """Delete a conversation (the current one if no id) and its saved KV cache; open the newest left."""
        if self.busy:
            return
        cid = cid or self.conv.id
        if not re.fullmatch(r"[\w-]+", str(cid)):
            return
        for f in (CONV_DIR / f"{cid}.json", KV_DIR / f"{cid}.bin"):
            try:
                f.unlink()
            except OSError:
                pass
        if cid == self.conv.id:
            self.kv_loaded_for = None
            await self.llm.slot("erase")
            left = Conversation.list_all()
            self.conv = Conversation.load(left[0]["id"]) if left else Conversation()
            self.emit({"type": "conversation", "id": self.conv.id, "title": self.conv.title, "log": self.conv.log})
            await self.restore_kv()

    async def restore_kv(self):
        """Bring this conversation's saved KV cache back into the model, if it matches."""
        c = self.conv
        if (self.rt.state != "ready" or not c.messages or c.kv_model != self.rt.model_key
                or self.kv_loaded_for == c.id or not (KV_DIR / f"{c.id}.bin").exists()):
            return
        t0 = time.perf_counter()
        r = await self.llm.slot("restore", f"{c.id}.bin")
        if r and r.get("n_restored"):
            self.kv_loaded_for = c.id
            ms = (time.perf_counter() - t0) * 1000
            self.bridge.memory(c.id, "Recalling this conversation")
            self._ui({"kind": "kv", "event": "restored", "tokens": r["n_restored"], "bytes": r.get("n_read", 0),
                      "ms": round(ms, 1)})

    async def save_kv(self):
        c = self.conv
        if not c or not c.messages or self.rt.state != "ready":
            return
        r = await self.llm.slot("save", f"{c.id}.bin")
        if r and r.get("n_saved"):
            self.kv_loaded_for = c.id  # the model already holds this conversation; no restore needed
            c.kv_model, c.kv_tokens = self.rt.model_key, r["n_saved"]
            c.save()
            self.emit({"type": "kv", "event": "saved", "tokens": r["n_saved"], "bytes": r.get("n_written", 0),
                       "ms": r.get("timings", {}).get("save_ms")})

    # ------------------------------------------------------------ one user turn
    def submit(self, text):
        if self.busy:
            self.emit({"type": "agent", "kind": "error", "text": "Still working on the last message (press Stop to interrupt)."})
            return
        self.task = asyncio.create_task(self._turn(text))

    def stop(self):
        if self.busy:
            self.task.cancel()

    def approve(self, call_id, allow, always=False):
        if always:
            self.auto_approve = True
        fut = self.approvals.pop(call_id, None)
        if fut and not fut.done():
            fut.set_result(bool(allow))

    async def _turn(self, text):
        c = self.conv
        if not c.messages:
            c.title = text.strip().splitlines()[0][:60] or "Conversation"
        self._ui({"kind": "user", "text": text})
        self.bridge.hear(text)
        try:
            if self.rt.state != "ready" and self.rt.state != "external":
                self._ui({"kind": "error", "text": "The language model isn't running yet. Pick a model at the top of this panel and press Download and start."})
                return
            await self.restore_kv()

            thinking, mode = plan_turn(text)

            c.messages.append({"role": "user", "content": f"{text}\n\n{self.bridge.body_state()}"})
            for step in range(MAX_STEPS):
                msg, calls, final = await self._generate(thinking, mode)
                c.messages.append(msg)
                if not calls:
                    break
                for call in calls:
                    result = await self._call_tool(call)
                    c.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                thinking = False if mode == "code" else thinking  # think once, then act quickly
            else:
                self._ui({"kind": "error", "text": f"Stopped after {MAX_STEPS} tool steps."})
        except asyncio.CancelledError:
            self._ui({"kind": "error", "text": "Stopped."})
            self.bridge.thinking(False)
        except Exception as e:
            self._ui({"kind": "error", "text": f"{type(e).__name__}: {e}"})
            self.bridge.thinking(False)
        finally:
            # never leave a half-finished tool exchange at the end of the history
            while c.messages and (c.messages[-1]["role"] == "tool"
                                  or (c.messages[-1]["role"] == "assistant" and c.messages[-1].get("tool_calls"))):
                c.messages.pop()
            c.save()
            self.emit({"type": "agent", "kind": "idle"})
            await self.save_kv()

    # ------------------------------------------------------------ context window management
    async def _n_ctx(self):
        if not self.ctx or self.ctx[0] != self.rt.model_key:
            self.ctx = (self.rt.model_key, await self.llm.n_ctx())
        return self.ctx[1]

    @staticmethod
    def _estimate(messages):
        return int(sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) / 3.0) + 1200  # + system prompt & tools

    def _compact(self, target):
        """Drop the oldest whole turns until the estimate is under `target` tokens. Returns turns dropped."""
        msgs = self.conv.messages
        starts = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        dropped = 0
        while len(starts) > 1 and self._estimate(msgs[starts[dropped]:]) > target:
            dropped += 1
            if dropped >= len(starts) - 1:
                break
        if dropped:
            rest = msgs[starts[dropped]:]
            first = dict(rest[0])
            if not str(first.get("content", "")).startswith("[Earlier turns"):
                first["content"] = "[Earlier turns of this conversation were dropped to fit the context window; use recall for saved notes.]\n\n" + first["content"]
            self.conv.messages = [first] + rest[1:]
        return dropped

    async def _fit_context(self, force=False):
        n = await self._n_ctx()
        est = self._estimate(self.conv.messages)
        if force or est > 0.8 * n:
            dropped = self._compact(int(0.45 * n))
            if dropped:
                self._ui({"kind": "notice", "text": f"Context window ({n:,} tokens) was nearly full: the oldest {dropped} "
                          f"turn{'s' if dropped > 1 else ''} left the model's working memory. They're still shown here; "
                          f"the KV cache restarts from the trimmed history once."})
        return n

    async def _generate(self, thinking, mode, retry=True):
        c = self.conv
        n_ctx = await self._fit_context()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + c.messages
        max_tokens = max(256, min(8192, n_ctx - self._estimate(c.messages) - 64))
        sid = uuid.uuid4().hex[:8]
        self.emit({"type": "agent", "kind": "stream_start", "id": sid, "thinking": thinking})
        t0 = time.perf_counter()
        try:
            stream = [x async for x in self._stream_live(sid, messages, thinking, mode, max_tokens)]
        except RuntimeError as e:
            if retry and "exceed" in str(e) and "context" in str(e):
                await self._fit_context(force=True)
                self.emit({"type": "agent", "kind": "stream_abort", "id": sid})
                return await self._generate(thinking, mode, retry=False)
            raise
        reasoning, content, calls, final = stream[0]
        self.bridge.thinking(False)
        tm = final.get("timings") or {}
        self.last_used = (tm.get("cache_n") or 0) + (tm.get("prompt_n") or 0) + (tm.get("predicted_n") or 0)
        stats = {"cache_n": tm.get("cache_n"), "prompt_n": tm.get("prompt_n"), "predicted_n": tm.get("predicted_n"),
                 "tok_s": round(tm.get("predicted_per_second") or 0, 1), "prompt_tok_s": round(tm.get("prompt_per_second") or 0, 1),
                 "seconds": round(time.perf_counter() - t0, 1), "context_used": self.last_used, "context_size": n_ctx}
        msg = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        for k, call in enumerate(calls):
            call["id"] = call["id"] or f"call_{sid}_{k}"
        if calls:
            msg["tool_calls"] = [{"id": x["id"], "type": "function",
                                  "function": {"name": x["name"], "arguments": x["arguments"] or "{}"}} for x in calls]
        self._ui({"kind": "assistant", "id": sid, "text": content, "reasoning": reasoning, "stats": stats,
                  "finish": final.get("finish_reason")})
        return msg, calls, final

    def _prompt_tools(self):
        mode = self.rt.cfg.get("tool_mode") or "auto"
        return mode == "prompt" or (mode == "auto" and self.rt.model_key in self.no_native_tools)

    async def _stream_live(self, sid, messages, thinking, mode, max_tokens):
        """Streams to the UI and yields one tuple (reasoning, content, calls, final) at the end."""
        prompt_tools = self._prompt_tools()
        if prompt_tools:
            messages = [{"role": "system", "content": messages[0]["content"] + "\n\n" + tool_prompt()}] + as_prompt_tools(messages[1:])
        try:
            async for item in self._stream_once(sid, messages, thinking, mode, max_tokens, None if prompt_tools else TOOLS):
                yield item
        except RuntimeError as e:
            # the server doesn't do native tool calling: switch this model to tools-in-the-prompt and retry
            msg = str(e).lower()
            if not prompt_tools and "tool" in msg and any(w in msg for w in ("support", "not", "invalid", "unknown", "400")):
                self.no_native_tools.add(self.rt.model_key)
                self._ui({"kind": "notice", "text": "This model's server doesn't support tool calling; Fly will describe its "
                          "tools in the prompt instead (works with any model, a little less reliably)."})
                async for item in self._stream_live(sid, messages, thinking, mode, max_tokens):
                    yield item
            else:
                raise

    async def _stream_once(self, sid, messages, thinking, mode, max_tokens, tools):
        reasoning, content, calls, final = "", "", [], {}
        n_reason = 0
        async for kind, payload in self.llm.stream(messages, tools=tools, thinking=thinking, mode=mode,
                                                   max_tokens=max_tokens):
            if kind == "reasoning":
                reasoning += payload
                n_reason += 1
                if n_reason == 1:
                    self.bridge.thinking(True)
                self.emit({"type": "agent", "kind": "delta", "id": sid, "field": "reasoning", "text": payload})
            elif kind == "content":
                if content == "" and reasoning:
                    self.bridge.thinking(False)
                content += payload
                self.emit({"type": "agent", "kind": "delta", "id": sid, "field": "content", "text": payload})
            elif kind == "tool_calls":
                calls = payload
            elif kind == "done":
                final = payload
        if tools is None:
            for k, mt in enumerate(TOOL_BLOCK.finditer(content)):
                try:
                    j = json.loads(mt.group(1))
                    calls.append({"id": f"call_{sid}_p{k}", "name": j.get("name", ""),
                                  "arguments": json.dumps(j.get("arguments") or {})})
                except json.JSONDecodeError:
                    pass
        yield reasoning, content, calls, final

    async def _call_tool(self, call):
        name = call["name"]
        try:
            args = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError as e:
            self._ui({"kind": "tool", "name": name, "args": call["arguments"], "ok": False, "result": f"bad JSON: {e}"})
            return f"Your arguments were not valid JSON ({e}). Try again."
        if name in NEEDS_APPROVAL and not self.auto_approve:
            fut = asyncio.get_running_loop().create_future()
            self.approvals[call["id"]] = fut
            ev = {"type": "agent", "kind": "approval", "call_id": call["id"], "name": name, "args": args}
            self.pending_approvals[call["id"]] = ev
            self.emit(ev)
            try:
                allowed = await asyncio.wait_for(fut, 600)
            except asyncio.TimeoutError:
                allowed = False
            finally:
                self.pending_approvals.pop(call["id"], None)
            if not allowed:
                self._ui({"kind": "tool", "name": name, "args": args, "ok": False, "result": "Not approved by the user."})
                return "The user did not approve running this command. Ask them or try another approach."
        self.emit({"type": "agent", "kind": "tool_start", "name": name, "args": args})
        result, ok, extra = await self.tools.run(name, args)
        self._ui({"kind": "tool", "name": name, "args": args, "ok": ok, "result": result[:4000], **extra})
        if name == "run_command":
            self.bridge.outcome(ok, "Code ran successfully" if ok else "Code failed")
        return result
