"""
A scripted stand-in for llama-server, for testing the agent and UI without downloading a model.

    python tools/mock_llm.py            # listens on 127.0.0.1:8799
    llm_config.json: {"external_url": "http://127.0.0.1:8799", "external_slots": true}

It speaks the same streaming chat API (reasoning_content, tool_calls, timings with cache_n) and
the /slots save/restore/erase endpoints. Behaviour: messages mentioning code make it write and
run a small Python script; messages mentioning a sense make it stimulate that sense; anything
else gets a short echo reply.
"""
import asyncio
import json
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
kv = {"text": ""}          # what the fake KV cache currently holds
saved = {}


def flat(messages):
    return json.dumps(messages, sort_keys=True)


def plan(messages, thinking):
    last = messages[-1]
    user = next(m["content"] for m in reversed(messages) if m["role"] == "user").lower()
    if last["role"] == "tool":
        prev_call = next(m for m in reversed(messages) if m["role"] == "assistant" and m.get("tool_calls"))
        name = prev_call["tool_calls"][0]["function"]["name"]
        if name == "write_file":
            return None, None, ("run_command", {"command": "python hello_fly.py"})
        if name == "learn_skill" and "type" in user:
            return None, "Learned. Now typing 'hi' with alternating taps.", ("do_skills", {"names": ["tap_front_left", "tap_front_left"]})
        return None, f"Done. The tool said:\n\n```\n{last['content'][:1200]}\n```", None
    reasoning = "The user wants something. Let me think step by step about the best approach." if thinking else None
    if "code" in user or "script" in user or "program" in user:
        code = "import math\n\nfor n in range(1, 6):\n    print(n, 'neurons' if n > 1 else 'neuron', round(math.sqrt(n), 3))\n"
        return reasoning, "I'll write a small script and run it.", ("write_file", {"path": "hello_fly.py", "content": code})
    if "tap" in user or "type" in user:
        plan = [{"t": 0, "LF.lift": 0, "LF.reach": 0}, {"t": 0.15, "LF.lift": 0.9, "LF.reach": 0.6},
                {"t": 0.4, "LF.lift": -0.6, "LF.grip": 0.8, "LF.reach": 0.3}, {"t": 0.6, "LF.lift": 0, "LF.grip": 0, "LF.reach": 0}]
        return reasoning, "I'll learn a tap with my front-left leg first.", ("learn_skill", {"name": "tap_front_left", "description": "lift the front-left leg and tap down", "duration": 0.8, "keyframes": plan})
    if "threat" in user or "escape" in user or "predator" in user:
        return reasoning, "Let me show you on my body.", ("body_experiment", {"experiment": "threat_left"})
    for sense in ("sugar", "bitter", "looming", "touch"):
        if sense in user:
            return reasoning, f"Let me try that with my body.", ("stimulate_senses", {"sense": sense, "seconds": 1})
    return reasoning, "Hi! I'm the mock model. Ask me to write code, or to taste sugar.", None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/slots/{sid}")
async def slots(sid: int, action: str, req: Request):
    body = {}
    try:
        body = await req.json()
    except Exception:
        pass
    n = len(kv["text"]) // 4
    if action == "save":
        saved[body["filename"]] = kv["text"]
        return {"id_slot": sid, "filename": body["filename"], "n_saved": n, "n_written": n * 1000, "timings": {"save_ms": 3.1}}
    if action == "restore":
        kv["text"] = saved.get(body["filename"], "")
        return {"id_slot": sid, "filename": body["filename"], "n_restored": len(kv["text"]) // 4,
                "n_read": len(kv["text"]) * 250, "timings": {"restore_ms": 2.4}}
    if action == "erase":
        kv["text"] = ""
        return {"id_slot": sid, "n_erased": n}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    msgs = body["messages"]
    thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", False)
    reasoning, content, call = plan(msgs, thinking)
    prompt = flat(msgs)
    common = 0
    for a, b in zip(prompt, kv["text"]):
        if a != b:
            break
        common += 1
    cache_n, prompt_n = common // 4, (len(prompt) - common) // 4
    cid = "chatcmpl-" + uuid.uuid4().hex[:10]

    async def gen():
        def chunk(delta, finish=None, extra=None):
            d = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if extra:
                d.update(extra)
            return f"data: {json.dumps(d)}\n\n"
        n = 0
        for part in (reasoning or "").split(" "):
            if part:
                n += 1
                yield chunk({"reasoning_content": part + " "})
                await asyncio.sleep(0.03)
        for part in (content or "").split(" "):
            if part:
                n += 1
                yield chunk({"content": part + " "})
                await asyncio.sleep(0.03)
        if call:
            args = json.dumps(call[1])
            yield chunk({"tool_calls": [{"index": 0, "id": "call_" + uuid.uuid4().hex[:6], "type": "function",
                                         "function": {"name": call[0], "arguments": ""}}]})
            for i in range(0, len(args), 20):
                yield chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[i:i + 20]}}]})
        kv["text"] = prompt + "ANSWER"
        yield chunk({}, "tool_calls" if call else "stop",
                    {"timings": {"cache_n": cache_n, "prompt_n": prompt_n, "predicted_n": n,
                                 "predicted_per_second": 42.0, "prompt_per_second": 900.0}})
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8799, log_level="warning")
