"""Streaming client for an OpenAI-compatible chat server (llama-server, or any other via external_url)."""
import json

import httpx

# Sampling recommended by Qwen for each mode.
SAMPLING = {
    "think": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    "code": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    "chat": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
}


class LLM:
    def __init__(self, runtime):
        self.rt = runtime
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0))

    async def stream(self, messages, tools=None, thinking=False, mode="chat", max_tokens=4096):
        """Yields (kind, payload): ('reasoning', str) | ('content', str) | ('tool_calls', list) | ('done', dict)."""
        cfg = self.rt.cfg
        samp = SAMPLING["think" if thinking and mode != "code" else mode]
        body = {
            "model": cfg.get("external_model") or self.rt.model_key or "local",
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "stream_options": {"include_usage": True},
        }
        headers = {}
        if cfg.get("external_api_key"):
            headers["Authorization"] = f"Bearer {cfg['external_api_key']}"
        if cfg.get("external_url") and (cfg.get("external_api_key") or cfg.get("external_strict")):
            # hosted APIs reject llama.cpp's extra parameters: send only the standard ones
            body.update(temperature=samp["temperature"], top_p=samp["top_p"])
        else:
            body.update(chat_template_kwargs={"enable_thinking": bool(thinking)}, cache_prompt=True, **samp)
        if tools:
            body["tools"] = tools
        calls = {}
        final = {"finish_reason": None, "timings": None, "usage": None}
        async with self.client.stream("POST", self.rt.base_url + "/v1/chat/completions", json=body, headers=headers) as r:
            if r.status_code != 200:
                text = (await r.aread()).decode(errors="replace")[:800]
                raise RuntimeError(f"model server returned {r.status_code}: {text}")
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                j = json.loads(data)
                if j.get("timings"):
                    final["timings"] = j["timings"]
                if j.get("usage"):
                    final["usage"] = j["usage"]
                for ch in j.get("choices", []):
                    d = ch.get("delta") or {}
                    if d.get("reasoning_content") or d.get("reasoning"):
                        yield "reasoning", d.get("reasoning_content") or d.get("reasoning")
                    if d.get("content"):
                        yield "content", d["content"]
                    for tc in d.get("tool_calls") or []:
                        c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        c["id"] = tc.get("id") or c["id"]
                        fn = tc.get("function") or {}
                        c["name"] += fn.get("name") or ""
                        c["arguments"] += fn.get("arguments") or ""
                    if ch.get("finish_reason"):
                        final["finish_reason"] = ch["finish_reason"]
        if calls:
            yield "tool_calls", [calls[k] for k in sorted(calls)]
        yield "done", final

    async def n_ctx(self, default=8192):
        """The context window the server actually allocated (llama.cpp's --fit may shrink it)."""
        try:
            r = await self.client.get(self.rt.base_url + "/props", timeout=5)
            n = r.json().get("default_generation_settings", {}).get("n_ctx")
            if n:
                return int(n)
        except Exception:
            pass
        return int(self.rt.cfg.get("context") or default)

    # ------------------------------------------------------------ KV cache persistence (llama-server slots)
    async def slot(self, action, filename=None):
        if not self.rt.supports_slots:
            return None
        try:
            r = await self.client.post(self.rt.base_url + f"/slots/0?action={action}",
                                       json={"filename": filename} if filename else None, timeout=60)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None
