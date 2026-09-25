"""
System 1: fast, typed snap judgements with Laya (Convai, Apache 2.0).

Laya is a 421M-parameter encoder that answers typed questions in one forward pass (~30 ms on a
GPU, ~100-300 ms on CPU) with calibrated probabilities. It never generates text, so it can't
hallucinate. Here it decides, before the LLM (System 2) starts:
  intent  - chat / question / code / design / brain
  depth   - how much deliberate reasoning the reply needs (turns the LLM's thinking on or off)
  tools   - whether the reply needs files or running code
  tone    - how the person seems to feel (drives taste neurons in the fly: praise tastes sweet)

Laya is used zero-shot; its authors note accuracy improves a lot with fine-tuning on your own
decisions. If Laya can't be installed or its checkpoint can't be downloaded, simple keyword rules
stand in and the UI says so.
"""
import re
import threading
import time

QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What is the person asking the assistant to do?",
        "criteria": {
            "chat": "casual conversation, greetings, feelings, small talk",
            "question": "asking for information, an explanation or an opinion",
            "code": "write, fix, run, test or review code or scripts",
            "design": "plan or design a system, architecture, product or approach",
            "brain": "about the fruit fly brain, its neurons, senses or behaviour, or controlling it",
        },
    },
    "depth": {
        "type": "score",
        "instructions": "How much careful step-by-step reasoning does a good reply need?",
        "criteria": ["an instant reply", "some thought", "careful multi-step reasoning"],
    },
    "tools": {
        "type": "noul",
        "instructions": "Does answering require creating, reading or running code or files?",
    },
    "tone": {
        "type": "choice",
        "instructions": "How does the person seem to feel?",
        "criteria": {
            "positive": "pleased, grateful, excited, praising",
            "neutral": "matter of fact",
            "frustrated": "annoyed, frustrated, upset, complaining that something is wrong",
        },
    },
}


class System1:
    def __init__(self):
        self.router = None
        self.state = "waiting"
        self.detail = "Laya loads when the language model starts"
        self.device = None
        self._started = False

    def start(self):
        """Load Laya in the background (downloads its checkpoint from Hugging Face the first time)."""
        if self._started:
            return
        self._started = True
        self.state, self.detail = "loading", "loading Laya…"
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            from laya import Router

            r = Router()
            r.predict("hello", {"intent": QUESTIONS["intent"]})  # downloads + warms the checkpoint
            self.router = r
            try:
                agent = next(iter(r._agents.values()))
                self.device = str(agent.device)
            except Exception:
                self.device = None
            self.state, self.detail = "ready", f"Laya on {self.device or 'CPU'}"
        except Exception as e:
            self.state, self.detail = "fallback", f"Laya unavailable ({type(e).__name__}: {str(e)[:160]}); using keyword rules"

    def status(self):
        return {"state": self.state, "detail": self.detail}

    def decide(self, text, context=""):
        """Returns a dict with intent, depth (0-2), tools (0-1), tone, confidences, ms, source."""
        t0 = time.perf_counter()
        state = (f"Earlier: {context[-600:]}\n" if context else "") + f"Message: {text[-1500:]}"
        if self.router is not None:
            try:
                res = self.router.predict(state, QUESTIONS)
                a = res["answers"]
                out = {
                    "intent": a["intent"]["choice"], "intent_conf": a["intent"]["confidence"],
                    "intent_probs": a["intent"].get("probabilities", {}),
                    "depth": float(a["depth"]["score"]), "depth_conf": a["depth"]["confidence"],
                    "tools": float(a["tools"]["noul"]),
                    "tone": a["tone"]["choice"], "tone_conf": a["tone"]["confidence"],
                    "source": "laya",
                }
                out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
                return out
            except Exception as e:
                self.detail = f"Laya error ({e}); using keyword rules for this message"
        self.start()
        out = heuristic(text)
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out


def heuristic(text):
    t = text.lower()
    has = lambda *ws: any(re.search(r"\b" + w, t) for w in ws)  # noqa: E731
    if "```" in text or has("code", "function", "script", "python", "javascript", "bug", "debug", "compile",
                             "implement", "refactor", "unit test", "error", "program", "api", "class "):
        intent = "code"
    elif has("design", "architect", "system", "scal", "plan ", "pipeline", "infrastructure"):
        intent = "design"
    elif has("neuron", "brain", "fly", "stimul", "sugar", "bitter", "odou?r", "mushroom", "connectome", "synap"):
        intent = "brain"
    elif "?" in t or has("what", "why", "how", "explain", "who", "when"):
        intent = "question"
    else:
        intent = "chat"
    depth = 0.0
    if intent in ("design", "code") or len(t) > 300:
        depth = 1.2
    if has("why", "prove", "derive", "optimi", "trade-?off", "compare", "step by step", "architecture"):
        depth = max(depth, 1.6)
    tools = 0.8 if intent == "code" and has("run", "write", "create", "build", "test", "make", "file") else 0.15
    tone = "neutral"
    if has("thank", "great", "awesome", "love", "nice", "perfect", "amazing", "cool"):
        tone = "positive"
    if has("wrong", "broken", "doesn.?t work", "not working", "frustrat", "annoy", "ugh", "useless", "bad"):
        tone = "frustrated"
    return {"intent": intent, "intent_conf": None, "intent_probs": {}, "depth": depth, "depth_conf": None,
            "tools": tools, "tone": tone, "tone_conf": None, "source": "rules"}
