"""
Runs an open LLM locally with llama.cpp's llama-server.

Picks the right build for the machine (Metal on Apple silicon, Vulkan on Windows/Linux, which is
the fastest backend on recent AMD cards), downloads it and a GGUF model from
Hugging Face with resumable progress, and starts the server with KV-cache persistence enabled.

One project folder serves several machines. Everything machine-specific is kept per "profile"
(macos-arm64, windows-x64, ...), detected at start-up: the llama.cpp build (runtime/<profile>/),
the chosen model (data/settings/<profile>.json), and saved KV caches (data/kv/<profile>/).
Model files (models/) are portable GGUF and shared.

Optional overrides in llm_config.json at the project root (top-level keys apply everywhere, a
section named after a profile applies only there):
  {"context": 32768,
   "macos-arm64": {"model": "qwen3.5-9b"},
   "windows-x64": {"extra_args": ["--n-cpu-moe", "20"]}}
Keys: model (one of the built-in choices), context, extra_args (passed to llama-server),
hf_model (any GGUF on Hugging Face, "owner/repo" or "owner/repo:QUANT", fetched by llama.cpp),
model_path (a GGUF file you already have), external_url + external_model + external_api_key (any
OpenAI-compatible server: LM Studio, Ollama, vLLM, or a hosted API), port.
The same choices can be made in the app ("Use another model" in the Chat tab); those are saved per
machine in data/settings/ (never in git).
"""
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .models import MODELS, recommend

def machine_profile():
    """A short name for this kind of machine (never anything identifying, like a hostname)."""
    mach = platform.machine().lower()
    arch = "arm64" if mach in ("arm64", "aarch64") else "x64"
    osname = {"darwin": "macos", "win32": "windows"}.get(sys.platform, "linux")
    return f"{osname}-{arch}"


PROFILE = machine_profile()
ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtime" / PROFILE
MODELS_DIR = ROOT / "models"
KV_DIR = ROOT / "data" / "kv" / PROFILE
LOG_DIR = ROOT / "data" / "logs"
CONFIG = ROOT / "llm_config.json"
SETTINGS = ROOT / "data" / "settings" / f"{PROFILE}.json"
PINNED_TAG = "b11163"  # the llama.cpp build this app was tested with
UA = {"User-Agent": "flybrain/0.2"}


def _get_json(url, timeout=20):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


KEYS = ("model", "context", "extra_args", "model_path", "hf_model", "external_url", "external_model", "external_api_key",
        "external_slots", "external_strict", "tool_mode", "port", "llama_tag")


def load_config():
    """defaults < llm_config.json < llm_config.json[profile] < choices made in the app on this machine."""
    cfg = {"model": None, "context": 32768, "extra_args": [], "model_path": None, "hf_model": None,
           "external_url": None, "external_model": None, "external_api_key": None, "port": 8766}
    if CONFIG.exists():
        try:
            user = json.loads(CONFIG.read_text())
            cfg.update({k: v for k, v in user.items() if k in KEYS})
            cfg.update({k: v for k, v in (user.get(PROFILE) or {}).items() if k in KEYS})
        except Exception as e:  # keep going with defaults, but say why
            print(f"llm_config.json ignored: {e}")
    if SETTINGS.exists():
        try:
            cfg.update(json.loads(SETTINGS.read_text()))
        except Exception:
            pass
    return cfg


def save_choice(**kw):
    """Remember a choice made in the app, for this machine profile only."""
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}
    cur.update(kw)
    SETTINGS.write_text(json.dumps(cur, indent=2))


# ---------------------------------------------------------------- hardware
def system_ram_gb():
    try:
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30
        if sys.platform.startswith("linux"):
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 2**20
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 2**30
    except Exception:
        pass
    return 0.0


def build_asset():
    """Which llama.cpp release file suits this machine, and what it will use."""
    mach = platform.machine().lower()
    if sys.platform == "darwin":
        return ("bin-macos-arm64.tar.gz", "Metal") if mach in ("arm64", "aarch64") else ("bin-macos-x64.tar.gz", "CPU")
    if sys.platform == "win32":
        return "bin-win-vulkan-x64.zip", "Vulkan"
    if mach in ("arm64", "aarch64"):
        return "bin-ubuntu-arm64.tar.gz", "CPU"
    return "bin-ubuntu-vulkan-x64.tar.gz", "Vulkan"


class Runtime:
    def __init__(self):
        self.cfg = load_config()
        self.state = "idle"          # idle | installing | downloading | starting | ready | error | external
        self.detail = ""
        self.progress = None         # {"done": bytes, "total": bytes}
        self.proc = None
        self.hw = {"os": sys.platform, "arch": platform.machine(), "ram_gb": round(system_ram_gb(), 1), "gpus": []}
        self.asset, self.backend = build_asset()
        self.server_bin = self._find_server()
        self.model_key = None
        self.model_path = None
        self.prepare_error = None
        self._preparing = False
        self._queued = None
        self._lock = threading.Lock()
        if self.cfg.get("external_url"):
            self.state = "external"
        if self.server_bin:
            self.hw["gpus"] = self.list_devices()
        self.recommended, self.reason = recommend(self.hw)

    # ------------------------------------------------------------ status
    @property
    def base_url(self):
        if self.cfg.get("external_url"):
            return self.cfg["external_url"].rstrip("/").removesuffix("/v1")
        return f"http://127.0.0.1:{self.cfg['port']}"

    @property
    def supports_slots(self):
        return not self.cfg.get("external_url") or bool(self.cfg.get("external_slots"))

    def custom(self):
        """The user's own model choice, if any: ("server" | "hf" | "path", value)."""
        if self.cfg.get("external_url"):
            return "server", self.cfg["external_url"]
        if self.cfg.get("hf_model"):
            return "hf", self.cfg["hf_model"]
        if self.cfg.get("model_path"):
            return "path", self.cfg["model_path"]
        return None, None

    def set_custom(self, kind, value, model=None, api_key=None):
        """Use any model: a Hugging Face GGUF, a local GGUF file, or an OpenAI-compatible server."""
        value = (value or "").strip()
        if not value:
            return
        clear = {"hf_model": None, "model_path": None, "external_url": None, "external_model": None, "external_api_key": None}
        new = dict(clear)
        if kind == "hf":
            new["hf_model"] = value.removeprefix("https://huggingface.co/")
        elif kind == "path":
            new["model_path"] = value
        elif kind == "server":
            new.update(external_url=value, external_model=(model or "").strip() or None,
                       external_api_key=(api_key or "").strip() or None)
        else:
            return
        self.cfg.update(new)
        save_choice(**new)
        if kind == "server":
            self.stop()
            self.state, self.detail = "external", ""
            self.model_key = self.cfg["external_model"]
        else:
            if self.state == "external":
                self.state = "idle"
            self.ensure_started(None)

    def choose_builtin(self, key):
        clear = {"hf_model": None, "model_path": None, "external_url": None, "external_model": None, "external_api_key": None}
        self.cfg.update(clear)
        save_choice(**clear)
        if self.state == "external":
            self.state = "idle"
        self.ensure_started(key)

    def status(self):
        kind, value = self.custom()
        key = self.model_key or (Path(self.cfg["model_path"]).stem if self.cfg.get("model_path") else None) \
            or self.cfg.get("hf_model") or self.cfg.get("model") or self.recommended
        m = MODELS.get(key, {})
        label = m.get("label", key)
        if kind == "server":
            label = f"{self.cfg.get('external_model') or 'Model'} at {self.cfg['external_url']}"
        return {
            "state": self.state, "detail": self.detail, "progress": self.progress,
            "hardware": self.hw, "backend": self.backend, "profile": PROFILE,
            "recommended": self.recommended, "reason": self.reason,
            "model": key, "model_label": label,
            "installed": {k: self.model_installed(k) for k in MODELS},
            "models": {k: {kk: v[kk] for kk in ("label", "size_gb", "mem_gb", "about")} for k, v in MODELS.items()},
            "external_url": self.cfg.get("external_url"),
            "custom": {"kind": kind, "value": value, "model": self.cfg.get("external_model"),
                       "has_key": bool(self.cfg.get("external_api_key"))} if kind else None,
            "context": self.cfg.get("context"),
            "engine": bool(self.server_bin),
            "note": self.prepare_error,
        }

    # ------------------------------------------------------------ engine
    def _find_server(self):
        exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        if RUNTIME.exists():
            hits = sorted(RUNTIME.rglob(exe))
            if hits:
                return hits[-1]
        found = shutil.which(exe)
        return Path(found) if found else None

    def install_engine(self):
        self.state, self.detail = "installing", f"Downloading llama.cpp ({self.backend} build)"
        # A fixed, tested build by default (reproducible). Set "llama_tag": "latest" in llm_config.json
        # to use the newest build instead (llama.cpp marks its builds as pre-releases).
        tag = self.cfg.get("llama_tag") or PINNED_TAG
        if tag == "latest":
            tag = PINNED_TAG
            try:
                for rel in _get_json("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20"):
                    if any(a["name"].endswith(self.asset) for a in rel.get("assets", [])):
                        tag = rel["tag_name"]
                        break
            except Exception:
                pass
        name = f"llama-{tag}-{self.asset}"
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}"
        dest_dir = RUNTIME / tag
        dest_dir.mkdir(parents=True, exist_ok=True)
        archive = RUNTIME / name
        self._download(url, archive)
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest_dir)
        else:
            with tarfile.open(archive) as t:
                t.extractall(dest_dir)
        archive.unlink(missing_ok=True)
        if sys.platform == "darwin":
            subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(dest_dir)], capture_output=True)
        self.server_bin = self._find_server()
        if not self.server_bin:
            raise RuntimeError(f"llama-server not found in {name}")
        if sys.platform != "win32":
            for f in self.server_bin.parent.iterdir():
                if f.is_file():
                    f.chmod(f.stat().st_mode | 0o111)
        self.hw["gpus"] = self.list_devices()
        self.recommended, self.reason = recommend(self.hw)

    def list_devices(self):
        """GPUs llama.cpp can use, with memory, e.g. 'Vulkan0: <GPU name> (16304 MiB, 15800 MiB free)'."""
        try:
            out = subprocess.run([str(self.server_bin), "--list-devices"], capture_output=True, text=True,
                                 timeout=60, cwd=self.server_bin.parent).stdout
        except Exception:
            return []
        gpus = []
        for line in out.splitlines():
            m = re.match(r"\s+(\S+): (.+) \((\d+) MiB, (\d+) MiB free\)", line)
            if m and int(m.group(3)) > 0:  # skip pseudo-devices like Apple's Accelerate (BLAS, 0 MiB)
                gpus.append({"id": m.group(1), "name": m.group(2), "total_mb": int(m.group(3)), "free_mb": int(m.group(4))})
        return gpus

    # ------------------------------------------------------------ model files
    def _model_files(self, key):
        m = MODELS[key]
        tree = _get_json(f"https://huggingface.co/api/models/{m['repo']}/tree/main?recursive=true")
        q = m["quant"].lower()
        files = [f for f in tree if f.get("type") == "file" and f["path"].lower().endswith(".gguf")
                 and q in f["path"].lower() and "mmproj" not in f["path"].lower()]
        if not files:
            avail = sorted({re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", f["path"].split("/")[-1])
                            for f in tree if f["path"].endswith(".gguf")})
            raise RuntimeError(f"{m['quant']} not found in {m['repo']}. Available: {', '.join(avail[:12])}")
        return sorted(files, key=lambda f: f["path"])

    def model_installed(self, key):
        d = MODELS_DIR / MODELS[key]["repo"].replace("/", "__")
        q = MODELS[key]["quant"].lower()
        return d.exists() and any(q in p.name.lower() and not p.name.endswith(".part") for p in d.rglob("*.gguf"))

    def _local_model(self, key):
        d = MODELS_DIR / MODELS[key]["repo"].replace("/", "__")
        q = MODELS[key]["quant"].lower()
        hits = sorted(p for p in d.rglob("*.gguf") if q in p.name.lower() and "mmproj" not in p.name.lower())
        return hits[0] if hits else None

    def download_model(self, key):
        m = MODELS[key]
        files = self._model_files(key)
        total = sum(f.get("lfs", {}).get("size", f.get("size", 0)) for f in files)
        self.state, self.detail = "downloading", f"Downloading {m['label']} ({total / 1e9:.1f} GB, one time)"
        base = MODELS_DIR / m["repo"].replace("/", "__")
        done_before = 0
        for f in files:
            dest = base / f["path"]
            size = f.get("lfs", {}).get("size", f.get("size", 0))
            if dest.exists() and dest.stat().st_size == size:
                done_before += size
                continue
            url = f"https://huggingface.co/{m['repo']}/resolve/main/{f['path']}"
            self._download(url, dest, offset_total=(done_before, total))
            done_before += size

    def _download(self, url, dest, offset_total=None):
        """Resumable download with progress."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={**UA, **({"Range": f"bytes={have}-"} if have else {})})
        with urllib.request.urlopen(req, timeout=60) as r:
            if have and r.status != 206:
                have = 0  # server ignored the range; start over
            size = int(r.headers.get("Content-Length", 0)) + have
            before, total = offset_total or (0, size)
            with open(part, "ab" if have else "wb") as out:
                done = have
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    self.progress = {"done": before + done, "total": total}
        part.rename(dest)
        self.progress = None

    def prepare(self):
        """First launch: fetch the (small) engine in the background so the GPU can be detected
        and the right model recommended. Models themselves are only downloaded when asked."""
        if self.server_bin or self.cfg.get("external_url"):
            return

        def run():
            try:
                self.install_engine()
                self.state, self.detail = "idle", ""
            except Exception as e:
                self.state, self.detail = "idle", ""
                self.prepare_error = f"Couldn't fetch the llama.cpp engine yet ({e}). It will be retried when you start a model."
            if self._queued is not None:
                key, self._queued = self._queued, None
                self.ensure_started(key or None)
        self._preparing = True
        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------ server process
    def ensure_started(self, key=None):
        """Install/download/start as needed. Runs in a background thread; poll status()."""
        if self.cfg.get("external_url"):
            self.state = "external"
            return
        with self._lock:
            if self.state == "installing" and self._preparing:
                self._queued = key or ""      # start as soon as the engine is in
                self._preparing = False
                return
            if self.state in ("installing", "downloading", "starting"):
                return
            self.state = "starting"
        threading.Thread(target=self._start, args=(key,), daemon=True).start()

    def _start(self, key):
        try:
            custom = self.cfg.get("model_path") if not key else None
            hf = self.cfg.get("hf_model") if not key else None
            if hf:
                custom = hf
                key, label = hf.split("/")[-1], hf
            elif custom:
                self.model_path = Path(custom).expanduser()
                if not self.model_path.exists():
                    raise RuntimeError(f"model_path {self.model_path} does not exist")
                key, label = self.model_path.stem, self.model_path.name
            else:
                key = key or self.cfg.get("model") or self.recommended
                if key not in MODELS:
                    raise RuntimeError(f"unknown model '{key}'")
                label = MODELS[key]["label"]
                if key != self.cfg.get("model") or self.cfg.get("model_path"):
                    self.cfg["model"], self.cfg["model_path"] = key, None
                    save_choice(model=key, model_path=None)
            if not self.server_bin:
                self.install_engine()
            if not custom and not self.model_installed(key):
                self.download_model(key)
            self.stop()
            self.state, self.detail = "starting", (f"Fetching {label} from Hugging Face (first time only) and loading it"
                                                   if hf else f"Loading {label} into memory")
            self.model_key = key
            if not custom:
                self.model_path = self._local_model(key)
            KV_DIR.mkdir(parents=True, exist_ok=True)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            source = ["-hf", hf] if hf else ["-m", str(self.model_path)]
            args = [str(self.server_bin), *source,
                    "--host", "127.0.0.1", "--port", str(self.cfg["port"]),
                    "-c", str(self.cfg["context"]), "-np", "1",
                    "--jinja", "--reasoning-format", "deepseek",
                    "--slot-save-path", str(KV_DIR),
                    "--cache-ram", "2048",
                    "--alias", key, "--no-webui", "--fit", "on"]
            gpus = self.hw.get("gpus") or []
            extra = [str(a) for a in self.cfg.get("extra_args", [])]
            if len(gpus) > 1 and not any(a in ("--device", "-dev") for a in extra):
                args += ["--device", max(gpus, key=lambda g: g["total_mb"])["id"]]
            args += extra
            log = open(LOG_DIR / "llama-server.log", "w")
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            env = dict(os.environ, LLAMA_CACHE=str(ROOT / "models" / "hf"))   # -hf downloads stay in this folder
            self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=self.server_bin.parent,
                                         creationflags=flags, env=env)
            t0 = time.time()
            while time.time() - t0 < (6 * 3600 if hf else 900):
                if self.proc.poll() is not None:
                    tail = (LOG_DIR / "llama-server.log").read_text(errors="replace")[-1500:]
                    raise RuntimeError(f"llama-server exited (code {self.proc.returncode}). Log tail:\n{tail}")
                try:
                    with urllib.request.urlopen(self.base_url + "/health", timeout=2) as r:
                        if json.load(r).get("status") == "ok":
                            break
                except Exception:
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError("llama-server did not become ready within 15 minutes")
            self.state, self.detail = "ready", f"{label} running on {self._device_summary()}"
        except urllib.error.URLError as e:
            self.state = "error"
            self.detail = (f"Download failed: {getattr(e, 'reason', e)}. Check the internet connection and press the "
                           f"button again; downloads resume where they stopped.")
            self.progress = None
        except Exception as e:
            self.state, self.detail = "error", str(e)
            self.progress = None

    def _device_summary(self):
        log = (LOG_DIR / "llama-server.log").read_text(errors="replace") if (LOG_DIR / "llama-server.log").exists() else ""
        gpus = self.hw.get("gpus") or []
        if gpus:
            g = gpus[0]
            where = f"{g['name']} ({g['total_mb'] / 1024:.0f} GB, {self.backend})"
        else:
            where = "CPU"
        m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
        if m and gpus:
            where += f", {m.group(1)}/{m.group(2)} layers on GPU"
        return where

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
