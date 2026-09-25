"""Open models the app knows how to run, and how to pick one for this machine."""

# All GGUF, all run by llama.cpp (Metal on Mac, Vulkan on AMD/NVIDIA/Intel GPUs, CPU fallback).
# size_gb is the download; mem_gb is roughly what it needs to run with a 32k context.
MODELS = {
    "qwen3.6-35b-a3b": {
        "label": "Qwen3.6 35B-A3B",
        "repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "quant": "UD-Q4_K_XL",
        "size_gb": 22.4,
        "mem_gb": 25,
        "moe": True,
        "about": "Best quality. Mixture-of-experts: only 3B parameters active per token, so it stays fast "
                 "even when part of it sits in system RAM (ideal for a 16 GB GPU + 64 GB RAM).",
    },
    "qwen3.6-35b-a3b-q3": {
        "label": "Qwen3.6 35B-A3B (3-bit)",
        "repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "quant": "UD-Q3_K_XL",
        "size_gb": 16.8,
        "mem_gb": 19,
        "moe": True,
        "about": "The same model at 3-bit for 32-36 GB Macs: fits the GPU's share of unified memory with room left for everything else. Slightly lower quality than 4-bit.",
    },
    "qwen3.6-27b": {
        "label": "Qwen3.6 27B",
        "repo": "unsloth/Qwen3.6-27B-GGUF",
        "quant": "UD-Q4_K_XL",
        "size_gb": 18,
        "mem_gb": 21,
        "moe": False,
        "about": "Dense 27B: strong coder, slower than the 35B-A3B unless it fits fully in GPU memory.",
    },
    "qwen3.5-9b": {
        "label": "Qwen3.5 9B",
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "quant": "UD-Q4_K_XL",
        "size_gb": 6.5,
        "mem_gb": 8,
        "moe": False,
        "about": "Fast and light. Good for 16 GB Macs, or as a quick option on the PC.",
    },
    "qwen3.5-4b": {
        "label": "Qwen3.5 4B",
        "repo": "unsloth/Qwen3.5-4B-GGUF",
        "quant": "UD-Q4_K_XL",
        "size_gb": 3,
        "mem_gb": 5,
        "moe": False,
        "about": "Smallest useful model, for 8 GB machines.",
    },
}


def recommend(hw):
    """Pick a model for the detected hardware. Returns (key, reason)."""
    ram = hw.get("ram_gb", 0)
    gpus = hw.get("gpus", [])
    vram = max([g["total_mb"] for g in gpus], default=0) / 1024
    if hw.get("os") == "darwin":
        # Apple silicon: the GPU may use about two thirds of unified memory; leave room for the
        # brain simulation, Laya and the browser.
        if ram >= 44:
            return "qwen3.6-35b-a3b", f"{ram:.0f} GB of unified memory fits the 4-bit 35B-A3B"
        if ram >= 28:
            return "qwen3.6-35b-a3b-q3", f"{ram:.0f} GB of unified memory fits the 3-bit 35B-A3B (16.8 GB) with room to spare"
        if ram >= 14:
            return "qwen3.5-9b", f"{ram:.0f} GB of unified memory: the 9B leaves room for everything else"
        return "qwen3.5-4b", f"{ram:.0f} GB of memory: the 4B is the safe choice"
    # Discrete GPU (Windows / Linux). MoE experts can live in system RAM at good speed.
    if vram >= 12 and ram >= 40:
        return "qwen3.6-35b-a3b", f"{vram:.0f} GB GPU + {ram:.0f} GB RAM: attention on the GPU, some experts in RAM"
    if vram >= 20:
        return "qwen3.6-27b", f"{vram:.0f} GB GPU fits the dense 27B"
    if vram >= 8:
        return "qwen3.5-9b", f"{vram:.0f} GB GPU fits the 9B"
    if ram >= 16:
        return "qwen3.5-9b", "no large GPU found: running on CPU"
    return "qwen3.5-4b", "limited memory: smallest model"
