# Flybrain

A whole fruit fly's nervous system, simulated live and shown in 3D, with a body it moves and an AI
that can teach it new movements.

- **The nervous system** is BANC, the first connectome of a fly's brain *and* nerve cord together
  (166,029 neurons, 11.5 million connections, from one adult female; Bates, Phelps, Kim, Yang et al.
  2026). Every neuron sits at its real position: the brain on top, the ventral nerve cord (the fly's
  spinal cord) below. It includes the ~800 motor neurons that drive the legs, wings, neck and
  proboscis, each annotated with the muscle it innervates.
- **The body** is a 3D fly on a floating ball, like a fly in a neuroscience rig. Its joints move only
  when their motor neurons fire: motor neurons → muscles → joints. Show it a threat and the giant
  fibre fires its jump muscle; touch sugar to its mouth and MN9 extends the proboscis. For every
  behaviour you get the pathway that caused it, traced neuron by neuron and lit up in 3D.
- **Fly, the agent**: an open language model running on your own computer (talking, reasoning,
  designing systems, writing and running code), with Laya as a fast "System 1" in front of it. The
  fly is its body. Ask it to do a movement the fly has never done, like tapping a key, and it plans
  the movement, the engine searches the wiring for the neurons that produce it, and the fly
  practises while you watch. The result is saved as a **skill**: the next time, it's instant.

## Setup

### What you need

- **macOS 14+** (Apple silicon recommended), **Windows 10/11**, or **Linux** (x64 or arm64).
- **16 GB of RAM** or more. The simulation itself needs about 3 GB; the rest is for the language model.
- About **5 GB of disk** for the app, plus the model you pick (3 to 23 GB).
- An internet connection for the first run. Afterwards everything runs offline.
- Nothing else: no Python, Git or GPU drivers to install. The app fetches its own tools into its folder.

### 1. Get the code

```bash
git clone https://github.com/<you>/flybrain.git
cd flybrain
```

(or download the ZIP from GitHub and unzip it anywhere).

### 2. Start it

| Machine | How |
|---|---|
| Mac | Double-click `Start Flybrain.command`, or run `./run.sh` in Terminal |
| Windows | Double-click `run.bat` |
| Linux | `./run.sh` |

The first start takes a few minutes. Inside the folder, it:

1. installs [uv](https://docs.astral.sh/uv/) (a small Python manager), Python 3.12 and the Python
   packages (about 1 GB, mostly PyTorch for Laya);
2. downloads the BANC connections (about 300 MB, from the BANC project's public Google Cloud bucket)
   and packs them for the simulator (under a minute);
3. downloads the llama.cpp engine for your machine (about 50 MB).

Then it opens http://localhost:8765 in your browser. Stop it with Ctrl+C in the terminal window (or
close the window). Later starts take a few seconds.

### 3. Pick a language model

The simulation and the body work without a model. Fly, the chat agent, needs one. In the **Chat**
tab, the app suggests a model for your hardware; press **Download and start**. Any other model works
too: open **Use another model** in the same place (details under "Any model" below).

### Troubleshooting

- **macOS says the file "can't be opened" or "Apple could not verify" it.** Files downloaded from the
  internet are quarantined. Right-click `Start Flybrain.command` and choose Open, or run once in
  Terminal: `xattr -dr com.apple.quarantine /path/to/flybrain`
- **Port 8765 is busy.** Start with another port: `PORT=8800 ./run.sh` (Windows: `set PORT=8800` then
  `run.bat`).
- **A download failed.** Run it again; downloads resume.
- **The top bar shows less than 1× real time.** Big inputs (odours, whole-eye vision) are heavy for
  one CPU core; lower the speed in the Senses tab or switch the input off.
- **Start from scratch.** Delete the hidden `.cache`, `.tools` and `.venv-*` folders (and
  `server/data/connectome_banc888.npz`). Your conversations and learned skills are in `data/`.

## What you're looking at

- **The fly** (left or top) is the body. Glowing parts are being driven by their motor neurons.
- **The nervous system** (right or bottom): dots are neurons, coloured by region (brain regions on
  top; the nerve cord's flight, front-, middle- and hind-leg and abdominal parts below). A neuron
  glows when it fires. Cyan marks the pathway being explained or the neurons a skill is driving.
- **Neurons → muscles** (right) shows how hard each body part's motor neurons are driving its
  muscles right now, plus the brain's main descending command neurons. Click a row to find those
  neurons.
- **Skills** (right): movements the fly has learned. ▶ replays one.
- **Senses** (left tab) drive real sensory neurons with Poisson spike trains: taste, Johnston's
  organ, olfactory, thermo- and hygrosensory neurons, leg touch, vision.
- **Click any neuron** to see what it is, what it connects to, and to stimulate or silence it.
  "Stimulate by cell type" does the same for a whole type (try `LPLC2`, `DNa02`, `KCab`).

The buttons at the top right switch between fly + brain, fly only and brain only. Press **▶ Demo**
under the fly for a tour (press it again to stop).

## The fly's body

Every frame, in simulated time:

1. **Stimuli → sensory neurons.** A dark object swooping at one eye fires that eye's LPLC2 looming
   detectors, at a rate that follows how fast the shadow grows. A sugar or bitter drop on the
   proboscis fires taste neurons. Dust on the antennae fires Johnston's organ neurons. "Activate
   neurons" drives descending neurons directly, like optogenetics.
2. **The whole connectome runs**, brain and nerve cord together.
3. **Motor neurons → muscles → joints** (`server/motor.py`). Motor neurons driving the same muscle on
   the same side form a pool (237 pools). A pool's firing rate sets its muscle's activation (with a
   twitch-like time constant), and each joint moves by agonist minus antagonist around a resting
   pose: trochanter flexors lift the femur and extensors press it down, the tibia flexors fold the
   leg and the extensor reaches, the tarsus depressor grips, the thorax-coxa muscles swing the leg.
   All 805 motor neurons are mapped to a body part: six legs, two wings, two halteres, neck,
   proboscis and pharynx, antennae, abdomen, and internal organs (crop, spiracles, salivary gland,
   uterus, eye), each a row in the Neurons → muscles panel.
   **How far joints move:** every leg joint rests in the middle of the range a real fly uses when
   walking (recorded kinematics). Everyday muscle activation moves it within that range; only
   near-full activation goes further, to 1.2× the walking range (1.7× for the front legs, which also
   groom and reach), and hard limits stop it there, so no amount of drive flails a leg or bends it
   backwards. Skills are held to the same limits: a plan asking for more is clipped, and Fly is told
   which joint stopped it.
   The ball turns under the feet that are down, in the fly's own frame (its head points forward):
   legs sweeping backwards roll it for forward walking, forwards for backward walking, and uneven
   left/right strokes spin it for turns.
   Wings: the indirect flight muscles' motor neurons (DLM, DVM) set flight power, the steering
   muscles set stroke amplitude and posture. Neck muscles turn the head; MN9 and the proboscis
   muscles extend the proboscis; the pharyngeal muscles pump. Nothing is animated by hand except the
   wingbeat waveform itself (a 200 Hz oscillation of the flight muscles, shown slowed down).

What the wiring produces, checked live and with `tools/validate.py`:

| Input | What happens | Pathway (traced live) |
|---|---|---|
| Sugar on the proboscis | proboscis extends and stays out | sugar taste neurons → ... → MN9 |
| Sugar + bitter | feeding suppressed | bitter neurons inhibit the feeding pathway |
| Threat from either side | escape jump, a burst of wingbeats, then backing away | LPLC2 → giant fibre → jump motor neuron (TTMn); LPLC2 → DNp06 → wing power motor neurons; Moonwalker neurons → walking backwards |
| Activate P9 / MDN / DNa02 / aDN | walks forwards / backwards / turns / grooms its antennae | the command neurons you drove, with the rhythm from a built-in skill (below) |
| Heat or cold | turns and walks (thermal avoidance); the flight muscles' motor neurons fire, but it stays on the ball | hot/cool sensors → VP2/VP3 projection neurons → DNb05 → DNp31 |
| Odours, humidity | the brain responds (antennal lobe, lateral horn, mushroom body), the body stays still | |
| Antenna dust | leg twitches only; the touch → grooming pathway is too weak in this wiring to reach the aDN command neurons | |

Three things to know:

- **Gap junctions.** The giant fibre drives the jump motor neuron mainly through electrical synapses,
  which an electron-microscopy connectome cannot see. Those few known ones (giant fibre → TTMn and
  PSI) are added by hand, and marked as such.
- **Feet on the ball stop flight.** In real flies, tarsal contact inhibits flight. The body follows
  that rule: the wings only beat after the giant-fibre jump has lifted the legs off, for as long as
  the flight power motor neurons keep firing.
- **Rhythms come from built-in skills.** Stepping and grooming rhythms are made by nerve-cord
  circuits whose ion-channel dynamics a leaky integrate-and-fire model doesn't have, so the command
  neurons alone only give twitches. The app ships five skills learned by the skill engine (below) on
  this same connectome: walking forwards and backwards, turning left and right, and antennal
  grooming (`server/data/skills_builtin.json`). When the brain's own command neurons for one of
  them fire (P9 or DNg97, MDN, DNa01/DNa02, aDN), the body replays that skill's neuron drive as the
  rhythm. Neurons driven by a skill don't count as commands, so a skill can't trigger itself.

The **Why?** card appears whenever the body starts a behaviour. It starts at the motor neurons (or
command neurons) involved and walks backwards through the wiring: at each step, the cell type sending
the most excitation right now (firing rate × synapse count), until it reaches a sensory neuron. It
also names the strongest inhibitor pushing against it. The same chain lights up cyan.

## Skills: teaching the fly new movements

Ask Fly in chat to do something its body has never done ("tap your front-left leg", "wave", "type
hi"). It works like this (`server/skills.py`):

1. **Plan.** The LLM describes the movement as keyframes of motor controls: per leg `swing`, `lift`,
   `reach`, `grip`, `spread`; wings `power`, `extend`, `stroke`; head `yaw`/`pitch`/`roll`;
   proboscis; antennae. Values from -1 to 1 over time.
2. **Search the wiring (the slow part, done once).** Each control direction maps to muscle pools
   (e.g. "front-left lift up" = the front-left trochanter flexor motor neurons). For every neuron, the
   engine computes its influence on every pool through up to three synapses (signed, normalised by
   each target's total input). Good drivers push their target hard and its antagonist and everything
   else weakly. It tries **descending neurons** first (brain → nerve cord), then **premotor
   nerve-cord interneurons**, and only as a last resort the motor neurons themselves.
3. **Practise, visibly.** The chosen neurons are driven on the live connectome, and the engine
   measures what the muscles actually did. It corrects the drive over the whole movement from the
   error (iterative learning control): harder where a muscle lagged, gentler where it overshot. A
   driver that can't move its muscle even at full drive is replaced one level down; one that makes
   other body parts move is swapped for a more specific one. If a movement is still only roughly
   right after three tries, that muscle's own motor neurons are added as a fine trim for the
   remaining error, while the upstream drivers keep doing the bulk of it. Up to eight tries, each starting from a
   quiet nervous system; you see the tries, the drivers and the scores in the Skills panel, and the
   driven neurons light up.
4. **Save.** The best version is stored in `data/skills/skills.json` (this computer only, not in
   git): the driver neurons (with their BANC ids), their firing schedule, the plan and the score.

After that, `do_skills` just replays the stored drive: no search, no practice. To type a word, Fly
learns one small tap skill per leg once, then chains them. The built-in walking, turning and grooming
skills are listed too, and Fly can use them like its own.

This is the same idea as the KV cache, one level down: work that is expensive the first time is
kept and reused. And the skill list is deliberately not in the LLM's system prompt. Every new skill
would change the prompt's start and throw away the cached tokens, so Fly looks skills up with a tool
(`list_skills`) and the answer lands at the end of the conversation, where it costs only its own
tokens.

A score of 1.0 means the muscles followed the plan exactly; 0.6-0.8 is typical, since the network
answers the same drive a little differently each time, as neurons do.

## Fly, the agent

### One folder, any machine

The same folder runs on a Mac and on a Windows PC (and Linux). At start-up the app detects what it
is running on and keeps everything machine-specific apart, per *profile* (`macos-arm64`,
`windows-x64`, `linux-x64`, ...): the Python environment (`.venv-<profile>`), the llama.cpp build
(`runtime/<profile>/`), the model chosen for that machine, and its saved KV caches. Model files are
portable and shared (`models/`). Everything the app installs, including uv, Python, packages and
the Laya checkpoint, stays inside the folder (`.tools/`, `.cache/`); nothing is installed system-wide.

The LLM runs in [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`, and the app
fetches the right build:

| Machine | llama.cpp build | Why |
|---|---|---|
| Mac with Apple silicon | `macos-arm64` (Metal) | Uses the GPU and unified memory directly |
| Windows PC with an AMD, NVIDIA or Intel GPU | `win-vulkan-x64` (Vulkan) | On recent AMD cards (RDNA4), Vulkan generates tokens ~35% faster than ROCm and needs no ROCm install |
| Linux | `ubuntu-vulkan-x64` / `ubuntu-arm64` | |

It lists the GPUs llama.cpp can see, with their memory, and picks a model:

| Model | Download | Picked for |
|---|---|---|
| Qwen3.6 35B-A3B (4-bit) | 22.4 GB | GPUs with 12 GB+ plus 40 GB+ system RAM; Macs with 48 GB+. Mixture-of-experts with only 3B active per token, so experts that don't fit on the GPU sit in system RAM and it stays fast |
| Qwen3.6 35B-A3B (3-bit) | 16.8 GB | Macs with 32-36 GB |
| Qwen3.6 27B | 18 GB | GPUs with 20 GB+ |
| Qwen3.5 9B | 6.5 GB | 16 GB Macs; GPUs with 8 GB+ |
| Qwen3.5 4B | 3 GB | 8 GB machines |

`--fit on` makes llama.cpp place layers and experts to fit the GPU memory it actually has. If a
machine shows more than one GPU (say a CPU's integrated graphics next to a graphics card), the app
uses the one with the most memory. You can switch models any time ("Change" in the Chat tab); the
choice is remembered for that machine only.

### Any model

The built-in list is just a starting point. Fly works with any chat model that can call tools. In the
Chat tab, **Use another model** takes one of:

| Option | Example | Runs |
|---|---|---|
| Any GGUF model on Hugging Face | `unsloth/gemma-3-12b-it-GGUF:Q4_K_M`, `Qwen/Qwen2.5-7B-Instruct-GGUF` | here, with llama.cpp; downloaded once into `models/hf/` |
| A GGUF file you already have | `/Users/you/models/my-model.gguf`, `D:\models\my-model.gguf` | here, with llama.cpp |
| Any OpenAI-compatible server | Ollama `http://localhost:11434/v1`, LM Studio `http://localhost:1234/v1`, vLLM, or a hosted API URL, plus the model name and an API key if it needs one | wherever that server runs |

The choice is remembered for that machine only, in `data/settings/` (which never goes to git; an API
key you enter is stored there too). Pick a built-in model again to switch back.

Or set it in a file. Create `llm_config.json` next to this README (it is git-ignored). Top-level keys
apply on every machine; a section named after a profile applies only there:

```json
{
  "context": 32768,
  "macos-arm64": {"model": "qwen3.5-9b"},
  "windows-x64": {"hf_model": "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL", "extra_args": ["--n-cpu-moe", "20"]},
  "linux-x64": {"external_url": "http://localhost:11434/v1", "external_model": "qwen3:14b"}
}
```

| Key | Meaning |
|---|---|
| `model` | a built-in choice: `qwen3.6-35b-a3b`, `qwen3.6-35b-a3b-q3`, `qwen3.6-27b`, `qwen3.5-9b`, `qwen3.5-4b` |
| `hf_model` | any Hugging Face GGUF repo, optionally `:QUANT` |
| `model_path` | a GGUF file on disk |
| `external_url`, `external_model`, `external_api_key` | an OpenAI-compatible server instead of the built-in engine |
| `external_strict` | `true` for hosted APIs that reject llama.cpp's extra sampling settings (automatic when an API key is set) |
| `context` | context window in tokens |
| `extra_args` | passed straight to llama-server |

Notes: the model needs tool calling for Fly to code and use its body (most recent instruct models
have it). KV-cache saving and restoring (below) needs llama.cpp's own server, so with an external
server each conversation is re-read after a restart. "Thinking" is shown for models that stream it
(`reasoning_content` or `reasoning`).

### System 1 and System 2

Every message first goes to **Laya** ([Convai](https://github.com/NandhaKishorM/laya), Apache 2.0),
a 421M-parameter decision model that answers typed questions in one forward pass with calibrated
probabilities and no text generation. It decides the intent (chat, question, code, design, brain),
how much reasoning the reply needs, whether tools are needed, and the tone. The chat shows each
decision. Depth switches the LLM's thinking on or off, which saves time on quick replies. Laya runs on
the GPU where PyTorch supports it (Apple silicon, NVIDIA) and otherwise on the CPU; either way it takes tens of milliseconds. Its
checkpoint downloads from Hugging Face the first time the agent starts. It is used zero-shot here;
its authors report big accuracy gains from fine-tuning on your own decisions, which is a natural
next step. If Laya can't load, simple keyword rules stand in and the chat labels them "rules".

The **LLM** (Qwen) then answers, streaming its thinking (collapsible) and its reply. It has tools:
read/write/edit files and run commands in the `workspace/` folder (commands ask for your approval
unless you tick Auto-approve), long-term notes (remember/recall), and its body
(body_experiment, stimulate_senses, brain_status, list_skills, learn_skill, do_skills).

### KV caching

- **Reuse within a conversation.** The system prompt and tools never change and turns are only
  appended, so each request reuses the keys/values already computed and only processes new tokens.
  Every reply shows "KV cache reused N · processed M new".
- **Persistence.** After each turn the KV cache is saved to `data/kv/<conversation>.bin`. Reopen a
  conversation (History) or restart the app and it is restored from disk in milliseconds instead
  of re-reading the whole history.
- **Context window.** When a conversation nears the model's context limit, the oldest turns leave
  the model's working memory in one step (so the cache restarts rarely), and the chat says so.

### The body link

What the agent does drives chosen neurons; the connectome does the rest. The choice of neurons is a
design decision, listed live in the "Body link" panel:

| Agent | Fly neurons |
|---|---|
| Reading your message | Johnston's organ auditory neurons (the fly's ear) |
| Laya hears praise / frustration | sugar / bitter taste neurons |
| LLM deliberating (thinking tokens) | PFN neurons into the central complex, the action-selection hub |
| Recalling a memory or a conversation | a sparse, memory-specific 5% of Kenyon cells (mushroom body) |
| Code runs / fails | sugar (reward) / bitter (aversive) |
| stimulate_senses tool | any of the senses |
| body_experiment tool | a full experiment on the body (threat, taste, dust, activating descending neurons); the agent gets back what the body did and the traced pathway, and you watch it happen |
| learn_skill / do_skills tools | the neurons the skill engine found for a movement (see Skills) |

The other way round, each message the LLM receives ends with the body's state (behaviour readouts,
busiest regions), so it knows when its fly is feeding or about to escape. One circuit you'll notice:
hearing at high intensity triggers the escape neuron, because Johnston's organ feeds the giant
fibre, as in real flies. So hearing is kept gentle.

To be clear: the fly brain does not do the reasoning or the coding; the LLM does. The brain is the
body, and its responses come from its real wiring.

## The brain model

A leaky integrate-and-fire network with the equations and constants of
[Shiu et al., *Nature* 2024](https://www.nature.com/articles/s41586-024-07763-9), which validated
it against real flies on the FlyWire brain. Here it runs on BANC's brain and nerve cord:

- **Weights**: synapse count × 0.275 mV × 1.5. BANC's synapse detector finds about half as many
  synapses per neuron as FlyWire's (206 vs 393 inputs per neuron), which the published constants were
  fitted to; ×1.5 restores the sugar → MN9 and looming → giant fibre responses without making the
  network more excitable than the original.
- **Signs**: each neuron's transmitter (verified where known, else BANC's prediction). GABA,
  glutamate (via GluCl in flies) and histamine inhibit; the rest excite.
- **Stabilised mode** (default): odours tip the plain model into firing that never stops (antennal
  lobe local neurons and Kenyon cells exciting each other). `tools/find_loops.py` probes every sense
  and command neuron and finds the ~13,000 neurons that keep firing after the input is gone. Only
  those get two things real neurons have: spike-frequency adaptation (6 mV per spike, 500 ms decay),
  and short-term synaptic depression of their outputs (each spike uses 70% of the available
  transmitter, recovering over 400 ms). Adaptation stops the endless firing; depression stops the
  burst at input onset from igniting thousands of neurons for a moment, which would otherwise twitch
  the wings and legs whenever any smell or temperature change arrived. With both, every input goes
  quiet within half a second of being switched off, and the behaviours in the table above hold.
  Only interneurons are eligible: sensory, descending and motor neurons carry the signals in and out,
  and slowing them would blunt the behaviours themselves.
  **No adaptation** (the other mode) is the plain model.

Speed: an exact event-driven scheme. A neuron whose voltage and input are both below threshold
can't fire until new input arrives, so it's left alone and caught up with the closed-form solution
when its next spike arrives. Only neurons near threshold are stepped every 0.1 ms. Taste, touch and
skills run faster than real time on one CPU core; odours and big visual inputs are heavier, and the
top bar shows when the CPU is the limit.

## Control API

Everything the buttons do goes through one command endpoint:

```
POST http://localhost:8765/api/command
{"cmd": "preset", "key": "sugar", "rate": 150}    # rate 0 switches it off
{"cmd": "type",   "type": "LPLC2", "rate": 100}
{"cmd": "neuron", "idx": 12345, "rate": 150}
{"cmd": "block",  "idx": 12345, "on": true}        # silence a neuron
{"cmd": "clear"} | {"cmd": "reset"} | {"cmd": "pause", "on": true}
{"cmd": "mode",   "value": "stable" | "paper"}
{"cmd": "speed",  "value": 0.5}
{"cmd": "body",   "action": "loom", "side": "left"}   # also sugar, bitter, dust, p9, mdn,
                                                     # dna02_l, dna02_r, gf, adn, stop, reset
{"cmd": "skill",  "action": "play", "name": "tap_front_left"}   # also stop, forget

GET /api/status         behaviour levels, region activity, top firing neurons
GET /api/neuron/{idx}   identity, transmitter, strongest inputs and outputs
GET /api/search?q=...   cell types by name
GET /api/presets        all senses and behaviour readouts with their neuron ids
GET /api/body/pathways  motor neurons per body part, descending command neurons
GET /api/skills         learned skills, and the motor controls a plan can use

{"cmd": "chat", "text": "..."} | {"cmd": "stop"} | {"cmd": "new_chat"} | {"cmd": "model_setup", "model": "qwen3.5-9b"}
{"cmd": "model_custom", "kind": "hf" | "path" | "server", "value": "...", "model": "...", "api_key": "..."}
GET /api/agent/status   model, hardware, Laya, conversation
```

The browser gets a WebSocket stream (`/ws`) of every spike (binary), status JSON 5× a second, the
body's state (joint offsets, muscle activation per body part) every frame, a `decision` message
with the traced pathway whenever the body starts a behaviour, and `skill` messages while a skill is
learned or performed.

## Files

```
server/app.py          web server, simulation loop, API
server/sim.py          the network simulator (numba)
server/connectome.py   downloads + packs the BANC connections (first run)
server/motor.py        motor neurons -> muscles -> joints
server/body.py         stimuli -> sensory neurons; what the body is doing
server/skills.py       learning new movements: wiring search, practice, skill library
server/agent/          the AI: runtime.py (llama.cpp setup), models.py, llm.py, system1.py (Laya),
                       agent.py (loop, conversations, KV cache), tools.py, bridge.py (body link)
server/data/           neuron labels and signs, senses, readouts, motor pools, loop-neuron mask,
                       built-in skills (skills_builtin.json)
web/                   the 3D viewer (three.js, bundled); web/fly.js is the 3D fly, web/chat.js the chat
web/data/fly_body.*    the fly's meshes, skeleton and resting pose (from NeuroMechFly)
tools/fetch_banc.sh    downloads the BANC annotation files build_banc.py needs
tools/build_banc.py    rebuilds web/data and server/data from the BANC files
tools/find_loops.py    finds the neurons that get adaptation in stabilised mode
tools/validate.py      checks known behaviours and which body parts each input moves
tools/build_body.py    rebuilds web/data/fly_body.* from the flygym package
tools/mock_llm.py      a scripted fake model server for testing the agent without a download

Created on your machine and kept out of git (see .gitignore): .tools/, .cache/ (uv, Python, the BANC
download), .venv-<profile>/, runtime/ (llama.cpp), models/ (GGUF files), data/ (your conversations,
KV caches, memory notes, learned skills, logs, per-machine settings), workspace/ (the agent's
code), llm_config.json (your overrides), server/data/connectome_*.npz (built from the download).
```

## Data and credit

- Connectome: BANC, Bates, Phelps, Kim, Yang et al., *Nature* 2026, "Distributed control circuits
  across a brain-and-cord connectome"; data from
  [flywire.ai/banc](https://flywire.ai) via the public bucket described in
  [sjcabs/fly_connectome_data_tutorial](https://github.com/sjcabs/fly_connectome_data_tutorial).
  Cross-dataset cell types (FAFB/FlyWire, MANC) come with BANC's annotations.
- Model: Shiu et al., *Nature* 2024.
- Fly body: [NeuroMechFly v2 / flygym](https://github.com/NeLy-EPFL/flygym) (Apache 2.0),
  Lobato-Rios et al., *Nature Methods* 2022 and Wang-Chen et al., *Nature Methods* 2024: the body
  meshes and skeleton (simplified here) and a resting pose from recorded walking.
- Muscles and motor neurons: Azevedo et al. 2024 (leg), Lindsay et al. 2017 and Muijres et al. 2014
  (wing steering), Schwarz et al. 2017 and McKellar et al. 2020 (proboscis). Giant fibre gap
  junctions: King and Wyman 1980, Allen et al. 2006.
- Behaviour neurons: LPLC2 and the giant fibre (von Reyn et al. 2014, Ache et al. 2019), P9 (Bidaye
  et al. 2020), MDN (Bidaye et al. 2014), DNa02 (Rayshubskiy et al. 2020), aDN (Hampel et al. 2015).
- LLM runtime: llama.cpp (MIT). Models: Qwen3.5 / Qwen3.6 (Apache 2.0), GGUF builds by Unsloth.
- System 1: Laya by Convai Innovations (Apache 2.0).

Check the BANC and FlyWire data terms before any commercial use.
