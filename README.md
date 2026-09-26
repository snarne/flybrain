# Flybrain

A whole fruit fly's nervous system, simulated live and shown in 3D, with a body it moves and an AI
that can teach it new movements.

- **The nervous system** is BANC, the first connectome of a fly's brain *and* nerve cord together
  (166,029 neurons, 11.5 million connections, from one adult female; Bates, Phelps, Kim, Yang et al.
  2026). Every neuron sits at its real position: the brain on top, the ventral nerve cord (the fly's
  spinal cord) below. It includes the ~800 motor neurons that drive the legs, wings, neck and
  proboscis, each annotated with the muscle it innervates.
- **The body** is a physical 3D fly (NeuroMechFly, simulated with MuJoCo) free in an arena. Its
  joints move only when their motor neurons fire: motor neurons → muscles → joints, with gravity,
  contact and friction doing the rest. It walks, jumps and flies wherever its neurons take it.
- **Its world** is what it senses. Put food, odours, warm or cool air, humid or dry air, a looming
  threat, a small moving target or turning stripes near it, and the matching sensory neurons in the connectome fire from
  where those things are relative to its antennae, eyes and feet. The legs' sense organs report back
  into the nerve cord every frame, so the loop is closed: world → senses → brain → muscles → body →
  senses. For every behaviour you get the pathway that caused it, traced neuron by neuron and lit up
  in 3D.
- **Walking commanded by its descending neurons**: when the walking command neurons fire (P9 for
  forwards, the moonwalker neurons for backwards, DNa02 for turning), the six legs step in a tripod
  with a real fly's recorded steps, as fast, which way and turning as the neurons say.
- **It learns.** Pair an odour with reward or punishment and the mushroom body's own synapses
  change: dopamine neurons weaken the connections from that odour's Kenyon cells onto the output
  neurons they gate, and not those of other odours.
- **A scoring suite** runs the experiments that were done on real flies (activate the moonwalker
  neurons, show a looming shadow, condition an odour, ...) and scores what the model does, so every
  change can be checked against biology. The same suite runs on **controls**: the wiring randomly
  shuffled (each neuron keeping its number of connections), without inhibition, or with groups of
  neurons silenced, to show what the real wiring contributes.
- **Fly, the agent**: an open language model running on your own computer (talking, reasoning,
  designing systems, writing and running code), with Laya as a fast "System 1" in front of it. The
  fly is its body. Ask it to do a movement the fly has never done, like tapping a key or jumping as
  high as it can, and it plans the movement and says what it should achieve; the engine searches the
  wiring for the neurons that produce it, and the fly practises many variations in background copies
  of itself, scored on what each try achieved. The result is saved as a **skill**: the next time,
  it's instant. It
  can also work as a **neuroscientist** on its own nervous system: look up cell types, trace
  pathways, activate, record and silence neurons, and score the results against real experiments.

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
- **The page says "reconnecting".** The server stopped; `run.sh` / `run.bat` start it again by
  themselves, and the page reconnects (a language model that was running is picked up again rather
  than reloaded). What went wrong is written to `data/logs/server.log`, including hard crashes.
- **The top bar shows less than 1× real time.** Big inputs (odours, whole-eye vision) are heavy for
  one CPU core; lower the speed in the Senses tab or switch the input off.
- **You pulled an update while the app was running.** The top bar shows *Updated · restart*; press it
  (or close and start the app again) so the new code is used. The language model keeps running.
- **Learning a skill takes a while, or the computer gets busy.** Skills practise in background copies
  of the fly: each takes a CPU core and about 1 GB of memory, and the first one takes ~15 s to start.
  The app picks how many from your cores and memory (1 to 4); set `FLYBRAIN_PRACTICE_WORKERS=2` (for
  example) to choose. Fewer tries (`trials`) are faster; more find better movements.
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
- **Agent / Lab** (top): *Agent* is for talking to Fly (chat on the left; skills and the body link on
  the right). *Lab* is for experiments: put things in the fly's world, do things to it, teach it,
  activate or silence neurons, and switch body settings (left), with the motor, region and firing
  panels (right). Food, smells and air patches go where you click: press *Fruit smell*, *Sugar drop*,
  *Warm* and so on, then click the floor (press the button again to put it next to the fly; Esc
  cancels).
- **Senses** (left tab) drive real sensory neurons with Poisson spike trains: taste, Johnston's
  organ, olfactory, thermo- and hygrosensory neurons, leg touch, vision.
- **Click any neuron** to see what it is, what it connects to, and to stimulate or silence it, with
  its whole cell type (how many, how active, its main partners, what the app uses it for) and the
  strongest path from it to any other cell type. "Stimulate by cell type" does the same for a whole
  type (try `LPLC2`, `DNa02`, `KCab`).
- **Fly + activity** (top right) swaps the brain for a recording view: a spike raster of neurons
  sampled from each stage, from sense organs (odour receptors, looming and small-object detectors,
  taste) through the antennal lobe and mushroom body (projection neurons, Kenyon cells, output and
  dopamine neurons) and the command neurons (giant fibre, P9, MDN, DNa02) to the leg and wing motor
  neurons, and each group's firing rate over the last 10 s. Add any cell type to it.

The buttons at the top right switch between fly + brain, fly only, brain only and fly + activity. Press **▶ Demo**
under the fly for a tour (press it again to stop).

## The fly's body

Every frame (30 a second, in simulated time):

1. **The world → sensory neurons** (`server/arena.py`; the neurons found in BANC by
   `tools/build_senses.py`). Everything is sensed from where it is relative to the fly's body:

   | In the world | Sensory neurons | How they fire |
   |---|---|---|
   | Sugar or bitter drop | taste neurons in each leg's tarsi (350); the labellum's when the extended proboscis reaches the drop | each foot standing in the drop tastes it; the fly drinks, and the drop shrinks |
   | Odour source (fruit, CO₂, geosmin, cVA) | olfactory receptor neurons of each antenna (374: Or42b/Or22a/Or59b; Gr21a/Gr63a; Or56a; Or67d) | concentration of a plume at that antenna, so left and right differ near the source |
   | Warm or cool air | hot and cold cells in each arista (28) | temperature at that antenna, and more strongly while it's changing |
   | Humid or dry air | hygrosensory neurons of each antenna (60) | humidity at that antenna |
   | Looming threat | LPLC2 and LC4 of each eye (289) | how fast the object grows on that side of the visual field |
   | Turning stripes | T4/T5 motion detectors of each eye (4,336): front-to-back, back-to-front | the slip between the stripes and the fly's own turning, so turning with them quietens them |
   | Moving target | LC10a small-object detectors of each eye (224) | while a small dark object moves across that eye's view (its size and speed on the retina) |

   The experimenter can also touch a drop to the proboscis, put dust on the antennae, or drive
   descending neurons directly ("Activate neurons", like optogenetics).
2. **The whole connectome runs**, brain and nerve cord together.
3. **Motor neurons → muscles** (`server/motor.py`). Motor neurons driving the same muscle on the same
   side form a pool (237 pools). A pool's firing rate sets its muscle's activation, with a
   twitch-like time constant. All 805 motor neurons are mapped to a body part: six legs, two wings,
   two halteres, neck, proboscis and pharynx, antennae, abdomen, and internal organs (crop,
   spiracles, salivary gland, uterus, eye), each a row in the Neurons → muscles panel.
4. **Muscles → a physical body** (`server/physics.py`). The body is NeuroMechFly (the same model
   the meshes come from), simulated with MuJoCo: a free body with 42 leg joints on the arena floor,
   gravity, friction and adhesive foot pads. Each leg joint is pulled by its agonist and antagonist
   muscles: their difference sets the angle the joint is pulled towards, their sum how stiffly it's
   held (both relaxed = a floppy joint that only holds some tone, both contracted = a stiff one, as
   in insects). Trochanter flexors lift the femur and extensors press it down, the tibia flexors fold
   the leg and the extensor reaches, the thorax-coxa muscles swing it (the coxa's roll moves with the
   swing, as in a walking fly: it has no muscles of its own in the annotation), the adductor pulls it
   in, the femur reductor twists the femur, and the claw muscles (tarsus depressor, long tendon)
   engage the sticky foot pads, which lets a leg pull as well as push. The
   brain and the body run side by side: the body plays out one frame's muscle commands while the
   brain computes the next (a 33 ms sensorimotor delay, about a fly's).
5. **The body → sense organs → nerve cord** (`server/feedback.py`). Each leg's proprioceptors,
   found in BANC by `tools/build_proprio.py` (1,032 neurons), fire from what the physics measured:
   femoral chordotonal organ claw neurons (joint angle, range-fractionated, so they're quiet around the
   standing posture; firing at the slight give of a standing leg set off reflexes that twitched the resting legs), hook neurons (direction of movement), club
   neurons (movement and vibration), hair plates (a joint near the end of its range), campaniform
   sensilla (mostly changes in load) and tarsal bristles (a foot touching down). While the fly makes
   a movement itself, claw and hook output is turned down, as presynaptic inhibition does in real
   flies; otherwise the resistance reflexes they drive fight every step.

**Jumping and flying.** The wings aren't simulated as airfoils, and the jump isn't simulated leg by
leg (in this model's resting stance the middle legs splay sideways, while real flies bring them
under the body first). While the jump muscle (tergotrochanter, driven by the giant fibre) is active
and the middle legs are still on the ground, it pushes the body steeply up and a little forward. The
wings give lift and thrust in proportion to the flight power, the steering muscles turn the fly (a
bigger right stroke turns left), the halteres keep it level and damp its rotation from the moment it
leaves the ground (a stabilising torque; halteres aren't in the neural model), and air drag slows it.
A flying fly turns away from a wall that looms ahead (visual collision avoidance). A fly left on its
back for a second is set back on its feet (a real fly rights itself with its legs; this model can't
make those movements).

The arena is a glass-walled box: the walls push the fly back in, whether it walks or flies into them,
and whatever you put in the world stays inside it.

**How far joints move:** every leg joint rests in the middle of the range a real fly uses when
walking (recorded kinematics). Everyday muscle activation moves it within that range; only
near-full activation goes further, to 1.2× the walking range (1.7× for the front legs, which also
groom and reach), and hard limits stop it there. Skills are held to the same limits.

Neck muscles turn the head; MN9 and the proboscis muscles extend the proboscis; the pharyngeal
muscles pump; wing posture follows the steering muscles. These are posed from their muscles directly
(the physics covers the body and legs). Nothing is animated by hand except the wingbeat waveform
itself (a 200 Hz oscillation of the flight muscles, shown slowed down).

**Body buttons** (under the fly): *Physics* switches between the physical body and a kinematic one
(joint angles straight from the muscles, movement estimated from leg sweeps; also used when MuJoCo
isn't installed, or with `FLYBRAIN_BODY=kinematic`). *Leg feedback* connects or disconnects the
legs' sense organs. *Walking model* and *Flight timer* switch the two hand-built parts (below) on or
off. *Re-centre* puts the fly back in the middle.

What the wiring produces, checked live and with `tools/validate.py`:

| Input | What happens | Pathway (traced live) |
|---|---|---|
| Sugar touched to the proboscis | proboscis extends and stays out | sugar taste neurons → ... → MN9 |
| Sugar + bitter | feeding suppressed | bitter neurons inhibit the feeding pathway |
| Threat from either side | escape jump, a couple of seconds' flight, landing on its feet | LPLC2 → giant fibre → jump motor neuron (TTMn); LPLC2 → DNp06 → wing power motor neurons |
| Activate P9 / MDN / DNa02 / aDN | walks forwards / backs up / turns that way / grooms its antennae | the command neurons you drove |
| Warm air | the brain responds; the walking command neurons don't | |
| Sugar under the feet | the feet taste it, but the proboscis doesn't extend: in this model the tarsal taste signal doesn't reach MN9 (in real flies it does) | |
| Small target darting ahead-left or right | turns towards it | LC10a → AOTU025 → DNa02 on that side, the pathway that steers male flies following a female (Ribeiro et al. 2018; Hindmarsh Sten et al. 2021) |
| Fruit odour + Punish (or Reward) | that odour's Kenyon cell → output neuron synapses weaken; other odours' much less (see Learning) | ORNs → projection neurons → Kenyon cells; PPL1 (or PAM) dopamine at the output synapses |
| Odours, humidity | the brain responds (antennal lobe, lateral horn, mushroom body); the walking command neurons hardly do, so the fly doesn't move towards or away | |
| Antenna dust | leg twitches only; the touch → grooming pathway is too weak in this wiring to reach the aDN command neurons | |

Three things to know, two of them **hand-built parts** of the body. Each can be switched off, and
skills practise with them off (see Skills), so it's always clear what comes from the connectome:

- **Gap junctions.** The giant fibre drives the jump motor neuron mainly through electrical synapses,
  which an electron-microscopy connectome cannot see. Those few known ones (giant fibre → TTMn and
  PSI) are added by hand, and marked as such.
- **Flight follows the tarsal reflex; the flight timer is hand-built.** In real flies, contact under
  the feet inhibits flight and losing it starts and keeps up flight. So the flight motor runs only once
  no foot touches the ground while the flight power motor neurons fire, and touching down on three or
  more feet ends the flight. The **flight timer** (switchable) adds: an escape jump starts flight
  straight away (the giant fibre drives the flight muscles as well as the jump muscle), and once
  airborne the fly keeps flying at a cruising power for 1.5 s, then eases off, sinks and lands. (The
  brain's flight power neurons fire only for a fraction of a second after a looming shadow, so without
  the timer the fly falls back after half a second of flight.) With it off, flight power is only what
  the wing power motor neurons give, so a skill can fly, or jump without flying, on its own.
- **Walking: the walking model is hand-built** (`server/locomotion.py`). The nerve-cord circuits that generate stepping in a fly
  (central pattern generators) aren't mapped yet, and a leaky integrate-and-fire model doesn't make
  rhythms by itself (with the walking model off the command neurons only give twitches). What is known is how
  the brain commands walking, so that part is the simulated brain's, and the stepping itself is
  modelled the way NeuroMechFly does it:
  - six leg oscillators in a tripod (front and hind legs of one side with the middle leg of the
    other), each leg following a real fly's recorded step, all seven joints (NeuroMechFly's kinematic
    recording, 7 steps a second at full speed here up to 6);
  - every frame the brain sets the gait from its walking command neurons: how fast (P9/DNg97, MDN,
    DNa01/DNa02; not other descending neurons: an earlier version added every descending neuron's
    wiring influence on the legs' networks, which counted feeding and grooming neurons too and set the
    fly walking while it tasted sugar), which way (MDN against
    P9/DNg97; backwards is the step cycle run in reverse), and turning (DNa01/DNa02 left against
    right; the legs on the inside of the turn take shorter strides, as Rayshubskiy et al. 2020 and
    Yang et al. 2023 found); the walking fly also holds its course against turns it didn't command;
  - each leg's premotor rhythm neurons, picked from the wiring (`server/cpg.py`: interneurons whose
    synapses go mostly to that leg's swing or stance motor neurons), fire in time with its steps, and
    through the connectome's own synapses so do its motor neurons;
  - the feet's adhesive pads are off while stepping: in this physics, sticky feet make fast stepping
    chaotic (the fly spins or tips over), while friction alone walks straight.

  An earlier version grew the rhythm from those premotor neurons themselves (mutual inhibition and a
  tiring current). They alternated, but the six legs didn't coordinate, the steps were weak and the fly
  could only walk forwards. Another replayed motor-neuron drive learned from recorded steps; it drove
  only 4 of each leg's 7 joints, so the hind legs dragged and turns went the wrong way. Grooming (aDN)
  plays a built-in movement while the aDN neurons fire.

The **Why?** card appears whenever the body starts a behaviour. It starts at the motor neurons (or
command neurons) involved and walks backwards through the wiring: at each step, the cell type sending
the most excitation right now (firing rate × synapse count), until it reaches a sensory neuron. It
also names the strongest inhibitor pushing against it. The same chain lights up cyan.

## Learning in the mushroom body

The mushroom body is where a fly learns what odours mean (`server/plasticity.py`). An odour activates
its own sparse set of Kenyon cells (KCs). They synapse onto mushroom body output neurons (MBONs),
which bias the fly towards or away from things. Dopamine neurons each cover a compartment of that
wiring: PAM neurons mostly signal reward, PPL1 neurons punishment. When a compartment's dopamine
neurons fire while KCs are active, those KCs' synapses onto its MBONs weaken (Hige et al. 2015;
Aso & Rubin 2016), so the next time that odour comes, those MBONs hear less of it. KC activity without
dopamine slowly restores them.

In the model the 15,534 KC → MBON connections of BANC are plastic: their weights change while it
runs. Which dopamine neurons gate which MBON is read from the wiring (their share of the dopamine
synapses onto it). Each KC keeps an eligibility trace of its recent spikes, and each frame a synapse
changes by −η × trace × dopamine × weight, down to a floor of 10% of its connectome strength. The
dopamine neurons' synapses onto KC axons act through slow receptors, so they have no fast effect
here. Taste and heat don't reach the dopamine neurons strongly enough in this model to teach, so
reinforcement is given the way many labs do it: **Reward** and **Punish** (under the fly, "Teach it")
activate the PAM or PPL1 neurons for a second, while whatever odour is in the world is smelled.
The status line under the fly then shows the memory, e.g. "memory: fruit avoid 72%": how much weaker
fruit's KCs now drive the punishment-gated MBONs. **Forget** restores every synapse; learning also
survives a brain reset.

**Odour-specific Kenyon cells needed a calibration** (`server/calibration.py`). As wired, with every
connection a fast chemical synapse, the antennal lobe's cholinergic local neurons spread any odour
to nearly every glomerulus within 20 ms: fruit, CO₂ and geosmin all reached the same ~290 projection
neurons and the same ~90 KCs (85 shared by fruit and geosmin), so pairing one odour with punishment
weakened all of them equally. Three changes, each following real flies, restore odour identity:
excitatory local neurons count 20% of their chemical synapses (they couple glomeruli mostly through
gap junctions, weakly; Yaksi & Wilson 2010, Huang et al. 2010); KC → KC axo-axonal synapses have no
fast effect; and the KCs' sparseness comes from the inhibitory APL neuron (its synapses onto KCs ×3,
as it inhibits by graded release, not spikes; Lin et al. 2014) instead of the stabilised mode's
generic adaptation, which projection neurons, KCs and APL are exempt from. Now fruit drives ~550 KCs
(12%), CO₂ ~45 (8 shared with fruit), and 3 fruit + punishment pairings weaken fruit's input to the
punishment-gated MBONs by ~80% and CO₂'s by ~20%.

What doesn't follow yet: the fly's behaviour. After training, it approaches fruit about as much as
before (the learned_avoidance experiment below). Measured on the model, three things stop it:

1. **The output neurons barely fire.** Fruit at the antennae makes the 110 MBONs fire 9 spikes in a
   second in total (real ones fire spontaneously, several to tens of spikes a second each); the learned weakening takes that to
   4. With a background input that makes them fire spontaneously, as real MBONs do, the odour's
   effect on them drowns in it (593 spikes before training, 590 after).
2. **Their output hardly reaches the command neurons.** Summed through up to three synapses, the
   punishment-gated MBONs' influence on the walking and steering command neurons is under 1% of
   those neurons' input, and in the spiking model the signal fades at each layer. Training changed
   the command neurons' firing by nothing measurable.
3. **The fly doesn't steer by smell.** Approach and avoidance are turns towards or away from an
   odour; in this wiring and model, fruit and CO₂ set the fly walking but not turning (Scoring, below),
   so there's no odour steering for a memory to change.

Real flies close this loop through circuits downstream of the MBONs (the fan-shaped body, the
lateral horn, and descending neurons for upwind steering) that the connectome contains but the
spiking model doesn't carry at these strengths.

## Scoring against real experiments

`tools/benchmark.py` does to the model what was done to real flies and scores what the body does,
0 to 1, from what a tracker and a camera on a real rig would measure (how far and which way the fly
moved and turned, which muscles fired and when, how far the proboscis extended), or, for learning,
how the synapses changed. Each experiment runs three times and the scores are averaged (the model is
stochastic). Scores are reported separately for **behaviours** the fly should produce and **controls**
where it should do nothing (stand still, not take off at a smell, not drink bitter), because a dead
network passes every control.

```bash
uv run tools/benchmark.py                          # physical body, walking on (~15 min)
uv run tools/benchmark.py --rhythm off             # no walking model: the connectome alone moves the legs
uv run tools/benchmark.py --flight-timer off       # no flight timer: flight power only from the motor neurons
uv run tools/benchmark.py --variant shuffled       # control wiring (below)
uv run tools/benchmark.py --ablate giant_fibre     # silence neurons: vision, mushroom_body, descending,
                                                   # giant_fibre, antennal_lobe, lateral_horn, or a cell type
uv run tools/benchmark.py --only loom_left,learn_punish --repeats 1
```

Results at the time of writing (physical body, leg feedback on). *Walking off* is the connectome
without the stepping model; *shuffled* is the connectome with every synapse's target reassigned at
random, keeping each neuron's number of inputs and outputs, its transmitter and each synapse's
strength (and the leg networks and mushroom body re-read from that wiring):

| Experiment | Real flies | Walking on | Walking off | Shuffled |
|---|---|---|---|---|
| **Activating command neurons** | | | | |
| P9 (DNp09) | walk forwards (Bidaye et al. 2020) | 0.80 (+3.9 mm/s) | 0.00 | 1.00 |
| MDN | walk backwards (Bidaye et al. 2014) | 1.00 (−6.5 mm/s) | 0.00 | 1.00 |
| DNa02, left / right | turn that way (Rayshubskiy et al. 2020) | 1.00 / 0.96 (0.6-0.8 rad/s) | 0.00 / 0.00 | 1.00 / 1.00 |
| The giant fibre | escape jump within ms (von Reyn et al. 2014) | 1.00 | 1.00 | 0.00 |
| aDN (DNg62) | antennal grooming (Hampel et al. 2015) | 1.00 | 1.00 | 1.00 |
| **Senses → behaviour** | | | | |
| Looming shadow, left / right | escape takeoff before contact | 1.00 / 1.00 | 0.67 / 1.00 | 0.00 / 0.00 |
| Looming shadow | takes off, flies away, lands on its feet (Card & Dickinson 2008) | 1.00 (2.5 s airborne) | 1.00 | 0.00 |
| Small target darting ahead-left / right | turns towards it (Ribeiro et al. 2018) | 1.00 / 1.00 (+91° / +59°) | 0.13 / 0.00 | 0.07 / 0.01 |
| Sugar on the proboscis | extension (Shiu et al. 2024) | 1.00 | 1.00 | 0.00 |
| Sugar under the front feet | extension (Dethier 1976) | 0.00 | 0.00 | 0.00 |
| Dust on the antennae | antennal grooming (Seeds et al. 2014) | 0.00 | 0.00 | 0.00 |
| Warm zone around it | walks out of it (Simões et al. 2021) | 0.00 | 0.00 | 0.00 |
| CO₂ source ahead | moves away (Suh et al. 2004) | 0.00 | 0.00 | 0.00 |
| Fruit odour 6 mm to the left / right | turns and walks towards it | 0.02 / 0.01 | 0.03 / 0.00 | 0.00 / 0.02 |
| Stripes turning left | turns with them (optomotor) | 0.00 | 0.11 | 0.01 |
| Fruit paired 3× with PPL1, CO₂ unpaired | paired odour's KC → MBON synapses weaken (Hige et al. 2015) | 1.00 (−82% vs −17%) | 1.00 | 0.00 |
| The same with PAM (reward) | the same in reward compartments (Cohn et al. 2015) | 1.00 (−80% vs −11%) | 1.00 | 0.00 |
| Fruit approach before / after punishment pairings | approaches less (Tully & Quinn 1985) | 0.00 | 0.00 | 0.01 |
| **Controls** (8): no input; bitter on the proboscis or under the feet; fruit, geosmin, CO₂, heat, cold | stand still, don't feed, don't take off | 0.99 | 1.00 | 1.00 |
| **Behaviours, activating command neurons** (6) | | **0.96** | 0.33 | 0.83 |
| **Behaviours, from the senses** (16) | | **0.50** | 0.37 | **0.01** |

What this says:

- **Walking and escape work.** The command neurons make the fly walk forwards, back up and turn
  either way, at 4-7 mm/s and 0.6-0.8 rad/s; it turns towards a moving target through LC10a →
  AOTU025 → DNa02, the pathway that steers male flies after a female; and a looming shadow makes it
  jump, fly for a couple of seconds and land on its feet.
- **The real wiring carries everything the fly senses.** With the same neurons, sensors, body and
  model machinery, the shuffled wiring scores 0.01 on the 15 behaviours that start at a sense organ,
  against 0.51 for the connectome. Activating a command neuron directly still makes the shuffled fly
  walk, turn and groom, because the stepping and grooming models read those neurons themselves; the
  giant fibre's jump, which goes through the nerve cord's wiring, fails. (Tasting sugar or bitter, or
  dust on the antennae, no longer sets the fly walking: see Walking.)
- **Ablations find the circuits where biology puts them**: silencing the 2 giant fibres abolishes the
  jump to a looming shadow (real flies still escape without them, with slower "long-mode" takeoffs,
  and here the fly runs instead); silencing the visual projection neurons abolishes escape and
  target following but not the giant fibre's own jump or feeding; silencing the Kenyon cells
  abolishes learning and nothing else tested (escape 5 of 5, feeding). Removing all inhibition makes
  the network seize (sugar on the proboscis sets 28,000 neurons firing within 0.1 s instead of 185)
  and the fly thrashes across the arena with no input at all.
- **What the wiring doesn't do in this model**, and so the fly doesn't either: it doesn't steer by
  odours, towards fruit or away from CO₂, and warmth doesn't reach its walking command neurons (an
  earlier version walked out of warm air and through CO₂ only because any descending activity counted
  as a walking command); turning stripes
  make it turn the wrong way, because in the simulation the right-side motion-sensing neurons (HS
  cells) respond to both directions and the right DNa02 fires either way; sugar under the feet
  reaches the proboscis motor neuron only through links carrying 0.5-1.4% of each next neuron's input;
  antennal dust reaches the grooming command neurons only through three weak steps. Learning changes
  the synapses but not yet the behaviour (see Learning in the mushroom body).

## Skills: teaching the fly new movements

Ask Fly in chat to do something its body has never done ("tap your front-left leg", "jump as high as
you can", "walk forwards without the walking model"). The idea: the LLM plans and judges, the
connectome and the body do the moving, and practice finds how far they can go. It works like this
(`server/skills.py`, `server/practice.py`):

1. **Plan and goal.** The LLM describes the movement as keyframes of motor controls: per leg `swing`,
   `lift`, `reach`, `grip`, `spread`, `twist`, and the middle legs' `jump` muscle; wings `power`,
   `extend`, `stroke`; head; proboscis; antennae; halteres; abdomen. Values from -1 to 1 over time
   (`motor_reference` has the joint ranges, timing facts and physics). For walking it can instead ask
   for a plan made from a real fly's recorded steps (`gait`). And it says what the movement should
   achieve: a **goal**, like `forward`, `turn_left`, `jump_high`, `still`, or its own weighting of the
   measured metrics (how far forward, sideways, how much it turned, how high, time in the air, whether
   it fell, flight power, the jump muscle, muscles the plan doesn't use).
2. **Search the wiring (done once).** Each control direction maps to muscle pools (e.g. "front-left
   lift up" = the front-left trochanter flexor motor neurons). For every neuron, the engine computes its
   influence on every pool through up to three synapses (signed, normalised by each target's total
   input). Good drivers push their target hard and its antagonist and everything else weakly. It tries
   **descending neurons** first (brain → nerve cord), then **premotor nerve-cord interneurons**, and
   only as a last resort the motor neurons themselves (`drive_at` can restrict this to the nerve cord
   or the motor neurons).
3. **Shape.** The drivers are driven and the engine measures what the muscles did, correcting the
   drive from the error (iterative learning control) until the muscles follow the plan; a driver that
   can't move its muscle is replaced one level down, one that moves other body parts is swapped.
4. **Practise for the goal.** Many tries (40 by default) in **background copies of the whole fly**
   (brain, muscles, body, physics; `server/practice_worker.py`, a few at once on a computer with the
   cores for it), while the live fly keeps running. Every try starts from the same standing fly; a
   numerical search (CMA-ES) varies how hard and when each kind of movement's neurons are driven, and
   the pace, and scores each try on the goal. The best few are tried again on fresh noise and the best
   on average wins. Falling over, muscles the plan doesn't use, the wings (if the plan doesn't use
   them) and the jump (if it doesn't use it) count against a try unless the goal says otherwise.
5. **Compare and save.** A built-in skill can be measured the same way as a **baseline** (e.g.
   `walk_forward`, with the walking model it relies on). The skill is stored in
   `data/skills/skills.json` (this computer only, not in git): the driver neurons (with their BANC ids),
   their firing schedule, the plan, the goal, what it achieved and the switches it was learned with; then
   it's shown once on the live fly.

**Hand-built parts are off in practice.** The walking model and the flight timer (see The fly's body)
are switched off while a skill practises, unless the LLM asks for them, and a skill replays with the
switches it learned with. So a learned skill is the neurons' and muscles' own work, and the built-in
versions become the baselines to beat. `test_skill` measures any skill with or without them, without
touching the live fly.

After that, `do_skills` just replays the stored drive: no search, no practice. To type a word, Fly
learns one small tap skill per leg once, then chains them; `combine_skills` saves a chain under one
name (e.g. `walk_forward` ×4, then a jump: the walking runs as one continuous 4 s walk). To improve a
skill Fly refines it rather than starting a new version from scratch, so it keeps what it learned
and never ends up worse. The built-in skills are listed too:
walking forwards and backwards and turning left and right activate those command neurons (P9, MDN,
DNa02 left or right) and the walking model steps the legs; grooming its antennae is a recorded
movement.

What it gets (about a second of movement, 40 tries, measured in the practice copies; forward
distance, mean of 3 runs):

| Movement | Before practice | After practice | Built-in `walk_forward` (walking model), same time |
|---|---|---|---|
| a real fly's recorded steps as a plan, driven at the motor neurons | 4.4 mm | 5.5 mm | 4.9 mm |
| a simple tripod plan (lift and swing only), brain's own routes | 9.8 mm, but by hopping (it set off the jump muscle) | 4.2 mm, walking | 4.6 mm |
| the recorded steps, brain's own routes (descending, premotor) | 0.3 mm, the wings working too | 2.2 mm | 6.7 mm |
| built-in `walk_forward` with the walking model off | 0 mm: the command neurons alone don't step the legs | | |
| a jump (the middle legs' jump muscle, through the giant fibre) | 13.5 mm high | 16.4 mm high, lands on its feet, doesn't fly off | |

So the body can walk from its muscles as well as the walking model does, and practice improves on
the plan it's given; what limits walking through the brain's own routes is how specific the upstream
neurons are (a premotor neuron that lifts a leg may also work the wings). Practice also catches
cheats: the simple plan first "walked" by hopping, and the goal's penalty on the jump muscle pushed it
back to walking.

This is the same idea as the KV cache, one level down: work that is expensive the first time is
kept and reused. And the skill list is deliberately not in the LLM's system prompt. Every new skill
would change the prompt's start and throw away the cached tokens, so Fly looks skills up with a tool
(`list_skills`) and the answer lands at the end of the conversation, where it costs only its own
tokens.

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

### Any model, offline or online

The built-in list is just a starting point. Fly works with any chat model, running on your machine
(fully offline once downloaded) or on a server or hosted API. In the Chat tab, **Use another model**
takes one of:

| Option | Example | Runs | Offline? |
|---|---|---|---|
| Any GGUF model on Hugging Face | `unsloth/gemma-3-12b-it-GGUF:Q4_K_M`, `Qwen/Qwen2.5-7B-Instruct-GGUF` | here, with llama.cpp; downloaded once into `models/hf/` | yes, after the download |
| A GGUF file you already have | `/Users/you/models/my-model.gguf`, `D:\models\my-model.gguf` | here, with llama.cpp | yes |
| A local server | Ollama `http://localhost:11434/v1`, LM Studio `http://localhost:1234/v1`, vLLM, llama.cpp | on your machine or network | yes |
| A hosted API that speaks the OpenAI chat format | the provider's base URL (usually ending in `/v1`), the model name, and your API key | the provider's servers | no |

The choice is remembered for that machine only, in `data/settings/` (which never goes to git). An API
key typed there is stored there too; to keep it out of files altogether, leave the key field empty
and set the environment variable `FLYBRAIN_API_KEY` before starting instead. Pick a built-in model
again to switch back.

**Models without tool calling** work too. If a server rejects tools, Fly switches that model to
describing its tools in the prompt and reading its calls from fenced blocks, and says so in the chat.
Native tool calling is more reliable, so prefer models that have it. Set `"tool_mode": "prompt"` to
force the prompt style, `"native"` to never fall back.

**Better models plan better movements.** When asked for a new movement, Fly reads `motor_reference`
(every motor control, each leg joint's range in degrees, real fly timing, the physics, the hand-built
parts, and the goal metrics), writes a keyframe timeline and a goal, and learns it. The result comes
back as what the movement achieved (before and after practice, and against a baseline), and Fly can
`refine_skill` with a new goal, more tries or a changed plan. A stronger model writes better plans and
goals and reads the results better; the simulated nervous system and the practice are the same for
every model.

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
| `external_url`, `external_model`, `external_api_key` | an OpenAI-compatible server instead of the built-in engine (or the key in `FLYBRAIN_API_KEY`) |
| `external_strict` | `true` for hosted APIs that reject llama.cpp's extra sampling settings (automatic when an API key is set) |
| `tool_mode` | `auto` (default), `native` or `prompt` |
| `context` | context window in tokens |
| `extra_args` | passed straight to llama-server |

Notes: KV-cache saving and restoring (below) needs llama.cpp's own server, so with an external
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
unless you tick Auto-approve), long-term notes (remember/recall), its body
(body_experiment, stimulate_senses, brain_status, list_skills, motor_reference, learn_skill,
refine_skill, test_skill, combine_skills, do_skills, body_parts), and neuroscience tools (below).

It also has a way to learn the system itself, so it doesn't guess: `guide` (a guide to what's real
connectome and what's hand-built, how the body and practice work, and what each tool is for;
`server/agent/guide.md`), `api_get` (reads the app's local HTTP API, no approval needed), and
`flylab.py`, a small Python helper the app puts in the workspace for scripts (`import flylab as fl;
fl.info("DNp01")`, `fl.rates([...])`, `fl.test("walk_forward", assists={"walking": False})`, ...;
plain Python over the local API).

### Fly as a neuroscientist

Fly can study its own nervous system the way a lab would, and is asked to report negative results
and keep claims to what it measured (`server/neuroscience.py`, `server/agent/tools.py`):

| Tool | What it does |
|---|---|
| `find_neurons` | search cell types by name, class or super class, with counts, transmitter, regions and current firing |
| `neuron_info` | one cell type: what it is, where, how active now, its strongest input and output types (synapses, sign, share), what the app uses it for |
| `trace_path` | the strongest chain of cell types from one to another, each step weighted by its share of the next type's input |
| `activate_neurons` | drive a cell type (optogenetics-style) and see what the body does, recording other types meanwhile |
| `body_experiment` (with `record`) | any world or optogenetic experiment, recording chosen cell types' firing |
| `silence_neurons` | ablate a named group (vision, mushroom_body, descending, giant_fibre, antennal_lobe, lateral_horn) or any cell type, until restored |
| `run_benchmark` | score experiments in a separate copy of the model, optionally with neurons silenced or on a control wiring; saved under a label |
| `compare_runs` | compare saved runs experiment by experiment |

For example, "which neurons make the fly turn towards a moving target?" can go: trace_path LC10a →
DNa02 (LC10a → AOTU025 → DNa02), body_experiment target_left recording AOTU025 and DNa02, then
run_benchmark follow_left with AOTU025 silenced and compare.

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
- **Olfactory calibration** (stabilised mode only): so that odours stay distinct in the mushroom body,
  excitatory antennal lobe local neurons count 20% of their chemical synapses, Kenyon cells don't
  excite each other, APL's inhibition of them is ×3, and projection neurons, Kenyon cells and APL
  have no adaptation or depression (see Learning in the mushroom body).

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
{"cmd": "body",   "action": "loom", "side": "left"}   # also dust, p9, mdn, dna02_l, dna02_r, gf, adn,
                                                     # stop, reset ("home": true re-centres the fly)
{"cmd": "body",   "action": "physics", "on": true}     # physical or kinematic body
{"cmd": "body",   "action": "feedback", "gain": 1}     # legs' sense organs -> nerve cord (0 = off)
{"cmd": "body",   "action": "rhythm", "mode": "neurons"} # the walking model: neurons (on) | off
{"cmd": "body",   "action": "assist", "name": "flight_timer", "on": false}  # a hand-built part on/off
{"cmd": "ablate", "group": "vision", "on": true}       # silence a neuron group or a cell type (false: restore)
{"cmd": "body",   "action": "fruit"}                   # put something in the world near the fly: sugar,
                                                     # bitter, fruit, co2, geosmin, cva, heat, cold,
                                                     # humid, dry, loom (side), stripes (side), clear;
                                                     # optional "x", "y" in mm
{"cmd": "body",   "action": "probe_sugar"}             # touch a drop to the proboscis (also probe_bitter)
{"cmd": "body",   "action": "target", "side": "left"}  # a small target darting ahead-left (or right)
{"cmd": "body",   "action": "punish"}                  # activate PPL1 dopamine neurons for 1 s (reward: PAM)
{"cmd": "body",   "action": "forget"}                  # restore every learned synapse
{"cmd": "body",   "action": "learning", "on": false}   # freeze the mushroom body's synapses
{"cmd": "skill",  "action": "play", "name": "tap_front_left"}   # also stop, forget

GET /api/status         behaviour levels, region activity, top firing neurons
GET /api/neuron/{idx}   identity, transmitter, strongest inputs and outputs
GET /api/search?q=...   cell types by name
GET /api/typeinfo/{type} a cell type: counts, transmitter, regions, firing, main partner types, roles
GET /api/path?src=&dst=  strongest path between two cell types
GET /api/activity_groups the neuron groups of the activity view
GET /api/presets        all senses and behaviour readouts with their neuron ids
GET /api/body/pathways  motor neurons per body part, descending command neurons
GET /api/skills         learned skills, and the motor controls a plan can use
GET /api/skill/{name}   one skill: plan, goal, what it achieved, drivers and their firing schedule
GET /api/controls       every motor control and its muscles, plus the planning reference
GET /api/body           what the body is doing, where it is, the hand-built parts
GET /api/rates?types=DNp01,MDN&neurons=12,34   firing rates now
GET /api/find?q=...     cell types by name, class or super class
GET /api/practice       the practice copies, goal metrics and presets
POST /api/skills/learn  {"name", "description", "keyframes" or "gait", "goal", "trials", "baseline",
                         "assists", "drive_at", "refine": true to improve a learned skill}
POST /api/skills/test   {"name", "assists", "goal", "runs"}: measured in a practice copy
POST /api/skills/combine {"name", "description", "steps": [{"skill": ..., "repeat": n}]}: a sequence
POST /api/skills/do     {"names": [...], "repeat": 1}: on the live fly
POST /api/restart       start the app again (run.sh / run.bat), e.g. after an update

{"cmd": "chat", "text": "..."} | {"cmd": "stop"} | {"cmd": "new_chat"} | {"cmd": "model_setup", "model": "qwen3.5-9b"}
{"cmd": "model_custom", "kind": "hf" | "path" | "server", "value": "...", "model": "...", "api_key": "..."}
GET /api/agent/status   model, hardware, Laya, conversation
```

The browser gets a WebSocket stream (`/ws`) of every spike (binary), status JSON 5× a second, the
body's state (joint angles, muscle activation per body part, where the fly is and how it moves,
which feet touch the ground, what it senses and what the legs' sense organs report, and what's in the
world) every frame, a `decision` message
with the traced pathway whenever the body starts a behaviour, and `skill` messages while a skill is
learned or performed.

## Files

```
server/app.py          web server, simulation loop, API
server/sim.py          the network simulator (numba)
server/connectome.py   downloads + packs the BANC connections (first run)
server/motor.py        motor neurons -> muscles -> joint targets and stiffness
server/physics.py      the physical body in its arena (MuJoCo, NeuroMechFly)
server/arena.py        the world: food, odours, air, threats, targets, stripes -> sensory neurons
server/locomotion.py   walking: leg oscillators and recorded steps, commanded by descending neurons
server/cpg.py          the legs' premotor rhythm neurons from the wiring, and descending drive to them
server/feedback.py     the legs' sense organs -> sensory neurons
server/plasticity.py   learning: dopamine-gated plasticity of the mushroom body's synapses
server/calibration.py  keeps odours distinct in the antennal lobe and mushroom body
server/neuroscience.py wiring queries: cell types, partners, strongest paths, activity-view groups
server/body.py         stimuli -> sensory neurons; runs the body; what the body is doing
server/skills.py       learning new movements: wiring search, shaping, practice for a goal, skill library
server/practice.py     practice: goal metrics, the search (CMA-ES), background copies of the fly
server/practice_worker.py  one background copy of the whole fly, run as fast as it goes
server/agent/          the AI: runtime.py (llama.cpp setup), models.py, llm.py, system1.py (Laya),
                       agent.py (loop, conversations, KV cache), tools.py, bridge.py (body link),
                       guide.md (the agent's guide to the system), flylab.py (Python helper for its scripts)
server/data/           neuron labels and signs, senses, readouts, motor pools, leg sense organs,
                       loop-neuron mask, built-in skills (skills_builtin.json), and the
                       NeuroMechFly body model (neuromechfly.xml, from flygym, Apache 2.0)
web/                   the 3D viewer (three.js, bundled); web/fly.js is the 3D fly, web/chat.js the chat,
                       web/activity.js the spike raster and rate plots
web/data/fly_body.*    the fly's meshes, skeleton and resting pose (from NeuroMechFly)
tools/fetch_banc.sh    downloads the BANC annotation files build_banc.py needs
tools/build_banc.py    rebuilds web/data and server/data from the BANC files
tools/find_loops.py    finds the neurons that get adaptation in stabilised mode
tools/benchmark.py     scores the model against published experiments
tools/validate.py      checks known behaviours and which body parts each input moves
tools/build_proprio.py finds each leg's sense organs in BANC (needs the BANC annotation file)
tools/build_senses.py  finds the sensory neurons the world reaches, by side and by leg (same)
tools/learn_builtin.py relearns the built-in grooming skill on the connectome
tools/build_body.py    rebuilds web/data/fly_body.* from the flygym package
tools/mock_llm.py      a scripted fake model server for testing the agent without a download

Created on your machine and kept out of git (see .gitignore): .tools/, .cache/ (uv, Python, the BANC
download), .venv-<profile>/, runtime/ (llama.cpp), models/ (GGUF files), data/ (your conversations,
KV caches, memory notes, learned skills, scoring results, logs, per-machine settings), workspace/ (the agent's
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
  model (MJCF), meshes and skeleton (simplified here), and recorded walking steps.
  Physics: [MuJoCo](https://mujoco.org) (Apache 2.0).
- Leg sense organs: Tuthill and Wilson 2016; Mamiya et al. 2018 and 2023 (femoral chordotonal
  organ); Dallmann et al. 2023 (presynaptic inhibition of leg proprioceptors during movement); BANC's
  sensory annotations.
- Muscles and motor neurons: Azevedo et al. 2024 (leg), Lindsay et al. 2017 and Muijres et al. 2014
  (wing steering), Schwarz et al. 2017 and McKellar et al. 2020 (proboscis). Giant fibre gap
  junctions: King and Wyman 1980, Allen et al. 2006.
- Behaviour neurons: LPLC2 and the giant fibre (von Reyn et al. 2014, Ache et al. 2019), P9 (Bidaye
  et al. 2020), MDN (Bidaye et al. 2014), DNa02 (Rayshubskiy et al. 2020), aDN (Hampel et al. 2015),
  LC10a (Ribeiro et al. 2018; Hindmarsh Sten et al. 2021).
- Learning: Hige et al. 2015, Aso and Rubin 2016, Cohn et al. 2015, Owald et al. 2015 (dopamine-gated
  plasticity of Kenyon cell → MBON synapses); Lin et al. 2014 (APL and Kenyon cell sparseness);
  Yaksi and Wilson 2010, Huang et al. 2010 (excitatory local neurons in the antennal lobe).
- LLM runtime: llama.cpp (MIT). Models: Qwen3.5 / Qwen3.6 (Apache 2.0), GGUF builds by Unsloth.
- System 1: Laya by Convai Innovations (Apache 2.0).

Check the BANC and FlyWire data terms before any commercial use.
