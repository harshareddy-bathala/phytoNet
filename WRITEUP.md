# PhytoNet: a brain-free plant that learns its own electrical signalling

*A draft results write-up. Figures referenced live in `figures/`; the benchmark
data is in `results/results.csv`. All numbers are reproducible from the seed
lists via the scripts named inline.*

## Abstract

Plants coordinate growth without a brain, using slow chemical and electrical
signals between tissues. PhytoNet asks whether a simulated plant can *discover*
such signalling rather than have it hand-designed. The plant's body is a graph of
tissues; every tissue runs the **same** interpretable rule, senses its local
environment, exchanges **electrical signals** with neighbours, and acts (grow,
thicken, steer, defend). The rule's parameters are **evolved** (Cross-Entropy
Method) to maximise lifetime fitness (biomass + reproduction, under survival
pressure) — no backprop, no central controller, no neural network. We compare
this brain-free controller against neural baselines (a PPO MLP and a per-tissue
GNN with learnable messages) and against fixed policies across a battery of
environments. **Main results:** (1) the brain-free controller is the best
performer overall and the only one to beat trivial baselines; (2) its advantage
is *statistically significant specifically under herbivory* (multi-seed,
p = 0.008), where an electrical wound signal coordinates a systemic defence;
(3) the learned signalling reproduces three signatures of real plant electrical
signalling — systemic spread, a slow finite-speed wave, and shoot-directional
wound response; and (4) across pure resource variability, non-stationary seasons,
and plant–plant competition, adaptive signalling *ties* a fixed strategy. We
conclude that **distributed signalling earns its keep for coordination problems
(systemic defence), not for mere environmental variability.**

## 1. Introduction

This is the inverse of connectome-driven work that simulates a fixed nervous
system to read out behaviour: we fix behaviour (survive, grow, reproduce) and let
a plant *write in* the communication. The research question: do the learned
signals resemble real plant signalling, or invent something different?

## 2. Methods

**Body.** A plant is a graph of tissues (`seed`, `root`, `stem`, `leaf`). It
germinates from a single seed; tissues are added by growth. Positions are
N-dimensional (2-D and 3-D supported).

**Physiology (per day).** Leaves convert local light to sugar (with **feedback
inhibition** and **self-/canopy-shading**); roots take up water from a dynamic
soil reservoir (rain refills, evaporation/uptake drain it); sugar and water
**diffuse** along edges (the hand-coded stand-in for transport/signalling);
maintenance burns sugar. Growth is an **action** with **asymmetric build costs** —
shoot/leaf expansion costs mostly water (turgor), roots cost mostly sugar
(carbon) — and a **resource-coupled budget**, so the optimal allocation can shift
with conditions. Lifecycle: mature leaves set seed (fitness); starvation kills.

**Environments (`phytonet.scenarios`).** `baseline`, `drought`, `shade`,
`monsoon`, `bright`, `patchy` (spatial light/water gradients), `seasonal`
(light fades over the episode), `herbivory` (a leaf is bitten on a schedule),
`lifecycle` (reproduction + death).

**Controllers.** (a) fixed global allocation (`balanced`, `random`); (b)
**MLP-PPO** — a CleanRL-style continuous PPO on a global summary observation;
(c) **per-tissue GNN** — hand-written message passing (no PyG); each tissue acts
from its message-informed embedding; trained with PPO; (d) **bioelectric (hand)**
— each cell has a membrane potential depolarised by local stimuli, electrotonic
coupling, action potentials, and grows where excited; (e) **bioelectric
(learned)** — the cell's signal is a multi-channel vector with a shared linear
production/coupling/readout rule (~30 weights) **evolved by CEM**. Wound signals
travel on a separate variation-potential-like channel and arm systemic defence.

**Evaluation.** Common outcome metric = final biomass + reproduction. The
paper-grade runner (`phytonet.experiments`, Hydra-configured) trains each method
**per seed**, evaluates on every scenario, and reports **bootstrap 95% CIs** and
**permutation-test** p-values (numpy-only). See `scripts/run_experiment.py`.

## 3. Results

### 3.1 Cross-paradigm benchmark (`figures/benchmark_all.png`)
The brain-free bioelectric controller is the best overall (16.9 vs 16.3 for all
others) and the **only** method to beat the trivial baselines, winning decisively
on the spatially heterogeneous `patchy` scenario (20.5 vs 17.4). The neural
controllers (MLP, GNN) tie the fixed baselines exactly — with a *global* action,
better sensing/messages cannot help.

### 3.2 Multi-seed significance (`results/experiment.png`, `results/results.csv`)
Across 5 methods × 5 scenarios × 5 seeds with 95% CIs and a permutation test, the
evolved bioelectric controller significantly beats `balanced` **only under
herbivory**: **+2.83, p = 0.008**. Everywhere else the methods lie within each
other's CIs. Evolution matters: hand 9.3 → evolved 12.8 on herbivory.

### 3.3 Learned signal content (`figures/learned_signal.png`)
Evolving the *content* of the signal (Phase 2) beats the hand-designed rule
overall (16.3 vs 16.0) and on herbivory (12.4 vs 10.0). The result is
*readable*: the plant discovered a channel encoding shoot-identity (+0.74), light
(+0.64), and **wound** (+0.49) — an emergent, inspectable signalling code.

### 3.4 Signalling vs real plants (`figures/phase3_signal.png`)
A single controlled leaf wound produces: **systemic** defence out to ~6 hops; a
**slow, finite-speed travelling wave** (~0.26 hops/step); and a **shoot-directional**
response (0.53 vs 0.00 in roots). These match three signatures of real plant
electrical signalling (systemic wound response; mm–cm/s action/variation
potentials; shoot-borne systemic wounding).

### 3.5 What does *not* help (the boundary)
- **Seasons** (`figures/seasonal.png`): a real non-stationary environment, but
  adaptive signalling ties the baseline (p = 0.73).
- **Competition** (`phytonet.competition`): genuinely lowers fitness and makes
  *strategy* matter (a tall/shoot-heavy racer **loses** to balanced — it starves
  its roots), yet adaptive signalling ties the fixed controller (19 = 19); both
  reach the same competitive equilibrium.

## 4. Discussion

The central, robust finding is a **boundary condition for the value of
signalling**: across resource variability (drought, shade, patchy), non-stationarity
(seasons), and competition, a fixed "balanced" policy is already near-optimal,
because the right response — "grow when you can, don't under-invest in either
organ" — is policy-independent. Distributed signalling provides a *statistically
significant* advantage in exactly one tested setting: **herbivory**, a genuine
**coordination problem** where a local event must trigger a systemic, anticipatory
response. There, a brain-free electrical controller not only wins but reproduces
the qualitative phenomenology of real plant wound signalling.

This reframes the headline contribution. It is not "learned signalling beats
neural control everywhere"; it is "a brain-free, evolved, interpretable electrical
controller **matches or beats** neural and fixed controllers, and **uniquely
excels at the coordination task that real plant signalling evolved to solve**."

## 5. Limitations & follow-ups (tested)

- **Bidirectional signalling (added).** Beyond the shoot→root wound signal, a
  root-origin water-stress signal now propagates **root→shoot** (verified: root
  2.1 → shoot 0.9 under drought) and biases growth root-ward — an ABA-like
  drought signal. The mechanism works, but drought fitness is flat (the resource
  plateau): signalling moves information, not the ceiling.
- **Competitive co-evolution (added & tested).** Evolving a rule *against* a
  fixed competitor in the arena yields no edge — co-evolved ties the opponent
  (relative biomass +0.00); the competitive equilibrium is inescapable.
- **The boundary is therefore robust.** Across resource variability, seasons,
  competition, bidirectional resource signalling, *and* co-evolution, distributed
  signalling gives a statistically significant advantage in exactly one tested
  setting — **herbivory** (the coordination task). This is the paper's claim.
- Remaining true open items: evolved (not hand-set) wound-propagation constants;
  richer multi-channel learned signals; self-play co-evolution of *both*
  competitors; and biological calibration of signal speeds against measured
  action/variation-potential velocities.

## 6. Reproducibility

`pip install -e ".[rl,repro]"` (CPU torch + Hydra; W&B optional). Every figure has
a generating script (`scripts/`), the benchmark is `scripts/run_experiment.py`
(seeds/days overridable on the CLI), and 110 tests cover the mechanisms and
statistics. Results are deterministic given the seed lists.
