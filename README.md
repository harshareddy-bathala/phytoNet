# PhytoNet

A simulated plant agent that **learns its own internal signaling**. The plant's
body is a graph of tissues; each tissue reads its local environment, exchanges
signals with its neighbours, and acts (grow, thicken, steer, defend). Growth is
shaped by a reward for survival, biomass, and reproduction. The research question:
do the learned signals resemble real plant signaling, or invent something new?

The headline controller is **brain-free and decentralized**: every tissue runs
the *same* interpretable rule, communicating via **electrical signals** (membrane
potentials, action potentials, systemic wound signals) like a real plant, with no
central unit and no neural network. Its rule is **evolved** (CEM, no backprop).
Neural controllers (PPO MLP, a per-tissue GNN with learnable messages) are kept as
*baselines* to compare against.

**Key results.** The brain-free bioelectric controller beats the neural baselines
overall, and — in a multi-seed, statistically-tested benchmark — significantly
under **herbivory** (+2.83, p=0.008), where its electrical wound→defense response
matters. The learned signaling reproduces three signatures of real plant
signaling: **systemic spread**, a **slow finite-speed wave**, and **shoot-directional**
wound response. See `figures/` and `results/`.

**Thesis-ready write-up:** [WRITEUP.md](WRITEUP.md) packages the methods, results,
and the central finding (signaling helps for *coordination* problems like defense,
not mere environmental variability) into a paper-style draft.

## Status / roadmap

- [x] **Step 1 — substrate on screen.** World (light + soil water) and a plant
  graph; render a static seedling. *No growth, no learning.*
- [x] **Step 2 — the watchable win.** Physiology (light→sugar, water uptake,
  transport/diffusion, maintenance) + a hand-coded grow-toward-light policy +
  a season-long animation and resource plot. Run with `scripts/run_season.py`.
- [x] **Step 3 — make it honest.** Resources no longer accumulate unbounded:
  added a **thickening sink** (nodes invest surplus into girth) and **sugar/water
  feedback inhibition** (production down-regulates as a tissue fills, like real
  sink-limited photosynthesis). All knobs live in `phytonet.config.SimConfig`.
  20 tests cover conservation, boundedness, and growth.
- [x] **3D + physics.** N-dimensional core now runs in 3D (`--dim 3`) with a
  matplotlib 3D renderer, a sun position, and tropism-based growth (gravitropism
  + phototropism). Realistic look: tapered woody branches, leaf discs, smooth
  interpolated growth with a day/biomass HUD.
- [x] **Root system + dynamic water.** Seedlings grow several roots that branch
  into a root system. Soil water is now a **reservoir** that roots drain, rain
  refills (on a schedule), and evaporation dries — visualised as the ground
  darkening, a rain effect, and a soil-% HUD.
- [x] **Scenarios.** `phytonet.scenarios` defines worlds to test the plant in —
  `baseline`, `drought`, `monsoon`, `shade`, `bright` — via `--scenario`.
- [x] **Growth as a trade-off.** Each step the plant funds a limited number of
  growth events (the budget is coupled to the *scarcer* pooled resource) and
  **allocates** them between shoots / roots / thickening — they compete. Build
  costs are **asymmetric**: shoot/leaf expansion costs mostly **water** (turgor),
  roots cost mostly **sugar** (carbon). So under drought shoots get expensive
  (favour roots) and under shade roots get expensive (favour shoots). The
  optimal *extreme* now flips by condition (see `figures/allocation.png`); the
  interior stays a broad plateau, so a single **global** allocation knob can't
  express a sharp per-condition optimum — motivating per-tissue control (GNN).
- [x] **Gymnasium wrapper.** `phytonet.env.PhytoNetEnv` (passes `check_env`).
  9-dim observation (counts, biomass, pooled sugar/water, **soil moisture**,
  **ambient light**), 3-logit action → **softmax allocation** (shoot/root/girth),
  reward = biomass gained. Episodes **randomise the scenario**.
- [x] **It learns (lightweight).** `phytonet.learn` — Cross-Entropy Method
  (numpy only). `scripts/train_cem.py` plots a learning curve.
- [x] **PPO (deep RL).** `phytonet.ppo` — a compact, readable CleanRL-style
  continuous PPO (actor-critic + GAE + clipped objective, torch). PPO **learns
  the optimal allocation** (reaches the measured ceiling, e.g. ~13.8 on drought,
  beating a naive shoot-heavy policy by ~+5). Run `scripts/train_ppo.py`. 48 tests.
  *Honest caveat:* it only ties **random**, because random allocation averages to
  ~balanced and balanced is robustly near-optimal (resource complementarity). The
  next lever is strengthening condition-dependent payoffs — see the project note.
- [x] **GNN policy with learnable messages (increment 1).** `phytonet.gnn` is a
  hand-written message-passing actor-critic (no PyG): each tissue reads its
  **local** features (kind, resources, height, local light/water), exchanges
  **learned messages** with neighbours over K rounds, then a graph readout picks
  the action. Per-tissue *observation* + learned inter-tissue *signals*; trained
  with `phytonet.ppo_gnn` (PPO). `scripts/train_gnn.py` compares it to the MLP
  baseline and prints the learned messages by tissue type. 54 tests.
- [x] **GNN increment 2 — per-tissue actions.** `phytonet.gnn.GNNNodeActorCritic`
  gives every tissue its *own* action ([grow, thicken]) from its message-informed
  embedding; the joint action's log-prob is the sum over tissues. The env runs in
  `per_tissue=True` mode (`step_local`), so growth is genuinely distributed —
  each tissue decides locally, gated by its own resources, no global budget.
  Trained with `phytonet.ppo_gnn.train_ppo_gnn_local`.
- [x] **Benchmark harness.** `phytonet.benchmark` runs controllers × scenarios ×
  seeds → mean ± std. First finding: the homogeneous env has a *spatially uniform,
  condition-independent* optimum ("balanced"), so per-tissue GNN == MLP == balanced
  everywhere — signaling has nothing to exploit. Motivated the next tier.
- [x] **Environment enrichment.** (1) **Spatial heterogeneity** — light brighter
  toward +x, soil water concentrated toward −x (`light_gradient`, `water_source_x`),
  so *where* a tissue sits decides its value (scenario `patchy`). (2) **Lifecycle**
  — mature leaves set seed (fitness) and the plant can starve to death (scenario
  `lifecycle`); the env reward adds reproduction and a death penalty. 69 tests.
- [x] **Per-tissue directional growth.** Per-tissue action `[grow, thicken, steer_x]`;
  each meristem steers along the heterogeneity axis. Hand-coded steering gains +2.0
  over no-steer on `patchy` — but PPO/GNN did **not** discover it (neural RL kept
  tying the global MLP). This, plus the project's "no brain" premise, motivated a
  pivot.
- [x] **Start-from-seed + viewer fixes.** Day 0 is a lone **seed** that germinates
  (no day-0 leaf); the 3D animation **pauses at the end** (no looping), the camera
  is **free to orbit**, and `--seed` is random per run for variety.
- [x] **Decentralized electrical-signaling controller** (`phytonet.signaling`).
  Every tissue runs the *same* interpretable rule (no central unit, no neural net):
  a **membrane potential** depolarised by local light/water, **electrotonically
  coupled** to neighbours, firing **action potentials** that propagate — and it
  *grows where excited*. Coordinated growth emerges from local cells exchanging
  electrical signals. Watch it: `run_season.py --dim 3 --signaling` colours tissue
  by potential. 76 tests.
- [x] **Evolved the electrical rule (CEM, no backprop, no network).**
  `phytonet.signaling.evolve_signaling` + `learn.cem_optimize` tune the 9 shared
  parameters for fitness (biomass + reproduction) across scenarios. Overall fitness
  16.1 → 16.5 with a clear **+2.0 in shade** (it lowered `grow_threshold` and raised
  `k_light` — interpretable). Gains are modest because most scenarios sit at
  plateaus (the recurring environment-headroom limit). `scripts/evolve_signaling.py`.
  79 tests.
- [x] **Signaling analysis (the research question).** `phytonet.analysis` +
  `scripts/analyze_signaling.py` instrument a run and report firing rate by tissue,
  membrane-potential traces, and whether firing **encodes the environment**. Key
  finding on `patchy` with the evolved rule: **root firing correlates +0.54 with
  local water** — the electrical signal genuinely tracks the water environment
  (resembling real plant electrical/hydraulic root signalling); seed & stem are the
  most active sources. 83 tests; all 9 entry-point scripts smoke-tested.
- [x] **Cross-paradigm benchmark.** `phytonet.benchmark` (outcome helpers) +
  `scripts/benchmark_all.py` train/evolve every paradigm — fixed baselines, MLP-PPO,
  per-tissue GNN, and the bioelectric controller — and score them on **one common
  metric** (final biomass + reproduction) across scenarios × seeds, with a single
  comparison figure (`figures/benchmark_all.png`). Answers: does brain-free
  bioelectric control rival the neural ones? 84 tests.
- [x] **Phase 1 — wound→defense electrical signalling + herbivory.** A bitten leaf
  loses biomass and emits a **wound signal** (a variation-potential-like channel,
  separate from the growth V) that **propagates systemically**; tissues carrying it
  **arm defense** (costing sugar), and defended tissue loses less to later bites.
  Verified: the signal spreads to up to 5 tissues and defense-on keeps more biomass
  than defense-off under herbivory (`scenario herbivory`). This is the canonical
  real-plant signal, emergent in the brain-free model. 89 tests.
- [x] **Phase 2 — learned signal *content*.** `phytonet.learned_signaling`: each
  cell carries a small **multi-channel signal**; one shared, interpretable linear
  rule — `s ← (1-leak)·s + W_in·sensors + W_couple·mean(neighbours)`, actions =
  `W_out·s` — whose ~30 weights are **evolved** (CEM, no backprop, no network). The
  plant now *discovers what to signal* (`W_in`), how tissues communicate
  (`W_couple`), and how to act (`W_out`); `channel_sensor_correlation` reads out
  what each learned channel came to encode. **Result:** the evolved learned rule
  beats the hand-designed Phase-1 rule overall (16.3 vs 16.0) and decisively under
  herbivory (12.4 vs 10.0), and it's *readable* — the plant discovered a channel
  encoding shoot-identity + light + **wound** (+0.74 / +0.64 / +0.49). 94 tests.
- [x] **Phase 3 — signalling vs real plants.** `phytonet.signal_analysis` +
  `scripts/phase3_analysis.py` run a controlled single-wound experiment and measure
  the response by graph hop-distance. **Findings** (and how they map to real plant
  electrical signalling): a local leaf wound arms **systemic** defense out to ~6
  hops (real: wound → defense in distant leaves); it travels as a **slow, finite
  wave** ~0.26 hops/step (real: mm–cm/s action/variation potentials); and it is
  **shoot-directional**, 0.53 vs 0.00 in roots (real: shoot-borne systemic wound
  signalling). 97 tests.
- [x] **Phase 4 — rigor & reproducibility.** `phytonet.experiments` runs every
  method **per-seed** (genuine training-seed variance for the evolved controllers),
  writes a tidy **results CSV**, and reduces it to **bootstrap 95% CIs** and a
  **permutation-test** significance comparison — numpy-only, no SciPy. Driven by
  **Hydra** configs (`configs/experiment.yaml`, `scripts/run_experiment.py`, e.g.
  `seeds=8 days=80`), with **optional W&B** logging (`wandb.enabled=true`; CSV is
  the reproducible source of truth either way). 101 tests.
- [x] **Richer environments — seasons.** Ambient light now **fades over the
  episode** (a growing season ending in autumn; `season_amplitude`, scenario
  `seasonal`), so conditions are **non-stationary** — a fixed allocation can't be
  optimal throughout. The plant front-loads growth while light is good and sets
  seed during the productive window. Added to the RL randomisation pool. 105 tests.
- [x] **Plant–plant competition.** `phytonet.competition.Arena` runs several
  plants in one world with **cross-plant shading** (a leaf is shaded by any taller
  leaf nearby) and a **fair shared water reservoir** (all roots draw from the same
  pre-step moisture). Findings: competition genuinely lowers fitness and makes
  *strategy* matter (a tall/shoot-heavy racer **loses** to balanced — it starves
  its roots), but the adaptive signaling controller **ties** the fixed one (19=19)
  — both converge to the same competitive equilibrium. 110 tests.
- [x] **Root-ward (bidirectional) signalling.** Added a root-origin **water-stress
  signal** (ABA-like): dry roots emit it, it propagates systemically **root→shoot**
  (verified: root stress 2.1 → shoot stress 0.9 under drought) and biases growth
  root-ward. Complements the shoot→root wound signal, so signalling is now
  bidirectional. (Mechanism works; drought fitness flat — the resource plateau.)
- [x] **Competitive co-evolution.** `phytonet.competition.coevolve_competitive`
  evolves a signalling rule (CEM) to *beat* an opponent head-to-head in the arena
  (relative biomass), both left/right arrangements to cancel position bias.
  **Finding:** co-evolved ties the fixed competitor (relative biomass +0.00, same
  as hand) — the arena has a strong equilibrium no strategy escapes. 113 tests.
- [x] **Boundary confirmed.** Both remaining directions (bidirectional signalling,
  co-evolution) work mechanically but yield no fitness edge — reinforcing that
  distributed signalling pays off *only* for the herbivory coordination task.
- [x] **Refinements.** (A) wound-propagation constants are now **evolved** (param
  vector 13→15); (B) the learned signal has **3 channels** (richer multiplexing);
  (C) **self-play co-evolution** (`coevolve_selfplay` — both sides co-adapt);
  (D) **signal-speed calibration** to mm/s vs measured plant action/variation
  potentials (`calibrate_speed`, shown in `phase3_analysis.py`). Plus
  `scripts/compare.py` for **side-by-side** controller comparison. 115 tests.
- [x] **FYP Tier 1 — rigor & validation.** Biological **validation** (the sim
  reproduces functional equilibrium emergently: root:shoot drought 1.0 > baseline
  0.6 > shade 0.3 — `validate.py`); a **baseline-fairness** audit (PPO converges,
  beats naive — `baseline_audit.py`); the herbivory win **characterised** as a
  dose-response with effect sizes (`herbivory_study.py`, Cohen's d = 2.57, p<0.001
  at moderate pressure); and a **dissertation skeleton** (`DISSERTATION.md`).
- [x] **FYP Tier 2 — rigor & reproducibility polish.** (a) **Holm** multiple-
  comparison correction + **Cohen's d** + a markdown **results table** in
  `experiments.py`/`run_experiment.py`; (b) **`reproduce_all.py`**, a pinned
  **`requirements-lock.txt`**, and **GitHub Actions CI** (`.github/workflows/ci.yml`);
  (c) the growth/signalling **timescale conflation** addressed via `signal_substeps`
  (faster signalling than growth). 123 tests.

## Reproducible experiments

```bash
pip install -e ".[rl,repro]"                 # torch + hydra (W&B optional: pip install wandb)
python scripts/run_experiment.py             # multi-seed, CSV + CIs + significance + figure
python scripts/run_experiment.py seeds=8 days=80 wandb.enabled=true
```
Outputs land in `results/` (`results.csv`, `experiment.png`). Every number is
reproducible from the seed list; the evolved controllers get a fresh CEM seed each.

## Model at a glance (MVP)

Each simulated day, in order:

| Stage | Where | What happens |
|-------|-------|--------------|
| Photosynthesis | `physiology` | leaves turn local light into sugar (down-regulated as they fill) |
| Water uptake | `physiology` | roots turn local water into water-credit (likewise) |
| Transport | `physiology` | sugar & water diffuse along edges toward equalization — the hand-coded *signaling* |
| Weather | `simulation` | rain refills the soil on a schedule; evaporation dries it |
| Maintenance | `physiology` | every tissue burns sugar proportional to its size |
| Growth (action) | `policy` | a limited budget is **allocated** between extending shoots, extending roots, and thickening — they compete |

Tune any of it via `SimConfig`, or pick a whole world with `--scenario`.

## 2D / 3D

The core (`world`, `plant`, `physiology`, `policy`, `simulation`) is
N-dimensional — positions are length-`dim` vectors and nothing hardcodes "2".
Only the renderers commit to a dimension: `viz.py` (2D) and `viz3d.py` (3D,
matplotlib mplot3d), chosen automatically by `visualize.py` based on `world.dim`.

**Physics (tropisms).** Growth direction is a weighted blend of:
- *gravitropism* — shoots grow up, roots grow down;
- *phototropism* — shoots bend toward the sun (`world.light_pos`);
- lateral wobble.

Move the sun off-centre to see the plant lean toward it — most visible in 3D.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -e ".[dev]"
```

## Run

```bash
python scripts/run_mvp.py                       # static 2D seedling
python scripts/run_season.py                    # 2D growing season (animated)
python scripts/run_season.py --dim 3 --sun 4 4 9            # 3D, rotate with the mouse
python scripts/run_season.py --dim 3 --scenario drought    # watch it cope with drought
python scripts/run_season.py --dim 3 --scenario monsoon    # wet soil (dark ground + rain)
python scripts/run_season.py --dim 3 --save-gif season3d.gif --no-show
python scripts/make_figures.py --outdir figures  # plant model + growth + strategy plots
python scripts/train_cem.py                     # lightweight learner (numpy CEM)
python scripts/train_ppo.py --scenario drought  # deep RL (PPO) — learns optimal allocation
python scripts/train_ppo.py                      # PPO across randomised conditions
python scripts/train_gnn.py --compare-mlp        # GNN (learnable messages) vs MLP baseline
python scripts/compare.py                        # side-by-side: controllers' performance + morphology
python scripts/benchmark_all.py                  # cross-paradigm benchmark (trains all, one figure)
python scripts/phase3_analysis.py --outdir figures  # wound signalling vs real plants (+ mm/s calibration)
python scripts/run_experiment.py                 # paper-grade multi-seed (CSV + CIs + significance)
pytest                                           # run the tests (115)
```

PPO needs torch (CPU is fine):
`pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu`

Scenarios: `baseline`, `drought`, `monsoon`, `shade`, `bright`, `patchy`
(spatially heterogeneous), `lifecycle` (reproduction + death). In 3D the ground
darkens with soil moisture and a rain effect appears on watering days; the HUD
shows day / biomass / soil %.

`--substeps` controls 3D smoothness/speed (in-between frames per day; higher =
slower and smoother). `--fps` sets GIF playback speed.

## The RL env (Gymnasium)

```python
from phytonet.env import PhytoNetEnv

# randomize=True -> each episode is a different scenario (drought/shade/...).
env = PhytoNetEnv(max_days=80, randomize=True)
obs, info = env.reset(seed=0)
for _ in range(80):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break
```

- **Observation** (9 floats, ~[0,1]): day, node/leaf/**root** counts, biomass,
  pooled sugar & water, **soil moisture**, **ambient light**.
- **Action** (3 logits, [-1,1] → softmax): how to split this step's growth
  **budget** between shoots / roots / girth — a real, competing trade-off.
- **Reward**: biomass gained per step (+ a small living bonus).
- **Conditions**: `randomize=True` samples a scenario per episode, so a policy
  must generalise across drought/shade/monsoon — pin one with
  `PhytoNetEnv(make_scenario("drought"), randomize=False)`.

Two learners ship: a numpy-only **CEM** (`phytonet.learn`) and a readable
**PPO** (`phytonet.ppo`, torch). When we add the GNN, the observation becomes the
per-tissue graph and the action includes the **learned inter-tissue messages**.
