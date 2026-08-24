# PhytoNet — Dissertation skeleton (FYP)

*A scaffold for the final report. Each section lists what to write and the
evidence/figure to use. `[CITE: ...]` marks where a real reference must be added
— **do not ship these placeholders**; fill them from the literature. Results and
figures referenced here exist in `figures/` and `results/`.*

---

## 0. Front matter
- Title, author, supervisor, institution, date, declaration of originality.
- **Abstract** (~250 words): see `WRITEUP.md` §Abstract — condense to the problem,
  approach (brain-free evolved electrical controller), and the three headline
  results (best overall; significant under herbivory; reproduces real-plant
  signalling signatures; boundary = coordination not variability).
- Acknowledgements, table of contents, list of figures/tables.

## 1. Introduction
- **Motivation.** Plants coordinate without a brain via slow chemical/electrical
  signals `[CITE: plant electrophysiology review, e.g. electrical signalling in
  plants]`. Contrast with connectome-driven simulation (fixed wiring → behaviour);
  PhytoNet inverts this (behaviour → discovered wiring).
- **Problem statement & research question.** Can a simulated plant *learn* its own
  inter-tissue signalling, and do the learned signals resemble real ones?
- **Aims & objectives** (numbered, measurable): build a graph-bodied plant
  simulator; implement controllers (fixed, neural, bioelectric); evolve a
  brain-free signalling rule; evaluate rigorously; compare to real plant
  signalling.
- **Contributions** (bullet list): the simulator + its validation; the brain-free
  evolvable electrical controller; the multi-seed comparison; the boundary finding.
- **Report structure.**

## 2. Background & Related Work
*This is the section most in need of real citations — prioritise it.*
- **Plant biology.** Tissues (root/stem/leaf), source–sink transport `[CITE]`,
  functional-equilibrium / balanced-growth theory (root:shoot shifts with the
  limiting resource) `[CITE]`, tropisms (photo/gravi) `[CITE]`, electrical
  signalling — action potentials & variation potentials, propagation velocities
  (~0.1–40 mm/s) `[CITE]`, systemic wound signalling & induced defence `[CITE]`,
  ABA-mediated root→shoot drought signalling `[CITE]`.
- **Computational plant models.** L-systems / functional-structural plant models
  `[CITE: Lindenmayer; FSPM reviews]` — note these model *form*, not *learned
  control*; this is the gap.
- **Learning & control.** Reinforcement learning and PPO `[CITE: Schulman et al.
  PPO]`; Graph Neural Networks & message passing `[CITE]`; neural cellular
  automata / self-organising systems `[CITE: Mordvintsev et al.]`; evolutionary
  strategies / CEM `[CITE]`; bioelectric / morphological computation `[CITE:
  Levin bioelectricity]`.
- **Gap.** No prior work lets a plant *evolve a decentralised electrical
  signalling rule* and asks whether it matches real signalling. PhytoNet does.

## 3. Design & Methodology
- **Plant as a graph of tissues** (§Methods/Body in `WRITEUP.md`): node kinds,
  germination, N-D positions.
- **Physiology** (per day): photosynthesis with feedback inhibition + canopy
  shading; water uptake from a dynamic reservoir; diffusive transport;
  maintenance; asymmetric, resource-coupled growth costs; lifecycle.
- **Environments**: the 8 scenarios; design rationale (each isolates a stressor).
- **Controllers**: fixed allocation; MLP-PPO; per-tissue GNN (message passing);
  hand bioelectric (membrane potential, electrotonic coupling, action potentials,
  wound + stress channels, systemic defence); learned bioelectric (evolved
  multi-channel linear rule). Give the equations for the electrical rule.
- **Learning**: CEM evolution of the shared parameters; PPO for the neural
  controllers. Reward = biomass + reproduction under survival/herbivory pressure.
- **Evaluation design**: common outcome metric; per-seed training; bootstrap CIs;
  permutation tests; effect sizes (Cohen's d); validation against known biology.

## 4. Implementation
- **Architecture**: ~23 modules, dependency layering (numpy/matplotlib core; torch
  for neural; hydra for experiments). Diagram the module graph.
- **Key components**: `world`, `plant`, `physiology`, `policy`, `simulation`,
  `signaling`, `learned_signaling`, `competition`, `experiments`, `signal_analysis`.
- **Reproducibility & testing**: 100+ pytest tests; Hydra-configured experiments;
  CSV as source of truth; seed control. (Note: add CI + pinned versions — Tier 2.)
- **Design decisions & trade-offs**: hand-rolled GNN vs PyG; matplotlib vs
  Polyscope; daily timestep (and its conflation of growth/signalling timescales).

## 5. Evaluation & Results
1. **Validation (biology).** Functional-equilibrium response reproduced
   *emergently*: root:shoot drought 1.00 > baseline 0.61 > shade 0.28
   (`figures/validation_functional_equilibrium.png`). Argue this evidences a
   reasonable abstraction.
2. **Baseline fairness.** PPO budget sweep — it learns, beats a naive fixed
   policy, and converges (`figures/baseline_audit.png`); ties random only because
   random allocation ≈ balanced ≈ optimal.
3. **Cross-paradigm benchmark.** `figures/benchmark_all.png` — bioelectric best
   overall; neural ties baselines.
4. **Multi-seed significance.** `results/experiment.png` / `results.csv` — evolved
   bioelectric vs balanced significant only under herbivory (+2.83, p = 0.008).
5. **Learned signal content.** `figures/learned_signal.png` — beats hand rule;
   channel encodes shoot/light/wound.
6. **Signalling vs real plants.** `figures/phase3_signal.png` — systemic spread,
   slow finite-speed wave (with mm/s calibration), shoot-directional.
7. **Herbivory characterisation.** `figures/herbivory_dose_response.png` — the
   defence benefit is **dose-dependent**: large & significant at moderate
   herbivory (intensity 0.3: Cohen's d = 2.57, p < 0.001), absent when the threat
   is trivial or overwhelming. This is the paper's positive contribution.
8. **The boundary (negative results, stated honestly).** Resource variability,
   seasons, competition, bidirectional signalling, and co-evolution all yield no
   significant edge — signalling pays off for coordination, not variability.

## 6. Discussion
- Interpret the boundary finding; relate to ecology (when do plants invest in
  signalling?) `[CITE]`.
- Compare the learned signalling code to real signalling qualitatively.
- **Threats to validity** (be candid): simplified physiology; 2-D/single-plant
  focus; hand-tuned scenario parameters; small N (mitigate by raising seeds).
  *Addressed:* the growth/signalling **timescale conflation** — `signal_substeps`
  now runs several propagation rounds per growth day, separating the fast
  signalling timescale from slow growth; **multiple comparisons** — Holm
  correction is applied (`experiments.holm_correction`) and **effect sizes**
  (Cohen's d) reported alongside p-values.

## 7. Conclusion & Future Work
- Restate contributions and the central finding.
- Future work: timescale separation; richer hormone/calcium channels; root-ward
  *defence*; 3-D multi-plant; biological calibration with measured velocities;
  PyG-batched training for larger sweeps.

## 8. References
*Replace every `[CITE: ...]` above with a real reference. Suggested anchor topics:
plant electrical/electrophysiological signalling; systemic wound signalling &
induced defence; ABA root–shoot drought signalling; source–sink transport;
functional-equilibrium / balanced-growth; L-systems & FSPMs; PPO; GNN message
passing; neural cellular automata; CEM/evolution strategies; bioelectricity &
morphological computation.*

## Appendices
- **A. Reproduce everything**: setup + the exact commands (see `README.md` Run
  section and `WRITEUP.md` §Reproducibility).
- **B. Full results tables** (from `results/results.csv`).
- **C. Hyperparameters** (PPOConfig, CEM settings, SignalingParams defaults).
