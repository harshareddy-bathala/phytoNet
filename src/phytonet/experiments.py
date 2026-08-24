"""Paper-grade experiment runner: multi-seed, CSV, CIs, significance.

Turns the project's claims into reproducible numbers. Each method is trained /
configured *per seed* (so learned controllers get genuine training-seed variance),
then evaluated on every scenario. Results are written as a tidy CSV and reduced
to bootstrap confidence intervals and permutation-test p-values -- no SciPy, no
hidden state, fully reproducible from the seed list.

Methods:
    random, balanced            -- fixed global baselines
    bioelectric_hand            -- the hand-designed electrical rule
    bioelectric_evolved         -- electrical rule, CEM-evolved per seed
    learned_evolved             -- learned signal-content rule, CEM-evolved per seed
"""

from __future__ import annotations

import csv

import numpy as np

from phytonet.benchmark import (
    eval_global_outcome, eval_learned_outcome, eval_signaling_outcome,
)

METHODS = ("random", "balanced", "bioelectric_hand",
           "bioelectric_evolved", "learned_evolved")


def _evaluator(method: str, seed: int, evolve_iters: int, evolve_pop: int,
               train_days: int):
    """Build/train a method for one seed; return ``eval(scenario, eval_seed, days)``."""
    if method == "random":
        rng = np.random.default_rng(seed)
        act = lambda o: rng.uniform(-1, 1, 3).astype(np.float32)
        return lambda sc, es, d: float(eval_global_outcome(act, sc, [es], d)[0])
    if method == "balanced":
        act = lambda o: np.zeros(3, np.float32)
        return lambda sc, es, d: float(eval_global_outcome(act, sc, [es], d)[0])
    if method == "bioelectric_hand":
        from phytonet.signaling import SignalingParams
        p = SignalingParams()
        return lambda sc, es, d: float(eval_signaling_outcome(p, sc, [es], d)[0])
    if method == "bioelectric_evolved":
        from phytonet.signaling import evolve_signaling
        best, _ = evolve_signaling(seed=seed, iterations=evolve_iters,
                                   population=evolve_pop, days=train_days)
        return lambda sc, es, d: float(eval_signaling_outcome(best, sc, [es], d)[0])
    if method == "learned_evolved":
        from phytonet.learned_signaling import evolve_learned
        best, _ = evolve_learned(seed=seed, iterations=evolve_iters,
                                 population=evolve_pop, days=train_days)
        return lambda sc, es, d: float(eval_learned_outcome(best, sc, [es], d)[0])
    raise ValueError(f"unknown method {method!r}")


def run_experiment(methods, scenarios, seeds, days: int = 70,
                   evolve_iters: int = 8, evolve_pop: int = 16, progress=print):
    """Train each method per seed, evaluate on each scenario. Returns tidy records."""
    records = []
    for method in methods:
        for s in seeds:
            progress(f"  {method} (seed {s}) ...")
            ev = _evaluator(method, s, evolve_iters, evolve_pop, days)
            for sc in scenarios:
                records.append({"method": method, "scenario": sc, "seed": int(s),
                                "outcome": ev(sc, s, days)})
    return records


def write_csv(records, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "scenario", "seed", "outcome"])
        w.writeheader()
        w.writerows(records)


# -- statistics (numpy only) ----------------------------------------------
def _grouped(records):
    out: dict[tuple, list[float]] = {}
    for r in records:
        out.setdefault((r["method"], r["scenario"]), []).append(r["outcome"])
    return out


def bootstrap_ci(values, n: int = 2000, alpha: float = 0.05, seed: int = 0):
    """(mean, lo, hi) bootstrap confidence interval."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n, v.size))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def cohens_d(a, b) -> float:
    """Standardised effect size (Cohen's d) for the difference in means a - b,
    pooled standard deviation. |d|: ~0.2 small, ~0.5 medium, ~0.8 large."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    na, nb = a.size, b.size
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    sp = np.sqrt(sp2)
    return float((a.mean() - b.mean()) / sp) if sp > 1e-12 else float("nan")


def permutation_test(a, b, n: int = 5000, seed: int = 0) -> float:
    """Two-sided permutation test p-value for a difference in means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[:a.size].mean() - pool[a.size:].mean()) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n + 1)


def holm_correction(pvalues) -> list:
    """Holm-Bonferroni step-down adjustment for ``len(pvalues)`` comparisons.
    Controls the family-wise error rate; returns adjusted p-values in input order."""
    pv = list(pvalues)
    m = len(pv)
    if m == 0:
        return []
    order = np.argsort(pv)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pv[idx])
        adj[idx] = min(1.0, running)
    return adj


def summarize(records):
    """Per (method, scenario): mean and 95% bootstrap CI."""
    g = _grouped(records)
    return {k: bootstrap_ci(v) for k, v in g.items()}


def compare(records, method_a: str, method_b: str, scenarios):
    """Per scenario: mean difference (a - b), effect size (Cohen's d), permutation
    p, and the **Holm-adjusted** p across the scenarios compared (multiple-comparison
    correction)."""
    g = _grouped(records)
    out = {}
    for sc in scenarios:
        a, b = g.get((method_a, sc), []), g.get((method_b, sc), [])
        if a and b:
            out[sc] = {"diff": float(np.mean(a) - np.mean(b)),
                       "d": cohens_d(a, b), "p": permutation_test(a, b)}
    scs = list(out)
    for sc, padj in zip(scs, holm_correction([out[s]["p"] for s in scs])):
        out[sc]["p_holm"] = padj
    return out


def results_markdown(summ, methods, scenarios) -> str:
    """A markdown results table (mean [95% CI]) for the dissertation."""
    head = "| method | " + " | ".join(scenarios) + " |"
    sep = "|" + "---|" * (len(scenarios) + 1)
    rows = [head, sep]
    for m in methods:
        cells = []
        for sc in scenarios:
            mean, lo, hi = summ.get((m, sc), (float("nan"),) * 3)
            cells.append(f"{mean:.1f} [{lo:.1f}, {hi:.1f}]")
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(rows)
