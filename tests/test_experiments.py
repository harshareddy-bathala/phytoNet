"""Tests for the paper-grade experiment runner and statistics."""

import csv

import numpy as np

from phytonet.experiments import (
    bootstrap_ci, compare, permutation_test, run_experiment, summarize, write_csv,
)


def test_bootstrap_ci_on_constant_data():
    mean, lo, hi = bootstrap_ci([5.0, 5.0, 5.0, 5.0])
    assert mean == 5.0 and lo == 5.0 and hi == 5.0


def test_permutation_test_distinguishes_distributions():
    same = permutation_test([5, 5, 5, 5], [5, 5, 5, 5])
    diff = permutation_test([10, 11, 9, 10], [1, 2, 1, 0])
    assert same > 0.5            # no difference -> large p
    assert diff < 0.1            # clear difference -> small p


def test_run_experiment_and_csv(tmp_path):
    recs = run_experiment(["random", "balanced"], ["baseline", "patchy"],
                          seeds=range(2), days=20, progress=lambda *_: None)
    assert len(recs) == 2 * 2 * 2          # methods x seeds x scenarios
    assert {r["method"] for r in recs} == {"random", "balanced"}
    assert all(np.isfinite(r["outcome"]) for r in recs)

    path = tmp_path / "results.csv"
    write_csv(recs, str(path))
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == len(recs) and "outcome" in rows[0]


def test_summarize_and_compare():
    recs = run_experiment(["random", "balanced"], ["baseline"],
                          seeds=range(3), days=20, progress=lambda *_: None)
    summ = summarize(recs)
    assert ("balanced", "baseline") in summ
    cmp = compare(recs, "balanced", "random", ["baseline"])
    assert "baseline" in cmp and "p" in cmp["baseline"]


def test_cohens_d_effect_size():
    from phytonet.experiments import cohens_d
    import numpy as np
    big = cohens_d([10, 11, 9, 10, 10], [1, 2, 1, 0, 1])     # clearly large
    none = cohens_d([5, 5.1, 4.9, 5, 5], [5, 4.9, 5.1, 5, 5])
    assert big > 2.0 and abs(none) < 0.5


def test_holm_correction_inflates_and_bounds():
    from phytonet.experiments import holm_correction
    adj = holm_correction([0.01, 0.01, 0.01])
    assert all(abs(a - 0.03) < 1e-9 for a in adj)         # 3 * 0.01
    one = holm_correction([0.5, 0.5])
    assert all(0.0 <= a <= 1.0 for a in one)
    assert holm_correction([]) == []


def test_compare_reports_effect_size_and_holm():
    recs = run_experiment(["balanced", "random"], ["baseline", "patchy"],
                          seeds=range(3), days=20, progress=lambda *_: None)
    cmp = compare(recs, "balanced", "random", ["baseline", "patchy"])
    for sc in ["baseline", "patchy"]:
        assert {"diff", "d", "p", "p_holm"} <= set(cmp[sc])
        assert cmp[sc]["p_holm"] >= cmp[sc]["p"] - 1e-9   # correction never decreases p


def test_results_markdown_table():
    from phytonet.experiments import results_markdown, summarize
    recs = run_experiment(["balanced"], ["baseline"], seeds=range(2), days=20,
                          progress=lambda *_: None)
    md = results_markdown(summarize(recs), ["balanced"], ["baseline"])
    assert md.startswith("| method |") and "balanced" in md
