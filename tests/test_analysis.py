"""Tests for the electrical-signal analysis."""

import numpy as np

from phytonet.analysis import (
    firing_rate_by_kind, firing_stimulus_correlation, mean_potential_by_kind,
    potential_stimulus_correlation, run_and_log, summarize,
)


def test_run_and_log_records_every_tissue_step():
    log = run_and_log(scenario="patchy", seed=0, days=20)
    n = len(log.day)
    assert n > 0
    # All parallel arrays have the same length.
    for attr in ("kind", "V", "fired", "light", "water"):
        assert len(getattr(log, attr)) == n
    assert set(log.kind) <= {"seed", "root", "stem", "leaf"}


def test_firing_rates_are_fractions():
    log = run_and_log(scenario="baseline", seed=0, days=25)
    fr = firing_rate_by_kind(log)
    assert set(fr) == {"seed", "root", "stem", "leaf"}
    assert all(0.0 <= v <= 1.0 for v in fr.values())


def test_correlations_are_in_range_or_nan():
    log = run_and_log(scenario="patchy", seed=1, days=25)
    for d in (potential_stimulus_correlation(log), firing_stimulus_correlation(log)):
        for v in d.values():
            assert np.isnan(v) or (-1.0 <= v <= 1.0)


def test_summary_is_a_string():
    log = run_and_log(scenario="drought", seed=0, days=20)
    s = summarize(log)
    assert isinstance(s, str) and "firing rate" in s
    assert isinstance(mean_potential_by_kind(log), dict)
