"""Analyse the electrical signals the plant uses -- the research question.

Runs a plant under the signaling controller while logging every cell's membrane
potential, firing events, and local stimulus, then summarises *what the signals
encode* and *how they're organised by tissue*:

* **firing rate by tissue type** -- which tissues are the electrically active
  signal sources;
* **does V encode the environment?** -- correlation of a tissue's potential with
  its local light / water (a signal that tracks the stimulus is doing real work);
* **shoot vs root activity** -- the directional organisation, to compare against
  real plant signalling (shoots as light/wound sources, source->sink propagation).

This needs no fitness headroom to be interesting -- it characterises the learned
signalling itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from phytonet.config import build_simulation
from phytonet.scenarios import make_scenario
from phytonet.signaling import SignalingParams, attach_signaling


@dataclass
class SignalLog:
    """Per-(step, node) records of the electrical activity."""
    day: list[int] = field(default_factory=list)
    kind: list[str] = field(default_factory=list)
    V: list[float] = field(default_factory=list)
    fired: list[bool] = field(default_factory=list)
    light: list[float] = field(default_factory=list)
    water: list[float] = field(default_factory=list)

    def as_arrays(self):
        return {k: np.array(getattr(self, k)) for k in
                ("day", "kind", "V", "fired", "light", "water")}


def run_and_log(params: SignalingParams | None = None, scenario: str = "patchy",
                seed: int = 0, days: int = 70) -> SignalLog:
    """Grow a plant under the signaling controller, logging electrical activity."""
    sim = build_simulation(make_scenario(scenario, start_from_seed=True, days=days,
                                         seed=seed, record_frames=False))
    ctrl = attach_signaling(sim, params)
    log = SignalLog()
    for _ in range(days):
        sim.step()
        for nid, node in sim.plant.nodes.items():
            log.day.append(sim.day)
            log.kind.append(node.kind.value)
            log.V.append(node.state.get("V", 0.0))
            log.fired.append(nid in ctrl.last_fired)
            log.light.append(sim.world.light_at(node.pos))
            log.water.append(sim.world.water_at(node.pos))
    return log


def firing_rate_by_kind(log: SignalLog) -> dict[str, float]:
    """Fraction of tissue-steps that fired an action potential, per tissue kind."""
    a = log.as_arrays()
    out = {}
    for k in ("seed", "root", "stem", "leaf"):
        mask = a["kind"] == k
        out[k] = float(a["fired"][mask].mean()) if mask.any() else 0.0
    return out


def potential_stimulus_correlation(log: SignalLog) -> dict[str, float]:
    """Correlation of membrane potential with the locally relevant stimulus:
    light for leaves/stems, water for roots. High = the signal tracks the env."""
    a = log.as_arrays()

    def corr(mask, stimulus):
        v, s = a["V"][mask], stimulus[mask]
        if v.size < 3 or v.std() < 1e-9 or s.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(v, s)[0, 1])

    return {
        "leaf~light": corr(a["kind"] == "leaf", a["light"]),
        "stem~light": corr(a["kind"] == "stem", a["light"]),
        "root~water": corr(a["kind"] == "root", a["water"]),
    }


def firing_stimulus_correlation(log: SignalLog) -> dict[str, float]:
    """Correlation of *firing* with the locally relevant stimulus. Positive means
    tissues in richer spots fire more -- the cleanest 'the signal encodes the
    environment' read (membrane V is muddied by post-firing resets)."""
    a = log.as_arrays()

    def corr(mask, stimulus):
        f, s = a["fired"][mask].astype(float), stimulus[mask]
        if f.size < 3 or f.std() < 1e-9 or s.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(f, s)[0, 1])

    return {
        "leaf fire~light": corr(a["kind"] == "leaf", a["light"]),
        "root fire~water": corr(a["kind"] == "root", a["water"]),
    }


def mean_potential_by_kind(log: SignalLog) -> dict[str, float]:
    a = log.as_arrays()
    return {k: (float(a["V"][a["kind"] == k].mean()) if (a["kind"] == k).any() else 0.0)
            for k in ("seed", "root", "stem", "leaf")}


def summarize(log: SignalLog) -> str:
    fr = firing_rate_by_kind(log)
    fc = firing_stimulus_correlation(log)
    mv = mean_potential_by_kind(log)
    lines = ["Electrical-signal analysis:", "  firing rate by tissue:"]
    lines += [f"    {k:5s}: {v:.3f}" for k, v in fr.items()]
    lines += ["  mean membrane potential by tissue:"]
    lines += [f"    {k:5s}: {v:.2f}" for k, v in mv.items()]
    lines += ["  firing ~ stimulus correlation (does the signal encode the env?):"]
    lines += [f"    {k:16s}: {v:+.2f}" for k, v in fc.items()]
    return "\n".join(lines)
