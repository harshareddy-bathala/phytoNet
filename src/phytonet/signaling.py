"""Decentralized electrical-signaling controller -- the brain-free alternative.

Every tissue runs the *same* small, interpretable rule (no central controller, no
neural network). Each cell carries a **membrane potential** ``V`` (its electrical
state) and:

1. is **depolarised by local stimuli** -- light for shoots/leaves, water for roots
   (a tissue "senses" its environment);
2. is **electrotonically coupled** to its neighbours -- ``V`` spreads along edges
   like a plant's electrical signal (the cable equation);
3. **fires an action potential** when ``V`` crosses a threshold: it resets and
   sends a spike to neighbours, so excitation *propagates* as a wave;
4. **grows where it is excited** -- a depolarised meristem extends (steering up a
   sensed light/water gradient), a depolarised node thickens.

Coordinated growth therefore *emerges* from identical local cells exchanging
electrical signals -- distributed plant cognition, no brain. The rule's ~12
scalar parameters (:class:`SignalingParams`) are interpretable and will later be
*evolved* (CEM) for fitness -- learning without backprop or a central network.

It plugs in exactly where the policy/GNN did: ``sim.grow_fn = controller.step``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phytonet.plant import NodeKind, Plant
from phytonet.policy import GrowToLightPolicy
from phytonet.world import World


@dataclass
class SignalingParams:
    # Membrane dynamics (interpretable, all per-cell, all shared).
    v_rest: float = 0.0
    leak: float = 0.25          # decay of V back toward rest
    coupling: float = 0.25      # electrotonic spread of V along edges
    dt: float = 1.0
    threshold: float = 2.5      # action-potential firing threshold
    ap_strength: float = 1.0    # spike delivered to neighbours when a cell fires
    # Stimuli: how local conditions depolarise a cell.
    k_light: float = 0.7        # shoots/leaves depolarise in light
    k_water: float = 0.7        # roots depolarise where there's water
    # Growth gates: what an excited cell does.
    grow_threshold: float = 0.6   # meristem extends when V exceeds this
    thicken_threshold: float = 1.6  # node thickens when V exceeds this
    steer_gain: float = 2.0       # steer growth up the sensed light/water gradient
    # Defense: a tissue carrying a *wound signal* arms its defenses (costing
    # sugar); defended tissue takes less damage from later bites. The wound
    # signal travels on its own channel ``wound`` (a variation-potential-like
    # signal, distinct from the growth excitation V), so defense only fires in
    # response to real damage -- not to ordinary light-driven activity.
    defense_threshold: float = 0.5  # wound-signal level above which a cell arms
    defense_gain: float = 0.4       # defense added per signalled step
    defense_cost: float = 0.3       # sugar spent to arm defense
    defense_decay: float = 0.05     # defense relaxes when the signal subsides
    # Wound-signal propagation (now EVOLVED -- see to_vector): how far/fast the
    # wound signal travels. Defaults give a systemic, several-hop wave.
    wound_coupling: float = 0.45
    wound_decay: float = 0.12
    # Root-origin water-stress signal (the other direction: root -> shoot, like a
    # real ABA/hydraulic drought signal). Dry roots emit it; it propagates
    # systemically and biases growth toward roots (invest where the water is).
    stress_threshold: float = 0.5  # local water below this marks a root stressed
    stress_gain: float = 1.0
    stress_coupling: float = 0.4
    stress_decay: float = 0.2
    stress_bias: float = 0.5       # how strongly stress shifts growth root-ward

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.leak, self.coupling, self.threshold, self.ap_strength,
            self.k_light, self.k_water, self.grow_threshold,
            self.thicken_threshold, self.steer_gain,
            self.defense_threshold, self.defense_gain, self.defense_cost,
            self.defense_decay, self.wound_coupling, self.wound_decay,
        ], dtype=float)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "SignalingParams":
        return cls(leak=float(v[0]), coupling=float(v[1]), threshold=float(v[2]),
                   ap_strength=float(v[3]), k_light=float(v[4]), k_water=float(v[5]),
                   grow_threshold=float(v[6]), thicken_threshold=float(v[7]),
                   steer_gain=float(v[8]), defense_threshold=float(v[9]),
                   defense_gain=float(v[10]), defense_cost=float(v[11]),
                   defense_decay=float(v[12]),
                   wound_coupling=float(v[13]) if len(v) > 13 else 0.45,
                   wound_decay=float(v[14]) if len(v) > 14 else 0.12)


class SignalingController:
    """Shared local electrical rule applied to every tissue (no central unit)."""

    def __init__(self, policy: GrowToLightPolicy, params: SignalingParams | None = None,
                 signal_substeps: int = 1):
        self.policy = policy          # reused only for the atomic growth ops
        self.p = params or SignalingParams()
        self.last_fired: set[int] = set()   # ids that fired an AP on the last step
        # Signalling runs faster than growth: ``signal_substeps`` rounds of
        # electrical/wound/stress propagation happen per growth day. This
        # separates the (fast) signalling timescale from the (slow) growth one --
        # addressing the single-timestep conflation. 1 = original behaviour.
        self.signal_substeps = max(1, int(signal_substeps))

    # -- electrical dynamics ------------------------------------------------
    def _electrical_step(self, plant: Plant, world: World) -> set[int]:
        """One synchronous update of every cell's membrane potential. Returns the
        set of cells that fired an action potential this step."""
        p = self.p
        V = {nid: n.state.get("V", 0.0) for nid, n in plant.nodes.items()}
        delta = {nid: 0.0 for nid in plant.nodes}
        for nid, node in plant.nodes.items():
            d = -p.leak * (V[nid] - p.v_rest)                  # leak toward rest
            if node.kind in (NodeKind.LEAF, NodeKind.STEM, NodeKind.SEED):
                d += p.k_light * world.light_at(node.pos)      # light stimulus
            if node.kind in (NodeKind.ROOT, NodeKind.SEED):
                d += p.k_water * world.water_at(node.pos)      # water stimulus
            delta[nid] += d
        for a, b in plant.edges:                                # electrotonic coupling
            diff = V[a] - V[b]
            delta[a] -= p.coupling * diff
            delta[b] += p.coupling * diff
        for nid, node in plant.nodes.items():
            node.state["V"] = V[nid] + p.dt * delta[nid]

        # Action potentials: cells over threshold fire -> reset + spike neighbours.
        fired = {nid for nid, n in plant.nodes.items() if n.state["V"] >= p.threshold}
        if fired:
            boost = {nid: 0.0 for nid in plant.nodes}
            for nid in fired:
                for nb in plant.neighbors(nid):
                    boost[nb] += p.ap_strength
            for nid in fired:
                plant.nodes[nid].state["V"] = p.v_rest
            for nid, b in boost.items():
                if nid not in fired:
                    plant.nodes[nid].state["V"] += b
        return fired

    # -- sensed gradient (local tropism, no global knowledge) ---------------
    def _steer(self, node, world: World) -> float:
        dx = 0.6
        plus = node.pos.copy(); plus[0] += dx
        minus = node.pos.copy(); minus[0] -= dx
        if node.kind == NodeKind.STEM:
            grad = world.light_at(plus) - world.light_at(minus)
        else:
            grad = world.water_at(plus) - world.water_at(minus)
        return float(np.clip(self.p.steer_gain * grad, -2.0, 2.0))

    def _wound_step(self, plant: Plant) -> None:
        """Propagate + fade the wound signal along edges (a slow systemic signal
        separate from V). Herbivory injects it at the bite site."""
        p = self.p
        W = {nid: n.state.get("wound", 0.0) for nid, n in plant.nodes.items()}
        delta = {nid: -p.wound_decay * W[nid] for nid in plant.nodes}
        for a, b in plant.edges:
            diff = W[a] - W[b]
            delta[a] -= p.wound_coupling * diff
            delta[b] += p.wound_coupling * diff
        for nid, node in plant.nodes.items():
            node.state["wound"] = max(0.0, W[nid] + delta[nid])

    def _stress_step(self, plant: Plant, world: World) -> None:
        """Dry roots emit a water-stress signal that propagates systemically
        (root -> shoot) and decays -- the up-ward counterpart of the wound signal."""
        p = self.p
        for root in plant.nodes_of(NodeKind.ROOT):
            deficit = p.stress_threshold - world.water_at(root.pos)
            if deficit > 0:
                root.state["stress"] = root.state.get("stress", 0.0) + p.stress_gain * deficit
        S = {nid: n.state.get("stress", 0.0) for nid, n in plant.nodes.items()}
        delta = {nid: -p.stress_decay * S[nid] for nid in plant.nodes}
        for a, b in plant.edges:
            diff = S[a] - S[b]
            delta[a] -= p.stress_coupling * diff
            delta[b] += p.stress_coupling * diff
        for nid, n in plant.nodes.items():
            n.state["stress"] = max(0.0, S[nid] + delta[nid])

    def _defense(self, plant: Plant) -> None:
        """Tissues carrying the wound signal arm their defenses; defense relaxes
        otherwise. Read back by herbivory to blunt the next bite."""
        p = self.p
        for node in plant.nodes.values():
            d = node.state.get("defense", 0.0) * (1.0 - p.defense_decay)
            if (node.state.get("wound", 0.0) >= p.defense_threshold
                    and node.state["sugar"] >= p.defense_cost):
                node.state["sugar"] -= p.defense_cost
                d = min(1.0, d + p.defense_gain)
            node.state["defense"] = d

    # -- the growth phase (matches the grow_fn signature) -------------------
    def step(self, plant: Plant, world: World) -> None:
        # Fast signalling timescale: several propagation rounds per growth day.
        fired: set[int] = set()
        for _ in range(self.signal_substeps):
            fired |= self._electrical_step(plant, world)
            self._wound_step(plant)
            self._stress_step(plant, world)
        self.last_fired = fired
        self._defense(plant)
        pol = self.policy
        for node in list(plant.nodes.values()):
            v = node.state.get("V", 0.0)
            # The stress signal lowers the bar to grow roots and raises it for
            # shoots -> the plant shifts investment toward water under drought.
            s = node.state.get("stress", 0.0)
            if node.kind == NodeKind.ROOT:
                thr = self.p.grow_threshold - self.p.stress_bias * s
            else:
                thr = self.p.grow_threshold + self.p.stress_bias * s
            if node.meristematic and (v >= thr or node.id in fired):
                pol.extend_tip(plant, world, node, steer_x=self._steer(node, world))
            if v >= self.p.thicken_threshold:
                pol.thicken_one(node)


def attach_signaling(sim, params: SignalingParams | None = None) -> SignalingController:
    """Make ``sim`` grow via the electrical controller instead of its policy."""
    controller = SignalingController(sim.policy, params)
    sim.grow_fn = controller.step
    return controller


# -- evolving the shared rule (learning, no backprop / no network) ---------
_PARAM_LO, _PARAM_HI = 0.0, 8.0   # all 9 parameters are non-negative


def fitness(params: SignalingParams, scenario: str, seed: int, days: int = 60,
            repro_weight: float = 3.0) -> float:
    """Lifetime fitness of a plant grown by ``params`` in one scenario: final
    biomass plus a reward for seeds set (reproduction). A plant that starves to
    death stops accumulating, so survival is implicitly rewarded."""
    from phytonet.config import build_simulation
    from phytonet.scenarios import make_scenario
    sim = build_simulation(make_scenario(scenario, start_from_seed=True,
                                         days=days, seed=seed, record_frames=False))
    attach_signaling(sim, params)
    sim.run(days)
    return float(sim.history["biomass"][-1] + repro_weight * sim.seeds)


def evaluate_params(vector: np.ndarray, scenarios, seeds, days: int = 60) -> float:
    """Mean fitness of a parameter vector over scenarios x seeds (for CEM)."""
    params = SignalingParams.from_vector(np.clip(vector, _PARAM_LO, _PARAM_HI))
    scores = [fitness(params, sc, s, days) for sc in scenarios for s in seeds]
    return float(np.mean(scores))


def evolve_signaling(scenarios=("baseline", "patchy", "drought", "shade",
                                "lifecycle", "herbivory"),
                     seeds=range(2), days: int = 60, iterations: int = 12,
                     population: int = 20, seed: int = 0):
    """Evolve the shared :class:`SignalingParams` with CEM. Returns
    ``(best_params, result)`` where ``result.history`` is the learning curve."""
    from phytonet.learn import cem_optimize
    x0 = SignalingParams().to_vector()
    result = cem_optimize(
        lambda v: evaluate_params(v, scenarios, seeds, days),
        x0, iterations=iterations, population=population, seed=seed,
        lo=_PARAM_LO, hi=_PARAM_HI,
    )
    best = SignalingParams.from_vector(result.best_theta
                                       if result.best_theta is not None else x0)
    return best, result
