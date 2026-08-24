"""Phase 2: a decentralized controller whose *signal content* is learned.

The hand-designed controller (``signaling.py``) wired the membrane potential to
light/water by hand. Here each cell instead carries a small **multi-channel
signal** ``s`` and the rule is one shared, interpretable linear law whose weights
are *evolved* (CEM, no backprop, no network):

    sensors x_i  = [bias, light, water, sugar, size, is_root, is_shoot, wound]
    s_i  <- clip( (1-leak) * s_i  +  W_in @ x_i  +  W_couple @ mean_neighbour(s) )
    a_i  =  W_out @ s_i            # action drives: [grow, thicken, steer, defend]

So the plant *discovers what to signal* (``W_in`` decides what each channel
encodes), *how tissues communicate* (``W_couple`` mixes neighbour signals), and
*how to act on signals* (``W_out``). Every cell shares the same weights -- no
central unit. The weights are few (~30) and readable: you can inspect what each
learned channel came to represent (see ``channel_sensor_correlation``).
"""

from __future__ import annotations

import numpy as np

from phytonet.plant import NodeKind, Plant
from phytonet.policy import GrowToLightPolicy
from phytonet.world import World

SENSOR_DIM = 8
N_CHANNELS = 3   # richer signal: more channels -> the plant can multiplex more
N_ACTIONS = 4    # grow, thicken, steer, defend
_THETA = N_CHANNELS * SENSOR_DIM + N_CHANNELS * N_CHANNELS + N_CHANNELS + N_ACTIONS * N_CHANNELS

SENSOR_NAMES = ["bias", "light", "water", "sugar", "size", "is_root", "is_shoot", "wound"]


class LearnedSignalingParams:
    """Flat parameter vector packing W_in, W_couple, leak, W_out."""

    def __init__(self, theta: np.ndarray | None = None):
        self.theta = (np.zeros(_THETA) if theta is None
                      else np.asarray(theta, dtype=float).copy())

    @property
    def size(self) -> int:
        return _THETA

    def unpack(self):
        k, s, a = N_CHANNELS, SENSOR_DIM, N_ACTIONS
        t, i = self.theta, 0
        W_in = t[i:i + k * s].reshape(k, s); i += k * s
        W_c = t[i:i + k * k].reshape(k, k); i += k * k
        leak = np.clip(t[i:i + k], 0.0, 1.0); i += k
        W_out = t[i:i + a * k].reshape(a, k); i += a * k
        return W_in, W_c, leak, W_out

    def to_vector(self) -> np.ndarray:
        return self.theta.copy()

    @classmethod
    def from_vector(cls, v) -> "LearnedSignalingParams":
        return cls(v)

    @classmethod
    def default(cls) -> "LearnedSignalingParams":
        """A sensible, *growing* starting point for evolution: channel 0 is an
        excitation driven by light+water that drives growth; channel 1 carries
        the wound signal and drives defense. Evolution refines from here."""
        W_in = np.zeros((N_CHANNELS, SENSOR_DIM))
        W_in[0, 0], W_in[0, 1], W_in[0, 2] = 0.3, 0.7, 0.7   # ch0: bias + light + water
        if N_CHANNELS > 1:
            W_in[1, 7] = 1.0                                  # ch1: wound
        W_c = np.eye(N_CHANNELS) * 0.3                        # mild self-persistence
        leak = np.full(N_CHANNELS, 0.3)
        W_out = np.zeros((N_ACTIONS, N_CHANNELS))
        W_out[0, 0], W_out[1, 0] = 1.0, 0.5                   # grow, thicken <- ch0
        if N_CHANNELS > 1:
            W_out[3, 1] = 1.0                                 # defend <- ch1
        # extra channels (>=2) start silent; evolution can recruit them.
        return cls(np.concatenate([W_in.ravel(), W_c.ravel(), leak, W_out.ravel()]))


def _sensors(node, world: World) -> np.ndarray:
    x = np.zeros(SENSOR_DIM)
    x[0] = 1.0
    x[1] = world.light_at(node.pos) / 2.0
    x[2] = world.water_at(node.pos)
    x[3] = node.state["sugar"] / 6.0
    x[4] = node.state["size"] / 4.0
    x[5] = 1.0 if node.kind == NodeKind.ROOT else 0.0
    x[6] = 1.0 if node.kind in (NodeKind.STEM, NodeKind.LEAF) else 0.0
    x[7] = node.state.get("wound", 0.0)
    return x


class LearnedSignalingController:
    """Shared evolved linear signalling rule applied to every tissue."""

    # Fixed (non-evolved) auxiliaries shared with the hand controller.
    defense_gain, defense_cost, defense_decay = 0.4, 0.3, 0.05
    wound_coupling, wound_decay = 0.45, 0.12

    def __init__(self, policy: GrowToLightPolicy, params: LearnedSignalingParams | None = None):
        self.policy = policy
        self.p = params or LearnedSignalingParams.default()

    def _wound_step(self, plant: Plant) -> None:
        W = {nid: n.state.get("wound", 0.0) for nid, n in plant.nodes.items()}
        delta = {nid: -self.wound_decay * W[nid] for nid in plant.nodes}
        for a, b in plant.edges:
            diff = W[a] - W[b]
            delta[a] -= self.wound_coupling * diff
            delta[b] += self.wound_coupling * diff
        for nid, n in plant.nodes.items():
            n.state["wound"] = max(0.0, W[nid] + delta[nid])

    def step(self, plant: Plant, world: World) -> None:
        W_in, W_c, leak, W_out = self.p.unpack()
        self._wound_step(plant)

        s = {nid: np.asarray(n.state.get("sig", np.zeros(N_CHANNELS)), dtype=float)
             for nid, n in plant.nodes.items()}
        nbr = {}
        for nid in plant.nodes:
            ns = plant.neighbors(nid)
            nbr[nid] = (np.mean([s[j] for j in ns], axis=0) if ns
                        else np.zeros(N_CHANNELS))
        for nid, node in plant.nodes.items():
            x = _sensors(node, world)
            new = (1.0 - leak) * s[nid] + W_in @ x + W_c @ nbr[nid]
            node.state["sig"] = np.clip(new, -4.0, 4.0)
            node.state["V"] = float(node.state["sig"][0])   # channel 0 for viz

        pol = self.policy
        for node in list(plant.nodes.values()):
            a = W_out @ node.state["sig"]
            if node.meristematic and a[0] > 0:
                pol.extend_tip(plant, world, node, steer_x=float(np.clip(a[2], -2, 2)))
            if a[1] > 0:
                pol.thicken_one(node)
            d = node.state.get("defense", 0.0) * (1.0 - self.defense_decay)
            if a[3] > 0 and node.state["sugar"] >= self.defense_cost:
                node.state["sugar"] -= self.defense_cost
                d = min(1.0, d + self.defense_gain)
            node.state["defense"] = d


def attach_learned_signaling(sim, params: LearnedSignalingParams | None = None):
    controller = LearnedSignalingController(sim.policy, params)
    sim.grow_fn = controller.step
    return controller


# -- evolving the learned rule --------------------------------------------
def evaluate_learned(vector: np.ndarray, scenarios, seeds, days: int = 60) -> float:
    from phytonet.config import build_simulation
    from phytonet.scenarios import make_scenario
    params = LearnedSignalingParams(np.clip(vector, -3.0, 3.0))
    scores = []
    for sc in scenarios:
        for s in seeds:
            sim = build_simulation(make_scenario(sc, days=days, seed=s,
                                                 record_frames=False))
            attach_learned_signaling(sim, params)
            sim.run(days)
            scores.append(sim.history["biomass"][-1] + 3.0 * sim.seeds)
    return float(np.mean(scores))


def evolve_learned(scenarios=("baseline", "patchy", "drought", "shade",
                              "lifecycle", "herbivory"),
                   seeds=range(2), days: int = 60, iterations: int = 14,
                   population: int = 28, seed: int = 0):
    from phytonet.learn import cem_optimize
    x0 = LearnedSignalingParams.default().to_vector()
    result = cem_optimize(lambda v: evaluate_learned(v, scenarios, seeds, days),
                          x0, iterations=iterations, population=population,
                          init_std=0.3, seed=seed, lo=-3.0, hi=3.0)
    best = LearnedSignalingParams(result.best_theta if result.best_theta is not None else x0)
    return best, result


def channel_sensor_correlation(params, scenario="patchy", seed=0, days=60):
    """What did each learned channel come to encode? Correlate each signal
    channel with each sensor across all tissue-steps of a run."""
    from phytonet.config import build_simulation
    from phytonet.scenarios import make_scenario
    sim = build_simulation(make_scenario(scenario, days=days, seed=seed,
                                         record_frames=False))
    attach_learned_signaling(sim, params)
    sig_log, sens_log = [], []
    for _ in range(days):
        sim.step()
        for node in sim.plant.nodes.values():
            sig_log.append(np.asarray(node.state.get("sig", np.zeros(N_CHANNELS))))
            sens_log.append(_sensors(node, sim.world))
    S = np.array(sig_log)
    X = np.array(sens_log)
    out = {}
    for c in range(N_CHANNELS):
        for j in range(SENSOR_DIM):
            if S[:, c].std() > 1e-9 and X[:, j].std() > 1e-9:
                out[(c, SENSOR_NAMES[j])] = float(np.corrcoef(S[:, c], X[:, j])[0, 1])
    return out
