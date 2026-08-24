"""Gymnasium environment wrapping the PhytoNet simulation.

This is the seam between the simulation and *learning*. *No agent is trained
here* -- it exposes the sim as a standard ``gymnasium.Env`` so PPO (and later a
GNN policy) can be pointed at it without touching the sim.

Each episode can use a different world (``randomize=True``), so a policy is
pushed to cope with drought, shade, monsoon, etc. -- "as it gets tested [in
varied conditions] it learns [to handle them]" (domain randomization).

What the agent controls (action, 3 logits in [-1, 1] -> softmax allocation):
    how to split this step's limited growth budget between
        shoots  (height + leaves -> capture light)
        roots   (-> capture water)
        girth   (thickening / storage)
    The right split depends on the world -- roots in drought, shoots in shade --
    so a fixed policy can't win everywhere and there is real signal to learn.

What it sees (observation, each ~[0, 1]):
    [day_fraction, n_nodes, n_leaves, n_roots, biomass, sugar_pool,
     water_pool, soil_moisture, ambient_light]

Reward: biomass gained this step (+ a small living bonus). The graph (per-node)
observation and richer reward (reproduction, survival) come with the GNN step.
"""

from __future__ import annotations

import dataclasses

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:  # pragma: no cover - helpful message
    raise ModuleNotFoundError(
        "PhytoNet's Gym env needs `gymnasium`. Install with `pip install -e .`."
    ) from exc

from phytonet.config import SimConfig, build_simulation
from phytonet.plant import NodeKind
from phytonet.scenarios import RANDOM_SCENARIOS, make_scenario

# Rough maxima to scale each observation into ~[0, 1] (presentation only).
_NORM = np.array([1.0, 200.0, 40.0, 40.0, 200.0, 100.0, 100.0, 1.0, 2.0],
                 dtype=np.float32)

# Per-tissue (graph) observation: one feature row per node. The GNN policy reads
# this instead of the global summary above.
_KINDS = [NodeKind.SEED, NodeKind.ROOT, NodeKind.STEM, NodeKind.LEAF]
NODE_FEATURE_DIM = len(_KINDS) + 8  # 4 one-hot + 8 scalars (see graph_observation)


class PhytoNetEnv(gym.Env):
    """A single plant growing over a season, as a Gymnasium environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Per-tissue action: each tissue gets [grow, thicken, steer_x].
    # grow/thicken: >0 means do it. steer_x: bias growth direction along x
    # (toward +x light / -x water), enabling tissues to navigate to good spots.
    NODE_ACTION_DIM = 3

    def __init__(self, config: SimConfig | None = None, max_days: int = 80,
                 randomize: bool = True, render_mode: str | None = None,
                 per_tissue: bool = False):
        super().__init__()
        self.base_config = config
        self.max_days = max_days
        # Randomise scenarios only when no fixed config was pinned.
        self.randomize = randomize and config is None
        self.render_mode = render_mode
        # per_tissue=True -> the growth phase is driven by per-node actions
        # supplied via step_local() (the GNN distributed controller); the global
        # Box action_space is then unused.
        self.per_tissue = per_tissue

        self.observation_space = spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self.sim = None
        self._prev_biomass = 0.0
        self._prev_seeds = 0
        self.repro_value = 2.0    # reward per seed set (lifecycle scenarios)
        self.death_penalty = 5.0  # one-off penalty if the plant starves to death
        self._graph_node_ids: list[int] = []
        self._pending: dict[int, np.ndarray] = {}
        self.scenario_name = "baseline"

    # -- helpers -----------------------------------------------------------
    def _sample_config(self) -> SimConfig:
        if self.base_config is not None:
            self.scenario_name = "fixed"
            return dataclasses.replace(self.base_config)
        if self.randomize:
            name = str(self.np_random.choice(list(RANDOM_SCENARIOS)))
            cfg = make_scenario(name)
            cfg.light_intensity *= float(self.np_random.uniform(0.8, 1.2))
            cfg.seed = int(self.np_random.integers(0, 1_000_000))
            self.scenario_name = name
            return cfg
        self.scenario_name = "baseline"
        return SimConfig()

    def _observation(self) -> np.ndarray:
        h = self.sim.history
        raw = np.array([
            self.sim.day / self.max_days,
            h["n_nodes"][-1], h["n_leaves"][-1], h["n_roots"][-1],
            h["biomass"][-1], h["total_sugar"][-1], h["total_water"][-1],
            h["moisture"][-1], self.sim.world.light_intensity,
        ], dtype=np.float32)
        return np.clip(raw / _NORM, 0.0, 1.0).astype(np.float32)

    def graph_observation(self):
        """Per-tissue graph: ``(node_features [N, F], edge_index [2, 2E])``.

        Each row is what one tissue 'senses' -- its kind, its own resource load
        and size, where it sits, and its *local* light/water -- so the GNN reads
        local environments and exchanges learned messages along the edges. Edges
        are undirected, emitted in both directions for message passing.
        """
        w = self.sim.world
        phys, gp = self.sim.physiology, self.sim.policy.p
        up = w.up_axis
        span = float(w.bounds_high[up] - w.ground_level)
        day_frac = self.sim.day / self.max_days
        light_norm = max(w.light_intensity, 1e-6)

        ids = sorted(self.sim.plant.nodes)
        self._graph_node_ids = ids  # row order, for mapping per-tissue actions back
        index = {nid: i for i, nid in enumerate(ids)}
        feats = np.zeros((len(ids), NODE_FEATURE_DIM), dtype=np.float32)
        for row, nid in enumerate(ids):
            node = self.sim.plant.nodes[nid]
            feats[row, _KINDS.index(node.kind)] = 1.0          # kind one-hot
            feats[row, 4] = node.state["sugar"] / phys.sugar_cap
            feats[row, 5] = node.state["water"] / phys.water_cap
            feats[row, 6] = node.state["size"] / gp.max_size
            feats[row, 7] = w.height_of(node.pos) / span        # signed height
            feats[row, 8] = 1.0 if node.meristematic else 0.0
            feats[row, 9] = w.light_at(node.pos) / light_norm    # local light
            feats[row, 10] = w.water_at(node.pos)                # local water
            feats[row, 11] = day_frac
        feats = np.clip(feats, -2.0, 2.0)

        edges = []
        for a, b in self.sim.plant.edges:
            edges.append((index[a], index[b]))
            edges.append((index[b], index[a]))
        edge_index = (np.array(edges, dtype=np.int64).T
                      if edges else np.zeros((2, 0), dtype=np.int64))
        return feats, edge_index

    def _apply_action(self, action: np.ndarray) -> None:
        # 3 logits -> softmax allocation of the growth budget (shoot/root/girth).
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        w = np.exp(2.0 * a)  # temperature 2 sharpens the split a little
        w = w / w.sum()
        p = self.sim.policy.p
        p.shoot_alloc = float(w[0])
        p.root_alloc = float(w[1])
        p.thicken_alloc = float(w[2])

    def _info(self) -> dict:
        h = self.sim.history
        p = self.sim.policy.p
        return {
            "scenario": self.scenario_name,
            "day": self.sim.day,
            "n_nodes": h["n_nodes"][-1],
            "n_roots": h["n_roots"][-1],
            "biomass": h["biomass"][-1],
            "moisture": h["moisture"][-1],
            "alloc": (p.shoot_alloc, p.root_alloc, p.thicken_alloc),
        }

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg = self._sample_config()
        cfg.days = self.max_days
        cfg.record_frames = self.render_mode is not None
        self.sim = build_simulation(cfg)
        if self.per_tissue:
            self.sim.grow_fn = self._apply_local  # per-node control replaces policy
            self._pending = {}
        self._prev_biomass = self.sim.history["biomass"][-1]
        self._prev_seeds = self.sim.seeds
        return self._observation(), self._info()

    def _advance_reward(self):
        """Reward after a sim step: biomass gained + seeds set, minus a death
        penalty; terminates if the plant starved to death."""
        biomass = self.sim.history["biomass"][-1]
        seeds = self.sim.seeds
        reward = ((biomass - self._prev_biomass)
                  + self.repro_value * (seeds - self._prev_seeds) + 0.01)
        self._prev_biomass = biomass
        self._prev_seeds = seeds
        terminated = not self.sim.alive
        if terminated:
            reward -= self.death_penalty
        truncated = self.sim.day >= self.max_days
        return float(reward), terminated, truncated

    def step(self, action: np.ndarray):
        self._apply_action(action)
        self.sim.step()
        reward, terminated, truncated = self._advance_reward()
        return self._observation(), reward, terminated, truncated, self._info()

    # -- per-tissue (distributed) control ----------------------------------
    def _apply_local(self, plant, world) -> None:
        """Growth phase for per-tissue mode: each tissue acts on its own decision.

        ``self._pending`` maps node id -> [grow, thicken] (raw, >0 means do it).
        Meristems with grow>0 try to extend; any node with thicken>0 tries to
        thicken. Both are gated by the tissue's *local* resources -- so this is
        genuine distributed control, no global budget."""
        pol = self.sim.policy  # reuse its atomic growth ops + helpers
        for nid in list(plant.nodes):
            dec = self._pending.get(nid)
            if dec is None:
                continue
            node = plant.nodes[nid]
            if node.meristematic and dec[0] > 0.0:
                pol.extend_tip(plant, world, node, steer_x=float(dec[2]))
            if dec[1] > 0.0:
                pol.thicken_one(node)

    def step_local(self, node_actions: np.ndarray):
        """Advance one day under per-tissue actions (rows aligned to the last
        ``graph_observation()``). Returns (reward, terminated, truncated, info)."""
        self._pending = {nid: node_actions[i]
                         for i, nid in enumerate(self._graph_node_ids)}
        self.sim.step()
        reward, terminated, truncated = self._advance_reward()
        return reward, terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None:
            return None
        from phytonet.visualize import render_frame
        ax = render_frame(self.sim.world, self.sim.plant,
                          title=f"PhytoNet — day {self.sim.day}")
        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            plt.show()
            return None
        fig = ax.figure
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buf.reshape(h, w, 4)[..., :3].copy()

    def close(self):
        import matplotlib.pyplot as plt
        plt.close("all")
