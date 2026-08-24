"""The world the plant lives in: bounds, a ground plane, light, and soil water.

Everything here is dimension-agnostic. The world has ``dim`` spatial axes and a
single "up" axis (the last one): in 2D up is ``y`` (index 1), in 3D up is ``z``
(index 2). Light comes from above the ground; water lives below it.

For the MVP, ``light_at`` and ``water_at`` are deliberately trivial (uniform
fields with a hard ground cutoff). They exist as *methods* so that later steps
can swap in real fields -- gradients, shading, drying soil -- without any caller
needing to change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class World:
    """A simple environment with a light source above ground and water below it.

    Parameters
    ----------
    dim:
        Number of spatial dimensions (2 for the MVP, 3 later).
    bounds_low, bounds_high:
        Corner coordinates of the world box, each a length-``dim`` vector. If
        left as ``None`` a sensible default box is built around the origin.
    ground_level:
        Position along the up-axis that separates air (>=) from soil (<).
    light_intensity:
        Light available to tissue in the air, in arbitrary energy units / step.
    water_availability:
        Water available to tissue in the soil, in arbitrary units / step.
    light_pos:
        Position of the sun. Used for the *directional* light cue (phototropism);
        the *energy* available (``light_at``) stays a uniform ambient field for
        now. Defaults to high above the centre of the box.
    """

    dim: int = 2
    bounds_low: np.ndarray | None = None
    bounds_high: np.ndarray | None = None
    ground_level: float = 0.0
    light_intensity: float = 1.0
    water_availability: float = 1.0
    light_pos: np.ndarray | None = None
    # Soil-water reservoir. ``water_capacity=None`` means an *infinite* reservoir
    # (constant availability -- the original back-compatible behaviour). Set a
    # finite capacity to model a soil that dries as roots drink and refills when
    # it rains / is watered; then ``soil_water`` tracks the current amount.
    water_capacity: float | None = None
    soil_water: float | None = None
    # Spatial heterogeneity (along lateral axis 0 = x). Defaults reproduce the
    # old uniform fields. With these set, *where* a tissue sits matters -- light
    # is brighter on one side, water concentrated on another -- so the best
    # action differs tissue-to-tissue (what a per-tissue / GNN policy can use and
    # a single global policy cannot).
    light_gradient: float = 0.0          # d(light)/dx as a fraction; +ve = +x brighter
    water_source_x: float | None = None  # x where soil water concentrates
    water_width: float = 3.0             # spread of that water patch along x

    def __post_init__(self) -> None:
        if self.dim < 2:
            raise ValueError(f"dim must be >= 2, got {self.dim}")
        # Default box: 10 wide on lateral axes, from -5 (underground) to +10 (up).
        if self.bounds_low is None:
            low = np.full(self.dim, -5.0)
            self.bounds_low = low
        if self.bounds_high is None:
            high = np.full(self.dim, 5.0)
            high[self.up_axis] = 10.0
            self.bounds_high = high
        self.bounds_low = np.asarray(self.bounds_low, dtype=float)
        self.bounds_high = np.asarray(self.bounds_high, dtype=float)
        if self.bounds_low.shape != (self.dim,) or self.bounds_high.shape != (self.dim,):
            raise ValueError("bounds must each be a length-dim vector")
        if self.light_pos is None:
            # Centre laterally, near the top of the box.
            sun = (self.bounds_low + self.bounds_high) / 2.0
            sun[self.up_axis] = 0.9 * self.bounds_high[self.up_axis]
            self.light_pos = sun
        if self.water_capacity is not None and self.soil_water is None:
            self.soil_water = self.water_capacity  # start with full soil
        self.light_pos = np.asarray(self.light_pos, dtype=float)
        if self.light_pos.shape != (self.dim,):
            raise ValueError("light_pos must be a length-dim vector")

    @property
    def up_axis(self) -> int:
        """Index of the vertical axis (last coordinate)."""
        return self.dim - 1

    def up_vector(self) -> np.ndarray:
        """Unit vector pointing 'up' (against gravity)."""
        v = np.zeros(self.dim)
        v[self.up_axis] = 1.0
        return v

    def light_direction(self, pos: np.ndarray) -> np.ndarray:
        """Unit vector from ``pos`` toward the sun (the phototropic cue)."""
        d = self.light_pos - np.asarray(pos, dtype=float)
        n = float(np.linalg.norm(d))
        return d / n if n > 0 else self.up_vector()

    def height_of(self, pos: np.ndarray) -> float:
        """Signed height above the ground plane (negative = underground)."""
        return float(pos[self.up_axis] - self.ground_level)

    def is_above_ground(self, pos: np.ndarray) -> bool:
        return self.height_of(pos) >= 0.0

    def light_at(self, pos: np.ndarray) -> float:
        """Light available at a position. MVP: uniform above ground, zero below.

        With ``light_gradient`` set, light varies across x so one side is
        brighter. ``gradient = 0`` reproduces the uniform field.
        """
        if not self.is_above_ground(pos):
            return 0.0
        factor = max(0.0, 1.0 + self.light_gradient * float(pos[0]))
        return self.light_intensity * factor

    def moisture(self) -> float:
        """Soil-water fraction in [0, 1]. Always 1.0 for an infinite reservoir."""
        if self.water_capacity is None:
            return 1.0
        return max(0.0, min(1.0, self.soil_water / self.water_capacity))

    def water_at(self, pos: np.ndarray) -> float:
        """Water available to a root at ``pos``.

        Below ground it is the richness (``water_availability``) scaled by how
        wet the soil currently is, so a drying soil starves the roots.
        """
        if self.is_above_ground(pos):
            return 0.0
        factor = 1.0
        if self.water_source_x is not None:
            d = float(pos[0]) - self.water_source_x
            factor = float(np.exp(-(d * d) / (2.0 * self.water_width ** 2)))
        return self.water_availability * self.moisture() * factor

    def add_water(self, amount: float) -> None:
        """Rain / watering: refill the reservoir up to capacity (no-op if infinite)."""
        if self.water_capacity is not None:
            self.soil_water = min(self.water_capacity, self.soil_water + amount)

    def evaporate(self, amount: float) -> None:
        """Lose soil water to evaporation/drainage (no-op if infinite)."""
        if self.water_capacity is not None:
            self.soil_water = max(0.0, self.soil_water - amount)

    def deplete(self, amount: float) -> None:
        """Remove water taken up by roots (no-op if infinite reservoir)."""
        if self.water_capacity is not None:
            self.soil_water = max(0.0, self.soil_water - amount)
