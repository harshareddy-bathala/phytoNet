"""Tests that the simulator reproduces known plant biology (validation)."""

import numpy as np

from phytonet.plant import NodeKind, Plant
from phytonet.validation import (
    functional_equilibrium, morphometrics, reproduces_functional_equilibrium,
    root_shoot_ratio,
)


def test_root_shoot_ratio_on_known_plant():
    p = Plant(dim=2)
    r = p.add_node(NodeKind.ROOT, [0.0, -1.0]); r.state["size"] = 2.0
    s = p.add_node(NodeKind.STEM, [0.0, 1.0]); s.state["size"] = 1.0
    l = p.add_node(NodeKind.LEAF, [0.3, 1.3]); l.state["size"] = 1.0
    assert abs(root_shoot_ratio(p) - 1.0) < 1e-9   # 2 root / (1+1) shoot


def test_functional_equilibrium_ordering():
    """The model should put more biomass into roots under water limitation
    (drought) and less under light limitation (shade) -- emergent, not coded."""
    fe = functional_equilibrium(seeds=range(3), days=60)
    assert fe["drought"][0] > fe["baseline"][0] > fe["shade"][0]
    assert reproduces_functional_equilibrium(seeds=range(3), days=60)


def test_morphometrics_fields():
    m = morphometrics("baseline", seed=0, days=40)
    assert set(m) == {"root_shoot", "height", "n_leaves", "n_roots", "biomass"}
    assert np.isfinite(m["root_shoot"]) and m["biomass"] > 0
