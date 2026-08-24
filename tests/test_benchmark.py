"""Tests for the benchmark harness and the message-ablation switch."""

import numpy as np
import pytest

from phytonet.benchmark import FIXED_CONTROLLERS, benchmark_table, eval_policy, format_table


def test_eval_policy_returns_one_value_per_seed():
    r = eval_policy("baseline", FIXED_CONTROLLERS["balanced"], seeds=range(3), days=20)
    assert r.shape == (3,)
    assert np.all(np.isfinite(r))


def test_benchmark_table_has_all_cells():
    scenarios = ["baseline", "drought"]
    table = benchmark_table(scenarios, seeds=range(3), days=20)
    assert set(table) == set(FIXED_CONTROLLERS)
    for row in table.values():
        for sc in scenarios:
            mean, std = row[sc]
            assert np.isfinite(mean) and std >= 0.0
    assert isinstance(format_table(table, scenarios), str)


def test_baseline_has_strategy_headroom():
    """In baseline, balanced should clearly beat root-heavy -- there's a real
    spread (unlike a flat-plateau scenario)."""
    table = benchmark_table(["baseline"], seeds=range(6), days=80)
    assert table["balanced"]["baseline"][0] > table["root-heavy"]["baseline"][0] + 2.0


def test_message_ablation_changes_dependence_on_edges():
    torch = pytest.importorskip("torch")
    from phytonet.gnn import GNNNodeActorCritic

    feats = torch.randn(5, 12)
    edge_a = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_b = torch.tensor([[0, 4], [4, 0]])

    on = GNNNodeActorCritic(12, 2, hidden=16, rounds=2, use_messages=True)
    off = GNNNodeActorCritic(12, 2, hidden=16, rounds=2, use_messages=False)

    # With messages, embeddings depend on the edges; without, they do not.
    assert not torch.allclose(on.embed_nodes(feats, edge_a),
                              on.embed_nodes(feats, edge_b))
    assert torch.allclose(off.embed_nodes(feats, edge_a),
                          off.embed_nodes(feats, edge_b))


# -- cross-paradigm outcome metric ----------------------------------------
def test_outcome_evals_return_one_value_per_seed():
    import numpy as np
    from phytonet.benchmark import eval_global_outcome, eval_signaling_outcome
    from phytonet.signaling import SignalingParams
    bal = eval_global_outcome(lambda o: np.zeros(3, np.float32),
                              "baseline", seeds=range(2), days=20)
    sig = eval_signaling_outcome(SignalingParams(), "baseline", seeds=range(2), days=20)
    assert bal.shape == (2,) and sig.shape == (2,)
    assert np.all(np.isfinite(bal)) and np.all(np.isfinite(sig))
    assert np.all(bal > 0) and np.all(sig > 0)
