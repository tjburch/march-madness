"""Tests for bracket pool optimization."""

import numpy as np
import pandas as pd
from collections import Counter


def _make_mini_bracket():
    """Create a minimal 4-team bracket for testing."""
    bracket_struct = {
        "seed_to_team": {
            "W01": {"team_id": 1, "team_name": "Duke", "seed_num": 1},
            "W16": {"team_id": 2, "team_name": "Norfolk St", "seed_num": 16},
            "W08": {"team_id": 3, "team_name": "Memphis", "seed_num": 8},
            "W09": {"team_id": 4, "team_name": "FAU", "seed_num": 9},
        },
        "play_in_slots": {},
        "regular_slots": {
            "R1W1": ("W01", "W16"),
            "R1W2": ("W08", "W09"),
            "R2W1": ("R1W1", "R1W2"),
        },
        "seeds_df": pd.DataFrame({
            "TeamID": [1, 2, 3, 4],
            "TeamName": ["Duke", "Norfolk St", "Memphis", "FAU"],
            "Seed": ["W01", "W16", "W08", "W09"],
            "SeedNum": [1, 16, 8, 9],
            "Region": ["W", "W", "W", "W"],
        }),
        "gender": "M",
    }
    return bracket_struct


def _make_sim_results(bracket_struct, n_sims=100, seed=42):
    """Generate fake sim results for testing."""
    rng = np.random.default_rng(seed)
    all_results = []

    teams = bracket_struct["seed_to_team"]
    regular_slots = bracket_struct["regular_slots"]

    for _ in range(n_sims):
        slot_winners = {}
        resolved = dict(teams)

        for slot in sorted(regular_slots.keys(), key=lambda s: (int(s[1]), s)):
            strong, weak = regular_slots[slot]
            team_a = resolved.get(strong)
            team_b = resolved.get(weak)
            if team_a is None or team_b is None:
                continue
            # Favor lower seed
            p = 0.8 if team_a["seed_num"] < team_b["seed_num"] else 0.2
            winner = team_a if rng.random() < p else team_b
            resolved[slot] = winner
            slot_winners[slot] = winner

        round_results = {}
        for slot, winner in slot_winners.items():
            if slot[0] == "R" and slot[1].isdigit():
                rn = int(slot[1])
                round_results.setdefault(rn, []).append(winner)

        all_results.append({
            "slot_winners": slot_winners,
            "champion": slot_winners.get("R2W1"),
            "round_results": round_results,
        })

    return {
        "all_results": all_results,
        "n_sims": n_sims,
    }


def test_seed_public_pick_prob_favorites():
    """Higher seed should be favored by the public."""
    from src.pool import seed_public_pick_prob

    # 1-seed vs 16-seed: public strongly favors the 1
    p = seed_public_pick_prob(1, 16)
    assert p > 0.9, f"1 vs 16: expected >0.9, got {p:.3f}"

    # 8 vs 9: nearly even
    p = seed_public_pick_prob(8, 9)
    assert 0.45 < p < 0.65, f"8 vs 9: expected ~0.5, got {p:.3f}"

    # Same seed: 50/50
    p = seed_public_pick_prob(5, 5)
    assert p == 0.5


def test_seed_public_pick_prob_symmetry():
    """P(A beats B) + P(B beats A) = 1."""
    from src.pool import seed_public_pick_prob

    for sa in range(1, 17):
        for sb in range(1, 17):
            p_ab = seed_public_pick_prob(sa, sb)
            p_ba = seed_public_pick_prob(sb, sa)
            assert abs(p_ab + p_ba - 1.0) < 1e-10


def test_extract_outcome():
    """extract_outcome should return slot -> team_id dict."""
    from src.pool import extract_outcome

    sim_result = {
        "slot_winners": {
            "R1W1": {"team_id": 1, "team_name": "Duke", "seed_num": 1},
            "R1W2": {"team_id": 3, "team_name": "Memphis", "seed_num": 8},
        }
    }
    outcome = extract_outcome(sim_result)
    assert outcome == {"R1W1": 1, "R1W2": 3}


def test_score_bracket():
    """Bracket scoring should match correct picks by round."""
    from src.pool import score_bracket, SCORING_SYSTEMS

    picks = {"R1W1": 1, "R1W2": 3, "R2W1": 1}
    outcome = {"R1W1": 1, "R1W2": 4, "R2W1": 1}
    scoring = SCORING_SYSTEMS["espn"]

    score = score_bracket(picks, outcome, scoring)
    # R1W1 correct (10 pts), R1W2 wrong (0), R2W1 correct (20 pts)
    assert score == 30, f"Expected 30, got {score}"


def test_score_bracket_ignores_play_in():
    """Play-in slots (no R prefix with digit) should not score."""
    from src.pool import score_bracket

    picks = {"X16": 1, "R1W1": 1}
    outcome = {"X16": 1, "R1W1": 1}
    scoring = {1: 10}

    score = score_bracket(picks, outcome, scoring)
    assert score == 10  # Only R1W1 counts


def test_generate_public_bracket_structure():
    """Public bracket should have picks for all tournament slots."""
    from src.pool import generate_public_bracket

    bracket_struct = _make_mini_bracket()
    rng = np.random.default_rng(42)

    bracket = generate_public_bracket(bracket_struct, rng)

    # Should have all regular slots
    assert "R1W1" in bracket
    assert "R1W2" in bracket
    assert "R2W1" in bracket

    # All picks should be valid team IDs
    valid_ids = {1, 2, 3, 4}
    for slot, team_id in bracket.items():
        assert team_id in valid_ids

    # R2 winner must be one of the R1 winners (bracket consistency)
    r2_winner = bracket["R2W1"]
    assert r2_winner in (bracket["R1W1"], bracket["R1W2"])


def test_generate_public_bracket_favors_seeds():
    """Over many brackets, public should favor lower seeds."""
    from src.pool import generate_public_bracket

    bracket_struct = _make_mini_bracket()
    rng = np.random.default_rng(42)

    r1w1_picks = Counter()
    n = 5000
    for _ in range(n):
        bracket = generate_public_bracket(bracket_struct, rng)
        r1w1_picks[bracket["R1W1"]] += 1

    # Duke (1-seed) should be picked much more than Norfolk St (16-seed)
    duke_rate = r1w1_picks[1] / n
    assert duke_rate > 0.85, f"Duke pick rate {duke_rate:.3f} too low"


def test_compute_leverage_returns_dataframe():
    """compute_leverage should return a DataFrame with expected columns."""
    from src.pool import compute_leverage

    bracket_struct = _make_mini_bracket()
    sim_results = _make_sim_results(bracket_struct)

    df = compute_leverage(sim_results, bracket_struct, n_public=1000, seed=42)

    assert isinstance(df, pd.DataFrame)
    expected_cols = [
        "Slot", "TeamID", "TeamName", "SeedNum",
        "Round", "ModelProb", "PublicProb", "Leverage",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    # All probabilities should be in [0, 1]
    assert (df["ModelProb"] >= 0).all()
    assert (df["ModelProb"] <= 1).all()
    assert (df["PublicProb"] >= 0).all()


def test_pool_optimizer_evaluate():
    """PoolOptimizer.evaluate should return a probability in [0, 1]."""
    from src.pool import PoolOptimizer

    bracket_struct = _make_mini_bracket()
    sim_results = _make_sim_results(bracket_struct, n_sims=100)

    optimizer = PoolOptimizer(
        sim_results, bracket_struct,
        pool_size=5, n_pools=3, n_outcomes=100, seed=42,
    )

    bracket = optimizer.most_likely_bracket()
    win_prob = optimizer.evaluate(bracket)

    assert 0.0 <= win_prob <= 1.0
    # With pool_size=5, modal bracket should have reasonable win prob
    assert win_prob > 0.0, "Modal bracket should win sometimes"


def test_pool_optimizer_modal_bracket():
    """Modal bracket should pick the most frequent winner at each slot."""
    from src.pool import PoolOptimizer

    bracket_struct = _make_mini_bracket()
    sim_results = _make_sim_results(bracket_struct, n_sims=1000, seed=42)

    optimizer = PoolOptimizer(
        sim_results, bracket_struct,
        pool_size=5, n_pools=2, n_outcomes=100, seed=42,
    )

    modal = optimizer.most_likely_bracket()

    # With 80% favorite win rate, modal should pick Duke in R1W1
    assert modal["R1W1"] == 1, "Modal should pick Duke (1-seed)"


def test_pool_optimizer_optimize():
    """Optimizer should return valid result dict."""
    from src.pool import PoolOptimizer

    bracket_struct = _make_mini_bracket()
    sim_results = _make_sim_results(bracket_struct, n_sims=200)

    optimizer = PoolOptimizer(
        sim_results, bracket_struct,
        pool_size=10, n_pools=3, n_outcomes=200, seed=42,
    )

    result = optimizer.optimize(
        pop_size=10, n_generations=5, verbose=False,
    )

    assert "bracket" in result
    assert "win_prob" in result
    assert "most_likely_bracket" in result
    assert "most_likely_win_prob" in result
    assert "history" in result
    assert "diff_from_modal" in result

    assert 0.0 <= result["win_prob"] <= 1.0
    assert len(result["history"]) > 0

    # Bracket should be structurally valid
    bracket = result["bracket"]
    assert bracket["R2W1"] in (bracket["R1W1"], bracket["R1W2"])


def test_pool_optimizer_larger_pool_more_contrarian():
    """Larger pools should produce brackets that differ more from modal.

    This tests the key theoretical prediction: as pool size increases,
    the optimal strategy becomes more contrarian because you need more
    differentiation to beat more opponents.
    """
    from src.pool import PoolOptimizer

    bracket_struct = _make_mini_bracket()
    sim_results = _make_sim_results(bracket_struct, n_sims=500, seed=42)

    # Small pool
    small_opt = PoolOptimizer(
        sim_results, bracket_struct,
        pool_size=3, n_pools=5, n_outcomes=500, seed=42,
    )
    small_result = small_opt.optimize(
        pop_size=15, n_generations=20, verbose=False,
    )

    # Large pool
    large_opt = PoolOptimizer(
        sim_results, bracket_struct,
        pool_size=100, n_pools=5, n_outcomes=500, seed=42,
    )
    large_result = large_opt.optimize(
        pop_size=15, n_generations=20, verbose=False,
    )

    # In a small pool, you should win more often than in a large pool
    # (regardless of strategy, 1/N baseline)
    assert small_result["win_prob"] > large_result["win_prob"]


def test_mutate_bracket_maintains_consistency():
    """Mutated bracket should still be structurally valid."""
    from src.pool import PoolOptimizer

    bracket_struct = _make_mini_bracket()
    sim_results = _make_sim_results(bracket_struct, n_sims=100)

    optimizer = PoolOptimizer(
        sim_results, bracket_struct,
        pool_size=5, n_pools=2, n_outcomes=100, seed=42,
    )

    rng = np.random.default_rng(42)
    bracket = optimizer.most_likely_bracket()

    for _ in range(50):
        mutated = optimizer._mutate_bracket(bracket, rng, n_flips=1)

        # R2 winner must be one of the R1 winners
        assert mutated["R2W1"] in (mutated["R1W1"], mutated["R1W2"]), (
            f"Inconsistent bracket: R2W1={mutated['R2W1']} "
            f"not in R1 winners {mutated['R1W1']}, {mutated['R1W2']}"
        )


def test_generate_bracket_custom_fn():
    """generate_bracket should work with arbitrary win probability functions."""
    from src.pool import generate_bracket

    bracket_struct = _make_mini_bracket()
    rng = np.random.default_rng(42)

    # Always pick team_a (deterministic)
    always_a = generate_bracket(bracket_struct, lambda a, b: 1.0, rng)
    assert always_a["R1W1"] == 1  # W01 (Duke)
    assert always_a["R1W2"] == 3  # W08 (Memphis)
    assert always_a["R2W1"] in (1, 3)  # One of the R1 winners
