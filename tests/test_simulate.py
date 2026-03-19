"""Tests for simulation engine with offense/defense parameters."""

import numpy as np
import inspect
from scipy.stats import norm


def test_play_game_offdef():
    """_play_game should accept off/def arrays and use sqrt(2) denominator."""
    from src.simulate import _play_game

    rng = np.random.default_rng(42)
    off = np.array([5.0, 3.0, 1.0, -1.0])
    deff = np.array([3.0, 2.0, 4.0, 0.0])
    sigma_val = 11.0
    team_id_to_idx = {101: 0, 102: 1, 103: 2, 104: 3}

    team_a = {"team_id": 101, "team_name": "A", "seed_num": 1}
    team_b = {"team_id": 102, "team_name": "B", "seed_num": 2}

    wins_a = 0
    n_trials = 5000
    for _ in range(n_trials):
        winner = _play_game(
            team_a, team_b, sigma_val=sigma_val,
            team_id_to_idx=team_id_to_idx, rng=rng,
            off=off, deff=deff,
        )
        if winner["team_id"] == 101:
            wins_a += 1

    # diff = (5-2) - (3-3) = 3, p = Phi(3 / (11*sqrt(2))) ~ 0.585
    expected_p = norm.cdf(3 / (11 * np.sqrt(2)))
    observed_p = wins_a / n_trials
    assert abs(observed_p - expected_p) < 0.03, (
        f"Expected ~{expected_p:.3f}, got {observed_p:.3f}"
    )


def test_play_game_theta_still_works():
    """Backward compat: theta-based _play_game should still work."""
    from src.simulate import _play_game

    rng = np.random.default_rng(42)
    theta = np.array([5.0, 0.0])
    sigma_val = 11.0
    team_id_to_idx = {101: 0, 102: 1}

    team_a = {"team_id": 101, "team_name": "A", "seed_num": 1}
    team_b = {"team_id": 102, "team_name": "B", "seed_num": 2}

    wins_a = 0
    n_trials = 5000
    for _ in range(n_trials):
        winner = _play_game(
            team_a, team_b, sigma_val=sigma_val,
            team_id_to_idx=team_id_to_idx, rng=rng,
            theta=theta,
        )
        if winner["team_id"] == 101:
            wins_a += 1

    expected_p = norm.cdf(5.0 / 11.0)
    observed_p = wins_a / n_trials
    assert abs(observed_p - expected_p) < 0.03


def test_simulate_tournament_accepts_offdef():
    """simulate_tournament should accept off_samples and def_samples."""
    from src.simulate import simulate_tournament
    sig = inspect.signature(simulate_tournament)
    params = list(sig.parameters.keys())
    assert "off_samples" in params
    assert "def_samples" in params


def test_simulate_tournament_single_locks_in_play_in(tmp_path):
    """Locked play-in result should always produce the specified winner."""
    from src.simulate import simulate_tournament_single

    bracket_struct = {
        "seed_to_team": {
            "X16a": {"team_id": 1250, "team_name": "UMBC", "seed_num": 16},
            "X16b": {"team_id": 1341, "team_name": "Howard", "seed_num": 16},
            "X01": {"team_id": 1196, "team_name": "Florida", "seed_num": 1},
        },
        "play_in_slots": {"X16": ("X16a", "X16b")},
        "regular_slots": {"R1X1": ("X01", "X16")},
        "gender": "M",
    }

    rng = np.random.default_rng(42)
    team_id_to_idx = {1250: 0, 1341: 1, 1196: 2}
    off = np.array([0.0, 0.0, 5.0])
    deff = np.array([0.0, 0.0, -3.0])

    actual_results = {
        "X16": {"winner": 1250, "loser": 1341, "winner_score": 72, "loser_score": 65},
    }

    result = simulate_tournament_single(
        bracket_struct, sigma_val=11.0, team_id_to_idx=team_id_to_idx,
        rng=rng, off=off, deff=deff,
        actual_results=actual_results,
    )

    # Play-in winner locked in
    assert result["slot_winners"]["X16"]["team_id"] == 1250
    # R1 should still be simulated (Florida vs UMBC)
    assert "R1X1" in result["slot_winners"]


def test_simulate_tournament_single_locks_in_regular_slot():
    """Locked R1 result should always produce the specified winner."""
    from src.simulate import simulate_tournament_single

    bracket_struct = {
        "seed_to_team": {
            "W01": {"team_id": 1181, "team_name": "Duke", "seed_num": 1},
            "W16": {"team_id": 1373, "team_name": "Norfolk St", "seed_num": 16},
        },
        "play_in_slots": {},
        "regular_slots": {"R1W1": ("W01", "W16")},
        "gender": "M",
    }

    rng = np.random.default_rng(42)
    team_id_to_idx = {1181: 0, 1373: 1}
    off = np.array([0.0, 0.0])
    deff = np.array([0.0, 0.0])

    # Lock in Norfolk St as the upset winner
    actual_results = {
        "R1W1": {"winner": 1373, "loser": 1181, "winner_score": 70, "loser_score": 65},
    }

    result = simulate_tournament_single(
        bracket_struct, sigma_val=11.0, team_id_to_idx=team_id_to_idx,
        rng=rng, off=off, deff=deff,
        actual_results=actual_results,
    )

    assert result["slot_winners"]["R1W1"]["team_id"] == 1373


def test_simulate_tournament_locked_results_give_deterministic_probs():
    """With a locked R1 result, loser gets 0.0 advancement past that round."""
    from src.simulate import simulate_tournament
    import pandas as pd

    bracket_struct = {
        "seed_to_team": {
            "W01": {"team_id": 1181, "team_name": "Duke", "seed_num": 1},
            "W16": {"team_id": 1373, "team_name": "Norfolk St", "seed_num": 16},
        },
        "play_in_slots": {},
        "regular_slots": {"R1W1": ("W01", "W16")},
        "seeds_df": pd.DataFrame({
            "TeamID": [1181, 1373],
            "TeamName": ["Duke", "Norfolk St"],
            "Seed": ["W01", "W16"],
            "SeedNum": [1, 16],
            "Region": ["W", "W"],
        }),
        "gender": "M",
    }

    n_sims = 100
    team_ids = np.array([1181, 1373])
    off = np.random.default_rng(0).normal(size=(200, 2))
    deff = np.random.default_rng(1).normal(size=(200, 2))
    sigma = np.full(200, 11.0)

    actual_results = {
        "R1W1": {"winner": 1181, "loser": 1373, "winner_score": 85, "loser_score": 60},
    }

    sim = simulate_tournament(
        bracket_struct, sigma_samples=sigma, team_ids=team_ids,
        n_sims=n_sims, off_samples=off, def_samples=deff,
        actual_results=actual_results,
    )

    adv = sim["advancement"]
    duke = adv[adv["TeamID"] == 1181].iloc[0]
    norfolk = adv[adv["TeamID"] == 1373].iloc[0]

    # Duke won R1 in every sim
    assert duke["Round of 32"] == 1.0
    # Norfolk St lost R1 in every sim
    assert norfolk["Round of 32"] == 0.0
