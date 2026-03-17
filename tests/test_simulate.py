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
