"""Tests for submission generation with offense/defense parameters."""

import numpy as np
from scipy.stats import norm


def test_batch_win_probs_offdef():
    """_batch_win_probs should accept off/def samples and use sqrt(2)."""
    from src.submission import _batch_win_probs

    rng = np.random.default_rng(42)
    n_samples, n_teams = 100, 4
    off = rng.normal(0, 3, size=(n_samples, n_teams))
    deff = rng.normal(0, 3, size=(n_samples, n_teams))
    sigma = np.full(n_samples, 11.0)

    id_low = np.array([1, 2])
    id_high = np.array([3, 4])
    id_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}

    probs = _batch_win_probs(
        id_low, id_high, sigma, id_to_idx,
        off_samples=off, def_samples=deff,
    )

    assert probs.shape == (2,)
    assert np.all((probs >= 0) & (probs <= 1))

    for i in range(2):
        il, ih = id_to_idx[id_low[i]], id_to_idx[id_high[i]]
        diff = (off[:, il] - deff[:, ih]) - (off[:, ih] - deff[:, il])
        expected = norm.cdf(diff / (sigma * np.sqrt(2))).mean()
        assert abs(probs[i] - expected) < 1e-10


def test_batch_win_probs_theta():
    """Backward compat: theta-based batch probs should still work."""
    from src.submission import _batch_win_probs

    rng = np.random.default_rng(42)
    n_samples, n_teams = 100, 4
    theta = rng.normal(0, 5, size=(n_samples, n_teams))
    sigma = np.full(n_samples, 11.0)

    id_low = np.array([1])
    id_high = np.array([3])
    id_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}

    probs = _batch_win_probs(
        id_low, id_high, sigma, id_to_idx,
        theta_samples=theta,
    )

    diff = theta[:, 0] - theta[:, 2]
    expected = norm.cdf(diff / sigma).mean()
    assert abs(probs[0] - expected) < 1e-10
