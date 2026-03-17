"""Tests for model specification."""

import numpy as np
import pymc as pm
import xarray as xr
import arviz as az
import pytest


def _make_test_data():
    """Minimal model data dict for 4 teams, 2 conferences, 6 games."""
    return {
        "team_names": np.array(["Alpha", "Bravo", "Charlie", "Delta"]),
        "team_ids": np.array([1101, 1102, 1103, 1104]),
        "conf_names": np.array(["big_ten", "sec"]),
        "team_idx_i": np.array([0, 1, 0, 2, 1, 0]),
        "team_idx_j": np.array([1, 2, 2, 3, 3, 3]),
        "margin": np.array([15.0, 10.0, 5.0, 10.0, 10.0, -4.0]),
        "score_i": np.array([75.0, 80.0, 70.0, 65.0, 90.0, 68.0]),
        "score_j": np.array([60.0, 70.0, 65.0, 55.0, 80.0, 72.0]),
        "home_i": np.array([1.0, -1.0, 0.0, 1.0, 0.0, 1.0]),
        "conf_idx": np.array([0, 0, 1, 1]),
        "n_teams": 4,
        "n_conferences": 2,
        "n_games": 6,
    }


# --- Task 2: build_offense_defense_model ---

def test_offense_defense_model_builds():
    """The offense/defense model should compile without error."""
    from src.model import build_offense_defense_model
    data = _make_test_data()
    model = build_offense_defense_model(data, likelihood="normal")
    assert isinstance(model, pm.Model)


def test_offense_defense_model_has_expected_variables():
    """Model should contain mu_intercept, alpha, sigma, and observed scores."""
    from src.model import build_offense_defense_model
    data = _make_test_data()
    model = build_offense_defense_model(data, likelihood="normal")

    var_names = {v.name for v in model.free_RVs}
    assert "mu_intercept" in var_names
    assert "alpha" in var_names
    assert "sigma" in var_names

    obs_names = {v.name for v in model.observed_RVs}
    assert "score_i" in obs_names
    assert "score_j" in obs_names


def test_offense_defense_model_studentt():
    """Student-t variant should include nu parameter."""
    from src.model import build_offense_defense_model
    data = _make_test_data()
    model = build_offense_defense_model(data, likelihood="studentt")

    var_names = {v.name for v in model.free_RVs}
    assert "nu" in var_names


def test_offense_defense_model_logp_finite():
    """Model log-probability should be finite at the initial point."""
    from src.model import build_offense_defense_model
    data = _make_test_data()
    model = build_offense_defense_model(data, likelihood="normal")
    logp = model.point_logps()
    for name, val in logp.items():
        assert np.isfinite(val), f"{name} has non-finite logp: {val}"


def test_offense_defense_model_invalid_likelihood():
    """Invalid likelihood should raise ValueError."""
    from src.model import build_offense_defense_model
    data = _make_test_data()
    with pytest.raises(ValueError, match="Unknown likelihood"):
        build_offense_defense_model(data, likelihood="poisson")


# --- Task 3: check_diagnostics ---

def test_check_diagnostics_accepts_off_def_model():
    """check_diagnostics should work with offense/defense model parameters."""
    from src.model import check_diagnostics

    n_chains, n_draws, n_teams, n_conf = 2, 50, 4, 2
    rng = np.random.default_rng(42)

    posterior = xr.Dataset({
        "off": xr.DataArray(rng.normal(size=(n_chains, n_draws, n_teams)), dims=["chain", "draw", "team"]),
        "def": xr.DataArray(rng.normal(size=(n_chains, n_draws, n_teams)), dims=["chain", "draw", "team"]),
        "mu_intercept": xr.DataArray(rng.normal(70, 1, size=(n_chains, n_draws)), dims=["chain", "draw"]),
        "alpha": xr.DataArray(rng.normal(3.5, 0.5, size=(n_chains, n_draws)), dims=["chain", "draw"]),
        "sigma": xr.DataArray(rng.normal(11, 0.5, size=(n_chains, n_draws)), dims=["chain", "draw"]),
        "sigma_off_conf": xr.DataArray(rng.normal(3, 0.3, size=(n_chains, n_draws)), dims=["chain", "draw"]),
        "sigma_def_conf": xr.DataArray(rng.normal(3, 0.3, size=(n_chains, n_draws)), dims=["chain", "draw"]),
        "mu_off_conf": xr.DataArray(rng.normal(size=(n_chains, n_draws, n_conf)), dims=["chain", "draw", "conference"]),
        "mu_def_conf": xr.DataArray(rng.normal(size=(n_chains, n_draws, n_conf)), dims=["chain", "draw", "conference"]),
        "lkj_corr": xr.DataArray(
            np.tile(np.eye(2), (n_chains, n_draws, 1, 1)),
            dims=["chain", "draw", "lkj_corr_dim_0", "lkj_corr_dim_1"],
        ),
    })
    sample_stats = xr.Dataset({
        "diverging": xr.DataArray(np.zeros((n_chains, n_draws), dtype=bool), dims=["chain", "draw"]),
    })
    idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)

    diag = check_diagnostics(idata)
    assert "pass" in diag
    assert isinstance(diag["pass"], (bool, np.bool_))
    assert "max_rhat" in diag


# --- Task 4: get_team_strengths ---

def test_get_team_strengths_offdef():
    """get_team_strengths should extract off/def and compute overall strength."""
    from src.model import get_team_strengths

    n_chains, n_draws, n_teams = 2, 50, 4
    rng = np.random.default_rng(42)

    off_vals = rng.normal(size=(n_chains, n_draws, n_teams))
    def_vals = rng.normal(size=(n_chains, n_draws, n_teams))

    posterior = xr.Dataset({
        "off": xr.DataArray(off_vals, dims=["chain", "draw", "team"]),
        "def": xr.DataArray(def_vals, dims=["chain", "draw", "team"]),
        "lkj_corr": xr.DataArray(
            np.tile(np.eye(2), (n_chains, n_draws, 1, 1)),
            dims=["chain", "draw", "lkj_corr_dim_0", "lkj_corr_dim_1"],
        ),
    })
    idata = az.InferenceData(posterior=posterior)

    data = {
        "team_names": np.array(["A", "B", "C", "D"]),
        "team_ids": np.array([1, 2, 3, 4]),
    }
    result = get_team_strengths(idata, data)

    assert "off_samples" in result
    assert "def_samples" in result
    assert "overall_means" in result
    assert result["off_samples"].shape == (n_chains * n_draws, n_teams)

    expected_overall = result["off_samples"] + result["def_samples"]
    np.testing.assert_array_almost_equal(
        result["overall_means"], expected_overall.mean(axis=0)
    )


# --- Task 5: win_probability_samples ---

def test_win_probability_offdef():
    """win_probability_samples should accept off/def and use sqrt(2) denominator."""
    from src.model import win_probability_samples
    from scipy.stats import norm

    rng = np.random.default_rng(42)
    n_samples, n_teams = 100, 4
    off = rng.normal(5, 2, size=(n_samples, n_teams))
    deff = rng.normal(3, 2, size=(n_samples, n_teams))
    sigma = np.full(n_samples, 11.0)

    p = win_probability_samples(
        team_i_idx=0, team_j_idx=1,
        sigma_samples=sigma,
        off_samples=off, def_samples=deff,
    )

    assert p.shape == (n_samples,)
    assert np.all((p >= 0) & (p <= 1))

    # Verify sqrt(2) denominator
    diff = (off[:, 0] - deff[:, 1]) - (off[:, 1] - deff[:, 0])
    expected = norm.cdf(diff / (sigma * np.sqrt(2)))
    np.testing.assert_array_almost_equal(p, expected)


def test_win_probability_theta_still_works():
    """Backward compat: theta-based win probability should still work."""
    from src.model import win_probability_samples
    from scipy.stats import norm

    rng = np.random.default_rng(42)
    n_samples, n_teams = 100, 4
    theta = rng.normal(0, 5, size=(n_samples, n_teams))
    sigma = np.full(n_samples, 11.0)

    p = win_probability_samples(
        team_i_idx=0, team_j_idx=1,
        sigma_samples=sigma,
        theta_samples=theta,
    )

    diff = theta[:, 0] - theta[:, 1]
    expected = norm.cdf(diff / sigma)
    np.testing.assert_array_almost_equal(p, expected)


def test_win_probability_requires_samples():
    """Should raise when neither theta nor off/def provided."""
    from src.model import win_probability_samples

    sigma = np.ones(10)
    with pytest.raises(ValueError, match="Provide either"):
        win_probability_samples(team_i_idx=0, team_j_idx=1, sigma_samples=sigma)
