"""Bayesian Bradley-Terry model for NCAA basketball."""

import numpy as np
import pymc as pm
import arviz as az


def build_bradley_terry(data: dict, non_centered: bool = True) -> pm.Model:
    """Build hierarchical Bayesian Bradley-Terry model.

    Model:
        margin_ij ~ Normal(theta_i - theta_j + alpha * home_i, sigma)
        theta_i ~ Normal(mu_conf[c_i], sigma_team)
        mu_conf ~ Normal(0, sigma_conf)

    Args:
        data: Output of build_model_data()
        non_centered: Use non-centered parameterization (recommended)
    """
    coords = {
        "team": data["team_names"],
        "conference": data["conf_names"],
        "game": np.arange(data["n_games"]),
    }

    with pm.Model(coords=coords) as model:
        # Data
        team_i = pm.Data("team_i", data["team_idx_i"], dims="game")
        team_j = pm.Data("team_j", data["team_idx_j"], dims="game")
        home = pm.Data("home", data["home_i"], dims="game")
        conf_of_team = pm.Data("conf_of_team", data["conf_idx"])

        # Hyperpriors
        sigma_conf = pm.HalfNormal("sigma_conf", sigma=5.0)
        sigma_team = pm.HalfNormal("sigma_team", sigma=5.0)

        # Conference-level effects
        mu_conf = pm.Normal("mu_conf", mu=0, sigma=sigma_conf, dims="conference")

        # Team strengths (hierarchical by conference)
        if non_centered:
            theta_offset = pm.Normal("theta_offset", 0, 1, dims="team")
            theta = pm.Deterministic(
                "theta", mu_conf[conf_of_team] + sigma_team * theta_offset, dims="team"
            )
        else:
            theta = pm.Normal(
                "theta", mu=mu_conf[conf_of_team], sigma=sigma_team, dims="team"
            )

        # Home court advantage
        alpha = pm.Normal("alpha", mu=3.5, sigma=2.0)

        # Game-level noise
        sigma = pm.HalfNormal("sigma", sigma=15.0)

        # Likelihood: score margin
        mu_margin = theta[team_i] - theta[team_j] + alpha * home
        margin = pm.Normal(
            "margin", mu=mu_margin, sigma=sigma, observed=data["margin"], dims="game"
        )

    return model


def fit_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> az.InferenceData:
    """Sample from the model using nutpie."""
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler="nutpie",
            target_accept=target_accept,
            random_seed=random_seed,
        )
    return idata


def check_diagnostics(idata: az.InferenceData) -> dict:
    """Run standard convergence diagnostics."""
    n_div = idata.sample_stats["diverging"].sum().item()

    summary = az.summary(
        idata,
        var_names=["alpha", "sigma", "sigma_conf", "sigma_team", "mu_conf"],
    )

    theta_summary = az.summary(idata, var_names=["theta"])
    max_rhat = float(theta_summary["r_hat"].max())
    min_ess_bulk = float(theta_summary["ess_bulk"].min())
    min_ess_tail = float(theta_summary["ess_tail"].min())

    diagnostics = {
        "divergences": n_div,
        "max_rhat_theta": max_rhat,
        "min_ess_bulk_theta": min_ess_bulk,
        "min_ess_tail_theta": min_ess_tail,
        "pass": (
            n_div < 10
            and max_rhat < 1.01
            and min_ess_bulk > 400
            and min_ess_tail > 400
        ),
        "hyperparameter_summary": summary,
    }
    return diagnostics


def get_team_strengths(idata: az.InferenceData, data: dict) -> dict:
    """Extract posterior team strength summaries."""
    theta = idata.posterior["theta"].values  # (chains, draws, teams)
    theta_flat = theta.reshape(-1, theta.shape[-1])  # (samples, teams)

    means = theta_flat.mean(axis=0)
    stds = theta_flat.std(axis=0)

    ranking = np.argsort(-means)

    return {
        "samples": theta_flat,
        "means": means,
        "stds": stds,
        "ranking": ranking,
        "team_names": data["team_names"],
        "team_ids": data["team_ids"],
    }


def win_probability_samples(
    theta_samples: np.ndarray,
    sigma_samples: np.ndarray,
    team_i_idx: int,
    team_j_idx: int,
) -> np.ndarray:
    """Compute win probability distribution for team_i over team_j.

    Returns array of P(i beats j) for each posterior sample.
    Uses the probit formula: P(i>j) = Phi((theta_i - theta_j) / sigma)
    """
    from scipy.stats import norm

    diff = theta_samples[:, team_i_idx] - theta_samples[:, team_j_idx]
    return norm.cdf(diff / sigma_samples)
