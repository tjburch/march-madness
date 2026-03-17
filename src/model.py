"""Bayesian Bradley-Terry model for NCAA basketball."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
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


def build_offense_defense_model(
    data: dict, likelihood: str = "normal"
) -> pm.Model:
    """Build offense/defense model with score-based likelihood.

    Decomposes team strength into offense (generates points) and defense
    (prevents points) via a multivariate normal prior with LKJ correlation.

    score_i ~ Likelihood(mu + off_i - def_j + alpha * home_i, sigma)
    score_j ~ Likelihood(mu + off_j - def_i - alpha * home_i, sigma)

    Args:
        data: Output of build_model_data() (must include score_i, score_j)
        likelihood: "normal" or "studentt"
    """
    coords = {
        "team": data["team_names"],
        "conference": data["conf_names"],
        "game": np.arange(data["n_games"]),
        "component": ["offense", "defense"],
    }

    with pm.Model(coords=coords) as model:
        # Data
        team_i = pm.Data("team_i", data["team_idx_i"], dims="game")
        team_j = pm.Data("team_j", data["team_idx_j"], dims="game")
        home = pm.Data("home", data["home_i"], dims="game")
        conf_of_team = pm.Data("conf_of_team", data["conf_idx"])

        # Global intercept (average score per team per game)
        mu_intercept = pm.Normal("mu_intercept", mu=70, sigma=10)

        # Conference-level effects (independent for offense/defense)
        sigma_off_conf = pm.HalfNormal("sigma_off_conf", sigma=5)
        sigma_def_conf = pm.HalfNormal("sigma_def_conf", sigma=5)
        mu_off_conf = pm.Normal(
            "mu_off_conf", mu=0, sigma=sigma_off_conf, dims="conference"
        )
        mu_def_conf = pm.Normal(
            "mu_def_conf", mu=0, sigma=sigma_def_conf, dims="conference"
        )

        # Team-level offense/defense (multivariate, non-centered)
        sd_dist = pm.HalfNormal.dist(sigma=5)
        chol, corr, stds = pm.LKJCholeskyCov(
            "lkj",
            n=2,
            eta=2,
            sd_dist=sd_dist,
            compute_corr=True,
        )
        z = pm.Normal("z", mu=0, sigma=1, shape=(data["n_teams"], 2))

        # Transform: team effects = conference mean + L @ z'
        team_effects = pm.Deterministic(
            "team_effects",
            pt.stack([mu_off_conf[conf_of_team], mu_def_conf[conf_of_team]], axis=1)
            + pt.dot(z, chol.T),
        )

        off = pm.Deterministic("off", team_effects[:, 0], dims="team")
        deff = pm.Deterministic("def", team_effects[:, 1], dims="team")

        # Home court advantage
        alpha = pm.Normal("alpha", mu=3.5, sigma=2.0)

        # Observation noise
        sigma = pm.HalfNormal("sigma", sigma=15)

        # Expected scores
        mu_score_i = mu_intercept + off[team_i] - deff[team_j] + alpha * home
        mu_score_j = mu_intercept + off[team_j] - deff[team_i] - alpha * home

        # Likelihood
        if likelihood == "normal":
            pm.Normal(
                "score_i", mu=mu_score_i, sigma=sigma,
                observed=data["score_i"], dims="game",
            )
            pm.Normal(
                "score_j", mu=mu_score_j, sigma=sigma,
                observed=data["score_j"], dims="game",
            )
        elif likelihood == "studentt":
            nu = pm.Gamma("nu", alpha=2, beta=0.1)
            pm.StudentT(
                "score_i", nu=nu, mu=mu_score_i, sigma=sigma,
                observed=data["score_i"], dims="game",
            )
            pm.StudentT(
                "score_j", nu=nu, mu=mu_score_j, sigma=sigma,
                observed=data["score_j"], dims="game",
            )
        else:
            raise ValueError(f"Unknown likelihood: {likelihood!r}")

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
    """Run standard convergence diagnostics.

    Auto-detects whether this is a margin-based (theta) or score-based
    (off/def) model based on which variables are in the posterior.
    """
    n_div = idata.sample_stats["diverging"].sum().item()

    posterior_vars = list(idata.posterior.data_vars)
    is_offdef = "off" in posterior_vars

    if is_offdef:
        hyper_vars = [
            v for v in [
                "alpha", "sigma", "mu_intercept",
                "sigma_off_conf", "sigma_def_conf",
                "mu_off_conf", "mu_def_conf",
            ]
            if v in posterior_vars
        ]
        if "nu" in posterior_vars:
            hyper_vars.append("nu")

        summary = az.summary(idata, var_names=hyper_vars)

        team_summary = az.summary(idata, var_names=["off", "def"])
        max_rhat = float(team_summary["r_hat"].max())
        min_ess_bulk = float(team_summary["ess_bulk"].min())
        min_ess_tail = float(team_summary["ess_tail"].min())

        if "lkj_corr" in posterior_vars:
            corr_summary = az.summary(idata, var_names=["lkj_corr"])
            max_rhat = max(max_rhat, float(corr_summary["r_hat"].max()))
            min_ess_bulk = min(min_ess_bulk, float(corr_summary["ess_bulk"].min()))
    else:
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
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "pass": (
            n_div < 10
            and max_rhat < 1.01
            and min_ess_bulk > 400
            and min_ess_tail > 400
        ),
        "hyperparameter_summary": summary,
    }

    # Backward compatibility
    if not is_offdef:
        diagnostics["max_rhat_theta"] = max_rhat
        diagnostics["min_ess_bulk_theta"] = min_ess_bulk
        diagnostics["min_ess_tail_theta"] = min_ess_tail

    return diagnostics


def get_team_strengths(idata: az.InferenceData, data: dict) -> dict:
    """Extract posterior team strength summaries.

    Auto-detects margin-based (theta) vs score-based (off/def) models.
    """
    posterior_vars = list(idata.posterior.data_vars)

    if "off" in posterior_vars:
        off = idata.posterior["off"].values
        deff = idata.posterior["def"].values
        off_flat = off.reshape(-1, off.shape[-1])
        def_flat = deff.reshape(-1, deff.shape[-1])

        overall = off_flat + def_flat
        means = overall.mean(axis=0)
        stds = overall.std(axis=0)
        ranking = np.argsort(-means)

        off_def_corr = None
        if "lkj_corr" in posterior_vars:
            corr_matrix = idata.posterior["lkj_corr"].values
            off_def_corr = corr_matrix[:, :, 0, 1].flatten()

        return {
            "off_samples": off_flat,
            "def_samples": def_flat,
            "overall_means": means,
            "overall_stds": stds,
            "off_means": off_flat.mean(axis=0),
            "def_means": def_flat.mean(axis=0),
            "ranking": ranking,
            "team_names": data["team_names"],
            "team_ids": data["team_ids"],
            "off_def_corr": off_def_corr,
            # Backward compat aliases
            "samples": overall,
            "means": means,
            "stds": stds,
        }
    else:
        theta = idata.posterior["theta"].values
        theta_flat = theta.reshape(-1, theta.shape[-1])
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
    team_i_idx: int,
    team_j_idx: int,
    sigma_samples: np.ndarray,
    theta_samples: np.ndarray | None = None,
    off_samples: np.ndarray | None = None,
    def_samples: np.ndarray | None = None,
) -> np.ndarray:
    """Compute win probability distribution for team_i over team_j.

    Returns array of P(i beats j) for each posterior sample.

    For theta model: P(i>j) = Phi((theta_i - theta_j) / sigma)
    For off/def model: P(i>j) = Phi(diff / (sigma * sqrt(2)))
        where diff = (off_i - def_j) - (off_j - def_i)
    """
    from scipy.stats import norm

    if off_samples is not None and def_samples is not None:
        diff = (
            (off_samples[:, team_i_idx] - def_samples[:, team_j_idx])
            - (off_samples[:, team_j_idx] - def_samples[:, team_i_idx])
        )
        return norm.cdf(diff / (sigma_samples * np.sqrt(2)))
    elif theta_samples is not None:
        diff = theta_samples[:, team_i_idx] - theta_samples[:, team_j_idx]
        return norm.cdf(diff / sigma_samples)
    else:
        raise ValueError("Provide either theta_samples or off_samples + def_samples")
