"""Run the full March Madness Bayesian bracketology pipeline."""

import json
import numpy as np
import pymc as pm
import arviz as az

from src.data import build_model_data
from src.model import (
    build_offense_defense_model, fit_model, check_diagnostics, get_team_strengths,
)
from src.simulate import build_bracket_structure, simulate_tournament, tail_analysis, championship_gini
from src.submission import generate_submission


def run_gender(gender: str, season: int = 2026, n_sims: int = 10000):
    """Fit offense/defense model and simulate tournament for one gender."""
    label = "Men's" if gender == "M" else "Women's"
    suffix = "mens" if gender == "M" else "womens"

    # 1. Load data
    print(f"\n{'='*60}")
    print(f"  {label} Tournament")
    print(f"{'='*60}")
    print(f"Loading {season} regular season data...")
    data = build_model_data(season, gender)
    print(f"  {data['n_teams']} teams, {data['n_conferences']} conferences, {data['n_games']} games")

    # 2. Build and fit offense/defense model
    model = build_offense_defense_model(data, likelihood="normal")

    print("\nChecking priors...")
    with model:
        prior = pm.sample_prior_predictive(draws=500, random_seed=42)
    prior_scores = prior.prior_predictive["score_i"].values.flatten()
    print(f"  Prior score std: {prior_scores.std():.1f} (observed: {data['score_i'].std():.1f})")

    print("\nFitting model (2000 draws, 2000 tune, 4 chains)...")
    idata = fit_model(model, draws=2000, tune=2000, chains=4)
    idata.to_netcdf(f"results/model_{season}_{suffix}.nc")

    # 3. Diagnostics
    diag = check_diagnostics(idata)
    print(f"\nDiagnostics:")
    print(f"  Divergences: {diag['divergences']}")
    print(f"  Max R-hat: {diag['max_rhat']:.4f}")
    print(f"  Min ESS bulk: {diag['min_ess_bulk']:.0f}")
    print(f"  PASS: {diag['pass']}")

    # 4. Top teams
    strengths = get_team_strengths(idata, data)
    print(f"\nTop 10 Teams:")
    for rank, idx in enumerate(strengths["ranking"][:10], 1):
        name = strengths["team_names"][idx]
        overall = strengths["overall_means"][idx]
        off_val = strengths["off_means"][idx]
        def_val = strengths["def_means"][idx]
        print(f"  {rank}. {name:<20s} overall={overall:.1f} (off={off_val:.1f}, def={def_val:.1f})")

    # 5. Simulate tournament
    print(f"\nSimulating {n_sims:,} tournaments...")
    bracket_struct = build_bracket_structure(season, gender)
    off_samples = strengths["off_samples"]
    def_samples = strengths["def_samples"]
    sigma_samples = idata.posterior["sigma"].values.flatten()
    alpha_samples = idata.posterior["alpha"].values.flatten()

    sim_results = simulate_tournament(
        bracket_struct, sigma_samples=sigma_samples,
        team_ids=data["team_ids"], n_sims=n_sims, seed=42,
        off_samples=off_samples, def_samples=def_samples,
        alpha_samples=alpha_samples if gender == "W" else None,
    )

    sim_results["advancement"].to_csv(
        f"results/advancement_probs_{suffix}.csv", index=False
    )
    with open(f"results/champions_{suffix}.json", "w") as f:
        json.dump(sim_results["champions"], f, indent=2)

    # 6. Results
    print(f"\nChampionship Probabilities:")
    for _, row in sim_results["advancement"].head(10).iterrows():
        print(f"  {row['TeamName']:<25} {row['Champion']:.1%}")

    gini = championship_gini(sim_results["advancement"])
    print(f"\nChampionship Gini coefficient: {gini:.3f}")

    tails = tail_analysis(sim_results)
    print(f"P(zero #1 seeds in Final Four): {tails['no_one_seeds_final_four_pct']:.1%}")

    return {
        "data": data,
        "idata": idata,
        "off_samples": off_samples,
        "def_samples": def_samples,
        "sigma_samples": sigma_samples,
        "sim_results": sim_results,
    }


def compare_likelihoods(data: dict, draws: int = 2000, tune: int = 2000) -> dict:
    """Compare Gaussian vs Student-t likelihoods via LOO-CV."""
    models = {}
    idatas = {}
    for likelihood in ["normal", "studentt"]:
        print(f"\nFitting {likelihood} model...")
        model = build_offense_defense_model(data, likelihood=likelihood)
        idata = fit_model(model, draws=draws, tune=tune, chains=4)
        pm.compute_log_likelihood(idata, model=model)
        models[likelihood] = model
        idatas[likelihood] = idata

    comparison = az.compare(
        {"Gaussian": idatas["normal"], "Student-t": idatas["studentt"]},
        ic="loo",
    )
    print("\nModel Comparison (LOO-CV):")
    print(comparison[["rank", "elpd_loo", "elpd_diff", "weight"]])

    return {
        "comparison": comparison,
        "idatas": idatas,
        "models": models,
    }


def main():
    results = {}
    for gender in ["M", "W"]:
        results[gender] = run_gender(gender, season=2026, n_sims=10000)

    # Generate Kaggle submission
    print(f"\n{'='*60}")
    print("  Generating Kaggle Submission")
    print(f"{'='*60}")
    generate_submission(
        men_sigma=results["M"]["sigma_samples"],
        men_team_ids=results["M"]["data"]["team_ids"],
        men_off=results["M"]["off_samples"],
        men_def=results["M"]["def_samples"],
        women_sigma=results["W"]["sigma_samples"],
        women_team_ids=results["W"]["data"]["team_ids"],
        women_off=results["W"]["off_samples"],
        women_def=results["W"]["def_samples"],
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
