"""Run the full offense/defense pipeline: compare likelihoods, backtest, plot, prep submission."""

import gc
import json
import numpy as np
import pymc as pm
import arviz as az
import pandas as pd

from src.data import build_model_data
from src.model import (
    build_offense_defense_model, fit_model, check_diagnostics, get_team_strengths,
)
from src.simulate import (
    build_bracket_structure, simulate_tournament, tail_analysis,
    championship_gini, get_most_likely_bracket,
)
from src.submission import generate_submission
from src.validate import run_validation
from src import visualize as viz


def step1_compare_likelihoods():
    """Compare Gaussian vs Student-t likelihoods via LOO-CV on men's data."""
    print("\n" + "=" * 70)
    print("  STEP 1: Gaussian vs Student-t Likelihood Comparison (Men's)")
    print("=" * 70)

    data = build_model_data(2026, "M")
    print(f"Data: {data['n_teams']} teams, {data['n_games']} games")

    idatas = {}
    models = {}
    for likelihood in ["normal", "studentt"]:
        print(f"\nFitting {likelihood} model...")
        model = build_offense_defense_model(data, likelihood=likelihood)
        idata = fit_model(model, draws=2000, tune=2000, chains=4)

        diag = check_diagnostics(idata)
        print(f"  Divergences: {diag['divergences']}")
        print(f"  Max R-hat: {diag['max_rhat']:.4f}")
        print(f"  Min ESS bulk: {diag['min_ess_bulk']:.0f}")
        print(f"  PASS: {diag['pass']}")

        # Compute log-likelihood (nutpie doesn't store it)
        print("  Computing log-likelihood for LOO...")
        pm.compute_log_likelihood(idata, model=model)

        # Combine score_i and score_j log-likelihoods into one array per game.
        # Each game has two independent observations; for LOO we treat each
        # game as one data point with combined log-lik = ll(score_i) + ll(score_j).
        import xarray as xr
        ll_i = idata.log_likelihood["score_i"]
        ll_j = idata.log_likelihood["score_j"]
        combined = ll_i + ll_j
        idata.log_likelihood = xr.Dataset({"score": combined})

        idatas[likelihood] = idata
        models[likelihood] = model

        idata.to_netcdf(f"results/model_2026_mens_offdef_{likelihood}.nc")

    # Compare
    comparison = az.compare(
        {"Gaussian": idatas["normal"], "Student-t": idatas["studentt"]},
        ic="loo",
        var_name="score",
    )
    print("\nLOO-CV Model Comparison:")
    print(comparison[["rank", "elpd_loo", "p_loo", "elpd_diff", "weight"]])
    comparison.to_csv("results/loo_comparison.csv")

    # Determine winner
    winner = comparison.index[0]
    winner_key = "normal" if winner == "Gaussian" else "studentt"
    print(f"\nWinner: {winner}")

    # If Student-t, print nu
    if "nu" in list(idatas["studentt"].posterior.data_vars):
        nu = idatas["studentt"].posterior["nu"].values.flatten()
        print(f"Student-t nu: {nu.mean():.1f} ± {nu.std():.1f}")

    return {
        "comparison": comparison,
        "idatas": idatas,
        "models": models,
        "winner": winner_key,
        "data": data,
    }


def step2_backtest(model_type="offense_defense"):
    """Run historical backtesting for the offense/defense model."""
    print("\n" + "=" * 70)
    print("  STEP 2: Historical Backtesting (Men's, 2015-2025)")
    print("=" * 70)

    # Use fewer draws on low-memory machines; 500 draws is enough for Brier calibration
    results = run_validation(
        gender="M",
        draws=500,
        tune=500,
        model_type=model_type,
    )

    print(f"\nBrier score: {results['brier_score']:.4f}")
    print(f"Log loss: {results['log_loss']:.4f}")

    # Per-season breakdown
    for season_result in results["seasons"]:
        preds = season_result["predictions"]
        y = preds["outcome"].values
        p = preds["predicted_p"].values
        brier = np.mean((p - y) ** 2)
        acc = np.mean((p > 0.5) == y)
        print(f"  {season_result['season']}: Brier={brier:.3f}, Accuracy={acc:.1%}, n={season_result['n_games']}")

    results["predictions"].to_csv("results/validation_offdef.csv", index=False)
    with open("results/validation_offdef_results.json", "w") as f:
        json.dump({
            "brier_score": results["brier_score"],
            "log_loss": results["log_loss"],
            "n_seasons": len(results["seasons"]),
            "n_games": sum(r["n_games"] for r in results["seasons"]),
        }, f, indent=2)

    return results


def step3_full_run_and_plots(comparison_result):
    """Fit winning model for both genders, simulate, generate all plots."""
    print("\n" + "=" * 70)
    print("  STEP 3: Full Pipeline + All Plots")
    print("=" * 70)

    winner_key = comparison_result["winner"]
    men_data = comparison_result["data"]
    men_idata = comparison_result["idatas"][winner_key]

    print(f"\nUsing {winner_key} likelihood")
    print(f"Men's model already fit from step 1")

    # Men's strengths and simulation
    men_strengths = get_team_strengths(men_idata, men_data)
    men_off = men_strengths["off_samples"]
    men_def = men_strengths["def_samples"]
    men_sigma = men_idata.posterior["sigma"].values.flatten()
    men_alpha = men_idata.posterior["alpha"].values.flatten()

    print(f"\nMen's Top 10:")
    for rank, idx in enumerate(men_strengths["ranking"][:10], 1):
        name = men_strengths["team_names"][idx]
        overall = men_strengths["overall_means"][idx]
        off_val = men_strengths["off_means"][idx]
        def_val = men_strengths["def_means"][idx]
        print(f"  {rank}. {name:<20s} overall={overall:.1f} (off={off_val:.1f}, def={def_val:.1f})")

    if men_strengths.get("off_def_corr") is not None:
        corr = men_strengths["off_def_corr"]
        print(f"\nOff/Def correlation: {corr.mean():.3f} ± {corr.std():.3f}")

    print(f"\nSimulating 10,000 men's tournaments...")
    men_bracket = build_bracket_structure(2026, "M")
    men_sim = simulate_tournament(
        men_bracket, sigma_samples=men_sigma, team_ids=men_data["team_ids"],
        n_sims=10000, seed=42, off_samples=men_off, def_samples=men_def,
    )
    men_sim["advancement"].to_csv("results/advancement_probs_mens.csv", index=False)
    with open("results/champions_mens.json", "w") as f:
        json.dump(men_sim["champions"], f, indent=2)

    print("\nMen's Championship Probabilities:")
    for _, row in men_sim["advancement"].head(10).iterrows():
        print(f"  {row['TeamName']:<25} {row['Champion']:.1%}")

    # Women's
    print(f"\nFitting women's model...")
    women_data = build_model_data(2026, "W")
    women_model = build_offense_defense_model(women_data, likelihood=winner_key)
    women_idata = fit_model(women_model, draws=2000, tune=2000, chains=4)
    women_idata.to_netcdf("results/model_2026_womens_offdef.nc")

    women_diag = check_diagnostics(women_idata)
    print(f"Women's diagnostics: div={women_diag['divergences']}, rhat={women_diag['max_rhat']:.4f}, PASS={women_diag['pass']}")

    women_strengths = get_team_strengths(women_idata, women_data)
    women_off = women_strengths["off_samples"]
    women_def = women_strengths["def_samples"]
    women_sigma = women_idata.posterior["sigma"].values.flatten()
    women_alpha = women_idata.posterior["alpha"].values.flatten()

    print(f"\nWomen's Top 10:")
    for rank, idx in enumerate(women_strengths["ranking"][:10], 1):
        name = women_strengths["team_names"][idx]
        overall = women_strengths["overall_means"][idx]
        print(f"  {rank}. {name:<20s} overall={overall:.1f}")

    print(f"\nSimulating 10,000 women's tournaments...")
    women_bracket = build_bracket_structure(2026, "W")
    women_sim = simulate_tournament(
        women_bracket, sigma_samples=women_sigma, team_ids=women_data["team_ids"],
        n_sims=10000, seed=42, off_samples=women_off, def_samples=women_def,
        alpha_samples=women_alpha,
    )
    women_sim["advancement"].to_csv("results/advancement_probs_womens.csv", index=False)

    # --- ALL PLOTS ---
    print(f"\nGenerating all plots...")

    # Men's team strength forest (overall)
    fig = viz.plot_team_strength_forest(men_idata, men_data["team_names"], top_n=30)
    viz.save_fig(fig, "team_strengths_top30")

    # Off/def scatter (marquee figure)
    fig = viz.plot_off_def_scatter(
        men_strengths, men_data["conf_idx"], men_data["conf_names"],
    )
    viz.save_fig(fig, "off_def_scatter")

    # Off/def rankings
    fig = viz.plot_off_def_rankings(men_strengths, top_n=20)
    viz.save_fig(fig, "off_def_rankings")

    # Conference off/def
    fig = viz.plot_conference_off_def(men_idata, men_data["conf_names"])
    viz.save_fig(fig, "conference_off_def")

    # Championship odds
    fig = viz.plot_championship_odds(men_sim["advancement"], top_n=20)
    viz.save_fig(fig, "championship_odds")

    # Advancement heatmap
    fig = viz.plot_advancement_heatmap(men_sim["advancement"], top_n=40)
    viz.save_fig(fig, "advancement_heatmap")

    # Bracket
    men_most_likely = get_most_likely_bracket(men_sim, men_bracket)
    fig = viz.plot_bracket(men_sim["advancement"], men_bracket, men_most_likely)
    viz.save_fig(fig, "bracket_forecast")

    # Upset probabilities
    fig = viz.plot_upset_probabilities(men_sim["advancement"])
    viz.save_fig(fig, "upset_probabilities")

    # Home court comparison
    fig = viz.plot_home_court_comparison(men_alpha, women_alpha)
    viz.save_fig(fig, "home_court")

    # Championship odds comparison
    fig = viz.plot_championship_odds_comparison(
        men_sim["advancement"], women_sim["advancement"],
    )
    viz.save_fig(fig, "championship_comparison")

    # LOO comparison
    fig = viz.plot_loo_comparison(comparison_result["comparison"])
    viz.save_fig(fig, "loo_comparison")

    # Posterior predictive scores
    fig = viz.plot_posterior_predictive_scores(
        men_idata, men_data["score_i"],
    )
    viz.save_fig(fig, "ppc_scores")

    # Women's plots
    fig = viz.plot_team_strength_forest(women_idata, women_data["team_names"], top_n=30)
    viz.save_fig(fig, "team_strengths_top30_womens")

    fig = viz.plot_off_def_scatter(
        women_strengths, women_data["conf_idx"], women_data["conf_names"],
    )
    viz.save_fig(fig, "off_def_scatter_womens")

    fig = viz.plot_advancement_heatmap(women_sim["advancement"], top_n=40)
    viz.save_fig(fig, "advancement_heatmap_womens")

    fig = viz.plot_bracket(women_sim["advancement"], women_bracket,
                           get_most_likely_bracket(women_sim, women_bracket))
    viz.save_fig(fig, "bracket_forecast_womens")

    print(f"\nAll plots saved to {viz.FIGURES_DIR}")

    return {
        "men": {
            "data": men_data, "idata": men_idata,
            "strengths": men_strengths, "sim": men_sim,
            "off": men_off, "def": men_def, "sigma": men_sigma,
        },
        "women": {
            "data": women_data, "idata": women_idata,
            "strengths": women_strengths, "sim": women_sim,
            "off": women_off, "def": women_def, "sigma": women_sigma,
        },
    }


def step4_prep_submission(results):
    """Prepare submission CSV but do not submit."""
    print("\n" + "=" * 70)
    print("  STEP 4: Prepare Kaggle Submission (NOT submitting)")
    print("=" * 70)

    men = results["men"]
    women = results["women"]

    sub = generate_submission(
        men_sigma=men["sigma"],
        men_team_ids=men["data"]["team_ids"],
        men_off=men["off"],
        men_def=men["def"],
        women_sigma=women["sigma"],
        women_team_ids=women["data"]["team_ids"],
        women_off=women["off"],
        women_def=women["def"],
        output_path="results/submission_offdef.csv",
    )

    print(f"\nSubmission shape: {sub.shape}")
    print(f"Prediction range: [{sub['Pred'].min():.4f}, {sub['Pred'].max():.4f}]")
    print(f"Mean prediction: {sub['Pred'].mean():.4f}")
    print(f"\nSubmission saved to results/submission_offdef.csv (NOT submitted)")

    return sub


if __name__ == "__main__":
    # Step 1: Gaussian vs Student-t
    comparison_result = step1_compare_likelihoods()
    gc.collect()

    # Step 2: Historical backtesting
    validation_result = step2_backtest()
    gc.collect()

    # Step 3: Full pipeline with plots
    results = step3_full_run_and_plots(comparison_result)
    gc.collect()

    # Step 4: Prep submission
    submission = step4_prep_submission(results)

    print("\n" + "=" * 70)
    print("  ALL DONE")
    print("=" * 70)
