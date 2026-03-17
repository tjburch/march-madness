"""Regenerate all blog figures from saved model artifacts (no re-fitting)."""

import arviz as az
import pandas as pd

from src.data import build_model_data
from src.model import get_team_strengths
from src.simulate import build_bracket_structure, get_most_likely_bracket, simulate_tournament
from src import visualize as viz


def main():
    # Load saved models
    print("Loading men's model...")
    m_idata = az.from_netcdf("results/model_2026_mens.nc")
    m_data = build_model_data(2026, "M")

    print("Loading women's model...")
    w_idata = az.from_netcdf("results/model_2026_womens.nc")
    w_data = build_model_data(2026, "W")

    # Get team strengths
    m_strengths = get_team_strengths(m_idata, m_data)
    w_strengths = get_team_strengths(w_idata, w_data)

    # Simulate tournaments (needed for bracket/advancement plots)
    print("Simulating men's tournament...")
    m_bracket = build_bracket_structure(2026, "M")
    m_sigma = m_idata.posterior["sigma"].values.flatten()
    m_alpha = m_idata.posterior["alpha"].values.flatten()
    m_sim = simulate_tournament(
        m_bracket, sigma_samples=m_sigma, team_ids=m_data["team_ids"],
        n_sims=10000, seed=42,
        off_samples=m_strengths["off_samples"],
        def_samples=m_strengths["def_samples"],
    )

    print("Simulating women's tournament...")
    w_bracket = build_bracket_structure(2026, "W")
    w_sigma = w_idata.posterior["sigma"].values.flatten()
    w_alpha = w_idata.posterior["alpha"].values.flatten()
    w_sim = simulate_tournament(
        w_bracket, sigma_samples=w_sigma, team_ids=w_data["team_ids"],
        n_sims=10000, seed=42,
        off_samples=w_strengths["off_samples"],
        def_samples=w_strengths["def_samples"],
        alpha_samples=w_alpha,
    )

    m_adv = m_sim["advancement"]
    w_adv = w_sim["advancement"]

    # --- Men's plots ---
    print("\nGenerating men's plots...")

    fig = viz.plot_team_strength_forest(m_idata, m_data["team_names"], top_n=30)
    viz.save_fig(fig, "team_strengths_top30")

    fig = viz.plot_off_def_scatter(m_strengths, m_data["conf_idx"], m_data["conf_names"])
    viz.save_fig(fig, "off_def_scatter")

    fig = viz.plot_off_def_rankings(m_strengths, top_n=20)
    viz.save_fig(fig, "off_def_rankings")

    fig = viz.plot_conference_off_def(m_idata, m_data["conf_names"])
    viz.save_fig(fig, "conference_off_def")

    fig = viz.plot_championship_odds(m_adv, top_n=20)
    viz.save_fig(fig, "championship_odds")

    fig = viz.plot_advancement_heatmap(m_adv, top_n=40)
    viz.save_fig(fig, "advancement_heatmap")

    m_most_likely = get_most_likely_bracket(m_sim, m_bracket)
    fig = viz.plot_bracket(m_adv, m_bracket, m_most_likely)
    viz.save_fig(fig, "bracket_forecast")

    fig = viz.plot_upset_probabilities(m_adv)
    viz.save_fig(fig, "upset_probabilities")

    # Home court comparison
    fig = viz.plot_home_court_comparison(m_alpha, w_alpha)
    viz.save_fig(fig, "home_court")

    # Championship comparison
    fig = viz.plot_championship_odds_comparison(m_adv, w_adv)
    viz.save_fig(fig, "championship_comparison")

    # Posterior predictive scores
    fig = viz.plot_posterior_predictive_scores(m_idata, m_data["score_i"])
    viz.save_fig(fig, "ppc_scores")

    # --- Women's plots ---
    print("\nGenerating women's plots...")

    fig = viz.plot_team_strength_forest(w_idata, w_data["team_names"], top_n=30)
    viz.save_fig(fig, "team_strengths_top30_womens")

    fig = viz.plot_off_def_scatter(w_strengths, w_data["conf_idx"], w_data["conf_names"])
    viz.save_fig(fig, "off_def_scatter_womens")

    fig = viz.plot_advancement_heatmap(w_adv, top_n=40)
    viz.save_fig(fig, "advancement_heatmap_womens")

    w_most_likely = get_most_likely_bracket(w_sim, w_bracket)
    fig = viz.plot_bracket(w_adv, w_bracket, w_most_likely)
    fig.texts[0].set_text("2026 NCAA Women's Tournament Bracket Forecast")
    viz.save_fig(fig, "bracket_forecast_womens")

    fig = viz.plot_championship_odds(w_adv, top_n=20)
    fig.axes[0].set_title("Women's — Who Wins It All?")
    viz.save_fig(fig, "championship_odds_womens")

    fig = viz.plot_upset_probabilities(w_adv)
    fig.axes[0].set_title("Women's First Round Upset Probabilities")
    viz.save_fig(fig, "upset_probabilities_womens")

    print(f"\nAll figures saved to {viz.FIGURES_DIR}")


if __name__ == "__main__":
    main()
