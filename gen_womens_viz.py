"""Generate women's tournament visualizations and comparison plots."""

import pandas as pd
import arviz as az

from src.data import build_model_data
from src.model import get_team_strengths
from src.simulate import build_bracket_structure, get_most_likely_bracket
from src.visualize import (
    plot_team_strength_forest,
    plot_championship_odds,
    plot_advancement_heatmap,
    plot_bracket,
    plot_upset_probabilities,
    plot_home_court_comparison,
    plot_championship_odds_comparison,
    save_fig,
)


def main():
    # Load women's model and data
    print("Loading women's model...")
    w_idata = az.from_netcdf("results/model_2026_womens.nc")
    w_data = build_model_data(2026, gender="W")
    w_adv = pd.read_csv("results/advancement_probs_womens.csv")

    # Load men's model for comparisons
    print("Loading men's model...")
    m_idata = az.from_netcdf("results/model_2026_mens.nc")
    m_adv = pd.read_csv("results/advancement_probs_mens.csv")

    # Women's team strengths
    print("Generating women's team strengths...")
    fig = plot_team_strength_forest(w_idata, w_data["team_names"], top_n=30)
    fig.axes[0].set_title("Women's Posterior Team Strength (94% HDI)")
    save_fig(fig, "team_strengths_top30_womens")

    # Women's championship odds
    print("Generating women's championship odds...")
    fig = plot_championship_odds(w_adv, top_n=20)
    fig.axes[0].set_title("Women's — Who Wins It All?")
    save_fig(fig, "championship_odds_womens")

    # Women's advancement heatmap
    print("Generating women's advancement heatmap...")
    fig = plot_advancement_heatmap(w_adv, top_n=40)
    fig.axes[0].set_title("Women's Tournament Advancement Probabilities")
    save_fig(fig, "advancement_heatmap_womens")

    # Women's bracket forecast
    print("Generating women's bracket forecast...")
    w_bracket = build_bracket_structure(2026, gender="W")
    # Need sim_results for most_likely bracket — load from saved advancement
    # Use advancement CSV as proxy (bracket plot only needs advancement + bracket_struct)
    fig = plot_bracket(w_adv, w_bracket, {})
    fig.texts[0].set_text("2026 NCAA Women's Tournament Bracket Forecast")
    save_fig(fig, "bracket_forecast_womens")

    # Women's upset probabilities
    print("Generating women's upset probabilities...")
    fig = plot_upset_probabilities(w_adv)
    fig.axes[0].set_title("Women's First Round Upset Probabilities")
    save_fig(fig, "upset_probabilities_womens")

    # Comparison: championship odds side-by-side
    print("Generating championship odds comparison...")
    fig = plot_championship_odds_comparison(m_adv, w_adv, top_n=10)
    save_fig(fig, "championship_odds_comparison")

    # Comparison: home court advantage
    print("Generating home court comparison...")
    m_alpha = m_idata.posterior["alpha"].values.flatten()
    w_alpha = w_idata.posterior["alpha"].values.flatten()
    fig = plot_home_court_comparison(m_alpha, w_alpha)
    save_fig(fig, "home_court_comparison")

    print("Done!")


if __name__ == "__main__":
    main()
