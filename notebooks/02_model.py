"""Fit the Bayesian Bradley-Terry model and simulate the tournament."""

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import build_model_data, load_seeds
from src.model import build_bradley_terry, fit_model, check_diagnostics, get_team_strengths
from src.simulate import build_bracket_structure, simulate_tournament, tail_analysis, get_most_likely_bracket, championship_gini
from src.visualize import (
    plot_team_strength_forest,
    plot_conference_effects,
    plot_advancement_heatmap,
    plot_championship_odds,
    plot_bracket,
    plot_upset_probabilities,
    save_fig,
)

az.style.use("arviz-darkgrid")

# %% [markdown]
# ## 1. Build Model Data

# %%
data = build_model_data(2026)
print(f"Teams: {data['n_teams']}")
print(f"Conferences: {data['n_conferences']}")
print(f"Games: {data['n_games']}")

# %% [markdown]
# ## 2. Prior Predictive Check

# %%
model = build_bradley_terry(data)
print(model)

# %%
with model:
    prior_pred = pm.sample_prior_predictive(draws=500, random_seed=42)

prior_margins = prior_pred.prior_predictive["margin"].values.flatten()
print(f"Prior predictive margin range: [{prior_margins.min():.0f}, {prior_margins.max():.0f}]")
print(f"Prior predictive margin mean: {prior_margins.mean():.1f}")
print(f"Prior predictive margin std: {prior_margins.std():.1f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(prior_margins, bins=100, density=True, alpha=0.7, label="Prior predictive")
ax.hist(data["margin"], bins=50, density=True, alpha=0.7, label="Observed data")
ax.set_xlabel("Score Margin")
ax.set_ylabel("Density")
ax.set_title("Prior Predictive Check: Score Margins")
ax.legend()
plt.tight_layout()
save_fig(fig, "prior_predictive")

# %% [markdown]
# ## 3. Fit the Model

# %%
idata = fit_model(model, draws=2000, tune=2000, chains=4)
idata.to_netcdf("../results/model_2026.nc")

# %% [markdown]
# ## 4. Convergence Diagnostics

# %%
diagnostics = check_diagnostics(idata)
print(f"Divergences: {diagnostics['divergences']}")
print(f"Max R-hat (theta): {diagnostics['max_rhat_theta']:.4f}")
print(f"Min ESS bulk (theta): {diagnostics['min_ess_bulk_theta']:.0f}")
print(f"Min ESS tail (theta): {diagnostics['min_ess_tail_theta']:.0f}")
print(f"Diagnostics pass: {diagnostics['pass']}")
print(f"\nHyperparameter summary:")
print(diagnostics["hyperparameter_summary"])

# %%
az.plot_trace(idata, var_names=["alpha", "sigma", "sigma_conf", "sigma_team"], compact=True)
plt.tight_layout()
plt.savefig("../blogimages/march-madness-2026/trace_hyperparams.png", bbox_inches="tight")
plt.show()

# %%
# Energy diagnostic
az.plot_energy(idata)
plt.savefig("../blogimages/march-madness-2026/energy.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Posterior Predictive Check

# %%
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=42)

idata.to_netcdf("../results/model_2026.nc")

az.plot_ppc(idata, num_pp_samples=200, kind="cumulative")
plt.title("Posterior Predictive Check: Score Margins")
plt.savefig("../blogimages/march-madness-2026/ppc.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Team Strengths

# %%
strengths = get_team_strengths(idata, data)

fig = plot_team_strength_forest(idata, data["team_names"], top_n=30)
save_fig(fig, "team_strengths_top30")

# Print top 30 with uncertainty
print("Top 30 Teams by Posterior Mean Strength:")
print(f"{'Rank':>4}  {'Team':<25}  {'Mean':>6}  {'Std':>5}  {'94% HDI':>15}")
for rank, idx in enumerate(strengths["ranking"][:30], 1):
    samples = strengths["samples"][:, idx]
    hdi = az.hdi(samples, hdi_prob=0.94)
    print(f"{rank:4d}  {strengths['team_names'][idx]:<25}  {strengths['means'][idx]:6.1f}  "
          f"{strengths['stds'][idx]:5.1f}  [{hdi[0]:5.1f}, {hdi[1]:5.1f}]")

# %%
fig = plot_conference_effects(idata, data["conf_names"])
save_fig(fig, "conference_effects")

# %% [markdown]
# ## 7. Home Court Advantage

# %%
alpha_samples = idata.posterior["alpha"].values.flatten()
print(f"Home court advantage: {alpha_samples.mean():.2f} ± {alpha_samples.std():.2f} points")
hdi = az.hdi(alpha_samples, hdi_prob=0.94)
print(f"94% HDI: [{hdi[0]:.2f}, {hdi[1]:.2f}]")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(alpha_samples, bins=50, density=True, alpha=0.7, color="steelblue")
ax.axvline(alpha_samples.mean(), color="red", linestyle="--")
ax.set_xlabel("Home Court Advantage (points)")
ax.set_title("Posterior Distribution of Home Court Advantage")
plt.tight_layout()
save_fig(fig, "home_court")

# %% [markdown]
# ## 8. Tournament Simulation (10,000 brackets)

# %%
bracket_struct = build_bracket_structure(2026)

theta_samples = idata.posterior["theta"].values.reshape(-1, data["n_teams"])
sigma_samples = idata.posterior["sigma"].values.flatten()

sim_results = simulate_tournament(
    bracket_struct,
    theta_samples,
    sigma_samples,
    data["team_ids"],
    n_sims=10000,
    seed=42,
)

# Save simulation results
sim_results["advancement"].to_csv("../results/advancement_probs.csv", index=False)

# %%
print("Championship Probabilities:")
print(f"{'Team':<25}  {'P(Champion)':>12}")
print("-" * 40)
for _, row in sim_results["advancement"].head(20).iterrows():
    print(f"{row['TeamName']:<25}  {row['Champion']:>11.1%}")

gini = championship_gini(sim_results["advancement"])
n_winners = (sim_results["advancement"]["Champion"] > 0).sum()
print(f"\nChampionship Gini: {gini:.3f}")
print(f"Teams with >0 championship probability: {n_winners}")

# %%
fig = plot_championship_odds(sim_results["advancement"], top_n=20)
save_fig(fig, "championship_odds")

# %%
fig = plot_advancement_heatmap(sim_results["advancement"], top_n=40)
save_fig(fig, "advancement_heatmap")

# %%
fig = plot_bracket(sim_results["advancement"], bracket_struct, get_most_likely_bracket(sim_results, bracket_struct))
save_fig(fig, "bracket_forecast")

# %%
fig = plot_upset_probabilities(sim_results["advancement"])
save_fig(fig, "upset_probabilities")

# %% [markdown]
# ## 9. Tail Analysis — The Wild Simulations

# %%
tails = tail_analysis(sim_results)

print("Deepest Cinderella Runs (across 10,000 simulations):")
round_labels = {1: "R64 Win", 2: "Sweet 16", 3: "Elite 8", 4: "Final Four", 5: "Champ Game", 6: "Champion"}
for seed_line, (team, round_num) in sorted(tails["deepest_cinderella"].items()):
    print(f"  {seed_line}: {team} reached {round_labels.get(round_num, f'Round {round_num}')}")

print(f"\nProbability zero #1 seeds make Final Four: {tails['no_one_seeds_final_four_pct']:.2%}")

# Show the chalkiest and most chaotic Final Fours
chalkiest = sim_results["all_results"][tails["chalkiest_sim_idx"]]
chaotic = sim_results["all_results"][tails["most_chaotic_sim_idx"]]

print("\nChalkiest simulated Final Four:")
for slot, winner in chalkiest["results"].items():
    if slot.startswith("R4") and isinstance(winner, dict):
        print(f"  {winner['team_name']} ({winner['seed_num']}-seed)")

print("\nMost chaotic simulated Final Four:")
for slot, winner in chaotic["results"].items():
    if slot.startswith("R4") and isinstance(winner, dict):
        print(f"  {winner['team_name']} ({winner['seed_num']}-seed)")

# %%
# Distribution of chalk scores across simulations
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(tails["chalk_scores"], bins=50, edgecolor="black", alpha=0.7)
ax.axvline(tails["chalk_scores"][tails["chalkiest_sim_idx"]], color="green",
           linestyle="--", label="Chalkiest")
ax.axvline(tails["chalk_scores"][tails["most_chaotic_sim_idx"]], color="red",
           linestyle="--", label="Most chaotic")
ax.set_xlabel("Chalk Score (higher = more favorites win)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Bracket 'Chalkiness' Across 10,000 Simulations")
ax.legend()
plt.tight_layout()
save_fig(fig, "chalk_distribution")

# %% [markdown]
# ## 10. Key Matchup Win Probabilities

# %%
from scipy.stats import norm

seeds_df = load_seeds(2026)
tourney_teams = seeds_df.merge(
    pd.DataFrame({"TeamID": data["team_ids"], "idx": range(len(data["team_ids"]))}),
    on="TeamID",
)

print("First Round Matchup Win Probabilities:")
print(f"{'Matchup':<40}  {'P(Higher Seed)':>14}  {'Uncertainty':>10}")
print("-" * 70)

regions = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}
for region in ["W", "X", "Y", "Z"]:
    region_teams = tourney_teams[tourney_teams["Region"] == region].sort_values("SeedNum")
    print(f"\n  {regions[region]} Region:")

    # Standard matchups (excluding play-ins for simplicity)
    for high_seed in range(1, 9):
        low_seed = 17 - high_seed
        high = region_teams[region_teams["SeedNum"] == high_seed]
        low = region_teams[region_teams["SeedNum"] == low_seed]

        if len(high) == 0 or len(low) == 0:
            continue

        high_row = high.iloc[0]
        low_row = low.iloc[0]

        diff = theta_samples[:, high_row["idx"]] - theta_samples[:, low_row["idx"]]
        p_win = norm.cdf(diff / sigma_samples)
        p_mean = p_win.mean()
        p_std = p_win.std()

        label = f"  ({high_seed}) {high_row['TeamName']:<15} vs ({low_seed}) {low_row['TeamName']:<15}"
        print(f"{label}  {p_mean:>13.1%}  ±{p_std:>8.1%}")
