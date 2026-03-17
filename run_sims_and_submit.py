"""Load fitted models and run simulations + submission generation.

Runs each gender in a subprocess to control memory.
"""

import subprocess
import sys


def main():
    # Step 1: Run men's sims
    print("Running men's simulation...")
    r = subprocess.run(
        [sys.executable, "-c", """
import json, numpy as np, arviz as az, gc
from src.data import build_model_data
from src.model import get_team_strengths
from src.simulate import build_bracket_structure, simulate_tournament, tail_analysis

data = build_model_data(2026, "M")
print(f"  {data['n_teams']} teams, {data['n_conferences']} conferences, {data['n_games']} games")

idata = az.from_netcdf("results/model_2026_mens.nc")
theta = idata.posterior["theta"].values.reshape(-1, data["n_teams"])
sigma = idata.posterior["sigma"].values.flatten()

# Save arrays for submission later
np.savez_compressed("results/mens_posterior.npz", theta=theta, sigma=sigma, team_ids=data["team_ids"])

strengths = get_team_strengths(idata, data)
print("\\nTop 10 Men's Teams:")
for rank, idx in enumerate(strengths["ranking"][:10], 1):
    print(f"  {rank}. {strengths['team_names'][idx]:<20s} theta={strengths['means'][idx]:.1f}")

del idata, strengths; gc.collect()

bracket = build_bracket_structure(2026, "M")
sim = simulate_tournament(bracket, theta, sigma, data["team_ids"], n_sims=10000, seed=42)

sim["advancement"].to_csv("results/advancement_probs_mens.csv", index=False)
with open("results/champions_mens.json", "w") as f:
    json.dump(sim["champions"], f, indent=2)

print("\\nMen's Championship Probabilities:")
for _, row in sim["advancement"].head(10).iterrows():
    print(f"  {row['TeamName']:<25} {row['Champion']:.1%}")

tails = tail_analysis(sim)
print(f"\\nP(zero #1 seeds in FF): {tails['no_one_seeds_final_four_pct']:.1%}")
print("\\nMen's simulation complete.")
"""],
        cwd="/Users/tburch/Developer/march_madness",
    )
    if r.returncode != 0:
        print(f"Men's failed with code {r.returncode}")
        return

    # Step 2: Run women's sims
    print("\nRunning women's simulation...")
    r = subprocess.run(
        [sys.executable, "-c", """
import json, numpy as np, arviz as az, gc
from src.data import build_model_data
from src.model import get_team_strengths
from src.simulate import build_bracket_structure, simulate_tournament, tail_analysis

data = build_model_data(2026, "W")
print(f"  {data['n_teams']} teams, {data['n_conferences']} conferences, {data['n_games']} games")

idata = az.from_netcdf("results/model_2026_womens.nc")
theta = idata.posterior["theta"].values.reshape(-1, data["n_teams"])
sigma = idata.posterior["sigma"].values.flatten()
alpha = idata.posterior["alpha"].values.flatten()

np.savez_compressed("results/womens_posterior.npz", theta=theta, sigma=sigma, team_ids=data["team_ids"])

strengths = get_team_strengths(idata, data)
print("\\nTop 10 Women's Teams:")
for rank, idx in enumerate(strengths["ranking"][:10], 1):
    print(f"  {rank}. {strengths['team_names'][idx]:<20s} theta={strengths['means'][idx]:.1f}")

del idata, strengths; gc.collect()

bracket = build_bracket_structure(2026, "W")
sim = simulate_tournament(bracket, theta, sigma, data["team_ids"], n_sims=10000, seed=42, alpha_samples=alpha)

sim["advancement"].to_csv("results/advancement_probs_womens.csv", index=False)
with open("results/champions_womens.json", "w") as f:
    json.dump(sim["champions"], f, indent=2)

print("\\nWomen's Championship Probabilities:")
for _, row in sim["advancement"].head(10).iterrows():
    print(f"  {row['TeamName']:<25} {row['Champion']:.1%}")

tails = tail_analysis(sim)
print(f"\\nP(zero #1 seeds in FF): {tails['no_one_seeds_final_four_pct']:.1%}")
print("\\nWomen's simulation complete.")
"""],
        cwd="/Users/tburch/Developer/march_madness",
    )
    if r.returncode != 0:
        print(f"Women's failed with code {r.returncode}")
        return

    # Step 3: Generate submission from saved numpy arrays
    print("\nGenerating Kaggle submission...")
    r = subprocess.run(
        [sys.executable, "-c", """
import numpy as np
from src.submission import generate_submission

m = np.load("results/mens_posterior.npz")
w = np.load("results/womens_posterior.npz")

generate_submission(
    men_theta=m["theta"], men_sigma=m["sigma"], men_team_ids=m["team_ids"],
    women_theta=w["theta"], women_sigma=w["sigma"], women_team_ids=w["team_ids"],
)
"""],
        cwd="/Users/tburch/Developer/march_madness",
    )
    if r.returncode != 0:
        print(f"Submission failed with code {r.returncode}")
        return

    print("\nDone!")


if __name__ == "__main__":
    main()
