"""Historical validation of the Bradley-Terry model."""

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

from src.validate import run_validation
from src.visualize import plot_calibration, save_fig

az.style.use("arviz-darkgrid")

# %% [markdown]
# ## Validate on Historical Seasons (2015-2025)
#
# For each season, we fit the model on that season's regular season data,
# then predict that year's tournament games. We skip 2020 (COVID cancellation).

# %%
validation = run_validation(
    seasons=[2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
    draws=1000,
    tune=1000,
)

# %%
print(f"Brier Score: {validation['brier_score']:.4f}")
print(f"Log Loss: {validation['log_loss']:.4f}")
print(f"Total predictions: {len(validation['predictions']) // 2} games")

# %% [markdown]
# ## Calibration Plot

# %%
preds = validation["predictions"]
fig = plot_calibration(preds["predicted_p"].values, preds["outcome"].values, n_bins=10)
save_fig(fig, "calibration")

# %% [markdown]
# ## Accuracy by Seed Matchup
#
# How well does the model predict upsets at each seed-line matchup?

# %%
# Per-season summary
for result in validation["seasons"]:
    season = result["season"]
    p = result["predictions"]
    winners = p[p["outcome"] == 1]
    correct = (winners["predicted_p"] > 0.5).mean()
    print(f"  {season}: {result['n_games']} games, accuracy = {correct:.1%}")
