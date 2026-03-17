"""Exploratory data analysis for March Madness 2026."""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, "..")

from src.data import (
    load_regular_season,
    load_seeds,
    load_conferences,
    load_teams,
    build_model_data,
)

sns.set_style("darkgrid")
plt.rcParams.update({"figure.dpi": 150})

# %% [markdown]
# ## 2026 Regular Season Data

# %%
games = load_regular_season(2026)
print(f"Total games: {len(games)}")
print(f"Score margin range: [{games['margin'].min()}, {games['margin'].max()}]")
print(f"\nMargin distribution:")
print(games["margin"].describe())

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(games["margin"], bins=50, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Score Margin (Winner - Loser)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Score Margins (2026 Regular Season)")
axes[0].axvline(games["margin"].mean(), color="red", linestyle="--", label=f"Mean: {games['margin'].mean():.1f}")
axes[0].legend()

# Home/away/neutral breakdown
loc_counts = games["loc"].value_counts()
axes[1].bar(loc_counts.index, loc_counts.values)
axes[1].set_xlabel("Game Location")
axes[1].set_ylabel("Count")
axes[1].set_title("Game Location Distribution")
for i, (loc, count) in enumerate(loc_counts.items()):
    axes[1].text(i, count + 20, str(count), ha="center")

plt.tight_layout()
plt.savefig("../blogimages/march-madness-2026/eda_margins.png", bbox_inches="tight")
plt.show()

# %%
# Home court advantage in the raw data
home_wins = len(games[games["loc"] == "H"])
away_wins = len(games[games["loc"] == "A"])
neutral_games = len(games[games["loc"] == "N"])
print(f"Home wins: {home_wins} ({home_wins/(home_wins+away_wins)*100:.1f}%)")
print(f"Away wins: {away_wins} ({away_wins/(home_wins+away_wins)*100:.1f}%)")
print(f"Neutral site games: {neutral_games}")

home_margin = games[games["loc"] == "H"]["margin"].mean()
away_margin = games[games["loc"] == "A"]["margin"].mean()
print(f"\nMean margin when home team wins: {home_margin:.1f}")
print(f"Mean margin when away team wins: {away_margin:.1f}")

# %% [markdown]
# ## Tournament Seeds

# %%
seeds = load_seeds(2026)
print(f"Tournament teams: {len(seeds)}")
print(f"\nTeams by region:")
for region in ["W", "X", "Y", "Z"]:
    region_teams = seeds[seeds["Region"] == region].sort_values("SeedNum")
    region_name = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}[region]
    print(f"\n{region_name}:")
    for _, row in region_teams.iterrows():
        play_in = " (Play-in)" if row["PlayIn"] else ""
        print(f"  {row['SeedNum']:2d}. {row['TeamName']}{play_in}")

# %% [markdown]
# ## Conference Breakdown

# %%
conferences = load_conferences(2026)
tourney_teams = seeds.merge(conferences[["TeamID", "ConfAbbrev", "Description"]], on="TeamID")

conf_counts = tourney_teams["Description"].value_counts()
print("Tournament teams by conference:")
print(conf_counts.to_string())

# %%
fig, ax = plt.subplots(figsize=(12, 6))
conf_counts.head(15).plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlabel("Number of Tournament Teams")
ax.set_title("Conference Representation in 2026 Tournament")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("../blogimages/march-madness-2026/eda_conferences.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Model Data Summary

# %%
data = build_model_data(2026)
print(f"Teams: {data['n_teams']}")
print(f"Conferences: {data['n_conferences']}")
print(f"Games (observations): {data['n_games']}")
print(f"Games per team (avg): {data['n_games'] * 2 / data['n_teams']:.1f}")
