"""Visualization functions for March Madness Bayesian model."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import arviz as az
from pathlib import Path

FIGURES_DIR = Path("blogimages/march-madness-2026")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

az.style.use("arviz-darkgrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})


def save_fig(fig, name: str):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def plot_team_strength_posterior(
    idata: az.InferenceData,
    team_names: np.ndarray,
    top_n: int = 30,
) -> plt.Figure:
    """Ridge plot of posterior team strengths for top N teams."""
    theta = idata.posterior["theta"].values.reshape(-1, len(team_names))
    means = theta.mean(axis=0)
    ranking = np.argsort(-means)[:top_n]

    fig, ax = plt.subplots(figsize=(10, top_n * 0.35))

    for i, idx in enumerate(reversed(ranking)):
        samples = theta[:, idx]
        color = plt.cm.viridis(i / top_n)
        ax.violinplot(
            samples,
            positions=[i],
            vert=False,
            showmedians=True,
            widths=0.8,
        )

    ax.set_yticks(range(top_n))
    ax.set_yticklabels([team_names[ranking[top_n - 1 - i]] for i in range(top_n)])
    ax.set_xlabel("Team Strength (θ)")
    ax.set_title("Posterior Team Strength Distributions (Top 30)")
    fig.tight_layout()
    return fig


def plot_team_strength_forest(
    idata: az.InferenceData,
    team_names: np.ndarray,
    top_n: int = 30,
) -> plt.Figure:
    """Forest plot showing HDI for top N teams."""
    theta = idata.posterior["theta"].values.reshape(-1, len(team_names))
    means = theta.mean(axis=0)
    ranking = np.argsort(-means)[:top_n]

    fig, ax = plt.subplots(figsize=(10, top_n * 0.3))

    for i, idx in enumerate(reversed(ranking)):
        samples = theta[:, idx]
        hdi = az.hdi(samples, hdi_prob=0.94)
        mean = samples.mean()
        ax.plot([hdi[0], hdi[1]], [i, i], color="steelblue", linewidth=2)
        ax.plot(mean, i, "o", color="steelblue", markersize=5)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels([team_names[ranking[top_n - 1 - i]] for i in range(top_n)])
    ax.set_xlabel("Team Strength (θ)")
    ax.set_title("Posterior Team Strength (94% HDI)")
    fig.tight_layout()
    return fig


def plot_conference_effects(idata: az.InferenceData, conf_names: np.ndarray) -> plt.Figure:
    """Forest plot of conference-level effects."""
    mu_conf = idata.posterior["mu_conf"].values.reshape(-1, len(conf_names))
    means = mu_conf.mean(axis=0)
    order = np.argsort(-means)

    fig, ax = plt.subplots(figsize=(10, len(conf_names) * 0.3))

    for i, idx in enumerate(reversed(order)):
        samples = mu_conf[:, idx]
        hdi = az.hdi(samples, hdi_prob=0.94)
        mean = samples.mean()
        ax.plot([hdi[0], hdi[1]], [i, i], color="coral", linewidth=2)
        ax.plot(mean, i, "o", color="coral", markersize=5)

    ax.set_yticks(range(len(conf_names)))
    ax.set_yticklabels([conf_names[order[len(conf_names) - 1 - i]] for i in range(len(conf_names))])
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Conference Effect (μ_conf)")
    ax.set_title("Conference Strength (94% HDI)")
    fig.tight_layout()
    return fig


def plot_advancement_heatmap(advancement: pd.DataFrame, top_n: int = 40) -> plt.Figure:
    """Heatmap of advancement probabilities by team."""
    df = advancement.head(top_n).copy()
    round_cols = [
        "Round of 32",
        "Sweet 16",
        "Elite Eight",
        "Final Four",
        "Championship",
        "Champion",
    ]
    labels = [f"{row['TeamName']} ({row['SeedNum']})" for _, row in df.iterrows()]

    data = df[round_cols].values

    fig, ax = plt.subplots(figsize=(12, top_n * 0.35))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(round_cols)))
    ax.set_xticklabels(round_cols, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if val > 0.005:
                text = f"{val:.0%}" if val >= 0.01 else "<1%"
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Probability", shrink=0.6)
    ax.set_title("Tournament Advancement Probabilities")
    return fig


def plot_championship_odds(advancement: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """Bar chart of championship probabilities."""
    df = advancement.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))

    bars = ax.barh(
        range(top_n),
        df["Champion"].values[::-1],
        color=colors,
    )

    labels = [
        f"{row['TeamName']} ({row['SeedNum']})" for _, row in df.iloc[::-1].iterrows()
    ]
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Championship Probability")
    ax.set_title("Who Wins It All?")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    fig.tight_layout()
    return fig


def plot_bracket(
    advancement: pd.DataFrame,
    bracket_struct: dict,
    most_likely: dict,
) -> plt.Figure:
    """Bracket-style visualization with win probabilities."""
    seeds_df = bracket_struct["seeds_df"]

    # Build per-region bracket display
    regions = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}
    fig, axes = plt.subplots(2, 2, figsize=(20, 24))

    for ax, (region_code, region_name) in zip(axes.flat, regions.items()):
        region_seeds = seeds_df[seeds_df["Region"] == region_code].sort_values("SeedNum")

        # Get advancement data for these teams
        region_data = []
        for _, seed_row in region_seeds.iterrows():
            adv_row = advancement[advancement["TeamID"] == seed_row["TeamID"]]
            if len(adv_row) > 0:
                adv_row = adv_row.iloc[0]
                region_data.append(
                    {
                        "name": seed_row["TeamName"],
                        "seed": seed_row["SeedNum"],
                        "r32": adv_row.get("Round of 32", 0),
                        "s16": adv_row.get("Sweet 16", 0),
                        "e8": adv_row.get("Elite Eight", 0),
                        "f4": adv_row.get("Final Four", 0),
                        "champ": adv_row.get("Champion", 0),
                    }
                )

        # Display as a table within the axes
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(region_data) + 1)
        ax.set_title(f"{region_name} Region", fontsize=14, fontweight="bold")
        ax.axis("off")

        headers = ["Seed", "Team", "R32", "S16", "E8", "F4", "Champ"]
        x_positions = [0.5, 2.0, 4.5, 5.8, 7.0, 8.2, 9.3]

        for col, (header, x) in enumerate(zip(headers, x_positions)):
            ax.text(x, len(region_data) + 0.5, header, fontweight="bold",
                    ha="center", fontsize=9)

        for i, team in enumerate(region_data):
            y = len(region_data) - i - 0.5
            ax.text(x_positions[0], y, str(team["seed"]), ha="center", fontsize=9)
            ax.text(x_positions[1], y, team["name"], ha="center", fontsize=9)
            for j, key in enumerate(["r32", "s16", "e8", "f4", "champ"]):
                val = team[key]
                text = f"{val:.0%}" if val >= 0.01 else "<1%" if val > 0 else "-"
                color = plt.cm.YlOrRd(val)
                ax.text(
                    x_positions[j + 2],
                    y,
                    text,
                    ha="center",
                    fontsize=8,
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=color,
                        alpha=0.6 if val > 0.01 else 0.2,
                    ),
                )

    fig.suptitle(
        "2026 NCAA Tournament Bracket Forecast", fontsize=16, fontweight="bold", y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_calibration(predicted_probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> plt.Figure:
    """Calibration plot: predicted vs actual win rates."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_counts = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (predicted_probs >= lo) & (predicted_probs < hi)
        if mask.sum() > 0:
            bin_means.append(outcomes[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_means.append(np.nan)
            bin_counts.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.plot(bin_centers, bin_means, "o-", color="steelblue", label="Model")
    ax1.fill_between(bin_centers, bin_means, bin_centers, alpha=0.2, color="steelblue")
    ax1.set_xlabel("Predicted Win Probability")
    ax1.set_ylabel("Observed Win Rate")
    ax1.set_title("Calibration Plot (Historical Tournament Validation)")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.bar(bin_centers, bin_counts, width=1 / n_bins * 0.8, color="steelblue", alpha=0.7)
    ax2.set_xlabel("Predicted Win Probability")
    ax2.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_home_court_comparison(
    men_alpha: np.ndarray,
    women_alpha: np.ndarray,
) -> plt.Figure:
    """Overlay men's and women's home court advantage posteriors."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(men_alpha, bins=60, density=True, alpha=0.6, color="steelblue", label=f"Men's ({men_alpha.mean():.2f} ± {men_alpha.std():.2f})")
    ax.hist(women_alpha, bins=60, density=True, alpha=0.6, color="coral", label=f"Women's ({women_alpha.mean():.2f} ± {women_alpha.std():.2f})")
    ax.axvline(men_alpha.mean(), color="steelblue", linestyle="--", alpha=0.8)
    ax.axvline(women_alpha.mean(), color="coral", linestyle="--", alpha=0.8)
    ax.set_xlabel("Home Court Advantage (points)")
    ax.set_ylabel("Density")
    ax.set_title("Home Court Advantage: Men's vs Women's")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_championship_odds_comparison(
    men_adv: pd.DataFrame,
    women_adv: pd.DataFrame,
    top_n: int = 10,
) -> plt.Figure:
    """Side-by-side championship odds for men's and women's."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=False)

    for ax, df, title, color in [
        (ax1, men_adv.head(top_n), "Men's", "steelblue"),
        (ax2, women_adv.head(top_n), "Women's", "coral"),
    ]:
        labels = [f"{row['TeamName']} ({row['SeedNum']})" for _, row in df.iloc[::-1].iterrows()]
        ax.barh(range(top_n), df["Champion"].values[::-1], color=color, alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Championship Probability")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.suptitle("Who Wins It All?", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_upset_probabilities(advancement: pd.DataFrame) -> plt.Figure:
    """Show first-round upset probabilities by seed matchup."""
    regions = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}
    region_colors = {"W": "#1b9e77", "X": "#d95f02", "Y": "#7570b3", "Z": "#e7298a"}
    matchups = [(i, 17 - i) for i in range(1, 9)]

    # Collect all games
    games = []
    for region_code in ["W", "X", "Y", "Z"]:
        region_data = advancement[advancement["Region"] == region_code]
        for high_seed, low_seed in matchups:
            high = region_data[region_data["SeedNum"] == high_seed]
            low = region_data[region_data["SeedNum"] == low_seed]
            if len(high) > 0 and len(low) > 0:
                upset_prob = 1 - high.iloc[0]["Round of 32"]
                games.append({
                    "matchup": f"{low_seed}v{high_seed}",
                    "upset_prob": upset_prob,
                    "region": region_code,
                    "label": f"({low_seed}) {low.iloc[0]['TeamName']} over ({high_seed}) {high.iloc[0]['TeamName']}",
                    "underdog": low.iloc[0]["TeamName"],
                })

    fig, ax = plt.subplots(figsize=(12, 6))

    for region_code, region_name in regions.items():
        region_games = [g for g in games if g["region"] == region_code]
        ax.scatter(
            [g["matchup"] for g in region_games],
            [g["upset_prob"] for g in region_games],
            s=100, alpha=0.8, color=region_colors[region_code],
            label=region_name, edgecolors="white", linewidth=0.5,
        )

    # Annotate notable upsets (P > 20%)
    for g in games:
        if g["upset_prob"] > 0.20:
            ax.annotate(
                g["underdog"], (g["matchup"], g["upset_prob"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7,
                alpha=0.8,
            )

    ax.set_ylabel("Upset Probability")
    ax.set_xlabel("Seed Matchup (underdog v favorite)")
    ax.set_title("First Round Upset Probabilities")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.3, label="Toss-up")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, max(g["upset_prob"] for g in games) + 0.08)
    fig.tight_layout()
    return fig
