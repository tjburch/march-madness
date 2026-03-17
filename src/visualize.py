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

CONF_DISPLAY_NAMES = {
    "sec": "SEC",
    "big_twelve": "Big 12",
    "big_ten": "Big Ten",
    "acc": "ACC",
    "big_east": "Big East",
    "mwc": "Mountain West",
    "wcc": "WCC",
    "a_ten": "A-10",
    "mvc": "Missouri Valley",
    "aac": "AAC",
    "wac": "WAC",
    "big_west": "Big West",
    "cusa": "C-USA",
    "big_sky": "Big Sky",
    "ivy": "Ivy",
    "caa": "CAA",
    "mac": "MAC",
    "southland": "Southland",
    "horizon": "Horizon",
    "summit": "Summit",
    "sun_belt": "Sun Belt",
    "big_south": "Big South",
    "southern": "Southern",
    "a_sun": "ASUN",
    "maac": "MAAC",
    "patriot": "Patriot",
    "ovc": "OVC",
    "nec": "NEC",
    "aec": "America East",
    "swac": "SWAC",
    "meac": "MEAC",
}


def _format_conf_name(abbrev: str) -> str:
    return CONF_DISPLAY_NAMES.get(abbrev, abbrev.upper())


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
    """Forest plot showing HDI for top N teams.

    Auto-detects margin-based (theta) vs score-based (off/def) models.
    """
    posterior_vars = list(idata.posterior.data_vars)

    if "off" in posterior_vars:
        off = idata.posterior["off"].values.reshape(-1, len(team_names))
        deff = idata.posterior["def"].values.reshape(-1, len(team_names))
        overall = off + deff
        xlabel = "Overall Strength (off + def)"
    else:
        overall = idata.posterior["theta"].values.reshape(-1, len(team_names))
        xlabel = "Team Strength (θ)"

    means = overall.mean(axis=0)
    ranking = np.argsort(-means)[:top_n]

    fig, ax = plt.subplots(figsize=(10, top_n * 0.3))

    for i, idx in enumerate(reversed(ranking)):
        samples = overall[:, idx]
        hdi = az.hdi(samples, hdi_prob=0.94)
        mean = samples.mean()
        ax.plot([hdi[0], hdi[1]], [i, i], color="steelblue", linewidth=2)
        ax.plot(mean, i, "o", color="steelblue", markersize=5)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels([team_names[ranking[top_n - 1 - i]] for i in range(top_n)])
    ax.set_xlabel(xlabel)
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
    ax.set_yticklabels([
        _format_conf_name(conf_names[order[len(conf_names) - 1 - i]])
        for i in range(len(conf_names))
    ])
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

    data = pd.DataFrame(df[round_cols].values, index=labels, columns=round_cols)

    fig, ax = plt.subplots(figsize=(12, top_n * 0.35))
    sns.heatmap(
        data,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0,
        linecolor="none",
        annot=True,
        fmt=".0%",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Probability", "shrink": 0.6},
    )

    # Fix annotation text: hide near-zero values, set contrast colors
    for text_obj in ax.texts:
        val_str = text_obj.get_text()
        try:
            val = float(val_str.strip("%")) / 100
        except ValueError:
            continue
        if val < 0.005:
            text_obj.set_text("")
        elif val < 0.01:
            text_obj.set_text("<1%")
            text_obj.set_color("black")
        else:
            text_obj.set_color("white" if val > 0.5 else "black")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("Tournament Advancement Probabilities")
    return fig


def plot_championship_odds(advancement: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """Bar chart of championship probabilities."""
    df = advancement.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        range(top_n),
        df["Champion"].values[::-1],
        color="steelblue",
        edgecolor="none",
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
    """Bracket-style visualization with win probabilities.

    Uses a vertical stack (4 rows, 1 col) so each region gets the full
    figure width, making all text large and readable.
    """
    seeds_df = bracket_struct["seeds_df"]

    regions = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}
    fig, axes = plt.subplots(4, 1, figsize=(18, 52))

    for ax, (region_code, region_name) in zip(axes.flat, regions.items()):
        region_seeds = seeds_df[seeds_df["Region"] == region_code].sort_values("SeedNum")

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

        n_teams = len(region_data)
        ax.set_xlim(-0.2, 10.2)
        ax.set_ylim(-0.5, n_teams + 1.2)
        ax.set_title(f"{region_name} Region", fontsize=20, fontweight="bold", pad=14)
        ax.axis("off")

        headers = ["Seed", "Team", "R32", "S16", "E8", "F4", "Champ"]
        x_positions = [0.3, 2.5, 5.2, 6.3, 7.4, 8.5, 9.6]

        for header, x in zip(headers, x_positions):
            ax.text(
                x, n_teams + 0.5, header,
                fontweight="bold", ha="center", fontsize=16,
            )

        # Divider line under headers
        ax.plot([0, 10.2], [n_teams + 0.05, n_teams + 0.05],
                color="gray", linewidth=0.5, alpha=0.4)

        for i, team in enumerate(region_data):
            y = n_teams - i - 0.5

            ax.text(x_positions[0], y, str(team["seed"]),
                    ha="center", va="center", fontsize=15, fontweight="bold")
            ax.text(x_positions[1], y, team["name"],
                    ha="center", va="center", fontsize=15)

            for j, key in enumerate(["r32", "s16", "e8", "f4", "champ"]):
                val = team[key]
                text = f"{val:.0%}" if val >= 0.01 else "<1%" if val > 0 else "-"
                facecolor = plt.cm.YlOrRd(val)
                text_color = "white" if val > 0.5 else "black"
                ax.text(
                    x_positions[j + 2],
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold" if val >= 0.5 else "normal",
                    color=text_color,
                    bbox=dict(
                        boxstyle="round,pad=0.35",
                        facecolor=facecolor,
                        alpha=0.7 if val > 0.01 else 0.2,
                        edgecolor="none",
                    ),
                )

    fig.suptitle(
        "2026 NCAA Tournament Bracket Forecast", fontsize=24, fontweight="bold", y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
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


def plot_off_def_scatter(
    strengths: dict,
    conf_idx: np.ndarray,
    conf_names: np.ndarray,
    top_n_label: int = 15,
) -> plt.Figure:
    """Offense vs defense posterior means, colored by conference."""
    off_means = strengths["off_means"]
    def_means = strengths["def_means"]
    team_names = strengths["team_names"]
    overall = strengths["overall_means"]
    ranking = np.argsort(-overall)

    top_confs = ["sec", "big_twelve", "big_ten", "acc", "big_east"]
    conf_colors = {
        "sec": "#e41a1c",
        "big_twelve": "#377eb8",
        "big_ten": "#4daf4a",
        "acc": "#984ea3",
        "big_east": "#ff7f00",
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(len(off_means)):
        conf = conf_names[conf_idx[i]]
        if conf not in top_confs:
            ax.scatter(off_means[i], def_means[i], c="lightgray", s=20, alpha=0.5, zorder=1)

    for conf in top_confs:
        mask = np.array([conf_names[conf_idx[i]] == conf for i in range(len(off_means))])
        if mask.any():
            ax.scatter(
                off_means[mask], def_means[mask],
                c=conf_colors[conf], s=50, alpha=0.8, zorder=2,
                label=_format_conf_name(conf),
                edgecolors="white", linewidth=0.5,
            )

    for rank_pos in range(min(top_n_label, len(ranking))):
        idx = ranking[rank_pos]
        ax.annotate(
            team_names[idx],
            (off_means[idx], def_means[idx]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=7, alpha=0.9,
        )

    if strengths.get("off_def_corr") is not None:
        corr_mean = strengths["off_def_corr"].mean()
        ax.text(
            0.02, 0.98,
            f"Posterior correlation: {corr_mean:.2f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Offensive Strength (higher = more points scored)")
    ax.set_ylabel("Defensive Strength (higher = fewer points allowed)")
    ax.set_title("Offense vs Defense: Who's Good at What?")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_off_def_rankings(
    strengths: dict,
    top_n: int = 20,
) -> plt.Figure:
    """Forest plots for top teams by offense and defense separately."""
    off_samples = strengths["off_samples"]
    def_samples = strengths["def_samples"]
    team_names = strengths["team_names"]

    off_means = off_samples.mean(axis=0)
    def_means = def_samples.mean(axis=0)
    off_ranking = np.argsort(-off_means)[:top_n]
    def_ranking = np.argsort(-def_means)[:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, top_n * 0.3))

    for ax, ranking, samples, title, color in [
        (ax1, off_ranking, off_samples, "Offensive Strength", "coral"),
        (ax2, def_ranking, def_samples, "Defensive Strength", "steelblue"),
    ]:
        for i, idx in enumerate(reversed(ranking)):
            s = samples[:, idx]
            hdi = az.hdi(s, hdi_prob=0.94)
            mean = s.mean()
            ax.plot([hdi[0], hdi[1]], [i, i], color=color, linewidth=2)
            ax.plot(mean, i, "o", color=color, markersize=5)

        ax.set_yticks(range(top_n))
        ax.set_yticklabels([team_names[ranking[top_n - 1 - i]] for i in range(top_n)])
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel(title)
        ax.set_title(f"Top {top_n} by {title} (94% HDI)")

    fig.tight_layout()
    return fig


def plot_conference_off_def(
    idata: az.InferenceData,
    conf_names: np.ndarray,
) -> plt.Figure:
    """Conference offense vs defense effects as a 2D scatter."""
    mu_off = idata.posterior["mu_off_conf"].values.reshape(-1, len(conf_names))
    mu_def = idata.posterior["mu_def_conf"].values.reshape(-1, len(conf_names))

    off_means = mu_off.mean(axis=0)
    def_means = mu_def.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(off_means, def_means, s=60, c="steelblue", edgecolors="white", zorder=2)
    for i, name in enumerate(conf_names):
        ax.annotate(
            _format_conf_name(name),
            (off_means[i], def_means[i]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=8,
        )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Conference Offensive Effect")
    ax.set_ylabel("Conference Defensive Effect")
    ax.set_title("Conference Strength: Offense vs Defense")
    fig.tight_layout()
    return fig


def plot_loo_comparison(comparison: pd.DataFrame) -> plt.Figure:
    """Plot ELPD with error bars from az.compare()."""
    fig, ax = plt.subplots(figsize=(8, 4))
    az.plot_compare(comparison, ax=ax)
    ax.set_title("Model Comparison: Gaussian vs Student-t Likelihood (LOO-CV)")
    fig.tight_layout()
    return fig


def plot_posterior_predictive_scores(
    idata: az.InferenceData,
    observed_scores: np.ndarray,
) -> plt.Figure:
    """Predicted score distributions overlaid on observed score histogram."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        observed_scores, bins=40, density=True, alpha=0.6,
        color="steelblue", label="Observed scores", edgecolor="none",
    )

    if hasattr(idata, "posterior_predictive"):
        pp_key = "score_i" if "score_i" in idata.posterior_predictive else None
        if pp_key:
            pp_scores = idata.posterior_predictive[pp_key].values.flatten()
            ax.hist(
                pp_scores, bins=40, density=True, alpha=0.4,
                color="coral", label="Posterior predictive", edgecolor="none",
            )

    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive Check: Score Distribution")
    ax.legend()
    fig.tight_layout()
    return fig
