"""Historical validation: fit on regular season, predict tournament."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data import build_model_data, load_tournament_results
from src.model import build_bradley_terry, fit_model


def validate_season(
    season: int,
    gender: str = "M",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
) -> dict:
    """Fit model on one season's regular season, predict that tournament."""
    label = "Men's" if gender == "M" else "Women's"
    print(f"\n{'='*60}")
    print(f"Validating {label} season {season}")
    print(f"{'='*60}")

    data = build_model_data(season, gender)
    model = build_bradley_terry(data)
    idata = fit_model(model, draws=draws, tune=tune, chains=chains)

    theta = idata.posterior["theta"].values.reshape(-1, data["n_teams"])
    sigma = idata.posterior["sigma"].values.flatten()

    # Get tournament results
    tourney = load_tournament_results([season], gender)
    if len(tourney) == 0:
        print(f"  No tournament results for {season}")
        return None

    team_id_to_idx = {tid: i for i, tid in enumerate(data["team_ids"])}

    predictions = []
    for _, game in tourney.iterrows():
        w_idx = team_id_to_idx.get(game["WTeamID"])
        l_idx = team_id_to_idx.get(game["LTeamID"])
        if w_idx is None or l_idx is None:
            continue

        # P(winner wins) across posterior samples
        diff = theta[:, w_idx] - theta[:, l_idx]
        p_win = norm.cdf(diff / sigma).mean()
        predictions.append({"predicted_p": p_win, "outcome": 1})

        # Also add the loser's perspective for symmetry
        predictions.append({"predicted_p": 1 - p_win, "outcome": 0})

    return {
        "season": season,
        "predictions": pd.DataFrame(predictions),
        "n_games": len(tourney),
    }


def run_validation(
    seasons: list[int] | None = None,
    gender: str = "M",
    draws: int = 1000,
    tune: int = 1000,
) -> dict:
    """Run validation across multiple historical seasons."""
    if seasons is None:
        if gender == "W":
            seasons = list(range(2015, 2026))
        else:
            seasons = list(range(2015, 2026))
        # Skip 2020 (COVID cancellation)
        seasons = [s for s in seasons if s != 2020]

    all_predictions = []
    season_results = []

    for season in seasons:
        result = validate_season(season, gender=gender, draws=draws, tune=tune)
        if result is not None:
            season_results.append(result)
            all_predictions.append(result["predictions"])

    if not all_predictions:
        return {"seasons": [], "predictions": pd.DataFrame()}

    combined = pd.concat(all_predictions, ignore_index=True)

    # Compute calibration metrics
    y = combined["outcome"].values
    p = combined["predicted_p"].values
    brier = np.mean((p - y) ** 2)
    # Clip to avoid log(0)
    p_clipped = np.clip(p, 1e-15, 1 - 1e-15)
    logloss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))

    print(f"\nOverall validation ({len(season_results)} seasons, {len(combined)//2} games):")
    print(f"  Brier score: {brier:.4f}")
    print(f"  Log loss:    {logloss:.4f}")

    return {
        "seasons": season_results,
        "predictions": combined,
        "brier_score": brier,
        "log_loss": logloss,
    }
