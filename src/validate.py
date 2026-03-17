"""Historical validation: fit on regular season, predict tournament."""

import gc
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
    model_type: str = "bradley_terry",
) -> dict:
    """Fit model on one season's regular season, predict that tournament.

    Args:
        model_type: "bradley_terry" or "offense_defense"
    """
    label = "Men's" if gender == "M" else "Women's"
    print(f"\n{'='*60}")
    print(f"Validating {label} season {season} ({model_type})")
    print(f"{'='*60}")

    data = build_model_data(season, gender)

    if model_type == "offense_defense":
        from src.model import build_offense_defense_model
        model = build_offense_defense_model(data)
    else:
        model = build_bradley_terry(data)

    idata = fit_model(model, draws=draws, tune=tune, chains=chains)

    sigma = idata.posterior["sigma"].values.flatten()
    team_id_to_idx = {tid: i for i, tid in enumerate(data["team_ids"])}

    if model_type == "offense_defense":
        off = idata.posterior["off"].values.reshape(-1, data["n_teams"])
        deff = idata.posterior["def"].values.reshape(-1, data["n_teams"])
    else:
        theta = idata.posterior["theta"].values.reshape(-1, data["n_teams"])

    tourney = load_tournament_results([season], gender)
    if len(tourney) == 0:
        print(f"  No tournament results for {season}")
        return None

    predictions = []
    for _, game in tourney.iterrows():
        w_idx = team_id_to_idx.get(game["WTeamID"])
        l_idx = team_id_to_idx.get(game["LTeamID"])
        if w_idx is None or l_idx is None:
            continue

        if model_type == "offense_defense":
            diff = (off[:, w_idx] - deff[:, l_idx]) - (off[:, l_idx] - deff[:, w_idx])
            p_win = norm.cdf(diff / (sigma * np.sqrt(2))).mean()
        else:
            diff = theta[:, w_idx] - theta[:, l_idx]
            p_win = norm.cdf(diff / sigma).mean()

        predictions.append({"predicted_p": p_win, "outcome": 1})
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
    model_type: str = "bradley_terry",
) -> dict:
    """Run validation across multiple historical seasons."""
    if seasons is None:
        if gender == "W":
            seasons = list(range(2015, 2026))
        else:
            seasons = list(range(2015, 2026))
        seasons = [s for s in seasons if s != 2020]

    all_predictions = []
    season_results = []

    for season in seasons:
        result = validate_season(
            season, gender=gender, draws=draws, tune=tune,
            model_type=model_type,
        )
        if result is not None:
            season_results.append(result)
            all_predictions.append(result["predictions"])
        gc.collect()

    if not all_predictions:
        return {"seasons": [], "predictions": pd.DataFrame()}

    combined = pd.concat(all_predictions, ignore_index=True)

    y = combined["outcome"].values
    p = combined["predicted_p"].values
    brier = np.mean((p - y) ** 2)
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
