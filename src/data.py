"""Data loading and preparation for March Madness Bayesian model."""

import pandas as pd
import numpy as np
from pathlib import Path

KAGGLE_DIR = Path("data/kaggle")
BARTTORVIK_PATH = Path("data/barttorvik_2026.csv")


def load_teams(gender: str = "M") -> pd.DataFrame:
    return pd.read_csv(KAGGLE_DIR / f"{gender}Teams.csv")


def load_conferences(season: int = 2026, gender: str = "M") -> pd.DataFrame:
    conf = pd.read_csv(KAGGLE_DIR / f"{gender}TeamConferences.csv")
    conf_names = pd.read_csv(KAGGLE_DIR / "Conferences.csv")
    conf = conf[conf["Season"] == season].merge(conf_names, on="ConfAbbrev")
    return conf


def load_regular_season(season: int = 2026, gender: str = "M") -> pd.DataFrame:
    """Load regular season results and reshape to one row per game with margin."""
    rs = pd.read_csv(KAGGLE_DIR / f"{gender}RegularSeasonCompactResults.csv")
    rs = rs[rs["Season"] == season].copy()

    # Each game appears once (winner perspective). Create symmetric representation:
    # team1 = WTeam, team2 = LTeam, margin = WScore - LScore, home indicator for team1
    games = pd.DataFrame(
        {
            "team1": rs["WTeamID"],
            "team2": rs["LTeamID"],
            "score1": rs["WScore"],
            "score2": rs["LScore"],
            "margin": rs["WScore"] - rs["LScore"],
            "loc": rs["WLoc"],  # H=team1 home, A=team1 away, N=neutral
        }
    )
    return games


def load_regular_season_symmetric(season: int = 2026, gender: str = "M") -> pd.DataFrame:
    """Load regular season with each game appearing twice (one per team perspective).

    This is the format needed for the Bradley-Terry model: each row is
    (team_i, team_j, margin_ij, home_i) where margin can be negative.
    """
    rs = pd.read_csv(KAGGLE_DIR / f"{gender}RegularSeasonCompactResults.csv")
    rs = rs[rs["Season"] == season].copy()

    rows = []
    for _, g in rs.iterrows():
        # Winner perspective
        if g["WLoc"] == "H":
            home_w, home_l = 1, -1
        elif g["WLoc"] == "A":
            home_w, home_l = -1, 1
        else:
            home_w, home_l = 0, 0

        rows.append(
            {
                "team_i": g["WTeamID"],
                "team_j": g["LTeamID"],
                "margin": g["WScore"] - g["LScore"],
                "home_i": home_w,
            }
        )
        rows.append(
            {
                "team_i": g["LTeamID"],
                "team_j": g["WTeamID"],
                "margin": g["LScore"] - g["WScore"],
                "home_i": home_l,
            }
        )

    return pd.DataFrame(rows)


def load_seeds(season: int = 2026, gender: str = "M") -> pd.DataFrame:
    """Load tournament seeds with parsed region, seed number, and play-in flag."""
    seeds = pd.read_csv(KAGGLE_DIR / f"{gender}NCAATourneySeeds.csv")
    seeds = seeds[seeds["Season"] == season].copy()
    seeds["Region"] = seeds["Seed"].str[0]
    seeds["SeedNum"] = seeds["Seed"].str[1:3].astype(int)
    seeds["PlayIn"] = seeds["Seed"].str[3:]

    teams = load_teams(gender)
    seeds = seeds.merge(teams[["TeamID", "TeamName"]], on="TeamID")
    return seeds


def load_tournament_results(seasons: list[int] | None = None, gender: str = "M") -> pd.DataFrame:
    """Load historical tournament results for validation."""
    tr = pd.read_csv(KAGGLE_DIR / f"{gender}NCAATourneyCompactResults.csv")
    if seasons is not None:
        tr = tr[tr["Season"].isin(seasons)]
    return tr


def load_slots(season: int = 2026, gender: str = "M") -> pd.DataFrame:
    return pd.read_csv(KAGGLE_DIR / f"{gender}NCAATourneySlots.csv").query(
        f"Season == {season}"
    )


def load_seed_round_slots(gender: str = "M") -> pd.DataFrame:
    path = KAGGLE_DIR / f"{gender}NCAATourneySeedRoundSlots.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_model_data(season: int = 2026, gender: str = "M") -> dict:
    """Build all arrays needed for the PyMC model.

    Returns dict with:
        - team_ids: sorted array of unique team IDs
        - team_names: corresponding team names
        - team_idx_i: index into team_ids for team_i in each game
        - team_idx_j: index into team_ids for team_j in each game
        - margin: score margin (team_i - team_j)
        - home_i: home court indicator (+1 home, -1 away, 0 neutral)
        - conf_idx: conference index for each team
        - conf_names: conference names
        - n_teams: number of teams
        - n_conferences: number of conferences
        - n_games: number of observations
        - gender: "M" or "W"
    """
    games = load_regular_season(season, gender)
    conferences = load_conferences(season, gender)
    teams = load_teams(gender)

    # Get all teams that played this season
    all_team_ids = sorted(
        set(games["team1"].unique()) | set(games["team2"].unique())
    )
    team_id_to_idx = {tid: i for i, tid in enumerate(all_team_ids)}

    team_names = []
    for tid in all_team_ids:
        name = teams.loc[teams["TeamID"] == tid, "TeamName"]
        team_names.append(name.values[0] if len(name) > 0 else str(tid))

    # Map conferences
    conf_map = conferences.set_index("TeamID")["ConfAbbrev"].to_dict()
    team_confs = [conf_map.get(tid, "unknown") for tid in all_team_ids]
    unique_confs = sorted(set(team_confs))
    conf_to_idx = {c: i for i, c in enumerate(unique_confs)}
    conf_idx = np.array([conf_to_idx[c] for c in team_confs])

    # Build game arrays. Assign perspective so team_i = lower ID to avoid
    # always putting the winner first (which would make all margins positive
    # and bias the likelihood).
    rs = pd.read_csv(KAGGLE_DIR / f"{gender}RegularSeasonCompactResults.csv")
    rs = rs[rs["Season"] == season].copy()

    team_idx_i_list = []
    team_idx_j_list = []
    margin_list = []
    home_i_list = []

    for _, g in rs.iterrows():
        w_id, l_id = g["WTeamID"], g["LTeamID"]
        w_score, l_score = g["WScore"], g["LScore"]
        loc = g["WLoc"]

        # team_i = lower ID for consistent, unbiased assignment
        if w_id < l_id:
            ti, tj = w_id, l_id
            m = w_score - l_score
            if loc == "H":
                h = 1.0
            elif loc == "A":
                h = -1.0
            else:
                h = 0.0
        else:
            ti, tj = l_id, w_id
            m = l_score - w_score
            if loc == "H":
                h = -1.0  # winner was home, but team_i is loser
            elif loc == "A":
                h = 1.0  # winner was away, so team_i (loser) was home
            else:
                h = 0.0

        team_idx_i_list.append(team_id_to_idx[ti])
        team_idx_j_list.append(team_id_to_idx[tj])
        margin_list.append(m)
        home_i_list.append(h)

    team_idx_i = np.array(team_idx_i_list)
    team_idx_j = np.array(team_idx_j_list)
    margin = np.array(margin_list, dtype=float)
    home_i = np.array(home_i_list)

    return {
        "team_ids": np.array(all_team_ids),
        "team_names": np.array(team_names),
        "team_idx_i": team_idx_i,
        "team_idx_j": team_idx_j,
        "margin": margin,
        "home_i": home_i,
        "conf_idx": conf_idx,
        "conf_names": np.array(unique_confs),
        "n_teams": len(all_team_ids),
        "n_conferences": len(unique_confs),
        "n_games": len(games),
        "gender": gender,
    }


def build_validation_data(season: int, gender: str = "M") -> dict:
    """Build model data + tournament results for a historical season."""
    model_data = build_model_data(season, gender)

    tourney = load_tournament_results([season], gender)
    tourney_games = []
    for _, g in tourney.iterrows():
        w_idx = np.where(model_data["team_ids"] == g["WTeamID"])[0]
        l_idx = np.where(model_data["team_ids"] == g["LTeamID"])[0]
        if len(w_idx) > 0 and len(l_idx) > 0:
            tourney_games.append(
                {
                    "team_i_idx": w_idx[0],
                    "team_j_idx": l_idx[0],
                    "margin": g["WScore"] - g["LScore"],
                    "winner_idx": w_idx[0],
                }
            )

    model_data["tourney_games"] = tourney_games
    return model_data
