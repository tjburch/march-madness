"""Tests for data pipeline."""

import numpy as np
import pytest
from unittest.mock import patch
import pandas as pd


def _make_fake_season_data():
    """Minimal fake data: 4 teams, 6 games."""
    results = pd.DataFrame({
        "Season": [2026] * 6,
        "WTeamID": [1101, 1102, 1101, 1103, 1102, 1104],
        "LTeamID": [1102, 1103, 1103, 1104, 1104, 1101],
        "WScore": [75, 80, 70, 65, 90, 72],
        "LScore": [60, 70, 65, 55, 80, 68],
        "WLoc": ["H", "A", "N", "H", "N", "A"],
        "NumOT": [0] * 6,
    })
    teams = pd.DataFrame({
        "TeamID": [1101, 1102, 1103, 1104],
        "TeamName": ["Alpha", "Bravo", "Charlie", "Delta"],
    })
    conferences = pd.DataFrame({
        "TeamID": [1101, 1102, 1103, 1104],
        "Season": [2026] * 4,
        "ConfAbbrev": ["big_ten", "big_ten", "sec", "sec"],
    })
    conf_names = pd.DataFrame({
        "ConfAbbrev": ["big_ten", "sec"],
        "Description": ["Big Ten", "SEC"],
    })
    return results, teams, conferences, conf_names


def _mock_read_csv(results, teams, conferences, conf_names):
    """Build a side_effect function for pd.read_csv."""
    def side_effect(path, *args, **kwargs):
        path_str = str(path)
        if "RegularSeasonCompact" in path_str:
            return results
        elif "TeamConferences" in path_str:
            return conferences
        elif "Conferences" in path_str:
            return conf_names
        elif "Teams" in path_str:
            return teams
        raise ValueError(f"Unexpected path: {path_str}")
    return side_effect


@patch("src.data.pd.read_csv")
def test_build_model_data_includes_scores(mock_read_csv):
    """build_model_data should return score_i and score_j arrays."""
    results, teams, conferences, conf_names = _make_fake_season_data()
    mock_read_csv.side_effect = _mock_read_csv(results, teams, conferences, conf_names)

    from src.data import build_model_data
    data = build_model_data(season=2026, gender="M")

    assert "score_i" in data, "Missing score_i"
    assert "score_j" in data, "Missing score_j"
    assert len(data["score_i"]) == data["n_games"]
    assert len(data["score_j"]) == data["n_games"]
    np.testing.assert_array_almost_equal(
        data["margin"], data["score_i"] - data["score_j"]
    )


@patch("src.data.pd.read_csv")
def test_scores_follow_lower_id_first_convention(mock_read_csv):
    """score_i corresponds to the lower-ID team, score_j to the higher-ID team."""
    results, teams, conferences, conf_names = _make_fake_season_data()
    mock_read_csv.side_effect = _mock_read_csv(results, teams, conferences, conf_names)

    from src.data import build_model_data
    data = build_model_data(season=2026, gender="M")

    # First game: 1101 beat 1102 75-60 at home
    # Lower ID = 1101, so score_i=75, score_j=60
    assert data["score_i"][0] == 75.0
    assert data["score_j"][0] == 60.0
