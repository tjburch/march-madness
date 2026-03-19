"""Tests for tournament results ingestion."""

import pytest


def test_build_espn_to_kaggle_map_matches_known_teams():
    """Mapping should resolve ESPN team entries to Kaggle TeamIDs by name."""
    from src.results import build_espn_to_kaggle_map
    import pandas as pd

    seeds_df = pd.DataFrame({
        "TeamID": [1181, 1242, 1301],
        "TeamName": ["Duke", "Houston", "Michigan St"],
        "Seed": ["W01", "W04", "Z11a"],
        "SeedNum": [1, 4, 11],
        "Region": ["W", "W", "Z"],
    })

    espn_teams = [
        {"id": 150, "displayName": "Duke Blue Devils", "shortDisplayName": "Duke", "abbreviation": "DUKE"},
        {"id": 248, "displayName": "Houston Cougars", "shortDisplayName": "Houston", "abbreviation": "HOU"},
        {"id": 127, "displayName": "Michigan State Spartans", "shortDisplayName": "Michigan State", "abbreviation": "MSU"},
    ]

    result = build_espn_to_kaggle_map(seeds_df, espn_teams)

    assert result[150] == 1181  # Duke
    assert result[248] == 1242  # Houston
    assert result[127] == 1301  # Michigan St -> Michigan State


def test_build_espn_to_kaggle_map_warns_on_unmatched(capsys):
    """Unmatched tournament teams should produce warnings."""
    from src.results import build_espn_to_kaggle_map
    import pandas as pd

    seeds_df = pd.DataFrame({
        "TeamID": [9999],
        "TeamName": ["Nonexistent U"],
        "Seed": ["W01"],
        "SeedNum": [1],
        "Region": ["W"],
    })
    espn_teams = []

    result = build_espn_to_kaggle_map(seeds_df, espn_teams)
    captured = capsys.readouterr()
    assert "Nonexistent U" in captured.out
    assert len(result) == 0


def test_fetch_espn_results_parses_completed_games(monkeypatch):
    """fetch_espn_results should parse ESPN scoreboard JSON into game dicts."""
    from src.results import fetch_espn_results

    fake_response = {
        "events": [
            {
                "id": "401856434",
                "competitions": [{
                    "competitors": [
                        {
                            "id": "152",
                            "winner": False,
                            "team": {"id": "152", "displayName": "NC State Wolfpack"},
                            "score": "66",
                        },
                        {
                            "id": "251",
                            "winner": True,
                            "team": {"id": "251", "displayName": "Texas Longhorns"},
                            "score": "68",
                        },
                    ]
                }],
                "status": {
                    "type": {"completed": True}
                },
            },
            {
                "id": "401856435",
                "competitions": [{
                    "competitors": [
                        {"id": "10", "winner": False, "team": {"id": "10", "displayName": "A"}, "score": "50"},
                        {"id": "20", "winner": False, "team": {"id": "20", "displayName": "B"}, "score": "50"},
                    ]
                }],
                "status": {
                    "type": {"completed": False}
                },
            },
        ]
    }

    import json, io

    class FakeResponse:
        def read(self):
            return json.dumps(fake_response).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, **kw: FakeResponse(),
    )

    games = fetch_espn_results("M")

    # Only completed game should be returned
    assert len(games) == 1
    g = games[0]
    assert g["espn_game_id"] == "401856434"
    assert g["winner_espn_id"] == 251
    assert g["score_a"] == 66
    assert g["score_b"] == 68


def test_fetch_espn_results_returns_empty_on_failure(monkeypatch):
    """Network failure should return empty list, not raise."""
    from src.results import fetch_espn_results

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, **kw: (_ for _ in ()).throw(OSError("no network")),
    )

    games = fetch_espn_results("M")
    assert games == []


def test_fetch_espn_teams_parses_team_list(monkeypatch):
    """fetch_espn_teams should extract id, displayName, shortDisplayName, abbreviation."""
    from src.results import fetch_espn_teams

    fake_response = {
        "sports": [{
            "leagues": [{
                "teams": [
                    {"team": {"id": "150", "displayName": "Duke Blue Devils",
                              "shortDisplayName": "Duke", "abbreviation": "DUKE"}},
                    {"team": {"id": "248", "displayName": "Houston Cougars",
                              "shortDisplayName": "Houston", "abbreviation": "HOU"}},
                ]
            }]
        }]
    }

    import json

    class FakeResponse:
        def read(self):
            return json.dumps(fake_response).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr("urllib.request.urlopen", lambda req, **kw: FakeResponse())

    teams = fetch_espn_teams("M")
    assert len(teams) == 2
    assert teams[0]["id"] == 150
    assert teams[0]["displayName"] == "Duke Blue Devils"
    assert teams[1]["abbreviation"] == "HOU"
