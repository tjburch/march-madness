"""Tests for export module."""


def test_mark_eliminated_play_in_loser():
    """Play-in loser should be marked eliminated with 'First Four' round."""
    from src.export import _mark_eliminated

    teams = {
        "1250": {"eliminated": False, "eliminated_round": None},
        "1341": {"eliminated": False, "eliminated_round": None},
    }
    actual_results = {
        "X16": {"winner": 1250, "loser": 1341, "winner_score": 72, "loser_score": 65},
    }
    bracket_struct = {"seed_to_team": {}, "regular_slots": {}, "play_in_slots": {}}

    _mark_eliminated(teams, actual_results, bracket_struct)

    assert teams["1341"]["eliminated"] is True
    assert teams["1341"]["eliminated_round"] == "First Four"
    assert teams["1250"]["eliminated"] is False


def test_mark_eliminated_r1_loser():
    """R1 loser should be marked eliminated with 'Round of 64'."""
    from src.export import _mark_eliminated

    teams = {
        "1181": {"eliminated": False, "eliminated_round": None},
        "1373": {"eliminated": False, "eliminated_round": None},
    }
    actual_results = {
        "R1W1": {"winner": 1181, "loser": 1373, "winner_score": 85, "loser_score": 60},
    }
    bracket_struct = {"seed_to_team": {}, "regular_slots": {}, "play_in_slots": {}}

    _mark_eliminated(teams, actual_results, bracket_struct)

    assert teams["1373"]["eliminated"] is True
    assert teams["1373"]["eliminated_round"] == "Round of 64"


def test_mark_eliminated_r2_loser():
    """R2 loser should be marked eliminated. This tests the R2+ fix."""
    from src.export import _mark_eliminated

    teams = {
        "1181": {"eliminated": False, "eliminated_round": None},
        "1395": {"eliminated": False, "eliminated_round": None},
    }
    actual_results = {
        "R2W1": {"winner": 1181, "loser": 1395, "winner_score": 78, "loser_score": 71},
    }
    bracket_struct = {
        "seed_to_team": {},
        "regular_slots": {"R2W1": ("R1W1", "R1W8")},
        "play_in_slots": {},
    }

    _mark_eliminated(teams, actual_results, bracket_struct)

    assert teams["1395"]["eliminated"] is True
    assert teams["1395"]["eliminated_round"] == "Round of 32"
    assert teams["1181"]["eliminated"] is False
