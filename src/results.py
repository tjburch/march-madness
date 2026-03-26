"""Fetch actual tournament results from ESPN and map to bracket slots."""

import csv
import json
import re
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.data import load_seeds
from src.simulate import build_bracket_structure

KAGGLE_DIR = Path("data/kaggle")

# Teams missing from ESPN's /teams directory (newer D1 programs, etc.)
# Maps Kaggle TeamID -> ESPN team ID directly.
_MANUAL_ESPN_IDS = {
    1274: 2390,   # Miami FL (ESPN "Miami" is ambiguous with Miami OH)
    1474: 2511,   # Queens NC (not in ESPN team directory)
    3474: 2511,   # Queens NC (women's)
}

# Tournament start dates by (season, gender)
_TOURNAMENT_START = {
    (2026, "M"): date(2026, 3, 17),
    (2026, "W"): date(2026, 3, 18),
}


def _espn_sport_path(gender: str) -> str:
    return "mens-college-basketball" if gender == "M" else "womens-college-basketball"


def _load_spellings(gender: str) -> dict[str, int]:
    """Load Kaggle team spellings file as normalized_name -> TeamID lookup."""
    prefix = "M" if gender == "M" else "W"
    path = KAGGLE_DIR / f"{prefix}TeamSpellings.csv"
    lookup = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row["TeamNameSpelling"].lower().strip()] = int(row["TeamID"])
    return lookup


# Regex to strip common mascot/nickname suffixes from ESPN display names.
_MASCOT_RE = re.compile(
    r"\s+(?:blue devils|red storm|red raiders|fighting irish|yellow jackets"
    r"|golden eagles|mountaineers|cornhuskers|wolverines|spartans"
    r"|cyclones|hurricanes|hoosiers|hawkeyes|boilermakers|bulldogs"
    r"|wildcats|huskies|bears|tigers|eagles|hawks|panthers|cavaliers"
    r"|knights|warriors|cougars|rebels|aggies|longhorns|cowboys"
    r"|cardinals|owls|lancers|roadrunners|gaels|billikens|jaguars"
    r"|phoenix|sharks|bison|royals|terrapins|gators|seminoles"
    r"|wolfpack|volunteers|crimson tide|demon deacons|tar heels"
    r"|jayhawks|sooners|pirates|musketeers|friars|flyers|dons"
    r"|shockers|commodores|deacons|orange|buckeyes|nittany lions"
    r"|red raiders|redhawks|devilettes|lady bears|ladyjacks"
    r"|ragin cajuns|ramblers|bearcats|braves|rams|salukis"
    r"|spiders|colonials|monarchs|dukes|paladins|terriers"
    r"|[\w]+)$",
    re.IGNORECASE,
)


def build_espn_to_kaggle_map(
    seeds_df,
    espn_teams: list[dict],
    gender: str = "M",
) -> dict[int, int]:
    """Build ESPN team ID -> Kaggle TeamID mapping for tournament teams.

    Uses the Kaggle TeamSpellings file for robust name matching instead of
    fragile substring matching.

    Args:
        seeds_df: DataFrame with TeamID, TeamName columns (from load_seeds).
        espn_teams: List of ESPN team dicts with keys: id, displayName,
            shortDisplayName, abbreviation.
        gender: "M" or "W" to select the correct spellings file.

    Returns:
        Dict mapping ESPN team ID (int) to Kaggle TeamID (int).
    """
    spellings = _load_spellings(gender)
    tournament_ids = set(seeds_df["TeamID"].astype(int))

    # For each ESPN team, try to resolve a Kaggle ID via spellings
    espn_id_to_kaggle = {}
    for team in espn_teams:
        espn_id = int(team["id"])

        # Try each ESPN name variant against spellings
        for key in ["shortDisplayName", "displayName", "abbreviation"]:
            name = team.get(key, "").lower().strip()
            if not name:
                continue

            kaggle_id = spellings.get(name)
            if kaggle_id and kaggle_id in tournament_ids:
                espn_id_to_kaggle[espn_id] = kaggle_id
                break

            # Strip mascot suffix and retry
            stripped = _MASCOT_RE.sub("", name).strip()
            if stripped and stripped != name:
                kaggle_id = spellings.get(stripped)
                if kaggle_id and kaggle_id in tournament_ids:
                    espn_id_to_kaggle[espn_id] = kaggle_id
                    break

    # Add manual ESPN IDs for teams not in ESPN's directory
    for kaggle_id, espn_id in _MANUAL_ESPN_IDS.items():
        if kaggle_id in tournament_ids:
            espn_id_to_kaggle[espn_id] = kaggle_id

    # Report any tournament teams that couldn't be mapped
    mapped_kaggle = set(espn_id_to_kaggle.values())
    for _, row in seeds_df.iterrows():
        kaggle_id = int(row["TeamID"])
        if kaggle_id not in mapped_kaggle:
            print(f"Warning: No ESPN match for {row['TeamName']} (Kaggle ID {kaggle_id})")

    return espn_id_to_kaggle


def fetch_espn_results(gender: str, season: int = 2026) -> list[dict]:
    """Fetch completed tournament games from ESPN scoreboard API.

    Returns list of dicts with keys: espn_game_id, team_a_espn_id,
    team_b_espn_id, score_a, score_b, winner_espn_id.

    Returns empty list on any network/parsing failure.
    """
    start = _TOURNAMENT_START.get((season, gender))
    if start is None:
        print(f"Warning: No tournament start date for {season} {gender}")
        return []

    sport = _espn_sport_path(gender)
    today = datetime.now(timezone.utc).date()
    games = []
    seen_ids = set()

    current = start
    while current <= today:
        date_str = current.strftime("%Y%m%d")
        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
            f"{sport}/scoreboard?dates={date_str}&groups=100&limit=100"
        )
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())

            for event in data.get("events", []):
                status = event.get("status", {}).get("type", {})
                if not status.get("completed", False):
                    continue

                comp = event["competitions"][0]
                competitors = comp["competitors"]
                if len(competitors) != 2:
                    continue

                c0, c1 = competitors[0], competitors[1]
                game_id = event["id"]
                if game_id in seen_ids:
                    continue
                seen_ids.add(game_id)
                game = {
                    "espn_game_id": game_id,
                    "team_a_espn_id": int(c0["team"]["id"]),
                    "team_b_espn_id": int(c1["team"]["id"]),
                    "score_a": int(c0["score"]),
                    "score_b": int(c1["score"]),
                    "winner_espn_id": (
                        int(c0["team"]["id"]) if c0.get("winner")
                        else int(c1["team"]["id"])
                    ),
                }
                games.append(game)

        except Exception as e:
            print(f"Warning: ESPN fetch failed for {date_str}: {e}")

        current += timedelta(days=1)

    return games


def map_results_to_slots(
    games: list[dict],
    bracket_struct: dict,
) -> dict[str, dict]:
    """Map completed games to bracket slots.

    Args:
        games: List of dicts with keys: winner_kaggle_id, loser_kaggle_id,
            winner_score, loser_score.
        bracket_struct: From build_bracket_structure().

    Returns:
        Dict mapping slot name to result dict with keys:
        winner, loser, winner_score, loser_score.
    """
    seed_to_team = bracket_struct["seed_to_team"]
    play_in_slots = bracket_struct.get("play_in_slots", {})
    regular_slots = bracket_struct.get("regular_slots", {})

    # resolved maps slot/seed names to team info dicts
    resolved = dict(seed_to_team)

    unmatched = list(games)
    actual_results = {}

    def _find_and_record(slot, team_a_id, team_b_id):
        for i, g in enumerate(unmatched):
            ids = {g["winner_kaggle_id"], g["loser_kaggle_id"]}
            if ids == {team_a_id, team_b_id}:
                actual_results[slot] = {
                    "winner": g["winner_kaggle_id"],
                    "loser": g["loser_kaggle_id"],
                    "winner_score": g["winner_score"],
                    "loser_score": g["loser_score"],
                }
                winner_id = g["winner_kaggle_id"]
                # Find winner's team info and store under this slot
                for s, info in resolved.items():
                    if isinstance(info, dict) and info.get("team_id") == winner_id:
                        resolved[slot] = info
                        break
                unmatched.pop(i)
                return True
        return False

    # Phase 1: Play-in games
    for slot, (strong, weak) in play_in_slots.items():
        team_a = resolved.get(strong)
        team_b = resolved.get(weak)
        if team_a and team_b:
            _find_and_record(slot, team_a["team_id"], team_b["team_id"])

    # Phase 2: Regular slots sorted by round number
    slot_order = sorted(
        regular_slots.keys(),
        key=lambda s: (int(s[1]), s),
    )

    for slot in slot_order:
        strong, weak = regular_slots[slot]
        team_a = resolved.get(strong)
        team_b = resolved.get(weak)
        if team_a is None or team_b is None:
            continue
        _find_and_record(slot, team_a["team_id"], team_b["team_id"])

    for g in unmatched:
        print(
            f"Warning: Unmatched game result: "
            f"{g['winner_kaggle_id']} beat {g['loser_kaggle_id']} "
            f"({g['winner_score']}-{g['loser_score']})"
        )

    return actual_results


def fetch_tournament_results(season: int, gender: str) -> dict:
    """Fetch actual tournament results and map to bracket slots.

    Returns dict mapping slot name to result dict, compatible with
    export_snapshot's actual_results parameter.

    Returns empty dict on failure (equivalent to no results known).
    """
    try:
        seeds_df = load_seeds(season, gender)
        espn_teams = fetch_espn_teams(gender)
        espn_to_kaggle = build_espn_to_kaggle_map(seeds_df, espn_teams, gender)

        if not espn_to_kaggle:
            print("Warning: ESPN-to-Kaggle mapping is empty, cannot resolve results")
            return {}

        espn_games = fetch_espn_results(gender, season)
        if not espn_games:
            return {}

        # Convert ESPN IDs to Kaggle IDs
        mapped_games = []
        for g in espn_games:
            winner_kaggle = espn_to_kaggle.get(g["winner_espn_id"])
            loser_espn = (
                g["team_b_espn_id"]
                if g["winner_espn_id"] == g["team_a_espn_id"]
                else g["team_a_espn_id"]
            )
            loser_kaggle = espn_to_kaggle.get(loser_espn)

            if winner_kaggle is None or loser_kaggle is None:
                print(
                    f"Warning: Could not map ESPN game {g['espn_game_id']} "
                    f"to Kaggle IDs (winner ESPN {g['winner_espn_id']}, "
                    f"loser ESPN {loser_espn})"
                )
                continue

            winner_score = (
                g["score_a"] if g["winner_espn_id"] == g["team_a_espn_id"]
                else g["score_b"]
            )
            loser_score = (
                g["score_b"] if g["winner_espn_id"] == g["team_a_espn_id"]
                else g["score_a"]
            )

            mapped_games.append({
                "winner_kaggle_id": winner_kaggle,
                "loser_kaggle_id": loser_kaggle,
                "winner_score": winner_score,
                "loser_score": loser_score,
            })

        bracket_struct = build_bracket_structure(season, gender)
        return map_results_to_slots(mapped_games, bracket_struct)

    except Exception as e:
        print(f"Warning: Failed to fetch tournament results: {e}")
        return {}


def fetch_espn_teams(gender: str) -> list[dict]:
    """Fetch ESPN team directory for the given gender.

    Returns list of dicts with keys: id, displayName, shortDisplayName, abbreviation.
    Returns empty list on failure.
    """
    sport = _espn_sport_path(gender)
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
        f"{sport}/teams?limit=500"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode())

        teams = []
        for entry in raw.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
            team = entry.get("team", {})
            teams.append({
                "id": int(team.get("id", 0)),
                "displayName": team.get("displayName", ""),
                "shortDisplayName": team.get("shortDisplayName", ""),
                "abbreviation": team.get("abbreviation", ""),
            })
        return teams

    except Exception as e:
        print(f"Warning: Could not fetch ESPN teams for {gender}: {e}")
        return []
