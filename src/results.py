"""Fetch actual tournament results from ESPN and map to bracket slots."""

import json
import urllib.request
from datetime import date, timedelta


# Kaggle name -> ESPN full name, same direction as export.py's name_overrides.
# Only needed for names that don't match via normalized substring.
NAME_OVERRIDES = {
    "St John's": "st. john's red storm",
    "St Mary's CA": "saint mary's gaels",
    "NC State": "nc state wolfpack",
    "Michigan St": "michigan state spartans",
    "Iowa St": "iowa state cyclones",
    "Ohio St": "ohio state buckeyes",
    "Texas Tech": "texas tech red raiders",
    "Texas A&M": "texas a&m aggies",
    "Utah St": "utah state aggies",
    "McNeese St": "mcneese cowboys",
    "Tennessee St": "tennessee state tigers",
    "Miami FL": "miami hurricanes",
    "Col Charleston": "charleston cougars",
    "South Florida": "south florida bulls",
    "Notre Dame": "notre dame fighting irish",
    "West Virginia": "west virginia mountaineers",
    "Hawaii": "hawai'i rainbow warriors",
    "Southern Univ": "southern university jaguars",
    "F Dickinson": "fdu knights",
    "WI Green Bay": "green bay phoenix",
    "N Dakota St": "north dakota state bison",
    "Cal Baptist": "california baptist lancers",
    "UT San Antonio": "utsa roadrunners",
}

# Tournament start dates by (season, gender)
_TOURNAMENT_START = {
    (2026, "M"): date(2026, 3, 17),
    (2026, "W"): date(2026, 3, 19),
}


def _espn_sport_path(gender: str) -> str:
    return "mens-college-basketball" if gender == "M" else "womens-college-basketball"


def build_espn_to_kaggle_map(
    seeds_df,
    espn_teams: list[dict],
) -> dict[int, int]:
    """Build ESPN team ID -> Kaggle TeamID mapping for tournament teams.

    Args:
        seeds_df: DataFrame with TeamID, TeamName columns (from load_seeds).
        espn_teams: List of ESPN team dicts with keys: id, displayName,
            shortDisplayName, abbreviation.

    Returns:
        Dict mapping ESPN team ID (int) to Kaggle TeamID (int).
    """
    # Build ESPN lookup: normalized name -> ESPN ID
    espn_lookup = {}
    for team in espn_teams:
        espn_id = int(team["id"])
        for key in ["displayName", "shortDisplayName", "abbreviation"]:
            name = team.get(key, "")
            if name:
                espn_lookup[name.lower().strip()] = espn_id

    mapping = {}
    for _, row in seeds_df.iterrows():
        kaggle_id = int(row["TeamID"])
        kaggle_name = row["TeamName"]

        # Try override first
        override = NAME_OVERRIDES.get(kaggle_name, "").lower()
        espn_id = espn_lookup.get(override)

        # Try exact lowercase match
        if espn_id is None:
            espn_id = espn_lookup.get(kaggle_name.lower())

        # Try substring: Kaggle name contained in ESPN name
        if espn_id is None:
            for espn_name, eid in espn_lookup.items():
                if kaggle_name.lower() in espn_name:
                    espn_id = eid
                    break

        if espn_id is not None:
            mapping[espn_id] = kaggle_id
        else:
            print(f"Warning: No ESPN match for {kaggle_name} (Kaggle ID {kaggle_id})")

    return mapping


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
    today = date.today()
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
