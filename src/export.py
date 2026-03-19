"""Export model results to dashboard JSON format."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone


def export_snapshot(
    data: dict,
    sigma_samples: np.ndarray,
    sim_results: dict,
    bracket_struct: dict,
    date: str,
    actual_results: dict | None = None,
    output_dir: Path = Path("."),
    alpha_samples: np.ndarray | None = None,
    idata=None,
    off_samples: np.ndarray | None = None,
    def_samples: np.ndarray | None = None,
    theta_samples: np.ndarray | None = None,
) -> Path:
    """Export a single day's snapshot as JSON.

    Accepts either off_samples + def_samples (offense/defense model)
    or theta_samples (legacy margin model).
    """
    if actual_results is None:
        actual_results = {}

    team_id_to_idx = {int(tid): i for i, tid in enumerate(data["team_ids"])}
    seeds_df = bracket_struct["seeds_df"]
    advancement = sim_results["advancement"]

    is_offdef = off_samples is not None and def_samples is not None

    # Teams section — tournament teams only
    teams = {}
    for _, seed_row in seeds_df.iterrows():
        tid = int(seed_row["TeamID"])
        idx = team_id_to_idx.get(tid)
        if idx is None:
            continue

        entry = {
            "name": seed_row["TeamName"],
            "seed": seed_row["Seed"],
            "seed_num": int(seed_row["SeedNum"]),
            "region": seed_row["Region"],
            "eliminated": False,
            "eliminated_round": None,
        }

        if is_offdef:
            off_mean = float(off_samples[:, idx].mean())
            off_std = float(off_samples[:, idx].std())
            def_mean = float(def_samples[:, idx].mean())
            def_std = float(def_samples[:, idx].std())
            overall = off_mean + def_mean
            entry["off_mean"] = round(off_mean, 2)
            entry["off_std"] = round(off_std, 2)
            entry["def_mean"] = round(def_mean, 2)
            entry["def_std"] = round(def_std, 2)
            entry["overall_mean"] = round(overall, 2)
        else:
            ref = theta_samples
            entry["theta_mean"] = round(float(ref[:, idx].mean()), 2)
            entry["theta_std"] = round(float(ref[:, idx].std()), 2)

        teams[str(tid)] = entry

    _mark_eliminated(teams, actual_results, bracket_struct)

    # Advancement probabilities
    round_names = [
        "Round of 64", "Round of 32", "Sweet 16",
        "Elite Eight", "Final Four", "Championship", "Champion",
    ]
    advancement_dict = {}
    for _, row in advancement.iterrows():
        tid = str(int(row["TeamID"]))
        advancement_dict[tid] = {
            rn: round(float(row[rn]), 4) for rn in round_names
        }

    bracket = _build_bracket_section(sim_results, bracket_struct, actual_results)

    champ_odds = {}
    for _, row in advancement.iterrows():
        champ_odds[str(int(row["TeamID"]))] = round(float(row["Champion"]), 4)

    # Hyperparameters
    hyper = {
        "sigma_mean": round(float(sigma_samples.mean()), 2),
        "sigma_std": round(float(sigma_samples.std()), 2),
    }
    if alpha_samples is not None:
        hyper["alpha_mean"] = round(float(alpha_samples.mean()), 2)
        hyper["alpha_std"] = round(float(alpha_samples.std()), 2)
    if idata is not None:
        try:
            post = idata.posterior
            for var in ["sigma_conf", "sigma_team", "sigma_off_conf",
                        "sigma_def_conf", "mu_intercept"]:
                if var in post:
                    hyper[f"{var}_mean"] = round(
                        float(post[var].values.flatten().mean()), 2
                    )
            if "lkj_corr" in post:
                corr = post["lkj_corr"].values[:, :, 0, 1].flatten()
                hyper["off_def_corr_mean"] = round(float(corr.mean()), 2)
        except Exception:
            pass

    hyper["model_type"] = "offense_defense" if is_offdef else "margin"

    snapshot = {
        "date": date,
        "games_in_training_data": int(data["n_games"]),
        "tournament_games_played": sum(1 for r in actual_results.values() if r),
        "model_fit_timestamp": datetime.now(timezone.utc).isoformat(),
        "teams": teams,
        "advancement": advancement_dict,
        "bracket": bracket,
        "championship_odds": champ_odds,
        "hyperparameters": hyper,
        "actual_results": actual_results,
    }

    output_dir = Path(output_dir)
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    filepath = snapshots_dir / f"{date}.json"
    with open(filepath, "w") as f:
        json.dump(snapshot, f, separators=(",", ":"))

    latest_path = output_dir / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(snapshot, f, separators=(",", ":"))

    return filepath


def _mark_eliminated(teams: dict, actual_results: dict, bracket_struct: dict):
    """Set eliminated=True for teams that lost in actual results."""
    round_name_map = {
        1: "Round of 64", 2: "Round of 32", 3: "Sweet 16",
        4: "Elite Eight", 5: "Final Four", 6: "Championship",
    }
    for slot, result in actual_results.items():
        if not result or "loser" not in result:
            continue
        loser_key = str(result["loser"])
        if loser_key in teams:
            if slot[0] == "R" and slot[1].isdigit():
                round_num = int(slot[1])
            else:
                round_num = 0
            teams[loser_key]["eliminated"] = True
            teams[loser_key]["eliminated_round"] = round_name_map.get(
                round_num, "First Four"
            )


def _build_bracket_section(sim_results: dict, bracket_struct: dict, actual_results: dict) -> dict:
    """Build bracket dict with per-game win probabilities from simulation."""
    seed_to_team = bracket_struct["seed_to_team"]
    regular_slots = bracket_struct["regular_slots"]
    n_sims = sim_results["n_sims"]

    slot_winner_counts = {}
    for result in sim_results["all_results"]:
        for slot, winner in result["slot_winners"].items():
            if slot not in slot_winner_counts:
                slot_winner_counts[slot] = {}
            tid = winner["team_id"]
            slot_winner_counts[slot][tid] = slot_winner_counts[slot].get(tid, 0) + 1

    play_in_slots = bracket_struct.get("play_in_slots", {})
    play_in_resolved = {}
    bracket = {}
    for pi_slot, (pi_strong, pi_weak) in play_in_slots.items():
        team_a = seed_to_team.get(pi_strong)
        team_b = seed_to_team.get(pi_weak)
        if team_a and team_b:
            pi_result = actual_results.get(pi_slot)
            if pi_result and "winner" in pi_result:
                winner_id = pi_result["winner"]
                play_in_resolved[pi_slot] = team_a if team_a["team_id"] == winner_id else team_b
            else:
                play_in_resolved[pi_slot] = team_a

            counts = slot_winner_counts.get(pi_slot, {})
            a_wins = counts.get(team_a["team_id"], 0)
            p_a = round(a_wins / n_sims, 4) if n_sims > 0 else 0.5

            bracket[pi_slot] = {
                "team_a": {"id": team_a["team_id"], "name": team_a["team_name"], "seed_num": team_a["seed_num"]},
                "team_b": {"id": team_b["team_id"], "name": team_b["team_name"], "seed_num": team_b["seed_num"]},
                "p_a_wins": p_a,
                "result": actual_results.get(pi_slot),
                "play_in": True,
            }

    def _resolve(name):
        return seed_to_team.get(name) or play_in_resolved.get(name)
    for slot, (strong, weak) in sorted(regular_slots.items()):
        if not slot.startswith("R1"):
            continue
        team_a = _resolve(strong)
        team_b = _resolve(weak)
        if not team_a or not team_b:
            continue

        counts = slot_winner_counts.get(slot, {})
        a_wins = counts.get(team_a["team_id"], 0)
        p_a = round(a_wins / n_sims, 4) if n_sims > 0 else 0.5

        play_in_note = None
        if weak in play_in_slots:
            pi_strong_s, pi_weak_s = play_in_slots[weak]
            pi_a = seed_to_team.get(pi_strong_s)
            pi_b = seed_to_team.get(pi_weak_s)
            if pi_a and pi_b:
                play_in_note = f"{pi_a['team_name']}/{pi_b['team_name']}"

        entry = {
            "team_a": {"id": team_a["team_id"], "name": team_a["team_name"], "seed_num": team_a["seed_num"]},
            "team_b": {"id": team_b["team_id"], "name": team_b["team_name"], "seed_num": team_b["seed_num"]},
            "p_a_wins": p_a,
            "result": actual_results.get(slot),
        }
        if play_in_note:
            entry["play_in"] = play_in_note

        bracket[slot] = entry

    for slot in sorted(regular_slots.keys()):
        if slot.startswith("R1"):
            continue
        counts = slot_winner_counts.get(slot, {})
        if not counts:
            continue

        sorted_teams = sorted(counts.items(), key=lambda x: -x[1])
        top = []
        for tid, count in sorted_teams[:6]:
            name = None
            for seed_info in seed_to_team.values():
                if seed_info["team_id"] == tid:
                    name = seed_info["team_name"]
                    break
            top.append({
                "id": tid,
                "name": name or str(tid),
                "p": round(count / n_sims, 4),
            })

        bracket[slot] = {
            "contenders": top,
            "result": actual_results.get(slot),
        }

    return bracket


def export_odds_timeline(snapshots_dir: Path, output_path: Path) -> None:
    """Read all snapshot files and produce a condensed timeline."""
    snapshots_dir = Path(snapshots_dir)
    timeline = {}
    dates = []

    for snapshot_file in sorted(snapshots_dir.glob("*.json")):
        with open(snapshot_file) as f:
            snap = json.load(f)
        date = snap["date"]
        dates.append(date)
        for tid, odds in snap.get("championship_odds", {}).items():
            if tid not in timeline:
                timeline[tid] = {}
            timeline[tid][date] = odds

    team_names = {}
    if dates:
        with open(snapshots_dir / f"{dates[-1]}.json") as f:
            snap = json.load(f)
        for tid, info in snap.get("teams", {}).items():
            team_names[tid] = info["name"]

    result = {
        "dates": dates,
        "teams": {
            tid: {"name": team_names.get(tid, tid), "odds": odds_by_date}
            for tid, odds_by_date in timeline.items()
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))


def export_team_branding(
    men_seeds_df: pd.DataFrame,
    women_seeds_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Build team_branding.json with ESPN logo URLs and school colors."""
    import urllib.request

    all_teams = {}
    for _, row in pd.concat([men_seeds_df, women_seeds_df]).iterrows():
        tid = int(row["TeamID"])
        if tid not in all_teams:
            all_teams[tid] = row["TeamName"]

    espn_teams = {}
    for gender_path in ["mens-college-basketball", "womens-college-basketball"]:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/{gender_path}/teams?limit=500"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = json.loads(resp.read().decode())
            for entry in raw.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
                team = entry.get("team", {})
                espn_id = int(team.get("id", 0))
                name = team.get("displayName", "")
                short = team.get("shortDisplayName", "")
                abbr = team.get("abbreviation", "")
                color = team.get("color", "")
                alt_color = team.get("alternateColor", "")

                def _hex(c):
                    if not c:
                        return "#666666"
                    return f"#{c}" if not c.startswith("#") else c

                info = {
                    "espn_id": espn_id,
                    "name": name,
                    "color": _hex(color),
                    "alt_color": _hex(alt_color),
                }
                espn_teams[name.lower()] = info
                if short:
                    espn_teams[short.lower()] = info
                if abbr:
                    espn_teams[abbr.lower()] = info
        except Exception as e:
            print(f"Warning: Could not fetch ESPN teams for {gender_path}: {e}")

    name_overrides = {
        "St John's": "st. john's red storm",
        "St Mary's CA": "saint mary's gaels",
        "St Louis": "saint louis billikens",
        "Miami FL": "miami hurricanes",
        "Miami OH": "miami (oh) redhawks",
        "LIU Brooklyn": "long island university sharks",
        "N Dakota St": "north dakota state bison",
        "Cal Baptist": "california baptist lancers",
        "Queens NC": "queens university royals",
        "High Point": "high point panthers",
        "Prairie View": "prairie view a&m panthers",
        "NC State": "nc state wolfpack",
        "South Florida": "south florida bulls",
        "Texas A&M": "texas a&m aggies",
        "McNeese St": "mcneese cowboys",
        "Tennessee St": "tennessee state tigers",
        "Utah St": "utah state aggies",
        "Michigan St": "michigan state spartans",
        "Iowa St": "iowa state cyclones",
        "Texas Tech": "texas tech red raiders",
        "Ohio St": "ohio state buckeyes",
        "Notre Dame": "notre dame fighting irish",
        "West Virginia": "west virginia mountaineers",
        "Wright St": "wright state raiders",
        "Northern Iowa": "northern iowa panthers",
        "Hawaii": "hawai'i rainbow warriors",
        "UT San Antonio": "utsa roadrunners",
        "F Dickinson": "fdu knights",
        "Southern Univ": "southern university jaguars",
        "WI Green Bay": "green bay phoenix",
        "Col Charleston": "charleston cougars",
    }

    branding = {}
    for tid, name in all_teams.items():
        override_key = name_overrides.get(name, "").lower()
        match = espn_teams.get(override_key) or espn_teams.get(name.lower())

        if not match:
            for espn_name, espn_data in espn_teams.items():
                if name.lower() in espn_name:
                    match = espn_data
                    break

        if match:
            branding[str(tid)] = {
                "name": name,
                "espn_id": match["espn_id"],
                "logo_url": f"https://a.espncdn.com/i/teamlogos/ncaa/500-dark/{match['espn_id']}.png",
                "primary_color": match["color"],
                "secondary_color": match["alt_color"],
            }
        else:
            branding[str(tid)] = {
                "name": name,
                "espn_id": None,
                "logo_url": None,
                "primary_color": "#666666",
                "secondary_color": "#999999",
            }
            print(f"Warning: No ESPN match for {name} (ID {tid})")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(branding, f, indent=2)
    print(f"Exported branding for {len(branding)} teams to {output_path}")


def generate_baseline(
    output_base: Path,
    date: str = "2026-03-17",
) -> None:
    """Generate all baseline data files from existing model results."""
    import gc
    import arviz as az
    from src.data import build_model_data, load_seeds
    from src.model import get_team_strengths
    from src.simulate import build_bracket_structure, simulate_tournament

    output_base = Path(output_base)

    for gender in ["M", "W"]:
        suffix = "mens" if gender == "M" else "womens"
        gender_dir = output_base / suffix

        print(f"\nGenerating baseline for {suffix}...")
        data = build_model_data(2026, gender)
        idata = az.from_netcdf(f"results/model_2026_{suffix}.nc")

        strengths = get_team_strengths(idata, data)
        sigma = idata.posterior["sigma"].values.flatten()
        alpha = idata.posterior["alpha"].values.flatten()

        bracket_struct = build_bracket_structure(2026, gender)

        # Detect model type
        is_offdef = "off_samples" in strengths
        if is_offdef:
            sim = simulate_tournament(
                bracket_struct, sigma_samples=sigma,
                team_ids=data["team_ids"], n_sims=10000, seed=42,
                off_samples=strengths["off_samples"],
                def_samples=strengths["def_samples"],
                alpha_samples=alpha if gender == "W" else None,
            )
            filepath = export_snapshot(
                data=data,
                sigma_samples=sigma,
                sim_results=sim,
                bracket_struct=bracket_struct,
                date=date,
                output_dir=gender_dir,
                alpha_samples=alpha,
                idata=idata,
                off_samples=strengths["off_samples"],
                def_samples=strengths["def_samples"],
            )
        else:
            sim = simulate_tournament(
                bracket_struct, sigma_samples=sigma,
                team_ids=data["team_ids"], n_sims=10000, seed=42,
                theta_samples=strengths["samples"],
                alpha_samples=alpha if gender == "W" else None,
            )
            filepath = export_snapshot(
                data=data,
                sigma_samples=sigma,
                sim_results=sim,
                bracket_struct=bracket_struct,
                date=date,
                output_dir=gender_dir,
                alpha_samples=alpha,
                idata=idata,
                theta_samples=strengths["samples"],
            )

        print(f"  Snapshot: {filepath}")

        export_odds_timeline(
            gender_dir / "snapshots",
            gender_dir / "odds_timeline.json",
        )

        del idata
        gc.collect()

    men_seeds = load_seeds(2026, "M")
    women_seeds = load_seeds(2026, "W")
    export_team_branding(men_seeds, women_seeds, output_base / "team_branding.json")

    print(f"\nBaseline generation complete. Output: {output_base}")
