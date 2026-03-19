"""Tournament simulation engine."""

import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import norm
from src.data import load_seeds, load_slots


def build_bracket_structure(season: int = 2026, gender: str = "M") -> dict:
    """Parse the Kaggle bracket structure into a simulatable format.

    The Kaggle slot structure works as follows:
    - Each seed (e.g. W01, X16a) maps to a team
    - Each slot (e.g. R1W1, R2W1, ..., R6CH) is a game
    - A slot's StrongSeed/WeakSeed are either seed names or other slot names
    - Play-in games are slots whose StrongSeed/WeakSeed contain 'a'/'b' suffixes
      (e.g., slot X16 has StrongSeed=X16a, WeakSeed=X16b)
    """
    seeds = load_seeds(season, gender)
    slots = load_slots(season, gender)

    seed_to_team = {}
    for _, row in seeds.iterrows():
        seed_to_team[row["Seed"]] = {
            "team_id": row["TeamID"],
            "team_name": row["TeamName"],
            "seed_num": row["SeedNum"],
        }

    # Separate play-in slots from regular tournament slots.
    # Play-in slots reference seeds with a/b suffixes.
    play_in_slots = {}
    regular_slots = {}
    for _, row in slots.iterrows():
        slot, strong, weak = row["Slot"], row["StrongSeed"], row["WeakSeed"]
        if strong.endswith("a") or strong.endswith("b"):
            play_in_slots[slot] = (strong, weak)
        else:
            regular_slots[slot] = (strong, weak)

    return {
        "seed_to_team": seed_to_team,
        "play_in_slots": play_in_slots,
        "regular_slots": regular_slots,
        "seeds_df": seeds,
        "gender": gender,
    }


def _play_game(
    team_a: dict,
    team_b: dict,
    sigma_val: float,
    team_id_to_idx: dict,
    rng: np.random.Generator,
    theta: np.ndarray | None = None,
    off: np.ndarray | None = None,
    deff: np.ndarray | None = None,
    home_advantage: float = 0.0,
    home_team_id: int | None = None,
) -> dict:
    """Simulate a single game between two teams. Returns the winner."""
    idx_a = team_id_to_idx.get(team_a["team_id"])
    idx_b = team_id_to_idx.get(team_b["team_id"])

    if idx_a is not None and idx_b is not None:
        if off is not None and deff is not None:
            diff = (off[idx_a] - deff[idx_b]) - (off[idx_b] - deff[idx_a])
            denom = sigma_val * np.sqrt(2)
        else:
            diff = theta[idx_a] - theta[idx_b]
            denom = sigma_val

        if home_team_id == team_a["team_id"]:
            diff += home_advantage
        elif home_team_id == team_b["team_id"]:
            diff -= home_advantage

        p_a_wins = norm.cdf(diff / denom)
        return team_a if rng.random() < p_a_wins else team_b
    return team_a  # fallback


def simulate_tournament_single(
    bracket_struct: dict,
    sigma_val: float,
    team_id_to_idx: dict,
    rng: np.random.Generator,
    theta: np.ndarray | None = None,
    off: np.ndarray | None = None,
    deff: np.ndarray | None = None,
    alpha_val: float = 0.0,
    actual_results: dict | None = None,
) -> dict:
    """Simulate one complete tournament bracket.

    Args:
        theta: Team strength array (margin model) or None if using off/def.
        off: Offensive strength array (score model) or None.
        deff: Defensive strength array (score model) or None.
        alpha_val: Home court advantage from the posterior. Used for women's
            tournament R1/R2 where top 16 seeds host on campus.

    Returns dict with:
        - slot_winners: maps each game slot to the winning team info
        - champion: the championship winner
        - round_results: dict mapping round number -> list of winning team dicts
    """
    seed_to_team = bracket_struct["seed_to_team"]
    play_in_slots = bracket_struct["play_in_slots"]
    regular_slots = bracket_struct["regular_slots"]
    is_womens = bracket_struct.get("gender") == "W"

    game_kwargs = dict(
        sigma_val=sigma_val, team_id_to_idx=team_id_to_idx, rng=rng,
        theta=theta, off=off, deff=deff,
    )

    # resolved maps a name (seed or slot) to the team occupying it
    resolved = dict(seed_to_team)

    slot_winners = {}

    # Resolve play-in games first
    for slot, (strong, weak) in play_in_slots.items():
        team_a = resolved[strong]
        team_b = resolved[weak]
        if actual_results and slot in actual_results:
            winner_id = actual_results[slot]["winner"]
            winner = team_a if team_a["team_id"] == winner_id else team_b
        else:
            winner = _play_game(team_a, team_b, **game_kwargs)
        resolved[slot] = winner
        slot_winners[slot] = winner

    # Process regular slots in round order
    slot_order = sorted(
        regular_slots.keys(),
        key=lambda s: (int(s[1]), s),
    )

    round_results = {}

    for slot in slot_order:
        strong, weak = regular_slots[slot]
        team_a = resolved.get(strong)
        team_b = resolved.get(weak)

        if team_a is None or team_b is None:
            continue

        # Women's R1/R2: top 16 seeds (seed 1-4 in each region) host
        home_advantage = 0.0
        home_team_id = None
        round_num = int(slot[1])
        if is_womens and round_num <= 2:
            if team_a["seed_num"] < team_b["seed_num"]:
                if team_a["seed_num"] <= 4:
                    home_advantage = alpha_val
                    home_team_id = team_a["team_id"]
            elif team_b["seed_num"] < team_a["seed_num"]:
                if team_b["seed_num"] <= 4:
                    home_advantage = alpha_val
                    home_team_id = team_b["team_id"]

        if actual_results and slot in actual_results:
            winner_id = actual_results[slot]["winner"]
            winner = team_a if team_a["team_id"] == winner_id else team_b
        else:
            winner = _play_game(
                team_a, team_b, **game_kwargs,
                home_advantage=home_advantage, home_team_id=home_team_id,
            )
        resolved[slot] = winner
        slot_winners[slot] = winner

        round_results.setdefault(round_num, []).append(winner)

    champ_slots = [s for s in slot_order if s.startswith("R6")]
    champion = resolved.get(champ_slots[0]) if champ_slots else None

    return {
        "slot_winners": slot_winners,
        "champion": champion,
        "round_results": round_results,
    }


def simulate_tournament(
    bracket_struct: dict,
    sigma_samples: np.ndarray,
    team_ids: np.ndarray,
    n_sims: int = 10000,
    seed: int = 42,
    theta_samples: np.ndarray | None = None,
    off_samples: np.ndarray | None = None,
    def_samples: np.ndarray | None = None,
    alpha_samples: np.ndarray | None = None,
    actual_results: dict | None = None,
) -> dict:
    """Run full tournament simulation across posterior samples.

    For each simulation, draw one posterior sample of team strengths,
    then simulate the entire bracket.

    Accepts either theta_samples (margin model) or off_samples + def_samples
    (score-based offense/defense model).
    """
    rng = np.random.default_rng(seed)
    team_id_to_idx = {int(tid): i for i, tid in enumerate(team_ids)}

    # Determine which samples to use for indexing
    ref_samples = off_samples if off_samples is not None else theta_samples
    n_posterior = ref_samples.shape[0]
    seeds_df = bracket_struct["seeds_df"]
    tourney_team_ids = set(int(t) for t in seeds_df["TeamID"].values)

    round_names = [
        "Round of 64",
        "Round of 32",
        "Sweet 16",
        "Elite Eight",
        "Final Four",
        "Championship",
        "Champion",
    ]

    advancement_counts = {tid: np.zeros(7) for tid in tourney_team_ids}
    champion_counts = Counter()
    all_results = []

    for sim_i in range(n_sims):
        sample_idx = rng.integers(0, n_posterior)
        sigma_val = sigma_samples[sample_idx]
        alpha_val = alpha_samples[sample_idx] if alpha_samples is not None else 0.0

        if off_samples is not None and def_samples is not None:
            off = off_samples[sample_idx]
            deff = def_samples[sample_idx]
            theta = None
        else:
            theta = theta_samples[sample_idx]
            off = None
            deff = None

        result = simulate_tournament_single(
            bracket_struct, sigma_val, team_id_to_idx, rng,
            theta=theta, off=off, deff=deff, alpha_val=alpha_val,
            actual_results=actual_results,
        )

        # Everyone in the tournament made Round of 64
        for tid in tourney_team_ids:
            advancement_counts[tid][0] += 1

        # Track advancement by round
        for round_num, winners in result["round_results"].items():
            # round_num 1 -> advancement index 1 (made R32)
            # round_num 6 -> advancement index 6 (champion)
            adv_idx = round_num
            if adv_idx > 6:
                continue
            for winner in winners:
                tid = int(winner["team_id"])
                if tid in advancement_counts:
                    advancement_counts[tid][adv_idx] += 1

        if result["champion"]:
            champion_counts[result["champion"]["team_name"]] += 1

        all_results.append(result)

    # Build advancement DataFrame
    rows = []
    for tid in tourney_team_ids:
        seed_rows = seeds_df[seeds_df["TeamID"] == tid]
        if len(seed_rows) == 0:
            continue
        seed_row = seed_rows.iloc[0]
        row = {
            "TeamID": tid,
            "TeamName": seed_row["TeamName"],
            "Seed": seed_row["Seed"],
            "SeedNum": seed_row["SeedNum"],
            "Region": seed_row["Region"],
        }
        for r, name in enumerate(round_names):
            row[name] = advancement_counts[tid][r] / n_sims
        rows.append(row)

    advancement_df = pd.DataFrame(rows).sort_values("Champion", ascending=False)

    return {
        "advancement": advancement_df,
        "champions": dict(champion_counts),
        "all_results": all_results,
        "round_names": round_names,
        "n_sims": n_sims,
    }


def get_most_likely_bracket(sim_results: dict, bracket_struct: dict) -> dict:
    """Extract the modal bracket (most common winner at each slot)."""
    all_slots = set()
    all_slots.update(bracket_struct["play_in_slots"].keys())
    all_slots.update(bracket_struct["regular_slots"].keys())

    slot_winners = {}
    for slot in sorted(all_slots):
        winners = []
        for result in sim_results["all_results"]:
            w = result["slot_winners"].get(slot)
            if w is not None:
                winners.append(w["team_name"])
        if winners:
            counter = Counter(winners)
            most_common = counter.most_common(1)[0]
            slot_winners[slot] = {
                "team": most_common[0],
                "probability": most_common[1] / len(winners),
            }

    return slot_winners


def championship_gini(advancement: pd.DataFrame) -> float:
    """Gini coefficient over championship probability distribution.

    0 = every team equally likely to win, 1 = single team wins everything.
    """
    probs = np.sort(advancement["Champion"].values)
    n = len(probs)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * probs) / np.sum(probs) - (n + 1)) / n


def tail_analysis(sim_results: dict) -> dict:
    """Analyze the most extreme simulation outcomes."""
    all_results = sim_results["all_results"]

    # Deepest Cinderella runs by seed line
    seed_best_round = {}
    for result in all_results:
        for slot, winner in result["slot_winners"].items():
            seed_num = winner.get("seed_num", 0)
            if seed_num >= 10:
                round_num = int(slot[1]) if slot[0] == "R" and slot[1].isdigit() else 0
                key = f"{seed_num}-seed"
                if key not in seed_best_round or round_num > seed_best_round[key][1]:
                    seed_best_round[key] = (winner["team_name"], round_num)

    # How often zero #1 seeds made Final Four
    no_one_seeds_ff = 0
    for result in all_results:
        one_seeds_in_ff = 0
        for winner in result["round_results"].get(4, []):
            if winner.get("seed_num") == 1:
                one_seeds_in_ff += 1
        if one_seeds_in_ff == 0:
            no_one_seeds_ff += 1

    # Chalkiest bracket (higher seeds winning = more chalk)
    chalk_scores = []
    for result in all_results:
        chalk = 0
        for slot, winner in result["slot_winners"].items():
            if slot[0] == "R" and slot[1].isdigit():
                round_num = int(slot[1])
                chalk += (17 - winner.get("seed_num", 8)) * round_num
        chalk_scores.append(chalk)

    chalkiest_idx = int(np.argmax(chalk_scores))
    most_chaotic_idx = int(np.argmin(chalk_scores))

    return {
        "deepest_cinderella": seed_best_round,
        "no_one_seeds_final_four_pct": no_one_seeds_ff / len(all_results),
        "chalkiest_sim_idx": chalkiest_idx,
        "most_chaotic_sim_idx": most_chaotic_idx,
        "chalk_scores": chalk_scores,
    }
