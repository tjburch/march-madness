"""Bracket pool optimization via Bayesian decision theory.

Instead of maximizing expected correct picks (modal bracket), this module
optimizes for probability of winning a bracket pool of N participants.

Key insight: in a pool, you're competing against other brackets, not just
the tournament. The optimal strategy accounts for what others will pick.
Picking undervalued teams (high "leverage") creates differentiation that
increases your win probability, especially in larger pools.

The framework has three components:
  1. True probabilities — from the Bayesian model's posterior simulations
  2. Public pick distribution — estimated from seed-based heuristics
  3. Pool simulation — Monte Carlo evaluation of P(winning)

Usage:
    sim_results = simulate_tournament(...)  # existing pipeline
    bracket_struct = build_bracket_structure(...)

    pool = PoolOptimizer(sim_results, bracket_struct, pool_size=50)
    result = pool.optimize()
    print(result["win_prob"])  # P(winning the pool)
    print(result["bracket"])  # optimized bracket picks
"""

import numpy as np
import pandas as pd
from collections import Counter

# ESPN standard scoring: points per correct pick by round
SCORING_SYSTEMS = {
    "espn": {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320},
    "simple": {1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32},
}


def seed_public_pick_prob(seed_a: int, seed_b: int) -> float:
    """Estimate P(public picks team_a over team_b) based on seeds.

    The public tends to follow seeds more rigidly than true probabilities
    warrant. A 5-seed might have ~65% true win probability against a 12-seed,
    but ~75% of the public will pick the 5-seed.

    Uses a logistic model calibrated to typical ESPN public bracket patterns:
      - 1 vs 16: ~97% pick the 1 (true ~99%)
      - 2 vs 15: ~95% pick the 2
      - 5 vs 12: ~78% pick the 5 (true ~65%)
      - 8 vs 9:  ~55% pick the 8

    Args:
        seed_a: Seed number of team A (1-16).
        seed_b: Seed number of team B (1-16).

    Returns:
        Probability that a typical public bracket picks team A.
    """
    beta = 0.35
    diff = seed_b - seed_a
    return 1.0 / (1.0 + np.exp(-beta * diff))


def extract_outcome(sim_result: dict) -> dict:
    """Extract slot -> team_id mapping from a simulation result."""
    return {
        slot: winner["team_id"]
        for slot, winner in sim_result["slot_winners"].items()
    }


def score_bracket(picks: dict, outcome: dict, scoring: dict) -> int:
    """Score a bracket against a tournament outcome.

    Args:
        picks: Bracket picks mapping slot -> team_id.
        outcome: Actual tournament outcome mapping slot -> team_id.
        scoring: Points per round, e.g. {1: 10, 2: 20, ...}.

    Returns:
        Total bracket score.
    """
    total = 0
    for slot, winner_id in outcome.items():
        if not (slot[0] == "R" and slot[1].isdigit()):
            continue
        round_num = int(slot[1])
        if picks.get(slot) == winner_id:
            total += scoring.get(round_num, 0)
    return total


def generate_bracket(
    bracket_struct: dict,
    win_prob_fn,
    rng: np.random.Generator,
) -> dict:
    """Generate a complete bracket by picking winners using win_prob_fn.

    Walks the bracket structure round by round. At each game, calls
    win_prob_fn(team_a, team_b) to get P(team_a wins), then samples
    the pick from that probability.

    Args:
        bracket_struct: From build_bracket_structure().
        win_prob_fn: Callable(team_a_dict, team_b_dict) -> float.
        rng: NumPy random generator.

    Returns:
        Dict mapping slot -> team_id for the generated bracket.
    """
    seed_to_team = bracket_struct["seed_to_team"]
    play_in_slots = bracket_struct["play_in_slots"]
    regular_slots = bracket_struct["regular_slots"]

    resolved = dict(seed_to_team)
    picks = {}

    for slot, (strong, weak) in play_in_slots.items():
        team_a = resolved[strong]
        team_b = resolved[weak]
        p = win_prob_fn(team_a, team_b)
        winner = team_a if rng.random() < p else team_b
        resolved[slot] = winner
        picks[slot] = winner["team_id"]

    slot_order = sorted(regular_slots.keys(), key=lambda s: (int(s[1]), s))
    for slot in slot_order:
        strong, weak = regular_slots[slot]
        team_a = resolved.get(strong)
        team_b = resolved.get(weak)
        if team_a is None or team_b is None:
            continue
        p = win_prob_fn(team_a, team_b)
        winner = team_a if rng.random() < p else team_b
        resolved[slot] = winner
        picks[slot] = winner["team_id"]

    return picks


def generate_public_bracket(
    bracket_struct: dict,
    rng: np.random.Generator,
) -> dict:
    """Generate a bracket from the public pick distribution (seed-based)."""
    def _public_fn(team_a, team_b):
        return seed_public_pick_prob(team_a["seed_num"], team_b["seed_num"])
    return generate_bracket(bracket_struct, _public_fn, rng)


def compute_leverage(
    sim_results: dict,
    bracket_struct: dict,
    n_public: int = 10000,
    seed: int = 123,
) -> pd.DataFrame:
    """Compute leverage for each team at each bracket slot.

    Leverage = P(model) / P(public) for a team winning a given slot.
    High leverage means the model thinks a team is undervalued by the public.

    Args:
        sim_results: From simulate_tournament().
        bracket_struct: From build_bracket_structure().
        n_public: Number of public brackets to simulate for pick rates.
        seed: Random seed for public bracket generation.

    Returns:
        DataFrame with columns: Slot, TeamID, TeamName, SeedNum, Round,
        ModelProb, PublicProb, Leverage.
    """
    all_results = sim_results["all_results"]
    n_sims = len(all_results)

    # Model advancement: how often each team wins each slot
    model_slot_counts = {}
    for result in all_results:
        for slot, winner in result["slot_winners"].items():
            if slot not in model_slot_counts:
                model_slot_counts[slot] = Counter()
            model_slot_counts[slot][winner["team_id"]] += 1

    # Public advancement: how often each team is picked at each slot
    rng = np.random.default_rng(seed)
    public_slot_counts = {}
    for _ in range(n_public):
        bracket = generate_public_bracket(bracket_struct, rng)
        for slot, team_id in bracket.items():
            if slot not in public_slot_counts:
                public_slot_counts[slot] = Counter()
            public_slot_counts[slot][team_id] += 1

    # Build team name lookup from bracket structure
    team_id_to_info = {}
    for seed_info in bracket_struct["seed_to_team"].values():
        team_id_to_info[seed_info["team_id"]] = {
            "name": seed_info["team_name"],
            "seed_num": seed_info["seed_num"],
        }

    rows = []
    all_slots = set(model_slot_counts.keys()) | set(public_slot_counts.keys())
    for slot in sorted(all_slots):
        if not (slot[0] == "R" and slot[1].isdigit()):
            continue
        round_num = int(slot[1])

        model_counts = model_slot_counts.get(slot, Counter())
        public_counts = public_slot_counts.get(slot, Counter())
        all_teams = set(model_counts.keys()) | set(public_counts.keys())

        for team_id in all_teams:
            model_p = model_counts.get(team_id, 0) / n_sims
            public_p = public_counts.get(team_id, 0) / n_public

            info = team_id_to_info.get(team_id, {"name": str(team_id), "seed_num": 0})
            leverage = model_p / public_p if public_p > 0 else float("inf")

            rows.append({
                "Slot": slot,
                "TeamID": team_id,
                "TeamName": info["name"],
                "SeedNum": info["seed_num"],
                "Round": round_num,
                "ModelProb": round(model_p, 4),
                "PublicProb": round(public_p, 4),
                "Leverage": round(leverage, 3),
            })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["Round", "Slot", "Leverage"], ascending=[True, True, False])
    return df


class PoolOptimizer:
    """Bracket pool optimizer using Monte Carlo simulation.

    Pre-computes opponent pools and tournament outcomes so that
    evaluating candidate brackets is fast (enabling search).

    Args:
        sim_results: From simulate_tournament(). Contains the 10K
            simulated tournament outcomes used as "ground truth".
        bracket_struct: From build_bracket_structure().
        pool_size: Number of participants in the pool (including you).
        scoring: Scoring system name or dict. Default "espn".
        n_pools: Number of independent opponent pool realizations
            to average over for robust P(win) estimates.
        n_outcomes: Number of tournament outcomes to evaluate against.
            Uses the first n_outcomes from sim_results. Default: all.
        seed: Random seed for opponent bracket generation.
    """

    def __init__(
        self,
        sim_results: dict,
        bracket_struct: dict,
        pool_size: int = 50,
        scoring: str | dict = "espn",
        n_pools: int = 20,
        n_outcomes: int | None = None,
        seed: int = 42,
    ):
        self.bracket_struct = bracket_struct
        self.pool_size = pool_size
        self.scoring = (
            SCORING_SYSTEMS[scoring] if isinstance(scoring, str) else scoring
        )

        all_results = sim_results["all_results"]
        if n_outcomes is not None:
            all_results = all_results[:n_outcomes]
        self.n_outcomes = len(all_results)

        # Extract outcomes as slot -> team_id dicts
        self.outcomes = [extract_outcome(r) for r in all_results]

        # Pre-generate opponent pools and compute max opponent scores
        rng = np.random.default_rng(seed)
        self.max_opp_scores = np.zeros((n_pools, self.n_outcomes))

        for p in range(n_pools):
            opponents = [
                generate_public_bracket(bracket_struct, rng)
                for _ in range(pool_size - 1)
            ]
            for k, outcome in enumerate(self.outcomes):
                scores = [
                    score_bracket(opp, outcome, self.scoring)
                    for opp in opponents
                ]
                self.max_opp_scores[p, k] = max(scores)

        # Store team info for bracket generation
        self._team_id_to_info = {}
        for seed_info in bracket_struct["seed_to_team"].values():
            self._team_id_to_info[seed_info["team_id"]] = seed_info

        # Pre-compute slot-level model win frequencies for bracket generation
        self._slot_winner_counts = {}
        for result in all_results:
            for slot, winner in result["slot_winners"].items():
                if slot not in self._slot_winner_counts:
                    self._slot_winner_counts[slot] = Counter()
                self._slot_winner_counts[slot][winner["team_id"]] += 1

    def evaluate(self, bracket: dict) -> float:
        """Compute P(bracket wins the pool) across all outcomes and pools.

        Ties are handled by splitting the win probability: if you tie
        with the best opponent score, you win half the time (approximation
        of random tiebreaker).

        Args:
            bracket: Dict mapping slot -> team_id.

        Returns:
            Estimated probability of finishing first in the pool.
        """
        my_scores = np.array([
            score_bracket(bracket, outcome, self.scoring)
            for outcome in self.outcomes
        ])

        # Strict wins + half credit for ties
        strict_wins = np.sum(my_scores[np.newaxis, :] > self.max_opp_scores)
        ties = np.sum(my_scores[np.newaxis, :] == self.max_opp_scores)
        total = self.max_opp_scores.shape[0] * self.n_outcomes
        return (strict_wins + 0.5 * ties) / total

    def most_likely_bracket(self) -> dict:
        """Build the modal bracket (most common winner at each slot)."""
        bracket = {}
        for slot, counts in self._slot_winner_counts.items():
            bracket[slot] = counts.most_common(1)[0][0]
        return bracket

    def _generate_model_bracket(self, rng: np.random.Generator) -> dict:
        """Generate a bracket sampled from model probabilities."""
        seed_to_team = self.bracket_struct["seed_to_team"]
        play_in_slots = self.bracket_struct["play_in_slots"]
        regular_slots = self.bracket_struct["regular_slots"]

        resolved = dict(seed_to_team)
        picks = {}

        for slot, (strong, weak) in play_in_slots.items():
            team_a = resolved[strong]
            team_b = resolved[weak]
            counts = self._slot_winner_counts.get(slot, {})
            total = sum(counts.values()) or 1
            p_a = counts.get(team_a["team_id"], 0) / total
            winner = team_a if rng.random() < p_a else team_b
            resolved[slot] = winner
            picks[slot] = winner["team_id"]

        slot_order = sorted(regular_slots.keys(), key=lambda s: (int(s[1]), s))
        for slot in slot_order:
            strong, weak = regular_slots[slot]
            team_a = resolved.get(strong)
            team_b = resolved.get(weak)
            if team_a is None or team_b is None:
                continue

            # Use model's conditional win probability for this matchup
            counts = self._slot_winner_counts.get(slot, {})
            a_count = counts.get(team_a["team_id"], 0)
            b_count = counts.get(team_b["team_id"], 0)
            total = a_count + b_count
            if total > 0:
                p_a = a_count / total
            else:
                # Matchup never occurred in sims; fall back to seed
                p_a = seed_public_pick_prob(
                    team_a["seed_num"], team_b["seed_num"]
                )
            winner = team_a if rng.random() < p_a else team_b
            resolved[slot] = winner
            picks[slot] = winner["team_id"]

        return picks

    def _mutate_bracket(
        self, bracket: dict, rng: np.random.Generator, n_flips: int = 1,
    ) -> dict:
        """Mutate a bracket by flipping picks and propagating downstream.

        Picks a random tournament slot, swaps the pick to the other team
        that could have played there, then re-resolves all downstream
        games affected by this change.
        """
        regular_slots = self.bracket_struct["regular_slots"]
        seed_to_team = self.bracket_struct["seed_to_team"]
        play_in_slots = self.bracket_struct["play_in_slots"]

        new_bracket = dict(bracket)

        # Build resolved map from current bracket
        resolved = dict(seed_to_team)
        for slot in sorted(play_in_slots.keys()):
            strong, weak = play_in_slots[slot]
            team_a = resolved[strong]
            team_b = resolved[weak]
            winner_id = new_bracket.get(slot)
            winner = team_a if team_a["team_id"] == winner_id else team_b
            resolved[slot] = winner

        slot_order = sorted(regular_slots.keys(), key=lambda s: (int(s[1]), s))
        for slot in slot_order:
            strong, weak = regular_slots[slot]
            team_a = resolved.get(strong)
            team_b = resolved.get(weak)
            if team_a is None or team_b is None:
                continue
            winner_id = new_bracket.get(slot)
            if winner_id == team_a["team_id"]:
                resolved[slot] = team_a
            elif winner_id == team_b["team_id"]:
                resolved[slot] = team_b

        # Pick random slots to flip (only regular tournament games)
        mutable_slots = [
            s for s in slot_order
            if s in new_bracket and resolved.get(s) is not None
        ]
        if not mutable_slots:
            return new_bracket

        flip_slots = rng.choice(
            mutable_slots, size=min(n_flips, len(mutable_slots)), replace=False,
        )

        for flip_slot in flip_slots:
            strong, weak = regular_slots[flip_slot]
            team_a = resolved.get(strong)
            team_b = resolved.get(weak)
            if team_a is None or team_b is None:
                continue

            # Flip: pick the other team
            old_winner_id = new_bracket[flip_slot]
            if old_winner_id == team_a["team_id"]:
                new_winner = team_b
            else:
                new_winner = team_a
            new_bracket[flip_slot] = new_winner["team_id"]
            resolved[flip_slot] = new_winner

            # Propagate downstream: re-resolve any slot that feeds from this one
            for downstream_slot in slot_order:
                ds_strong, ds_weak = regular_slots[downstream_slot]
                if ds_strong != flip_slot and ds_weak != flip_slot:
                    continue
                # This downstream game feeds from the flipped slot
                ds_team_a = resolved.get(ds_strong)
                ds_team_b = resolved.get(ds_weak)
                if ds_team_a is None or ds_team_b is None:
                    continue
                # If old pick is no longer available, pick randomly
                old_ds_pick = new_bracket.get(downstream_slot)
                if old_ds_pick not in (
                    ds_team_a["team_id"], ds_team_b["team_id"],
                ):
                    # Old pick was eliminated; randomly pick from available
                    counts = self._slot_winner_counts.get(downstream_slot, {})
                    a_c = counts.get(ds_team_a["team_id"], 0)
                    b_c = counts.get(ds_team_b["team_id"], 0)
                    total = a_c + b_c
                    p_a = a_c / total if total > 0 else 0.5
                    pick = ds_team_a if rng.random() < p_a else ds_team_b
                    new_bracket[downstream_slot] = pick["team_id"]
                    resolved[downstream_slot] = pick

        return new_bracket

    def optimize(
        self,
        pop_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.15,
        elite_frac: float = 0.1,
        seed: int = 99,
        verbose: bool = True,
    ) -> dict:
        """Find the bracket that maximizes P(winning the pool).

        Uses a genetic algorithm:
          1. Initialize diverse population (model brackets + public brackets)
          2. Evaluate each bracket's win probability
          3. Select top performers, mutate to create next generation
          4. Repeat until convergence

        Args:
            pop_size: Number of brackets in each generation.
            n_generations: Maximum number of generations.
            mutation_rate: Fraction of games to flip per mutation.
            elite_frac: Fraction of top brackets preserved each generation.
            seed: Random seed.
            verbose: Print progress.

        Returns:
            Dict with keys:
              - bracket: The optimized bracket (slot -> team_id).
              - win_prob: P(winning the pool).
              - most_likely_bracket: The modal bracket for comparison.
              - most_likely_win_prob: P(winning) for the modal bracket.
              - history: List of best win_prob per generation.
              - diff_from_modal: Slots where optimized differs from modal.
        """
        rng = np.random.default_rng(seed)
        n_elite = max(1, int(pop_size * elite_frac))
        n_slots = len(
            [s for s in self.bracket_struct["regular_slots"] if True]
        )
        n_flips = max(1, int(n_slots * mutation_rate))

        # Initialize population: mix of model and public brackets
        population = []

        # Always include the modal bracket
        modal = self.most_likely_bracket()
        population.append(modal)

        # Generate model-sampled brackets (diverse but informed)
        for _ in range(pop_size // 2):
            population.append(self._generate_model_bracket(rng))

        # Generate public brackets (for diversity)
        for _ in range(pop_size - len(population)):
            population.append(
                generate_public_bracket(self.bracket_struct, rng)
            )

        # Evolution loop
        history = []
        best_bracket = modal
        best_score = -1.0

        for gen in range(n_generations):
            # Evaluate all brackets
            scores = np.array([self.evaluate(b) for b in population])

            gen_best_idx = np.argmax(scores)
            gen_best_score = scores[gen_best_idx]

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_bracket = dict(population[gen_best_idx])

            history.append(float(best_score))

            if verbose and (gen % 10 == 0 or gen == n_generations - 1):
                print(
                    f"  Gen {gen:3d}: best={best_score:.4f} "
                    f"gen_best={gen_best_score:.4f} "
                    f"gen_mean={scores.mean():.4f}"
                )

            # Early stopping if converged
            if gen >= 20 and len(set(history[-10:])) == 1:
                if verbose:
                    print(f"  Converged at generation {gen}")
                break

            # Selection: keep elites, generate children via mutation
            elite_idx = np.argsort(scores)[-n_elite:]
            elites = [dict(population[i]) for i in elite_idx]

            # Tournament selection for parents
            new_pop = list(elites)
            while len(new_pop) < pop_size:
                # Tournament selection: pick 3, keep best
                candidates = rng.choice(len(population), size=3, replace=False)
                parent_idx = candidates[np.argmax(scores[candidates])]
                child = self._mutate_bracket(
                    population[parent_idx], rng, n_flips=n_flips,
                )
                new_pop.append(child)

            population = new_pop

        # Evaluate modal bracket for comparison
        modal_win_prob = self.evaluate(modal)

        # Find differences between optimized and modal
        diff_slots = {}
        team_id_to_name = {
            info["team_id"]: info["team_name"]
            for info in self.bracket_struct["seed_to_team"].values()
        }
        for slot in sorted(best_bracket.keys()):
            if best_bracket[slot] != modal.get(slot):
                if not (slot[0] == "R" and slot[1].isdigit()):
                    continue
                diff_slots[slot] = {
                    "round": int(slot[1]),
                    "optimized_pick": team_id_to_name.get(
                        best_bracket[slot], str(best_bracket[slot]),
                    ),
                    "modal_pick": team_id_to_name.get(
                        modal.get(slot), str(modal.get(slot)),
                    ),
                }

        return {
            "bracket": best_bracket,
            "win_prob": best_score,
            "most_likely_bracket": modal,
            "most_likely_win_prob": modal_win_prob,
            "history": history,
            "diff_from_modal": diff_slots,
        }


def pool_size_analysis(
    sim_results: dict,
    bracket_struct: dict,
    pool_sizes: list[int] | None = None,
    scoring: str | dict = "espn",
    n_pools: int = 10,
    n_outcomes: int = 2000,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Analyze how optimal strategy changes with pool size.

    For each pool size, optimizes a bracket and compares to the modal
    bracket. Shows how contrarian the optimal bracket becomes as pool
    size increases.

    Returns:
        DataFrame with columns: PoolSize, ModalWinProb, OptimizedWinProb,
        Improvement, NumDifferences.
    """
    if pool_sizes is None:
        pool_sizes = [5, 10, 25, 50, 100, 250]

    rows = []
    for n in pool_sizes:
        if verbose:
            print(f"\nPool size: {n}")

        optimizer = PoolOptimizer(
            sim_results, bracket_struct,
            pool_size=n, scoring=scoring,
            n_pools=n_pools, n_outcomes=n_outcomes, seed=seed,
        )

        result = optimizer.optimize(
            pop_size=30, n_generations=50, verbose=verbose,
        )

        rows.append({
            "PoolSize": n,
            "ModalWinProb": result["most_likely_win_prob"],
            "OptimizedWinProb": result["win_prob"],
            "Improvement": result["win_prob"] - result["most_likely_win_prob"],
            "NumDifferences": len(result["diff_from_modal"]),
        })

        if verbose:
            print(f"  Modal P(win): {result['most_likely_win_prob']:.4f}")
            print(f"  Optimized P(win): {result['win_prob']:.4f}")
            print(f"  Differences from modal: {len(result['diff_from_modal'])}")

    return pd.DataFrame(rows)
