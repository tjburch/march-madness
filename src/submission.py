"""Generate Kaggle submission CSV with predictions for all matchups."""

import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path


def generate_submission(
    men_theta: np.ndarray,
    men_sigma: np.ndarray,
    men_team_ids: np.ndarray,
    women_theta: np.ndarray,
    women_sigma: np.ndarray,
    women_team_ids: np.ndarray,
    template_path: str = "data/kaggle/SampleSubmissionStage2.csv",
    output_path: str = "results/submission.csv",
) -> pd.DataFrame:
    """Generate Kaggle submission from posterior samples for both genders.

    For each matchup ID (2026_TeamLow_TeamHigh), compute:
        P(TeamLow beats TeamHigh) = mean over posterior of Phi((theta_low - theta_high) / sigma)

    Vectorized: precompute team index lookups, then batch-compute probabilities.
    """
    template = pd.read_csv(template_path)

    # Parse IDs
    parts = template["ID"].str.split("_", expand=True)
    id_low = parts[1].astype(int).values
    id_high = parts[2].astype(int).values

    # Build index maps
    men_id_to_idx = {int(tid): i for i, tid in enumerate(men_team_ids)}
    women_id_to_idx = {int(tid): i for i, tid in enumerate(women_team_ids)}

    # Determine gender mask (women's team IDs start at 3000)
    is_womens = id_low >= 3000

    predictions = np.full(len(template), 0.5)

    # Process men's matchups
    men_mask = ~is_womens
    if men_mask.any():
        men_low = id_low[men_mask]
        men_high = id_high[men_mask]
        predictions[men_mask] = _batch_win_probs(
            men_low, men_high, men_theta, men_sigma, men_id_to_idx
        )

    # Process women's matchups
    if is_womens.any():
        w_low = id_low[is_womens]
        w_high = id_high[is_womens]
        predictions[is_womens] = _batch_win_probs(
            w_low, w_high, women_theta, women_sigma, women_id_to_idx
        )

    template["Pred"] = predictions

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(template)} rows)")

    return template


def _batch_win_probs(
    id_low: np.ndarray,
    id_high: np.ndarray,
    theta_samples: np.ndarray,
    sigma_samples: np.ndarray,
    id_to_idx: dict,
) -> np.ndarray:
    """Compute P(team_low beats team_high) for arrays of matchups.

    For teams not in the model (didn't play regular season), returns 0.5.
    """
    n_matchups = len(id_low)
    probs = np.full(n_matchups, 0.5)

    # Map team IDs to model indices
    idx_low = np.array([id_to_idx.get(int(t), -1) for t in id_low])
    idx_high = np.array([id_to_idx.get(int(t), -1) for t in id_high])

    # Only compute for matchups where both teams are in the model
    valid = (idx_low >= 0) & (idx_high >= 0)
    if not valid.any():
        return probs

    # Vectorized: for each valid matchup, compute mean over posterior samples
    # theta_samples shape: (n_samples, n_teams)
    # sigma_samples shape: (n_samples,)
    vl = idx_low[valid]
    vh = idx_high[valid]

    # Extract theta columns for all valid matchups at once
    theta_low = theta_samples[:, vl]   # (n_samples, n_valid)
    theta_high = theta_samples[:, vh]  # (n_samples, n_valid)
    diff = theta_low - theta_high      # (n_samples, n_valid)

    # Divide by sigma (broadcast: n_samples,1 / n_samples -> n_samples,n_valid)
    p_samples = norm.cdf(diff / sigma_samples[:, np.newaxis])
    probs[valid] = p_samples.mean(axis=0)

    return probs
