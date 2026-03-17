"""Generate Kaggle submission CSV with predictions for all matchups."""

import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path


def generate_submission(
    men_sigma: np.ndarray,
    men_team_ids: np.ndarray,
    women_sigma: np.ndarray,
    women_team_ids: np.ndarray,
    men_theta: np.ndarray | None = None,
    men_off: np.ndarray | None = None,
    men_def: np.ndarray | None = None,
    women_theta: np.ndarray | None = None,
    women_off: np.ndarray | None = None,
    women_def: np.ndarray | None = None,
    template_path: str = "data/kaggle/SampleSubmissionStage2.csv",
    output_path: str = "results/submission.csv",
) -> pd.DataFrame:
    """Generate Kaggle submission from posterior samples for both genders.

    Accepts either theta_samples (margin model) or off/def samples (score model).
    """
    template = pd.read_csv(template_path)

    parts = template["ID"].str.split("_", expand=True)
    id_low = parts[1].astype(int).values
    id_high = parts[2].astype(int).values

    men_id_to_idx = {int(tid): i for i, tid in enumerate(men_team_ids)}
    women_id_to_idx = {int(tid): i for i, tid in enumerate(women_team_ids)}

    is_womens = id_low >= 3000
    predictions = np.full(len(template), 0.5)

    men_mask = ~is_womens
    if men_mask.any():
        predictions[men_mask] = _batch_win_probs(
            id_low[men_mask], id_high[men_mask], men_sigma, men_id_to_idx,
            theta_samples=men_theta, off_samples=men_off, def_samples=men_def,
        )

    if is_womens.any():
        predictions[is_womens] = _batch_win_probs(
            id_low[is_womens], id_high[is_womens], women_sigma, women_id_to_idx,
            theta_samples=women_theta, off_samples=women_off, def_samples=women_def,
        )

    template["Pred"] = predictions

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(template)} rows)")

    return template


def _batch_win_probs(
    id_low: np.ndarray,
    id_high: np.ndarray,
    sigma_samples: np.ndarray,
    id_to_idx: dict,
    theta_samples: np.ndarray | None = None,
    off_samples: np.ndarray | None = None,
    def_samples: np.ndarray | None = None,
) -> np.ndarray:
    """Compute P(team_low beats team_high) for arrays of matchups.

    For teams not in the model (didn't play regular season), returns 0.5.
    """
    n_matchups = len(id_low)
    probs = np.full(n_matchups, 0.5)

    idx_low = np.array([id_to_idx.get(int(t), -1) for t in id_low])
    idx_high = np.array([id_to_idx.get(int(t), -1) for t in id_high])

    valid = (idx_low >= 0) & (idx_high >= 0)
    if not valid.any():
        return probs

    vl = idx_low[valid]
    vh = idx_high[valid]

    if off_samples is not None and def_samples is not None:
        off_low = off_samples[:, vl]
        off_high = off_samples[:, vh]
        def_low = def_samples[:, vl]
        def_high = def_samples[:, vh]
        diff = (off_low - def_high) - (off_high - def_low)
        p_samples = norm.cdf(diff / (sigma_samples[:, np.newaxis] * np.sqrt(2)))
    else:
        theta_low = theta_samples[:, vl]
        theta_high = theta_samples[:, vh]
        diff = theta_low - theta_high
        p_samples = norm.cdf(diff / sigma_samples[:, np.newaxis])

    probs[valid] = p_samples.mean(axis=0)
    return probs
