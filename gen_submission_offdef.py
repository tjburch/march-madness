"""Generate submission from saved offense/defense model results, one gender at a time."""

import gc
import numpy as np
import pandas as pd
import arviz as az
from itertools import combinations
from scipy.stats import norm
from pathlib import Path

from src.data import build_model_data
from src.model import get_team_strengths


def compute_gender_predictions(gender: str):
    """Load model, compute all pairwise predictions for one gender, then free memory."""
    suffix = "mens" if gender == "M" else "womens"
    nc_path = f"results/model_2026_{suffix}_offdef.nc"
    if not Path(nc_path).exists():
        nc_path = f"results/model_2026_{suffix}_offdef_normal.nc"

    print(f"Loading {suffix} model from {nc_path}...")
    idata = az.from_netcdf(nc_path)
    data = build_model_data(2026, gender)

    off = idata.posterior["off"].values.reshape(-1, data["n_teams"])
    deff = idata.posterior["def"].values.reshape(-1, data["n_teams"])
    sigma = idata.posterior["sigma"].values.flatten()

    team_ids = sorted(int(t) for t in data["team_ids"])
    id_to_idx = {int(tid): i for i, tid in enumerate(data["team_ids"])}

    # Free idata early
    del idata
    gc.collect()

    # Generate pairings and compute predictions in chunks
    pairs = list(combinations(team_ids, 2))
    print(f"  {len(pairs):,} pairings")

    chunk_size = 5000
    rows = []
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start:start + chunk_size]
        id_low = np.array([p[0] for p in chunk])
        id_high = np.array([p[1] for p in chunk])

        idx_low = np.array([id_to_idx.get(int(t), -1) for t in id_low])
        idx_high = np.array([id_to_idx.get(int(t), -1) for t in id_high])

        valid = (idx_low >= 0) & (idx_high >= 0)
        probs = np.full(len(chunk), 0.5)

        if valid.any():
            vl = idx_low[valid]
            vh = idx_high[valid]
            off_low = off[:, vl]
            off_high = off[:, vh]
            def_low = deff[:, vl]
            def_high = deff[:, vh]
            diff = (off_low - def_high) - (off_high - def_low)
            p_samples = norm.cdf(diff / (sigma[:, np.newaxis] * np.sqrt(2)))
            probs[valid] = p_samples.mean(axis=0)

        for i, (lo, hi) in enumerate(chunk):
            rows.append({"ID": f"2026_{lo}_{hi}", "Pred": probs[i]})

    # Free model arrays
    del off, deff, sigma
    gc.collect()

    return rows


if __name__ == "__main__":
    all_rows = []

    for gender in ["M", "W"]:
        rows = compute_gender_predictions(gender)
        all_rows.extend(rows)
        gc.collect()

    df = pd.DataFrame(all_rows)
    output_path = "results/submission_offdef.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSubmission shape: {df.shape}")
    print(f"Prediction range: [{df['Pred'].min():.4f}, {df['Pred'].max():.4f}]")
    print(f"Mean prediction: {df['Pred'].mean():.4f}")
    print(f"Saved to {output_path} (NOT submitted)")
