"""Generate Kaggle submission from saved posterior arrays.

Processes matchups in chunks to limit memory.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

CHUNK_SIZE = 5000

template = pd.read_csv("data/kaggle/SampleSubmissionStage2.csv")
parts = template["ID"].str.split("_", expand=True)
id_low = parts[1].astype(int).values
id_high = parts[2].astype(int).values
del parts

predictions = np.full(len(template), 0.5)


def process_gender(mask, npz_path, label):
    if not mask.any():
        return
    data = np.load(npz_path)
    theta = data["theta"]    # (8000, n_teams)
    sigma = data["sigma"]    # (8000,)
    id_to_idx = {int(t): i for i, t in enumerate(data["team_ids"])}
    del data

    low_ids = id_low[mask]
    high_ids = id_high[mask]
    il = np.array([id_to_idx.get(int(t), -1) for t in low_ids])
    ih = np.array([id_to_idx.get(int(t), -1) for t in high_ids])
    valid = (il >= 0) & (ih >= 0)

    # Get global indices where this mask applies
    global_indices = np.where(mask)[0]

    # Process in chunks to avoid huge intermediate arrays
    valid_indices = np.where(valid)[0]
    n_valid = len(valid_indices)
    print(f"{label}: {mask.sum()} matchups, {n_valid} with both teams in model")

    for start in range(0, n_valid, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_valid)
        chunk_idx = valid_indices[start:end]

        chunk_il = il[chunk_idx]
        chunk_ih = ih[chunk_idx]

        # (n_samples, chunk_size)
        diff = theta[:, chunk_il] - theta[:, chunk_ih]
        p = norm.cdf(diff / sigma[:, np.newaxis])
        chunk_probs = p.mean(axis=0)

        # Write back to predictions at the correct global positions
        predictions[global_indices[chunk_idx]] = chunk_probs

        if start % 20000 == 0:
            print(f"  Processed {start + len(chunk_idx)}/{n_valid}")

    del theta, sigma


process_gender(id_low < 3000, "results/mens_posterior.npz", "Men's")
process_gender(id_low >= 3000, "results/womens_posterior.npz", "Women's")

template["Pred"] = predictions
template.to_csv("results/submission.csv", index=False)
print(f"\nSubmission saved: {len(template)} rows")
print(f"  Pred range: [{predictions.min():.4f}, {predictions.max():.4f}]")
print(f"  Mean pred: {predictions.mean():.4f}")
