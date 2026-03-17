# Offense/Defense Reparameterization — Implementation Summary

## Overview

Successor to the single-parameter Bradley-Terry model. Decomposes team strength into separate **offense** and **defense** parameters, modeling individual team scores rather than score margin. Uses a multivariate normal prior with LKJ correlation to capture the relationship between offensive and defensive ability. Supports both Gaussian and Student-t likelihoods for comparing blowout robustness.

---

## Model Specification (`src/model.py`)

### `build_offense_defense_model(data, likelihood="normal")`

```
# Global intercept (average score per team per game)
mu ~ Normal(70, 10)

# Conference-level effects (independent for offense/defense)
mu_off_conf[c] ~ Normal(0, sigma_off_conf)
mu_def_conf[c] ~ Normal(0, sigma_def_conf)
sigma_off_conf ~ HalfNormal(5)
sigma_def_conf ~ HalfNormal(5)

# Team-level offense/defense (multivariate, non-centered)
L, corr, stds ~ LKJCholeskyCov(n=2, eta=2, sd_dist=HalfNormal(5))
z_i ~ Normal(0, 1)    shape: (n_teams, 2)

[off_i, def_i] = [mu_off_conf[c_i], mu_def_conf[c_i]] + z_i @ L'

# Home court advantage
alpha ~ Normal(3.5, 2)

# Observation noise
sigma ~ HalfNormal(15)

# Likelihood (two observations per game):
score_i ~ Normal(mu + off_i - def_j + alpha * home_i, sigma)
score_j ~ Normal(mu + off_j - def_i - alpha * home_i, sigma)
```

Student-t variant replaces Normal likelihood with `StudentT(nu, ...)` where `nu ~ Gamma(2, 0.1)`.

### Design Decisions

- **Sign convention**: Positive `off_i` = strong offense (scores more), positive `def_i` = strong defense (allows fewer). Score equation: `mu + off_i - def_j`.
- **Identifiability**: Global intercept `mu` pins the scale. Offense and defense parameters centered near zero through hierarchical structure.
- **Two observations per game**: Each game produces `score_i` and `score_j`, modeled as conditionally independent given parameters. No within-game pace/tempo latent — offense/defense parameters absorb the meaningful variation.
- **LKJ(eta=2)**: Weakly informative prior favoring small correlations but permitting anything. Non-centered parameterization: raw `z ~ Normal(0,1)` offsets transformed via Cholesky factor.
- **`sd_dist=HalfNormal.dist(5)`**: Uses the `.dist()` API for an unregistered distribution, as required by `pm.LKJCholeskyCov`.

### Win Probability (Probit Formula)

```
diff = (off_i - def_j) - (off_j - def_i)
P(i beats j) = Phi(diff / (sigma * sqrt(2)))
```

The `sqrt(2)` arises because `sigma` is per-score noise. Since `Var(score_i - score_j) = 2 * sigma^2` for conditionally independent scores, the probit denominator is `sigma * sqrt(2)`. This is not a correction to the old model — `sigma` has a different semantic meaning between the margin-based and score-based models.

---

## Data Pipeline Changes (`src/data.py`)

`build_model_data()` now returns two additional arrays:

- `score_i`: individual score for team_i (lower-ID team, consistent with existing convention)
- `score_j`: individual score for team_j

Existing `margin` field preserved: `margin = score_i - score_j`.

---

## Diagnostics and Strengths (`src/model.py`)

### `check_diagnostics(idata)`

Auto-detects model type by checking for `"off"` in posterior variables. For offense/defense models:
- Checks `off` and `def` jointly for R-hat and ESS
- Checks LKJ correlation convergence
- Includes `mu_intercept`, conference sigmas, and `nu` (Student-t) in hyperparameter summary
- Same thresholds: R-hat < 1.01, ESS > 400, divergences < 10
- Backward-compatible: still works on theta-based models

### `get_team_strengths(idata, data)`

For offense/defense models, returns:
- `off_samples`, `def_samples`: full posterior arrays (n_samples × n_teams)
- `overall_means`: `off + def` averaged over posterior (used for ranking)
- `off_means`, `def_means`: marginal posterior means
- `off_def_corr`: posterior samples of the off/def correlation from LKJ
- `ranking`: teams sorted by overall strength
- Backward compat aliases: `samples`, `means`, `stds`

### `win_probability_samples(...)`

Accepts either `theta_samples` (margin model) or `off_samples + def_samples` (score model). Applies the appropriate probit formula with or without `sqrt(2)`.

---

## Simulation Engine (`src/simulate.py`)

### `_play_game(team_a, team_b, sigma_val, team_id_to_idx, rng, theta=None, off=None, deff=None, ...)`

Signature changed to keyword arguments for model arrays. Internally selects between:
- Theta: `diff = theta[a] - theta[b]`, denominator = `sigma`
- Off/def: `diff = (off[a] - def[b]) - (off[b] - def[a])`, denominator = `sigma * sqrt(2)`

### `simulate_tournament_single(...)`

Now takes `theta`, `off`, `deff` as keyword arguments. Passes them through to `_play_game` via a shared `game_kwargs` dict.

### `simulate_tournament(...)`

Interface change: `theta_samples` moved from positional to keyword. New keywords: `off_samples`, `def_samples`. Per-simulation, extracts the appropriate arrays by sample index.

---

## Submission (`src/submission.py`)

### `generate_submission(men_sigma, men_team_ids, women_sigma, women_team_ids, ...)`

Required arguments are now `sigma` and `team_ids` per gender. Team strength arrays are keyword-only: either `men_theta` or `men_off + men_def` (same for women).

### `_batch_win_probs(...)`

Accepts `theta_samples` or `off_samples + def_samples`. Vectorized computation with `sqrt(2)` denominator for off/def.

---

## Validation (`src/validate.py`)

### `validate_season(season, ..., model_type="bradley_terry")`

New `model_type` parameter: `"bradley_terry"` or `"offense_defense"`. Selects which model to build and which probit formula to use for tournament prediction.

### `run_validation(seasons, ..., model_type="bradley_terry")`

Passes `model_type` through to `validate_season`.

---

## Main Pipeline (`main.py`)

### `run_gender(gender, season, n_sims)`

Now uses `build_offense_defense_model` by default. Extracts `off_samples` and `def_samples` from the posterior. Prints top teams with offense/defense breakdown.

### `compare_likelihoods(data, draws, tune)`

Fits both Normal and Student-t variants on the same data. Computes log-likelihood (nutpie doesn't store it automatically), runs `az.compare()` for LOO-CV, and prints the comparison table.

---

## Visualizations (`src/visualize.py`)

### New Plots

| Function | Description |
|----------|-------------|
| `plot_off_def_scatter(strengths, conf_idx, conf_names)` | Offense vs defense posterior means, colored by top 5 conferences, top 15 teams labeled, posterior correlation annotated |
| `plot_off_def_rankings(strengths, top_n=20)` | Side-by-side forest plots: top 20 by offense, top 20 by defense (94% HDI) |
| `plot_conference_off_def(idata, conf_names)` | Conference offensive vs defensive effects as 2D scatter |
| `plot_loo_comparison(comparison)` | ELPD with error bars from `az.compare()` |
| `plot_posterior_predictive_scores(idata, observed_scores)` | Predicted vs observed score distributions |

### Updated Plots

- `plot_team_strength_forest`: Auto-detects model type. Uses `off + def` for overall strength when offense/defense model is detected.
- `plot_team_strength_posterior`: Still theta-only (violin plot for margin model).
- `plot_conference_effects`: Still uses `mu_conf` (margin model). For offense/defense, use `plot_conference_off_def` instead.

### Unchanged

Advancement heatmap, championship odds, bracket, upset probabilities, calibration, home court comparison — all operate on simulation results, not model parameters.

---

## Tests (`tests/`)

17 tests across 4 files:

| File | Tests | Coverage |
|------|-------|----------|
| `test_data.py` | 2 | score_i/score_j arrays, lower-ID-first convention |
| `test_model.py` | 7 | Model builds (Normal, Student-t), expected variables, logp finite, diagnostics, strengths extraction, win probability (off/def and theta), error on missing samples |
| `test_simulate.py` | 3 | `_play_game` off/def (probabilistic check), theta backward compat, `simulate_tournament` interface |
| `test_submission.py` | 2 | `_batch_win_probs` off/def (exact formula check), theta backward compat |

All tests are unit tests with mock data — no MCMC sampling required. Run with `uv run pytest tests/ -v`.

---

## Backward Compatibility

- `build_bradley_terry()` preserved in `src/model.py`
- All updated functions auto-detect model type from posterior variables or accept both theta and off/def via keyword arguments
- Existing notebooks and the margin-based validation pipeline still work unchanged

---

## File Changes

```
src/data.py       +10 lines   (score_i, score_j arrays)
src/model.py      +189 lines  (new model, updated diagnostics/strengths/win_prob)
src/simulate.py   +30 lines   (off/def support in simulation)
src/submission.py  rewritten   (off/def support, keyword args)
src/validate.py   +20 lines   (model_type parameter)
src/visualize.py  +160 lines  (5 new plot functions, 1 updated)
main.py            rewritten   (offense/defense pipeline, compare_likelihoods)
pyproject.toml    +3 lines    (pytest pythonpath config)
tests/            4 new files (17 tests)
```
