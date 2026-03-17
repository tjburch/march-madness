# Dashboard Updates for Offense/Defense Reparameterization

**Companion to**: `docs/specs/03_dashboard_spec.md`

This document covers only the changes required to the dashboard and data pipeline after reparameterizing team strength from a single `theta` parameter into separate `off` (offensive) and `def` (defensive) components. Everything in spec 03 not mentioned here is unchanged.

---

## 1. Snapshot JSON Schema Changes

### `teams` entries

Replace `theta_mean`/`theta_std` with per-component posteriors plus a derived overall:

```json
"1181": {
  "name": "Duke",
  "seed": "W01",
  "seed_num": 1,
  "region": "W",
  "conference": "ACC",
  "off_mean": 14.2,
  "off_std": 1.3,
  "def_mean": 8.3,
  "def_std": 1.1,
  "overall_mean": 22.5,
  "overall_std": 1.6,
  "eliminated": false,
  "eliminated_round": null
}
```

`overall_mean = off_mean + def_mean`. `overall_std` is computed from the joint posterior samples (not the naive sum of standard deviations, since the components are correlated within the hierarchical model).

### `hyperparameters` section

```json
"hyperparameters": {
  "alpha_mean": 2.78,
  "alpha_std": 0.17,
  "sigma_mean": 10.89,
  "sigma_std": 0.11,
  "sigma_off_conf_mean": 6.12,
  "sigma_def_conf_mean": 5.44,
  "mu_intercept_mean": 0.41,
  "off_def_correlation": 0.384
}
```

Changes from spec 03:
- `sigma_conf_mean` → `sigma_off_conf_mean` + `sigma_def_conf_mean`
- `sigma_team_mean` removed (absorbed into off/def conf scales)
- Add `mu_intercept_mean` (posterior mean of the shared intercept)
- Add `off_def_correlation`: marginal Pearson r between off and def means across all tournament teams (pre-computed scalar, not a posterior quantity)

### Win probability formula note

The `p_a_wins` field in bracket slots is computed from a score-margin model using a `sqrt(2)` normalization in the denominator. The formula is unchanged in the JSON — it is still a probability in [0, 1] — but `src/export.py` must draw from `off_samples` and `def_samples` rather than `theta_samples` when computing it. See §6 below.

---

## 2. New Section: Offense/Defense Scatter Plot

Add a new dashboard section between the Championship Odds chart and the Advancement Heatmap. This is the marquee visualization for the reparameterized model.

### What it shows

Interactive Plotly scatter plot of all 68 tournament teams:
- x-axis: `off_mean` (offensive strength)
- y-axis: `def_mean` (defensive strength)
- Each point colored by conference
- Hover tooltip: team name, seed, conference, off/def/overall values
- Axis labels: "Offensive Strength" / "Defensive Strength"

### Simpson's Paradox annotation

The marginal correlation across all teams is +0.384 (stored as `off_def_correlation` in hyperparameters). Within conferences, the correlation is typically negative (teams that outscore opponents tend to do so through offense or through defense, not both equally). The annotation should make this visible:

- Add a note on the plot: `"Marginal r = +0.384 (between conferences)"`
- Add a short plain-text caption below the chart: *"Strong teams tend to rank highly in both offense and defense, but within a conference the relationship reverses — teams that dominate offensively are less likely to lead defensively. This is Simpson's paradox."*

A tooltip on the annotation text is not necessary; the caption handles the explanation.

### Conference coloring

Use Plotly's built-in discrete color sequences (e.g., `plotly.colors.qualitative.Plotly`). Do not attempt to use school colors here — conference coloring is the right grouping for showing the within/between paradox. Add a legend for conferences.

### Plotly implementation sketch

```javascript
Plotly.newPlot('offdef-scatter', [{
  type: 'scatter',
  mode: 'markers+text',
  x: offMeans,          // array
  y: defMeans,          // array
  text: teamLabels,     // short name or seed+name
  textposition: 'top center',
  marker: {
    color: conferenceColors,  // per-team, from conference index
    size: 10,
  },
  hovertemplate: '<b>%{customdata[0]}</b><br>Seed: %{customdata[1]}<br>' +
    'Off: %{x:.1f} ± %{customdata[2]:.1f}<br>' +
    'Def: %{y:.1f} ± %{customdata[3]:.1f}<br>' +
    'Overall: %{customdata[4]:.1f}<extra></extra>',
  customdata: hoverData,  // [name, seed, off_std, def_std, overall_mean]
}], layout);
```

Text labels clutter quickly at 68 teams — consider showing labels only on hover, or only for top 16 seeds by default with a toggle to show all.

---

## 3. Team Deep Dive Updates

When a team is selected (click from any chart), the detail panel should show two posterior histograms side-by-side instead of the single theta histogram from spec 03.

### Layout

```
[ Offensive Strength ]   [ Defensive Strength ]
      histogram                 histogram
```

Both histograms drawn with Plotly using the team's `primary_color` (offense) and a muted variant (defense). X-axis labels: "Offensive Strength (pts)" and "Defensive Strength (pts)".

### Data

The histograms require per-sample data, not just mean/std. Two options:

**Option A (preferred)**: Pre-compute histogram bins in `export.py` and store them in the snapshot alongside the mean/std. Add `off_hist` and `def_hist` arrays (20–30 bins each) to each team entry. This avoids loading posterior NetCDF files client-side. Adds ~2 KB per team, ~136 KB per snapshot — within the < 200 KB target (the original spec's 50 KB target was for a simpler model; this is an acceptable increase).

**Option B**: Draw the histogram from a normal approximation using `off_mean` and `off_std` client-side. Loses the true posterior shape but avoids the size increase.

Recommendation: use Option A if the snapshot stays under 200 KB; fall back to Option B otherwise.

### Conference context line

Optionally, overlay a vertical line on each histogram at the conference mean (`mu_off_conf` / `mu_def_conf` for that team's conference). This provides immediate visual context — is this team above or below their conference average offensively vs. defensively?

---

## 4. Conference Effects Section

Add a small table or bar chart below the Offense/Defense scatter showing per-conference posterior means. This replaces the single `mu_conf` visualization that would have been natural in the original model.

### Data to store

In `export.py`, extract `mu_off_conf` and `mu_def_conf` posterior means by conference and store them in the snapshot under a new top-level key:

```json
"conference_effects": {
  "ACC": {"mu_off": 12.4, "mu_def": 9.1},
  "Big Ten": {"mu_off": 11.8, "mu_def": 9.6},
  "SEC": {"mu_off": 13.1, "mu_def": 8.7}
}
```

Only include conferences with at least one tournament team in the snapshot.

### Dashboard display

A grouped bar chart (or a small table) showing each conference's offensive and defensive mean. Sort by overall (`mu_off + mu_def`) descending. Use two colors: one for offense, one for defense — consistent with the scatter plot color scheme, not school colors.

This section can be collapsed by default (a `<details>` element works fine here) since it is secondary to the team-level scatter.

---

## 5. `src/export.py` Changes

### `export_snapshot` signature

```python
def export_snapshot(
    data: dict,
    off_samples: np.ndarray,    # (n_samples, n_teams) — replaces theta_samples
    def_samples: np.ndarray,    # (n_samples, n_teams) — new
    sigma_samples: np.ndarray,
    sim_results: dict,
    bracket_struct: dict,
    date: str,
    actual_results: dict | None = None,
    output_dir: Path = Path("."),
    alpha_samples: np.ndarray | None = None,
    idata=None,
) -> Path:
```

### Team entry construction

```python
off = off_samples[:, idx]
def_ = def_samples[:, idx]
overall = off + def_

teams[str(tid)] = {
    "name": ...,
    "seed": ...,
    "seed_num": ...,
    "region": ...,
    "conference": ...,
    "off_mean": round(float(off.mean()), 2),
    "off_std": round(float(off.std()), 2),
    "def_mean": round(float(def_.mean()), 2),
    "def_std": round(float(def_.std()), 2),
    "overall_mean": round(float(overall.mean()), 2),
    "overall_std": round(float(overall.std()), 2),
    "eliminated": False,
    "eliminated_round": None,
}
```

If using histogram bins (Option A from §3):
```python
off_counts, off_edges = np.histogram(off, bins=25)
teams[str(tid)]["off_hist"] = {
    "counts": off_counts.tolist(),
    "edges": [round(e, 2) for e in off_edges.tolist()],
}
# same for def_hist
```

### Hyperparameters extraction

```python
hyper = {
    "sigma_mean": ...,
    "sigma_std": ...,
}
if idata is not None:
    post = idata.posterior
    for key, var in [
        ("sigma_off_conf_mean", "sigma_off_conf"),
        ("sigma_def_conf_mean", "sigma_def_conf"),
        ("mu_intercept_mean", "mu_intercept"),
    ]:
        if var in post:
            hyper[key] = round(float(post[var].values.flatten().mean()), 2)
```

The `off_def_correlation` scalar is computed once from the tournament teams' off/def means (not per-sample) and stored in hyperparameters.

### Conference effects extraction

```python
if idata is not None and "mu_off_conf" in idata.posterior:
    conf_effects = {}
    conf_names = data["conference_names"]  # list aligned to mu_off_conf dimension
    mu_off = idata.posterior["mu_off_conf"].values.reshape(-1, len(conf_names))
    mu_def = idata.posterior["mu_def_conf"].values.reshape(-1, len(conf_names))
    for i, conf in enumerate(conf_names):
        if conf in tournament_conferences:  # only include tourney conferences
            conf_effects[conf] = {
                "mu_off": round(float(mu_off[:, i].mean()), 2),
                "mu_def": round(float(mu_def[:, i].mean()), 2),
            }
    snapshot["conference_effects"] = conf_effects
```

### `generate_baseline` update

Replace the `theta` extraction with off/def:

```python
# Old
theta = idata.posterior["theta"].values.reshape(-1, data["n_teams"])

# New
off_samples = idata.posterior["off"].values.reshape(-1, data["n_teams"])
def_samples = idata.posterior["def"].values.reshape(-1, data["n_teams"])
```

Pass both to `export_snapshot` and `simulate_tournament` (if the simulator expects them separately — verify against `src/simulate.py`).

---

## 6. Model Comparison Metadata (Optional)

If LOO model comparison results (Gaussian vs. Student-t likelihood) are computed, store them as static metadata in a file that does not change day-over-day:

`assets/data/march-madness-2026/model_comparison.json`:
```json
{
  "models": [
    {"name": "Gaussian", "loo": -18432.1, "p_loo": 284.2},
    {"name": "Student-t (nu=4)", "loo": -18201.7, "p_loo": 291.8}
  ],
  "preferred": "Student-t (nu=4)",
  "note": "LOO comparison on 2026 regular season data."
}
```

This file is generated once, committed to the site repo manually, and referenced in a small static info box on the dashboard (e.g., "Model: Student-t Bradley-Terry — see comparison"). It does not need to be fetched dynamically alongside snapshots.

---

## Summary of File Changes

| File | Change |
|---|---|
| `src/export.py` | `theta_samples` → `off_samples` + `def_samples`; update team entry fields; update hyperparameter extraction; add conference effects extraction |
| `snapshots/YYYY-MM-DD.json` | Schema per §1 and §4 |
| `march-madness-dashboard.js` | Add off/def scatter section; update team deep dive to two histograms; add conference effects section |
| `assets/data/march-madness-2026/model_comparison.json` | New static file (optional) |
