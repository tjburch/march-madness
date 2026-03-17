---
layout: single
title: "Bayesian Bracketology 2026: Decomposing Offense and Defense with a Hierarchical Bradley-Terry Model"
date: 2026-03-17
categories: misc
tags: [bayesian, pymc, sports, march-madness]
toc: true
toc_label: "Contents"
toc_sticky: true
excerpt: "Last year's model treated every team as a single number. This year, we split that number into offense and defense — and found a Simpson's paradox hiding in the correlation."
header:
  teaser: /blogimages/march-madness-2026/off_def_scatter.png
---

Every March, millions of Americans fill out tournament brackets. Most rely on some combination of seed numbers, team names they recognize, and gut feelings about which mid-major "looks dangerous." The results are predictably bad — and predictably fun.

[Last year](https://tylerjamesburch.com/blog/misc/hockey-bayes), I built a Bayesian Bradley-Terry model for college basketball: each team gets a single latent strength $\theta_i$, teams are partially pooled within conferences, and the model predicts score margins. It worked — Brier score of 0.188 across 10 seasons of historical validation, 70.3% accuracy. But it left something on the table.

A single $\theta$ treats Houston's suffocating defense the same as Michigan's prolific offense. Both might rate similarly overall, but they get there in fundamentally different ways. If we're going to simulate a tournament bracket, we should know *how* a team wins, not just *that* it wins.

This year, we decompose.

## The Upgrade: Offense and Defense

### From Margins to Scores

Last year's model observed the margin of each game — one number per contest:

$$\text{margin}_{ij} \sim \text{Normal}(\theta_i - \theta_j + \alpha \cdot \text{home}_i, \sigma)$$

The new model observes both scores separately, giving us two data points per game and a natural place to decompose team strength:

$$\text{score}_i \sim \text{Normal}(\mu + \text{off}_i - \text{def}_j + \alpha \cdot \text{home}_i, \sigma)$$

$$\text{score}_j \sim \text{Normal}(\mu + \text{off}_j - \text{def}_i - \alpha \cdot \text{home}_i, \sigma)$$

Each team gets two parameters: an offensive strength $\text{off}_i$ (how many points they generate above average) and a defensive strength $\text{def}_i$ (how many points they *prevent* above average). The global intercept $\mu$ anchors the average score per team per game — it estimates at about 70 points. Home court advantage $\alpha$ applies in the usual way: the home team's expected score goes up, the away team's goes down.

This is the same framework Baio & Blangiardo used for [modeling the Premier League](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf), adapted for college basketball. The key insight from that work carries over: by modeling scores directly instead of margins, you can identify teams that win by outscoring opponents versus teams that win by shutting them down.

### The LKJ Correlation Prior

How should offense and defense relate within a team? We could make them independent, but that ignores an interesting structural question: do teams that invest in offense tend to sacrifice defense, or do good programs tend to be good at both?

We let the model learn this relationship through an LKJ correlation prior on the team-level effects:

$$\begin{bmatrix} \text{off}_i \\ \text{def}_i \end{bmatrix} = \begin{bmatrix} \mu^{\text{off}}_{c[i]} \\ \mu^{\text{def}}_{c[i]} \end{bmatrix} + L \cdot z_i$$

where $L$ is the Cholesky factor of a $2 \times 2$ covariance matrix with an LKJ($\eta=2$) prior on the correlation, and $z_i \sim \text{Normal}(0, 1)$. Conference-level means for offense ($\mu^{\text{off}}_c$) and defense ($\mu^{\text{def}}_c$) are estimated separately — the SEC might produce strong defenses while the Big Ten generates potent offenses, or vice versa.

The non-centered parameterization via $z_i$ is essential for sampling efficiency in hierarchical models. If you've spent time with PyMC, you know the drill.

### Gaussian over Student-t

With individual scores instead of margins, outliers become a concern — blowout games could pull the model around. We fit both a Gaussian and a Student-t likelihood and compared them via LOO cross-validation:

![LOO model comparison](/blogimages/march-madness-2026/loo_comparison.png)

The ELPD difference was 0.78 in favor of Gaussian — essentially no difference. The Student-t model estimated $\nu \approx 53$, which is close enough to Gaussian that the heavier tails weren't doing any work. We kept the Gaussian for simplicity.

### Full Model Specification

For the record, here's the complete specification:

```python
with pm.Model(coords=coords) as model:
    mu_intercept = pm.Normal("mu_intercept", mu=70, sigma=10)

    sigma_off_conf = pm.HalfNormal("sigma_off_conf", sigma=5)
    sigma_def_conf = pm.HalfNormal("sigma_def_conf", sigma=5)
    mu_off_conf = pm.Normal("mu_off_conf", mu=0, sigma=sigma_off_conf, dims="conference")
    mu_def_conf = pm.Normal("mu_def_conf", mu=0, sigma=sigma_def_conf, dims="conference")

    sd_dist = pm.HalfNormal.dist(sigma=5)
    chol, corr, stds = pm.LKJCholeskyCov("lkj", n=2, eta=2, sd_dist=sd_dist,
                                          compute_corr=True)
    z = pm.Normal("z", mu=0, sigma=1, shape=(n_teams, 2))

    team_effects = pt.stack([mu_off_conf[conf_of_team],
                             mu_def_conf[conf_of_team]], axis=1) + pt.dot(z, chol.T)
    off = pm.Deterministic("off", team_effects[:, 0], dims="team")
    deff = pm.Deterministic("def", team_effects[:, 1], dims="team")

    alpha = pm.Normal("alpha", mu=3.5, sigma=2.0)
    sigma = pm.HalfNormal("sigma", sigma=15)

    mu_score_i = mu_intercept + off[team_i] - deff[team_j] + alpha * home
    mu_score_j = mu_intercept + off[team_j] - deff[team_i] - alpha * home

    pm.Normal("score_i", mu=mu_score_i, sigma=sigma, observed=score_i, dims="game")
    pm.Normal("score_j", mu=mu_score_j, sigma=sigma, observed=score_j, dims="game")
```

Sampled with [nutpie](https://github.com/pymc-devs/nutpie) — 4 chains, 2,000 draws each after 2,000 tuning steps. Zero divergences, $\hat{R} \leq 1.01$ for all parameters.

## The Data

The model is fit on the **2025-26 regular season** — all 5,647 Division I games across 365 teams and 31 conferences. Data comes from the [Kaggle March ML Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition dataset.

Each game provides: which teams played, both scores, and whether it was a home, away, or neutral-site game. The switch from margins to individual scores doubles our observed data points without requiring any additional data collection.

### Posterior Predictive Check

![Posterior predictive scores](/blogimages/march-madness-2026/ppc_scores.png)

The posterior predictive distribution matches the observed score distribution well. The model generates plausible game scores across the full range — from 40-point defensive slugfests to 100-point shootouts.

## What the Model Learned

### Overall Team Strength

The overall strength of a team is the sum of its offensive and defensive contributions: $\text{off}_i + \text{def}_i$. This is analogous to last year's $\theta_i$.

![Top 30 team strengths](/blogimages/march-madness-2026/team_strengths_top30.png)

Michigan and Duke sit at the top, followed by Arizona and Florida. The overall picture is similar to last year's model — the same teams are good, and their posteriors overlap in the same familiar way. But the interesting part is *how* they get there.

### The Offense-Defense Decomposition

This is the headline result.

![Offense vs defense scatter](/blogimages/march-madness-2026/off_def_scatter.png)

Every dot is a team. The x-axis is offensive strength (higher = generates more points above average), the y-axis is defensive strength (higher = allows fewer points than average). Teams in the upper-right are good at both; teams in the lower-left are bad at both.

A few teams jump out:

- **Houston** (overall 23.6): off = 7.0, def = 16.6. This is a defense-first team by a wide margin — their defensive rating is elite, while their offense is merely decent. If you watch Houston, this isn't a surprise. Kelvin Sampson's teams are built to grind.
- **Duke** (overall 28.0): off = 12.0, def = 16.0. Also defense-first, but with a more balanced profile than Houston.
- **Michigan** (overall 28.9): off = 17.5, def = 11.4. The opposite story — Michigan wins by outscoring you.

![Top 20 by offense and defense](/blogimages/march-madness-2026/off_def_rankings.png)

The side-by-side rankings make the decomposition concrete. The list of top-20 offenses and top-20 defenses are quite different — a team can rank in the top 5 offensively while barely cracking the top 20 defensively, or vice versa. The single-$\theta$ model couldn't see any of this.

### Conference Effects: Offense vs Defense

![Conference offensive vs defensive effects](/blogimages/march-madness-2026/conference_off_def.png)

Conferences separate in interesting ways. Some conferences produce teams with strong offenses and weak defenses (or vice versa), which reflects coaching philosophies, pace of play, and recruiting patterns across the league. The power conferences tend to cluster in the upper-right — they're above average on both dimensions — while mid-majors scatter more widely.

### Home Court Advantage

![Home court advantage posterior](/blogimages/march-madness-2026/home_court.png)

The model estimates home court advantage at about **2.8 ± 0.2 points** — consistent with last year's estimate and the published literature. This parameter is learned from regular season data and zeroed out for neutral-site tournament games.

## Simpson's Paradox in the Offense-Defense Correlation

This is the most interesting finding from the decomposition, and it deserves some space.

Look at the scatter plot above. The marginal correlation between offense and defense across all 365 teams is **+0.384**. That's positive — teams that are good at offense tend to also be good at defense. Intuitively, this makes sense: programs with good coaching, recruiting, and resources tend to be good at everything.

But the LKJ model parameter — the *within-conference* correlation that the model learned — is **-0.204**. That's negative. Within a conference, offense and defense *trade off*.

This is a textbook Simpson's paradox. The relationship reverses when you condition on the confounding variable (conference membership). Here's why:

**Between conferences**, the correlation is **+0.827**. Strong conferences (SEC, Big 12, Big Ten) produce teams that are above average on *both* offense and defense. Weak conferences produce teams below average on both. This between-group correlation is strong and positive, and when you pool all teams together ignoring conference, it dominates the marginal relationship.

**Within a conference**, though, teams face a different optimization problem. Every team in the SEC plays the same schedule of SEC opponents. Given that shared context, a team that invests heavily in up-tempo offense might sacrifice some defensive discipline, and vice versa. The data suggests a modest negative correlation of **-0.228** within conferences — not a dramatic tradeoff, but a real one.

The LKJ model parameter of **-0.204** captures precisely this within-conference structure. That's what it's designed to do: after the conference means ($\mu^{\text{off}}_c$, $\mu^{\text{def}}_c$) absorb the between-conference variation, the LKJ correlation models the *residual* relationship between offense and defense at the team level. The fact that it's negative tells us something genuinely interesting about how teams within the same competitive environment allocate resources.

This is exactly the kind of insight you can't get from a single-$\theta$ model. The decomposition into offense and defense reveals structure that the aggregated parameter hides.

## The Payoff: Simulating the Tournament

Win probabilities come from the Gaussian CDF applied to the strength difference. For two teams $i$ and $j$ on a neutral court:

$$P(i \text{ beats } j) = \Phi\left(\frac{(\text{off}_i - \text{def}_j) - (\text{off}_j - \text{def}_i)}{\sigma\sqrt{2}}\right)$$

The $\sqrt{2}$ comes from the fact that we're comparing the *difference* of two independent score random variables, each with variance $\sigma^2$.

For each of 10,000 simulations, we draw a complete set of team strengths from the posterior and play through the 68-team bracket. The result is a distribution of 10,000 brackets, each reflecting a different plausible world.

### Championship Odds

![Championship probabilities](/blogimages/march-madness-2026/championship_odds.png)

| Team | Seed | P(Champion) |
|------|------|-------------|
| Michigan | 1 | 16.2% |
| Duke | 1 | 14.9% |
| Arizona | 1 | 10.8% |
| Florida | 1 | 8.1% |
| Houston | 2 | 6.4% |

Michigan edges out Duke as the favorite, but the gap is narrower than last year's margin model suggested. The offense-defense decomposition changes the landscape slightly: Houston, despite its relatively modest offense, benefits from a defensive profile that translates well in tournament play. Its championship odds sit higher than you might expect from a 2-seed with a mediocre offensive rating.

### Full Bracket Forecast

![Tournament advancement probabilities](/blogimages/march-madness-2026/advancement_heatmap.png)

The advancement heatmap tells the full story. Even #1 seeds only have around 50% probability of reaching the Final Four. Single-elimination tournaments are brutal — and the model quantifies exactly how brutal.

![Regional bracket forecasts](/blogimages/march-madness-2026/bracket_forecast.png)

### First Round Upset Watch

![Upset probabilities](/blogimages/march-madness-2026/upset_probabilities.png)

The offense-defense model produces similar upset probabilities to last year's margin model. This makes sense — win probability depends on the *total* strength difference, which is similar between models. Where the decomposition pays off is in understanding *why* matchups play out the way they do. A strong offense against a weak defense is a different game than two mediocre teams — even if the overall strength gap is the same. The bracket simulation captures this because it draws correlated offense/defense values from the posterior rather than collapsing them into a single number.

## Historical Validation

You shouldn't trust a model that hasn't been tested. We validated by fitting the offense-defense model on each regular season from 2015–2025 (skipping 2020) and predicting that year's tournament.

**Results across 669 tournament games:**
- **Brier score: 0.1886** — essentially identical to last year's margin model (0.188)
- **Accuracy: 70.3%**

The decomposition doesn't improve aggregate prediction accuracy. That's not surprising — the total strength difference is what drives win probability, and the offense-defense model preserves that total. What we gain is *interpretability*: understanding team profiles, identifying stylistic matchup effects, and uncovering structural relationships like the Simpson's paradox in the correlation.

Sometimes the best reason to build a more complex model is to understand the system better, not to improve a point metric.

## The Women's Tournament: Same Model, Different Story

Since the Kaggle competition requires predictions for both tournaments, we fit an independent model on the women's regular season using the identical offense-defense specification.

### UConn and the Chalk

The men's tournament has two co-favorites separated by 1.3 percentage points. The women's tournament has *UConn*:

![Championship odds comparison](/blogimages/march-madness-2026/championship_comparison.png)

| Team | Seed | P(Champion) |
|------|------|-------------|
| Connecticut | 1 | 38.7% |
| South Carolina | 1 | 18.4% |
| UCLA | 1 | 15.6% |

UConn's 34-0 record translates to a posterior strength distribution that barely overlaps with the rest of the field. The offense-defense decomposition reveals that UConn is elite on both dimensions — unlike Houston's lopsided defensive profile, UConn doesn't sacrifice anything.

![Women's team strengths](/blogimages/march-madness-2026/team_strengths_top30_womens.png)

The structural difference between the men's and women's fields is stark. The men's top 30 is a smooth gradient with overlapping posteriors throughout. The women's field has a clear break: five teams in one tier, then a gap of nearly 7 points before the next cluster.

![Women's offense vs defense](/blogimages/march-madness-2026/off_def_scatter_womens.png)

![Women's advancement probabilities](/blogimages/march-madness-2026/advancement_heatmap_womens.png)

### Home Court in the Women's Tournament

The women's tournament format gives the top 16 seeds home court for rounds 1 and 2. Combined with the already-wider talent gap, this pushes early-round upset probabilities even lower. The ~2.8-point home advantage compounds on top of strength differentials that are already larger than the men's equivalents.

![Women's bracket forecast](/blogimages/march-madness-2026/bracket_forecast_womens.png)

## Our Brackets

Based on the most likely outcome at each matchup:

### Men's

**Final Four:** Duke, Florida, Michigan, Arizona

**Championship Game:** Michigan vs. Duke

**Champion:** Michigan (16.2% — meaning we're 83.8% sure this is wrong)

### Women's

**Final Four:** UConn, South Carolina, UCLA, Texas

**Championship Game:** UConn vs. South Carolina

**Champion:** UConn (38.7% — meaning we're 61.3% sure this is wrong)

The contrast captures the structural difference between the two fields. The men's champion is nearly a coin-flip among three or four teams. The women's champion is the clearest favorite the model produces — but 38.7% still means there's a better-than-even chance someone else lifts the trophy. Single-elimination tournaments are designed to produce uncertainty, and even a dominant team can only be so dominant across six consecutive games.

## What's Next

The big item from last year's to-do list — the offense-defense decomposition — is done. The Simpson's paradox finding alone justified the effort. Potential next steps:

- **Temporal weighting** — games in February should probably matter more than games in November. A team that lost three starters to injury in January is not the same team as the one that started the season.
- **Matchup-specific effects** — the current model uses total strength differences. In principle, a great offense facing a great defense could play out differently than the sum suggests (fast-paced teams vs. slow-paced teams, for example). Whether there's enough data to estimate these interaction effects is an open question.
- **Additional covariates** — incorporating tempo, four factors, or experience data from sources like Barttorvik.

But the model is already doing its job: giving us calibrated win probabilities with honest uncertainty, plus structural insight into how teams differ. Sometimes that's enough.

## Code

All code for this analysis is available on [GitHub](https://github.com/tjburch/march_madness). The model is implemented in PyMC 5 with nutpie sampling, and the full pipeline runs both tournaments end-to-end in about 10 minutes.

---

*Want to see how the model did? Check back after the tournament for a retrospective post comparing our predictions to actual results.*
