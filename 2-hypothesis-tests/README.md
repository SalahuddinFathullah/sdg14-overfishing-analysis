# 2-hypothesis-tests — Formal tests of H1a, H1b, H2a, H2b, H2c

`hypothesis_tests.py` runs the analysis pipeline from the main README (§4) on `data/raw/`. It reuses the data builders from [`1-statistical-tests/stat_tests.py`](../1-statistical-tests/stat_tests.py) and the chart style from [`0-visualisation/visualise.py`](../0-visualisation/visualise.py).

```bash
python 2-hypothesis-tests/hypothesis_tests.py                 # everything (~10 s)
python 2-hypothesis-tests/hypothesis_tests.py --only h1a h2c
python 2-hypothesis-tests/hypothesis_tests.py --only i2_cpue i2_h1b i2_h2b   # iteration 2 (effort)
```

Outputs:
- `results/decisions.csv`: verdict per hypothesis.
- `results/*.csv`: one table per test family.
- `figures/<section>/*.png`: 6 figures.

## Verdicts

| Hypothesis | Verdict | Key evidence |
|---|---|---|
| **H1a** World trend | **Reject H₀**: significant upward trend | Hamed-Rao Mann-Kendall p < 0.001, τ = 0.84; Sen's slope **+0.55 pp/yr** [0.50, 0.59]. Same result on assessment years only (+0.56) and in Sea Around Us (+0.98 pp/yr, 1974–2019). |
| | No slowdown after the SDGs | Slope changes −0.15 pp/yr after 2015 (p = 0.49); Chow test p = 0.89 |
| **H1b** World catch → overexploitation | **Fail to reject H₀** | CCF of catch growth vs Δ% overexploited, lags 0–10: max \|r\| = 0.23, all Holm p = 1.00. Prewhitened, Granger (lags 1–4) and the Sea Around Us catch series give the same result. Levels are not cointegrated (p = 0.20). |
| **Stage rule** | **Not met** (H1a yes, H1b no) | Stage 2 below is therefore **exploratory/conditional**, as the plan requires |
| **H2a** SEA trend | **Reject H₀**: % overexploited rises | SEA +7.4 pp/decade (p < 0.001); all 5 countries and all 11 EEZs rise over 1950–2019 after Holm. Since 1990 Vietnam is the only exception (falling). |
| | Mean trophic level: mixed | Falls significantly only in the **Philippines** (both windows), and in Vietnam and Malaysia Peninsula West over 1950–2019. SEA-wide MTL rises. |
| **H2b** SEA catch → MTL | **Fail to reject H₀** | 0 of 66 lag tests significant after Holm. The smallest raw p is the Philippines at lag 9 (r = −0.31, p = 0.016 → Holm 0.18). |
| **H2c** SEA vs world slope | **Mixed** | SEA as a whole rises at the world's pace since 1990: +0.97 vs +0.97 pp/yr, difference 0.00 [−0.09, +0.08], p = 0.91. |
| | Three countries worse | **Indonesia (+0.97 pp/yr faster), Thailand (+0.42), Malaysia (+0.26)** worsen significantly faster than the world (Holm). Vietnam is slower (−1.41) and the Philippines are not different. |
| | FAO Area 71 | Rises faster than the world (+1.03 pp/yr, p = 0.027), but only because of the 2021 value (without it p = 0.084). |

**Reading this:**
- Overfishing is clearly rising, globally and in Southeast Asia, and the SDGs have not bent the curve.
- *Catch volume* does not explain the rise, because world catch has been flat since the 1990s. Catch is a weak proxy for fishing pressure (README caveat 1); effort or fleet-capacity data would be needed to test the cause properly.
- Within SEA, the problem is concentrated in Indonesia, Thailand and Malaysia.

## Iteration 2 — fishing effort as the pressure variable

Iteration 1 found no link between **catch** and overexploitation (H1b, H2b). Catch is only a proxy for fishing pressure (main README caveat 1). It can stay flat while pressure rises, if each unit of fishing lands less. Iteration 2 repeats H1b and H2b with **fishing effort**, the direct measure of pressure. The source is the IMAS Global Fishing Effort database (vessels × kW × days at sea, 1950–2017). Iteration-1 results above are kept unchanged, so the analysis history stays visible.

| Test | Result |
|---|---|
| **Catch per unit effort (CPUE)**, Mann-Kendall on log CPUE 1990–2017 | **Falling.** World −35% (1990–94 → 2013–17; effort +44%, catch −6%; Sen −1.9%/yr, p < 0.001). SEA −50% (Sen −3.6%/yr). Significant after Holm in Malaysia (−68%), Indonesia (−54%) and the Philippines (−43%). Not significant in Thailand (source breaks) or Vietnam (−12%, catch still growing). |
| **H1b with effort** (world effort growth → Δ% overexploited, lags 0–10) | **Fail to reject H₀.** Max r = +0.35 at lag 3 (raw p = 0.026 → Holm 0.29). Sea Around Us series: Holm min p = 0.27. Granger min Holm p = 0.07. Long-run cointegration p = 0.078 (with constant; 10% level only). |
| **H2b with effort** (EEZ effort growth → Δ% overexploited and ΔMTL, SEA + 5 countries) | **Weak.** 1 of 132 tests is significant within its unit: Indonesia ΔMTL, same year, r = +0.35 (Holm p = 0.043). It is not significant once all 132 tests are corrected together (p = 0.52). |
| **Stage rule** | Still not met: H1b is not rejected with either catch or effort. |

**What the second iteration adds:**
- **The long run explains the rise.** More effort is needed for the same or less catch, the classic sign of depleted stocks. That is why catch can stay flat (iteration 1) while overexploitation rises.
- **Short-run changes don't line up.** Stock status responds slowly and cumulatively, so a 0–10-year correlation of yearly changes cannot pick up the link.
- **Stock status becomes testable.** Effort, unlike catch, is independent of Sea Around Us stock status, which removes the circularity from iteration 1.

**Method choices:**
- **Nominal effort is the main series.** Effective effort includes an assumed technological creep, grows several hundred-fold and stays non-stationary, so it is used for sensitivity only.
- **Source breaks are removed.** Single-year jumps above 65% are dropped from the effort growth series (Thailand 2002 and 2011, Malaysia 1965). Thailand's series stays non-stationary even so, and its CCF is flagged as unreliable.
- **CPUE change uses 5-year means** (1990–94 vs 2013–17), so a single noisy year can't drive it.

| | | |
|---|---|---|
| ![](figures/7_iteration2_effort/I2.01_cpue_effort_catch_index.png) | ![](figures/7_iteration2_effort/I2.02_h1b_ccf_effort.png) | ![](figures/7_iteration2_effort/I2.03_h2b_ccf_effort_heatmap.png) |

Tables: `7_I2_stationarity_effort.csv`, `7_I2_cpue_trends.csv`, `7_I2_H1b_ccf_effort.csv`, `7_I2_H1b_granger_effort.csv`, `7_I2_H1b_cointegration_effort.csv`, `7_I2_H2b_ccf_effort.csv`. Verdicts are rows in `decisions.csv`.

## Sections

| Section | Tests | Table(s) | Figure |
|---|---|---|---|
| Stationarity | ADF (H₀ unit root) + KPSS (H₀ stationary) on levels and on the transformation used: first difference / log-difference, detrended if ADF still fails | `0_stationarity_adf_kpss.csv` | — |
| H1a | Mann-Kendall (original + Hamed-Rao), Sen's slope + 95% CI, OLS + Newey-West, segmented regression (break 2015), Chow test | `1_H1a_*.csv` | H1a.01 |
| H1b | CCF lags 0–10 + Holm, AR prewhitening, Granger (F-test, lags 1–4), Engle-Granger cointegration | `2_H1b_*.csv` | H1b.01 |
| H2a | Hamed-Rao Mann-Kendall + Sen's slope for SEA, 5 countries and 11 EEZs × 2 windows, Holm per family; FAO Areas 71/57 | `3_H2a_*.csv` | H2a.01 |
| H2b | CCF catch growth → ΔMTL for SEA + 5 countries, Holm within unit and across all 66 tests | `4_H2b_ccf_catch_mtl.csv` | H2b.01 |
| H2c | Interaction regression y = b0 + b1·year + b2·SEA + b3·year×SEA; b3 tested on the SEA − world gap with HAC errors | `5_H2c_interaction_gap_regression.csv` | H2c.01 |
| Diagnostics | Shapiro-Wilk, Durbin-Watson, Ljung-Box, Breusch-Pagan for every regression | `6_regression_diagnostics.csv` | D.01 |

## Figures

| | |
|---|---|
| ![](figures/1_H1a_global_trend/H1a.01_world_trend_mann_kendall_segmented.png) | ![](figures/2_H1b_global_ccf/H1b.01_ccf_catch_vs_overexploited.png) |
| ![](figures/3_H2a_sea_trends/H2a.01_sen_slopes_sea_countries_eez.png) | ![](figures/4_H2b_sea_ccf/H2b.01_ccf_catch_vs_mtl.png) |
| ![](figures/5_H2c_sea_vs_world/H2c.01_sea_vs_world_slopes.png) | ![](figures/6_diagnostics/D.01_h1a_residual_diagnostics.png) |

## Notes for the report

- **Autocorrelation:** the H1a residuals have Durbin-Watson 0.58 and Ljung-Box p < 0.001, and 8 of the 13 regressions fail Ljung-Box. That is why every regression uses HAC standard errors and every trend uses the Hamed-Rao Mann-Kendall test.
- **Non-normal residuals:** these occur in the H1a and SEA-gap regressions (Shapiro p < 0.05). This supports Mann-Kendall / Sen as the primary trend evidence; OLS is the secondary check.
- **KPSS disagreements:** 7 transformed series pass ADF but not KPSS (flagged in the stationarity table). The CCF results there are the least certain, but none was close to significance.
- **SEA aggregate:** % overexploited is the mean of the 5 countries, so each country counts equally. A mean over the 11 EEZs would give Malaysia (4 EEZs) and Indonesia (3) more weight, and SEA would then look worse than the world.
- **FAO Area 71 vs world:** this comparison uses only 10 assessment years. The 2021 value (47%, up from 20% in 2019) drives the significance.
