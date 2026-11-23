# 4 · Overview — the whole analysis at a glance

`overview.py` reads the result tables written by the other folders and draws report-level summaries. It recomputes nothing, so run it **last**:

```bash
python 0-visualisation/visualise.py
python 3-data-quality/clean_data.py && python 3-data-quality/quality_tests.py
python 1-statistical-tests/stat_tests.py
python 2-hypothesis-tests/hypothesis_tests.py
python 4-overview/overview.py                 # all sections
python 4-overview/overview.py --only map      # one section: map | stationarity | assumptions | longrun | index
```

## Figures

| # | Figure | What it shows | Reads |
|---|---|---|---|
| O.01 | [Analysis map](figures/O.01_analysis_map.png) | Data → preparation → EDA → six test iterations → answers. Every hypothesis chip is coloured by its verdict (reject / mixed or weak / fail to reject), read from `decisions.csv`, so the map stays in step with the results | `2-hypothesis-tests/results/decisions.csv` |
| O.02 | [Stationarity of every series](figures/O.02_stationarity_all_series.png) | ADF and KPSS p-values for all 42 series tested in iterations 1–4, before (○) and after (●) differencing or detrending. 25 pass both tests after transforming, 12 pass ADF only, 5 stay non-stationary and are flagged | stationarity tables `0_`, `7_I2_`, `8_I3_`, `9_I4_` |
| O.03 | [Assumption checks and test choice](figures/O.03_assumption_checks_and_test_choice.png) | Left: residual diagnostics (Shapiro-Wilk, Ljung-Box, Breusch-Pagan, Durbin-Watson) for 20 regressions; autocorrelation fails in 15, which is why HAC standard errors are used. Right: parametric vs non-parametric p-values from the course-method tests, showing where the choice changes the verdict | diagnostics table, `1-statistical-tests/results/` B1/B2 |
| O.04 | [Long-run tests](figures/O.04_long_run_tests.png) | Engle-Granger cointegration and Granger causality across iterations. Only 2 of 23 cointegration tests pass, and neither holds in both data sources | cointegration and Granger tables |

These cover the tests that were previously reported only as tables, so every test in the analysis now has a figure.

## Figure catalogue

`figure_index.csv` lists all 71 figures in the repo (34 EDA, 8 course methods, 23 hypothesis tests, 2 data quality, 4 overview) with:

- `figure_id`, `title`, `chart` type and one-line `message`
- `step` (pipeline folder) and `rubric_section` (Introduction … Conclusion)
- `hypotheses` and `iteration` it supports
- `file` path and `report_role`: main-body **Fig 1–20**, **Appendix** (13), **Overview** (O.*) or **Gallery** (the rest)

Captions come from the gallery tables in `0-visualisation/README.md` plus a captions dictionary in the script for the other folders.

## Interactive dashboard

The same catalogue, the verdict table and the H5d forecast are in the offline interactive page [`dashboard/index.html`](../dashboard/README.md) (rebuild with `python dashboard/build_dashboard.py` after this script).
