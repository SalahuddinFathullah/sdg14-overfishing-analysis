# SDG 14 — Is Overfishing Getting Worse? (Global → Southeast Asia)

**KIE4033 Assignment 1 — Exploratory Data Analytics** · Universiti Malaya · Semester 1, 2026/2027
Theme: **UN Sustainable Development Goal 14 — Life Below Water**

We apply exploratory data analysis (EDA) and statistical hypothesis testing in Python to test whether overfishing is a genuine and worsening problem — first **globally**, and then, **only if the global result holds**, for **Southeast Asia (SEA)**.

> **Status:** 🟡 Data collected — all raw datasets in `data/raw/`; cleaning & analysis not started yet. See [Progress](#progress).

---

## 0. Assignment deliverables

- Python project developed in this GitHub repo (link goes in the report)
- Brief report (~10 pages), submitted as `KIE4033_Asgn1_GroupXY.pdf`
- Pair project (2 students)

### How the work maps to the rubric

| Report section (weight) | What we produce | Where in this repo |
|---|---|---|
| Introduction (10%) | Problem statement, objectives, research questions (H1a–H2c below) | [§1](#1-goal), [§2](#2-hypotheses) |
| Data Understanding & Preparation (15%) | Dataset descriptions, cleaning, missing-value handling, summary statistics, correlation matrix | [§3](#3-datasets), `notebooks/01_*` |
| EDA (25%) | Labelled time-series plots, ACF plots, regional comparisons, anomaly notes | `notebooks/02_*`, `figures/` |
| Statistical Analysis (25%) | ADF, Mann-Kendall, segmented regression, CCF, interaction regression; **assumptions checked**; **post-hoc tests** | [§4](#4-analysis-pipeline), `notebooks/03_*` |
| Discussion (15%) | Link EDA + test results; limitations | [§5](#5-known-caveats-for-the-limitations-section) |
| Conclusion (5%) | Direct answer to each hypothesis + SDG 14.4 relevance | [Findings](#findings-so-far) |
| Code quality & references (5%) | Commented, reproducible code; all sources cited | `src/`, `requirements.txt`, [§3](#3-datasets) |

---

## 1. Goal

Answer two questions with evidence rather than opinion:

1. **Is overfishing increasing globally, and is fishing pressure driving it?** (Stage 1)
2. **If yes — is Southeast Asia following the same pattern, or doing worse than the world average?** (Stage 2)

SDG target **14.4** calls for ending overfishing and restoring stocks to sustainable levels; indicator **14.4.1** is the share of fish stocks within biologically sustainable levels. If the data shows overexploitation is still rising after the SDGs were adopted (2015), the target is not being met.

---

## 2. Hypotheses

### Stage 1 — Global

| ID | Question | H₀ (no problem) | H₁ (problem exists) |
|----|----------|-----------------|---------------------|
| **H1a** | Trend | No significant upward trend in % of stocks overexploited | Significant upward trend |
| **H1b** | Cause | No significant cross-correlation between global catch and % stocks overexploited at any lag | Significant **positive** cross-correlation at lag *k* (high catch now → more overexploited stocks later) |

**Decision rule:** reject H₀ for **both** H1a and H1b → overfishing is confirmed as a real global problem → proceed to Stage 2.

### Stage 2 — Southeast Asia (only if Stage 1 holds)

| ID | Question | H₀ | H₁ |
|----|----------|----|----|
| **H2a** | Trend | No upward trend in SEA overexploitation (or no decline in SEA mean trophic level) | Significant worsening trend |
| **H2b** | Cause | No significant cross-correlation between SEA catch and SEA stock status / mean trophic level | Significant cross-correlation at lag *k* |
| **H2c** | Comparison | SEA trend slope = global trend slope | SEA slope is significantly worse |

**SEA focus countries:** Malaysia, Indonesia, Thailand, Vietnam, Philippines. (Singapore and Brunei are excluded — catch series are too small to be meaningful.)

---

## 3. Datasets

✅ **All datasets are downloaded and committed in [`data/raw/`](data/raw/)** — see the [data dictionary](data/raw/README.md) for every column, unit and known issue. Re-download from source with `python src/download_data.py`.

### Stage 1 (global)

| Dataset | Variable | Coverage | File |
|---------|----------|----------|------|
| FAO via OWID — *Share of fish stocks overexploited* | % of assessed stocks overexploited | World 1974–2021 (48 yrs) | `owid/share-of-fish-stocks-overexploited.csv` |
| FAO via OWID — *Fish stocks within sustainable levels* (SDG 14.4.1) | % sustainable / % overexploited | World 1974–2021 | `owid/fish-stocks-within-sustainable-levels.csv` |
| FAO/World Bank via OWID — *Capture fishery production* | Wild catch (t) | World 1960–2024 | `owid/capture-fishery-production.csv` |

### Stage 2 (Southeast Asia — Malaysia, Indonesia, Thailand, Vietnam, Philippines)

| Dataset | What it gives | Coverage | Folder / file |
|---------|---------------|----------|---------------|
| **Sea Around Us — stock status** | % of stocks (and % of catch) collapsed / over-exploited / exploited / developing / rebuilding | 11 SEA EEZs, 1950–2019 | `sau/stock_status/` |
| **Sea Around Us — Marine Trophic Index** | Mean trophic level of catch, FiB index, mean max fish length | 11 SEA EEZs, 1950–2019 | `sau/marine_trophic_index/` |
| **Sea Around Us — catch summary** | Total catch (t) & value by sector, landings/discards, reported/unreported, domestic/foreign fleet | 11 SEA EEZs, 1950–2019 | `sau/catch_summary/` |
| **FAO via OWID — Areas 71 & 57** | % stocks overexploited in Western Central Pacific (71) and Eastern Indian Ocean (57) | 2004–2021, biennial (10 pts) | `owid/share-of-fish-stocks-overexploited.csv` |
| FAO/World Bank via OWID — capture production | Wild catch (t) per country | SEA5 + World, 1960–2024 | `owid/capture-fishery-production.csv` |
| Sea Around Us via OWID — catch by gear, discards | Context for EDA (trawling share, waste) | SEA5 + World, 1950–2018 | `owid/fish-catch-gear-type.csv`, `owid/fish-discards.csv` |
| FAO/World Bank via OWID — aquaculture | Context: farmed fish growth while wild catch plateaus | SEA5 + World, 1960–2024 | `owid/aquaculture-farmed-fish-production.csv` |

**Why Sea Around Us became the main SEA source:** FAO publishes stock status only by large fishing area, and the OWID series for Areas 71/57 has just 10 uneven points — fine for a Mann-Kendall test but too short for ACF/CCF. Sea Around Us gives an annual 70-year series per EEZ, which supports the full pipeline and lets us compare the 5 countries (Kruskal-Wallis + Dunn).

**Dropped:** Global Fishing Watch (needs a registered account and its effort series is only 2012–2024) and SEAFDEC (PDF tables). Neither is needed for the hypotheses above; GFW can be added later as an optional extension.

---

## 4. Analysis pipeline

Fishing data is strongly **trended**. Two trending series will look correlated even when unrelated (*spurious correlation*), so correlation tests are run only on **stationary** series.

```
Load & clean ──► Explore (plots, missing years)
                     │
                     ▼
             ACF of each series ──► shows trend / persistence
                     │
                     ▼
             ADF test ──► non-stationary? ──► difference / detrend ──► re-test
                     │
          ┌──────────┴───────────┐
          ▼                      ▼
   TREND tests (H1a, H2a)   RELATIONSHIP tests (H1b, H2b)
   • Mann-Kendall           • Cross-correlation (CCF), lags 0–10 yrs
   • Linear regression      • Pearson / Spearman on stationary series
   • Segmented regression   • (optional) Granger causality
     (before vs after 2015)
          │                      │
          └──────────┬───────────┘
                     ▼
         COMPARISON test (H2c)
         • Regression with region × year interaction
                     │
                     ▼
         Accept / reject H₀ ──► conclusions & limitations
```

### Why each method

| Method | Purpose |
|--------|---------|
| **ACF (autocorrelation)** | Compares a series with its own past values. Used to *show* persistence/trend — it does **not** compare two datasets. |
| **ADF test** | Checks stationarity. Required before CCF so results aren't spurious. |
| **Differencing** | Removes trend (Δxₜ = xₜ − xₜ₋₁) so a series becomes stationary. |
| **Mann-Kendall** | Non-parametric trend test — no normality assumption, robust to outliers. |
| **Segmented regression** | Tests whether the slope changed after SDG adoption in 2015. |
| **CCF (cross-correlation)** | Correlation between two series at different lags — e.g. catch today vs overexploitation *k* years later. This is the test for "does one dataset relate to another". |
| **Granger causality** *(optional)* | Does past catch help *predict* future overexploitation beyond the series' own past? |
| **Interaction regression** | Tests whether the SEA slope differs from the global slope (H2c). |

### Assumption checks (required by the rubric)

| Check | Applies to | If it fails |
|-------|-----------|-------------|
| **Shapiro-Wilk** on residuals | Linear / segmented / interaction regression | Rely on Mann-Kendall (non-parametric) and Spearman instead of Pearson |
| **Durbin-Watson / Ljung-Box** on residuals | All regressions on time series | Use Newey-West (HAC) standard errors |
| **Breusch-Pagan** | Regressions | Use robust (HC) standard errors |
| **ADF stationarity** | CCF, correlation, Granger | Difference / detrend first |
| **Even spacing / no gaps** | ACF, CCF | Interpolate short gaps or trim the series, and report it |

### Post-hoc tests

- **CCF across many lags:** testing lags 0–10 is 11 tests, so significance is adjusted with a **Bonferroni / Holm correction** to avoid false positives.
- **Comparing the 5 SEA countries (Stage 2):** if a **Kruskal-Wallis** test shows the countries differ (e.g. in year-on-year change of mean trophic level), **Dunn's post-hoc test** with Holm correction identifies which country pairs differ.
- **Segmented regression:** if a breakpoint at 2015 is significant, a **Chow test** confirms the structural break.

---

## 5. Known caveats (for the Limitations section)

1. **Catch ≠ overfishing.** Catch rises while a fishery develops, then falls once stocks collapse. Lagged correlation captures part of this, but catch is a proxy, not a direct measure.
2. **FAO changed its stock-assessment method in 2025.** In Area 71 the number of assessed stocks jumped from 43 to 265, and the sustainable share fell from ~77–80% to 52.5%. Our FAO data ends in 2021, so the break is not in our series — but mention that newer FAO figures are not comparable.
3. **Interpolated FAO years.** The World overexploited series has stretches that rise by a constant step each year (linear interpolation between assessments), which inflates autocorrelation. Robustness check: re-run H1a on assessment years only.
4. **Sea Around Us stock status is derived from catch.** A stock is labelled "over-exploited"/"collapsed" from its catch relative to its historical peak (Kleisner et al. 2013). So for **H2b**, correlating catch with stock status is partly circular — use **mean trophic level** (independent of catch volume) as the main overfishing response, and treat stock status as supporting evidence.
5. **Country-level FAO stock status is not published.** FAO reports by fishing area only; that is why Sea Around Us EEZ data is used for SEA.
6. **Different end years.** FAO stocks end 2021, SAU 2019, capture 2024 — align on the common window before correlating.

---

## 6. Repository structure

```
sdg14-overfishing-analysis/
├── README.md            ← this file (project plan + progress log)
├── LICENSE              ← MIT
├── requirements.txt     ← Python dependencies
├── data/
│   ├── raw/             ← original downloads (OWID + Sea Around Us), see data/raw/README.md
│   └── processed/       ← cleaned, analysis-ready CSVs
├── notebooks/           ← Jupyter notebooks (stage1_global, stage2_sea, …)
├── src/
│   └── download_data.py ← re-downloads everything in data/raw/
└── figures/             ← exported charts for the report
```

---

## 7. Setup

```bash
git clone https://github.com/SalahuddinFathullah/sdg14-overfishing-analysis.git
cd sdg14-overfishing-analysis
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# data is already in data/raw/; to re-download from source:
python src/download_data.py
jupyter lab
```

---

## Progress

| Step | Stage | Status | Notebook / file |
|------|-------|--------|-----------------|
| Repo & plan set up | — | ✅ Done | `README.md` |
| Download global datasets | 1 | ✅ Done | `data/raw/owid/`, `src/download_data.py` |
| Clean global datasets | 1 | ⬜ Not started | — |
| Exploratory plots + ACF | 1 | ⬜ Not started | — |
| ADF + differencing | 1 | ⬜ Not started | — |
| H1a trend tests (Mann-Kendall, segmented regression) | 1 | ⬜ Not started | — |
| H1b cross-correlation (CCF) | 1 | ⬜ Not started | — |
| **Stage 1 decision** | 1 | ⬜ Pending | — |
| Download SEA datasets | 2 | ✅ Done | `data/raw/sau/`, `data/raw/owid/` |
| Clean SEA datasets | 2 | ⬜ Not started | — |
| H2a / H2b tests on SEA | 2 | ⬜ Not started | — |
| H2c SEA vs global comparison | 2 | ⬜ Not started | — |
| Final figures & write-up | — | ⬜ Not started | — |

### Findings so far

_None yet — results will be added here as each test is completed._

---

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-11-18 | Salah | Initial repo: project plan, hypotheses, datasets, analysis pipeline, rubric mapping, assumption & post-hoc tests, MIT licence |
| 2026-11-18 | Salah | Added all raw datasets (6 OWID + 33 Sea Around Us files), `src/download_data.py`, data dictionary; switched SEA source to Sea Around Us EEZ data; added circularity caveat for SAU stock status |

---

## Contributing

- Work on a branch or pull before you push (`git pull --rebase`) to avoid overwriting each other.
- **Whoever pushes code updates the [Progress](#progress) table and adds a line to the [Changelog](#changelog)** in the same commit, so this README always reflects the current state of the analysis.
- Keep notebooks numbered in run order (e.g. `01_stage1_load_clean.ipynb`, `02_stage1_acf_adf.ipynb`).
- Don't commit large raw data files — add a download link/script instead.

---

## License

Code in this repository is released under the [MIT License](LICENSE). Datasets remain under their original providers' licences (OWID: CC BY 4.0; Sea Around Us: free for non-commercial use with citation; FAO: see FAO terms). Full citations are in [`data/raw/README.md`](data/raw/README.md#references-for-the-report).
