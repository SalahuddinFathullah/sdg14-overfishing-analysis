# SDG 14 — Is Overfishing Getting Worse? (Global → Southeast Asia)

**KIE4033 Assignment 1 — Exploratory Data Analytics** · Universiti Malaya · Semester 1, 2026/2027
Theme: **UN Sustainable Development Goal 14 — Life Below Water**

We apply exploratory data analysis (EDA) and statistical hypothesis testing in Python to test whether overfishing is a genuine and worsening problem — first **globally**, and then, **only if the global result holds**, for **Southeast Asia (SEA)**.

> **Status:** 🟡 Planning — repo set up, no analysis code pushed yet. See [Progress](#progress).

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

### Stage 1 (global)

| Dataset | Variable | Coverage | Link |
|---------|----------|----------|------|
| OWID / FAO — *Share of fish stocks overexploited* | % of assessed stocks overexploited | Global, 1974–2021 | [ourworldindata.org](https://ourworldindata.org/grapher/share-of-fish-stocks-overexploited) |
| OWID / World Bank — *Capture fishery production* (`ER.FSH.CAPT.MT`) | Tonnes of wild fish landed | Global & per country, 1960–2023 | [ourworldindata.org](https://ourworldindata.org/grapher/capture-fishery-production) |

Both are single CSVs (direct download link on each chart), so Stage 1 needs no heavy data wrangling.

### Stage 2 (Southeast Asia)

| Dataset | What it gives | Notes |
|---------|---------------|-------|
| **FAO stock status — Areas 71 & 57** | % of stocks fished sustainably | Area 71 = South China Sea, Gulf of Thailand, Indonesia/Philippines. Area 57 = Andaman Sea, Strait of Malacca. ⚠️ See caveat on the 2025 method change below. |
| **Sea Around Us** | Reconstructed catch per EEZ (incl. discards & unreported catch) and **Mean Trophic Index** | Every SEA EEZ, annual from 1950. Also on AWS: [registry.opendata.aws](https://registry.opendata.aws/sau-global-fisheries-catch-data/) |
| **OWID / World Bank capture production** | Catch per SEA country | Sum the 5 focus countries for a regional series |
| **Global Fishing Watch** *(optional)* | Fishing effort (hours) by EEZ & flag, 2012–2024 | Free registration, non-commercial licence. Use monthly data — annual gives only ~13 points. [globalfishingwatch.org](https://globalfishingwatch.org/?p=6131) |
| **SEAFDEC Fishery Statistical Bulletin** *(optional)* | Official ASEAN fishery statistics | Mostly PDF tables, more cleaning. [seafdec.org](https://www.seafdec.org/stat2022/) |

Raw downloads go in `data/raw/` (not committed if large — see `.gitignore`). Cleaned versions go in `data/processed/`.

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
2. **FAO changed its stock-assessment method in 2025.** In Area 71 the number of assessed stocks jumped from 43 to 265, and the sustainable share fell from ~77–80% to 52.5%. FAO states this does not necessarily mean stocks got worse. **Do not run a trend test across that break** — use pre-2025 data only, with Sea Around Us mean trophic level as a backup indicator.
3. **Gaps in early FAO years.** ACF and CCF assume evenly spaced data — check and handle missing years before running them.
4. **Short series.** Global Fishing Watch annual data gives ~13 points — too few for a reliable CCF. Use monthly aggregation if effort data is included.
5. **Country-level overexploitation is not published.** SEA stock status is only available by FAO fishing area, not per country.

---

## 6. Repository structure

```
sdg14-overfishing-analysis/
├── README.md            ← this file (project plan + progress log)
├── LICENSE              ← MIT
├── requirements.txt     ← Python dependencies
├── data/
│   ├── raw/             ← original downloads, untouched
│   └── processed/       ← cleaned, analysis-ready CSVs
├── notebooks/           ← Jupyter notebooks (stage1_global, stage2_sea, …)
├── src/                 ← reusable Python functions (loading, tests, plots)
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
jupyter lab
```

---

## Progress

| Step | Stage | Status | Notebook / file |
|------|-------|--------|-----------------|
| Repo & plan set up | — | ✅ Done | `README.md` |
| Download & clean global datasets | 1 | ⬜ Not started | — |
| Exploratory plots + ACF | 1 | ⬜ Not started | — |
| ADF + differencing | 1 | ⬜ Not started | — |
| H1a trend tests (Mann-Kendall, segmented regression) | 1 | ⬜ Not started | — |
| H1b cross-correlation (CCF) | 1 | ⬜ Not started | — |
| **Stage 1 decision** | 1 | ⬜ Pending | — |
| Download & clean SEA datasets | 2 | ⬜ Not started | — |
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

---

## Contributing

- Work on a branch or pull before you push (`git pull --rebase`) to avoid overwriting each other.
- **Whoever pushes code updates the [Progress](#progress) table and adds a line to the [Changelog](#changelog)** in the same commit, so this README always reflects the current state of the analysis.
- Keep notebooks numbered in run order (e.g. `01_stage1_load_clean.ipynb`, `02_stage1_acf_adf.ipynb`).
- Don't commit large raw data files — add a download link/script instead.

---

## License

Code in this repository is released under the [MIT License](LICENSE). Datasets remain under their original providers' licences (OWID: CC BY 4.0; Sea Around Us, FAO, Global Fishing Watch and SEAFDEC: see each provider's terms).
