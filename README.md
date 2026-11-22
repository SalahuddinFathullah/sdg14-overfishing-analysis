# SDG 14 — Is Overfishing Getting Worse? (Global → Southeast Asia)

**KIE4033 Assignment 1 — Exploratory Data Analytics** · Universiti Malaya · Semester 1, 2026/2027
Theme: **UN Sustainable Development Goal 14 — Life Below Water**

We apply exploratory data analysis (EDA) and statistical hypothesis testing in Python to test whether overfishing is a genuine and worsening problem — first **globally**, and then, **only if the global result holds**, for **Southeast Asia (SEA)**.

> **Status:** 🟢 Analysis complete — data collected and cleaned ([`data/processed/`](data/processed/)), validation and integrity formally tested ([`3-data-quality/`](3-data-quality/)), 34 exploratory figures ([`0-visualisation/`](0-visualisation/)), course-method tests ([`1-statistical-tests/`](1-statistical-tests/)) and formal hypothesis tests H1a–H2c in five iterations — catch, fishing effort, ocean warming, shipping & built structures, pooled panel + stock rebuilding + Fishing-in-Balance + 2030 forecast ([`2-hypothesis-tests/`](2-hypothesis-tests/)) done; only the report write-up remains (figure set: [`figures/README.md`](figures/README.md)). See [Progress](#progress).

---

## 0. Assignment deliverables

- Python project developed in this GitHub repo (link goes in the report)
- Brief report (~10 pages), submitted as `KIE4033_Asgn1_GroupXY.pdf`
- Pair project (2 students)

### How the work maps to the rubric

| Report section (weight) | What we produce | Where in this repo |
|---|---|---|
| Introduction (10%) | Problem statement, objectives, research questions (H1a–H2c below) | [§1](#1-goal), [§2](#2-hypotheses) |
| Data Understanding & Preparation (15%) | Dataset descriptions, cleaning, missing-value handling, summary statistics, correlation matrix, **validation against alternative sources** | [§3](#3-datasets), [`1-statistical-tests/`](1-statistical-tests/) §A (summary stats, outliers, missing-value map), [`3-data-quality/`](3-data-quality/) (cleaning → `data/processed/`, formal validation and integrity tests) |
| EDA (25%) | Labelled time-series plots, ACF plots, regional comparisons, anomaly notes | [`0-visualisation/`](0-visualisation/) |
| Statistical Analysis (25%) | ADF, Mann-Kendall, segmented regression, CCF, interaction regression; **assumptions checked**; **post-hoc tests**; course methods (t-test/ANOVA vs Wilcoxon/Kruskal-Wallis, rank-sum, chi-square, k-means + hierarchical clustering) | [§4](#4-analysis-pipeline), [`1-statistical-tests/`](1-statistical-tests/) §B–E, [`2-hypothesis-tests/`](2-hypothesis-tests/) |
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

### Alternative drivers — added in later iterations of the analysis

| ID | Iteration | H₀ | H₁ |
|----|-----------|----|----|
| **H3a / H3b** | 3 · ocean warming | Sea temperature is not associated with % overexploited (world) or with % overexploited / MTL / catch (SEA) beyond their shared trend | Association beyond the trend (detrended, cointegration with trend, or CCF) |
| **H4a / H4b** | 4 · shipping & structures | Port traffic, seaborne trade, liner connectivity, dam building and aquaculture are not associated with % overexploited / MTL beyond the shared trend | Association beyond the trend, consistent across sources and detrending |
| **H5a** | 5 · pooled panel | Pooling all 11 SEA EEZs with EEZ and year fixed effects, no pressure measure (effort, industrial / unreported / foreign shares, catch) raises % overexploited or lowers MTL | A robust, significant effect in the 'worse' direction |
| **H5b** | 5 · stock rebuilding | The share of rebuilding stocks did not increase after 2015 | Significant increase after 2015 |
| **H5c** | 5 · Fishing-in-Balance | The FiB index has no downward trend since 1990 | FiB falling (catch no longer offsets trophic-level change) |
| **H5d** | 5 · forecast to 2030 | The 2030 forecast of % overexploited is not above today's level | 95% prediction interval for 2030 lies above today's level (with the trend validated in a back-test) |
| **H4c** | 4 · artificial islands | EEZs facing the Spratly Islands did not change after China's 2013–16 island building, relative to other SEA EEZs | Significant difference-in-differences |

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

| **Ocean warming** — *added in iteration 3*: HadSST world SST anomaly and NOAA upper-ocean heat content (via OWID); NOAA ERSST v5 SST averaged over each SEA5 EEZ | °C anomaly, 10²² J, annual EEZ SST, warm-month count | World 1950–2025; 5 SEA EEZs 1950–2025 | `owid/sea-surface-temperature-anomaly.csv`, `owid/ocean-heat-content-upper.csv`, `validation/climate/` |
| **Shipping & structures** — *added in iteration 4*: container port traffic and liner shipping connectivity (World Bank WDI / UNCTAD), seaborne trade (UNCTADstat), Global Dam Tracker, DeepOWT offshore wind, OOGP offshore oil & gas platforms | TEU, index, t, dam completions, turbine and platform counts | Shipping 2000–2024; dams 1900–2019; wind 2016–21; platforms 2017–23 | `validation/human_use/` |
| **IMAS Global Fishing Effort (Rousseau et al.)** — *added in iteration 2* | Fishing effort (kW × days at sea), vessels, engine power; world and inside each SEA5 EEZ | World + 5 SEA EEZs, 1950–2017 | `validation/effort/` |

**Why Sea Around Us became the main SEA source:** FAO publishes stock status only by large fishing area, and the OWID series for Areas 71/57 has just 10 uneven points — fine for a Mann-Kendall test but too short for ACF/CCF. Sea Around Us gives an annual 70-year series per EEZ, which supports the full pipeline and lets us compare the 5 countries (Kruskal-Wallis + Dunn).

### Alternative sources for cross-validation (`data/raw/validation/`)

| Source | Checks | Type of check |
|--------|--------|---------------|
| UN SDG Global Database — indicator 14.4.1 | FAO % overexploited (OWID) | Same data, official publisher → processing check |
| World Bank WDI API — `ER.FSH.CAPT.MT` | Capture production (OWID) | Same data, original publisher → processing & revision check |
| Sea Around Us — World, FAO Areas 57/71 | FAO % overexploited, FAO capture | **Different method** (catch-based status; reconstructed catch) |
| Sea Around Us — 5 SEA Large Marine Ecosystems | Sea Around Us EEZ series | **Different spatial unit** (ecosystem vs national zone) |

Details and a first look at agreement: [data dictionary §3](data/raw/README.md#3-alternative-sources-for-cross-validation--validation).

### Data-integrity sources — is officially reported catch believable? (`data/raw/validation/`)

| Source | Check |
|--------|-------|
| SDG 14.7.1 (fisheries % of GDP, via OWID) + World Bank GDP & capture, ~100 countries | **Implied value per tonne** — does reported catch fit the money it earns, compared with peers? |
| UN Comtrade, fish exports (HS 03), 2011–2023 | **Mirror trade** — exports a country reports vs imports its partners report from it |
| Sea Around Us taxon-level catch records | **Benford's law** on leading digits of reported vs reconstructed catch |
| Sea Around Us catch summary (already in repo) | **Reconstructed ÷ reported** — how much catch is missing from official statistics |

Details: [data dictionary §4](data/raw/README.md#4-data-integrity-sources--is-reported-catch-plausible).

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

### Data validation against alternative sources (Data Understanding & Preparation)

Before testing hypotheses, each main series is checked against an alternative source over their common years. Two sources rarely agree on **levels** (different coverage and methods: e.g. FAO capture includes inland catch, Sea Around Us adds unreported catch), so validity means they agree on **direction and trend**, while any level gap is explained.

| Question | Test | Pairs |
|---|---|---|
| Is there a systematic difference between the two sources? | **Wilcoxon signed-rank** on paired years (paired *t*-test only if the differences pass Shapiro-Wilk) | all pairs |
| How large is the bias and how wide is the disagreement? | **Bland-Altman**: mean difference ± 1.96 SD (limits of agreement); use % difference when levels differ in scale | same-unit pairs (% stocks, tonnes) |
| Do they agree point-for-point? | **Lin's concordance correlation coefficient (CCC)** — penalises both scatter and bias, unlike Pearson | same-unit pairs |
| Do they move together? | **Spearman ρ on levels *and* on first differences** — level correlation is inflated by shared trends; differences show year-to-year agreement | all pairs |
| Do they show the same trend? | **Sen's slope with 95% CI** for each source, plus Mann-Kendall direction; overlapping CIs → same trend | all pairs |

Because years are autocorrelated, the paired-test *p*-values are optimistic; we report them alongside effect sizes (bias, CCC, Sen's slope) rather than relying on *p* alone.

---

## 5. Known caveats (for the Limitations section)

1. **Catch ≠ overfishing.** Catch rises while a fishery develops, then falls once stocks collapse. Lagged correlation captures part of this, but catch is a proxy, not a direct measure.
2. **FAO changed its stock-assessment method in 2025.** In Area 71 the number of assessed stocks jumped from 43 to 265, and the sustainable share fell from ~77–80% to 52.5%. Our FAO data ends in 2021, so the break is not in our series — but mention that newer FAO figures are not comparable.
3. **Interpolated FAO years.** The World overexploited series has stretches that rise by a constant step each year (linear interpolation between assessments), which inflates autocorrelation. Robustness check: re-run H1a on assessment years only.
4. **Sea Around Us stock status is derived from catch.** A stock is labelled "over-exploited"/"collapsed" from its catch relative to its historical peak (Kleisner et al. 2013). So for **H2b**, correlating catch with stock status is partly circular — use **mean trophic level** (independent of catch volume) as the main overfishing response, and treat stock status as supporting evidence.
5. **Country-level FAO stock status is not published.** FAO reports by fishing area only; that is why Sea Around Us EEZ data is used for SEA.
6. **Different end years.** FAO stocks end 2021, SAU 2019, capture 2024 — align on the common window before correlating.
7. **Shared trends (iterations 3–5).** Effort, warming, shipping, dams and aquaculture all rise with overexploitation, so level correlations are expected whatever the cause. Linear-detrended correlations of curved series can also mislead, which is why links had to hold in both sources and under linear and quadratic detrending.
8. **Driver data limits.** Effort has source breaks (Thailand 2002/2011, Malaysia 1965) and ends in 2017; shipping series start in 2000; 34% of GDAT dams are undated; offshore platforms and wind overlap the fisheries data by 3–4 years; artificial islands are a single dated event.
9. **Coarse resolution.** EEZ- and country-level yearly data can't detect local effects (e.g. reef loss from island building) or separate drivers that rise together; pooling EEZs (H5a) didn't change that.
10. **Few clusters.** The pooled panel has 11 EEZs (5 countries for effort), so clustered standard errors over-reject; p-values come from a wild cluster bootstrap.
11. **Catch-based categories.** SAU "rebuilding" (H5b) and the FiB index (H5c) both come from catch data, so they describe fisheries, not surveyed biomass.

---

## 6. Repository structure

```
sdg14-overfishing-analysis/
├── README.md            ← this file (project plan + progress log)
├── 0-visualisation/
│   ├── visualise.py     ← creates every exploratory figure from data/raw/
│   ├── README.md        ← figure gallery with descriptions
│   └── figures/         ← 34 PNGs in 5 sections (global, sea_owid, sea_sau, validation, integrity)
├── 1-statistical-tests/
│   ├── stat_tests.py    ← course methods: EDA profiling, t-test/ANOVA vs non-parametric, rank-sum,
│   │                      chi-square, k-means + hierarchical clustering
│   ├── README.md        ← method ↔ course-topic map, results tables and interpretation
│   ├── results/         ← one CSV per test + test_summary.csv
│   └── figures/         ← 8 PNGs in 5 sections (A profile … E clustering)
├── 2-hypothesis-tests/
│   ├── hypothesis_tests.py ← stationarity, H1a, H1b, H2a, H2b, H2c, residual diagnostics;
│   │                         iterations 2 (effort), 3 (warming, H3), 4 (shipping & structures, H4)
│   ├── README.md        ← verdicts, method per hypothesis, notes for the report
│   ├── results/         ← one CSV per test family + decisions.csv
│   └── figures/         ← 20 PNGs (iteration 1: 6; iterations 2–5: 3 + 2 + 4 + 4; summary S.01)
├── 3-data-quality/
│   ├── clean_data.py    ← builds data/processed/ (rules R1–R6, cleaning_log.csv)
│   ├── quality_tests.py ← formal validation (21 source pairs) and integrity tests (Benford,
│   │                      mirror trade, peer outliers, unreported share)
│   ├── README.md        ← cleaning rules, validation and integrity results
│   ├── results/         ← one CSV per test
│   └── figures/         ← V.01 validation summary, I.01 integrity summary
├── LICENSE              ← MIT
├── requirements.txt     ← Python dependencies
├── data/
│   ├── raw/             ← original downloads (OWID + Sea Around Us), see data/raw/README.md
│   │   └── validation/  ← alternative sources for cross-checking (UN SDG, World Bank, SAU regions)
│   │                      data-integrity sources (economic/, comtrade/, benford/) and
│   │                      iteration drivers (effort/, climate/, human_use/)
│   └── processed/       ← cleaned, analysis-ready CSVs + cleaning_log.csv (see its README)
├── notebooks/           ← (unused placeholder — the analysis lives in the numbered script folders)
├── src/
│   └── download_data.py ← re-downloads everything in data/raw/
└── figures/README.md    ← final figure set for the report (links to the generated figures)
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
# regenerate all exploratory figures:
python 0-visualisation/visualise.py
# run the course-method statistical tests (tables + figures):
python 1-statistical-tests/stat_tests.py
# run the formal hypothesis tests (H1a–H2c, iterations 2–4):
python 2-hypothesis-tests/hypothesis_tests.py
# build the cleaned tables and run the validation / integrity tests:
python 3-data-quality/clean_data.py
python 3-data-quality/quality_tests.py
jupyter lab
```

---

## Progress

| Step | Stage | Status | Notebook / file |
|------|-------|--------|-----------------|
| Repo & plan set up | — | ✅ Done | `README.md` |
| Download global datasets | 1 | ✅ Done | `data/raw/owid/`, `src/download_data.py` |
| Download alternative sources for validation | 1–2 | ✅ Done | `data/raw/validation/` |
| Clean global datasets | 1 | ✅ Done — `data/processed/global_annual.csv`, rules R1–R6 logged, 0 range/duplicate violations | `3-data-quality/clean_data.py` |
| Course-method tests: assumption checks, t-test/ANOVA vs Wilcoxon/Kruskal-Wallis, rank-sum before/after 2015, chi-square, k-means + hierarchical clustering | 1–2 | ✅ Done | `1-statistical-tests/stat_tests.py` |
| Validate main data vs alternative sources | 1–2 | ✅ Done — 21 pairs: 7 identical, 10 of 14 others agree on trend, 4 on direction | `3-data-quality/quality_tests.py` §V |
| Exploratory plots — SEA | 2 | ✅ Done | `0-visualisation/figures/2_sea_owid/`, `3_sea_sau/` |
| Data-integrity checks (GDP, mirror trade, Benford, reconstruction gap) | 1–2 | ✅ Done — digits genuine, no peer outliers; Vietnam re-export gap, Malaysia gap widening; unreported share falling in 3 countries | `3-data-quality/quality_tests.py` §I |
| Exploratory plots + ACF | 1 | ✅ Done | `0-visualisation/figures/1_global/` |
| ADF + KPSS + differencing | 1–2 | ✅ Done | `2-hypothesis-tests/` (stationarity) |
| H1a trend tests (Mann-Kendall, Sen, segmented regression, Chow) | 1 | ✅ Done — reject H₀ | `2-hypothesis-tests/` |
| H1b cross-correlation (CCF, prewhitening, Granger, cointegration) | 1 | ✅ Done — fail to reject H₀ | `2-hypothesis-tests/` |
| **Stage 1 decision** | 1 | ⚠️ Rule not met (H1a yes, H1b no) → Stage 2 reported as exploratory | `2-hypothesis-tests/results/decisions.csv` |
| Download SEA datasets | 2 | ✅ Done | `data/raw/sau/`, `data/raw/owid/` |
| Clean SEA datasets | 2 | ✅ Done — `sea_country_annual.csv`, `sea_eez_annual.csv` | `3-data-quality/clean_data.py` |
| H2a / H2b tests on SEA | 2 | ✅ Done — H2a reject H₀, H2b fail to reject | `2-hypothesis-tests/` |
| H2c SEA vs global comparison | 2 | ✅ Done — SEA = world; Indonesia, Thailand, Malaysia worse | `2-hypothesis-tests/` |
| **Iteration 2:** fishing effort replaces catch as the pressure variable (H1b, H2b re-tested; CPUE) | 1–2 | ✅ Done — CPUE falling; short-run links still not significant | `2-hypothesis-tests/` §7, `data/raw/validation/effort/` |
| **Iteration 3:** ocean warming as an alternative driver (new H3a global, H3b SEA) | 1–2 | ✅ Done — shared trend only; no link beyond it | `2-hypothesis-tests/` §8, `data/raw/validation/climate/` |
| **Iteration 4:** shipping and built structures — ports, seaborne trade, liner connectivity, dams, aquaculture, offshore platforms & wind, artificial islands (new H4a global, H4b SEA, H4c island-building DiD) | 1–2 | ✅ Done — shared trend only; island building shows no effect; platforms/wind too recent to test | `2-hypothesis-tests/` §9, `data/raw/validation/human_use/` |
| **Iteration 5:** pooled panel with fixed effects (H5a), stock rebuilding after 2015 (H5b), Fishing-in-Balance index (H5c), forecast to 2030 (H5d) | 1–2 | ✅ Done — no pressure effect even pooled; SEA rebuilding stalled; FiB rising (expansion), falling only in Thailand; world ~41% overexploited by 2030 | `2-hypothesis-tests/` §11 |
| Final figure selection (15 figures mapped to report sections; report outline drafted) | — | ✅ Done | [`figures/README.md`](figures/README.md) |
| Report write-up (~10 pages) | — | ⬜ Team task | — |

### Findings so far

_Exploratory figures: [`0-visualisation/`](0-visualisation/). Course-method tests: [`1-statistical-tests/`](1-statistical-tests/). Hypothesis tests: [`2-hypothesis-tests/`](2-hypothesis-tests/)._

**Hypothesis verdicts**

| | Verdict | Evidence |
|---|---|---|
| H1a World trend | ✅ Reject H₀ | Hamed-Rao Mann-Kendall p < 0.001, Sen +0.55 pp/yr [0.50, 0.59]; holds without interpolated years and in Sea Around Us; no slowdown after 2015 (slope change p = 0.49, Chow p = 0.89) |
| H1b World catch → overexploitation | ❌ Fail to reject | No lag 0–10 significant after Holm (max \|r\| 0.23); prewhitened, Granger and cointegration (p = 0.20) agree — catch has been flat since the 1990s |
| Stage rule | ⚠️ Not met | Stage 2 reported as exploratory |
| H2a SEA trend | ✅ Reject H₀ | SEA % overexploited +7.4 pp/decade (p < 0.001); every country and EEZ rises 1950–2019; mean trophic level falls only in the Philippines (1990–2019) |
| H2b SEA catch → MTL | ❌ Fail to reject | 0 of 66 lag tests significant after Holm |
| H2c SEA vs world | ➖ Mixed | SEA as a whole = world pace since 1990 (difference 0.00 pp/yr, p = 0.91); Indonesia (+0.97), Thailand (+0.42), Malaysia (+0.26 pp/yr) significantly faster; Vietnam slower |
| *Iteration 2* H1b with effort | ❌ Fail to reject | Effort growth vs Δ% overexploited: max r = +0.35 (lag 3), Holm p = 0.29; Granger min Holm p = 0.07; long-run cointegration p = 0.078 (10% level only) |
| *Iteration 2* H2b with effort | ➖ Weak | 1 of 132 lag tests significant (Indonesia, same-year, ΔMTL, r = +0.35); none after correcting all 132 together. Stock status is now testable (effort is not catch-based) |
| *Iteration 2* CPUE (explanation) | 📉 Falling | Catch per kW-day 1990–94 → 2013–17: world −35% (effort +44%, catch −6%), SEA −50%; significant in Malaysia (−68%), Indonesia (−54%), Philippines (−43%) |
| *Iteration 3* H3a warming (global) | ❌ Fail to reject | SST anomaly vs % overexploited: levels ρ = +0.86, but detrended ρ = −0.11 (FAO) / +0.44 (SAU, curvature artefact); cointegrated only without a trend term (p = 0.05) and not jointly with effort (p = 0.19); year-to-year CCF and Granger all Holm p ≥ 0.56 |
| *Iteration 3* H3b warming (SEA) | ➖ Weak | SEA seas warm +0.12 °C/decade (p < 0.001); warm-anomaly months 0.2/yr (1950–89) → 5.6/yr (1990–2025). 2 of 198 lag tests significant within row (SEA Δ% overexploited at lag 10, Indonesia ΔMTL at lag 4); none across all tests |
| *Iteration 4* H4a shipping & structures (global) | ❌ Fail to reject | Container traffic, seaborne trade, dams, aquaculture: levels ρ = +0.93 to +1.00 (shared trend). No beyond-trend link is consistent across FAO and SAU and across linear and quadratic detrending (dams: +0.56 FAO vs −0.65 SAU); not cointegrated (p ≥ 0.06); 0 of 74 CCF tests significant across all |
| *Iteration 4* H4b shipping & structures (SEA) | ➖ Weak | 2 of 492 lag tests significant within row (Thailand dam completions → ΔMTL, lags 6–7); none across all tests |
| *Iteration 4* H4c artificial islands (SEA) | ❌ Fail to reject | Difference-in-differences, Spratly-facing EEZs vs other SEA EEZs around China's 2014 reclamation: % overexploited β = +0.7 pp (randomization p = 0.91); MTL and catch also n.s.; placebo 2009 n.s. Military outposts sit on the same reefs; offshore platforms/wind overlap the data by 3–4 years only (descriptive) |
| *Iteration 5* H5a pooled panel (SEA) | ❌ Fail to reject | 11 EEZs × 70 years, EEZ + year fixed effects, wild cluster bootstrap: 0 of 36 coefficients significant after Holm; the strongest (foreign-fleet share, raw p = 0.02) points the 'wrong' way |
| *Iteration 5* H5b stock rebuilding | ➖ Mixed | World rebuilding sped up after 2015 (+1.00 pp/yr, p = 0.016); SEA's slowed (−0.43 pp/yr, p = 0.002; 9.0% → 9.3%); 6 of 11 EEZs up (Wilcoxon p = 0.26); Vietnam and the Philippines down |
| *Iteration 5* H5c Fishing-in-Balance | ❌ Fail to reject (SEA) | SEA FiB rising +0.15/decade (p < 0.001), mostly from catch growth: an expansion phase that explains the rising SEA trophic level; FiB falling only in Thailand (−0.10/decade, Holm p = 0.017) |
| *Iteration 5* H5d forecast to 2030 | ✅ Reject H₀ (world) | FAO world 35.5% (2021) → 41.3% in 2030 [95% PI 36.7–46.1], back-test validated; SEA (SAU) has flattened since 2015, so its projected rise isn't supported — no improvement, not a further rise |
| Stage rule (both iterations) | ⚠️ Not met | Stage 2 stays exploratory |

**Why iteration 2:** iteration 1 found no link between catch and overexploitation. Catch is a weak measure of pressure, because it can stay flat while fishing intensifies if each trip lands less. Effort measures pressure directly. Result: effort rose 44% worldwide and ~170% in SEA since the early 1990s while catch per unit effort fell. That is the long-run explanation (depletion). Year-to-year effort changes still don't predict year-to-year stock-status changes, which fits stock status being a slow, cumulative indicator.

**Why iteration 3:** with fishing pressure not explaining the short-run pattern, we tested whether ocean warming does. The oceans are clearly warming, worldwide and in every SEA EEZ. But warming and overexploitation are only linked through a shared upward trend. Once the trend is removed, or effort is added to the model, the link disappears. So the data cannot attribute the rise in overexploitation to warming any more than to effort. Both are steadily rising, and so are demand and fleet technology.

**Why iteration 4:** none of catch, effort or warming was linked to overexploitation beyond a shared trend, so we tested human uses of the sea that harm stocks through habitat rather than catch: shipping, dams (river flow and sediment to the coast), aquaculture, offshore platforms and wind, and artificial islands. Same result: everything that grows steadily correlates with overexploitation in levels, and nothing does once the trend is removed. A new robustness check (quadratic as well as linear detrending, and both data sources) shows the few "beyond-trend" correlations are curvature artefacts. Island building, the one driver with a clean before/after event, left no trace on EEZ-wide indicators. Across four iterations the consistent message is that overexploitation rises **slowly and cumulatively with total human pressure**. Yearly national data cannot single out one driver; that needs stock-level or spatial data, which goes in the report's limitations.

**Why iteration 5:** every null result so far could be blamed on short single series. Pooling all 11 EEZs with fixed effects (which also removes the shared trend) still finds no pressure effect, so the nulls reflect the data's resolution, not too few years. Two new questions close the SDG story: rebuilding sped up worldwide after 2015 but stalled in SEA, so the "restore stocks" half of SDG 14.4 is missed there too; and the Fishing-in-Balance index shows SEA catches are still expanding into new and higher-trophic stocks, which is why SEA mean trophic level rises while overexploitation also rises. Thailand is the only country past that phase (catch and FiB falling). Finally, a back-tested forecast puts the world at about 41% overexploited by 2030, so the SDG deadline will be missed by a wider margin than today.



- **World:** % of stocks overexploited rose from ~10% (1974) to 35.5% (2021); the fitted slope is +0.55 pp/yr before 2015 and +0.48 pp/yr after — little sign of SDG-era improvement (fig 1.01).
- **Catch vs overexploitation:** world wild catch plateaued at ~90–95 Mt from the mid-1990s while overexploitation kept rising (figs 1.03–1.04).
- **ACF:** both series are strongly autocorrelated in levels and close to white noise after differencing → correlation tests must use differenced series (fig 1.05).
- **SEA:** FAO Area 71 (South China Sea etc.) went from ~6% to 47% overexploited between 2004 and 2021, the largest rise of any FAO area (fig 1.07). In Sea Around Us data, ~50% of stocks in most SEA EEZs are collapsed or over-exploited by 2019, tracking or exceeding the world level; Vietnam is the exception (~20%) (figs 3.01–3.03).
- **Validation (formal, `3-data-quality`):** 7 source pairs are identical (same data, two publishers); of 14 pairs from different sources, 10 agree on trend (overlapping Sen's-slope CIs) and 4 on direction only — none disagrees. Level gaps have known causes (FAO capture includes inland catch; Sea Around Us stock status is catch-based and sits higher, bias −4.1 pp, CCC 0.74) (figs 4.01–4.06, V.01).
- **Missing values:** no data loss — the only gaps are reporting cycles (FAO areas 71/57 assessed ~every 2 years, SDG 14.7.1 published every 2 years); 5 FAO world years look linearly interpolated (fig A.01).
- **Test choice:** normality fails for most series (Shapiro-Wilk), so non-parametric tests are used. FAO is systematically 4.1 pp below Sea Around Us (Wilcoxon p < 0.001, r = −0.58). Countries' yearly MTL changes do not differ (Kruskal-Wallis p = 0.32); a plain ANOVA would have wrongly said they do (p = 0.033, unequal variances) (fig B.01).
- **SDG effect:** no indicator changed significantly after 2015 (rank-sum / Welch, all p ≥ 0.22); the world rise looks slower (median +1.1 → +0.5 pp/yr) but 6 post-2015 years are too few (fig C.01).
- **Chi-square:** SEA overexploitation class depends strongly on period (Cramér's V = 0.68; 28% of EEZ-years ≥ 50% overexploited in 2000–19 vs 0% before 1980) and on country in 2010–19 (V = 0.51; Indonesia and Thailand highest, Vietnam lowest) (fig D.01). Chi-square rejects Benford only because N is huge — MAD confirms conformity (fig D.02).
- **Clustering:** globally, Area 71 (SEA) sits with the lower-level FAO areas but has one of the steepest trends (+1.6 pp/yr). Among SEA EEZs, Vietnam is an outlier; the rest form 3 groups that cut across borders — Indonesia (fast-rising overexploitation, little unreported catch), Malaysia Peninsula/Sabah + Philippines (high unreported share), Sarawak + Thailand (most overexploited, catch ~40% below peak) (figs E.01–E.03).
- **Data integrity (SEA5):** no sign of fabricated catch figures — leading digits of reported catch follow Benford's law closely for every country (MAD 0.0007–0.0028, Nigrini "close conformity"), and no country's fisheries value-vs-catch is a peer outlier (|z| ≤ 1.4). The real gap is **under-reporting**: Sea Around Us estimates 45% (Philippines), 39% (Malaysia), 28% (Vietnam), 24% (Thailand) and 10% (Indonesia) of 2015–19 catch is missing from official statistics. Mirror trade, formally tested (one-sample Wilcoxon 2011–2023, Holm): **Vietnam** is significantly below 0 (median −10.8%, re-exports/transshipment); the **Philippines** median is +13.9% (above the 0–10% band, with +61–64% in 2011–12) but not significant after Holm (p = 0.14, n = 13); **Malaysia**'s median is −1% but the gap is widening +2.8 pp/yr (Holm p = 0.02, +10% to +22% since 2019). Unreported share is falling significantly since 1990 in Indonesia, Vietnam and Malaysia; the integrity signals do not rank countries consistently (all Spearman p ≥ 0.19, n = 5) (figs 5.01–5.06, I.01).

---

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-11-18 | Salah | Initial repo: project plan, hypotheses, datasets, analysis pipeline, rubric mapping, assumption & post-hoc tests, MIT licence |
| 2026-11-18 | Salah | Added all raw datasets (6 OWID + 33 Sea Around Us files), `src/download_data.py`, data dictionary; switched SEA source to Sea Around Us EEZ data; added circularity caveat for SAU stock status |
| 2026-11-22 | Salah | Completed remaining steps: `3-data-quality/` (clean_data.py → `data/processed/` with cleaning log; quality_tests.py → formal validation of 21 source pairs and integrity tests, 2 figures, 6 tables); final figure set in `figures/README.md`; README progress/status/findings updated |
| 2026-11-23 | Salah | Iteration 5 · H5d: forecast of % overexploited to 2030 (`i5_h5d`: linear trend / damped ETS / ARIMA on logit %, 5-year back-test vs no-change forecast), figure I5.04, 4 tables, decision row |
| 2026-11-23 | Salah | Iteration 5: `hypothesis_tests.py` sections i5_h5a (pooled two-way fixed-effects panel, wild cluster bootstrap), i5_h5b (rebuilding share before/after 2015), i5_h5c (Fishing-in-Balance index + decomposition); 3 figures, 4 tables, H5 decision rows; caveats §5 extended with iterations 3–5 limitations; iterations 1–4 kept |
| 2026-11-23 | Salah | Added `summary` section to `hypothesis_tests.py`: all 8 world drivers from iterations 1–4 on one common method (levels vs linear/quadratic-detrended ρ, CCF), figure S.01 and 2 tables; S.01 replaces I3.01 + I4.01 in the report figure set |
| 2026-11-22 | Salah | Iteration 4: added shipping and structure data (`data/raw/validation/human_use/`: WDI container traffic + liner connectivity incl. Dec-2020 archive, UNCTADstat seaborne trade, GDAT dams, DeepOWT offshore wind, OOGP platforms; `download_data.py --only human_use`); `hypothesis_tests.py` sections i4_data, i4_h4a, i4_h4b, i4_h4c (island-building DiD with randomization inference), i4_descriptive (4 figures, 11 tables, H4 decision rows); `ccf_table` takes an optional `max_lag`; iterations 1–3 kept |
| 2026-11-22 | Salah | Iteration 3: added ocean-warming data (OWID HadSST + ocean heat content; NOAA ERSST v5 averaged over SEA EEZs; `download_data.py --only climate`); `hypothesis_tests.py` sections i3_stationarity, i3_h3a, i3_h3b (2 figures, 8 tables, H3 decision rows); iterations 1–2 kept |
| 2026-11-22 | Salah | Iteration 2: added IMAS fishing-effort data (`data/raw/validation/effort/`, `download_data.py --only effort`); `hypothesis_tests.py` sections i2_stationarity, i2_cpue, i2_h1b, i2_h2b (3 figures, 6 tables, decisions rows); iteration-1 results kept. Fixed undefined `RAW` in `download_integrity()` |
| 2026-11-22 | Salah | Added `2-hypothesis-tests/` (hypothesis_tests.py + 6 figures + 12 result tables + README): ADF/KPSS stationarity, H1a Mann-Kendall/Sen/segmented/Chow, H1b CCF/prewhitening/Granger/cointegration, H2a SEA trends, H2b SEA CCF, H2c interaction regression, residual diagnostics, decisions table; README status, progress, verdicts |
| 2026-11-22 | Salah | Added `1-statistical-tests/` (stat_tests.py + 8 figures + 13 result tables + README): EDA profiling and missing-value map, Shapiro/Levene-gated t-test/ANOVA vs Wilcoxon/Kruskal-Wallis, rank-sum before/after 2015, chi-square independence with Cramér's V, k-means + hierarchical clustering; added scikit-learn to requirements |
| 2026-11-22 | Salah | Filled Philippines 2014–2023 Comtrade mirror-trade data (previously blocked by the API rate limit); re-rendered figs 5.04 and 5.06 |
| 2026-11-22 | Salah | Added data-integrity sources recommended for validation (SDG 14.7.1 + World Bank GDP/capture, UN Comtrade mirror trade, Benford digit counts from SAU records), `download_data.py --only integrity`, and visualisation section 5 (6 figures) |
| 2026-11-21 | Salah | Added `0-visualisation/` (visualise.py + 28 figures + gallery README); fixed domestic/foreign fleet labels in `sau/catch_summary/` for the 9 split EEZs (and in `download_data.py`); first exploratory findings |
| 2026-11-18 | Salah | Added alternative sources for cross-validation (UN SDG 14.4.1, World Bank API, Sea Around Us World / FAO areas 57 & 71 / 5 SEA LMEs; 26 files), validation test plan, download script `--only validation` |

---

## Contributing

- Work on a branch or pull before you push (`git pull --rebase`) to avoid overwriting each other.
- **Whoever pushes code updates the [Progress](#progress) table and adds a line to the [Changelog](#changelog)** in the same commit, so this README always reflects the current state of the analysis.
- Keep notebooks numbered in run order (e.g. `01_stage1_load_clean.ipynb`, `02_stage1_acf_adf.ipynb`).
- Don't commit large raw data files — add a download link/script instead.

---

## License

Code in this repository is released under the [MIT License](LICENSE). Datasets remain under their original providers' licences (OWID: CC BY 4.0; Sea Around Us: free for non-commercial use with citation; FAO: see FAO terms). Full citations are in [`data/raw/README.md`](data/raw/README.md#references-for-the-report).
