# Raw data — data dictionary

Files in this folder are **untouched downloads** from the original sources (OWID files are only filtered to the entities listed below). Do not edit them by hand: clean in a notebook and write results to `data/processed/`.

Regenerate everything with:

```bash
python src/download_data.py
```

Retrieved: **2026-11-18** (see `manifest.json`).

---

## Which file answers which hypothesis

| Hypothesis | Variable | File |
|---|---|---|
| **H1a** global trend | % of stocks overexploited, World, 1974–2021 | `owid/share-of-fish-stocks-overexploited.csv` |
| **H1b** global cause | Global capture production (t), 1960–2024 | `owid/capture-fishery-production.csv` |
| **H2a** SEA trend | % stocks collapsed / over-exploited per EEZ, 1950–2019 | `sau/stock_status/*.csv` |
| **H2a** SEA trend (backup) | Mean trophic level of catch per EEZ | `sau/marine_trophic_index/*.csv` |
| **H2a** regional check | % overexploited, FAO Area 71 & 57, 2004–2021 | `owid/share-of-fish-stocks-overexploited.csv` |
| **H2b** SEA cause | Total catch per EEZ (t) vs stock status / MTL | `sau/catch_summary/*.csv` (sum `tonnes` by year) vs `sau/stock_status/*.csv`, `sau/marine_trophic_index/*.csv` |
| **H2c** SEA vs World | Same indicators, SEA vs World | combine the above |
| EDA context | Catch by gear, discards, aquaculture | `owid/fish-catch-gear-type.csv`, `owid/fish-discards.csv`, `owid/aquaculture-farmed-fish-production.csv` |

---

## 1. Our World in Data — `owid/`

Licence: **CC BY 4.0** (OWID), underlying data from FAO, World Bank and Sea Around Us.
Columns are always `Entity, Code, Year, <value columns>`.

| File | Value column(s) | Unit | Original source | Coverage in this file |
|---|---|---|---|---|
| `share-of-fish-stocks-overexploited.csv` | Share of fish stocks that are overexploited | % of assessed stocks | FAO, *State of World Fisheries and Aquaculture* | **Full file**: World 1974–2021 (48 yrs); 15 FAO fishing areas 2004–2021 (biennial, 10 pts); some countries 2018/2021 |
| `fish-stocks-within-sustainable-levels.csv` | Biologically sustainable, Overexploited | % of assessed stocks | FAO (SDG indicator 14.4.1) | Full file, same coverage as above |
| `capture-fishery-production.csv` | Capture fisheries production (metric tons) | tonnes / year | FAO via World Bank (`ER.FSH.CAPT.MT`) | World + SEA5, 1960–2024 |
| `fish-catch-gear-type.csv` | Unknown gear, Other gear, Longline, Gillnet, Small scale, Purse seine, Pelagic trawl, Bottom trawl | tonnes / year | Sea Around Us (reconstructed catch) | World + SEA5, 1950–2018 |
| `fish-discards.csv` | Discards | tonnes / year | Sea Around Us | World + SEA5, 1950–2018 (Malaysia & Thailand from 1961, Vietnam from 1987) |
| `aquaculture-farmed-fish-production.csv` | Aquaculture production (metric tons) | tonnes / year | FAO via World Bank | World + SEA5, 1960–2024 |

**FAO fishing areas relevant to SEA** (Entity names in the two stock files):
- `Western Central Pacific (FAO)` = **Area 71** — South China Sea, Gulf of Thailand, Indonesia/Philippines
- `Eastern Indian Ocean (FAO)` = **Area 57** — Andaman Sea, Strait of Malacca

**SEA5** = Malaysia, Indonesia, Thailand, Vietnam, Philippines.

---

## 2. Sea Around Us — `sau/`

Licence: Sea Around Us data are free for non-commercial use with citation (Pauly, Zeller & Palomares, *Sea Around Us*, www.seaaroundus.org). Dataset version **v50-1**.
One file per **Exclusive Economic Zone (EEZ)**; several countries are split into sub-EEZs:

| EEZ id | EEZ | Country |
|---|---|---|
| 459 | Malaysia (Peninsula West) — Strait of Malacca | Malaysia |
| 460 | Malaysia (Peninsula East) — South China Sea | Malaysia |
| 461 | Malaysia (Sabah) | Malaysia |
| 463 | Malaysia (Sarawak) | Malaysia |
| 937 | Indonesia (Central) | Indonesia |
| 361 | Indonesia (Eastern) | Indonesia |
| 938 | Indonesia (Indian Ocean) | Indonesia |
| 956 | Thailand (Andaman Sea) | Thailand |
| 957 | Thailand (Gulf of Thailand) | Thailand |
| 704 | Viet Nam | Vietnam |
| 608 | Philippines | Philippines |

File names are `<eez_id>_<eez_name>.csv`.

### `sau/stock_status/` — 1950–2019

`Data_Set, Year, Collapsed, Over-exploited, Exploited, Developing, Rebuilding`

| Data_Set | Meaning |
|---|---|
| `css` | **Cumulative stock status** — % of **catch** (tonnage) coming from stocks in each status |
| `nss` | **Number of stocks by status** — % of **stocks** in each status |

Stock status is assigned by Sea Around Us from each stock's catch relative to its peak catch (Kleisner et al. 2013). Rows sum to ~100 per year. **Overfishing indicator = Collapsed + Over-exploited.**

### `sau/marine_trophic_index/` — 1950–2019

Settings used: trophic level range 2–5, transfer efficiency 0.1 (website defaults).

| Column | Meaning |
|---|---|
| `year` | Year |
| `catch_sum` | Catch of taxa that have an assigned trophic level 2–5 (tonnes). **Lower than total catch** (70–96% of it) — use `catch_summary/` for total catch |
| `mean_trophic_level` | **Mean trophic level (MTL) of the catch** — falling MTL = "fishing down the food web" |
| `mtl_smoothed`, `catch_smoothed` | 3-year running means of the above |
| `fib_index` | Fishing-in-Balance index — rising FiB with falling MTL suggests geographic expansion of the fishery |
| `expansion_factor` | Factor by which the fishery area has expanded relative to 1950 |
| `mean_max_length` | Mean maximum body length of caught species (cm) — falling = smaller fish being caught |
| `RMTI_<year>` | Region-based MTI starting in that year (corrects for geographic expansion) |
| `*_unsmoothed`, `UNSMOOTHED_<year>` | Same values without smoothing |

### `sau/catch_summary/` — 1950–2019

The full SAU catch export (one row per taxon × fishing country × gear × end use; 13–53 MB per EEZ) **aggregated** by `src/download_data.py` (`summarise_catch`) to one row per:

| Column | Values |
|---|---|
| `year` | 1950–2019 |
| `fishing_sector` | Industrial, Artisanal, Subsistence, Recreational |
| `catch_type` | Landings, Discards |
| `reporting_status` | Reported (in official FAO statistics), Unreported (reconstructed by SAU) |
| `fleet` | `domestic` = fished by the EEZ's own country (the name outside or inside the brackets of the EEZ name, e.g. *Malaysia* for "Malaysia (Peninsula West)" and "Sabah (Malaysia)"); `foreign` = other countries' fleets |
| `tonnes` | Catch (t), summed |
| `landed_value` | Landed value (real 2010 USD), summed |

Total catch per year = `df.groupby("year").tonnes.sum()`. Useful extra angles: share of unreported catch, foreign fishing pressure, discard share.

---

## 3. Alternative sources for cross-validation — `validation/`

Each file overlaps a main series (same quantity, same years) but comes from a **different publisher, method or spatial unit**, so the main data can be checked against it.

| File | Cross-checks main file | Why it's a useful check |
|---|---|---|
| `un_sdg/14_4_1_ER_H2O_FWTL.csv` | `owid/share-of-fish-stocks-overexploited.csv` (overexploited = 100 − sustainable) | Official UN publication of FAO's numbers → tests that OWID's processing is faithful. Also extends World to **2023–2024** (FAO's new 2025 method — do not join to the old series) and gives **SEA country values 2016–2024** |
| `world_bank/ER.FSH.CAPT.MT.csv` | `owid/capture-fishery-production.csv` | Original publisher via API (vintage 2026-11-11, newer than OWID's 2026-07-27) → tests OWID processing and data revisions |
| `sau_regions/stock_status/global.csv` | FAO World % overexploited | **Different method**: FAO stock assessments vs SAU catch-based status |
| `sau_regions/stock_status/fao_71_*.csv`, `fao_57_*.csv` | FAO Areas 71 & 57 % overexploited | Same regions, different method |
| `sau_regions/catch/global.csv` | `owid/capture-fishery-production.csv` (World) | Officially **reported** catch vs SAU **reconstructed** catch (reported + unreported) |
| `sau_regions/*/lme_34…38_*.csv` | `sau/*` EEZ files | Same method, **different spatial unit** (ecosystem vs national zone) → tests that SEA conclusions don't depend on how the region is drawn |
| `sau_regions/marine_trophic_index/fao_*`, `global` | `sau/marine_trophic_index/*` | Regional/global mean trophic level as a benchmark for the EEZ series |

### `validation/sau_regions/` — 1950–2019

Same formats as the EEZ files (`stock_status/`, `marine_trophic_index/`), plus `catch/` with `year, landings, discards, reported, unreported` (tonnes; landings + discards = reported + unreported = total catch).

| Region file | Area |
|---|---|
| `global` | All oceans |
| `fao_57_indian_ocean_eastern` | FAO Area 57 (Andaman Sea, Strait of Malacca, Bay of Bengal…) |
| `fao_71_pacific_western_central` | FAO Area 71 (South China Sea, Gulf of Thailand, Indonesian & Philippine seas…) |
| `lme_34_bay_of_bengal` … `lme_38_indonesian_sea` | Large Marine Ecosystems 34 Bay of Bengal, 35 Gulf of Thailand, 36 South China Sea, 37 Sulu-Celebes Sea, 38 Indonesian Sea |

### `validation/un_sdg/14_4_1_ER_H2O_FWTL.csv`

`geo_area_code, geo_area_name, year, value_pct_sustainable, observation_status, nature, source` — % of fish stocks **within** biologically sustainable levels. Rows: World (1974–2021, 2023, 2024), Malaysia, Indonesia, Thailand, Viet Nam, Philippines (2016–2024), and the 16 FAO Major Fishing Areas.

### `validation/world_bank/ER.FSH.CAPT.MT.csv`

`country, iso3, year, capture_fisheries_production_t` — World + SEA5, 1960–2024 (WDI, last updated 2026-11-11).

### First look at agreement (Spearman ρ; preview only — formal tests go in the notebook)

| Pair | Years | ρ levels | ρ year-to-year changes |
|---|---|---|---|
| FAO % overexploited World: OWID vs UN SDG | 1974–2021 | 1.00 | 1.00 (identical) |
| Capture World: OWID vs World Bank API | 1960–2024 | 1.00 | 1.00 (tiny revisions) |
| % overexploited World: FAO vs SAU | 1974–2019 | 0.94 | 0.03 |
| % overexploited Area 71: FAO vs SAU | 2004–2019 (9 pts) | 0.40 | 0.19 |
| % overexploited Area 57: FAO vs SAU | 2004–2019 (9 pts) | 0.55 | −0.33 |
| Catch World: FAO capture vs SAU reported | 1960–2019 | 0.98 | 0.93 |
| Catch per SEA country: FAO vs SAU reported (EEZ sum) | 1960–2019 | 0.85–1.00 | 0.39–0.95 |

Reading: the two methods agree on the **long-run trend** of overexploitation but not on **year-to-year moves** (FAO's World series is partly interpolated), so trend tests are robust while lag correlations on differenced FAO data need caution. FAO/SAU catch ratios above 1 for Thailand (1.30) and the Philippines (1.77) are expected: FAO capture includes **inland** fisheries, SAU is marine only.

---

## 4. Data-integrity sources — is reported catch plausible?

These test whether the officially **reported** catch of each SEA country is believable, using numbers that are produced independently or are harder to fake.

| File | What it gives | Integrity check |
|---|---|---|
| `validation/economic/fisheries_gdp_capture.csv` | SDG 14.7.1 (sustainable fisheries value added, % of GDP), GDP (current USD), reported capture (t) | **Implied value per tonne** = (14.7.1 % × GDP) ÷ capture. A country far above peers may under-report tonnes; far below may over-report |
| `validation/comtrade/fish_hs03_mirror_trade.csv` | Fish exports (HS chapter 03) a SEA country reported vs imports from it reported by all its partners | **Mirror trade gap**. Partner imports should be ~0–10% higher (CIF includes freight/insurance; exports are FOB). Much larger gaps suggest under-reported exports |
| `validation/benford/sau_catch_leading_digits.csv` | Leading-digit counts of every taxon-level catch value in the SAU export, Reported vs Unreported | **Benford's law** — fabricated or heavily rounded numbers break the natural leading-digit pattern |
| `sau/catch_summary/` (already in repo) | Reported vs reconstructed catch | **Reconstructed ÷ reported ratio** — direct estimate of how much catch is missing from official statistics |

### `validation/economic/fisheries_gdp_capture.csv`

`iso3, country, year, sustainable_fisheries_pct_gdp, gdp_current_usd, capture_t`

- Rows: SEA5 and World for every SDG year (2011, 2013 … 2023); **all other countries for 2021 and 2023** (peer group, ~100 countries) where all three values exist.
- Sources: SDG 14.7.1 via OWID (same values as the UN SDG database, ISO3-coded); World Bank WDI `NY.GDP.MKTP.CD` and `ER.FSH.CAPT.MT` (vintage 2026-11-11).
- ⚠️ **Circularity:** FAO computes 14.7.1 by scaling fisheries value added by the share of stocks fished sustainably in the region. Do **not** correlate it with % overexploited stocks. For the integrity check, compare countries within the same FAO region (SEA5 are all in Areas 57/71).
- ⚠️ Capture includes inland fisheries; 14.7.1 is marine capture only.

### `validation/comtrade/fish_hs03_mirror_trade.csv`

`country, m49, year, reported_exports_usd, partner_imports_usd, n_partners` — SEA5, 2011–2023. Values are annual totals (all partners, all transport modes) in current USD. Source: UN Comtrade public API (preview endpoint).
- ⚠️ **Philippines 2014–2023 partner imports are blank**: the free Comtrade quota ran out during download (HTTP 403). Re-run `python src/download_data.py --only integrity` on another day to fill them.

### `validation/benford/sau_catch_leading_digits.csv`

`eez_id, country, reporting_status, d1 … d9` — counts of the first significant digit of `tonnes` for every row (taxon × fishing country × sector × gear × year) of the full SAU EEZ catch export with tonnes > 0. ~30,000–115,000 values per EEZ and status.
- Use **MAD** (mean absolute deviation from Benford proportions, Nigrini 2012) rather than chi-square: with this many values chi-square rejects even trivial deviations.
- ⚠️ SAU disaggregates national totals spatially and by taxon, which can itself shape the digits — interpret Reported vs Unreported **relative to each other**.

---

## Known data issues (handle in the cleaning notebook)

1. **World overexploited series has interpolated years.** Several stretches (e.g. 1974–1978, 1993–1995) change by a constant step each year, i.e. linear interpolation between FAO assessment years. This inflates autocorrelation; mention it, and consider testing on FAO assessment years only as a robustness check.
2. **FAO area series are biennial with gaps** (2004, 2006, 2008, 2009, 2011 … 2021). Unevenly spaced → do **not** use for ACF/CCF; Mann-Kendall only.
3. **FAO 2025 method change** is not in these files (they end in 2021), so no structural break to remove.
4. **Different end years:** OWID capture to 2024, SAU to 2018/2019, FAO stocks to 2021. Align on the common window before correlating.
5. **Country vs EEZ:** OWID values are per country; SAU values are per EEZ. Sum sub-EEZs to get country totals for SAU.
6. **Name mismatch:** OWID uses `Vietnam`, SAU uses `Viet Nam`.
7. **Fixed 2026-11-21:** the first version of `catch_summary/` labelled all catch in the 9 split EEZs (Malaysia, Indonesia, Thailand) as `foreign` because the fishing country was compared with the full EEZ name. Files were rebuilt; totals are unchanged, only the domestic/foreign split.
8. **SAU discard series start late** for some countries in OWID (Malaysia/Thailand 1961, Vietnam 1987) — missing ≠ zero.

---

## How the committed files were obtained

The cloud workspace used to build this repo could not reach OWID or SAU directly, so the files were fetched in a web browser using exactly the URLs and filtering in `src/download_data.py`, bundled with SHA-256 checksums, and verified after transfer (39/39 files matched; the 26 validation files likewise 26/26). Running `python src/download_data.py` re-creates them from source.

---

## References (for the report)

- Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data. "Share of fish stocks that are overexploited – FAO" [dataset]; "Biologically sustainable – FAO" [dataset]. FAO, "SDG Indicators" [original data]. https://ourworldindata.org/grapher/share-of-fish-stocks-overexploited
- Food and Agriculture Organization of the United Nations (FAO), via World Bank (2026) – processed by Our World in Data. "Capture fisheries production (metric tons)" and "Aquaculture production (metric tons)" [datasets]. World Development Indicators 129 [original data]. https://ourworldindata.org/grapher/capture-fishery-production
- Sea Around Us (2022) – with minor processing by Our World in Data. "Wild fish catch by gear type" and "Wild fish discards" [datasets]. Sea Around Us reconstructed marine fisheries catch [original data]. https://ourworldindata.org/grapher/fish-catch-gear-type
- Pauly, D., Zeller, D. & Palomares, M.L.D. (Eds.) (2020). *Sea Around Us Concepts, Design and Data*. www.seaaroundus.org — EEZ catch, Marine Trophic Index and stock-status data, dataset version v50-1, accessed 18 Nov 2026.
- Kleisner, K., Zeller, D., Froese, R. & Pauly, D. (2013). Using global catch data for inferences on the world's marine fisheries. *Fish and Fisheries*, 14(3), 293–311. (stock-status method)
- Pauly, D. et al. (1998). Fishing down marine food webs. *Science*, 279(5352), 860–863. (mean trophic level indicator)
- United Nations Statistics Division. *Global SDG Indicators Database*, series ER_H2O_FWTL (indicator 14.4.1). Source: FAO (2025), *Review of the state of world marine fishery resources – 2025*, FAO Fisheries and Aquaculture Technical Paper No. 721. https://unstats.un.org/sdgs/dataportal, accessed 18 Nov 2026.
- World Bank. *World Development Indicators*, ER.FSH.CAPT.MT — Capture fisheries production (metric tons), last updated 11 Nov 2026. https://data.worldbank.org/indicator/ER.FSH.CAPT.MT
- Sea Around Us regional data (Global, FAO Areas 57 & 71, LMEs 34–38), v50-1, accessed 18 Nov 2026 — cite as Pauly, Zeller & Palomares (2020) above.
- FAO (2025) – processed by Our World in Data. "Sustainable fisheries as a proportion of GDP" (SDG 14.7.1, series EN_SCP_FSHGDP) [dataset]. https://ourworldindata.org/grapher/sustainable-fisheries-as-a-proportion-of-gdp, accessed 22 Nov 2026.
- World Bank. *World Development Indicators*, NY.GDP.MKTP.CD — GDP (current US$), last updated 11 Nov 2026.
- United Nations. *UN Comtrade Database*, HS chapter 03 (fish, crustaceans, molluscs), annual, 2011–2023. https://comtradeplus.un.org, accessed 22 Nov 2026.
- Nigrini, M. J. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*. Wiley. (MAD conformity thresholds)
