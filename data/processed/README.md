# data/processed — checked and cleaned data (the only input of steps 1–4)

Created by [`0-data-quality/clean_data.py`](../../0-data-quality/clean_data.py), step 0 of the pipeline, from `data/raw/`. Do not edit by hand. Each check (C1–C4), cleaning rule (R1–R6) and count is listed in `cleaning_log.csv`.

## `clean/` — checked copy of every raw file

Same folder layout, file names and columns as `data/raw/`, so the shared loaders in [`src/loaders.py`](../../src/loaders.py) read it unchanged. Lines are copied byte for byte except where a rule applies:
- R1: "Viet Nam" → "Vietnam" in 5 files.
- R5: the partial year 2026 is dropped from the world SST series.
- R4: 6 GDAT dams with completion years after 2025 (planned) are removed.

`_quality_checks_passed.json` is written only when every check passes. It holds the SHA-256 of every raw file; steps 1–4 compare it with `data/raw/` before running.

## `global_annual.csv` — World, 1950–2025 (76 rows)

| Column | Unit | Source |
|---|---|---|
| `year` | | |
| `fao_pct_overexploited` | % of assessed stocks | FAO via OWID (1974–2021) |
| `fao_interpolated_flag` | bool | year lies on a straight line between its neighbours (likely interpolated) |
| `sdg_pct_sustainable` | % | UN SDG 14.4.1 (≤ 2021; new-method years excluded) |
| `sau_pct_overexploited` | % of stocks collapsed + over-exploited | Sea Around Us (1950–2019) |
| `fao_capture_t` | t | FAO capture production via OWID (1960–2024) |
| `aquaculture_t` | t | FAO aquaculture via OWID |
| `sau_catch_t` | t | Sea Around Us reconstructed catch, landings + discards (1950–2019) |
| `effort_nominal_kwdays`, `effort_effective_kwdays` | kW × days at sea | IMAS Global Fishing Effort (1950–2017) |
| `cpue_kg_per_kwday` | kg per kW-day | `sau_catch_t` × 1000 ÷ nominal effort |
| `sst_anomaly_c` | °C vs 1961–1990 | HadSST via OWID |
| `ocean_heat_0_700m_1e22j` | 10²² J | NOAA via OWID (1955–2025) |

## `sea_country_annual.csv` — SEA aggregate + 5 countries, 1950–2025 (456 rows)

| Column | Unit | Notes |
|---|---|---|
| `year`, `country` | | `SEA`, Malaysia, Indonesia, Thailand, Vietnam, Philippines |
| `pct_overexploited` | % | mean of the country's EEZs; SEA = mean of the 5 countries |
| `mean_trophic_level` | level | catch-weighted over EEZs |
| `sau_catch_t` | t | Sea Around Us reconstructed catch |
| `unreported_share_pct`, `foreign_share_pct`, `industrial_share_pct` | % of catch | catch-weighted over EEZs |
| `fao_capture_t` | t | FAO via OWID (SEA = sum of 5) |
| `effort_nominal_kwdays` | kW × days at sea | all fleets inside the EEZ (IMAS) |
| `effort_source_break_flag` | bool | \|Δlog effort\| > 0.5: change of source, not of fishing |
| `cpue_kg_per_kwday` | kg per kW-day | catch ÷ effort |
| `sst_mean_c`, `sst_anomaly_c`, `sst_hot_months` | °C, °C vs 1971–2000, months | NOAA ERSST v5, EEZ-area weighted |
| `sdg1471_fisheries_pct_gdp` | % of GDP | SDG 14.7.1, biennial, not interpolated |
| `mirror_gap_pct` | % | partner-reported imports ÷ own exports − 1 (UN Comtrade, 2011–2023) |

## `sea_eez_annual.csv` — 11 Sea Around Us EEZs, 1950–2019 (770 rows)

`eez_id`, `eez_name`, `country`, `year`, `pct_overexploited`, the 5 stock-status shares (`pct_collapsed`, `pct_over_exploited`, `pct_exploited`, `pct_developing`, `pct_rebuilding`), `mean_trophic_level`, `catch_t`, `unreported_share_pct`, `foreign_share_pct`, `industrial_share_pct`.
