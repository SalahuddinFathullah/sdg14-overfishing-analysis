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
| **H3** warming (iteration 3) | SST anomaly, ocean heat content, EEZ SST | `owid/sea-surface-temperature-anomaly.csv`, `owid/ocean-heat-content-upper.csv`, `validation/climate/*.csv` |
| **H4** shipping & structures (iteration 4) | Port traffic, seaborne trade, liner connectivity, dams, aquaculture, platforms, offshore wind | `validation/human_use/*.csv`, `owid/aquaculture-farmed-fish-production.csv` |
| **H7** pollution (iteration 7) | Fertiliser, pesticides, chlorophyll-a, plastic, oil spills, seawater pH; nutrient loading per SEA sea (H7f, with `validation/sau_regions/` LMEs 34–38), plastic to the ocean, beach litter (descriptive) | the 11 `owid/` files listed in §8 |
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

### `validation/benford/sau_catch_leading_digits.csv`

`eez_id, country, reporting_status, d1 … d9` — counts of the first significant digit of `tonnes` for every row (taxon × fishing country × sector × gear × year) of the full SAU EEZ catch export with tonnes > 0. ~30,000–115,000 values per EEZ and status.
- Use **MAD** (mean absolute deviation from Benford proportions, Nigrini 2012) rather than chi-square: with this many values chi-square rejects even trivial deviations.
- ⚠️ SAU disaggregates national totals spatially and by taxon, which can itself shape the digits — interpret Reported vs Unreported **relative to each other**.

---

## 5. Fishing effort — `validation/effort/` (added for iteration 2 of the analysis)

Source: **Rousseau et al., Global Fishing Effort database** (IMAS, University of Tasmania), 1950–2017, CC BY 4.0, https://doi.org/10.25959/MNGY-0Q43. Effort is reconstructed from national fleet statistics (vessel numbers, engine power, days at sea). Re-create with `python src/download_data.py --only effort`, which streams two of the database's aggregated CSVs and sums them by year.

| File | Rows | Columns |
|---|---|---|
| `world_effort.csv` | 1950–2017 | `year`, `nom_kwdays`, `eff_kwdays`, `nom_kwdays_industrial`, `vessels`, `power_kw` |
| `sea_eez_effort.csv` | 5 EEZs × 1950–2017 | `eez` (Indonesia, Malaysia, Philippines, Thailand, Viet Nam), then the same columns |

- `nom_kwdays`: **nominal effort**, active vessels × engine power (kW) × days at sea. This is the main pressure variable.
- `eff_kwdays`: **effective effort**, nominal effort plus the database's technological-creep factor. It grows several hundred-fold over 1950–2017, so use it for sensitivity checks only.
- `nom_kwdays_industrial`: nominal effort of the industrial sector (sector `I`). The remainder is artisanal (powered and unpowered).
- `vessels`, `power_kw`: active vessels and their total engine power.
- `world_effort.csv` sums all fishing countries (`TotalEffortby_FGroup_FishingCountry_Sector.csv`), including the high seas.
- `sea_eez_effort.csv` is the effort by **all fleets inside each country's EEZ** (`TotalEffortby_FGroup_EEZ_Sector.csv`). That matches the Sea Around Us EEZ catch data. The database gives one EEZ per country, so Malaysia's 4 and Indonesia's 3 Sea Around Us EEZs are combined here.

Transfer check: the two files were aggregated in the browser and transferred with SHA-256 checksums. Both matched (`world_effort.csv` `1b79710a…`, `sea_eez_effort.csv` `1f3a24d8…`).

---

## 6. Ocean warming — `owid/` + `validation/climate/` (added for iteration 3 of the analysis)

| File | Source | Coverage | Columns |
|---|---|---|---|
| `owid/sea-surface-temperature-anomaly.csv` | Met Office HadSST via OWID, World rows | 1850–2026 (2026 partial; analysis uses ≤ 2025) | `Average` = anomaly (°C) vs 1961–1990, plus 95% bounds |
| `owid/ocean-heat-content-upper.csv` | NOAA / IAP / MRI-JMA / CSIRO via OWID, World rows | 1955–2025 (NOAA complete) | upper-700 m ocean heat content anomaly (10²² J); analysis uses `NOAA` |
| `validation/climate/eez_ersst_cell_weights.csv` | built from IMAS `CellstoLME_EEZ.csv` | 5 SEA EEZs | `eez`, `lat`, `lon` (centre of a 2° ERSST cell), `area_km2` of the EEZ inside it |
| `validation/climate/sea_eez_sst_ersst.csv` | NOAA ERSST v5 monthly SST (PSL OPeNDAP) | SEA + 5 EEZs, 1950–2025 | `sst_mean_c` (annual mean, °C), `sst_anom_c` (vs 1971–2000 mean), `hot_months` (months above that calendar month's 1971–2000 90th-percentile anomaly) |

The EEZ SST is an **area-weighted** mean. Every 0.5° cell of a country's EEZ (IMAS cell list) is assigned to the 2° ERSST cell that contains it, and its area becomes that cell's weight. Land cells are skipped. "SEA" pools all 5 EEZs. Re-create with `python src/download_data.py --only climate`.

Transfer check (SHA-256, all matched): `sea-surface-temperature-anomaly.csv` `3bfbc754…`, `ocean-heat-content-upper.csv` `f9c6d6c3…`, `eez_ersst_cell_weights.csv` `8bd4d24c…`, `sea_eez_sst_ersst.csv` `3234e8ba…`.

---

## 7. Human use of the sea — `validation/human_use/` (added for iteration 4 of the analysis)

| File | Source | Coverage | Columns |
|---|---|---|---|
| `wb_shipping.csv` | World Bank WDI API: current release + WDI Database Archives (source 57), version Dec 2020 | Container port traffic `IS.SHP.GOOD.TU` (TEU; UNCTAD data): archive 2000–2019, current 2010–2024, World + SEA5. Liner shipping connectivity `IS.SHP.GCNW.XQ` (index): archive 2004–2019, current 2006–2021, SEA5 only | `source` (`WDI_current` / `WDI_archive_202012`), `iso3`, `year`, `indicator`, `value` |
| `unctad_seaborne_trade.csv` | UNCTADstat `US.SeaborneTrade` (version 10003, updated 17 Mar 2026), datamart API | 2000–2024, World + SEA5 (Indonesia 2000–2002 under its old code 960) | `iso3`, `year`, `flow` (`loaded` / `discharged`), `thousand_tonnes` |
| `gdat_dams_by_year.csv` | Global Dam Tracker v1, Zenodo record 7616852 (35,140 dams), aggregated from `GDAT_v1_dams.dbf` | World + SEA5 | `region`, `year` (completion year `Year_Fin`; blank if undated), `year_precision` (`exact` / `decade` = 3-digit year such as "195" / `undated`), `n_dams`, `n_with_volume`, `volume_mcm` (sum of reported reservoir volume `Volume_Rep`, million m³) |
| `deepowt_offshore_wind.csv` | DeepOWT, Zenodo record 5933967, `DeepOWT.geojson` (9,941 locations) | 2016Q3–2021Q2, World and Vietnam (102–108°E, 8–21.5°N; the only SEA5 cluster) | `region`, `year`, `quarter`, `n_class1` (under construction), `n_class2` (turbine), `n_class3` (substation) |
| `oogp_platforms.csv` | Offshore Oil and Gas Platforms v1.0.0, Zenodo record 18350974, `OOGPs_all_v1.0.0.gpkg` (9,334 records) | 2017–2023, all 6 basins (`ALL6`) + MYS, THA, VNM, IDN, KHM (Gulf of Thailand / South China Sea) | `region`, `year`, `n_platforms_detected` (records whose `Year_label` contains the year), `n_installed`, `n_removed` (by year of `Installation_date` / `Removal_date`) |

Aquaculture uses the existing `owid/aquaculture-farmed-fish-production.csv` (World + SEA5, 1960–2024).

Artificial islands and military outposts have no downloadable time series. Iteration 4 uses only the dated event, China's Spratly reclamation of >3,200 acres between Dec 2013 and 2016 (US DoD 2016; CSIS AMTI), with outposts and airstrips built on the same reefs through 2018. Vietnam's reclamation (~120 acres by 2016, 540 by Dec 2022, ~3,300 by Mar 2025; AMTI) falls mostly after the fisheries data end in 2019.

Re-create with `python src/download_data.py --only human_use`. Transfer check (SHA-256, all matched): `wb_shipping.csv` `25484fc0…`, `unctad_seaborne_trade.csv` `d98ed488…`, `gdat_dams_by_year.csv` `6899d1b8…`, `deepowt_offshore_wind.csv` `aa6edf46…`, `oogp_platforms.csv` `563032c5…`.

---

## 8. Sea and water pollution — `owid/` (added for iteration 7 of the analysis)

All from Our World in Data (CC BY 4.0), columns `Entity, Code, Year, <value>` unless noted. Filtered to World + SEA5 except where stated.

| File | Value | Unit | Original source | Coverage |
|---|---|---|---|---|
| `nitrogen-fertilizer-application-per-hectare-of-cropland.csv` | Use of nutrient nitrogen per area of cropland | kg N / ha | FAO (FAOSTAT inputs) | World + SEA5, 1961–2024 |
| `pesticide-use-per-hectare-of-cropland.csv` | Use of pesticides per area of cropland | kg / ha | FAO | World + SEA5, 1990–2024 (Thailand drops ~90% in 2013: reporting break) |
| `chlorophyll-a-deviation-from-the-global-average.csv` | Chlorophyll-a deviations, remote sensing | % | UNEP, SDG 14.1.1a (coastal eutrophication) | World + SEA5, 2005–2022 |
| `global-plastics-production.csv` | Annual plastic production | t / year | Geyer, Jambeck & Law (2017) | World, 1950–2019 (1974 missing; 2016–19 extrapolated by the source) |
| `number-oil-spills.csv` | Large (> 700 t) and medium (7–700 t) tanker spills | count | ITOPF | World, 1970–2025 |
| `quantity-oil-spills.csv` | Quantity of oil spilled | t | ITOPF | World, 1970–2025 |
| `seawater-ph.csv` | Surface seawater pH; columns `Entity, Day, Annual average, Monthly average` (whole file) | pH | Hawaii Ocean Time-series, Station ALOHA | Hawaii, Oct 1988 – Dec 2024, one row per cruise |
| `coastal-eutrophication.csv` | Nitrogen, Phosphorus loading; columns `Entity, Year, Nitrogen, Phosphorus` | t / year | UNEP / Global NEWS 2 model (SDG 14.1.1 support) | 5 SEA large marine ecosystems (Bay of Bengal, Gulf of Thailand, South China Sea, Sulu-Celebes Sea, Indonesian Sea), 2000–2015; nitrogen constant 2010–2015 in all five |
| `plastic-waste-emitted-to-the-ocean.csv` | Plastic waste emitted to the ocean | t | Meijer et al. (2021) | World + SEA5, 2019 only |
| `mismanaged-plastic-waste-per-capita.csv` | Mismanaged plastic waste per capita | kg / person / year | Meijer et al. (2021) | World + SEA5, 2019 only |
| `beach-litter.csv` | Beach litter per km² | items | UNEP, SDG 14.1.1b | SEA5 2016–18 and 2021–22; World 2015–2022 |

How they are used (`src/loaders.py`): fertiliser and pesticides in logs (growth rates); acidity as the hydrogen-ion concentration 10^(−pH) from the calendar-year mean of the deseasonalised running annual average (1990–2024); "SEA" = simple mean of the 5 countries. Nutrient loading is too short as one series, so `sau_lme_panel()` matches it to Sea Around Us for the same 5 large marine ecosystems (`validation/sau_regions/`, LMEs 34–38) and H7f tests the 5-sea panel; plastic emissions and beach litter are too short or sparse to test and are shown descriptively (figure I7.03).

Re-create with `python src/download_data.py --only pollution`. Files were fetched in the browser and passed through chat; transfer check (SHA-256 of each file, all matched): fertiliser `e31f97de…`, pesticides `fb83ad8c…`, chlorophyll-a `965ae973…`, plastic production `68e2dd59…`, oil spills number `4e98a53c…`, quantity `99a30b02…`, seawater pH `d9bc4265…`, coastal eutrophication `e7be389c…`, plastic to ocean `43d0acea…`, mismanaged plastic `7d614139…`, beach litter `d9e9624a…`.

## 9. Causes that can be confirmed — `causes/` (added for iteration 8 of the analysis)

| File | Content | Unit | Source | Coverage |
|------|---------|------|--------|----------|
| `ram_v4.66_timeseries.csv` | `stockid, year, BdivBmsypref, UdivUmsypref, TBbest, TCbest`: biomass relative to Bmsy (preferred series), fishing pressure relative to Umsy (exploitation rate or F), total biomass, total catch | ratios; t | RAM Legacy Stock Assessment Database v4.66 (2024), Zenodo 10.5281/zenodo.14043031 | 1,438 stocks; 432 with both ratios, 1874–2023 |
| `ram_v4.66_stocks.csv` | Stock metadata: names, RAM region, primary country and its ISO3 code, FAO area, taxonomic group, management authority | — | same | 1,514 stocks |
| `iuu_fishing_index_2019-2025_indicator_scores.csv` | 40 indicators (`Indicator ID, Indicator name, Country, Region, Ocean, Resp, Type, Year, Score`); Type = Vulnerability / Prevalence / Response; Resp = Flag / Coastal / Port / General | score 1 (best) – 5 (worst) | IUU Fishing Risk Index (Poseidon Aquatic Resource Management & Global Initiative against Transnational Organized Crime) | 152 coastal states, 2019, 2021, 2023, 2025 |
| `sumaila2019_fisheries_subsidies_2018.csv` | Subsidies by country, class (A beneficial, B capacity-enhancing, C ambiguous) and type; `Data Type` = Reported / Modeled / not found | constant 2018 USD | Sumaila et al. (2019) *Data in Brief* 27:104706 (supplement `Sumaila_dataset.csv`) | 152 countries, 2018 |
| `epi2026_regional_marine_trophic_index_raw.csv` | RMS: Regional Marine Trophic Index, a rolling trend in the mean trophic level of each country's EEZ catch (< 0 = fishing down the food web) | trophic level per year | Yale EPI 2026 raw data (from Sea Around Us) | 220 countries (164 with values), 1989–2019 |
| `owid_capture-fishery-production_all_countries.csv` | Capture fisheries production, all countries | t | FAO via OWID | 1960–2024 |
| `owid_gdp-per-capita-worldbank_all_countries.csv` | GDP per capita, PPP | constant international $ | World Bank via OWID | 1990–2025 |

How they are used (`src/loaders.py`): `ram_panel()` keeps stock-years with both ratios > 0 and adds the stock's ISO3; `iuu_index()` averages the indicator scores by Type per country and year; `subsidies_2018()` sums by category; `epi_rmti()` reshapes RMS to long; country names are matched to ISO3 through OWID's own names plus `I8_NAME_FIX`.

Re-create with `python src/download_data.py --only causes` (needs `pyreadr`). These sites refuse the analysis shell's proxy, so the committed files were fetched in the browser on 2026-11-27: the IUU, Sumaila (Elsevier supplement), EPI and OWID files whole; from the 112 MB RAM zip only the R data file was read, by HTTP range requests, inflated in the browser and checked against the zip's CRC-32 (`04a672e3`), then the two CSVs were extracted with `pyreadr`. SHA-256 of every committed file is in `CAUSES_SHA256` in `src/download_data.py`.

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
9. **Effort source breaks:** single-year jumps of more than 65% in EEZ effort mark changes in the national statistics, not in fishing. They occur in Thailand 2002 (+75%) and 2011 (−52%), and in Malaysia 1965 (+67%). `3-hypothesis-tests` drops these years from the effort growth series. Thailand's catch-per-unit-effort dip in 2002–2010 is an artefact of the same breaks.
10. **Effort ends in 2017**, two years before Sea Around Us (2019) and four before FAO stock status (2021). Iteration-2 tests use the common window.
11. **Warming data:**
    - **Two products:** the global SST (HadSST) and the SEA SST (ERSST v5) come from different products. They are never compared directly; each is only correlated with the fisheries series at its own scale.
    - **Coarse grid:** ERSST's 2° grid is coarse, so coastal EEZ cells are partly land, and narrow seas (e.g. the Malacca Strait) are smoothed.
    - **Short ocean-heat series:** NOAA ocean heat content starts in 1955.
12. **Human-use data (iteration 4):**
    - **Container series break in 2010:** the current WDI series starts in 2010 and differs slightly from the Dec-2020 archive (current/archive ratio 0.92–1.02 over 2010–2019). `3-hypothesis-tests` rescales archive years 2000–2009 by the median ratio before joining.
    - **GDAT dates:** 11,866 of 35,140 dams (34%) have no completion year, including 139 of 248 in Indonesia, 139 of 181 in Thailand and 112 of 198 in Vietnam. 25 world dams have only a decade ("195" = 1950s). Completions thin out after ~2005 because the database lags construction, and 2020–2040 entries are planned or under construction.
    - **GDAT volume outlier:** 1921 has a reported volume of 11,969,027 million m³, about 1,000× any other year, so it is a data-entry error. The analysis uses dam counts, not volumes.
    - **Seaborne trade:** the world row is goods loaded (= discharged at world level). Country rows are summed over both flows to give total tonnes handled.
    - **Offshore structures start late:** platforms (2017) and wind (2016) overlap the fisheries data (to 2019) by only 3–4 years.
13. **Iteration-8 data:**
    - **RAM coverage:** assessed stocks are mostly from high-income countries; only 3 stocks (two tunas, blue shark) touch SEA waters. B and U come from the same assessment model.
    - **RAM "primary country":** shared or high-seas stocks are assigned to one country (e.g. a NAFO redfish stock to Russia), so country aggregates are approximate; only countries with ≥ 3 stocks are used.
    - **IUU index Timor-Leste:** spelled both "Timor Leste" and "Timor-Leste" across years; both map to TLS.
    - **Subsidies are one year (2018)**; 'Modeled' rows are estimates where a subsidy was found but no amount reported.

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
- Rousseau, Y., Blanchard, J., Novaglio, C., Pinnell, K., Tittensor, D., Watson, R. & Ye, Y. (2022). *Global Fishing Effort* [dataset]. Institute for Marine and Antarctic Studies, University of Tasmania. https://doi.org/10.25959/MNGY-0Q43, accessed 22 Nov 2026. Method: Rousseau et al. (2024), A database of mapped global fishing activity 1950–2017, *Scientific Data* 11, 48.
- Kennedy, J. J. et al. (2019). An ensemble data set of sea surface temperature change from 1850: the Met Office Hadley Centre HadSST.4.0.0.0 data set. *JGR Atmospheres*, 124. Via Our World in Data, "Sea surface temperature anomaly" [dataset], accessed 22 Nov 2026.
- NOAA NCEI / IAP / MRI-JMA / CSIRO ocean heat content (0–700 m), via Our World in Data, "Ocean heat content in the upper 700 m" [dataset], accessed 22 Nov 2026.
- Huang, B. et al. (2017). Extended Reconstructed Sea Surface Temperature, Version 5 (ERSSTv5). *Journal of Climate*, 30, 8179–8205. Data: NOAA PSL, https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html, accessed 22 Nov 2026.
- World Bank. *World Development Indicators*, IS.SHP.GOOD.TU (container port traffic, TEU) and IS.SHP.GCNW.XQ (liner shipping connectivity index), current release (updated 11 Nov 2026) and WDI Database Archives, December 2020 version. Original data: UNCTAD. https://data.worldbank.org, accessed 22 Nov 2026.
- UNCTAD. *UNCTADstat*, "Seaborne trade by cargo type, annual" (US.SeaborneTrade), updated 17 Mar 2026. https://unctadstat.unctad.org/datacentre/dataviewer/US.SeaborneTrade, accessed 22 Nov 2026.
- Zhang, A. T. & Gu, V. X. (2023). Global Dam Tracker: a database of more than 35,000 dams with location, catchment, and attribute information. *Scientific Data*, 10, 111. Data: https://doi.org/10.5281/zenodo.7616852, accessed 22 Nov 2026.
- Hoeser, T., Feuerstein, S. & Kuenzer, C. (2022). DeepOWT: a global offshore wind turbine data set derived with deep learning from Sentinel-1 data. *Earth System Science Data*, 14, 4251–4270. Data: https://doi.org/10.5281/zenodo.5933967, accessed 22 Nov 2026.
- Si, L., Zhou, S., Irakulis-Loitxate, I., Roger, J. & Guanter, L. (2026). Building a dataset of offshore oil and gas extraction platforms from satellite data (2017–2023). *Earth System Science Data*, 18, 6545–6564. Data: https://doi.org/10.5281/zenodo.18350974, accessed 22 Nov 2026.
- CSIS Asia Maritime Transparency Initiative (AMTI). Island Tracker and reports "Vietnam's Major Spratly Expansion" (Dec 2022) and "Vietnam Ramps Up Spratly Island Dredging" (Nov 2023). https://amti.csis.org; US Department of Defense (2016), *Annual Report to Congress: Military and Security Developments Involving the People's Republic of China* (>3,200 acres reclaimed).
- FAO (2026) – processed by Our World in Data. "Use of nutrient nitrogen per area of cropland" and "Use of pesticides per area of cropland" [datasets]. FAOSTAT [original data]. https://ourworldindata.org/grapher/nitrogen-fertilizer-application-per-hectare-of-cropland, accessed 26 Nov 2026.
- UNEP (2025) – processed by Our World in Data. SDG 14.1.1a "Chlorophyll-a deviations, remote sensing" and 14.1.1b "Beach litter per square kilometer" [datasets]. UN SDG Indicators Database [original data], accessed 26 Nov 2026.
- Geyer, R., Jambeck, J. R. & Law, K. L. (2017). Production, use, and fate of all plastics ever made. *Science Advances*, 3(7), e1700782. Via Our World in Data, "Global plastics production" [dataset], accessed 26 Nov 2026.
- ITOPF (2026). *Oil Tanker Spill Statistics*. Via Our World in Data, "Number of oil spills" and "Quantity of oil spilled" [datasets], accessed 26 Nov 2026.
- Hawaii Ocean Time-series (HOT), Station ALOHA, University of Hawai'i. Via Our World in Data, "Ocean acidification: mean seawater pH" [dataset], accessed 26 Nov 2026.
- Meijer, L. J. J., van Emmerik, T., van der Ent, R., Schmidt, C. & Lebreton, L. (2021). More than 1000 rivers account for 80% of global riverine plastic emissions into the ocean. *Science Advances*, 7(18), eaaz5803. Via Our World in Data, accessed 26 Nov 2026.
- Nigrini, M. J. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*. Wiley. (MAD conformity thresholds)
