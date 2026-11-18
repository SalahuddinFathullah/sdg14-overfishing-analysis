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
| `fleet` | `domestic` = fished by the EEZ's own country; `foreign` = fished by other countries' fleets |
| `tonnes` | Catch (t), summed |
| `landed_value` | Landed value (real 2010 USD), summed |

Total catch per year = `df.groupby("year").tonnes.sum()`. Useful extra angles: share of unreported catch, foreign fishing pressure, discard share.

---

## Known data issues (handle in the cleaning notebook)

1. **World overexploited series has interpolated years.** Several stretches (e.g. 1974–1978, 1993–1995) change by a constant step each year, i.e. linear interpolation between FAO assessment years. This inflates autocorrelation; mention it, and consider testing on FAO assessment years only as a robustness check.
2. **FAO area series are biennial with gaps** (2004, 2006, 2008, 2009, 2011 … 2021). Unevenly spaced → do **not** use for ACF/CCF; Mann-Kendall only.
3. **FAO 2025 method change** is not in these files (they end in 2021), so no structural break to remove.
4. **Different end years:** OWID capture to 2024, SAU to 2018/2019, FAO stocks to 2021. Align on the common window before correlating.
5. **Country vs EEZ:** OWID values are per country; SAU values are per EEZ. Sum sub-EEZs to get country totals for SAU.
6. **Name mismatch:** OWID uses `Vietnam`, SAU uses `Viet Nam`.
7. **SAU discard series start late** for some countries in OWID (Malaysia/Thailand 1961, Vietnam 1987) — missing ≠ zero.

---

## How the committed files were obtained

The cloud workspace used to build this repo could not reach OWID or SAU directly, so the files were fetched in a web browser using exactly the URLs and filtering in `src/download_data.py`, bundled with SHA-256 checksums, and verified after transfer (39/39 files matched). Running `python src/download_data.py` re-creates them from source.

---

## References (for the report)

- Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data. "Share of fish stocks that are overexploited – FAO" [dataset]; "Biologically sustainable – FAO" [dataset]. FAO, "SDG Indicators" [original data]. https://ourworldindata.org/grapher/share-of-fish-stocks-overexploited
- Food and Agriculture Organization of the United Nations (FAO), via World Bank (2026) – processed by Our World in Data. "Capture fisheries production (metric tons)" and "Aquaculture production (metric tons)" [datasets]. World Development Indicators 129 [original data]. https://ourworldindata.org/grapher/capture-fishery-production
- Sea Around Us (2022) – with minor processing by Our World in Data. "Wild fish catch by gear type" and "Wild fish discards" [datasets]. Sea Around Us reconstructed marine fisheries catch [original data]. https://ourworldindata.org/grapher/fish-catch-gear-type
- Pauly, D., Zeller, D. & Palomares, M.L.D. (Eds.) (2020). *Sea Around Us Concepts, Design and Data*. www.seaaroundus.org — EEZ catch, Marine Trophic Index and stock-status data, dataset version v50-1, accessed 18 Nov 2026.
- Kleisner, K., Zeller, D., Froese, R. & Pauly, D. (2013). Using global catch data for inferences on the world's marine fisheries. *Fish and Fisheries*, 14(3), 293–311. (stock-status method)
- Pauly, D. et al. (1998). Fishing down marine food webs. *Science*, 279(5352), 860–863. (mean trophic level indicator)
