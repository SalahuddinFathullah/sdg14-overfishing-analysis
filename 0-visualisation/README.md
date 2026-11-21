# 0-visualisation — exploratory figures

All figures are produced by [`visualise.py`](visualise.py) from the raw data in [`data/raw/`](../data/raw/). Re-create them with:

```bash
pip install -r requirements.txt
python 0-visualisation/visualise.py                    # all 28 figures
python 0-visualisation/visualise.py --only global      # one section: global | sea_owid | sea_sau | validation
```

PNG, 200 dpi, saved under `figures/<section>/`. Several datasets are shown with more than one chart type, because each type answers a different question (trend, composition, distribution, relationship, agreement).

---

## 1 · Global (Stage 1 — H1a, H1b)

| # | Figure | Chart type | What it shows |
|---|---|---|---|
| 1.01 | ![](figures/1_global/1.01_world_overexploited_trend.png) | Line + fitted segments | % of world stocks overexploited 1974–2021 with separate linear fits before and after 2015 (SDG adoption) — **H1a** |
| 1.02 | ![](figures/1_global/1.02_world_stock_composition.png) | 100% stacked area | Same data as a composition: overexploited vs sustainable (SDG 14.4.1) |
| 1.03 | ![](figures/1_global/1.03_world_capture_vs_aquaculture.png) | Line | Wild catch plateaued in the 1990s while aquaculture kept growing |
| 1.04 | ![](figures/1_global/1.04_capture_vs_overexploited_connected_scatter.png) | Connected scatter | Catch vs % overexploited, coloured by year — the relationship behind **H1b** |
| 1.05 | ![](figures/1_global/1.05_acf_levels_vs_differences.png) | ACF stem plots | Autocorrelation of levels vs first differences — why correlation must use differenced series |
| 1.06 | ![](figures/1_global/1.06_world_overexploited_annual_changes.png) | Bar + histogram | Year-to-year changes; identical steps expose interpolated years in the FAO series |
| 1.07 | ![](figures/1_global/1.07_fao_areas_dumbbell_2004_2021.png) | Dumbbell | 2004 → 2021 change for every FAO fishing area; SEA areas 71 & 57 highlighted |

## 2 · Southeast Asia — country data (OWID)

| # | Figure | Chart type | What it shows |
|---|---|---|---|
| 2.01 | ![](figures/2_sea_owid/2.01_sea_capture_by_country.png) | Line | Wild capture per country, 1960–2024 |
| 2.02 | ![](figures/2_sea_owid/2.02_sea_capture_stacked.png) | Stacked area | Same data as the regional total and each country's share |
| 2.03 | ![](figures/2_sea_owid/2.03_sea_capture_vs_aquaculture.png) | Small multiples | Wild capture vs aquaculture per country |
| 2.04 | ![](figures/2_sea_owid/2.04_sea_gear_share_by_decade.png) | 100% stacked bars | Share of catch by fishing gear per decade (rise of bottom trawling) |
| 2.05 | ![](figures/2_sea_owid/2.05_sea_discard_rate.png) | Line | Discards as % of catch, SEA5 vs World |

## 3 · Southeast Asia — Sea Around Us EEZ data (Stage 2 — H2a, H2b, H2c)

| # | Figure | Chart type | What it shows |
|---|---|---|---|
| 3.01 | ![](figures/3_sea_sau/3.01_stock_status_by_eez.png) | Small multiples, stacked area | Stock status (collapsed → rebuilding) in all 11 SEA EEZs — **H2a** |
| 3.02 | ![](figures/3_sea_sau/3.02_overexploited_heatmap_eez_year.png) | Heatmap | % collapsed + over-exploited, EEZ × year — same data at a glance |
| 3.03 | ![](figures/3_sea_sau/3.03_overexploited_country_vs_world.png) | Line | Country averages vs the world — preview of **H2c** |
| 3.04 | ![](figures/3_sea_sau/3.04_mean_trophic_level_by_eez.png) | Small multiples, line | Mean trophic level of the catch per EEZ ("fishing down the food web") |
| 3.05 | ![](figures/3_sea_sau/3.05_fishing_down_catch_vs_mtl.png) | Connected scatter | Catch vs mean trophic level per country — preview of **H2b** |
| 3.06 | ![](figures/3_sea_sau/3.06_mtl_change_boxplot_by_country.png) | Box + strip plot | Distribution of yearly MTL changes by country, with a Kruskal-Wallis preview |
| 3.07 | ![](figures/3_sea_sau/3.07_mean_max_length_by_country.png) | Line | Average maximum size of the species caught |
| 3.08 | ![](figures/3_sea_sau/3.08_sea_catch_by_sector.png) | Stacked area | Total SEA5 catch by sector (industrial, artisanal, …) |
| 3.09 | ![](figures/3_sea_sau/3.09_unreported_and_foreign_share.png) | Two-panel line | Share of catch that is unreported, and share taken by foreign fleets |
| 3.10 | ![](figures/3_sea_sau/3.10_indicator_correlation_matrices.png) | Correlation heatmaps | Spearman ρ between indicators — levels vs year-to-year differences |

## 4 · Validation against alternative sources

| # | Figure | Chart type | What it shows |
|---|---|---|---|
| 4.01 | ![](figures/4_validation/4.01_owid_vs_un_sdg.png) | Line + markers | OWID vs official UN SDG database — identical 1974–2021 |
| 4.02 | ![](figures/4_validation/4.02_fao_vs_sau_world_overexploited.png) | Line | FAO assessments vs Sea Around Us catch-based status, world |
| 4.03 | ![](figures/4_validation/4.03_bland_altman_fao_vs_sau.png) | Bland-Altman | Bias and limits of agreement for the same pair, with Wilcoxon test |
| 4.04 | ![](figures/4_validation/4.04_fao_vs_sau_areas_71_57.png) | Line | FAO vs Sea Around Us for the two SEA fishing areas |
| 4.05 | ![](figures/4_validation/4.05_catch_fao_vs_sau_scatter.png) | Scatter + 1:1 line | FAO capture vs Sea Around Us reported catch, with Lin's CCC |
| 4.06 | ![](figures/4_validation/4.06_eez_vs_lme_overexploited.png) | Small multiples | Same indicator for national zones (EEZs) vs Large Marine Ecosystems |

---

### Notes for the report

- **Previews, not final tests.** Statistics printed on figures (Spearman ρ, Kruskal-Wallis, Wilcoxon, Lin's CCC) are exploratory; the formal tests with assumption checks go in the analysis notebooks.
- **Country values from Sea Around Us** are built from EEZs: stock status = unweighted mean of a country's EEZs; mean trophic level and max length = catch-weighted mean; catch = sum.
- **Colours** follow a colour-vision-checked palette in a fixed order; a country keeps the same colour in every figure.
