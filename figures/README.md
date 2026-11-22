# Final figure set for the report

This is the shortlist of figures for the ~10-page report, in reading order and mapped to the rubric sections. It is about 1–2 figures per page (15 figures). The links point to the figures where they are generated, so re-running a script updates the report figure as well. Everything else stays in the folder galleries as supporting material (appendix or the repo link).

| # | Report section (weight) | Figure | What it shows |
|---|---|---|---|
| 1 | Introduction (10%) | [1.01 World % overexploited](../0-visualisation/figures/1_global/1.01_world_overexploited_trend.png) | The problem: ~10% → 35.5% of stocks overexploited, 1974–2021 |
| 2 | Introduction | [1.07 FAO areas 2004 vs 2021](../0-visualisation/figures/1_global/1.07_fao_areas_dumbbell_2004_2021.png) | Area 71 (SEA) has the largest rise of any FAO area |
| 3 | Data Understanding & Preparation (15%) | [A.01 Missing-value map](../1-statistical-tests/figures/A_profile/A.01_missing_value_map.png) | Coverage of every source; gaps are reporting cycles, not data loss |
| 4 | Data Understanding & Preparation | [V.01 Validation summary](../3-data-quality/figures/V.01_validation_summary.png) | Sources agree on trend; level gaps explained |
| 5 | Data Understanding & Preparation | [I.01 Integrity summary](../3-data-quality/figures/I.01_integrity_summary.png) | Digits genuine; unreported catch and trade gaps are the issue |
| 6 | EDA (25%) | [1.03 World capture vs aquaculture](../0-visualisation/figures/1_global/1.03_world_capture_vs_aquaculture.png) | Wild catch flat since the 1990s while farming grows |
| 7 | EDA | [3.02 Overexploitation heatmap, EEZ × year](../0-visualisation/figures/3_sea_sau/3.02_overexploited_heatmap_eez_year.png) | Where and when SEA stocks became overexploited |
| 8 | EDA | [1.05 ACF levels vs differences](../0-visualisation/figures/1_global/1.05_acf_levels_vs_differences.png) | Why correlation tests must use differenced series |
| 9 | EDA | [E.02 Clustering of SEA EEZs](../1-statistical-tests/figures/E_clustering/E.02_sea_eez_dendrogram_heatmap.png) | Vietnam outlier + 3 groups that cut across borders |
| 10 | Statistical Analysis (25%) | [H1a.01 World trend tests](../2-hypothesis-tests/figures/1_H1a_global_trend/H1a.01_world_trend_mann_kendall_segmented.png) | H1a rejected: significant rise, no slowdown after 2015 |
| 11 | Statistical Analysis | [H1b.01 Catch CCF](../2-hypothesis-tests/figures/2_H1b_global_ccf/H1b.01_ccf_catch_vs_overexploited.png) | H1b not rejected (iteration 1) |
| 12 | Statistical Analysis | [H2a.01 SEA trends](../2-hypothesis-tests/figures/3_H2a_sea_trends/H2a.01_sen_slopes_sea_countries_eez.png) | H2a rejected across countries and EEZs |
| 13 | Statistical Analysis | [H2c.01 SEA vs world](../2-hypothesis-tests/figures/5_H2c_sea_vs_world/H2c.01_sea_vs_world_slopes.png) | Indonesia, Thailand, Malaysia worsen faster than the world |
| 14 | Statistical Analysis | [S.01 All drivers: levels vs beyond the trend](../2-hypothesis-tests/figures/10_summary_all_drivers/S.01_all_drivers_levels_vs_beyond_trend.png) | Iterations 1–4 in one figure: every driver shares the trend; none is linked beyond it (replaces I3.01 and I4.01 in the main body) |
| 15 | Discussion (15%) | [I2.01 Catch per unit effort](../2-hypothesis-tests/figures/7_iteration2_effort/I2.01_cpue_effort_catch_index.png) | Iteration 2: more effort for less fish — the depletion explanation |

**Appendix candidates:**
- [B.01 Q-Q normality checks](../1-statistical-tests/figures/B_assumptions/B.01_qq_plots_normality.png)
- [D.01 Residual diagnostics](../2-hypothesis-tests/figures/6_diagnostics/D.01_h1a_residual_diagnostics.png)
- [D.01 Chi-square class × period](../1-statistical-tests/figures/D_chi_square/D.01_overexploitation_class_vs_period.png)
- [I4.03 Island-building difference-in-differences](../2-hypothesis-tests/figures/9_iteration4_human_use/I4.03_h4c_island_building_did.png) (planned for the main body in the report outline)
- [I5.02 Stock rebuilding before/after 2015](../2-hypothesis-tests/figures/11_iteration5/I5.02_h5b_rebuilding_before_after_2015.png), [I5.03 Fishing-in-Balance](../2-hypothesis-tests/figures/11_iteration5/I5.03_h5c_fishing_in_balance.png) (main-body candidates: SDG 14.4 "restore" verdict and the SEA trophic-level puzzle), [I5.01 Pooled panel](../2-hypothesis-tests/figures/11_iteration5/I5.01_h5a_panel_fixed_effects.png)
- [I3.01 Warming: levels vs detrended](../2-hypothesis-tests/figures/8_iteration3_warming/I3.01_h3a_warming_levels_detrended_ccf.png), [I4.01 Shipping, dams, aquaculture](../2-hypothesis-tests/figures/9_iteration4_human_use/I4.01_h4a_global_drivers.png): summarised by S.01
- [I4.02 SEA shipping & structures CCF](../2-hypothesis-tests/figures/9_iteration4_human_use/I4.02_h4b_sea_ccf_heatmap.png), [I4.04 Platforms and wind](../2-hypothesis-tests/figures/9_iteration4_human_use/I4.04_structures_descriptive.png)
- [I3.02 SEA warming CCF](../2-hypothesis-tests/figures/8_iteration3_warming/I3.02_h3b_sea_sst_ccf.png)
- [5.04 Mirror trade](../0-visualisation/figures/5_integrity/5.04_mirror_trade_hs03.png)
