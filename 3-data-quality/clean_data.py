"""
clean_data.py — Build the cleaned, analysis-ready tables in data/processed/
==========================================================================

KIE4033 Assignment 1 · SDG 14 overfishing analysis (Global -> Southeast Asia)

Turns the raw files in ``data/raw/`` into three tidy tables (one row per unit and
year) plus a cleaning log. The analysis scripts in 0-, 1- and 2-* build the same
variables from the raw files with the same loaders; this script applies every
cleaning rule in one place, writes the result to disk for the report and for
anyone re-using the data, and records each rule and check in a log.

    python 3-data-quality/clean_data.py

Outputs (data/processed/)
-------------------------
global_annual.csv        World, one row per year 1950–2025
sea_country_annual.csv   SEA aggregate + 5 countries, one row per country-year 1950–2025
sea_eez_annual.csv       11 Sea Around Us EEZs, one row per EEZ-year 1950–2019
cleaning_log.csv         every rule applied and every check run, with counts

Cleaning rules (each one is logged)
-----------------------------------
R1  Harmonise names: "Viet Nam" -> "Vietnam"; one country name per table.
R2  Derive variables: % overexploited = collapsed + over-exploited stocks; unreported
    and foreign-fleet shares; industrial share; catch per unit effort (CPUE).
R3  Aggregate EEZs to countries: % overexploited = mean of the country's EEZs (it is a
    share of stocks), mean trophic level = catch-weighted, shares = catch-weighted.
R4  Flag, do not delete: FAO world years that look linearly interpolated; effort
    source breaks (|Δlog effort| > 0.5); FAO's 2023–24 new-method values are excluded
    from the old series (not comparable).
R5  Missing values stay missing: biennial series (FAO areas, SDG 14.7.1) are NOT
    interpolated; partial years (2026 SST) are dropped.
R6  Checks: duplicate keys, value ranges (0–100 for percentages, ≥ 0 for amounts),
    missing-value counts per column.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("0-visualisation", "1-statistical-tests", "2-hypothesis-tests"):
    sys.path.insert(0, str(ROOT / sub))
import hypothesis_tests as ht  # noqa: E402  (effort, warming and SEA builders)
import stat_tests as st  # noqa: E402  (FAO/SAU builders, interpolation flag)
import visualise as v  # noqa: E402  (raw loaders)

OUT = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
LOG: list[dict] = []


def log(table: str, rule: str, detail: str, count: int | float | str = "") -> None:
    LOG.append({"table": table, "rule": rule, "detail": detail, "count": count})


def checks(df: pd.DataFrame, table: str, keys: list[str], pct_cols: list[str], nonneg_cols: list[str]) -> None:
    """R6: duplicates, ranges and missing values, logged per column."""
    log(table, "R6 duplicate keys", f"rows sharing {keys}", int(df.duplicated(keys).sum()))
    for c in pct_cols:
        bad = int(((df[c] < 0) | (df[c] > 100)).sum())
        log(table, "R6 range 0–100", c, bad)
    for c in nonneg_cols:
        log(table, "R6 range ≥ 0", c, int((df[c] < 0).sum()))
    for c in df.columns:
        if c not in keys:
            log(table, "R6 missing values", c, int(df[c].isna().sum()))


# --------------------------------------------------------------------------- #
# Global table
# --------------------------------------------------------------------------- #
def build_global() -> pd.DataFrame:
    t = "global_annual"
    years = pd.Index(range(1950, 2026), name="year")
    fao = st.world_fao()
    interp = st.interpolated_years(fao)
    sdg = pd.read_csv(RAW / "validation" / "un_sdg" / "14_4_1_ER_H2O_FWTL.csv")
    sdg = sdg[sdg["geo_area_name"] == "World"].set_index("year")["value_pct_sustainable"]
    n_new = int((sdg.index > 2021).sum())
    sdg = sdg.loc[:2021]
    log(t, "R4 exclude", "UN SDG 14.4.1 world values after 2021 (FAO 2025 method change)", n_new)
    aqua = v.owid_series("aquaculture-farmed-fish-production", "World")
    we = ht.world_effort()
    sau_catch = ht.sau_world_catch()
    g = pd.DataFrame(index=years)
    g["fao_pct_overexploited"] = fao
    g["fao_interpolated_flag"] = g.index.isin(interp)
    g["sdg_pct_sustainable"] = sdg
    g["sau_pct_overexploited"] = st.world_sau()
    g["fao_capture_t"] = st.capture("World")
    g["aquaculture_t"] = aqua
    g["sau_catch_t"] = sau_catch
    g["effort_nominal_kwdays"] = we["nom_kwdays"]
    g["effort_effective_kwdays"] = we["eff_kwdays"]
    g["cpue_kg_per_kwday"] = g["sau_catch_t"] / g["effort_nominal_kwdays"] * 1000
    g["sst_anomaly_c"] = ht.world_sst()
    g["ocean_heat_0_700m_1e22j"] = ht.world_ohc()
    log(t, "R4 flag", "FAO world years on a straight line between neighbours (likely interpolated)", len(interp))
    log(t, "R5 drop", "partial year 2026 of the SST series", 1)
    log(t, "R2 derive", "cpue_kg_per_kwday = SAU catch (t) × 1000 ÷ nominal effort", int(g["cpue_kg_per_kwday"].notna().sum()))
    g = g.reset_index()
    checks(g, t, ["year"], ["fao_pct_overexploited", "sdg_pct_sustainable", "sau_pct_overexploited"],
           ["fao_capture_t", "aquaculture_t", "sau_catch_t", "effort_nominal_kwdays", "cpue_kg_per_kwday"])
    return g


# --------------------------------------------------------------------------- #
# SEA tables
# --------------------------------------------------------------------------- #
def build_eez() -> pd.DataFrame:
    t = "sea_eez_annual"
    d = v.load_sau_eez()
    stock = d["stock"].rename(columns={"Collapsed": "pct_collapsed", "Over-exploited": "pct_over_exploited",
                                       "Exploited": "pct_exploited", "Developing": "pct_developing",
                                       "Rebuilding": "pct_rebuilding"})
    stock["pct_overexploited"] = stock["pct_collapsed"] + stock["pct_over_exploited"]
    p = st.eez_panel()[["year", "eez", "mtl", "catch", "unrep_share", "foreign_share"]]
    c = d["catch"]
    tot = c.groupby(["eez", "year"])["tonnes"].sum()
    ind = c[c["fishing_sector"] == "Industrial"].groupby(["eez", "year"])["tonnes"].sum()
    ind_share = (ind.reindex(tot.index, fill_value=0) / tot * 100).rename("industrial_share_pct").reset_index()
    e = (stock.merge(p, on=["year", "eez"]).merge(ind_share, on=["eez", "year"])
         .rename(columns={"eez": "eez_id", "mtl": "mean_trophic_level", "catch": "catch_t",
                          "unrep_share": "unreported_share_pct", "foreign_share": "foreign_share_pct"}))
    e["eez_name"] = e["eez_id"].map(st.eez_name)
    e = e[["eez_id", "eez_name", "country", "year", "pct_overexploited", "pct_collapsed", "pct_over_exploited",
           "pct_exploited", "pct_developing", "pct_rebuilding", "mean_trophic_level", "catch_t",
           "unreported_share_pct", "foreign_share_pct", "industrial_share_pct"]].sort_values(["eez_id", "year"])
    log(t, "R2 derive", "pct_overexploited = collapsed + over-exploited (share of stocks)", len(e))
    log(t, "R2 derive", "unreported, foreign-fleet and industrial shares of total reconstructed catch", len(e))
    checks(e, t, ["eez_id", "year"], ["pct_overexploited", "unreported_share_pct", "foreign_share_pct",
                                       "industrial_share_pct"], ["catch_t"])
    return e


def build_country(eez: pd.DataFrame) -> pd.DataFrame:
    t = "sea_country_annual"
    sea, cs = ht.sea_series(), ht.country_series()
    se, ss = ht.sea_effort(), ht.sea_sst()
    econ = pd.read_csv(RAW / "validation" / "economic" / "fisheries_gdp_capture.csv")
    iso = {"Malaysia": "MYS", "Indonesia": "IDN", "Thailand": "THA", "Vietnam": "VNM", "Philippines": "PHL"}
    ct = pd.read_csv(RAW / "validation" / "comtrade" / "fish_hs03_mirror_trade.csv")
    ct["mirror_gap_pct"] = (ct["partner_imports_usd"] / ct["reported_exports_usd"] - 1) * 100
    rows = []
    for n in ["SEA"] + v.SEA5:
        src = sea if n == "SEA" else cs[n]
        sub = eez if n == "SEA" else eez[eez["country"] == n]
        w = sub.assign(u=sub["unreported_share_pct"] * sub["catch_t"], f=sub["foreign_share_pct"] * sub["catch_t"],
                       i=sub["industrial_share_pct"] * sub["catch_t"]).groupby("year")[["u", "f", "i", "catch_t"]].sum()
        df = pd.DataFrame(index=pd.Index(range(1950, 2026), name="year"))
        df["pct_overexploited"] = src["over"]
        df["mean_trophic_level"] = src["mtl"]
        df["sau_catch_t"] = src["catch"]
        df["unreported_share_pct"] = w["u"] / w["catch_t"]
        df["foreign_share_pct"] = w["f"] / w["catch_t"]
        df["industrial_share_pct"] = w["i"] / w["catch_t"]
        df["fao_capture_t"] = (sum(st.capture(c) for c in v.SEA5) if n == "SEA" else st.capture(n))
        eff = se[n]["nom_kwdays"]
        dl = np.log(eff).diff()
        df["effort_nominal_kwdays"] = eff
        df["effort_source_break_flag"] = df.index.isin(dl.index[dl.abs() > ht.BREAK_DLOG])
        df["cpue_kg_per_kwday"] = df["sau_catch_t"] / df["effort_nominal_kwdays"] * 1000
        df["sst_mean_c"] = ss[n]["sst_mean_c"]
        df["sst_anomaly_c"] = ss[n]["sst_anom_c"]
        df["sst_hot_months"] = ss[n]["hot_months"]
        if n in iso:
            g = econ[econ["iso3"] == iso[n]].set_index("year")
            df["sdg1471_fisheries_pct_gdp"] = g["sustainable_fisheries_pct_gdp"]
            df["mirror_gap_pct"] = ct[ct["country"] == n].set_index("year")["mirror_gap_pct"]
        df.insert(0, "country", n)
        rows.append(df.reset_index())
        log(t, "R4 flag", f"{n}: effort source-break years", int(df["effort_source_break_flag"].sum()))
    out = pd.concat(rows, ignore_index=True)
    log(t, "R1 harmonise", "'Viet Nam' (SAU, IMAS, ERSST) -> 'Vietnam' (OWID)", "all tables")
    log(t, "R3 aggregate", "country % overexploited = mean of its EEZs; MTL and shares = catch-weighted", len(out))
    log(t, "R3 aggregate", "SEA % overexploited = mean of 5 countries; SEA catch, effort = sums; SEA SST = area-weighted", 76)
    log(t, "R5 keep missing", "SDG 14.7.1 (biennial) and mirror trade (2011–2023) are not interpolated",
        int(out["sdg1471_fisheries_pct_gdp"].isna().sum()))
    checks(out, t, ["country", "year"], ["pct_overexploited", "unreported_share_pct", "foreign_share_pct",
                                          "industrial_share_pct"],
           ["sau_catch_t", "fao_capture_t", "effort_nominal_kwdays", "cpue_kg_per_kwday"])
    return out


def main() -> None:
    v.setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    g = build_global()
    e = build_eez()
    c = build_country(e)
    for name, df in [("global_annual", g), ("sea_country_annual", c), ("sea_eez_annual", e)]:
        df.round(6).to_csv(OUT / f"{name}.csv", index=False)
        print(f"  saved data/processed/{name}.csv  ({len(df)} rows × {df.shape[1]} columns)")
    lg = pd.DataFrame(LOG)
    lg.to_csv(OUT / "cleaning_log.csv", index=False)
    bad = lg[lg["rule"].str.startswith("R6 range") | lg["rule"].str.startswith("R6 duplicate")]
    print(f"  saved data/processed/cleaning_log.csv  ({len(lg)} entries; "
          f"{int(pd.to_numeric(bad['count']).sum())} range/duplicate violations)")


if __name__ == "__main__":
    main()
