"""
clean_data.py — Step 0 of the pipeline: check and clean every input before any analysis
=========================================================================================

KIE4033 Assignment 1 · SDG 14 overfishing analysis (Global -> Southeast Asia)

The ONLY script that reads ``data/raw/``. Every later step (1-visualisation,
2-statistical-tests, 3-hypothesis-tests, 4-overview, dashboard) reads what this
script writes, and refuses to run until its checks have passed on the current raw
files (``loaders.require_checked_data``).

    python 0-data-quality/clean_data.py      # or: python run_all.py

Stage A — checked copies (data/processed/clean/)
    Every raw file is read, checked (C1–C4) and written to the same relative path
    under data/processed/clean/ with the same columns, after the file-level rules
    (R1 names, R4 not-yet-real values, R5 partial years). Lines that need no change
    are copied byte-for-byte, so values are exactly the raw values.
Stage B — tidy tables (data/processed/), built from the checked copies
    global_annual.csv        World, one row per year 1950–2025
    sea_country_annual.csv   SEA aggregate + 5 countries, one row per country-year 1950–2025
    sea_eez_annual.csv       11 Sea Around Us EEZs, one row per EEZ-year 1950–2019
    cleaning_log.csv         every rule applied and every check run, with counts
Gate
    If every check passes, data/processed/clean/_quality_checks_passed.json records the
    SHA-256 of every raw file; otherwise the script exits with status 1 and lists the
    failures, and the later steps will not run.

File checks (Stage A; a failure stops the pipeline)
---------------------------------------------------
C1  The file parses as a table and has at least one row.
C2  No exact-duplicate rows (a repeated download or append).
C3  No infinite numbers (division artefacts in a source).
C4  Missing values counted per file (reported, not a failure: several series are
    biennial or start late by design).

Cleaning rules (each one is logged)
-----------------------------------
R1  Harmonise names: "Viet Nam" -> "Vietnam"; one country name per table.
R2  Derive variables: % overexploited = collapsed + over-exploited stocks; unreported
    and foreign-fleet shares; industrial share; catch per unit effort (CPUE).
R3  Aggregate EEZs to countries: % overexploited = mean of the country's EEZs (it is a
    share of stocks), mean trophic level = catch-weighted, shares = catch-weighted.
R4  Flag, do not delete: FAO world years that look linearly interpolated; effort
    source breaks (|Δlog effort| > 0.5); FAO's 2023–24 new-method values are excluded
    from the old series (not comparable). Values that are not real yet are removed:
    GDAT dams with a completion year after LAST_YEAR (planned, not built).
R5  Missing values stay missing: biennial series (FAO areas, SDG 14.7.1) are NOT
    interpolated; partial years (2026 SST) are dropped.
R6  Tidy-table checks (a failure stops the pipeline): duplicate keys, value ranges
    (0–100 for percentages, ≥ 0 for amounts); missing-value counts per column.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))   # step 0 imports shared code only, never a later step
import loaders as ld  # noqa: E402  (every data loader; reads data/raw or the checked copies)
import style as v  # noqa: E402  (chart style)

OUT = ld.PROC
RAW = ld.RAW_DIR
CLEAN = ld.CLEAN_DIR
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
# Stage A — checked copies of every raw file
# --------------------------------------------------------------------------- #
LAST_YEAR = 2025          # last complete calendar year in the data
# Rows removed from a file: (relative path, year column, reason). Everything else is kept.
DROP_AFTER_LAST_YEAR = {
    "owid/sea-surface-temperature-anomaly.csv": ("Year", "R5 drop", "partial year (only some months observed)"),
    "validation/human_use/gdat_dams_by_year.csv": ("year", "R4 remove", "planned dams: completion year in the future"),
}
RENAME = {"Viet Nam": "Vietnam"}   # R1: one spelling per country (OWID's)
FAILURES: list[str] = []


def write_clean_copy(rel: str, text: str, df: pd.DataFrame, keep: np.ndarray) -> None:
    """
    Write the checked copy. If no rows are dropped, the (renamed) text is written as is;
    otherwise the kept data lines are written verbatim, so numbers are never re-formatted.
    Falls back to pandas only if lines and rows do not map one-to-one (quoted newlines).
    """
    out = CLEAN / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if keep.all():
        out.write_text(text, encoding="utf-8")
        return
    lines = text.splitlines(keepends=True)
    if len(lines) - 1 == len(df):
        out.write_text(lines[0] + "".join(l for l, k in zip(lines[1:], keep) if k), encoding="utf-8")
    else:
        df[keep].to_csv(out, index=False)


def clean_copies() -> None:
    """Stage A: check (C1–C4) and clean (R1, R4, R5) every file in data/raw/ into data/processed/clean/."""
    t = "clean copies"
    if CLEAN.exists():
        shutil.rmtree(CLEAN)       # stale copies (and the pass marker) never survive a re-run
    n_files = 0
    for path in sorted(RAW.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(RAW).as_posix()
        if path.suffix != ".csv":                       # README, manifest: copied unchanged
            (CLEAN / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, CLEAN / rel)
            continue
        n_files += 1
        text = path.read_text(encoding="utf-8")
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:                        # C1
            FAILURES.append(f"{rel}: does not parse ({exc})")
            continue
        if df.empty:
            FAILURES.append(f"{rel}: no rows")
        dup = int(df.duplicated().sum())                # C2
        if dup:
            FAILURES.append(f"{rel}: {dup} exact-duplicate rows")
        num = df.select_dtypes("number")
        n_inf = int(np.isinf(num.to_numpy(float)).sum()) if not num.empty else 0   # C3
        if n_inf:
            FAILURES.append(f"{rel}: {n_inf} infinite values")
        log(rel, "C1–C3 file checks", f"{len(df)} rows × {df.shape[1]} columns; duplicates {dup}; infinite {n_inf}",
            dup + n_inf)
        log(rel, "C4 missing values", "empty cells across all columns", int(df.isna().sum().sum()))
        n_ren = sum(text.count(a) for a in RENAME)      # R1
        for a, b in RENAME.items():
            text = text.replace(a, b)
        if n_ren:
            log(rel, "R1 harmonise", "'Viet Nam' -> 'Vietnam'", n_ren)
        keep = np.ones(len(df), bool)
        if rel in DROP_AFTER_LAST_YEAR:                 # R4 / R5
            col, rule, why = DROP_AFTER_LAST_YEAR[rel]
            keep = ~(pd.to_numeric(df[col], errors="coerce") > LAST_YEAR).to_numpy()
            log(rel, rule, f"rows with {col} > {LAST_YEAR}: {why}", int((~keep).sum()))
        write_clean_copy(rel, text, df, keep)
    log(t, "Stage A", f"{n_files} CSV files checked and written to data/processed/clean/", n_files)


# --------------------------------------------------------------------------- #
# Global table
# --------------------------------------------------------------------------- #
def build_global() -> pd.DataFrame:
    t = "global_annual"
    years = pd.Index(range(1950, 2026), name="year")
    fao = ld.world_fao()
    interp = ld.interpolated_years(fao)
    sdg = pd.read_csv(CLEAN / "validation" / "un_sdg" / "14_4_1_ER_H2O_FWTL.csv")
    sdg = sdg[sdg["geo_area_name"] == "World"].set_index("year")["value_pct_sustainable"]
    n_new = int((sdg.index > 2021).sum())
    sdg = sdg.loc[:2021]
    log(t, "R4 exclude", "UN SDG 14.4.1 world values after 2021 (FAO 2025 method change)", n_new)
    aqua = ld.owid_series("aquaculture-farmed-fish-production", "World")
    we = ld.world_effort()
    sau_catch = ld.sau_world_catch()
    g = pd.DataFrame(index=years)
    g["fao_pct_overexploited"] = fao
    g["fao_interpolated_flag"] = g.index.isin(interp)
    g["sdg_pct_sustainable"] = sdg
    g["sau_pct_overexploited"] = ld.world_sau()
    g["fao_capture_t"] = ld.capture("World")
    g["aquaculture_t"] = aqua
    g["sau_catch_t"] = sau_catch
    g["effort_nominal_kwdays"] = we["nom_kwdays"]
    g["effort_effective_kwdays"] = we["eff_kwdays"]
    g["cpue_kg_per_kwday"] = g["sau_catch_t"] / g["effort_nominal_kwdays"] * 1000
    g["sst_anomaly_c"] = ld.world_sst()
    g["ocean_heat_0_700m_1e22j"] = ld.world_ohc()
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
    d = ld.load_sau_eez()
    stock = d["stock"].rename(columns={"Collapsed": "pct_collapsed", "Over-exploited": "pct_over_exploited",
                                       "Exploited": "pct_exploited", "Developing": "pct_developing",
                                       "Rebuilding": "pct_rebuilding"})
    stock["pct_overexploited"] = stock["pct_collapsed"] + stock["pct_over_exploited"]
    p = ld.eez_panel()[["year", "eez", "mtl", "catch", "unrep_share", "foreign_share"]]
    c = d["catch"]
    tot = c.groupby(["eez", "year"])["tonnes"].sum()
    ind = c[c["fishing_sector"] == "Industrial"].groupby(["eez", "year"])["tonnes"].sum()
    ind_share = (ind.reindex(tot.index, fill_value=0) / tot * 100).rename("industrial_share_pct").reset_index()
    e = (stock.merge(p, on=["year", "eez"]).merge(ind_share, on=["eez", "year"])
         .rename(columns={"eez": "eez_id", "mtl": "mean_trophic_level", "catch": "catch_t",
                          "unrep_share": "unreported_share_pct", "foreign_share": "foreign_share_pct"}))
    e["eez_name"] = e["eez_id"].map(ld.eez_name)
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
    sea, cs = ld.sea_series(), ld.country_series()
    se, ss = ld.sea_effort(), ld.sea_sst()
    econ = pd.read_csv(CLEAN / "validation" / "economic" / "fisheries_gdp_capture.csv")
    iso = {"Malaysia": "MYS", "Indonesia": "IDN", "Thailand": "THA", "Vietnam": "VNM", "Philippines": "PHL"}
    ct = pd.read_csv(CLEAN / "validation" / "comtrade" / "fish_hs03_mirror_trade.csv")
    ct["mirror_gap_pct"] = (ct["partner_imports_usd"] / ct["reported_exports_usd"] - 1) * 100
    rows = []
    for n in ["SEA"] + ld.SEA5:
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
        df["fao_capture_t"] = (sum(ld.capture(c) for c in ld.SEA5) if n == "SEA" else ld.capture(n))
        eff = se[n]["nom_kwdays"]
        dl = np.log(eff).diff()
        df["effort_nominal_kwdays"] = eff
        df["effort_source_break_flag"] = df.index.isin(dl.index[dl.abs() > ld.BREAK_DLOG])
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
    print("Stage A: checked copies of data/raw/ -> data/processed/clean/")
    clean_copies()
    ld.use_source("clean")          # Stage B is built from the checked copies, not from raw
    print("Stage B: tidy tables")
    g = build_global()
    e = build_eez()
    c = build_country(e)
    for name, df in [("global_annual", g), ("sea_country_annual", c), ("sea_eez_annual", e)]:
        df.round(6).to_csv(OUT / f"{name}.csv", index=False)
        print(f"  saved data/processed/{name}.csv  ({len(df)} rows × {df.shape[1]} columns)")
    lg = pd.DataFrame(LOG)
    lg.to_csv(OUT / "cleaning_log.csv", index=False)
    bad = lg[lg["rule"].str.startswith("R6 range") | lg["rule"].str.startswith("R6 duplicate")]
    for _, r in bad[pd.to_numeric(bad["count"]) > 0].iterrows():
        FAILURES.append(f"{r['table']}: {r['rule']} {r['detail']} ({r['count']})")
    print(f"  saved data/processed/cleaning_log.csv  ({len(lg)} entries; "
          f"{int(pd.to_numeric(bad['count']).sum())} range/duplicate violations in the tidy tables)")

    # Gate: later steps run only if every check passed on exactly these raw files
    if FAILURES:
        print("DATA-QUALITY CHECKS FAILED — fix data/raw/ before any analysis:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    ld.PASSED.write_text(json.dumps({"checks": "passed", "csv_files": int((lg["rule"] == "C1–C3 file checks").sum()),
                                     "raw_sha256": ld.raw_fingerprint()}, indent=1) + "\n", encoding="utf-8")
    print(f"  all data-quality checks passed -> {ld.PASSED.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
