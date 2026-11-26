"""
loaders.py — Every data loader of the project, in one place
===========================================================

KIE4033 Assignment 1 · SDG 14 overfishing analysis (Global -> Southeast Asia)

Shared by all pipeline steps (moved here unchanged from visualise.py, stat_tests.py and
hypothesis_tests.py, so that step 0 — data quality — no longer imports code from the
analysis steps that run after it).

Which copy of the data is read
------------------------------
* By default every loader reads the CHECKED copies in ``data/processed/clean/``,
  written by ``0-data-quality/clean_data.py`` (same folder layout and columns as
  ``data/raw/``, after the file-level cleaning rules).
* Only ``clean_data.py`` reads ``data/raw/`` (``use_source("raw")``).
* ``require_checked_data()`` stops an analysis step when the quality checks have not
  passed, or when ``data/raw/`` changed after they ran (SHA-256 fingerprint).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
CLEAN_DIR = PROC / "clean"
PASSED = CLEAN_DIR / "_quality_checks_passed.json"   # written by clean_data.py when every check passes

DATA = CLEAN_DIR          # rebound by use_source(); loaders resolve paths at call time
VAL = DATA / "validation"
HUMAN = VAL / "human_use"
EFFORT = VAL / "effort"
CLIMATE = VAL / "climate"

BREAK_DLOG = 0.5    # |Δlog effort| > 0.5 (a >65% jump in one year) = break in the source data (rule R4)


def use_source(which: str) -> None:
    """Point every loader at data/raw ('raw', data-quality step only) or data/processed/clean ('clean')."""
    global DATA, VAL, HUMAN, EFFORT, CLIMATE
    DATA = RAW_DIR if which == "raw" else CLEAN_DIR
    VAL = DATA / "validation"
    HUMAN, EFFORT, CLIMATE = VAL / "human_use", VAL / "effort", VAL / "climate"


def raw_fingerprint() -> dict[str, str]:
    """
    SHA-256 of every file in data/raw/ (relative path -> hash). Line endings are normalised
    first, so a Windows checkout (CRLF) has the same fingerprint as the committed files.
    """
    return {p.relative_to(RAW_DIR).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(RAW_DIR.rglob("*")) if p.is_file()}


def require_checked_data() -> None:
    """
    Gate for every analysis step: stop unless the data-quality step has passed on the
    current raw files. Raises SystemExit with the command to run.
    """
    fix = "run  python 0-data-quality/clean_data.py  (or  python run_all.py)  first"
    if not PASSED.exists():
        raise SystemExit(f"Data-quality checks have not passed yet: {fix}.")
    ok = json.loads(PASSED.read_text())
    if ok.get("raw_sha256") != raw_fingerprint():
        raise SystemExit(f"data/raw/ changed after the last data-quality check: {fix}.")


SEA5 = ["Malaysia", "Indonesia", "Thailand", "Vietnam", "Philippines"]


# Stock-status categories, ordered from worst to best, gray = neutral midpoint.
STATUS_ORDER = ["Collapsed", "Over-exploited", "Exploited", "Developing", "Rebuilding"]


# Sea Around Us EEZ ids grouped by country.
EEZ_COUNTRY = {
    459: "Malaysia", 460: "Malaysia", 461: "Malaysia", 463: "Malaysia",
    361: "Indonesia", 937: "Indonesia", 938: "Indonesia",
    956: "Thailand", 957: "Thailand",
    704: "Vietnam",
    608: "Philippines",
}


EEZ_LABEL = {
    459: "Peninsula West", 460: "Peninsula East", 461: "Sabah", 463: "Sarawak",
    361: "Eastern", 937: "Central", 938: "Indian Ocean",
    956: "Andaman Sea", 957: "Gulf of Thailand",
    704: "Viet Nam", 608: "Philippines",
}


def owid(slug: str) -> pd.DataFrame:
    """Load an OWID grapher CSV (columns: Entity, Code, Year, <value columns>)."""
    return pd.read_csv(DATA / "owid" / f"{slug}.csv")


def owid_series(slug: str, entity: str, col: int = -1) -> pd.Series:
    """One entity's series from an OWID CSV, indexed by year."""
    df = owid(slug)
    s = df[df["Entity"] == entity].set_index("Year").iloc[:, col]
    return s.astype(float).sort_index()


def eez_file(folder: str, eez_id: int) -> Path:
    return next((DATA / "sau" / folder).glob(f"{eez_id}_*.csv"))


def sau_stock(path: Path, dataset: str = "nss") -> pd.DataFrame:
    """SAU stock status. nss = % of stocks by status; css = % of catch by status."""
    df = pd.read_csv(path)
    return df[df["Data_Set"] == dataset].set_index("Year")[STATUS_ORDER].astype(float)


def sau_overexploited(path: Path) -> pd.Series:
    """% of stocks collapsed or over-exploited — the overfishing indicator."""
    s = sau_stock(path)
    return s["Collapsed"] + s["Over-exploited"]


def load_sau_eez() -> dict[str, pd.DataFrame]:
    """
    Long tables for all 11 SEA EEZs:
      stock  : year, eez, country, status columns (% of stocks)
      mti    : year, eez, country, catch_sum, mean_trophic_level, fib_index, mean_max_length
      catch  : the aggregated catch summary with eez + country columns
    """
    stock, mti, catch = [], [], []
    for eez, country in EEZ_COUNTRY.items():
        s = sau_stock(eez_file("stock_status", eez)).reset_index().rename(columns={"Year": "year"})
        s["eez"], s["country"] = eez, country
        stock.append(s)
        m = pd.read_csv(eez_file("marine_trophic_index", eez))
        m = m[["year", "catch_sum", "mean_trophic_level", "fib_index", "mean_max_length"]].copy()
        m["eez"], m["country"] = eez, country
        mti.append(m)
        c = pd.read_csv(eez_file("catch_summary", eez))
        c["eez"], c["country"] = eez, country
        catch.append(c)
    return {"stock": pd.concat(stock), "mti": pd.concat(mti), "catch": pd.concat(catch)}


def country_mtl(mti: pd.DataFrame) -> pd.DataFrame:
    """Country mean trophic level = catch-weighted mean of its EEZs; catch = sum."""
    g = mti.assign(w=mti["mean_trophic_level"] * mti["catch_sum"],
                   wf=mti["mean_max_length"] * mti["catch_sum"])
    out = g.groupby(["country", "year"]).agg(catch=("catch_sum", "sum"), w=("w", "sum"), wf=("wf", "sum"))
    out["mtl"] = out["w"] / out["catch"]
    out["max_length"] = out["wf"] / out["catch"]
    return out[["catch", "mtl", "max_length"]].reset_index()


def world_fao() -> pd.Series:
    """FAO % of world stocks overexploited (OWID), 1974–2021."""
    return owid_series("share-of-fish-stocks-overexploited", "World")


def world_sau() -> pd.Series:
    """Sea Around Us % of world stocks collapsed + over-exploited, 1950–2019."""
    return sau_overexploited(VAL / "sau_regions" / "stock_status" / "global.csv")


def interpolated_years(s: pd.Series) -> list[int]:
    """
    Years that lie exactly on the straight line between their neighbours
    (second difference = 0). In the FAO world series these are years filled by
    linear interpolation between assessments rather than measured.
    """
    dd = s.diff().diff().round(6)
    return [int(s.index[s.index.get_loc(y) - 1]) for y in dd.index[dd == 0]]


def eez_panel() -> pd.DataFrame:
    """
    One row per Southeast Asian EEZ and year (11 EEZs x 1950–2019):
      over          % of stocks collapsed or over-exploited
      mtl, catch    mean trophic level and total catch (t)
      unrep_share   % of reconstructed catch that is unreported
      foreign_share % of catch taken by foreign fleets
    """
    d = load_sau_eez()
    st = d["stock"].assign(over=lambda x: x["Collapsed"] + x["Over-exploited"])[["year", "eez", "country", "over"]]
    m = d["mti"][["year", "eez", "catch_sum", "mean_trophic_level"]].rename(
        columns={"catch_sum": "catch", "mean_trophic_level": "mtl"})
    c = d["catch"]
    tot = c.groupby(["eez", "year"])["tonnes"].sum()
    unrep = c[c["reporting_status"] == "Unreported"].groupby(["eez", "year"])["tonnes"].sum()
    foreign = c[c["fleet"] == "foreign"].groupby(["eez", "year"])["tonnes"].sum()
    shares = pd.DataFrame({
        "unrep_share": unrep.reindex(tot.index, fill_value=0) / tot * 100,
        "foreign_share": foreign.reindex(tot.index, fill_value=0) / tot * 100,
    }).reset_index()
    return st.merge(m, on=["year", "eez"]).merge(shares, on=["eez", "year"])


def eez_name(eez: int) -> str:
    """Readable EEZ label, e.g. 'Malaysia – Sabah'; single-EEZ countries keep one name."""
    c, z = EEZ_COUNTRY[eez], EEZ_LABEL[eez]
    return c if z in (c, "Viet Nam") else f"{c} – {z}"


def country_panel(p: pd.DataFrame) -> pd.DataFrame:
    """
    Country-year values: % overexploited = simple mean of the country's EEZs (it is
    already a share of stocks); MTL = catch-weighted mean of its EEZs, so a small
    EEZ cannot dominate; unreported share = catch-weighted as well.
    """
    g = p.assign(wm=p["mtl"] * p["catch"], wu=p["unrep_share"] * p["catch"])
    out = g.groupby(["country", "year"]).agg(over=("over", "mean"), catch=("catch", "sum"),
                                             wm=("wm", "sum"), wu=("wu", "sum"))
    out["mtl"] = out["wm"] / out["catch"]
    out["unrep_share"] = out["wu"] / out["catch"]
    return out[["over", "mtl", "unrep_share", "catch"]].reset_index()


def capture(entity: str) -> pd.Series:
    """FAO capture fisheries production (t), via OWID, 1960–2024."""
    return owid_series("capture-fishery-production", entity)


def sea_series() -> pd.DataFrame:
    """
    SEA aggregate, 1950–2019:
      over  = mean of the 5 countries' % overexploited (each country counts equally)
      mtl   = catch-weighted mean trophic level of all 11 EEZs
      catch = total catch of the 11 EEZs (t)
    """
    p = eez_panel()
    cp = country_panel(p)
    agg = p.assign(wm=p["mtl"] * p["catch"]).groupby("year").agg(catch=("catch", "sum"), wm=("wm", "sum"))
    agg["mtl"] = agg["wm"] / agg["catch"]
    agg["over"] = cp.groupby("year")["over"].mean()
    return agg[["over", "mtl", "catch"]]


def country_series() -> dict[str, pd.DataFrame]:
    cp = country_panel(eez_panel())
    return {c: cp[cp["country"] == c].set_index("year")[["over", "mtl", "catch"]] for c in SEA5}


def sau_world_catch() -> pd.Series:
    """Sea Around Us global reconstructed catch = landings + discards (t), 1950–2019."""
    d = pd.read_csv(VAL / "sau_regions" / "catch" / "global.csv").set_index("year")
    return d["landings"] + d["discards"]


def sau_world_mtl() -> pd.Series:
    d = pd.read_csv(VAL / "sau_regions" / "marine_trophic_index" / "global.csv").set_index("year")
    return d["mean_trophic_level"]


def fao_area(name: str) -> pd.Series:
    o = owid("share-of-fish-stocks-overexploited")
    return o[o["Entity"] == f"{name} (FAO)"].set_index("Year").iloc[:, -1].astype(float)


def world_effort() -> pd.DataFrame:
    """World fishing effort 1950–2017 (kW × days at sea; nominal and with technological creep)."""
    return pd.read_csv(EFFORT / "world_effort.csv").set_index("year")


def sea_effort() -> dict[str, pd.DataFrame]:
    """Effort by all fleets inside each SEA5 EEZ, 1950–2017; 'SEA' = sum of the five."""
    d = pd.read_csv(EFFORT / "sea_eez_effort.csv")
    d["eez"] = d["eez"].replace({"Viet Nam": "Vietnam"})
    out = {c: d[d["eez"] == c].set_index("year").drop(columns="eez") for c in SEA5}
    out["SEA"] = sum(out[c] for c in SEA5)
    return out


def world_sst() -> pd.Series:
    """World sea-surface temperature anomaly (°C vs 1961–1990), HadSST via OWID; 1950–2025 (2026 is partial)."""
    d = owid("sea-surface-temperature-anomaly")
    return d[d["Entity"] == "World"].set_index("Year")["Average"].loc[1950:2025]


def world_ohc() -> pd.Series:
    """Upper-700 m ocean heat content anomaly (10^22 J), NOAA series via OWID; 1955–2025."""
    d = owid("ocean-heat-content-upper")
    return d[d["Entity"] == "World"].set_index("Year")["NOAA"].dropna()


def sea_sst() -> dict[str, pd.DataFrame]:
    """Annual mean SST (°C), anomaly vs 1971–2000 and hot-month count per SEA5 EEZ and 'SEA', 1950–2025."""
    d = pd.read_csv(CLIMATE / "sea_eez_sst_ersst.csv")
    d["region"] = d["region"].replace({"Viet Nam": "Vietnam"})
    return {r: g.set_index("year").drop(columns="region") for r, g in d.groupby("region")}


ISO3 = {"MYS": "Malaysia", "IDN": "Indonesia", "THA": "Thailand", "VNM": "Vietnam", "PHL": "Philippines",
        "WLD": "World"}


SPLICE: list[dict] = []        # container-series splice check -> 9_I4_container_splice.csv


def container_teu() -> pd.DataFrame:
    """
    Container port throughput (TEU) per region, 2000–2024. The current WDI series starts in
    2010; the WDI December-2020 archive covers 2000–2019. Archive years 2000–2009 are rescaled
    by the median current/archive ratio over the 2010–2019 overlap so the joined series has no
    level jump (the match is logged to SPLICE). SEA = sum of the 5 countries.
    """
    d = pd.read_csv(HUMAN / "wb_shipping.csv")
    d = d[d["indicator"] == "IS.SHP.GOOD.TU"]
    out = {}
    SPLICE.clear()
    for iso, g in d.groupby("iso3"):
        a = g[g["source"] == "WDI_archive_202012"].set_index("year")["value"]
        c = g[g["source"] == "WDI_current"].set_index("year")["value"]
        ratio = (c / a).dropna()
        SPLICE.append({"region": ISO3[iso], "overlap_years": f"{ratio.index.min()}–{ratio.index.max()}",
                       "median_ratio_current_to_archive": ratio.median(), "min_ratio": ratio.min(),
                       "max_ratio": ratio.max(),
                       "growth_corr_overlap": np.corrcoef(np.log(a).diff().loc[2011:2019],
                                                          np.log(c).diff().loc[2011:2019])[0, 1]})
        out[ISO3[iso]] = pd.concat([a.loc[:2009] * ratio.median(), c]).sort_index()
    df = pd.DataFrame(out)
    df["SEA"] = df[SEA5].sum(axis=1, min_count=5)
    return df


def lsci() -> pd.DataFrame:
    """UNCTAD liner shipping connectivity index (current WDI, 2006–2021; no world value). SEA = mean."""
    d = pd.read_csv(HUMAN / "wb_shipping.csv")
    d = d[(d["indicator"] == "IS.SHP.GCNW.XQ") & (d["source"] == "WDI_current")]
    df = d.pivot(index="year", columns="iso3", values="value").rename(columns=ISO3)
    df["SEA"] = df[SEA5].mean(axis=1)
    return df


def seaborne() -> pd.DataFrame:
    """
    UNCTAD seaborne trade volume (thousand t), 2000–2024. Countries: goods loaded + discharged
    (total handled). World: goods loaded only (world loaded = world discharged). SEA = sum.
    """
    d = pd.read_csv(HUMAN / "unctad_seaborne_trade.csv")
    tot = d.groupby(["iso3", "year"])["thousand_tonnes"].sum().unstack(0).rename(columns=ISO3)
    tot["World"] = d[(d["iso3"] == "WLD") & (d["flow"] == "loaded")].set_index("year")["thousand_tonnes"]
    tot["SEA"] = tot[SEA5].sum(axis=1, min_count=5)
    return tot


def dams(last: int = 2019) -> pd.DataFrame:
    """
    Cumulative number of dams with a known completion year (GDAT v1), 1900–`last`.
    Decade-only years (e.g. '195') sit at the decade start. Undated dams (34% of GDAT, and
    more than half in Indonesia, Thailand and Vietnam) cannot be placed and are excluded,
    so the series is a lower bound; completions also thin out after ~2005 because the
    database lags construction.
    """
    d = pd.read_csv(HUMAN / "gdat_dams_by_year.csv").dropna(subset=["year"])
    d["year"] = d["year"].astype(int)
    yrs = range(1900, last + 1)
    out = {}
    for reg, g in d.groupby("region"):
        per_year = g.groupby("year")["n_dams"].sum()
        before = per_year.loc[:1899].sum()
        out[ISO3[reg]] = per_year.reindex(yrs, fill_value=0).cumsum() + before
    df = pd.DataFrame(out)
    df["SEA"] = df[SEA5].sum(axis=1)
    return df


def aquaculture() -> pd.DataFrame:
    """Aquaculture production (t, all farmed aquatic animals), FAO via OWID, 1960–2024. SEA = sum."""
    o = owid("aquaculture-farmed-fish-production")
    df = o[o["Entity"].isin(SEA5 + ["World"])].pivot(index="Year", columns="Entity",
                                                     values="Aquaculture production (metric tons)")
    df["SEA"] = df[SEA5].sum(axis=1, min_count=5)
    return df


# --------------------------------------------------------------------------- #
# Iteration 7 — sea and water pollution (all OWID files in data/raw/owid/)
# --------------------------------------------------------------------------- #
def _owid_wide(slug: str, col: str | int = -1, mean_sea: bool = True) -> pd.DataFrame:
    """Year × entity table of one OWID value column (World + SEA5); 'SEA' = simple mean of the 5 countries."""
    o = owid(slug)
    c = col if isinstance(col, str) else o.columns[col]
    df = o[o["Entity"].isin(SEA5 + ["World"])].pivot(index="Year", columns="Entity", values=c).astype(float)
    if mean_sea:
        df["SEA"] = df[SEA5].mean(axis=1, skipna=False)
    return df


def n_fertiliser() -> pd.DataFrame:
    """Nitrogen fertiliser applied per hectare of cropland (kg N/ha), FAO via OWID, 1961–2024."""
    return _owid_wide("nitrogen-fertilizer-application-per-hectare-of-cropland")


def pesticides() -> pd.DataFrame:
    """Pesticides applied per hectare of cropland (kg/ha), FAO via OWID, 1990–2024."""
    return _owid_wide("pesticide-use-per-hectare-of-cropland")


def chlorophyll() -> pd.DataFrame:
    """SDG 14.1.1a coastal eutrophication proxy: chlorophyll-a deviation (%), remote sensing, 2005–2022."""
    return _owid_wide("chlorophyll-a-deviation-from-the-global-average")


def plastic_production() -> pd.Series:
    """World plastic production (t/year), Geyer et al. via OWID, 1950–2019 (1974 not reported)."""
    return owid_series("global-plastics-production", "World")


def oil_spills() -> pd.DataFrame:
    """World tanker oil spills ≥ 7 t (ITOPF via OWID), 1970–2025: counts by size and tonnes spilled."""
    n = owid("number-oil-spills").set_index("Year")
    q = owid("quantity-oil-spills").set_index("Year")
    out = pd.DataFrame({"large": n.iloc[:, -2], "medium": n.iloc[:, -1]}).astype(float)
    out["n_spills"] = out["large"] + out["medium"]
    out["tonnes"] = q.iloc[:, -1].astype(float)
    return out


def seawater_ph() -> pd.Series:
    """
    Ocean acidification: surface seawater pH at Station ALOHA, Hawaii (HOT), the longest open-ocean pH
    record (OWID). Calendar-year mean of the deseasonalised running annual average, 1990–2024 (the
    running mean is first available in Nov 1989). Lower pH = more acidic.
    """
    d = owid("seawater-ph").dropna(subset=["Annual average"])
    d["year"] = pd.to_datetime(d["Day"]).dt.year
    s = d.groupby("year")["Annual average"].mean()
    return s.loc[1990:]


def coastal_nutrients() -> pd.DataFrame:
    """Modelled nitrogen and phosphorus loading (t/yr) to the 5 SEA large marine ecosystems, 2000–2015."""
    return owid("coastal-eutrophication").rename(columns={"Entity": "lme", "Year": "year"})


# Sea Around Us LME files (validation/sau_regions) ↔ the LME names used in the OWID nutrient file
SAU_LME = {"lme_34_bay_of_bengal": "Bay of Bengal", "lme_35_gulf_of_thailand": "Gulf of Thailand",
           "lme_36_south_china_sea": "South China Sea", "lme_37_sulu_celebes_sea": "Sulu-Celebes Sea",
           "lme_38_indonesian_sea": "Indonesian Sea"}


def sau_lme_panel() -> pd.DataFrame:
    """
    Long table (lme, year, over, mtl) for the 5 SEA large marine ecosystems, 1950–2019:
    over = Sea Around Us % of stocks collapsed or over-exploited, mtl = mean trophic level of the catch.
    LME names match coastal_nutrients(), so the two can be joined on (lme, year).
    """
    rows = []
    for stem, lme in SAU_LME.items():
        over = sau_overexploited(VAL / "sau_regions" / "stock_status" / f"{stem}.csv").rename("over")
        mtl = pd.read_csv(VAL / "sau_regions" / "marine_trophic_index" / f"{stem}.csv").set_index("year")[
            "mean_trophic_level"].rename("mtl")
        d = pd.concat([over, mtl], axis=1).rename_axis("year").reset_index()
        rows.append(d.assign(lme=lme))
    return pd.concat(rows, ignore_index=True)[["lme", "year", "over", "mtl"]]


def plastic_cross_section() -> pd.DataFrame:
    """2019 snapshot per entity: plastic emitted to the ocean (t) and mismanaged plastic waste (kg/person)."""
    e = owid("plastic-waste-emitted-to-the-ocean").set_index("Entity").iloc[:, -1]
    m = owid("mismanaged-plastic-waste-per-capita").set_index("Entity").iloc[:, -1]
    return pd.DataFrame({"plastic_to_ocean_t": e, "mismanaged_kg_per_person": m})


def beach_litter() -> pd.DataFrame:
    """SDG 14.1.1b beach litter (items per km²), 2015–2022, sparse (no 2019–2020 country values)."""
    return _owid_wide("beach-litter", mean_sea=False)


def eez_full_panel() -> pd.DataFrame:
    """Cleaned EEZ-year table (0-data-quality/clean_data.py output): adds industrial and rebuilding shares."""
    p = pd.read_csv(PROC / "sea_eez_annual.csv")
    return p.rename(columns={"eez_id": "unit", "pct_overexploited": "over", "mean_trophic_level": "mtl",
                             "catch_t": "catch"})
