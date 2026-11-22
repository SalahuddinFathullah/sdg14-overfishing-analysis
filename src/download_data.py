"""
download_data.py
================
Re-downloads every raw dataset used in this project into ``data/raw/``.

Running this script reproduces the CSV files that are committed in the repo,
so anyone can check the data came from the original sources unchanged.

Sources
-------
1. Our World in Data (OWID) grapher CSVs  ->  data/raw/owid/
   - share-of-fish-stocks-overexploited       (FAO; World 1974-2021, FAO areas, countries)
   - fish-stocks-within-sustainable-levels    (FAO; SDG indicator 14.4.1)
   - capture-fishery-production               (FAO via World Bank; 1960-2024)
   - fish-catch-gear-type                     (Sea Around Us; 1950-2018)
   - fish-discards                            (Sea Around Us; 1950-2018)
   - aquaculture-farmed-fish-production       (FAO via World Bank; context only)
   The last four are filtered to World + the 5 SEA focus countries
   (Malaysia, Indonesia, Thailand, Vietnam, Philippines). The first two are kept
   in full because they also contain the FAO fishing-area series.

2. Sea Around Us (SAU) API, per Exclusive Economic Zone (EEZ)  ->  data/raw/sau/
   - marine_trophic_index/   mean trophic level of the catch (MTI), 1950-2019
   - stock_status/           % of stocks / % of catch by status, 1950-2019
   - catch_summary/          total catch (t) and landed value by year x sector x
                             landings/discards x reported/unreported x domestic/foreign fleet
   for the 11 EEZs belonging to the 5 SEA focus countries.

3. Alternative sources for cross-validation  ->  data/raw/validation/
   - sau_regions/   SAU stock status, MTI and catch for the World, FAO areas 57/71
                    and the 5 SEA Large Marine Ecosystems (LMEs 34-38)
   - un_sdg/        UN SDG Global Database, indicator 14.4.1 (series ER_H2O_FWTL)
   - world_bank/    World Bank WDI API, capture production ER.FSH.CAPT.MT

4. Data-integrity sources  ->  data/raw/validation/{economic,comtrade,benford}/
   - economic/   SDG 14.7.1 + World Bank GDP + capture (implied value per tonne)
   - comtrade/   UN Comtrade mirror trade statistics for fish (HS 03)
   - benford/    leading-digit counts of SAU taxon-level catch values

5. Fishing effort (iteration 2 of the analysis)  ->  data/raw/validation/effort/
   Rousseau et al. Global Fishing Effort database, IMAS (University of Tasmania),
   1950-2017, CC BY 4.0, https://doi.org/10.25959/MNGY-0Q43
   - world_effort.csv     world total effort by year (all fishing countries)
   - sea_eez_effort.csv   effort inside each SEA5 country's EEZ by year (all fleets)
   Effort = active vessels x engine power (kW) x days at sea. "Nominal" has no
   technological creep; "effective" adds the database's creep factor.

6. Ocean warming (iteration 3 of the analysis)
   - owid/sea-surface-temperature-anomaly.csv   World SST anomaly (HadSST), World rows
   - owid/ocean-heat-content-upper.csv           upper-700 m ocean heat content, World rows
   - validation/climate/eez_ersst_cell_weights.csv  EEZ area (km2) of each SEA5 EEZ inside
                                                each 2-degree ERSST grid cell (from the IMAS
                                                CellstoLME_EEZ 0.5-degree cell list)
   - validation/climate/sea_eez_sst_ersst.csv   annual EEZ-mean SST per SEA5 country and 'SEA'
                                                from NOAA ERSST v5 (PSL OPeNDAP), 1950-2025

7. Human use of the sea (iteration 4 of the analysis)  ->  data/raw/validation/human_use/
   - wb_shipping.csv            World Bank WDI: container port traffic (TEU, UNCTAD data) and liner
                                shipping connectivity index; current WDI + the WDI Dec-2020 archive
                                (source 57), which still holds container data for 2000-2009
   - unctad_seaborne_trade.csv  UNCTADstat US.SeaborneTrade, goods loaded / discharged, 2000-2024
   - gdat_dams_by_year.csv      Global Dam Tracker v1 (Zhang & Gu 2023, Zenodo 7616852, CC BY 4.0),
                                dam counts and reported reservoir volume by completion year
   - deepowt_offshore_wind.csv  DeepOWT (Hoeser et al. 2022, Zenodo 5933967, CC BY 4.0), quarterly
                                offshore wind turbine counts, World and Vietnam, 2016Q3-2021Q2
   - oogp_platforms.csv         OOGP (Si et al. 2026, Zenodo 18350974, CC BY 4.0), offshore oil & gas
                                platforms detected per year, 6 basins incl. the Gulf of Thailand

Usage
-----
    python src/download_data.py                   # download everything
    python src/download_data.py --only owid
    python src/download_data.py --only sau
    python src/download_data.py --only validation
    python src/download_data.py --only integrity    # SDG 14.7.1/GDP, Comtrade, Benford (slow)
    python src/download_data.py --only effort       # streams ~110 MB of IMAS CSVs
    python src/download_data.py --only climate      # OWID SST/ocean heat + NOAA ERSST v5 for SEA EEZs
    python src/download_data.py --only human_use    # shipping, dams, offshore wind/platforms (~20 MB)

Notes
-----
- The SAU API can be slow (30-60 s per request); a generous timeout is used.
- SAU returns a ZIP per request; only the main CSV inside it is kept.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
import zipfile
from datetime import date
from pathlib import Path

import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

# Country / aggregate names exactly as they appear in OWID's "Entity" column.
SEA_COUNTRIES = ["Malaysia", "Indonesia", "Thailand", "Vietnam", "Philippines"]
OWID_KEEP_ENTITIES = ["World"] + SEA_COUNTRIES

OWID_BASE = "https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=false"

# slug -> True if the file is filtered to OWID_KEEP_ENTITIES, False if kept whole.
OWID_DATASETS = {
    "share-of-fish-stocks-overexploited": False,
    "fish-stocks-within-sustainable-levels": False,
    "capture-fishery-production": True,
    "fish-catch-gear-type": True,
    "fish-discards": True,
    "aquaculture-farmed-fish-production": True,
}

SAU_BASE = "https://api.seaaroundus.org/api/v1/eez"

# Sea Around Us EEZ id -> EEZ name. Several countries are split into sub-EEZs.
SAU_EEZS = {
    459: "Malaysia (Peninsula West)",
    460: "Malaysia (Peninsula East)",
    461: "Malaysia (Sabah)",
    463: "Malaysia (Sarawak)",
    937: "Indonesia (Central)",
    361: "Indonesia (Eastern)",
    938: "Indonesia (Indian Ocean)",
    956: "Thailand (Andaman Sea)",
    957: "Thailand (Gulf of Thailand)",
    704: "Viet Nam",
    608: "Philippines",
}

# measure folder -> URL template (same parameters the SAU website uses by default)
SAU_MEASURES = {
    "marine_trophic_index": SAU_BASE
    + "/marine-trophic-index/?format=csv&region_id={id}&transfer_efficiency=0.1"
    "&sub_area_id=&tl_min=2&tl_max=5",
    "stock_status": SAU_BASE + "/stock-status/?format=csv&region_id={id}&sub_area_id=",
    # Full catch record (every taxon x fleet x gear, 13-53 MB per EEZ). It is
    # aggregated by `summarise_catch` before saving, so it is stored as catch_summary/.
    "catch_summary": SAU_BASE + "/tonnage/catchtype/?format=csv&region_id={id}&limit=10",
}

HEADERS = {"User-Agent": "sdg14-overfishing-analysis (KIE4033 coursework)"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def eez_slug(eez_id: int) -> str:
    """File-name-safe slug, e.g. 459 -> '459_malaysia_peninsula_west'."""
    name = SAU_EEZS[eez_id].lower()
    for ch in "()":
        name = name.replace(ch, "")
    return f"{eez_id}_" + "_".join(name.split())


def get(url: str, timeout: int = 120, retries: int = 3) -> bytes:
    """GET a URL with simple retry/back-off; returns the raw response body."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.content
        except requests.RequestException as err:
            if attempt == retries:
                raise
            print(f"    retry {attempt}/{retries - 1} after error: {err}")
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def first_csv_in_zip(blob: bytes) -> str:
    """SAU returns a ZIP; return the text of the first .csv file inside it."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        return zf.read(name).decode("utf-8")


def eez_countries(area: str) -> list[str]:
    """
    Candidate owner-country names for an SAU EEZ name. The country is either the
    part outside the brackets ("Malaysia (Peninsula West)") or inside them
    ("Sabah (Malaysia)"); unsplit EEZs are just the country ("Viet Nam").
    """
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", area)
    return [m.group(1).strip(), m.group(2).strip()] if m else [area.strip()]


def summarise_catch(text: str) -> str:
    """
    Aggregate the full SAU catch export (one row per taxon x fleet x gear x ...)
    to one row per year x fishing_sector x catch_type x reporting_status x fleet.

    fleet = "domestic" if the fishing country is the EEZ's country, else "foreign".
    Sums are accumulated in file order with plain floats (no pandas) so the
    output is identical to the committed files.
    """
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    col = {name: header.index(name) for name in (
        "area_name", "year", "fishing_entity", "fishing_sector",
        "catch_type", "reporting_status", "tonnes", "landed_value")}

    def num(s: str) -> float:
        try:
            x = float(s)
            return 0.0 if x != x else x  # treat NaN as 0
        except ValueError:
            return 0.0

    totals: dict[tuple, list[float]] = {}
    for row in reader:
        if not row:
            continue
        fleet = "domestic" if row[col["fishing_entity"]] in eez_countries(row[col["area_name"]]) else "foreign"
        key = (int(row[col["year"]]), row[col["fishing_sector"]], row[col["catch_type"]],
               row[col["reporting_status"]], fleet)
        acc = totals.setdefault(key, [0.0, 0.0])
        acc[0] += num(row[col["tonnes"]])
        acc[1] += num(row[col["landed_value"]])

    lines = ["year,fishing_sector,catch_type,reporting_status,fleet,tonnes,landed_value"]
    for key in sorted(totals):
        t, v = totals[key]
        lines.append(",".join(map(str, key)) + f",{t:.3f},{v:.0f}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Downloaders
# --------------------------------------------------------------------------- #
def download_owid() -> None:
    out_dir = RAW_DIR / "owid"
    out_dir.mkdir(parents=True, exist_ok=True)
    for slug, filtered in OWID_DATASETS.items():
        print(f"OWID  {slug}")
        lines = get(OWID_BASE.format(slug=slug)).decode("utf-8").strip().split("\n")
        if filtered:
            # Filter on the text lines (not via pandas) so numbers keep their
            # original formatting and the output matches the committed files.
            keep = set(OWID_KEEP_ENTITIES)
            header, rows = lines[0], lines[1:]
            rows = [r for r in rows if next(csv.reader([r]))[0] in keep]
            lines = [header] + rows
        (out_dir / f"{slug}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_sau() -> None:
    for measure, template in SAU_MEASURES.items():
        out_dir = RAW_DIR / "sau" / measure
        out_dir.mkdir(parents=True, exist_ok=True)
        for eez_id in SAU_EEZS:
            print(f"SAU   {measure:22s} {SAU_EEZS[eez_id]}")
            text = first_csv_in_zip(get(template.format(id=eez_id), timeout=300))
            if measure == "catch_summary":
                text = summarise_catch(text)
            else:
                # Normalise Windows line endings so diffs stay clean in git.
                text = text.replace("\r\n", "\n").rstrip("\n") + "\n"
            (out_dir / f"{eez_slug(eez_id)}.csv").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Validation (alternative) sources  ->  data/raw/validation/
# --------------------------------------------------------------------------- #
# Sea Around Us at other spatial scales: the whole ocean, the two FAO areas that
# cover SEA, and the 5 SEA Large Marine Ecosystems (LMEs).
SAU_REGIONS = {
    ("global", 1): "global",
    ("fao", 57): "fao_57_indian_ocean_eastern",
    ("fao", 71): "fao_71_pacific_western_central",
    ("lme", 34): "lme_34_bay_of_bengal",
    ("lme", 35): "lme_35_gulf_of_thailand",
    ("lme", 36): "lme_36_south_china_sea",
    ("lme", 37): "lme_37_sulu_celebes_sea",
    ("lme", 38): "lme_38_indonesian_sea",
}
SAU_API = "https://api.seaaroundus.org/api/v1"
UN_SDG_URL = ("https://unstats.un.org/SDGAPI/v1/sdg/Series/Data"
              "?seriesCode=ER_H2O_FWTL&pageSize=10000")          # SDG 14.4.1
WB_URL = ("https://api.worldbank.org/v2/country/WLD;MYS;IDN;THA;VNM;PHL"
          "/indicator/ER.FSH.CAPT.MT?format=json&per_page=1000")  # capture production


def js_num(x) -> str:
    """Format a number the way JavaScript's String(x) does (used for the committed files)."""
    if x is None:
        return ""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return repr(x) if isinstance(x, float) else str(x)


def q(s) -> str:
    """Quote a CSV field only when needed (same rule as the committed files)."""
    s = "" if s is None else str(s)
    return '"' + s.replace('"', '""') + '"' if any(c in s for c in '",\n\r') else s


def download_validation() -> None:
    root = RAW_DIR / "validation"

    # 1) Sea Around Us regional series ---------------------------------------
    for (region, rid), slug in SAU_REGIONS.items():
        print(f"SAU   region {slug}")
        for measure, url in {
            "stock_status": f"{SAU_API}/{region}/stock-status/?format=csv&region_id={rid}&sub_area_id=",
            "marine_trophic_index": f"{SAU_API}/{region}/marine-trophic-index/?format=csv&region_id={rid}"
                                    "&transfer_efficiency=0.1&sub_area_id=&tl_min=2&tl_max=5",
        }.items():
            text = first_csv_in_zip(get(url, timeout=300))
            out = root / "sau_regions" / measure / f"{slug}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text.replace("\r\n", "\n").rstrip("\n") + "\n", encoding="utf-8")

        # Catch: two small JSON chart series (catch type, reporting status) joined by year.
        series: dict[str, dict[int, float]] = {}
        for dim in ("reporting-status", "catchtype"):
            js = json.loads(get(f"{SAU_API}/{region}/tonnage/{dim}/?region_id={rid}", timeout=300))
            for d in js["data"]:
                series[d["key"]] = {int(y): v for y, v in d["values"]}
        years = sorted({y for s in series.values() for y in s})
        lines = ["year,landings,discards,reported,unreported"]
        for y in years:
            vals = [js_num(series.get(k, {}).get(y)) for k in ("Landings", "Discards", "Reported", "Unreported")]
            lines.append(",".join([str(y)] + vals))
        out = root / "sau_regions" / "catch" / f"{slug}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 2) UN SDG Global Database, indicator 14.4.1 -----------------------------
    print("UN SDG 14.4.1 (ER_H2O_FWTL)")
    keep = {"World", "Malaysia", "Indonesia", "Thailand", "Viet Nam", "Philippines"}
    rows = [d for d in json.loads(get(UN_SDG_URL))["data"]
            if d["geoAreaName"] in keep or d["geoAreaName"].startswith("FAO Major Fishing Area")]
    rows.sort(key=lambda d: (d["geoAreaName"], float(d["timePeriodStart"])))
    lines = ["geo_area_code,geo_area_name,year,value_pct_sustainable,observation_status,nature,source"]
    for d in rows:
        a = d["attributes"]
        lines.append(",".join([str(d["geoAreaCode"]), q(d["geoAreaName"]), js_num(d["timePeriodStart"]),
                               str(d["value"]), q(a.get("Observation Status")), q(a.get("Nature")), q(d["source"])]))
    out = root / "un_sdg" / "14_4_1_ER_H2O_FWTL.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 3) World Bank WDI API, capture fisheries production ---------------------
    print("World Bank ER.FSH.CAPT.MT")
    data = json.loads(get(WB_URL))[1]
    data.sort(key=lambda d: (d["countryiso3code"], int(d["date"])))
    lines = ["country,iso3,year,capture_fisheries_production_t"]
    for d in data:
        lines.append(",".join([q(d["country"]["value"]), d["countryiso3code"], d["date"], js_num(d["value"])]))
    out = root / "world_bank" / "ER.FSH.CAPT.MT.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Data-integrity sources  ->  data/raw/validation/{economic,comtrade,benford}/
# --------------------------------------------------------------------------- #
OWID_SDG1471 = OWID_BASE.format(slug="sustainable-fisheries-as-a-proportion-of-gdp")
WB_ALL = ("https://api.worldbank.org/v2/country/all/indicator/{ind}"
          "?format=json&per_page=20000&date=2011:2023")
COMTRADE = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
COMTRADE_M49 = {458: "Malaysia", 360: "Indonesia", 764: "Thailand", 704: "Vietnam", 608: "Philippines"}
SEA_ISO3 = ["MYS", "IDN", "THA", "VNM", "PHL"]


def download_integrity() -> None:
    """
    1) economic/fisheries_gdp_capture.csv
       SDG 14.7.1 (sustainable fisheries, % of GDP; OWID copy, ISO3-coded) + World Bank
       GDP (current USD) + World Bank capture production (t). SEA5 and World for every
       SDG year (2011-2023, biennial); all other countries for 2021 and 2023 (peer group).
    2) comtrade/fish_hs03_mirror_trade.csv
       UN Comtrade, HS chapter 03 (fish): exports each SEA5 country reported vs the sum
       of imports from it reported by all partner countries ("mirror" statistics).
       The public preview API is slow (~15 s/request) and rate-limited.
    3) benford/sau_catch_leading_digits.csv
       Leading-digit counts (1-9) of every taxon-level catch value in the full SAU
       EEZ catch export, split into Reported and Unreported (reconstructed) catch.
    """
    root = RAW_DIR / "validation"

    # 1) economic table -----------------------------------------------------
    print("OWID SDG 14.7.1 + World Bank GDP and capture")
    sdg = {}
    lines = get(OWID_SDG1471).decode("utf-8").strip().split("\n")
    for row in csv.reader(lines[1:]):
        if row[1]:
            sdg[(row[1], int(row[2]))] = (row[0], row[3])

    def wb(ind: str) -> dict:
        data = json.loads(get(WB_ALL.format(ind=ind)))[1]
        return {(d["countryiso3code"], int(d["date"])): d["value"] for d in data if d["value"] is not None}

    gdp, cap = wb("NY.GDP.MKTP.CD"), wb("ER.FSH.CAPT.MT")
    rows = []

    def add(iso: str, sdg_code: str, year: int) -> None:
        s = sdg.get((sdg_code, year))
        if s and (iso, year) in gdp and (iso, year) in cap:
            rows.append((iso, s[0], year, s[1], js_num(gdp[(iso, year)]), js_num(cap[(iso, year)])))

    years = [2011, 2013, 2015, 2017, 2019, 2021, 2023]
    for iso in SEA_ISO3:
        for y in years:
            add(iso, iso, y)
    for y in years:
        add("WLD", "OWID_WRL", y)
    for iso, y in sdg:
        if not iso.startswith("OWID") and iso not in SEA_ISO3 and y in (2021, 2023):
            add(iso, iso, y)
    rows.sort(key=lambda r: (r[0], r[2]))
    out = root / "economic" / "fisheries_gdp_capture.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("iso3,country,year,sustainable_fisheries_pct_gdp,gdp_current_usd,capture_t\n"
                   + "".join(",".join([r[0], q(r[1]), str(r[2]), r[3], r[4], r[5]]) + "\n" for r in rows),
                   encoding="utf-8")

    # 2) Comtrade mirror trade -------------------------------------------------
    def is_total(d: dict) -> bool:
        return d["partner2Code"] == 0 and d["motCode"] == 0 and d["customsCode"] == "C00"

    lines = ["country,m49,year,reported_exports_usd,partner_imports_usd,n_partners"]
    for code, name in COMTRADE_M49.items():
        for y in range(2011, 2024):
            print(f"Comtrade {name} {y}")
            exp = [d for d in json.loads(get(f"{COMTRADE}?reporterCode={code}&period={y}&cmdCode=03"
                                             f"&flowCode=X&partnerCode=0", timeout=120))["data"] if is_total(d)]
            imp = [d for d in json.loads(get(f"{COMTRADE}?partnerCode={code}&period={y}&cmdCode=03"
                                             f"&flowCode=M", timeout=120))["data"]
                   if is_total(d) and d["partnerCode"] == code and d["reporterCode"] != code]
            e = sum(d["primaryValue"] for d in exp)
            m = sum(d["primaryValue"] for d in imp)
            lines.append(f"{name},{code},{y},{e:.0f},{m:.0f},{len(imp)}")
            time.sleep(1.2)
    out = root / "comtrade" / "fish_hs03_mirror_trade.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 3) Benford leading-digit counts from the full SAU catch export -----------
    lines = ["eez_id,country,reporting_status,d1,d2,d3,d4,d5,d6,d7,d8,d9"]
    country = {459: "Malaysia", 460: "Malaysia", 461: "Malaysia", 463: "Malaysia", 361: "Indonesia",
               937: "Indonesia", 938: "Indonesia", 956: "Thailand", 957: "Thailand", 704: "Vietnam", 608: "Philippines"}
    for eez_id in sorted(SAU_EEZS):
        print(f"SAU   benford {SAU_EEZS[eez_id]}")
        text = first_csv_in_zip(get(SAU_MEASURES["catch_summary"].format(id=eez_id), timeout=300))
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
        it, ir = header.index("tonnes"), header.index("reporting_status")
        counts = {"Reported": [0] * 9, "Unreported": [0] * 9}
        for row in reader:
            if not row or row[ir] not in counts:
                continue
            try:
                v = float(row[it])
            except ValueError:
                continue
            if v > 0:
                counts[row[ir]][int(f"{v:e}"[0]) - 1] += 1
        for st in ("Reported", "Unreported"):
            lines.append(",".join([str(eez_id), country[eez_id], st] + [str(n) for n in counts[st]]))
    out = root / "benford" / "sau_catch_leading_digits.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


IMAS_EFFORT = "https://data.imas.utas.edu.au/attachments/1241a51d-c8c2-4432-aa68-3d2bae142794/"
EFFORT_EEZS = {"Malaysia": "Malaysia", "Indonesia": "Indonesia", "Thailand": "Thailand",
               "Viet Nam": "Viet Nam", "Philippines": "Philippines"}   # EEZ names as written in the IMAS file


def sig6(x: float) -> str:
    """6 significant digits in plain (non-scientific) notation, e.g. 4577500000 or 59007.2."""
    r = float(f"{x:.6g}")
    return str(int(r)) if r == int(r) else repr(r)


def download_effort() -> None:
    """
    Stream two aggregated IMAS files and sum them by year:
    1) TotalEffortby_FGroup_FishingCountry_Sector.csv (42 MB) -> world_effort.csv
       (all fishing countries = world fishing effort, incl. high seas)
    2) TotalEffortby_FGroup_EEZ_Sector.csv (69 MB) -> sea_eez_effort.csv
       (effort by all fleets inside each SEA5 EEZ, matching the SAU EEZ catch data)
    Columns kept: nominal and effective effort (kW x days at sea), nominal effort of
    the industrial sector, number of active vessels and engine power (kW).
    """
    out = RAW_DIR / "validation" / "effort"
    out.mkdir(parents=True, exist_ok=True)

    def stream(name: str, key_col: str, keep) -> dict:
        totals: dict = {}
        with requests.get(IMAS_EFFORT + name, headers=HEADERS, stream=True, timeout=600) as r:
            r.raise_for_status()
            reader = csv.DictReader(line.decode("utf-8") for line in r.iter_lines() if line)
            for row in reader:
                if row["Year"] == "NA" or not keep(row[key_col]):
                    continue
                k = (row[key_col], int(row["Year"]))
                t = totals.setdefault(k, [0.0] * 5)
                nom = float(row["NomActive"])
                t[0] += nom
                t[1] += float(row["EffActive"])
                t[2] += nom if row["Sector"] == "I" else 0.0
                t[3] += float(row["NV"])
                t[4] += float(row["P"])
        return totals

    print("IMAS effort by fishing country -> world totals")
    world: dict = {}
    for (_, year), t in stream("TotalEffortby_FGroup_FishingCountry_Sector.csv", "Country", lambda c: True).items():
        w = world.setdefault(year, [0.0] * 5)
        for i in range(5):
            w[i] += t[i]
    lines = ["year,nom_kwdays,eff_kwdays,nom_kwdays_industrial,vessels,power_kw"]
    lines += [",".join([str(y)] + [sig6(v) for v in world[y]]) for y in sorted(world)]
    (out / "world_effort.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("IMAS effort by EEZ -> SEA5 EEZs")
    sea = stream("TotalEffortby_FGroup_EEZ_Sector.csv", "EEZ", lambda e: e in EFFORT_EEZS)
    lines = ["eez,year,nom_kwdays,eff_kwdays,nom_kwdays_industrial,vessels,power_kw"]
    lines += [",".join([z, str(y)] + [sig6(v) for v in sea[(z, y)]]) for z, y in sorted(sea)]
    (out / "sea_eez_effort.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


ERSST_ASCII = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc.ascii?"
ERSST_T0 = 1152            # time index of 1950-01 (the file starts at 1854-01)
ERSST_BOX = (33, 51, 46, 71)   # lat index 33..51 = 22N..14S, lon index 46..71 = 92E..142E (2-degree grid)
OWID_CLIMATE = ["sea-surface-temperature-anomaly", "ocean-heat-content-upper"]


def download_climate() -> None:
    """
    1) OWID World SST anomaly and upper-ocean heat content (World rows only).
    2) EEZ weights: every 0.5-degree IMAS cell of a SEA5 EEZ is assigned to the
       2-degree ERSST cell containing it; weight = summed cell area (km2).
    3) NOAA ERSST v5 monthly SST for the SEA box, 1950-01 to the last complete year,
       averaged per EEZ with those area weights (land cells skipped), then per year:
       mean SST, anomaly vs the 1971-2000 mean, and 'hot months' = months whose
       anomaly vs the 1971-2000 same-month mean exceeds that month's 90th percentile.
    """
    for slug in OWID_CLIMATE:
        lines = get(OWID_BASE.format(slug=slug)).decode("utf-8").strip().split("\n")
        keep = [lines[0]] + [ln for ln in lines[1:] if ln.startswith("World,")]
        (RAW_DIR / "owid" / f"{slug}.csv").write_text("\n".join(keep) + "\n", encoding="utf-8")

    out = RAW_DIR / "validation" / "climate"
    out.mkdir(parents=True, exist_ok=True)
    cells = get(IMAS_EFFORT + "CellstoLME_EEZ.csv").decode("utf-8").splitlines()
    weights: dict = {}
    for row in csv.reader(cells[1:]):
        if len(row) < 6 or row[5] not in EFFORT_EEZS:
            continue
        lon, lat, area = float(row[1]), float(row[2]), float(row[3])
        glon = round(lon / 2) * 2 % 360
        glat = round(lat / 2) * 2
        k = (row[5], glat, glon)
        weights[k] = weights.get(k, 0.0) + area
    lines = ["eez,lat,lon,area_km2"] + [f"{e},{la},{lo},{round(a)}" for (e, la, lo), a in
                                        sorted(weights.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), str(kv[0][2])))]
    (out / "eez_ersst_cell_weights.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    la0, la1, lo0, lo1 = ERSST_BOX
    t_last = len(re.findall(r"[\d.]+", get(ERSST_ASCII + "time").decode().split("time[")[-1].split("\n", 1)[1])) - 1
    n_years = (t_last - ERSST_T0 + 1) // 12
    grid: dict = {}
    for s in range(ERSST_T0, ERSST_T0 + 12 * n_years, 120):
        e = min(ERSST_T0 + 12 * n_years - 1, s + 119)
        text = get(ERSST_ASCII + f"sst[{s}:1:{e}][{la0}:1:{la1}][{lo0}:1:{lo1}]").decode()
        for m in re.finditer(r"^\[(\d+)\]\[(\d+)\], (.*)$", text, re.M):
            grid[(s + int(m.group(1)) - ERSST_T0, int(m.group(2)))] = [float(x) for x in m.group(3).split(",")]

    def monthly(cell_list) -> list[float]:
        series = []
        for k in range(12 * n_years):
            sw = sv = 0.0
            for la, lo, a in cell_list:
                val = grid[(k, (88 - la) // 2 - la0)][lo // 2 - lo0]
                if val > -100:          # land / missing = -9.97e36
                    sw, sv = sw + a, sv + a * val
            series.append(sv / sw)
        return series

    by_eez: dict = {}
    for (e, la, lo), a in weights.items():
        by_eez.setdefault(e, []).append((la, lo, a))
    regions = {"SEA": [c for v in by_eez.values() for c in v], **by_eez}
    rows = ["region,year,sst_mean_c,sst_anom_c,hot_months"]
    for reg in ["SEA", "Indonesia", "Malaysia", "Philippines", "Thailand", "Viet Nam"]:
        a = monthly(regions[reg])
        clim, p90 = [], []
        for m in range(12):
            base = [a[(y - 1950) * 12 + m] for y in range(1971, 2001)]
            clim.append(sum(base) / len(base))
            an = sorted(x - clim[m] for x in base)
            p90.append(an[int(0.9 * (len(an) - 1))])
        ann_base = sum(clim) / 12
        for i in range(n_years):
            vals = a[12 * i: 12 * i + 12]
            mean = sum(vals) / 12
            hot = sum(v - clim[m] > p90[m] for m, v in enumerate(vals))
            rows.append(f"{reg},{1950 + i},{mean:.3f},{mean - ann_base:.3f},{hot}")
    (out / "sea_eez_sst_ersst.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


WB_API = "https://api.worldbank.org/v2"
WB_ISO = "WLD;MYS;IDN;THA;VNM;PHL"
UNCTAD_FACTS = "https://unctadstat-api.unctad.org/datamart-api/US.SeaborneTrade/10003/Facts"
GDAT_ZIP = "https://zenodo.org/api/records/7616852/files/GDAT_data_v1.zip/content"
DEEPOWT = "https://zenodo.org/api/records/5933967/files/DeepOWT.geojson/content"
OOGP_ZIP = "https://zenodo.org/api/records/18350974/files/OOGPs_v1.0.0.zip/content"


def _num(x: float) -> str:
    """Write numbers as JavaScript does (3797948 not 3797948.0), matching the committed files."""
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def _round1(x: float) -> str:
    return _num(int(x * 10 + 0.5) / 10)        # Math.round(x*10)/10, half away from zero for x ≥ 0


def _read_dbf(raw: bytes, columns: list[str]) -> list[dict]:
    """Minimal dBASE III reader: returns the requested character/numeric columns as stripped strings."""
    n, hlen, rlen = int.from_bytes(raw[4:8], "little"), int.from_bytes(raw[8:10], "little"), \
        int.from_bytes(raw[10:12], "little")
    fields, p, off = {}, 32, 1
    while raw[p] != 0x0D:
        name = raw[p:p + 11].split(b"\0")[0].decode()
        size = raw[p + 16]
        fields[name] = (off, size)
        off += size
        p += 32
    rows = []
    for i in range(n):
        base = hlen + i * rlen
        if raw[base] == 0x2A:                     # deleted record
            continue
        rows.append({c: raw[base + fields[c][0]: base + fields[c][0] + fields[c][1]].decode("utf-8", "replace").strip()
                     for c in columns})
    return rows


def download_human_use() -> None:
    """
    Iteration 4 drivers -> data/raw/validation/human_use/
      wb_shipping.csv             WDI container port traffic (IS.SHP.GOOD.TU) and liner shipping
                                  connectivity (IS.SHP.GCNW.XQ): current WDI + WDI Dec-2020 archive
      unctad_seaborne_trade.csv   UNCTADstat seaborne trade, goods loaded/discharged (thousand t)
      gdat_dams_by_year.csv       Global Dam Tracker v1: dams and reported reservoir volume by
                                  completion year (exact / decade-only / undated), World + SEA5
      deepowt_offshore_wind.csv   DeepOWT turbine counts by quarter: World and Vietnam (102-108E, 8-21.5N)
      oogp_platforms.csv          OOGP platforms detected / installed / removed per year, 6 basins + SEA
    """
    out = RAW_DIR / "validation" / "human_use"
    out.mkdir(parents=True, exist_ok=True)

    print("World Bank: container port traffic + liner shipping connectivity")
    rows = []
    for ind in ["IS.SHP.GOOD.TU", "IS.SHP.GCNW.XQ"]:
        cur = json.loads(get(f"{WB_API}/country/{WB_ISO}/indicator/{ind}?format=json&per_page=2000&date=1960:2025"))
        rows += [("WDI_current", x["countryiso3code"], int(x["date"]), ind, x["value"])
                 for x in cur[1] if x["value"] is not None]
        arc = json.loads(get(f"{WB_API}/sources/57/country/{WB_ISO}/series/{ind}/version/202012/time/all"
                             "?format=json&per_page=2000"))
        for d in arc["source"]["data"]:
            if d["value"] is None:
                continue
            var = {q["concept"]: q["id"] for q in d["variable"]}
            rows.append(("WDI_archive_202012", var["Country"], int(var["Time"][2:]), ind, d["value"]))
    rows.sort(key=lambda r: (r[3] + r[0] + r[1], r[2]))
    (out / "wb_shipping.csv").write_text("source,iso3,year,indicator,value\n" + "".join(
        f"{s},{c},{y},{i},{_num(val)}\n" for s, c, y, i, val in rows), encoding="utf-8")

    print("UNCTADstat: seaborne trade")
    q = ("$select=Economy/Code,CargoType/Code,Year,M2200/Value&$filter=Economy/Code in "
         "('0000','360','960','458','608','764','704') and CargoType/Code in ('10','20')"
         "&$orderby=Economy/Code,CargoType/Code,Year&$format=csv&compress=false")
    text = get(UNCTAD_FACTS + "?" + q.replace(" ", "%20").replace("'", "%27")).decode("utf-8")
    iso = {"0000": "WLD", "360": "IDN", "960": "IDN", "458": "MYS", "608": "PHL", "764": "THA", "704": "VNM"}
    flow = {"10": "loaded", "20": "discharged"}
    lines = ["iso3,year,flow,thousand_tonnes"]
    for ln in text.strip().split("\n")[1:]:
        e, c, y, val = ln.split(",")
        if val != "":
            lines.append(f"{iso[e]},{y},{flow[c]},{int(float(val) / 1000 + 0.5)}")
    (out / "unctad_seaborne_trade.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("GDAT v1 dams (Zenodo, ~14 MB zip)")
    z = zipfile.ZipFile(io.BytesIO(get(GDAT_ZIP)))
    dams = _read_dbf(z.read("GDAT_data_v1/data/GDAT_v1_dams.dbf"), ["Admin0", "Year_Fin", "Volume_Rep"])
    sea = {"Indonesia": "IDN", "Malaysia": "MYS", "Philippines": "PHL", "Thailand": "THA", "Vietnam": "VNM"}
    agg: dict = {}
    for r in dams:
        y = r["Year_Fin"]
        yr, prec = (int(y), "exact") if re.fullmatch(r"\d{4}", y) else \
            ((int(y) * 10, "decade") if re.fullmatch(r"\d{3}", y) else ("", "undated"))
        try:
            vol = float(r["Volume_Rep"]) if r["Volume_Rep"] and not r["Volume_Rep"].startswith("*") else None
        except ValueError:
            vol = None
        for reg in ["WLD", sea.get(r["Admin0"])]:
            if reg:
                a = agg.setdefault((reg, yr, prec), [0, 0, 0.0])
                a[0] += 1
                if vol is not None:
                    a[1] += 1
                    a[2] += vol
    keys = sorted(agg, key=lambda k: (k[0], 1e9 if k[1] == "" else k[1], k[2]))
    (out / "gdat_dams_by_year.csv").write_text("region,year,year_precision,n_dams,n_with_volume,volume_mcm\n" + "".join(
        f"{r},{y},{p},{agg[(r, y, p)][0]},{agg[(r, y, p)][1]},{_round1(agg[(r, y, p)][2])}\n" for r, y, p in keys),
        encoding="utf-8")

    print("DeepOWT offshore wind turbines (Zenodo)")
    feats = json.loads(get(DEEPOWT))["features"]
    quarters = list(feats[0]["properties"])
    lines = ["region,year,quarter,n_class1,n_class2,n_class3"]
    for reg in ["WLD", "VNM"]:
        for k in quarters:
            n = [0, 0, 0]
            for f in feats:
                lon, lat = f["geometry"]["coordinates"][:2]
                if reg == "VNM" and not (102 <= lon < 108 and 8 <= lat <= 21.5):
                    continue
                c = f["properties"][k]
                if c >= 1:
                    n[c - 1] += 1
            lines.append(f"{reg},{k[1:5]},{k[5:]},{n[0]},{n[1]},{n[2]}")
    (out / "deepowt_offshore_wind.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("OOGP offshore oil & gas platforms (Zenodo GeoPackage)")
    import sqlite3
    import tempfile
    z = zipfile.ZipFile(io.BytesIO(get(OOGP_ZIP)))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "oogp.gpkg"
        path.write_bytes(z.read("OOGPs_all_v1.0.0.gpkg"))
        con = sqlite3.connect(path)
        plat = con.execute("SELECT Country, Installation_date, Removal_date, Year_label FROM platforms").fetchall()
        con.close()
    iso2 = {"Malaysia": "MYS", "Thailand": "THA", "Vietnam": "VNM", "Indonesia": "IDN", "Cambodia": "KHM"}
    agg = {}
    for country, inst, rem, label in plat:
        for reg in ["ALL6", iso2.get(country)]:
            if not reg:
                continue
            for y in [s.strip() for s in (label or "").split(",") if s.strip()]:
                agg.setdefault((reg, int(y)), [0, 0, 0])[0] += 1
            if inst:
                agg.setdefault((reg, int(inst[:4])), [0, 0, 0])[1] += 1
            if rem:
                agg.setdefault((reg, int(rem[:4])), [0, 0, 0])[2] += 1
    (out / "oogp_platforms.csv").write_text("region,year,n_platforms_detected,n_installed,n_removed\n" + "".join(
        f"{r},{y},{a[0]},{a[1]},{a[2]}\n" for (r, y), a in sorted(agg.items())), encoding="utf-8")


def write_manifest(parts: list[str]) -> None:
    manifest = {
        "retrieved": date.today().isoformat(),
        "parts": parts,
        "owid_entities_kept": OWID_KEEP_ENTITIES,
        "sau_eezs": {str(k): v for k, v in SAU_EEZS.items()},
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--only", choices=["owid", "sau", "validation", "integrity", "effort", "climate", "human_use"],
                        help="download one part only")
    args = parser.parse_args()

    parts = [args.only] if args.only else ["owid", "sau", "validation", "integrity", "effort", "climate", "human_use"]
    if "owid" in parts:
        download_owid()
    if "sau" in parts:
        download_sau()
    if "validation" in parts:
        download_validation()
    if "integrity" in parts:
        download_integrity()
    if "effort" in parts:
        download_effort()
    if "climate" in parts:
        download_climate()
    if "human_use" in parts:
        download_human_use()
    write_manifest(parts)
    print("Done ->", RAW_DIR)


if __name__ == "__main__":
    main()
