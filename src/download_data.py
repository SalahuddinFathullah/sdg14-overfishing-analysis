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

Usage
-----
    python src/download_data.py                   # download everything
    python src/download_data.py --only owid
    python src/download_data.py --only sau
    python src/download_data.py --only validation
    python src/download_data.py --only integrity    # SDG 14.7.1/GDP, Comtrade, Benford (slow)
    python src/download_data.py --only effort       # streams ~110 MB of IMAS CSVs

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
    parser.add_argument("--only", choices=["owid", "sau", "validation", "integrity", "effort"],
                        help="download one part only")
    args = parser.parse_args()

    parts = [args.only] if args.only else ["owid", "sau", "validation", "integrity", "effort"]
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
    write_manifest(parts)
    print("Done ->", RAW_DIR)


if __name__ == "__main__":
    main()
