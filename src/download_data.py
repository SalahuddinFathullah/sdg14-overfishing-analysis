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

Usage
-----
    python src/download_data.py            # download everything
    python src/download_data.py --only owid
    python src/download_data.py --only sau

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


def summarise_catch(text: str) -> str:
    """
    Aggregate the full SAU catch export (one row per taxon x fleet x gear x ...)
    to one row per year x fishing_sector x catch_type x reporting_status x fleet.

    fleet = "domestic" if the fishing country owns the EEZ, else "foreign".
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
        fleet = "domestic" if row[col["fishing_entity"]] == row[col["area_name"]] else "foreign"
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
    parser.add_argument("--only", choices=["owid", "sau"], help="download one source only")
    args = parser.parse_args()

    parts = [args.only] if args.only else ["owid", "sau"]
    if "owid" in parts:
        download_owid()
    if "sau" in parts:
        download_sau()
    write_manifest(parts)
    print("Done ->", RAW_DIR)


if __name__ == "__main__":
    main()
