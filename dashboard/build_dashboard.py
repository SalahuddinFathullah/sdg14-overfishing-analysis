"""
build_dashboard.py — Standalone interactive results dashboard
=============================================================

KIE4033 Assignment 1 · SDG 14 overfishing analysis (Global -> Southeast Asia)

Builds dashboard/index.html: one self-contained HTML page (no server, no internet,
no JavaScript libraries) that shows

    1. verdict summary — how many hypothesis tests reject H0, are mixed/weak, or fail to reject
    2. analysis map — figure O.01
    3. verdict table — every hypothesis; click a row for its question, evidence and note,
       and to filter the figure gallery to that hypothesis
    4. forecast to 2030 (H5d) — observed series, chosen model's forecast and 95% interval
       for each of the 8 series, plus the back-test table
    5. figure gallery — all 68 figures, filterable by report section, pipeline step,
       hypothesis, or main-body figures only

The script only READS result tables written by the other folders, so run it last:

    python 2-hypothesis-tests/hypothesis_tests.py
    python 4-overview/overview.py          # writes 4-overview/figure_index.csv
    python dashboard/build_dashboard.py    # this script

Then open dashboard/index.html in a browser (double-click). Images are linked with
relative paths to the PNGs already in the repo, so the page must stay inside the repo
folder; the data tables are embedded in the page itself.

Inputs:
    2-hypothesis-tests/results/decisions.csv                  verdict rows
    2-hypothesis-tests/results/11_I5_H5d_forecast_2030.csv    chosen model + 2030 forecast per series
    2-hypothesis-tests/results/11_I5_H5d_forecast_paths.csv   yearly forecast path + 95% interval
    4-overview/figure_index.csv                               figure catalogue
    observed % overexploited series — rebuilt with the same loader functions that H5d uses
    (hypothesis_tests.py), so the chart shows exactly the data the forecast was fitted on
Template:
    dashboard/template.html   (page layout, styles and drawing code; data goes in at /*__DATA__*/)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "2-hypothesis-tests" / "results"
TEMPLATE = HERE / "template.html"
OUTPUT = HERE / "index.html"
PLACEHOLDER = "/*__DATA__*/null"


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------
def short_code(label: str) -> str:
    """Short id for a decisions.csv row, e.g. 'H2b (effort)', 'CPUE', 'Rule 1'."""
    if label.startswith("Explanation"):
        return "CPUE"
    if label.startswith("Stage rule"):
        return "Rule 1" if "catch" in label else "Rule 2"
    m = re.match(r"(H\d[a-d]?)", label)
    code = m.group(1) if m else label
    return code + (" (effort)" if "effort" in label else "")


def iteration_of(label: str) -> int:
    """Iteration number written in the label; iteration 1 rows carry none."""
    m = re.search(r"iteration (\d)", label)
    return int(m.group(1)) if m else 1


def verdict_class(decision: str) -> str:
    """Group the free-text decision into one of six verdict classes (first words decide)."""
    d = decision.lower()
    if d.startswith("reject"):
        return "Reject H0"
    if d.startswith("mixed"):
        return "Mixed"
    if d.startswith("weak"):
        return "Weak"
    if d.startswith("fail"):
        return "Fail to reject H0"
    if "rule not met" in d:
        return "Rule not met"
    return "Descriptive"


def load_verdicts() -> list[dict]:
    d = pd.read_csv(RESULTS / "decisions.csv")
    out = pd.DataFrame({
        "code": d["hypothesis"].map(short_code),
        "iteration": d["hypothesis"].map(iteration_of),
        "test": d["hypothesis"],
        "question": d["question"],
        "verdict": d["decision"].map(verdict_class),
        "decision": d["decision"],
        "evidence": d["evidence"],
        "note": d["note"],
    })
    return out.fillna("").to_dict("records")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def load_figures() -> list[dict]:
    f = pd.read_csv(ROOT / "4-overview" / "figure_index.csv").fillna("")
    f["iteration"] = f["iteration"].astype(str)
    # Path from dashboard/ to the PNG; flag any figure whose file is missing.
    f["src"] = "../" + f["file"]
    missing = [p for p in f["file"] if not (ROOT / p).exists()]
    if missing:
        print(f"  warning: {len(missing)} figure file(s) not found, e.g. {missing[0]}")
    return f.to_dict("records")


# ---------------------------------------------------------------------------
# Forecast (H5d)
# ---------------------------------------------------------------------------
def load_forecast() -> tuple[list[dict], list[dict]]:
    """Observed + forecast rows per series, and the per-series summary table."""
    sys.path.insert(0, str(ROOT / "2-hypothesis-tests"))
    import hypothesis_tests as ht  # noqa: E402  (its own module sets up the data paths)

    sea, cs = ht.sea_series(), ht.country_series()
    units = [("World (FAO)", ht.st.world_fao()), ("World (SAU)", ht.st.world_sau()), ("SEA (SAU)", sea["over"])]
    units += [(f"{c} (SAU)", cs[c]["over"]) for c in ht.SEA5]

    summary = pd.read_csv(RESULTS / "11_I5_H5d_forecast_2030.csv")
    paths = pd.read_csv(RESULTS / "11_I5_H5d_forecast_paths.csv")

    rows: list[dict] = []
    for unit, s in units:
        s = s.dropna()
        s = s[s.index >= ht.FORECAST_FROM]
        rows += [{"unit": unit, "year": int(y), "kind": "Observed", "value": round(float(v), 3),
                  "lower": None, "upper": None} for y, v in s.items()]
    for _, r in summary.iterrows():
        p = paths[paths["unit"] == r["unit"]]
        if p["model"].nunique() > 1:          # paths file may hold more than the chosen model
            p = p[p["model"] == r["chosen_model"]]
        rows += [{"unit": r["unit"], "year": int(x["year"]), "kind": "Forecast", "value": round(x["mean"], 3),
                  "lower": round(x["lower"], 3), "upper": round(x["upper"], 3)} for _, x in p.iterrows()]

    keep = ["unit", "chosen_model", "backtest_mae_pp", "naive_mae_pp", "last_year", "last_value",
            "forecast_2030", "pi95_low_2030", "pi95_high_2030", "beats_naive_in_backtest", "rise_credible"]
    return rows, summary[keep].round(4).to_dict("records")


# ---------------------------------------------------------------------------
def main() -> None:
    forecast, forecast_summary = load_forecast()
    data = {
        "verdicts": load_verdicts(),
        "figures": load_figures(),
        "forecast": forecast,
        "forecastSummary": forecast_summary,
    }
    # "</" is escaped so text inside the data can never close the <script> tag early.
    payload = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise SystemExit(f"placeholder {PLACEHOLDER!r} not found in {TEMPLATE}")
    OUTPUT.write_text(html.replace(PLACEHOLDER, payload), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {len(data['verdicts'])} verdict rows, "
          f"{len(data['figures'])} figures, {len(forecast_summary)} forecast series")


if __name__ == "__main__":
    main()
