"""
overview.py — Report-level visual summaries of the whole analysis
==================================================================

KIE4033 Assignment 1 · SDG 14 overfishing analysis (Global -> Southeast Asia)

Reads only the result tables written by the other folders (it recomputes nothing),
so run it last (``python run_all.py`` runs every step in order, data quality first):

    python 0-data-quality/clean_data.py && python 0-data-quality/quality_tests.py
    python 1-visualisation/visualise.py
    python 2-statistical-tests/stat_tests.py
    python 3-hypothesis-tests/hypothesis_tests.py
    python 4-overview/overview.py              # this script
    python 4-overview/overview.py --only map

Figures (figures/*.png):
    O.01  analysis map — data → quality checks & preparation → EDA → 6 iterations of tests → answers,
          with every hypothesis verdict taken from 3-hypothesis-tests/results/decisions.csv
    O.02  stationarity of every series tested (ADF + KPSS, before and after transforming)
    O.03  assumption checks and the test choices they drove (regression residual
          diagnostics; parametric vs non-parametric p-values)
    O.04  long-run tests across iterations (Engle-Granger cointegration, Granger causality)

Table:
    figure_index.csv  every figure in the repo: id, title, rubric section, hypotheses, iteration,
                      one-line message, file, and its role in the report (main-body Fig N,
                      appendix or gallery) — the report's figure catalogue

The figures exist so that tests which were reported only as tables (stationarity,
diagnostics, cointegration, Granger) can also be read at a glance.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
import style as v  # noqa: E402  (chart style + header helpers, shared by every step)

FIG = HERE / "figures"
HT = ROOT / "3-hypothesis-tests" / "results"
STT = ROOT / "2-statistical-tests" / "results"
ALPHA = 0.05

# Verdict classes used on the map (one colour each; text always in ink)
REJECT, MIXED, FAIL = "Reject H₀ / confirmed", "Mixed or weak", "Fail to reject H₀"
VERDICT_STYLE = {REJECT: dict(fc=v.SERIES[0], ec=v.SERIES[0], tc="#ffffff"),
                 MIXED: dict(fc="#cde2fb", ec="#cde2fb", tc=v.INK),
                 FAIL: dict(fc=v.SURFACE, ec=v.INK_2, tc=v.INK)}


def save(fig, name: str, source: str) -> None:
    fig.text(0.01, -0.01, f"Source: {source}", ha="left", va="top", fontsize=8, color=v.MUTED)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("  saved", (FIG / f"{name}.png").relative_to(ROOT))


def verdict_class(decision: str) -> str:
    """Map a decisions.csv verdict onto the three map classes."""
    d = decision.lower()
    if d.startswith(("weak", "mixed")) or "reject for" in d or "falling in" in d:
        return MIXED
    if d.startswith("reject") or d == "cpue falling":
        return REJECT
    return FAIL


# =========================================================================== #
# O.01 — analysis map
# =========================================================================== #
def box(ax, x, y, w, h, title, body, fc="#f0efec", ec=None, title_size=9.5, body_size=8.2, wrap=None):
    """Rounded box with a bold title line and wrapped body text, top-left aligned."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc, ec=ec or fc, lw=1.2, zorder=1))
    ax.text(x + 1.2, y + h - 1.4, title, ha="left", va="top", fontsize=title_size, fontweight="bold",
            color=v.INK, zorder=3)
    if body:
        wrapped = "\n".join(textwrap.fill(line, wrap or int(w * 1.55)) for line in body.split("\n"))
        ax.text(x + 1.2, y + h - 4.6, wrapped, ha="left", va="top", fontsize=body_size, color=v.INK_2,
                linespacing=1.3, zorder=3)


def arrow(ax, x0, y0, x1, y1, rad=0.0, color=None, lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, lw=lw,
                                 color=color or v.AXIS, connectionstyle=f"arc3,rad={rad}", zorder=2))


def chip(ax, x, y, label, cls, w=7.6, h=3.4):
    s = VERDICT_STYLE[cls]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=s["fc"], ec=s["ec"], lw=1.2, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.6, fontweight="bold",
            color=s["tc"], zorder=4)


def section_map() -> None:
    print("O.01 analysis map")
    dec = pd.read_csv(HT / "decisions.csv")
    cls = {}
    for _, r in dec.iterrows():
        h = r["hypothesis"]
        key = ("CPUE" if h.startswith("Explanation") else
               h.split(" (")[0] + ("·e" if "iteration 2" in h else ""))
        cls[key] = verdict_class(r["decision"])

    fig = plt.figure(figsize=(19, 10.6))
    ax = fig.add_axes([0.005, 0.02, 0.99, 0.86])
    ax.set_xlim(0, 190)
    ax.set_ylim(0, 100)
    ax.axis("off")

    cols = [(2, "1 · Data"), (36, "2 · Check & clean first"), (68, "3 · Explore"), (100, "4 · Test (7 iterations)"),
            (158, "5 · Answer")]
    for x, t in cols:
        ax.text(x, 99, t, ha="left", va="top", fontsize=11.5, fontweight="bold", color=v.INK)

    # 1 · Data
    data = [("FAO via OWID", "% stocks overexploited (world 1974–2021, FAO areas), capture, aquaculture"),
            ("Sea Around Us", "Stock status, mean trophic level, catch by sector and fleet; 11 SEA EEZs, 1950–2019"),
            ("Validation & integrity", "UN SDG 14.4.1, World Bank, SAU regions, UN Comtrade, GDP, Benford digits"),
            ("Driver data (iterations 2–7)", "Effort (IMAS), warming (HadSST, ERSST), shipping (WDI, UNCTAD), "
                                             "dams (GDAT), platforms (OOGP), wind (DeepOWT); pollution: fertiliser, "
                                             "pesticides, plastic, oil spills, pH, chlorophyll-a (OWID)")]
    ys = [78, 60, 42, 16]
    hs = [14, 14, 14, 22]
    for (t, b), y, h in zip(data, ys, hs):
        box(ax, 2, y, 30, h, t, b, wrap=44)

    # 2 · Check & clean first (step 0: a failed check stops every later step)
    prep = [("Check & clean (gate)", "All 90 input files checked (parse, duplicates, ranges); rules R1–R6 → "
                                     "data/processed. Any failure stops the pipeline"),
            ("Validate", "21 source pairs: 7 identical, 10 agree on trend, 4 on direction only, none disagree"),
            ("Integrity", "Benford digits genuine; under-reporting 10–45% of catch; Vietnam mirror-trade gap")]
    for (t, b), y in zip(prep, [76, 56, 36]):
        box(ax, 36, y, 28, 16, t, b, wrap=40)

    # 3 · Explore
    box(ax, 68, 66, 28, 26, "EDA (34 figures)",
        "Trends, ACF (levels vs differences), regional comparisons, EEZ × year heatmap, correlation matrices, "
        "anomaly notes", wrap=40)
    box(ax, 68, 32, 28, 30, "Course methods",
        "Assumption checks (Shapiro, Levene) → t-test/ANOVA vs Wilcoxon/Kruskal-Wallis; rank-sum before/after "
        "2015; chi-square + Cramér's V; k-means + Ward clustering", wrap=40)

    # 4 · Test lanes
    lanes = [("Stage 1 · global", "catch", ["H1a", "H1b"]),
             ("Stage 2 · Southeast Asia", "catch", ["H2a", "H2b", "H2c"]),
             ("Iteration 2", "fishing effort", ["H1b·e", "H2b·e", "CPUE"]),
             ("Iteration 3", "ocean warming", ["H3a", "H3b"]),
             ("Iteration 4", "shipping & structures", ["H4a", "H4b", "H4c"]),
             ("Iteration 5", "panel, rebuilding, FiB, 2030", ["H5a", "H5b", "H5c", "H5d"]),
             ("Iteration 6", "drivers in combination", ["H6a", "H6b", "H6c"]),
             ("Iteration 7", "sea & water pollution", ["H7a", "H7b"]),
             ("Iteration 7 (cont.)", "pollution combined, per sea", ["H7c", "H7d", "H7e", "H7f"])]
    lane_h, gap, top = 8.0, 1.0, 92
    lane_y = []
    for i, (t, sub, hyps) in enumerate(lanes):
        y = top - (i + 1) * lane_h - i * gap
        lane_y.append(y)
        ax.add_patch(FancyBboxPatch((100, y), 56, lane_h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="#f0efec", ec="#f0efec", zorder=1))
        ax.text(101.2, y + lane_h - 1.4, t, ha="left", va="top", fontsize=9.5, fontweight="bold", color=v.INK)
        ax.text(101.2, y + lane_h - 5.0, sub, ha="left", va="top", fontsize=8.2, color=v.INK_2)
        for j, hname in enumerate(hyps):
            label = hname.replace("·e", " (effort)") if hname.endswith("·e") else hname
            w = 12.0 if hname.endswith("·e") else 7.6
            x = 120.5 + sum((12.0 if hh.endswith("·e") else 7.6) + 1.2 for hh in hyps[:j])
            chip(ax, x, y + lane_h / 2 - 1.7, label, cls.get(hname, FAIL), w=w)
    ax.text(128, lane_y[0] + lane_h + 0.6, "Stage rule (H1a and H1b rejected) not met → Stage 2 exploratory",
            ha="center", va="bottom", fontsize=7.8, color=v.MUTED)
    # iteration loop: a straight arrow down the right edge of the lanes
    arrow(ax, 157.8, lane_y[2] + lane_h - 0.5, 157.8, lane_y[-1] + 1.0, color=v.INK_2, lw=1.4)
    ax.text(159.6, (lane_y[2] + lane_h + lane_y[-1]) / 2, "each null result → test a new driver", rotation=-90,
            ha="center", va="center", fontsize=7.8, color=v.INK_2)

    # 5 · Answer
    ans = [("Is overfishing getting worse?", "Yes. World +0.55 pp/yr (1974–2021), SEA +7.4 pp/decade; "
                                             "no slowdown after 2015. Indonesia, Thailand, Malaysia worsen "
                                             "faster than the world."),
           ("What drives it?", "No single yearly driver, pair or combination: catch, effort, warming, "
                               "shipping, dams, aquaculture and pollution share only the trend. More effort for "
                               "less fish (CPUE −35% world, −50% SEA) → slow, cumulative depletion."),
           ("SDG 14.4?", "Missed. Rebuilding stalled in SEA after 2015; world on course for ~41% "
                         "overexploited by 2030.")]
    for (t, b), y in zip(ans, [70, 40, 16]):
        box(ax, 166, y, 23, 22 if y != 16 else 16, t, b, fc="#e7f1fb", wrap=32)

    # flow arrows
    for y0, y1 in ((85, 84), (67, 64), (49, 44)):
        arrow(ax, 32.3, y0, 35.7, y1)
    for y in (84, 64, 44):
        arrow(ax, 64.3, y, 67.7, 79 if y > 60 else 47)
    arrow(ax, 96.3, 85, 99.7, 85)
    arrow(ax, 96.3, 58, 99.7, 70)
    for y in (81, 51, 24):
        arrow(ax, 161.3, y, 165.7, y)

    # legend
    lx = 2
    for i, c in enumerate([REJECT, MIXED, FAIL]):
        chip(ax, lx, 4.2, "", c, w=3.2, h=2.6)
        ax.text(lx + 4.2, 5.5, c, ha="left", va="center", fontsize=8.6, color=v.INK_2)
        lx += 30
    ax.text(lx + 2, 5.5, "Verdicts read from 3-hypothesis-tests/results/decisions.csv; "
                         "H2c, H5b, H5c are mixed (true for some countries or the world, not SEA as a whole); "
                         "H6a, H7c weak (one source only); H7b rests on 3 countries",
            ha="left", va="center", fontsize=8.2, color=v.MUTED)

    v.suptitle(fig, "Analysis map — from data to answers, with every hypothesis verdict",
               "Overfishing is confirmed and worsening; seven rounds of driver tests (pollution included) find no yearly "
               "cause, alone or in combination, and the evidence points to slow, cumulative depletion")
    save(fig, "O.01_analysis_map", "this repository's result tables")


# =========================================================================== #
# O.02 — stationarity of every series tested
# =========================================================================== #
def section_stationarity() -> None:
    print("O.02 stationarity overview")
    parts = [("Iteration 1", "0_stationarity_adf_kpss.csv"), ("Iteration 2", "7_I2_stationarity_effort.csv"),
             ("Iteration 3", "8_I3_stationarity_warming.csv"), ("Iteration 4", "9_I4_stationarity.csv"),
             ("Iteration 7", "13_I7_stationarity.csv")]
    rows = []
    for it, f in parts:
        d = pd.read_csv(HT / f)
        d["iteration"] = it
        d["ok"] = (d["transformed_adf_p"] < ALPHA) & (d["transformed_kpss_p"] >= ALPHA)
        d["adf_only"] = (d["transformed_adf_p"] < ALPHA) & (d["transformed_kpss_p"] < ALPHA)
        rows.append(d[["iteration", "series", "level_adf_p", "level_kpss_p", "transformed_adf_p",
                       "transformed_kpss_p", "transformation_used", "ok", "adf_only"]])
    d = pd.concat(rows, ignore_index=True)
    n = len(d)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 0.34 * n + 2.4), sharey=True,
                                 gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05})
    yy = np.arange(n)[::-1]
    for ax, lv, tr, title in [(a1, "level_adf_p", "transformed_adf_p", "ADF (H₀ unit root): want p < 0.05"),
                              (a2, "level_kpss_p", "transformed_kpss_p", "KPSS (H₀ stationary): want p ≥ 0.05")]:
        for y, (_, r) in zip(yy, d.iterrows()):
            good = r["ok"]
            col = v.SERIES[0] if good else (v.SERIES[3] if r["adf_only"] else v.SERIES[1])
            ax.plot([max(r[lv], 1e-6), max(r[tr], 1e-6)], [y, y], color=v.GRID, lw=2, zorder=1)
            ax.scatter(max(r[lv], 1e-6), y, s=30, facecolor=v.SURFACE, edgecolor=v.INK_2, lw=1.2, zorder=3)
            ax.scatter(max(r[tr], 1e-6), y, s=36, color=col, edgecolor=v.SURFACE, lw=0.8, zorder=4)
        ax.set_xscale("log")
        ax.axvline(ALPHA, color=v.INK_2, lw=1, ls=(0, (3, 3)))
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False)
    a1.set_xlim(1e-6, 1.5)
    a2.set_xlim(5e-3, 0.15)
    a1.set_xlabel("p-value (log scale): ○ level → ● after transformation")
    a2.set_xlabel("p-value (KPSS table is bounded at 0.01 and 0.10)")
    a1.set_yticks(yy, [f"{s}" for s in d["series"]], fontsize=8)
    edges = d.index[d["iteration"] != d["iteration"].shift()].tolist()[1:]
    for e in edges:
        for ax in (a1, a2):
            ax.axhline(yy[e] + 0.5, color=v.AXIS, lw=0.8)
    for it in d["iteration"].unique():
        top = yy[d.index[d["iteration"] == it][0]]
        a2.text(1.02, top, it, ha="left", va="center", fontsize=8, color=v.MUTED, fontweight="bold",
                transform=a2.get_yaxis_transform())
    handles = [plt.Line2D([], [], marker="o", ls="", color=v.SERIES[0], label="stationary after transform (both tests agree)"),
               plt.Line2D([], [], marker="o", ls="", color=v.SERIES[3], label="ADF yes, KPSS no (flagged, read with care)"),
               plt.Line2D([], [], marker="o", ls="", color=v.SERIES[1], label="not stationary (flagged unreliable)")]
    a1.legend(handles=handles, loc="lower left", fontsize=8)
    ok, adf_only = int(d["ok"].sum()), int(d["adf_only"].sum())
    v.suptitle(fig, "Stationarity — almost every series trends in levels and is fixed by differencing",
               f"{n} series across iterations 1–4 and 7: {ok} pass both tests after transforming, {adf_only} pass ADF only, "
               f"{n - ok - adf_only} stay non-stationary. Correlation tests use the transformed series")
    fig.subplots_adjust(top=1 - 1.05 / fig.get_figheight(), bottom=0.9 / fig.get_figheight(), left=0.25, right=0.92)
    save(fig, "O.02_stationarity_all_series", "3-hypothesis-tests results (ADF + KPSS)")


# =========================================================================== #
# O.03 — assumption checks and the test choices they drove
# =========================================================================== #
def section_assumptions() -> None:
    print("O.03 assumption checks")
    diag = pd.read_csv(HT / "6_regression_diagnostics.csv")
    checks = [("shapiro_p", "Normal residuals\n(Shapiro p ≥ 0.05)", lambda x: x >= ALPHA, "{:.3f}"),
              ("durbin_watson", "No autocorrelation\n(Durbin-Watson 1.5–2.5)", lambda x: 1.5 <= x <= 2.5, "{:.2f}"),
              ("ljung_box_p", "No autocorrelation\n(Ljung-Box p ≥ 0.05)", lambda x: x >= ALPHA, "{:.3f}"),
              ("breusch_pagan_p", "Equal variance\n(Breusch-Pagan p ≥ 0.05)", lambda x: x >= ALPHA, "{:.3f}")]
    b = pd.read_csv(STT / "B1_B2_assumption_checks_and_tests.csv")

    fig = plt.figure(figsize=(16, 9.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.42)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    n = len(diag)
    for i, (_, r) in enumerate(diag.iterrows()):
        y = n - 1 - i
        for j, (col, _, ok, fmt) in enumerate(checks):
            val = r[col]
            passed = bool(ok(val)) if pd.notna(val) else True
            a1.add_patch(plt.Rectangle((j - 0.47, y - 0.42), 0.94, 0.84, color="#f0efec" if passed else "#f9ece7",
                                       ec=v.SURFACE))
            a1.text(j, y, ("" if pd.isna(val) else fmt.format(val)) + ("" if passed else "  ✗"), ha="center",
                    va="center", fontsize=7.8, color=v.INK if not passed else v.INK_2,
                    fontweight="bold" if not passed else "normal")
    a1.set_xlim(-0.5, len(checks) - 0.5)
    a1.set_ylim(-0.6, n - 0.4)
    a1.set_xticks(range(len(checks)), [c[1] for c in checks], fontsize=8.2)
    a1.xaxis.tick_top()
    a1.set_yticks(range(n)[::-1], [m.replace("H2c gap: ", "H2c ").replace(" segmented", "")
                                   .replace("rebuilding", "rebuild.") for m in diag["model"]], fontsize=7.8)
    a1.grid(False)
    for s_ in a1.spines.values():
        s_.set_visible(False)
    fails = {c[0]: int((~diag[c[0]].apply(c[2])).sum()) for c in checks}
    a1.set_title(f"1 · Residuals of {n} regressions: Ljung-Box fails in {fails['ljung_box_p']}\n"
                 "→ Newey-West (HAC) errors used everywhere", fontsize=9.5, pad=34, loc="left")

    # panel 2: parametric vs non-parametric p
    pairs = [("B1 FAO vs SAU\n(paired, n = 46)", ["Paired t-test", "Wilcoxon signed-rank"]),
             ("B2 MTL change by country\n(5 groups, n = 345)", ["One-way ANOVA", "Welch ANOVA", "Kruskal-Wallis"])]
    yy, labels, y = [], [], 0
    for grp, tests in pairs:
        sub = b[b["comparison"].str.startswith(grp[:2]) & b["test"].isin(tests)]
        for _, r in sub.iterrows():
            yy.append(y)
            labels.append(f"{grp.split()[0]} · {r['test']}")
            col = v.SERIES[0] if r["chosen"] else v.NEUTRAL
            a2.barh(y, -np.log10(r["p_value"]), color=col, height=0.6)
            a2.text(-np.log10(r["p_value"]) + 0.06, y, f"p = {r['p_value']:.4f}" + (" · chosen" if r["chosen"] else ""),
                    va="center", fontsize=8, color=v.INK_2)
            y -= 1
        a2.text(-0.05, y + 0.4 + len(sub) - 0.2, grp, ha="right", va="bottom", fontsize=8, color=v.INK,
                fontweight="bold", transform=a2.get_yaxis_transform()) if False else None
        y -= 0.8
    a2.axvline(-np.log10(ALPHA), color=v.INK_2, lw=1, ls=(0, (3, 3)))
    a2.text(-np.log10(ALPHA), 0.6, " p = 0.05", fontsize=8, color=v.INK_2)
    a2.set_yticks(yy, labels, fontsize=8.5)
    a2.set_xlim(0, 5.2)
    a2.set_xlabel("−log10(p)")
    a2.grid(axis="y", visible=False)
    a2.text(0.0, -0.1, "B1: normality fails (Shapiro p = 0.004) → Wilcoxon\nB2: variances unequal (Levene p < 0.001), "
            "4 of 5 groups\nnon-normal → Kruskal-Wallis. A plain ANOVA would have\nwrongly found a difference (p = 0.033)",
            transform=a2.transAxes, ha="left", va="top", fontsize=8.2, color=v.INK_2)
    a2.set_title("2 · Course tests: the assumption check picks the test,\nand here it changes the answer",
                 fontsize=9.5, pad=34, loc="left")
    v.suptitle(fig, "Assumption checks — they decided which tests were valid",
               "Left: residual diagnostics of every regression (✗ = check failed). Right: parametric vs non-parametric "
               "p-values for the course comparisons; blue = the test the assumption checks selected")
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.2, right=0.97)
    save(fig, "O.03_assumption_checks_and_test_choice", "2-statistical-tests and 3-hypothesis-tests results")


# =========================================================================== #
# O.04 — long-run tests across iterations
# =========================================================================== #
def section_longrun() -> None:
    print("O.04 long-run tests")
    rows = []
    c1 = pd.read_csv(HT / "2_H1b_cointegration.csv").iloc[0]
    rows.append(("Iteration 1", "Catch × FAO", "cointegration, with trend", c1["p"]))
    for _, r in pd.read_csv(HT / "7_I2_H1b_cointegration_effort.csv").iterrows():
        rows.append(("Iteration 2", f"Effort × {r['response'].split()[0]}",
                     "cointegration, constant" if r["trend_term"] == "c" else "cointegration, with trend", r["p"]))
    for _, r in pd.read_csv(HT / "8_I3_H3a_cointegration.csv").iterrows():
        kind = ("cointegration, joint with effort" if "+" in r["drivers"] else
                "cointegration, constant" if r["trend_term"] == "c" else "cointegration, with trend")
        rows.append(("Iteration 3", f"{r['drivers'].replace(' + log effort', '')} × {r['response'].split()[0]}",
                     kind, r["p"]))
    for _, r in pd.read_csv(HT / "9_I4_H4a_cointegration.csv").dropna(subset=["p"]).iterrows():
        rows.append(("Iteration 4", f"{r['driver'].replace(' (cumulative)', '')} × {r['response'].split()[0]}",
                     "cointegration, constant" if r["trend_term"] == "c" else "cointegration, with trend", r["p"]))
    for it, f, lbl in [("Granger", "2_H1b_granger.csv", "Catch → FAO (iteration 1)"),
                       ("Granger", "7_I2_H1b_granger_effort.csv", "Effort → FAO (iteration 2)"),
                       ("Granger", "8_I3_H3a_granger.csv", "SST → FAO (iteration 3)")]:
        g = pd.read_csv(HT / f)
        rows.append((it, lbl, "Granger causality (min Holm p, lags 1–4)", g["p_holm"].min()))
    d = pd.DataFrame(rows, columns=["iteration", "pair", "test", "p"])
    d["label"] = d["pair"]
    d = d.groupby(["iteration", "label"], sort=False)

    kinds = {"cointegration, constant": ("o", v.SERIES[0]), "cointegration, with trend": ("D", v.SERIES[1]),
             "cointegration, joint with effort": ("s", v.SERIES[2]),
             "Granger causality (min Holm p, lags 1–4)": ("^", v.SERIES[3])}
    groups = list(d.groups.keys())
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(groups) + 2.6))
    yy = np.arange(len(groups))[::-1]
    prev = None
    for y, key in zip(yy, groups):
        g = d.get_group(key)
        for _, r in g.iterrows():
            m, c = kinds[r["test"]]
            ax.scatter(r["p"], y, marker=m, s=60, color=c, edgecolor=v.SURFACE, lw=0.8, zorder=3)
        if prev is not None and key[0] != prev:
            ax.axhline(y + 0.5, color=v.AXIS, lw=0.8)
        prev = key[0]
    for it in dict.fromkeys(k[0] for k in groups):
        top = yy[[k[0] for k in groups].index(it)]
        ax.text(1.02, top, it, ha="left", va="center", fontsize=8, color=v.MUTED, fontweight="bold",
                transform=ax.get_yaxis_transform())
    ax.axvline(0.05, color=v.INK_2, lw=1, ls=(0, (3, 3)))
    ax.axvline(0.10, color=v.AXIS, lw=1, ls=(0, (1, 2)))
    ax.set_xscale("log")
    ax.set_xlim(0.01, 1.2)
    ax.set_yticks(yy, [k[1] for k in groups], fontsize=8.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("p-value (log scale); dashed = 0.05, dotted = 0.10")
    handles = [plt.Line2D([], [], marker=m, ls="", color=c, label=k) for k, (m, c) in kinds.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True, facecolor=v.SURFACE, edgecolor=v.SURFACE,
              framealpha=1)
    sig_c = d.obj[(d.obj["p"] < ALPHA) & (d.obj["test"] == "cointegration, constant")]
    sig_t = d.obj[(d.obj["p"] < ALPHA) & (d.obj["test"] == "cointegration, with trend")]
    n_coint = int(d.obj["test"].str.startswith("cointegration").sum())
    v.suptitle(fig, f"Long-run tests — only {len(sig_c) + len(sig_t)} of {n_coint} cointegration tests pass, "
               "and neither holds in both data sources",
               f"{len(d.obj)} Engle-Granger and Granger tests across iterations 1–4. Significant: {len(sig_c)} with a "
               f"constant only, {len(sig_t)} with a trend term ("
               + (", ".join(sig_t['pair']) or "none") + "), 0 jointly with effort, 0 Granger after Holm. "
               "Each passing pair fails in the other source (FAO vs Sea Around Us)")
    fig.subplots_adjust(top=1 - 1.0 / fig.get_figheight(), bottom=0.9 / fig.get_figheight(), left=0.22, right=0.9)
    save(fig, "O.04_long_run_tests", "3-hypothesis-tests results (Engle-Granger, Granger)")


# =========================================================================== #
# Figure index — one row per figure in the repo (report catalogue + dashboard data)
# =========================================================================== #
# Main-body figure numbers follow the report outline's figure plan; "Appendix" = moved out of
# the main body because another figure carries the same message.
REPORT_FIG = {"1.07": 1, "A.01": 2, "3.10": 3, "V.01": 4, "I.01": 5, "1.03": 6, "1.05": 7, "3.02": 8, "E.02": 9,
              "B.01": 10, "H1a.01": 11, "H1b.01": 12, "H2a.01": 13, "H2c.01": 14, "S.01": 15, "I4.03": 16,
              "I2.01": 17, "I5.02": 18, "I5.03": 19, "I5.04": 20}
APPENDIX = {"1.01", "D.01 (residuals)", "H2b.01", "I2.02", "I2.03", "I3.02", "I4.02", "I3.01", "I4.01", "C.01",
            "D.01 (chi-square)", "I4.04", "I5.01", "I6.01", "I6.02", "I6.03",
            "I7.01", "I7.02", "I7.03", "I7.04", "I7.05", "I7.06", "I7.07"}

# id: (title, rubric section, hypotheses, iteration, message) for figures outside 1-visualisation
CAPTIONS = {
    "A.01": ("Missing-value map", "Data Understanding & Preparation", "", "", "Coverage of every source; gaps are reporting cycles, not data loss; 5 FAO years look interpolated"),
    "B.01": ("Q-Q plots, normality", "Statistical Analysis", "", "", "Most series fail normality, so non-parametric tests lead"),
    "C.01": ("Before vs after 2015 (rank-sum)", "Statistical Analysis", "H1a", "1", "No indicator changed significantly after the SDGs (all p ≥ 0.22)"),
    "D.01 (chi-square)": ("Overexploitation class vs period", "Statistical Analysis", "H2a", "1", "SEA overexploitation class depends strongly on period (Cramér's V = 0.68) and country"),
    "D.02": ("Chi-square vs sample size", "Data Understanding & Preparation", "", "", "Chi-square rejects Benford only because N is huge; MAD confirms conformity"),
    "E.01": ("Clusters of FAO areas", "Statistical Analysis", "", "", "Area 71 sits with lower-level areas but has one of the steepest trends"),
    "E.02": ("SEA EEZ clustering", "Exploratory Data Analysis", "", "", "Vietnam is an outlier; the rest form 3 groups that cut across borders"),
    "E.03": ("Silhouette and PCA", "Statistical Analysis", "", "", "How many clusters, and how well separated (silhouette, PCA view)"),
    "S.01": ("All drivers: levels vs beyond the trend", "Statistical Analysis", "H1b H3a H4a", "1–4", "Every driver rises with overexploitation; none is linked beyond the shared trend"),
    "I5.01": ("Pooled panel, fixed effects", "Statistical Analysis", "H5a", "5", "Pooling 11 EEZs with the trend removed still finds no pressure effect (0 of 36)"),
    "I5.02": ("Stock rebuilding before/after 2015", "Statistical Analysis", "H5b", "5", "Rebuilding sped up worldwide after 2015 but stalled in SEA"),
    "I5.03": ("Fishing-in-Balance index", "Discussion", "H5c", "5", "SEA catches still expanding, which explains its rising trophic level; Thailand past that phase"),
    "I6.01": ("Pairs of drivers (joint tests)", "Statistical Analysis", "H6a", "6", "No pair of drivers explains overexploitation in both data sources; levels-with-trend 'links' are spurious"),
    "I6.02": ("Interactions between pressures", "Statistical Analysis", "H6b", "6", "No pressure changes the effect of another in the SEA country panel (0 of 16)"),
    "I6.03": ("Best combination of drivers", "Statistical Analysis", "H6c", "6", "No combination predicts unseen years better than the trend alone"),
    "I7.01": ("Pollution drivers (global)", "Statistical Analysis", "H7a", "7", "Fertiliser, pesticides, plastic and acidity rise and oil spills fall with overexploitation; no link beyond the trend in both sources"),
    "I7.02": ("SEA pollution: CCF and panel", "Statistical Analysis", "H7b", "7", "SEA run-off and chlorophyll-a don't track the fisheries indicators (0 of 312 lag tests, 0 of 24 panel terms)"),
    "I7.03": ("Pollution series too short to test", "Data Understanding & Preparation", "H7", "7", "Nutrient loading (2000–15), plastic (2019) and beach litter are too short or sparse to test alone; nutrients are tested per sea in H7f"),
    "I7.04": ("Pollution + other drivers (pairs)", "Statistical Analysis", "H7c", "7", "No pair with a pollution driver explains overexploitation in both data sources"),
    "I7.05": ("Pollution × pressure interactions", "Statistical Analysis", "H7d", "7", "Run-off does not change the effect of fishing, warming or aquaculture (0 of 16)"),
    "I7.06": ("Best combination with pollution", "Statistical Analysis", "H7e", "7", "Adding pollution doesn't make any combination predict unseen years better than the trend"),
    "I7.07": ("Nutrient loading per sea (5 LMEs)", "Statistical Analysis", "H7f", "7", "Seas whose nutrient loading rose faster than the others' do not become more overexploited (0 of 16)"),
    "I5.04": ("Forecast to 2030", "Conclusion", "H5d", "5", "World on course for ~41% overexploited by 2030; SEA levelled off, no improvement"),
    "H1a.01": ("World trend tests", "Statistical Analysis", "H1a", "1", "Significant rise (+0.55 pp/yr); no slowdown after 2015"),
    "H1b.01": ("Catch CCF", "Statistical Analysis", "H1b", "1", "World catch does not lead overexploitation at any lag"),
    "H2a.01": ("SEA Sen slopes", "Statistical Analysis", "H2a", "1", "Every SEA country and EEZ worsens; trophic level falls only in places"),
    "H2b.01": ("SEA catch → trophic level CCF", "Statistical Analysis", "H2b", "1", "Catch changes don't predict trophic-level changes (0 of 66 tests)"),
    "H2c.01": ("SEA vs world slopes", "Statistical Analysis", "H2c", "1", "SEA as a whole = world pace; Indonesia, Thailand, Malaysia faster"),
    "D.01 (residuals)": ("H1a residual diagnostics", "Statistical Analysis", "H1a", "1", "Residuals are autocorrelated → HAC errors and Hamed-Rao Mann-Kendall"),
    "I2.01": ("Catch per unit effort", "Discussion", "H1b", "2", "More effort for the same fish: CPUE −35% world, −50% SEA"),
    "I2.02": ("Effort CCF", "Statistical Analysis", "H1b", "2", "World effort growth doesn't lead overexploitation"),
    "I2.03": ("SEA effort CCF heatmap", "Statistical Analysis", "H2b", "2", "1 of 132 lag tests significant; none after full correction"),
    "I3.01": ("Warming: levels vs detrended", "Statistical Analysis", "H3a", "3", "Warming shares the trend only"),
    "I3.02": ("SEA warming CCF", "Statistical Analysis", "H3b", "3", "SEA seas warm +0.12 °C/decade, but don't track fisheries indicators"),
    "I4.01": ("Shipping, dams, aquaculture (global)", "Statistical Analysis", "H4a", "4", "Human-use drivers share the trend; no link beyond it"),
    "I4.02": ("SEA shipping & structures CCF", "Statistical Analysis", "H4b", "4", "2 of 492 lag tests significant; none after full correction"),
    "I4.03": ("Island-building DiD", "Statistical Analysis", "H4c", "4", "Spratly island building left no measurable mark on neighbouring EEZs"),
    "I4.04": ("Offshore platforms and wind", "Statistical Analysis", "H4", "4", "Satellite records start too late to test (3–4 overlapping years)"),
    "I.01": ("Integrity summary", "Data Understanding & Preparation", "", "", "Digits genuine; under-reporting and trade gaps are the issue"),
    "V.01": ("Validation summary", "Data Understanding & Preparation", "", "", "Independent sources agree on trend; level gaps explained"),
    "O.01": ("Analysis map", "Introduction", "all", "1–7", "Data → preparation → EDA → 7 test iterations → answers, with every verdict"),
    "O.02": ("Stationarity of every series", "Statistical Analysis", "", "1–4, 7", "Almost every series trends in levels and is fixed by differencing"),
    "O.03": ("Assumption checks and test choice", "Statistical Analysis", "", "", "Assumption checks decided which tests were valid; here they change the answer"),
    "O.04": ("Long-run tests", "Statistical Analysis", "H1b H3a H4a", "1–4", "Only 2 of 23 cointegration tests pass, neither in both sources"),
}
STEP = {"0-data-quality": "0 · Data quality", "1-visualisation": "1 · EDA figures",
        "2-statistical-tests": "2 · Course methods", "3-hypothesis-tests": "3 · Hypothesis tests",
        "4-overview": "4 · Overview"}


def section_index() -> None:
    print("Figure index")
    rows = []
    # 1-visualisation: captions from its README gallery table
    for line in (ROOT / "1-visualisation" / "README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and ".png)" in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            fid, img, kind, msg = cells[0], cells[1], cells[2], cells[3]
            rel = "1-visualisation/" + img[img.index("(") + 1: img.index(")")]
            sec = rel.split("/")[2]
            rubric = "Data Understanding & Preparation" if sec.startswith(("4_", "5_")) else "Exploratory Data Analysis"
            hyp = " ".join(sorted(set(h for h in ("H1a", "H1b", "H2a", "H2b", "H2c") if h in msg)))
            title = Path(rel).stem.split("_", 1)[1].replace("_", " ")
            for low, up in [("fao", "FAO"), ("acf", "ACF"), ("eez", "EEZ"), ("sau", "SAU"), ("owid", "OWID"),
                            ("un sdg", "UN SDG"), ("mtl", "MTL"), ("hs03", "HS03"), ("lme", "LME"), ("sea", "SEA")]:
                title = " ".join(up if w.lower() == low else w for w in title.replace(up.lower(), low).split(" ")) \
                    if " " not in low else title.replace(low, up)
            rows.append({"figure_id": fid, "title": title[0].upper() + title[1:], "chart": kind,
                         "step": STEP["1-visualisation"], "rubric_section": rubric, "hypotheses": hyp, "iteration": "",
                         "message": msg.replace("**", ""), "file": rel})
    for f in sorted(ROOT.glob("[0-4]-*/figures/**/*.png")):
        if f.relative_to(ROOT).parts[0] == "1-visualisation":
            continue                                   # EDA figures: captions come from its README (above)
        rel = f.relative_to(ROOT).as_posix()
        fid = f.stem.split("_", 1)[0]
        if fid == "D.01":
            fid = "D.01 (residuals)" if "2-hypothesis" in rel else "D.01 (chi-square)"
        title, rubric, hyp, it, msg = CAPTIONS[fid]
        rows.append({"figure_id": fid, "title": title, "chart": "", "step": STEP[rel.split("/")[0]],
                     "rubric_section": rubric, "hypotheses": hyp, "iteration": it, "message": msg, "file": rel})
    d = pd.DataFrame(rows)
    d["report_role"] = d["figure_id"].map(lambda i: f"Fig {REPORT_FIG[i]}" if i in REPORT_FIG else
                                          ("Appendix" if i in APPENDIX else
                                           ("Overview" if i.startswith("O.") else "Gallery")))
    # the outline places two EDA-folder figures in other sections
    d.loc[d["figure_id"] == "1.07", "rubric_section"] = "Introduction"
    d.loc[d["figure_id"] == "3.10", "rubric_section"] = "Data Understanding & Preparation"
    d["report_fig_no"] = d["figure_id"].map(REPORT_FIG)
    d = d.sort_values(["report_fig_no", "file"], na_position="last").drop(columns="report_fig_no")
    d.to_csv(HERE / "figure_index.csv", index=False)
    print(f"  saved 4-overview/figure_index.csv ({len(d)} figures, {d['report_role'].str.startswith('Fig').sum()} in the main body)")


SECTIONS = {"map": section_map, "stationarity": section_stationarity, "assumptions": section_assumptions,
            "longrun": section_longrun, "index": section_index}


def main() -> None:
    parser = argparse.ArgumentParser(description="Report-level overview figures.")
    parser.add_argument("--only", nargs="+", choices=SECTIONS)
    args = parser.parse_args()
    v.setup_style()
    for name in args.only or SECTIONS:
        SECTIONS[name]()


if __name__ == "__main__":
    main()
