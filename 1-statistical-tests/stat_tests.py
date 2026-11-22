"""
stat_tests.py — Course-syllabus statistical methods applied to the SDG 14 data
===============================================================================

KIE4033 Assignment 1 · SDG 14 overfishing analysis (Global -> Southeast Asia)

Applies the EDA / statistics / unsupervised-learning methods from the course
(weeks 6–7) to the datasets in ``data/raw/``. Every test writes a results table
(CSV) to ``1-statistical-tests/results/`` and a figure (PNG, 200 dpi) to
``1-statistical-tests/figures/<section>/``. One-line results of every test are
collected in ``results/test_summary.csv``.

Run from anywhere (paths are resolved from this file):

    python 1-statistical-tests/stat_tests.py                  # everything
    python 1-statistical-tests/stat_tests.py --only chisquare clustering

Sections (course topic -> what is done here)
--------------------------------------------
A  profile      EDA: tabular summaries, IQR outlier detection, missing-value
                analysis, frequency distribution table
B  assumptions  Hypothesis testing: Shapiro-Wilk + Levene decide between the
                parametric test (paired t-test / one-way ANOVA) and its
                non-parametric counterpart (Wilcoxon signed-rank / Kruskal-Wallis);
                both are reported, the chosen one is marked
C  ranksum      Wilcoxon rank-sum (Mann-Whitney U) + Welch t-test: did yearly
                changes differ before vs after the SDGs were adopted (2015)?
D  chisquare    Chi-square test of independence (stock-status class x period,
                class x country, Benford leading digit x reporting status) with
                Cramér's V, and a demonstration of chi-square's sample-size problem
E  clustering   Unsupervised learning: hierarchical (Ward) clustering + k-means
                (k chosen by silhouette) on the 15 FAO fishing areas (global) and
                the 11 Southeast Asian EEZs

Caveat that applies throughout
------------------------------
Yearly observations of the same series are not independent (autocorrelation).
Most textbook tests assume independence, so p-values here are optimistic and are
always reported next to an effect size (Cohen's d, eta², rank-biserial r,
Cramér's V). Treat p-values as indicative, effect sizes as the main result.

The chart style (palette, fonts, header layout) and the data loaders are reused
from ``0-visualisation/visualise.py`` so both folders look and load the same.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from matplotlib.colors import ListedColormap
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from statsmodels.stats.oneway import anova_oneway

# =========================================================================== #
# Paths + shared style/loaders from the visualisation script
# =========================================================================== #
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0-visualisation"))
import visualise as v  # noqa: E402  (style constants, suptitle/layout helpers, loaders)

RAW = ROOT / "data" / "raw"
VAL = RAW / "validation"
FIG = HERE / "figures"
RES = HERE / "results"

SEA5 = v.SEA5
ALPHA = 0.05            # significance level used for every decision
SDG_YEAR = 2015         # SDGs adopted September 2015
PRE_WINDOW = (2006, 2015)   # yearly changes ending in these years = "before"
# "after" = yearly changes ending in 2016 .. last year of each series

SUMMARY: list[dict] = []    # one row per test -> results/test_summary.csv


# =========================================================================== #
# Small helpers
# =========================================================================== #
def save_fig(fig, section: str, name: str, source: str) -> None:
    """Add a source line and save under figures/<section>/<name>.png."""
    fig.text(0.01, -0.01, f"Source: {source}", ha="left", va="top", fontsize=8, color=v.MUTED)
    path = FIG / section / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("  saved", path.relative_to(ROOT))


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    """Write a results table as CSV (rounded for readability)."""
    RES.mkdir(parents=True, exist_ok=True)
    path = RES / f"{name}.csv"
    df.round(4).to_csv(path, index=index)
    print("  saved", path.relative_to(ROOT))


def record(section: str, test: str, data: str, n: str, statistic: str,
           p: float | None, effect: str, conclusion: str) -> None:
    """Add one line to the cross-section summary table."""
    SUMMARY.append({"section": section, "test": test, "data": data, "n": n,
                    "statistic": statistic, "p_value": p, "effect_size": effect,
                    "conclusion": conclusion})


def fmt_p(p: float) -> str:
    """p-value with its relation sign for titles: '= 0.123', or '< 0.001' for very small p."""
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def shapiro_p(x) -> float:
    """Shapiro-Wilk normality p-value (NaN when n < 3, where the test is undefined)."""
    x = np.asarray(x, float)
    return float(stats.shapiro(x).pvalue) if len(x) >= 3 else np.nan


def lag1_autocorr(x) -> float:
    """Lag-1 autocorrelation — how strongly a value depends on the previous year."""
    x = pd.Series(np.asarray(x, float))
    return float(x.autocorr(1)) if len(x) > 3 else np.nan


def cramers_v(chi2: float, n: int, shape: tuple[int, int]) -> float:
    """Cramér's V = sqrt(chi² / (n · (min(rows, cols) − 1))): 0 = independent, 1 = perfect association."""
    return float(np.sqrt(chi2 / (n * (min(shape) - 1))))


def adjusted_residuals(obs: np.ndarray, exp: np.ndarray) -> np.ndarray:
    """
    Haberman adjusted residuals for a contingency table:
    (O − E) / sqrt(E · (1 − row share) · (1 − column share)).
    |value| > 1.96 marks a cell that differs from independence at the 5% level.
    """
    n = obs.sum()
    row = obs.sum(1, keepdims=True) / n
    col = obs.sum(0, keepdims=True) / n
    return (obs - exp) / np.sqrt(exp * (1 - row) * (1 - col))


def cochran_ok(exp: np.ndarray) -> bool:
    """Cochran's rule for chi-square: every expected count ≥ 1 and at most 20% below 5."""
    return bool(exp.min() >= 1 and (exp < 5).mean() <= 0.2)


# =========================================================================== #
# Data builders
# =========================================================================== #
def world_fao() -> pd.Series:
    """FAO % of world stocks overexploited (OWID), 1974–2021."""
    return v.owid_series("share-of-fish-stocks-overexploited", "World")


def world_sau() -> pd.Series:
    """Sea Around Us % of world stocks collapsed + over-exploited, 1950–2019."""
    return v.sau_overexploited(VAL / "sau_regions" / "stock_status" / "global.csv")


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
    d = v.load_sau_eez()
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
    c, z = v.EEZ_COUNTRY[eez], v.EEZ_LABEL[eez]
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
    return v.owid_series("capture-fishery-production", entity)


# =========================================================================== #
# Section A — Data profiling: tabular summaries, outliers, missing values,
#             frequency distribution
# =========================================================================== #
def iqr_outliers(x: pd.Series) -> list[int]:
    """Years outside Tukey's fences [Q1 − 1.5·IQR, Q3 + 1.5·IQR]."""
    q1, q3 = x.quantile([0.25, 0.75])
    k = 1.5 * (q3 - q1)
    return [int(y) for y in x.index[(x < q1 - k) | (x > q3 + k)]]


def describe_series(name: str, source: str, unit: str, s: pd.Series) -> dict:
    """
    One row of the descriptive-statistics table. Outliers are searched in the
    level AND in the yearly change: a trending series has no level outliers by
    construction, but an abrupt jump shows up in the change.
    """
    s = s.dropna().sort_index()
    span = range(int(s.index.min()), int(s.index.max()) + 1)
    ch = s.diff().dropna()
    return {
        "variable": name, "source": source, "unit": unit,
        "first_year": int(s.index.min()), "last_year": int(s.index.max()),
        "n": len(s), "missing_in_span": len(span) - len(s),
        "mean": s.mean(), "sd": s.std(ddof=1), "min": s.min(), "q1": s.quantile(0.25),
        "median": s.median(), "q3": s.quantile(0.75), "max": s.max(),
        "skewness": stats.skew(s, bias=False),
        "outlier_years_level": " ".join(map(str, iqr_outliers(s))),
        "outlier_years_change": " ".join(map(str, iqr_outliers(ch))),
    }


def section_profile() -> None:
    sec = "A_profile"
    print("Section A: data profiling")
    p = eez_panel()
    cp = country_panel(p)

    # A1 Descriptive statistics + IQR outliers ---------------------------------
    rows = [
        describe_series("World % stocks overexploited", "FAO (OWID)", "%", world_fao()),
        describe_series("World % stocks overexploited", "Sea Around Us", "%", world_sau()),
        describe_series("World capture production", "FAO (OWID)", "t", capture("World")),
    ]
    for c in SEA5:
        g = cp[cp["country"] == c].set_index("year")
        rows += [
            describe_series(f"{c} capture production", "FAO (OWID)", "t", capture(c)),
            describe_series(f"{c} % stocks overexploited", "Sea Around Us", "%", g["over"]),
            describe_series(f"{c} mean trophic level", "Sea Around Us", "level", g["mtl"]),
            describe_series(f"{c} unreported share of catch", "Sea Around Us", "%", g["unrep_share"]),
        ]
    desc = pd.DataFrame(rows)
    save_table(desc, "A1_descriptive_statistics")
    n_jump = (desc["outlier_years_change"] != "").sum()
    record("A", "Descriptive statistics + IQR (Tukey) outliers", "23 series", "23 series", "—", None,
           f"{(desc['outlier_years_level'] != '').sum()} series with level outliers, {n_jump} with outlier yearly changes",
           "Level outliers mark regime shifts (e.g. Indonesia 2008–19); many SAU series change in steps, "
           "so the IQR of yearly changes is narrow and flags many years — read with the figures")

    # A2 Missing-value analysis -------------------------------------------------
    econ = pd.read_csv(VAL / "economic" / "fisheries_gdp_capture.csv")
    iso = {"Malaysia": "MYS", "Indonesia": "IDN", "Thailand": "THA", "Vietnam": "VNM", "Philippines": "PHL"}
    ct = pd.read_csv(VAL / "comtrade" / "fish_hs03_mirror_trade.csv")
    over_area = v.owid("share-of-fish-stocks-overexploited")
    series: list[tuple[str, str, pd.Series]] = [
        ("FAO stock status", "World", world_fao()),
        ("FAO stock status", "Area 71 W. Central Pacific",
         over_area[over_area["Entity"] == "Western Central Pacific (FAO)"].set_index("Year").iloc[:, -1]),
        ("FAO stock status", "Area 57 E. Indian Ocean",
         over_area[over_area["Entity"] == "Eastern Indian Ocean (FAO)"].set_index("Year").iloc[:, -1]),
        ("SAU stock status", "World", world_sau()),
    ]
    series += [("SAU stock status", c, cp[cp["country"] == c].set_index("year")["over"]) for c in SEA5]
    series += [("FAO capture", e, capture(e)) for e in ["World"] + SEA5]
    series += [("SDG 14.7.1 fisheries % GDP", c,
                econ[econ["iso3"] == iso[c]].set_index("year")["sustainable_fisheries_pct_gdp"]) for c in SEA5]
    series += [("Comtrade mirror trade", c,
                ct[ct["country"] == c].set_index("year")["partner_imports_usd"]) for c in SEA5]

    years = np.arange(1950, 2025)
    interp = set(interpolated_years(world_fao()))
    # States: 0 outside source coverage · 1 observed · 2 flagged interpolated · 3 missing inside span
    grid = np.zeros((len(series), len(years)), int)
    miss_rows = []
    for i, (grp, ent, s) in enumerate(series):
        s = s.dropna()
        lo, hi = int(s.index.min()), int(s.index.max())
        for j, y in enumerate(years):
            if lo <= y <= hi:
                grid[i, j] = 1 if y in s.index else 3
                if grp == "FAO stock status" and ent == "World" and y in interp:
                    grid[i, j] = 2
        span = hi - lo + 1
        missing = [y for y in range(lo, hi + 1) if y not in s.index]
        miss_rows.append({"dataset": grp, "entity": ent, "first_year": lo, "last_year": hi,
                          "years_in_span": span, "observed": len(s), "missing": len(missing),
                          "pct_missing": 100 * len(missing) / span,
                          "flagged_interpolated": len(interp) if (grp, ent) == ("FAO stock status", "World") else 0,
                          "missing_years": " ".join(map(str, missing))})
    miss = pd.DataFrame(miss_rows)
    save_table(miss, "A2_missing_values")
    n_gappy = (miss["missing"] > 0).sum()
    record("A", "Missing-value analysis", "24 series, 1950–2024", "24 series", "—", None,
           f"{n_gappy} series have gaps; {len(interp)} FAO world years look interpolated",
           "Gaps come from assessment/reporting cycles (FAO areas, SDG 14.7.1 every 2 yrs), not data loss")

    cmap = ListedColormap(["#f0efec", v.SERIES[0], "#86b6ef", v.SERIES[1]])
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest",
              extent=(years[0] - 0.5, years[-1] + 0.5, len(series) - 0.5, -0.5))
    for b in range(1, len(series)):   # 2 px surface-coloured gap between rows
        ax.axhline(b - 0.5, color=v.SURFACE, lw=2)
    prev = None
    for i, (grp, _, _) in enumerate(series):   # thicker divider between datasets
        if prev is not None and grp != prev:
            ax.axhline(i - 0.5, color=v.INK_2, lw=0.8)
        prev = grp
    ax.set_yticks(range(len(series)), [f"{g} · {e}" for g, e, _ in series], fontsize=8.5)
    ax.grid(False)
    for spn in ax.spines.values():
        spn.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(k)) for k in range(4)]
    ax.legend(handles, ["Not covered by source", "Observed", "Observed but looks interpolated", "Missing inside span"],
              loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4)
    v.suptitle(fig, "A.01 — Missing-value map: which years does each dataset actually have?",
               "Gaps inside a series come from assessment cycles: FAO areas every ~2 years, SDG 14.7.1 every 2 years")
    save_fig(fig, sec, "A.01_missing_value_map", "FAO via OWID; Sea Around Us; UN SDG 14.7.1; UN Comtrade")

    # A3 Frequency distribution table -------------------------------------------
    freq = frequency_table(p)
    out = freq.copy()
    out.columns = [f"{c} (EEZ-years)" for c in out.columns]
    pct = freq.div(freq.sum(axis=1), axis=0) * 100
    for c in freq.columns:
        out[f"{c} (% of row)"] = pct[c]
    save_table(out.reset_index(), "A3_frequency_distribution")
    record("A", "Frequency distribution table", "SEA EEZ-years, % overexploited class x period",
           str(int(freq.values.sum())), "—", None, "see A3 table", "Input to chi-square test D1")


OVER_BINS = [0, 10, 25, 50, 100.01]   # no EEZ-year reaches 75%, so the top class starts at 50%
OVER_LABELS = ["< 10%", "10–25%", "25–50%", "≥ 50%"]
PERIODS = [(1950, 1979), (1980, 1999), (2000, 2019)]


def period_label(y: int) -> str:
    return next(f"{a}–{b}" for a, b in PERIODS if a <= y <= b)


def frequency_table(p: pd.DataFrame) -> pd.DataFrame:
    """EEZ-years counted by % of stocks overexploited (4 classes) and period."""
    t = p.assign(cls=pd.cut(p["over"], OVER_BINS, right=False, labels=OVER_LABELS),
                 period=p["year"].map(period_label))
    return pd.crosstab(t["period"], t["cls"]).reindex(columns=OVER_LABELS, fill_value=0)


# =========================================================================== #
# Section B — Assumption checks: parametric vs non-parametric
# =========================================================================== #
def section_assumptions() -> None:
    sec = "B_assumptions"
    print("Section B: assumption checks (Shapiro-Wilk, Levene) -> test choice")
    rows = []

    # B1 Paired: FAO vs Sea Around Us, world % overexploited ------------------
    pair = pd.concat([world_fao().rename("fao"), world_sau().rename("sau")], axis=1).dropna()
    d = pair["fao"] - pair["sau"]
    sw = shapiro_p(d)
    t_res = stats.ttest_rel(pair["fao"], pair["sau"])
    w_res = stats.wilcoxon(pair["fao"], pair["sau"])
    dz = d.mean() / d.std(ddof=1)                    # Cohen's d for paired data
    # Matched-pairs rank-biserial r = (W+ − W−) / total rank sum
    ranks = stats.rankdata(np.abs(d[d != 0]))
    r_rb = (ranks[d[d != 0] > 0].sum() - ranks[d[d != 0] < 0].sum()) / ranks.sum()
    parametric = sw >= ALPHA
    rows += [
        {"comparison": "B1 FAO vs SAU, world % overexploited (paired years)", "n": len(d),
         "shapiro_p_differences": sw, "lag1_autocorr_differences": lag1_autocorr(d),
         "test": "Paired t-test", "statistic": t_res.statistic, "p_value": t_res.pvalue,
         "effect_size": f"Cohen's dz = {dz:.2f}", "chosen": parametric},
        {"comparison": "B1 FAO vs SAU, world % overexploited (paired years)", "n": len(d),
         "shapiro_p_differences": sw, "lag1_autocorr_differences": lag1_autocorr(d),
         "test": "Wilcoxon signed-rank", "statistic": w_res.statistic, "p_value": w_res.pvalue,
         "effect_size": f"rank-biserial r = {r_rb:.2f}", "chosen": not parametric},
    ]
    chosen = (t_res, "paired t-test", f"dz = {dz:.2f}") if parametric else (w_res, "Wilcoxon signed-rank", f"r = {r_rb:.2f}")
    record("B", f"Shapiro-Wilk on differences (p = {sw:.3f}) -> {chosen[1]}",
           "World % overexploited, FAO vs SAU", str(len(d)), f"{chosen[0].statistic:.2f}", chosen[0].pvalue,
           f"mean diff {d.mean():+.1f} pp, {chosen[2]}",
           "Systematic difference between the two sources" if chosen[0].pvalue < ALPHA else "No systematic difference")

    # B2 Groups: yearly change in mean trophic level by country ----------------
    cp = country_panel(eez_panel()).sort_values(["country", "year"])
    cp["d_mtl"] = cp.groupby("country")["mtl"].diff()
    groups = {c: cp.loc[cp["country"] == c, "d_mtl"].dropna().to_numpy() for c in SEA5}
    data = list(groups.values())
    sw_g = {c: shapiro_p(x) for c, x in groups.items()}
    lev = stats.levene(*data, center="median")          # Brown-Forsythe version, robust to non-normality
    f_res = stats.f_oneway(*data)
    welch = anova_oneway(data, use_var="unequal")
    kw = stats.kruskal(*data)
    allv = np.concatenate(data)
    ss_between = sum(len(x) * (x.mean() - allv.mean()) ** 2 for x in data)
    eta2 = ss_between / ((allv - allv.mean()) ** 2).sum()
    eps2 = kw.statistic / (len(allv) - 1)                # epsilon² effect size for Kruskal-Wallis
    normal = all(p >= ALPHA for p in sw_g.values())
    equal_var = lev.pvalue >= ALPHA
    choice = "One-way ANOVA" if normal and equal_var else ("Welch ANOVA" if normal else "Kruskal-Wallis")
    common = {"comparison": "B2 Yearly change in mean trophic level, 5 countries", "n": len(allv),
              "shapiro_p_differences": np.nan, "lag1_autocorr_differences": np.nan}
    rows += [
        {**common, "test": "Levene (median-centred)", "statistic": lev.statistic, "p_value": lev.pvalue,
         "effect_size": "", "chosen": False},
        {**common, "test": "One-way ANOVA", "statistic": f_res.statistic, "p_value": f_res.pvalue,
         "effect_size": f"eta² = {eta2:.3f}", "chosen": choice == "One-way ANOVA"},
        {**common, "test": "Welch ANOVA", "statistic": welch.statistic, "p_value": welch.pvalue,
         "effect_size": f"eta² = {eta2:.3f}", "chosen": choice == "Welch ANOVA"},
        {**common, "test": "Kruskal-Wallis", "statistic": kw.statistic, "p_value": kw.pvalue,
         "effect_size": f"epsilon² = {eps2:.3f}", "chosen": choice == "Kruskal-Wallis"},
    ]
    for c in SEA5:
        rows.append({"comparison": f"B2 group normality: {c}", "n": len(groups[c]),
                     "shapiro_p_differences": sw_g[c], "lag1_autocorr_differences": lag1_autocorr(groups[c]),
                     "test": "Shapiro-Wilk", "statistic": stats.shapiro(groups[c]).statistic,
                     "p_value": sw_g[c], "effect_size": "", "chosen": False})
    save_table(pd.DataFrame(rows), "B1_B2_assumption_checks_and_tests")
    chosen_res = {"One-way ANOVA": f_res, "Welch ANOVA": welch, "Kruskal-Wallis": kw}[choice]
    chosen_p = float(chosen_res.pvalue)
    record("B", f"Shapiro-Wilk per group + Levene (p = {lev.pvalue:.3f}) -> {choice}",
           "Yearly change in MTL, 5 SEA countries", str(len(allv)), f"{float(chosen_res.statistic):.2f}",
           chosen_p, f"eta² = {eta2:.3f}, epsilon² = {eps2:.3f}",
           "Countries differ" if chosen_p < ALPHA else "No difference between countries' yearly MTL changes")

    # Post-hoc only when the omnibus test is significant
    if chosen_p < ALPHA:
        long = cp.dropna(subset=["d_mtl"])
        post = (sp.posthoc_dunn(long, val_col="d_mtl", group_col="country", p_adjust="holm")
                if choice == "Kruskal-Wallis" else
                pd.DataFrame(stats.tukey_hsd(*data).pvalue, index=SEA5, columns=SEA5))
        save_table(post.reset_index(names="country"), "B3_posthoc_pairwise_p")

    # Figure B.01: Q-Q plots, the visual version of Shapiro-Wilk ----------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
    panels = [("FAO − SAU, world (pp)", d.to_numpy(), sw, v.INK_2)] + \
             [(f"{c}: yearly ΔMTL", groups[c], sw_g[c], v.COUNTRY_COLOR[c]) for c in SEA5]
    for ax, (title, x, pval, colour) in zip(axes.flat, panels):
        (osm, osr), (slope, icpt, _) = stats.probplot(x, dist="norm")
        ax.plot(osm, icpt + slope * osm, color=v.AXIS, lw=1.2, zorder=1)
        ax.scatter(osm, osr, s=18, color=colour, edgecolor=v.SURFACE, linewidth=1, zorder=3)
        verdict = "normal" if pval >= ALPHA else "not normal"
        ax.set_title(f"{title}\nShapiro-Wilk p {fmt_p(pval)} → {verdict}", fontsize=9.5)
        ax.grid(axis="x")
    for ax in axes[1]:
        ax.set_xlabel("Theoretical normal quantile")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed value")
    v.suptitle(fig, "B.01 — Normality check (Q-Q plots): which test is allowed?",
               f"Points on the line = normal → t-test / ANOVA allowed. Chosen: {chosen[1]} (panel 1), "
               f"{choice} (countries; Levene p {fmt_p(lev.pvalue)})")
    v.layout(fig, 0)
    save_fig(fig, sec, "B.01_qq_plots_normality", "FAO via OWID; Sea Around Us")


# =========================================================================== #
# Section C — Wilcoxon rank-sum (Mann-Whitney U): before vs after 2015
# =========================================================================== #
def before_after(changes: pd.Series, years: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Split yearly changes into the pre-SDG window (2006–2015) and post-SDG (2016+)."""
    pre = changes[(years >= PRE_WINDOW[0]) & (years <= PRE_WINDOW[1])].dropna().to_numpy()
    post = changes[years > SDG_YEAR].dropna().to_numpy()
    return pre, post


def section_ranksum() -> None:
    sec = "C_rank_sum"
    print("Section C: Wilcoxon rank-sum (Mann-Whitney U), before vs after 2015")
    p = eez_panel().sort_values(["eez", "year"])
    p["d_over"] = p.groupby("eez")["over"].diff()
    p["d_mtl"] = p.groupby("eez")["mtl"].diff()
    wf = world_fao()
    cap = capture("World")
    cases = [
        ("World % stocks overexploited (FAO)", "pp per year", *before_after(wf.diff(), wf.index.to_series()),
         "FAO via OWID", f"{PRE_WINDOW[0]}–{PRE_WINDOW[1]} vs 2016–{wf.index.max()}"),
        ("World capture production (FAO)", "% per year",
         *before_after(cap.pct_change() * 100, cap.index.to_series()),
         "FAO via OWID", f"{PRE_WINDOW[0]}–{PRE_WINDOW[1]} vs 2016–{cap.index.max()}"),
        ("SEA EEZs % stocks overexploited (SAU)", "pp per year", *before_after(p["d_over"], p["year"]),
         "Sea Around Us, 11 EEZs pooled", f"{PRE_WINDOW[0]}–{PRE_WINDOW[1]} vs 2016–2019"),
        ("SEA EEZs mean trophic level (SAU)", "change per year", *before_after(p["d_mtl"], p["year"]),
         "Sea Around Us, 11 EEZs pooled", f"{PRE_WINDOW[0]}–{PRE_WINDOW[1]} vs 2016–2019"),
    ]
    rows = []
    for name, unit, pre, post, src, window in cases:
        u = stats.mannwhitneyu(post, pre, alternative="two-sided")
        welch = stats.ttest_ind(post, pre, equal_var=False)
        r_rb = 2 * u.statistic / (len(pre) * len(post)) - 1   # >0: post-2015 changes tend to be larger
        sw_pre, sw_post = shapiro_p(pre), shapiro_p(post)
        normal = sw_pre >= ALPHA and sw_post >= ALPHA
        rows.append({"variable": name, "unit": unit, "windows": window, "n_pre": len(pre), "n_post": len(post),
                     "median_pre": np.median(pre), "median_post": np.median(post),
                     "mean_pre": pre.mean(), "mean_post": post.mean(),
                     "shapiro_p_pre": sw_pre, "shapiro_p_post": sw_post,
                     "mann_whitney_U": u.statistic, "mann_whitney_p": u.pvalue, "rank_biserial_r": r_rb,
                     "welch_t": welch.statistic, "welch_p": welch.pvalue,
                     "chosen_test": "Welch t-test" if normal else "Wilcoxon rank-sum"})
        pc = welch.pvalue if normal else u.pvalue
        direction = "smaller" if np.median(post) < np.median(pre) else "larger"
        record("C", f"{'Welch t-test' if normal else 'Wilcoxon rank-sum (Mann-Whitney U)'}, before vs after 2015",
               name, f"{len(pre)} vs {len(post)}",
               f"t = {welch.statistic:.2f}" if normal else f"U = {u.statistic:.0f}", pc,
               f"rank-biserial r = {r_rb:+.2f}",
               (f"Post-2015 yearly changes significantly {direction}" if pc < ALPHA
                else f"No significant change after 2015 (post median {direction}"
                     + (", few post-2015 years)" if len(post) < 10 else ")")))
    res = pd.DataFrame(rows)
    save_table(res, "C1_rank_sum_before_after_2015")

    # Figure C.01: before/after distributions -----------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.9))
    rng = np.random.default_rng(0)
    for ax, (name, unit, pre, post, *_), r in zip(axes, cases, rows):
        for k, (vals, colour) in enumerate([(pre, v.NEUTRAL), (post, v.SERIES[0])]):
            ax.boxplot([vals], positions=[k], widths=0.45, showfliers=False,
                       medianprops={"color": v.INK, "lw": 1.5}, whiskerprops={"color": v.MUTED},
                       capprops={"color": v.MUTED}, boxprops={"color": v.MUTED})
            ax.scatter(k + rng.uniform(-0.15, 0.15, len(vals)), vals, s=12, color=colour,
                       edgecolor=v.SURFACE, linewidth=0.6, alpha=0.85, zorder=3)
        ax.axhline(0, color=v.AXIS, lw=0.8)
        ax.set_xticks([0, 1], [f"{PRE_WINDOW[0]}–{PRE_WINDOW[1]}\n(n = {len(pre)})", f"2016+\n(n = {len(post)})"])
        ptxt = r["welch_p"] if r["chosen_test"] == "Welch t-test" else r["mann_whitney_p"]
        ax.set_title(f"{name}\n{r['chosen_test']} p {fmt_p(ptxt)}, r = {r['rank_biserial_r']:+.2f}", fontsize=9)
        ax.set_ylabel(f"Yearly change ({unit})")
    v.suptitle(fig, "C.01 — Did yearly changes slow down after the SDGs were adopted (2015)?",
               "Each dot = one year (EEZ-year for SEA). Same-length recent window before 2015 so the "
               "1950s–80s fishery expansion does not dominate")
    v.layout(fig, 0)
    save_fig(fig, sec, "C.01_before_after_2015", "FAO via OWID; Sea Around Us")


# =========================================================================== #
# Section D — Chi-square tests of independence
# =========================================================================== #
def chi_square(table: pd.DataFrame, label: str, n_perm: int = 10_000) -> dict:
    """
    Chi-square test of independence on a contingency table of counts, with
    Cramér's V, Cochran's expected-count rule and a permutation p-value
    (valid even when expected counts are small).
    """
    obs = table.to_numpy(float)
    chi2, pval, dof, exp = stats.chi2_contingency(obs, correction=False)
    # Permutation test: shuffle column labels across units, recompute chi-square.
    rows = np.repeat(np.arange(obs.shape[0]), obs.sum(1).astype(int))
    cols = np.concatenate([np.repeat(np.arange(obs.shape[1]), r.astype(int)) for r in obs])
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(cols)
        t = np.zeros_like(obs)
        np.add.at(t, (rows, perm), 1)
        hits += stats.chi2_contingency(t, correction=False)[0] >= chi2 - 1e-9
    return {"test": label, "n": int(obs.sum()), "chi2": chi2, "dof": dof, "p_value": pval,
            "p_permutation": (hits + 1) / (n_perm + 1), "cramers_v": cramers_v(chi2, int(obs.sum()), obs.shape),
            "min_expected": exp.min(), "pct_expected_below_5": 100 * (exp < 5).mean(),
            "cochran_rule_ok": cochran_ok(exp), "expected": exp}


def section_chisquare() -> None:
    sec = "D_chi_square"
    print("Section D: chi-square tests of independence")
    p = eez_panel()
    results = []

    # D1 Overexploitation class x period (SEA EEZ-years) -----------------------
    t1 = frequency_table(p)
    r1 = chi_square(t1, "D1 % overexploited class x period (SEA EEZ-years)")
    results.append(r1)
    record("D", "Chi-square independence: overexploitation class x period", "11 SEA EEZs, 1950–2019",
           str(r1["n"]), f"χ²({r1['dof']}) = {r1['chi2']:.1f}", r1["p_value"], f"Cramér's V = {r1['cramers_v']:.2f}",
           "Class depends on period: EEZs moved into higher overexploitation classes over time"
           if r1["p_value"] < ALPHA else "Class independent of period")

    # D2 Overexploitation class x country, recent decade -----------------------
    recent = p[p["year"].between(2010, 2019)]
    t2 = pd.crosstab(recent["country"], np.where(recent["over"] >= 50, "≥ 50% overexploited", "< 50% overexploited"))
    t2 = t2.reindex(SEA5)
    r2 = chi_square(t2, "D2 ≥50% overexploited x country (SEA EEZ-years 2010–19)")
    results.append(r2)
    p_use = r2["p_value"] if r2["cochran_rule_ok"] else r2["p_permutation"]
    record("D", "Chi-square independence: ≥50% overexploited x country (2010–19)"
           + ("" if r2["cochran_rule_ok"] else ", permutation p (Cochran rule failed)"),
           "11 SEA EEZs", str(r2["n"]), f"χ²({r2['dof']}) = {r2['chi2']:.1f}", p_use,
           f"Cramér's V = {r2['cramers_v']:.2f}",
           "Overexploitation level depends on country" if p_use < ALPHA else "No country difference")

    # D3 Benford: leading digit x reporting status, and goodness-of-fit ---------
    bf = pd.read_csv(VAL / "benford" / "sau_catch_leading_digits.csv")
    dcols = [f"d{i}" for i in range(1, 10)]
    g = bf.groupby(["country", "reporting_status"])[dcols].sum()
    bens = []
    for c in SEA5:
        t = g.loc[c]                                              # 2 x 9: Reported / Unreported
        r = chi_square(t, f"D3 leading digit x reporting status: {c}", n_perm=0)
        rep = g.loc[(c, "Reported")].to_numpy(float)
        gof = stats.chisquare(rep, f_exp=rep.sum() * v.BENFORD)
        bens.append({"country": c, "n_values": r["n"], "chi2_digit_x_status": r["chi2"],
                     "p_digit_x_status": r["p_value"], "cramers_v": r["cramers_v"],
                     "n_reported": int(rep.sum()), "chi2_gof_reported_vs_benford": gof.statistic,
                     "p_gof_reported_vs_benford": gof.pvalue, "mad_reported": v.benford_mad(rep),
                     "nigrini_label": v.nigrini_label(v.benford_mad(rep))})
    bens = pd.DataFrame(bens)
    save_table(bens, "D3_benford_chi_square_vs_mad")
    record("D", "Chi-square: leading digit x reporting status, and goodness-of-fit vs Benford",
           "SAU catch records, 5 countries", f"{bens['n_values'].sum():,}",
           f"GOF p < 0.05 for {(bens['p_gof_reported_vs_benford'] < ALPHA).sum()}/5", None,
           f"Cramér's V {bens['cramers_v'].min():.3f}–{bens['cramers_v'].max():.3f}; "
           f"MAD {bens['mad_reported'].min():.4f}–{bens['mad_reported'].max():.4f}",
           "Large N makes chi-square flag trivial deviations; tiny V and MAD → digits conform")

    summary = pd.DataFrame([{k: val for k, val in r.items() if k != "expected"} for r in results])
    save_table(summary, "D1_D2_chi_square_independence")
    for name, tab, r in [("D1", t1, r1), ("D2", t2, r2)]:
        exp = pd.DataFrame(r["expected"], index=tab.index, columns=tab.columns)
        res = pd.DataFrame(adjusted_residuals(tab.to_numpy(float), r["expected"]), index=tab.index, columns=tab.columns)
        save_table(pd.concat({"observed": tab, "expected": exp, "adjusted_residual": res}, axis=1).reset_index(),
                   f"{name}_observed_expected_residuals")

    # Figure D.01: class shares by period + adjusted residuals -----------------
    ramp = ["#cde2fb", "#86b6ef", "#3987e5", "#184f95"]     # sequential: magnitude classes, light -> dark
    shares = t1.div(t1.sum(axis=1), axis=0) * 100
    resid = adjusted_residuals(t1.to_numpy(float), r1["expected"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    left = np.zeros(len(shares))
    for cls, colour in zip(OVER_LABELS, ramp):
        a1.barh(shares.index, shares[cls], left=left, color=colour, edgecolor=v.SURFACE, linewidth=2,
                height=0.6, label=cls)
        for y, (l, w) in enumerate(zip(left, shares[cls])):
            if w >= 7:
                a1.text(l + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=9,
                        color="#ffffff" if colour in ramp[2:] else v.INK)
        left += shares[cls].to_numpy()
    a1.invert_yaxis()
    a1.set_xlim(0, 100)
    a1.set_xlabel("% of EEZ-years in period")
    a1.grid(axis="x")
    a1.grid(axis="y", visible=False)
    a1.legend(title="% of stocks overexploited", loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4,
              title_fontsize=9)
    a1.set_title("Share of EEZ-years in each overexploitation class", fontsize=10)
    lim = np.ceil(np.abs(resid).max())
    im = a2.imshow(resid, cmap=v.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(resid.shape[0]):
        for j in range(resid.shape[1]):
            a2.text(j, i, f"{resid[i, j]:+.1f}\n({int(t1.iloc[i, j])})", ha="center", va="center", fontsize=9,
                    color="#ffffff" if abs(resid[i, j]) > 0.6 * lim else v.INK)
    a2.set_xticks(range(len(OVER_LABELS)), OVER_LABELS)
    a2.set_yticks(range(len(t1)), t1.index)
    a2.grid(False)
    for spn in a2.spines.values():
        spn.set_visible(False)
    cb = fig.colorbar(im, ax=a2, fraction=0.04, pad=0.02)
    cb.set_label("Adjusted residual", color=v.INK_2)
    cb.outline.set_visible(False)
    a2.set_title("Adjusted residuals (count in brackets); |r| > 1.96 = significant cell", fontsize=10)
    v.suptitle(fig, "D.01 — Chi-square test of independence: overexploitation class vs period (SEA EEZs)",
               f"χ²({r1['dof']}) = {r1['chi2']:.0f}, p {fmt_p(r1['p_value'])}, "
               f"Cramér's V = {r1['cramers_v']:.2f}. Red = more EEZ-years than expected if class and period were unrelated")
    v.layout(fig, 0.02)
    save_fig(fig, sec, "D.01_overexploitation_class_vs_period", "Sea Around Us (11 SEA EEZs)")

    # Figure D.02: why chi-square fails for very large samples -----------------
    # Use the country whose reported digits deviate most (largest MAD, still 'close conformity')
    demo = bens.loc[bens["mad_reported"].idxmax(), "country"]
    rep_all = g.loc[(demo, "Reported")].to_numpy(float)
    p_hat = rep_all / rep_all.sum()
    ns = np.unique(np.logspace(2, 6, 80).astype(int))
    chi = ns * ((p_hat - v.BENFORD) ** 2 / v.BENFORD).sum()   # χ² = N · Σ (p̂ − P)² / P, same p̂ at every N
    pv = stats.chi2.sf(chi, 8)
    mad_all = v.benford_mad(rep_all)
    n_cross = ns[np.argmax(pv < ALPHA)] if (pv < ALPHA).any() else None
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(ns, pv, color=v.SERIES[0], label="Chi-square goodness-of-fit p-value")
    ax.axhline(ALPHA, color=v.SERIES[7], lw=1.2, label="p = 0.05")
    if n_cross:
        ax.axvline(n_cross, color=v.AXIS, lw=1)
        ax.text(rep_all.sum() * 1.08, 0.55, f"Solid line: chi-square rejects\nBenford from N ≈ {n_cross:,}",
                ha="left", fontsize=9, color=v.INK_2)
    ax.axvline(rep_all.sum(), color=v.AXIS, lw=1, ls=(0, (3, 3)))
    ax.text(rep_all.sum() * 1.08, 0.85, f"{demo}'s actual N\n{int(rep_all.sum()):,}", ha="left", fontsize=9,
            color=v.INK_2)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Number of catch values tested (same digit proportions)")
    ax.set_ylabel("p-value")
    ax.legend(loc="lower left")
    v.suptitle(fig, "D.02 — Why we judge Benford with MAD, not chi-square",
               f"{demo}'s reported digit shares held fixed (MAD = {mad_all:.4f}, 'close conformity' at every N) — "
               "chi-square turns 'significant' purely because N grows")
    save_fig(fig, sec, "D.02_chi_square_sample_size_effect", f"Sea Around Us taxon-level catch records ({demo}, reported)")


# =========================================================================== #
# Section E — Unsupervised learning: hierarchical clustering + k-means
# =========================================================================== #
def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise each feature (mean 0, SD 1) so no feature dominates the distances."""
    return (df - df.mean()) / df.std(ddof=0)


def cluster(features: pd.DataFrame, k_range: range) -> dict:
    """
    Ward hierarchical clustering and k-means on standardised features.
    k is chosen by the highest mean silhouette score of k-means; the dendrogram
    is cut at the same k and the two partitions are compared with the adjusted
    Rand index (1 = identical grouping, ~0 = unrelated).
    """
    z = zscore(features)
    sil = {}
    labels = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(z)
        sil[k] = silhouette_score(z, km.labels_)
        labels[k] = km.labels_
    k_best = max(sil, key=sil.get)
    link = linkage(z.to_numpy(), method="ward")
    hc = fcluster(link, k_best, criterion="maxclust")
    km = labels[k_best]
    # Renumber k-means clusters 1..k in order of first appearance in the dendrogram leaves,
    # so cluster numbers (and colours) read top-to-bottom in the figures.
    order = dendrogram(link, no_plot=True)["leaves"]
    mapping = {}
    for i in order:
        mapping.setdefault(km[i], len(mapping) + 1)
    km = np.array([mapping[x] for x in km])
    return {"z": z, "silhouette": sil, "k": k_best, "link": link, "hc": hc, "km": km,
            "ari": adjusted_rand_score(hc, km), "order": order}


# Label nudges (points per name) where two direct labels would collide; others use (6, 3).
LABEL_NUDGE = {"Eastern Central Atlantic": (6, -11), "Eastern Indian Ocean": (6, -11),
               "Malaysia – Peninsula East": (6, -11)}


def label_points(ax, xs, ys, names) -> None:
    """Direct labels beside scatter points, in secondary ink, nudged where they would collide."""
    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y), xytext=LABEL_NUDGE.get(name, (6, 3)), textcoords="offset points",
                    fontsize=8, color=v.INK_2)


def section_clustering() -> None:
    sec = "E_clustering"
    print("Section E: hierarchical clustering + k-means")

    # E1 Global: 15 FAO fishing areas -------------------------------------------
    o = v.owid("share-of-fish-stocks-overexploited")
    o = o[o["Entity"].str.endswith("(FAO)")]
    feats = {}
    for ent, g in o.groupby("Entity"):
        s = g.set_index("Year").iloc[:, -1].astype(float)
        b, a = np.polyfit(s.index, s.values, 1)
        feats[ent.replace(" (FAO)", "")] = {
            "mean_pct_overexploited": s.mean(),
            "trend_pp_per_year": b,
            "volatility_sd_resid": np.std(s.values - (a + b * s.index), ddof=1),
            "latest_pct_overexploited": s.iloc[-1],
        }
    gf = pd.DataFrame(feats).T
    X1 = gf[["mean_pct_overexploited", "trend_pp_per_year", "volatility_sd_resid"]]
    c1 = cluster(X1, range(2, 7))
    out1 = gf.assign(kmeans_cluster=c1["km"], hierarchical_cluster=c1["hc"])
    save_table(out1.reset_index(names="fao_area"), "E1_global_fao_area_clusters")
    record("E", f"K-means + Ward hierarchical clustering (k = {c1['k']} by silhouette)",
           "15 FAO fishing areas, 2004–2021", "15",
           f"silhouette = {c1['silhouette'][c1['k']]:.2f}", None, f"ARI k-means vs hierarchical = {c1['ari']:.2f}",
           "Areas group by overexploitation level and trend; see E1 table for members")

    # E2 Southeast Asia: 11 EEZs -----------------------------------------------
    p = eez_panel()
    rows = {}
    for eez, g in p.groupby("eez"):
        g = g.set_index("year")
        rec = g.loc[2015:2019]
        win = g.loc[1990:2019]
        catch5 = g["catch"].rolling(5, center=True).mean()
        rows[eez_name(eez)] = {
            "pct_overexploited_2015_19": rec["over"].mean(),
            "overexploited_trend_pp_per_yr_1990_2019": np.polyfit(win.index, win["over"], 1)[0],
            "mtl_trend_per_decade_1990_2019": np.polyfit(win.index, win["mtl"], 1)[0] * 10,
            "unreported_share_2015_19": (rec["unrep_share"] * rec["catch"]).sum() / rec["catch"].sum(),
            "foreign_fleet_share_2015_19": (rec["foreign_share"] * rec["catch"]).sum() / rec["catch"].sum(),
            "catch_vs_peak_pct": (rec["catch"].mean() / catch5.max() - 1) * 100,
        }
    ef = pd.DataFrame(rows).T
    # Step 1: all 11 EEZs. A cluster with a single member is an outlier, not a group,
    # so it is set aside and the remaining EEZs are clustered again (step 2).
    c2 = cluster(ef, range(2, 6))
    sizes = pd.Series(c2["km"]).value_counts()
    outliers = [ef.index[i] for i, k in enumerate(c2["km"]) if sizes[k] == 1]
    rest = ef.drop(outliers)
    c3 = cluster(rest, range(2, 6))
    final = pd.Series(c3["km"], index=rest.index).reindex(ef.index).fillna(0).astype(int)  # 0 = outlier
    out2 = ef.assign(step1_kmeans=c2["km"], step1_hierarchical=c2["hc"],
                     step2_hierarchical=pd.Series(c3["hc"], index=rest.index).reindex(ef.index),
                     final_cluster=final)
    save_table(out2.reset_index(names="eez"), "E2_sea_eez_clusters")
    profile = out2.groupby("final_cluster")[list(ef.columns)].mean()
    profile.insert(0, "members", out2.groupby("final_cluster").apply(lambda d: "; ".join(d.index)))
    save_table(profile.reset_index(), "E3_sea_cluster_profiles")
    record("E", f"Step 1: k-means + Ward clustering, all 11 EEZs (k = {c2['k']} by silhouette)",
           "11 SEA EEZs, 6 indicators", "11", f"silhouette = {c2['silhouette'][c2['k']]:.2f}", None,
           f"ARI k-means vs hierarchical = {c2['ari']:.2f}",
           f"Single-member cluster → outlier: {', '.join(outliers) or 'none'}")
    record("E", f"Step 2: k-means + Ward clustering without the outlier (k = {c3['k']} by silhouette)",
           f"{len(rest)} SEA EEZs, 6 indicators", str(len(rest)), f"silhouette = {c3['silhouette'][c3['k']]:.2f}",
           None, f"ARI k-means vs hierarchical = {c3['ari']:.2f}",
           "Weak-to-moderate structure (silhouette < 0.5); see E3 cluster profiles")

    short = {"pct_overexploited_2015_19": "% overexploited\n2015–19",
             "overexploited_trend_pp_per_yr_1990_2019": "Overexploited trend\npp/yr 1990–2019",
             "mtl_trend_per_decade_1990_2019": "MTL trend\nper decade",
             "unreported_share_2015_19": "Unreported\nshare %",
             "foreign_fleet_share_2015_19": "Foreign fleet\nshare %",
             "catch_vs_peak_pct": "Catch vs\npeak %"}
    # Cluster colours: start with hues not used for countries elsewhere (green, violet, red).
    clr = dict(zip(range(1, 7), [v.SERIES[5], v.SERIES[6], v.SERIES[7], v.SERIES[3], v.SERIES[1], v.SERIES[0]]))
    clr[0] = v.NEUTRAL   # outlier
    cname = {0: "Outlier"} | {k: f"Cluster {k}" for k in range(1, 7)}

    # Figure E.01: global dendrogram + scatter -----------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
    dendrogram(c1["link"], labels=list(X1.index), orientation="left", ax=a1,
               color_threshold=0, above_threshold_color=v.INK_2, leaf_font_size=9)
    a1.set_xlabel("Ward linkage distance")
    a1.grid(axis="y", visible=False)
    a1.grid(axis="x")
    for lab in a1.get_yticklabels():
        k = out1.loc[lab.get_text(), "kmeans_cluster"]
        lab.set_color(v.INK)
        lab.set_bbox({"facecolor": clr[k], "alpha": 0.25, "edgecolor": "none", "pad": 1.5})
    a1.set_title("Hierarchical clustering (labels shaded by k-means cluster)", fontsize=10)
    for k in sorted(out1["kmeans_cluster"].unique()):
        g = out1[out1["kmeans_cluster"] == k]
        a2.scatter(g["mean_pct_overexploited"], g["trend_pp_per_year"], s=70, color=clr[k],
                   edgecolor=v.SURFACE, linewidth=2, zorder=3, label=f"Cluster {k} (n = {len(g)})")
    label_points(a2, out1["mean_pct_overexploited"], out1["trend_pp_per_year"], out1.index)
    a2.axhline(0, color=v.AXIS, lw=0.8)
    a2.set_xlabel("Mean % of stocks overexploited, 2004–2021")
    a2.set_ylabel("Trend (pp per year)")
    a2.grid(axis="x")
    a2.legend(loc="upper left")
    a2.set_title("K-means clusters in the two main features", fontsize=10)
    v.suptitle(fig, f"E.01 — Clustering the 15 FAO fishing areas (k = {c1['k']}, chosen by silhouette)",
               f"Features: mean level, trend, volatility (standardised). Silhouette {c1['silhouette'][c1['k']]:.2f}; "
               f"k-means and hierarchical agree with ARI = {c1['ari']:.2f}")
    v.layout(fig, 0)
    save_fig(fig, sec, "E.01_global_fao_areas_clusters", "FAO via OWID")

    # Figure E.02: standardised feature heatmap + dendrogram ----------------
    # Heatmap on the left (EEZ names as its row labels), dendrogram on the right with
    # its leaves facing the heatmap, so long names are never cut off.
    fig = plt.figure(figsize=(15, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.6, 0.9, 0.08], wspace=0.03)
    a2, a1, cax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    dn = dendrogram(c2["link"], labels=list(ef.index), orientation="right", ax=a1, no_labels=True,
                    color_threshold=0, above_threshold_color=v.INK_2)
    a1.grid(axis="y", visible=False)
    a1.grid(axis="x")
    a1.set_xlabel("Ward linkage distance")
    a1.spines["left"].set_visible(False)
    a1.set_yticks([])
    leaves = dn["ivl"]                       # bottom-to-top order in the dendrogram
    zz = c2["z"].loc[leaves]
    raw = ef.loc[leaves]
    im = a2.imshow(zz.to_numpy(), cmap=v.DIVERGING, vmin=-2.5, vmax=2.5, aspect="auto",
                   origin="lower", extent=(0, zz.shape[1], 0, 10 * zz.shape[0]))
    fmt = ["{:.0f}", "{:+.2f}", "{:+.3f}", "{:.0f}", "{:.0f}", "{:+.0f}"]
    for i in range(zz.shape[0]):
        for j in range(zz.shape[1]):
            a2.text(j + 0.5, 10 * i + 5, fmt[j].format(raw.iloc[i, j]), ha="center", va="center", fontsize=9,
                    color="#ffffff" if abs(zz.iloc[i, j]) > 1.5 else v.INK)
    a2.set_yticks(10 * np.arange(len(leaves)) + 5, leaves, fontsize=9)
    for lab in a2.get_yticklabels():
        k = out2.loc[lab.get_text(), "final_cluster"]
        lab.set_color(v.INK)
        lab.set_bbox({"facecolor": clr[k], "alpha": 0.3, "edgecolor": "none", "pad": 1.5})
    a2.set_xticks(np.arange(zz.shape[1]) + 0.5, [short[c] for c in zz.columns], fontsize=8.5)
    a2.xaxis.tick_top()
    a2.grid(False)
    for spn in a2.spines.values():
        spn.set_visible(False)
    a1.set_ylim(0, 10 * zz.shape[0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Standardised value (z)", color=v.INK_2)
    cb.outline.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=clr[k], alpha=0.4) for k in sorted(final.unique())]
    fig.legend(handles, [f"{cname[k]} (n = {(final == k).sum()})" for k in sorted(final.unique())],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, title="Name shading = final cluster",
               title_fontsize=9)
    v.suptitle(fig, f"E.02 — Clustering the 11 Southeast Asian EEZs: {', '.join(outliers)} is an outlier, "
               f"the other {len(rest)} form {c3['k']} groups",
               "Cells show raw values, colour = z-score (red = high, blue = low). "
               f"Step-2 k-means vs hierarchical ARI = {c3['ari']:.2f}")
    fig.subplots_adjust(top=0.76, bottom=0.15, left=0.17, right=0.95)
    save_fig(fig, sec, "E.02_sea_eez_dendrogram_heatmap", "Sea Around Us (11 SEA EEZs)")

    # Figure E.03: silhouette vs k + PCA map of the EEZs -------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.4]})
    for (name, c), colour in zip([("15 FAO areas", c1), ("11 SEA EEZs (step 1)", c2),
                                  (f"{len(rest)} SEA EEZs without outlier (step 2)", c3)],
                                 [v.SERIES[0], v.SERIES[1], v.SERIES[2]]):
        ks = list(c["silhouette"])
        a1.plot(ks, [c["silhouette"][k] for k in ks], color=colour, marker="o", markersize=7,
                markeredgecolor=v.SURFACE, markeredgewidth=1.5, label=name)
        a1.scatter([c["k"]], [c["silhouette"][c["k"]]], s=160, facecolor="none", edgecolor=v.INK, lw=1.2, zorder=4)
    a1.set_xlabel("Number of clusters k")
    a1.set_ylabel("Mean silhouette score")
    a1.set_xticks(range(2, 7))
    a1.legend(loc="upper right")
    a1.set_title("Choosing k: highest silhouette (circled)", fontsize=10)
    pca = PCA(n_components=2).fit(c2["z"])
    pcs = pca.transform(c2["z"])
    for k in sorted(final.unique()):
        m = (final == k).to_numpy()
        a2.scatter(pcs[m, 0], pcs[m, 1], s=80, color=clr[k], edgecolor=v.SURFACE, linewidth=2, zorder=3,
                   label=f"{cname[k]} (n = {m.sum()})")
    label_points(a2, pcs[:, 0], pcs[:, 1], ef.index)
    a2.axhline(0, color=v.AXIS, lw=0.8)
    a2.axvline(0, color=v.AXIS, lw=0.8)
    a2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.0f}% of variance)")
    a2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.0f}% of variance)")
    a2.grid(axis="x")
    a2.legend(loc="best")
    a2.set_title("SEA EEZs projected on 2 principal components (display only)", fontsize=10)
    v.suptitle(fig, "E.03 — How many clusters, and how well separated are the SEA EEZs?",
               "Silhouette: +1 = well separated, 0 = overlapping. PCA squeezes the 6 indicators into 2 axes for plotting")
    v.layout(fig, 0)
    save_fig(fig, sec, "E.03_silhouette_and_pca", "FAO via OWID; Sea Around Us")


# =========================================================================== #
SECTIONS = {
    "profile": section_profile,
    "assumptions": section_assumptions,
    "ranksum": section_ranksum,
    "chisquare": section_chisquare,
    "clustering": section_clustering,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the course-syllabus statistical tests.")
    parser.add_argument("--only", nargs="+", choices=SECTIONS, help="run only these sections")
    args = parser.parse_args()
    v.setup_style()
    for name in args.only or SECTIONS:
        SECTIONS[name]()
    if SUMMARY:
        s = pd.DataFrame(SUMMARY)
        path = RES / ("test_summary.csv" if not args.only else f"test_summary_{'_'.join(args.only)}.csv")
        RES.mkdir(parents=True, exist_ok=True)
        s.to_csv(path, index=False)
        print("  saved", path.relative_to(ROOT))
    print("Done ->", HERE.relative_to(ROOT))


if __name__ == "__main__":
    main()
