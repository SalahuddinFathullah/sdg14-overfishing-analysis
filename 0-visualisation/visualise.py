"""
visualise.py — Exploratory visualisation of every dataset in data/raw/
=====================================================================

KIE4033 Assignment 1 · SDG 14 overfishing analysis (Global -> Southeast Asia)

Run from the repository root (or anywhere — paths are resolved from this file):

    python 0-visualisation/visualise.py            # all figures
    python 0-visualisation/visualise.py --only global sea_sau

Figures are written to ``0-visualisation/figures/<section>/`` as PNG (200 dpi),
ready to paste into the report. One dataset is often shown with several chart
types, each answering a different question (trend, composition, distribution,
relationship, comparison).

Sections
--------
1_global      Stage 1 — FAO stock status & world capture (H1a, H1b)
2_sea_owid    Southeast Asia country series from OWID (capture, gear, discards)
3_sea_sau     Southeast Asia Sea Around Us EEZ series (stock status, MTL, catch) (H2a-H2c)
4_validation  Main datasets vs alternative sources (data validity checks)
5_integrity   Is reported catch plausible? GDP, mirror trade, Benford, reconstruction gap

Style
-----
Colours follow a colour-vision-deficiency-checked categorical palette in a fixed
order (a country always keeps the same colour in every figure); one y-axis per
chart; 2 px lines; hairline solid gridlines; a legend whenever there are two
or more series.

SEA5 = Malaysia, Indonesia, Thailand, Vietnam, Philippines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from statsmodels.tsa.stattools import acf

# =========================================================================== #
# Paths
# =========================================================================== #
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "raw"
OUT = HERE / "figures"

# =========================================================================== #
# Style: palette, ink and chart chrome
# =========================================================================== #
SURFACE = "#fcfcfb"      # chart background
INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text
MUTED = "#898781"        # axis ticks / labels
GRID = "#e1e0d9"         # hairline gridlines
AXIS = "#c3c2b7"         # baseline / axis
NEUTRAL = "#c3c2b7"      # neutral / de-emphasised marks

# Categorical palette — fixed order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]

SEA5 = ["Malaysia", "Indonesia", "Thailand", "Vietnam", "Philippines"]
# A country keeps the same colour in every figure.
COUNTRY_COLOR = dict(zip(SEA5, SERIES[:5]))
WORLD_COLOR = INK_2

# Sequential (magnitude) ramp: one hue, light -> dark.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
# Diverging ramp: blue <-> red with a neutral gray midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#5598e7", "#f0efec", "#ec835a", "#b8302f"])

# Stock-status categories, ordered from worst to best, gray = neutral midpoint.
STATUS_ORDER = ["Collapsed", "Over-exploited", "Exploited", "Developing", "Rebuilding"]
STATUS_COLOR = dict(zip(STATUS_ORDER, ["#e34948", "#eda100", "#c3c2b7", "#2a78d6", "#1baf7a"]))

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


def setup_style() -> None:
    """Global matplotlib defaults so every figure shares the same quiet look."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ["Inter", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"]
                   if f in available), "sans-serif")
    plt.rcParams.update({
        "font.family": family,
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


HEADER_IN = 0.75  # inches reserved at the top of every figure for title + subtitle


def header_top(fig) -> float:
    """Figure fraction below which the plotting area must stay (under the header)."""
    return 1 - HEADER_IN / fig.get_figheight()


def suptitle(fig, title: str, subtitle: str | None = None) -> None:
    """
    Left-aligned figure title + subtitle at fixed distances (in inches) from the
    top edge, so they never collide whatever the figure size; the axes are then
    pushed below the header.
    """
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.05 / h, ha="left", va="top",
                 fontsize=13, fontweight="semibold", color=INK)
    if subtitle:
        fig.text(0.01, 1 - 0.36 / h, subtitle, ha="left", va="top", fontsize=9, color=INK_2)
    # Squeeze every axes (colorbars included) proportionally below the header,
    # so space already reserved for a colorbar or legend is kept.
    top = max(ax.get_position().y1 for ax in fig.axes)
    # Panel titles sit above their axes: leave room for them under the subtitle.
    target = header_top(fig) - (0.22 / h if any(ax.get_title() for ax in fig.axes) else 0)
    if top > target:
        k = target / top
        for ax in fig.axes:
            p = ax.get_position()
            ax.set_position([p.x0, p.y0 * k, p.width, p.height * k])


def layout(fig, bottom: float = 0.0) -> None:
    """tight_layout for multi-panel figures, keeping the header space free."""
    fig.tight_layout(rect=(0, bottom, 1, header_top(fig)))


def save(fig, section: str, name: str, source: str) -> Path:
    """Add a source line, save as PNG under figures/<section>/ and close."""
    fig.text(0.01, -0.01, f"Source: {source}", ha="left", va="top", fontsize=8, color=MUTED)
    path = OUT / section / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("  saved", path.relative_to(ROOT))
    return path


def end_label(ax, x, y, text, dx=0.6, **kw) -> None:
    """Direct label at the end of a line, in text ink (never the series colour)."""
    ax.annotate(text, (x, y), xytext=(dx * 6, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9, color=INK_2, **kw)


def end_labels(ax, items: list[tuple[float, float, str]], min_gap: float = 0.06) -> None:
    """
    Direct end labels for several lines, but only for lines whose end is clear of
    its neighbours (closer than `min_gap` of the y-range -> no label; the legend
    carries those). Labels are never nudged away from their line.
    """
    lo, hi = ax.get_ylim()
    gap = min_gap * (hi - lo)
    ys = sorted(y for _, y, _ in items)
    for x, y, text in items:
        others = [o for o in ys if o != y]
        if all(abs(y - o) >= gap for o in others):
            end_label(ax, x, y, text)


def legend_right(ax, **kw) -> None:
    """Legend outside the plot on the right, used when it would cover the data."""
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), **kw)


def millions(ax, axis="y") -> None:
    """Show tonnes as millions of tonnes on an axis (clean tick labels: 0, 0.5, 1, 15...)."""
    fmt = mticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def stacked_area(ax, x, ys: list, colors: list, labels: list) -> None:
    """Stacked area with a thin surface-coloured edge acting as the gap between layers."""
    ax.stackplot(x, *ys, colors=colors, labels=labels, edgecolor=SURFACE, linewidth=1.0, alpha=0.95)


# =========================================================================== #
# Data loading
# =========================================================================== #
def owid(slug: str) -> pd.DataFrame:
    """Load an OWID grapher CSV (columns: Entity, Code, Year, <value columns>)."""
    return pd.read_csv(RAW / "owid" / f"{slug}.csv")


def owid_series(slug: str, entity: str, col: int = -1) -> pd.Series:
    """One entity's series from an OWID CSV, indexed by year."""
    df = owid(slug)
    s = df[df["Entity"] == entity].set_index("Year").iloc[:, col]
    return s.astype(float).sort_index()


def eez_file(folder: str, eez_id: int) -> Path:
    return next((RAW / "sau" / folder).glob(f"{eez_id}_*.csv"))


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


# =========================================================================== #
# Section 1 — Global (Stage 1)
# =========================================================================== #
def section_global() -> None:
    sec = "1_global"
    print("Section 1: global")
    over = owid_series("share-of-fish-stocks-overexploited", "World")
    sust = owid_series("fish-stocks-within-sustainable-levels", "World", col=-2)
    capture = owid_series("capture-fishery-production", "World")
    aqua = owid_series("aquaculture-farmed-fish-production", "World")
    src_fao = "FAO via Our World in Data"

    # 1.1 Trend line with pre/post-2015 fitted segments (H1a) -----------------
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(over.index, over.values, color=SERIES[0], label="Overexploited stocks")
    for lo, hi in [(1974, 2015), (2015, 2021)]:
        seg = over.loc[lo:hi]
        b, a = np.polyfit(seg.index, seg.values, 1)
        xs = np.array([lo, hi])
        ax.plot(xs, a + b * xs, color=INK_2, lw=1.2, label="Linear fit (before / after 2015)" if lo == 1974 else None)
        ax.annotate(f"{b:+.2f} pp/yr", (xs.mean(), a + b * xs.mean()), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=9, color=INK_2)
    ax.axvline(2015, color=AXIS, lw=1)
    ax.text(2015.3, over.min(), "SDGs adopted (2015)", fontsize=9, color=MUTED, va="bottom")
    ax.scatter([over.index[-1]], [over.iloc[-1]], s=40, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    end_label(ax, over.index[-1], over.iloc[-1], f"{over.iloc[-1]:.1f}%")
    ax.set_ylabel("% of assessed stocks overexploited")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    suptitle(fig, "Share of world fish stocks that are overexploited, 1974–2021",
             "Overexploitation has risen almost every decade; the slope after 2015 is the SDG 14.4 test (H1a)")
    save(fig, sec, "1.01_world_overexploited_trend", src_fao)

    # 1.2 Composition: 100% stacked area (sustainable vs overexploited) -------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    stacked_area(ax, over.index, [over.values, sust.reindex(over.index).values],
                 [SERIES[7], SERIES[0]], ["Overexploited", "Within sustainable levels (SDG 14.4.1)"])
    ax.set_ylim(0, 100)
    ax.set_xlim(over.index.min(), over.index.max())
    ax.set_ylabel("% of assessed stocks")
    legend_right(ax)
    suptitle(fig, "World fish stocks: overexploited vs sustainable", "The two shares always sum to 100%")
    save(fig, sec, "1.02_world_stock_composition", src_fao)

    # 1.3 Capture vs aquaculture (same unit, one axis) ------------------------
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for s, c, lab in [(capture, SERIES[0], "Wild capture"), (aqua, SERIES[2], "Aquaculture (farmed)")]:
        ax.plot(s.index, s.values, color=c, label=lab)
        end_label(ax, s.index[-1], s.iloc[-1], f"{s.iloc[-1] / 1e6:.0f} Mt")
    millions(ax)
    ax.set_ylabel("Million tonnes per year")
    ax.legend(loc="upper left")
    suptitle(fig, "World fish production: wild catch has plateaued since the 1990s",
             "Growth in supply now comes from farming, not from more wild fishing")
    save(fig, sec, "1.03_world_capture_vs_aquaculture", "FAO via World Bank / Our World in Data")

    # 1.4 Relationship: connected scatter capture vs % overexploited (H1b) ----
    both = pd.concat([capture.rename("capture"), over.rename("over")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(both["capture"], both["over"], color=GRID, lw=1.2, zorder=1)
    sc = ax.scatter(both["capture"], both["over"], c=both.index, cmap=SEQ_BLUE, s=42,
                    edgecolor=SURFACE, linewidth=1.5, zorder=2)
    for yr in [both.index.min(), 1996, both.index.max()]:
        ax.annotate(str(yr), (both.loc[yr, "capture"], both.loc[yr, "over"]), xytext=(6, -2),
                    textcoords="offset points", fontsize=9, color=INK_2)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Year", color=INK_2)
    cb.outline.set_visible(False)
    millions(ax, "x")
    ax.set_xlabel("World wild capture (million tonnes)")
    ax.set_ylabel("% of stocks overexploited")
    ax.grid(axis="x")
    rho = stats.spearmanr(both["capture"], both["over"])[0]
    suptitle(fig, "Catch vs overexploitation, 1974–2021",
             f"Overexploitation kept rising after catch stopped growing (Spearman ρ = {rho:.2f}, levels)")
    save(fig, sec, "1.04_capture_vs_overexploited_connected_scatter", src_fao)

    # 1.5 ACF of each series, levels vs first differences ---------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    for row, (s, name) in enumerate([(over, "% stocks overexploited"), (capture, "World capture")]):
        for col, (series, kind) in enumerate([(s, "levels"), (s.diff().dropna(), "first differences")]):
            ax = axes[row, col]
            vals = acf(series.values, nlags=15, fft=False)
            conf = 1.96 / np.sqrt(len(series))
            ax.axhspan(-conf, conf, color=GRID, alpha=0.6, lw=0)
            ax.vlines(range(len(vals)), 0, vals, color=SERIES[0], lw=2)
            ax.scatter(range(len(vals)), vals, s=18, color=SERIES[0], zorder=3)
            ax.axhline(0, color=AXIS, lw=0.8)
            ax.set_title(f"{name} — {kind}", fontsize=10)
            ax.set_xlabel("Lag (years)")
    axes[0, 0].set_ylabel("Autocorrelation")
    axes[1, 0].set_ylabel("Autocorrelation")
    suptitle(fig, "Autocorrelation (ACF): levels are trending, differences are close to white noise",
             "Shaded band = 95% bound for no autocorrelation. Correlate differenced series, not levels")
    layout(fig, 0)
    save(fig, sec, "1.05_acf_levels_vs_differences", src_fao)

    # 1.6 Distribution of year-to-year changes (shows interpolated years) -----
    d = over.diff().dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax1.bar(d.index, d.values, width=0.7, color=[SERIES[7] if v > 0 else SERIES[0] for v in d.values])
    ax1.axhline(0, color=AXIS, lw=0.8)
    ax1.set_ylabel("Change vs previous year (pp)")
    ax1.set_title("Year-to-year change", fontsize=10)
    ax2.hist(d.values, bins=14, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
    ax2.set_title("Distribution of changes", fontsize=10)
    ax2.set_xlabel("pp per year")
    repeats = d.round(2).value_counts()
    suptitle(fig, "Annual changes in world % overexploited",
             f"Red = increase. Identical consecutive steps (e.g. {repeats.index[0]:+.2f} pp × {repeats.iloc[0]}) "
             "reveal linear interpolation between FAO assessments")
    layout(fig, 0)
    save(fig, sec, "1.06_world_overexploited_annual_changes", src_fao)

    # 1.7 Dumbbell: FAO fishing areas 2004 -> 2021 ----------------------------
    df = owid("share-of-fish-stocks-overexploited")
    areas = df[df["Entity"].str.endswith("(FAO)")].pivot(index="Entity", columns="Year", values=df.columns[-1])
    areas = areas[[2004, 2021]].dropna().sort_values(2021)
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(areas))
    sea_areas = {"Western Central Pacific (FAO)", "Eastern Indian Ocean (FAO)"}
    for i, (name, r) in enumerate(areas.iterrows()):
        hi = name in sea_areas
        ax.plot([r[2004], r[2021]], [i, i], color=SERIES[0] if hi else NEUTRAL, lw=2, zorder=1)
    ax.scatter(areas[2004], y, s=46, color=SURFACE, edgecolor=INK_2, linewidth=1.5, zorder=2, label="2004")
    ax.scatter(areas[2021], y, s=52, color=[SERIES[0] if n in sea_areas else INK_2 for n in areas.index],
               edgecolor=SURFACE, linewidth=2, zorder=3, label="2021")
    ax.set_yticks(y, [n.replace(" (FAO)", "") + ("  ← SEA" if n in sea_areas else "") for n in areas.index])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")
    ax.set_xlabel("% of stocks overexploited")
    ax.legend(loc="lower right")
    suptitle(fig, "Overexploited stocks by FAO fishing area, 2004 → 2021",
             "Blue = areas covering Southeast Asia (71 Western Central Pacific, 57 Eastern Indian Ocean)")
    save(fig, sec, "1.07_fao_areas_dumbbell_2004_2021", src_fao)


# =========================================================================== #
# Section 2 — Southeast Asia from OWID
# =========================================================================== #
def section_sea_owid() -> None:
    sec = "2_sea_owid"
    print("Section 2: Southeast Asia (OWID)")
    cap = owid("capture-fishery-production").pivot(index="Year", columns="Entity", values="Capture fisheries production (metric tons)")
    aqua = owid("aquaculture-farmed-fish-production").pivot(index="Year", columns="Entity", values="Aquaculture production (metric tons)")

    # 2.1 Capture per country (lines, direct labels + legend) -----------------
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ends = []
    for c in SEA5:
        s = cap[c].dropna()
        ax.plot(s.index, s.values, color=COUNTRY_COLOR[c], label=c)
        ends.append((s.index[-1], s.iloc[-1], f"{c} {s.iloc[-1] / 1e6:.1f} Mt"))
    millions(ax)
    ax.set_ylabel("Million tonnes per year")
    ax.set_xlim(None, cap.index.max() + 12)
    ax.set_xticks(range(1960, 2030, 10))
    end_labels(ax, ends)
    ax.legend(loc="upper left")
    suptitle(fig, "Wild fish capture in Southeast Asia, 1960–2024",
             "Indonesia dominates; Thailand peaked in the 1990s and has fallen since")
    save(fig, sec, "2.01_sea_capture_by_country", "FAO via World Bank / Our World in Data")

    # 2.2 Composition: stacked area of the regional total --------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    stacked_area(ax, cap.index, [cap[c].fillna(0).values for c in SEA5], [COUNTRY_COLOR[c] for c in SEA5], SEA5)
    millions(ax)
    ax.set_xlim(cap.index.min(), cap.index.max())
    ax.set_ylabel("Million tonnes per year")
    ax.legend(loc="upper left", ncol=5)
    share = cap[SEA5].sum(axis=1) / cap["World"] * 100
    suptitle(fig, "SEA5 combined wild capture and each country's share",
             f"SEA5 share of world wild catch: {share.iloc[0]:.0f}% (1960) → {share.iloc[-1]:.0f}% (2024)")
    save(fig, sec, "2.02_sea_capture_stacked", "FAO via World Bank / Our World in Data")

    # 2.3 Small multiples: capture vs aquaculture per country -----------------
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharey=False)
    for ax, c in zip(axes, SEA5):
        ax.plot(cap.index, cap[c], color=SERIES[0], label="Wild capture")
        ax.plot(aqua.index, aqua[c], color=SERIES[2], label="Aquaculture")
        millions(ax)
        ax.set_title(c, fontsize=10)
    axes[0].set_ylabel("Million tonnes")
    axes[0].legend(loc="upper left")
    suptitle(fig, "Wild capture vs aquaculture by country", "Each panel has its own y-scale")
    layout(fig, 0)
    save(fig, sec, "2.03_sea_capture_vs_aquaculture", "FAO via World Bank / Our World in Data")

    # 2.4 100% stacked bars by decade: catch by gear --------------------------
    gear = owid("fish-catch-gear-type")
    gear_cols = ["Bottom trawl", "Pelagic trawl", "Purse seine", "Small scale",
                 "Gillnet", "Longline", "Other gear", "Unknown gear"]
    gear["decade"] = (gear["Year"] // 10) * 10
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.4), sharey=True)
    for ax, c in zip(axes, SEA5):
        g = gear[gear["Entity"] == c].groupby("decade")[gear_cols].sum()
        pct = g.div(g.sum(axis=1), axis=0) * 100
        bottom = np.zeros(len(pct))
        for col, colour in zip(gear_cols, SERIES):
            ax.bar(pct.index.astype(str), pct[col], bottom=bottom, color=colour, width=0.7,
                   edgecolor=SURFACE, linewidth=1, label=col)
            bottom += pct[col].values
        ax.set_title(c, fontsize=10)
        ax.tick_params(axis="x", labelrotation=90)
    axes[0].set_ylabel("% of catch")
    axes[0].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.08))
    suptitle(fig, "How fish are caught: share of catch by gear type, by decade",
             "Bottom trawling — the most damaging gear for seabed habitats — grew sharply in Thailand, Malaysia and Vietnam")
    layout(fig, 0.02)
    save(fig, sec, "2.04_sea_gear_share_by_decade", "Sea Around Us via Our World in Data")

    # 2.5 Discard rate (% of catch thrown back) -------------------------------
    disc = owid("fish-discards").pivot(index="Year", columns="Entity", values="Discards")
    total = gear.groupby(["Year", "Entity"])[gear_cols].sum().sum(axis=1).unstack()
    rate = (disc / total * 100).loc[:, SEA5 + ["World"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for c in SEA5:
        s = rate[c].dropna()
        ax.plot(s.index, s.values, color=COUNTRY_COLOR[c], label=c)
    ax.plot(rate.index, rate["World"], color=WORLD_COLOR, lw=1.4, label="World")
    ax.set_ylabel("Discards as % of catch")
    ax.set_ylim(0, None)
    legend_right(ax)
    suptitle(fig, "Discard rate: share of the catch thrown back dead",
             "World discards have fallen; check whether SEA follows (gaps = no data, not zero)")
    save(fig, sec, "2.05_sea_discard_rate", "Sea Around Us via Our World in Data")


# =========================================================================== #
# Section 3 — Southeast Asia, Sea Around Us EEZ data (Stage 2)
# =========================================================================== #
def section_sea_sau() -> None:
    sec = "3_sea_sau"
    print("Section 3: Southeast Asia (Sea Around Us)")
    src = "Sea Around Us (v50-1), EEZ data"
    d = load_sau_eez()
    stock, mti, catch = d["stock"], d["mti"], d["catch"]

    # 3.1 Small multiples: stock status stacked area per EEZ ------------------
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5), sharex=True, sharey=True)
    for ax, (eez, country) in zip(axes.flat, EEZ_COUNTRY.items()):
        s = stock[stock["eez"] == eez].set_index("year")
        stacked_area(ax, s.index, [s[c].values for c in STATUS_ORDER],
                     [STATUS_COLOR[c] for c in STATUS_ORDER], STATUS_ORDER)
        ax.set_title(f"{country} — {EEZ_LABEL[eez]}" if country != EEZ_LABEL[eez] else country, fontsize=10)
        ax.set_xlim(1950, 2019)
        ax.set_ylim(0, 100)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles[::-1], labels[::-1], loc="center", title="Stock status", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of stocks")
    suptitle(fig, "Stock status in each Southeast Asian EEZ, 1950–2019",
             "Red + yellow = collapsed or over-exploited. Status is derived from catch relative to its peak")
    layout(fig, 0)
    save(fig, sec, "3.01_stock_status_by_eez", src)

    # 3.2 Heatmap: % collapsed + over-exploited, EEZ x year -------------------
    stock["over"] = stock["Collapsed"] + stock["Over-exploited"]
    hm = stock.pivot(index="eez", columns="year", values="over")
    order = hm[2019].sort_values().index
    hm = hm.loc[order]
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(hm.values, aspect="auto", cmap=SEQ_BLUE, vmin=0, vmax=100, interpolation="nearest")
    ax.set_yticks(range(len(hm)), [f"{EEZ_COUNTRY[e]} — {EEZ_LABEL[e]}" if EEZ_COUNTRY[e] != EEZ_LABEL[e]
                                   else EEZ_COUNTRY[e] for e in hm.index])
    years = hm.columns.to_numpy()
    ticks = np.where(years % 10 == 0)[0]
    ax.set_xticks(ticks, years[ticks])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("% of stocks collapsed or over-exploited", color=INK_2)
    cb.outline.set_visible(False)
    suptitle(fig, "Overfishing intensity across SEA EEZs over time",
             "Darker = larger share of stocks collapsed or over-exploited. Sorted by 2019 value")
    save(fig, sec, "3.02_overexploited_heatmap_eez_year", src)

    # 3.3 Country lines vs world (H2c preview) --------------------------------
    cty = stock.groupby(["country", "year"])["over"].mean().unstack(0)
    world = sau_overexploited(RAW / "validation" / "sau_regions" / "stock_status" / "global.csv")
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for c in SEA5:
        ax.plot(cty.index, cty[c], color=COUNTRY_COLOR[c], label=c)
    ax.plot(world.index, world.values, color=WORLD_COLOR, lw=2.6, label="World (SAU)")
    end_label(ax, world.index[-1], world.iloc[-1], "World")
    ax.set_ylabel("% of stocks collapsed or over-exploited")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", ncol=2)
    suptitle(fig, "Southeast Asia vs world: share of stocks overfished",
             "Country = unweighted mean of its EEZs. H2c tests whether the SEA slope is steeper than the world's")
    save(fig, sec, "3.03_overexploited_country_vs_world", src)

    # 3.4 Mean trophic level, small multiples by country ----------------------
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharey=True)
    for ax, c in zip(axes, SEA5):
        eezs = [e for e, cc in EEZ_COUNTRY.items() if cc == c]
        for i, e in enumerate(eezs):
            m = mti[mti["eez"] == e]
            ax.plot(m["year"], m["mean_trophic_level"], color=SERIES[i], lw=1.6, label=EEZ_LABEL[e])
        ax.set_title(c, fontsize=10)
        if len(eezs) > 1:
            ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("Mean trophic level of catch")
    suptitle(fig, "Mean trophic level (MTL) of the catch by EEZ",
             "A falling MTL = 'fishing down the food web': large predators depleted, catches shift to smaller species")
    layout(fig, 0)
    save(fig, sec, "3.04_mean_trophic_level_by_eez", src)

    # 3.5 'Fishing down' connected scatter: catch vs MTL per country ---------
    cm = country_mtl(mti)
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.4), gridspec_kw={"wspace": 0.35})
    for ax, c in zip(axes, SEA5):
        g = cm[cm["country"] == c].set_index("year")
        ax.plot(g["catch"], g["mtl"], color=GRID, lw=1, zorder=1)
        sc = ax.scatter(g["catch"], g["mtl"], c=g.index, cmap=SEQ_BLUE, s=16, zorder=2)
        for yr in (1950, 2019):
            ax.annotate(str(yr), (g.loc[yr, "catch"], g.loc[yr, "mtl"]), xytext=(4, 3),
                        textcoords="offset points", fontsize=8, color=INK_2)
        millions(ax, "x")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("Catch (Mt)")
        ax.grid(axis="x")
    axes[0].set_ylabel("Mean trophic level")
    cb = fig.colorbar(sc, ax=axes, pad=0.01, fraction=0.015)
    cb.set_label("Year", color=INK_2)
    cb.outline.set_visible(False)
    suptitle(fig, "Catch vs mean trophic level ('fishing down' plot)",
             "Moving right and down = more catch made of lower-level species; hooking back left = catch falling")
    save(fig, sec, "3.05_fishing_down_catch_vs_mtl", src)

    # 3.6 Distribution: year-on-year MTL change by country (KW preview) ------
    cm = cm.sort_values(["country", "year"])
    cm["d_mtl"] = cm.groupby("country")["mtl"].diff()
    data = [cm.loc[cm["country"] == c, "d_mtl"].dropna().values for c in SEA5]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bp = ax.boxplot(data, tick_labels=SEA5, widths=0.45, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "lw": 1.5}, whiskerprops={"color": MUTED},
                    capprops={"color": MUTED})
    for patch, c in zip(bp["boxes"], SEA5):
        patch.set_facecolor(COUNTRY_COLOR[c])
        patch.set_alpha(0.25)
        patch.set_edgecolor(COUNTRY_COLOR[c])
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(data, SEA5), start=1):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=10, color=COUNTRY_COLOR[c], alpha=0.7, zorder=3)
    ax.axhline(0, color=AXIS, lw=0.8)
    h, p = stats.kruskal(*data)
    ax.set_ylabel("Change in MTL vs previous year")
    suptitle(fig, "Year-on-year change in mean trophic level by country",
             f"Box = middle 50%, line = median, dots = years. Kruskal-Wallis H = {h:.2f}, p = {p:.3f} (preview)")
    save(fig, sec, "3.06_mtl_change_boxplot_by_country", src)

    # 3.7 Mean maximum length of caught species -------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for c in SEA5:
        g = cm[cm["country"] == c]
        ax.plot(g["year"], g["max_length"], color=COUNTRY_COLOR[c], label=c)
    ax.set_ylabel("Mean maximum length (cm)")
    legend_right(ax)
    suptitle(fig, "Average size of the species being caught",
             "Catch-weighted mean maximum body length; a decline means catches are shifting to smaller fish")
    save(fig, sec, "3.07_mean_max_length_by_country", src)

    # 3.8 Catch by fishing sector, SEA total (stacked area) -------------------
    sector = catch.groupby(["year", "fishing_sector"])["tonnes"].sum().unstack().fillna(0)
    sectors = [s for s in ["Industrial", "Artisanal", "Subsistence", "Recreational"] if s in sector]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    stacked_area(ax, sector.index, [sector[s].values for s in sectors], SERIES[:len(sectors)], sectors)
    millions(ax)
    ax.set_xlim(1950, 2019)
    ax.set_ylabel("Million tonnes per year")
    ax.legend(loc="upper left")
    suptitle(fig, "Total catch in SEA5 waters by fishing sector",
             "Industrial fleets drove the growth in catch from the 1960s")
    save(fig, sec, "3.08_sea_catch_by_sector", src)

    # 3.9 Unreported share and foreign-fleet share (two measures -> two panels)
    def share(col, val):
        t = catch.groupby(["country", "year"])["tonnes"].sum()
        part = catch[catch[col] == val].groupby(["country", "year"])["tonnes"].sum()
        return (part / t * 100).unstack(0)
    unrep, foreign = share("reporting_status", "Unreported"), share("fleet", "foreign")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6), sharex=True)
    for c in SEA5:
        a1.plot(unrep.index, unrep[c], color=COUNTRY_COLOR[c], label=c)
        a2.plot(foreign.index, foreign[c], color=COUNTRY_COLOR[c], label=c)
    a1.set_title("Unreported catch (% of total)", fontsize=10)
    a2.set_title("Catch by foreign fleets (% of total)", fontsize=10)
    for a in (a1, a2):
        a.set_ylim(0, None)
        a.set_ylabel("%")
    a1.legend(loc="upper center", bbox_to_anchor=(1.05, -0.12), ncol=5)
    suptitle(fig, "Hidden and foreign fishing pressure in SEA waters",
             "Unreported = catch missing from official FAO statistics, reconstructed by Sea Around Us")
    layout(fig, 0)
    save(fig, sec, "3.09_unreported_and_foreign_share", src)

    # 3.10 Correlation matrices: levels vs first differences ------------------
    over_c = cty
    tot_c = catch.groupby(["country", "year"])["tonnes"].sum().unstack(0)
    rows = []
    for c in SEA5:
        g = cm[cm["country"] == c].set_index("year")
        rows.append(pd.DataFrame({
            "Catch": tot_c[c], "Overfished %": over_c[c], "MTL": g["mtl"],
            "Max length": g["max_length"], "Unreported %": unrep[c], "Foreign %": foreign[c],
        }).assign(country=c))
    panel = pd.concat(rows)
    vars_ = ["Catch", "Overfished %", "MTL", "Max length", "Unreported %", "Foreign %"]
    lev = panel[vars_].corr(method="spearman")
    dif = panel.groupby("country")[vars_].diff().corr(method="spearman")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, m, t in [(axes[0], lev, "Levels"), (axes[1], dif, "Year-to-year differences")]:
        im = ax.imshow(m.values, cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(vars_)), vars_, rotation=40, ha="right")
        ax.set_yticks(range(len(vars_)), vars_)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i in range(len(vars_)):
            for j in range(len(vars_)):
                v = m.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if abs(v) > 0.6 else INK)
        ax.set_title(t, fontsize=10)
    fig.subplots_adjust(bottom=0.2)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Spearman ρ", color=INK_2)
    cb.outline.set_visible(False)
    suptitle(fig, "Correlation between SEA overfishing indicators (pooled SEA5, 1950–2019)",
             "Level correlations are inflated by shared trends; differences show which indicators really move together")
    save(fig, sec, "3.10_indicator_correlation_matrices", src)


# =========================================================================== #
# Section 4 — Validation against alternative sources
# =========================================================================== #
def section_validation() -> None:
    sec = "4_validation"
    print("Section 4: validation")
    V = RAW / "validation"
    over = owid_series("share-of-fish-stocks-overexploited", "World")
    sdg = pd.read_csv(V / "un_sdg" / "14_4_1_ER_H2O_FWTL.csv")
    sdg_w = 100 - sdg[sdg["geo_area_name"] == "World"].set_index("year")["value_pct_sustainable"].astype(float)

    # 4.1 OWID vs UN SDG (same data, two publishers) --------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(over.index, over.values, color=SERIES[0], label="OWID (main dataset)")
    old = sdg_w.loc[:2021]
    new = sdg_w.loc[2022:]
    ax.scatter(old.index, old.values, s=22, color=SURFACE, edgecolor=INK_2, linewidth=1.2, zorder=3,
               label="UN SDG database")
    ax.scatter(new.index, new.values, s=46, color=SERIES[1], edgecolor=SURFACE, linewidth=2, zorder=3,
               label="UN SDG, FAO new method (2023–24)")
    for y, v in new.items():
        end_label(ax, y, v, f"{v:.1f}%")
    ax.set_ylabel("% of stocks overexploited")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    diff = (over - sdg_w).dropna().abs().max()
    suptitle(fig, "Check 1 — OWID vs the official UN SDG database",
             f"Identical for 1974–2021 (max difference {diff:.4f} pp). 2023–24 use FAO's new method: do not join")
    save(fig, sec, "4.01_owid_vs_un_sdg", "Our World in Data; UN Global SDG Indicators Database (ER_H2O_FWTL)")

    # 4.2 FAO vs SAU, world % overexploited (different method) ---------------
    sau_w = sau_overexploited(V / "sau_regions" / "stock_status" / "global.csv")
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(over.index, over.values, color=SERIES[0], label="FAO (stock assessments)")
    ax.plot(sau_w.index, sau_w.values, color=SERIES[1], label="Sea Around Us (catch-based)")
    ax.set_ylabel("% of stocks overexploited")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    pair = pd.concat([over.rename("fao"), sau_w.rename("sau")], axis=1).dropna()
    r_lev = stats.spearmanr(pair["fao"], pair["sau"])[0]
    r_dif = stats.spearmanr(pair["fao"].diff().dropna(), pair["sau"].diff().dropna())[0]
    suptitle(fig, "Check 2 — Two methods, same question: how many stocks are overfished?",
             f"Spearman ρ: levels {r_lev:.2f}, year-to-year changes {r_dif:.2f} → agree on trend, not on annual moves")
    save(fig, sec, "4.02_fao_vs_sau_world_overexploited", "FAO via OWID; Sea Around Us")

    # 4.3 Bland-Altman plot for the same pair --------------------------------
    mean = pair.mean(axis=1)
    dd = pair["fao"] - pair["sau"]
    bias, sd = dd.mean(), dd.std(ddof=1)
    w_stat, w_p = stats.wilcoxon(pair["fao"], pair["sau"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean, dd, s=36, c=pair.index, cmap=SEQ_BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for v, lab in [(bias, f"Bias {bias:+.1f} pp"), (bias + 1.96 * sd, f"+1.96 SD {bias + 1.96 * sd:+.1f}"),
                   (bias - 1.96 * sd, f"−1.96 SD {bias - 1.96 * sd:+.1f}")]:
        ax.axhline(v, color=INK_2 if lab.startswith("Bias") else MUTED, lw=1.2 if lab.startswith("Bias") else 0.9)
        ax.text(mean.max() + 0.5, v, lab, va="center", fontsize=9, color=INK_2)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Mean of FAO and SAU (% overexploited)")
    ax.set_ylabel("FAO − SAU (pp)")
    ax.grid(axis="x")
    suptitle(fig, "Check 2b — Bland-Altman agreement plot, FAO vs Sea Around Us",
             f"Wilcoxon signed-rank p = {w_p:.3g}. Colour = year (light → dark). A trend in the dots = disagreement grows")
    save(fig, sec, "4.03_bland_altman_fao_vs_sau", "FAO via OWID; Sea Around Us")

    # 4.4 FAO areas 71 & 57: FAO vs SAU --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, (ent, f, title) in zip(axes, [
        ("Western Central Pacific (FAO)", "fao_71_pacific_western_central", "Area 71 — Western Central Pacific"),
        ("Eastern Indian Ocean (FAO)", "fao_57_indian_ocean_eastern", "Area 57 — Eastern Indian Ocean")]):
        fao = owid_series("share-of-fish-stocks-overexploited", ent)
        sau = sau_overexploited(V / "sau_regions" / "stock_status" / f"{f}.csv").loc[2000:]
        ax.plot(sau.index, sau.values, color=SERIES[1], label="Sea Around Us (annual)")
        ax.plot(fao.index, fao.values, color=SERIES[0], marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label="FAO (assessment years)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("% of stocks overexploited")
    axes[0].legend(loc="upper left")
    suptitle(fig, "Check 3 — Southeast Asian fishing areas: FAO vs Sea Around Us",
             "Only ~9 overlapping FAO points: enough for a trend check, too few for correlation")
    layout(fig, 0)
    save(fig, sec, "4.04_fao_vs_sau_areas_71_57", "FAO via OWID; Sea Around Us")

    # 4.5 Catch: FAO capture vs SAU reported, scatter with 1:1 line ----------
    cap = owid("capture-fishery-production").pivot(index="Year", columns="Entity",
                                                   values="Capture fisheries production (metric tons)")
    d = load_sau_eez()["catch"]
    rep = d[d["reporting_status"] == "Reported"].groupby(["country", "year"])["tonnes"].sum().unstack(0)
    glob_c = pd.read_csv(V / "sau_regions" / "catch" / "global.csv").set_index("year")
    rep["World"] = glob_c["reported"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    for ax, c in zip(axes.flat, ["World"] + SEA5):
        pr = pd.concat([cap[c].rename("fao"), rep[c].rename("sau")], axis=1).dropna()
        lim = [0, pr.max().max() * 1.05]
        ax.plot(lim, lim, color=AXIS, lw=1, zorder=1)
        ax.scatter(pr["sau"], pr["fao"], c=pr.index, cmap=SEQ_BLUE, s=16, zorder=2)
        ccc = 2 * np.cov(pr["fao"], pr["sau"])[0, 1] / (pr["fao"].var() + pr["sau"].var() + (pr["fao"].mean() - pr["sau"].mean()) ** 2)
        ax.set_title(f"{c}  (Lin's CCC = {ccc:.2f})", fontsize=10)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        millions(ax)
        millions(ax, "x")
        ax.grid(axis="x")
    for ax in axes[:, 0]:
        ax.set_ylabel("FAO capture (Mt)")
    for ax in axes[1]:
        ax.set_xlabel("Sea Around Us reported catch (Mt)")
    suptitle(fig, "Check 4 — Catch: FAO statistics vs Sea Around Us reported catch, 1960–2019",
             "Gray line = perfect agreement (1:1). Dots above it: FAO higher, as FAO also counts inland catch. Colour = year")
    layout(fig, 0)
    save(fig, sec, "4.05_catch_fao_vs_sau_scatter", "FAO via OWID; Sea Around Us")

    # 4.6 EEZ vs Large Marine Ecosystem: same method, different region ------
    stock = load_sau_eez()["stock"]
    stock["over"] = stock["Collapsed"] + stock["Over-exploited"]
    pairs = [  # LME file, matching EEZ ids (approximate geographic overlap)
        ("lme_35_gulf_of_thailand", "Gulf of Thailand", [957]),
        ("lme_36_south_china_sea", "South China Sea", [704, 460, 461, 463]),
        ("lme_37_sulu_celebes_sea", "Sulu-Celebes Sea", [608]),
        ("lme_38_indonesian_sea", "Indonesian Sea", [937, 361]),
        ("lme_34_bay_of_bengal", "Bay of Bengal", [956, 459]),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.9), sharey=True)
    for ax, (f, title, ids) in zip(axes, pairs):
        lme = sau_overexploited(V / "sau_regions" / "stock_status" / f"{f}.csv")
        eez = stock[stock["eez"].isin(ids)].groupby("year")["over"].mean()
        ax.plot(lme.index, lme.values, color=SERIES[0], label="LME (ecosystem)")
        ax.plot(eez.index, eez.values, color=SERIES[1], label="Matching EEZs (mean)")
        r = stats.spearmanr(lme.loc[eez.index], eez)[0]
        ax.set_title(f"{title}\nρ = {r:.2f}", fontsize=10)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("% of stocks overfished")
    axes[0].legend(loc="upper left", fontsize=8)
    suptitle(fig, "Check 5 — Does the result depend on how the region is drawn? EEZ vs ecosystem (LME)",
             "Sea Around Us data for national zones (EEZs) vs Large Marine Ecosystems that overlap them")
    layout(fig, 0)
    save(fig, sec, "4.06_eez_vs_lme_overexploited", "Sea Around Us")


# =========================================================================== #
# Section 5 — Data integrity: is reported catch plausible?
# =========================================================================== #
ISO_TO_COUNTRY = {"MYS": "Malaysia", "IDN": "Indonesia", "THA": "Thailand", "VNM": "Vietnam", "PHL": "Philippines"}
BENFORD = np.log10(1 + 1 / np.arange(1, 10))  # expected share of leading digits 1..9


def benford_mad(counts: np.ndarray) -> float:
    """Mean absolute deviation from Benford's law (Nigrini). Preferred over chi-square
    for very large samples, where chi-square rejects even trivial deviations."""
    p = counts / counts.sum()
    return float(np.mean(np.abs(p - BENFORD)))


def nigrini_label(mad: float) -> str:
    """Nigrini (2012) conformity bands for first-digit MAD."""
    return ("close conformity" if mad < 0.006 else "acceptable" if mad < 0.012
            else "marginal" if mad < 0.015 else "nonconformity")


def section_integrity() -> None:
    sec = "5_integrity"
    print("Section 5: data integrity")
    V = RAW / "validation"
    econ = pd.read_csv(V / "economic" / "fisheries_gdp_capture.csv")
    # Sustainable fisheries value added (USD) = SDG 14.7.1 (% of GDP) x GDP
    econ["fish_va_usd"] = econ["sustainable_fisheries_pct_gdp"] / 100 * econ["gdp_current_usd"]
    econ["usd_per_t"] = econ["fish_va_usd"] / econ["capture_t"]
    d = load_sau_eez()["catch"]

    # 5.1 Value per tonne: GDP-implied vs Sea Around Us landed value ---------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8))
    for iso, c in ISO_TO_COUNTRY.items():
        g = econ[econ["iso3"] == iso]
        a1.plot(g["year"], g["usd_per_t"], color=COUNTRY_COLOR[c], marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=c)
    w = econ[econ["iso3"] == "WLD"]
    a1.plot(w["year"], w["usd_per_t"], color=WORLD_COLOR, lw=1.4, label="World")
    lv = d.groupby(["country", "year"])[["landed_value", "tonnes"]].sum()
    lv = (lv["landed_value"] / lv["tonnes"]).unstack(0)
    for c in SEA5:
        a2.plot(lv.index, lv[c], color=COUNTRY_COLOR[c], label=c)
    a1.set_title("GDP-implied: fisheries value added ÷ reported catch (current USD/t)", fontsize=10)
    a2.set_title("Sea Around Us landed value ÷ catch (real 2010 USD/t)", fontsize=10)
    a1.set_ylabel("USD per tonne")
    a1.set_ylim(0, None)
    a2.set_ylim(0, None)
    a1.legend(loc="upper right", fontsize=8)
    suptitle(fig, "Integrity check 1 — Value per tonne of fish: does reported catch fit the money?",
             "A country far above its peers may be under-reporting tonnes; far below, over-reporting. "
             "Note: SDG 14.7.1 is scaled by stock sustainability, and units differ between panels")
    layout(fig, 0)
    save(fig, sec, "5.01_value_per_tonne", "UN SDG 14.7.1 via OWID; World Bank WDI; Sea Around Us")

    # 5.2 Peer cross-section: fisheries value added vs reported catch (log-log)
    year = 2021
    x = econ[(econ["year"] == year) & (econ["iso3"] != "WLD") & (econ["fish_va_usd"] > 0)].copy()
    lx, ly = np.log10(x["capture_t"]), np.log10(x["fish_va_usd"])
    slope, icpt, r, p, _ = stats.linregress(lx, ly)
    x["resid"] = ly - (icpt + slope * lx)
    x["z"] = (x["resid"] - x["resid"].mean()) / x["resid"].std(ddof=1)
    fig, ax = plt.subplots(figsize=(9, 6))
    peers = x[~x["iso3"].isin(ISO_TO_COUNTRY)]
    ax.scatter(peers["capture_t"], peers["fish_va_usd"], s=24, color=NEUTRAL, edgecolor=SURFACE, lw=1, label="Other countries")
    xs = np.logspace(lx.min(), lx.max(), 50)
    ax.plot(xs, 10 ** (icpt + slope * np.log10(xs)), color=INK_2, lw=1.2, label=f"Log-log fit (R² = {r**2:.2f})")
    for iso, c in ISO_TO_COUNTRY.items():
        row = x[x["iso3"] == iso]
        if row.empty:
            continue
        # SEA points sit close together, so their z-scores go in the legend instead of on the plot.
        ax.scatter(row["capture_t"], row["fish_va_usd"], s=70, color=COUNTRY_COLOR[c], edgecolor=SURFACE, lw=2,
                   zorder=3, label=f"{c} (z = {row['z'].iloc[0]:+.1f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reported wild capture (tonnes, log scale)")
    ax.set_ylabel("Sustainable fisheries value added (USD, log scale)")
    ax.grid(axis="x")
    legend_right(ax)
    suptitle(fig, f"Integrity check 2 — Fisheries value added vs reported catch, {len(x)} countries, {year}",
             "z = how far a country sits above (+) or below (−) the fitted line, in standard deviations. |z| > 2 would be a red flag")
    save(fig, sec, "5.02_peer_value_vs_catch", "UN SDG 14.7.1 via OWID; World Bank WDI")

    # 5.3 Reconstructed ÷ reported catch ratio (Sea Around Us) ---------------
    tot = d.groupby(["country", "year"])["tonnes"].sum()
    rep = d[d["reporting_status"] == "Reported"].groupby(["country", "year"])["tonnes"].sum()
    ratio = (tot / rep).unstack(0)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for c in SEA5:
        ax.plot(ratio.index, ratio[c], color=COUNTRY_COLOR[c], label=c)
    ax.axhline(1, color=AXIS, lw=1)
    ax.text(1951, 1.02, "1 = everything was reported", fontsize=8, color=MUTED, va="bottom")
    ax.set_ylabel("Reconstructed ÷ reported catch")
    ax.set_ylim(0.9, None)
    legend_right(ax)
    suptitle(fig, "Integrity check 3 — How much catch is missing from official statistics?",
             "Sea Around Us total (reported + reconstructed unreported) divided by the officially reported part")
    save(fig, sec, "5.03_reconstructed_vs_reported_ratio", "Sea Around Us (v50-1), EEZ data")

    # 5.4 Mirror trade: own exports vs what partners say they imported -------
    ct = pd.read_csv(V / "comtrade" / "fish_hs03_mirror_trade.csv")
    ct["gap_pct"] = (ct["partner_imports_usd"] / ct["reported_exports_usd"] - 1) * 100
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for j, c in enumerate(SEA5):
        g = ct[ct["country"] == c]
        a, b = axes[0, j], axes[1, j]
        a.plot(g["year"], g["reported_exports_usd"], color=COUNTRY_COLOR[c], label="Exports it reported")
        a.plot(g["year"], g["partner_imports_usd"], color=INK_2, lw=1.4, label="Imports partners reported")
        a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1e9:g}"))
        a.set_title(c, fontsize=10)
        b.axhspan(0, 10, color=GRID, alpha=0.7, lw=0)
        b.bar(g["year"], g["gap_pct"], color=[COUNTRY_COLOR[c] if v >= 0 else NEUTRAL for v in g["gap_pct"]], width=0.7)
        b.axhline(0, color=AXIS, lw=0.8)
        b.set_xticks(range(2011, 2024, 4))
        b.tick_params(axis="x", labelrotation=90)
    axes[0, 0].set_ylabel("Fish exports (billion USD)")
    axes[1, 0].set_ylabel("Partner − own (%)")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    missing = sorted(ct.loc[ct["partner_imports_usd"].isna(), "country"].unique())
    note = f" Missing partner data: {', '.join(missing)} (some years, API quota)." if missing else ""
    suptitle(fig, "Integrity check 4 — Mirror trade: fish exports (HS 03) vs what trading partners recorded",
             "Shaded band 0–10% = normal gap (imports are valued with freight and insurance, CIF; exports without, FOB). "
             "Larger positive gaps suggest under-reported exports." + note)
    layout(fig, 0)
    save(fig, sec, "5.04_mirror_trade_hs03", "UN Comtrade (public preview API), HS chapter 03")

    # 5.5 Benford's law on catch values: reported vs reconstructed ------------
    bf = pd.read_csv(V / "benford" / "sau_catch_leading_digits.csv")
    dcols = [f"d{i}" for i in range(1, 10)]
    g = bf.groupby(["country", "reporting_status"])[dcols].sum()
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.4), sharey=True)
    digits = np.arange(1, 10)
    mads = {}
    for ax, c in zip(axes, SEA5):
        for k, (st, colour, off) in enumerate([("Reported", COUNTRY_COLOR[c], -0.2), ("Unreported", NEUTRAL, 0.2)]):
            cnt = g.loc[(c, st)].to_numpy(float)
            mads[(c, st)] = benford_mad(cnt)
            ax.bar(digits + off, cnt / cnt.sum(), width=0.38, color=colour,
                   label=f"{'Reported' if st == 'Reported' else 'Reconstructed (unreported)'}")
        ax.plot(digits, BENFORD, color=INK, lw=1.4, marker="o", markersize=4, label="Benford's law")
        ax.set_title(f"{c}\nMAD reported {mads[(c, 'Reported')]:.4f} ({nigrini_label(mads[(c, 'Reported')])})", fontsize=9)
        ax.set_xticks(digits)
        ax.set_xlabel("Leading digit")
    axes[0].set_ylabel("Share of catch values")
    axes[0].legend(loc="upper right", fontsize=8)
    suptitle(fig, "Integrity check 5 — Benford's law: do catch figures have natural leading digits?",
             "Fabricated numbers often break Benford's law. MAD = mean absolute deviation; Nigrini bands: <0.006 close, "
             "<0.012 acceptable, <0.015 marginal, else nonconformity")
    layout(fig, 0)
    save(fig, sec, "5.05_benford_catch_values", "Sea Around Us (v50-1) taxon-level catch records")

    # 5.6 Summary: all checks side by side ------------------------------------
    unrep = (1 - 1 / ratio.loc[2015:2019]).mean() * 100
    z21 = x.set_index("iso3")["z"].rename(index=ISO_TO_COUNTRY)
    gap = ct[ct["year"].between(2015, 2023)].groupby("country")["gap_pct"].median()
    mad_rep = pd.Series({c: mads[(c, "Reported")] for c in SEA5})
    summary = pd.DataFrame({
        "Unreported share of catch\n(SAU, 2015–19 mean, %)": unrep,
        "Value-vs-catch z-score\n(2021, vs peers)": z21,
        "Mirror trade gap\n(2015–23 median, %)": gap,
        "Benford MAD\n(reported catch)": mad_rep,
    }).loc[SEA5]
    # Colour each column by rank (1 = least suspicious, 5 = most) so different units can share one scale.
    suspicion = pd.DataFrame({
        summary.columns[0]: summary.iloc[:, 0].rank(),
        summary.columns[1]: summary.iloc[:, 1].abs().rank(),
        summary.columns[2]: summary.iloc[:, 2].rank(),
        summary.columns[3]: summary.iloc[:, 3].rank(),
    })
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.imshow(suspicion.values, cmap=SEQ_BLUE, vmin=0.5, vmax=5.5, aspect="auto")
    fmt = ["{:.0f}%", "{:+.2f}", "{:+.0f}%", "{:.4f}"]
    for i in range(len(SEA5)):
        for j in range(summary.shape[1]):
            v = summary.iloc[i, j]
            ax.text(j, i, "n/a" if pd.isna(v) else fmt[j].format(v), ha="center", va="center", fontsize=10,
                    color="#ffffff" if suspicion.iloc[i, j] >= 4 else INK)
    ax.set_xticks(range(summary.shape[1]), summary.columns, fontsize=9)
    ax.set_yticks(range(len(SEA5)), SEA5)
    ax.xaxis.tick_top()
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ok = summary.iloc[:, [0, 2]].dropna()
    rho_txt = (f"Spearman ρ, unreported share vs mirror gap = {stats.spearmanr(ok.iloc[:, 0], ok.iloc[:, 1])[0]:.2f} "
               f"(n = {len(ok)}, indicative only)") if len(ok) >= 3 else "Too few countries for a correlation"
    suptitle(fig, "Integrity summary — which countries' official numbers look least reliable?",
             f"Darker = more suspicious within SEA5 (ranked per column). {rho_txt}")
    fig.subplots_adjust(top=0.72)
    save(fig, sec, "5.06_integrity_summary", "Sea Around Us; UN SDG 14.7.1; World Bank; UN Comtrade")


# =========================================================================== #
SECTIONS = {
    "global": section_global,
    "sea_owid": section_sea_owid,
    "sea_sau": section_sea_sau,
    "validation": section_validation,
    "integrity": section_integrity,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create all exploratory figures.")
    parser.add_argument("--only", nargs="+", choices=SECTIONS, help="run only these sections")
    args = parser.parse_args()
    setup_style()
    for name in args.only or SECTIONS:
        SECTIONS[name]()
    print("Done ->", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
