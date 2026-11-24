"""
style.py — Shared chart style: palette, matplotlib defaults, figure header helpers
================================================================================

KIE4033 Assignment 1 · SDG 14 overfishing analysis. Moved unchanged from
visualise.py so every step (data quality first) draws figures the same way.

Colours follow a colour-vision-deficiency-checked categorical palette in a fixed order
(a country always keeps the same colour in every figure); one y-axis per chart; 2 px
lines; hairline solid gridlines; a legend whenever there are two or more series.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

from loaders import SEA5, STATUS_ORDER


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


# A country keeps the same colour in every figure.
COUNTRY_COLOR = dict(zip(SEA5, SERIES[:5]))


WORLD_COLOR = INK_2


# Sequential (magnitude) ramp: one hue, light -> dark.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


# Diverging ramp: blue <-> red with a neutral gray midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#5598e7", "#f0efec", "#ec835a", "#b8302f"])


STATUS_COLOR = dict(zip(STATUS_ORDER, ["#e34948", "#eda100", "#c3c2b7", "#2a78d6", "#1baf7a"]))


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
