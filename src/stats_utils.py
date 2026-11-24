"""
stats_utils.py — Small statistical helpers shared by several pipeline steps
==========================================================================

KIE4033 Assignment 1 · SDG 14 overfishing analysis. Moved unchanged from the analysis
scripts so the data-quality step (step 0) can use them without importing later steps:
p-value formatting, Holm adjustment, Mann-Kendall + Sen's slope, Benford MAD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats
from statsmodels.stats.multitest import multipletests



def fmt_p(p: float) -> str:
    """p-value with its relation sign for titles: '= 0.123', or '< 0.001' for very small p."""
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def holm(p) -> np.ndarray:
    """Holm step-down adjustment (controls the family-wise error rate, less conservative than Bonferroni)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def mann_kendall(s: pd.Series) -> dict:
    """
    Mann-Kendall trend test + Sen's slope (median of all pairwise slopes) with its 95% CI.
    Hamed-Rao variant (variance corrected for autocorrelation) when n ≥ 20, original otherwise.
    """
    s = s.dropna()
    res = mk.hamed_rao_modification_test(s.values) if len(s) >= 20 else mk.original_test(s.values)
    sen, _, lo, hi = stats.theilslopes(s.values, s.index.to_numpy(), alpha=0.95)
    return {"n": len(s), "first": int(s.index.min()), "last": int(s.index.max()),
            "mk_variant": "Hamed-Rao" if len(s) >= 20 else "original", "tau": res.Tau, "z": res.z,
            "p": res.p, "trend": res.trend, "sen_slope": sen, "sen_ci_low": lo, "sen_ci_high": hi}


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
