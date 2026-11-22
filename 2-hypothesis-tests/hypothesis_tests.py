"""
hypothesis_tests.py — Formal tests of hypotheses H1a, H1b, H2a, H2b, H2c
=========================================================================

KIE4033 Assignment 1 · SDG 14 overfishing analysis (Global -> Southeast Asia)

Runs the analysis pipeline in README §4 on the raw data:

    stationarity  ADF + KPSS on every series (levels and transformed); picks the
                  transformation used for correlation tests (first difference,
                  or detrended first difference if still non-stationary)
    h1a           Global trend in % stocks overexploited: Mann-Kendall (original +
                  Hamed-Rao autocorrelation-corrected), Sen's slope with 95% CI,
                  OLS trend with Newey-West (HAC) errors, segmented regression with
                  a break at 2015, Chow test; robustness on assessment years only
                  and on the Sea Around Us world series
    h1b           Global catch -> % overexploited: cross-correlation (CCF) of the
                  stationary series at lags 0–10 with Holm correction, the same after
                  AR prewhitening, Granger causality (lags 1–4), Engle-Granger
                  cointegration of the levels
    h2a           SEA trends: Hamed-Rao Mann-Kendall + Sen's slope for % overexploited
                  and mean trophic level (MTL), for SEA, 5 countries and 11 EEZs,
                  windows 1950–2019 and 1990–2019, Holm-corrected; FAO Areas 71/57
    h2b           SEA catch -> MTL: CCF per country and for SEA, Holm-corrected
    h2c           Is SEA worse than the world? Interaction regression
                  y = b0 + b1·year + b2·SEA + b3·year×SEA, tested through the gap series
                  (SEA − World) with HAC errors; SAU % overexploited, SAU MTL, FAO areas
    diagnostics   Shapiro-Wilk, Durbin-Watson, Ljung-Box, Breusch-Pagan on every
                  regression's residuals; residual plots for the H1a trend model

    python 2-hypothesis-tests/hypothesis_tests.py               # everything
    python 2-hypothesis-tests/hypothesis_tests.py --only h1a h1b

Outputs: ``results/*.csv`` (one per test family, plus ``decisions.csv`` with the
accept/reject verdict per hypothesis) and ``figures/<section>/*.png``.

Reading the results
-------------------
* Yearly series are autocorrelated. Trend p-values use the Hamed-Rao corrected
  Mann-Kendall test and regressions use HAC standard errors for that reason.
* Correlation tests are only run on stationary series; correlating two trending
  series gives spurious results.
* Many lags/series are tested at once, so p-values are Holm-adjusted inside each
  family; "significant" always means after adjustment.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.stattools import acf, adfuller, coint, grangercausalitytests, kpss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0-visualisation"))
sys.path.insert(0, str(ROOT / "1-statistical-tests"))
import stat_tests as st  # noqa: E402  (data builders: eez_panel, country_panel, world_fao, ...)
import visualise as v  # noqa: E402  (chart style + header helpers)

warnings.filterwarnings("ignore", category=FutureWarning)
from statsmodels.tools.sm_exceptions import InterpolationWarning  # noqa: E402

# KPSS p-values come from a lookup table bounded at 0.01–0.10; values at the bounds are
# reported as such (the warning only says the true p is beyond the table).
warnings.filterwarnings("ignore", category=InterpolationWarning)

FIG = HERE / "figures"
RES = HERE / "results"
VAL = ROOT / "data" / "raw" / "validation"
SEA5 = v.SEA5
ALPHA = 0.05
BREAK = 2015          # SDG adoption year, break point for segmented regression / Chow test
MAX_LAG = 10          # CCF lags tested (years), catch leading the response
DIAGNOSTICS: list[dict] = []   # residual checks of every regression -> diagnostics.csv
DECISIONS: list[dict] = []     # one verdict per hypothesis -> decisions.csv


# =========================================================================== #
# Helpers: output, formatting
# =========================================================================== #
def save_fig(fig, section: str, name: str, source: str) -> None:
    fig.text(0.01, -0.01, f"Source: {source}", ha="left", va="top", fontsize=8, color=v.MUTED)
    path = FIG / section / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("  saved", path.relative_to(ROOT))


def save_table(df: pd.DataFrame, name: str) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    path = RES / f"{name}.csv"
    df.round(6).to_csv(path, index=False)
    print("  saved", path.relative_to(ROOT))


fmt_p = st.fmt_p   # '= 0.123' or '< 0.001'


def holm(p) -> np.ndarray:
    """Holm step-down adjustment (controls the family-wise error rate, less conservative than Bonferroni)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def decide(hyp: str, question: str, evidence: str, reject: bool, note: str = "", verdict: str | None = None) -> None:
    """Log a verdict; `verdict` overrides the default wording for mixed results."""
    DECISIONS.append({"hypothesis": hyp, "question": question, "evidence": evidence,
                      "decision": verdict or ("Reject H0" if reject else "Fail to reject H0"), "note": note})


# =========================================================================== #
# Data
# =========================================================================== #
def sea_series() -> pd.DataFrame:
    """
    SEA aggregate, 1950–2019:
      over  = mean of the 5 countries' % overexploited (each country counts equally)
      mtl   = catch-weighted mean trophic level of all 11 EEZs
      catch = total catch of the 11 EEZs (t)
    """
    p = st.eez_panel()
    cp = st.country_panel(p)
    agg = p.assign(wm=p["mtl"] * p["catch"]).groupby("year").agg(catch=("catch", "sum"), wm=("wm", "sum"))
    agg["mtl"] = agg["wm"] / agg["catch"]
    agg["over"] = cp.groupby("year")["over"].mean()
    return agg[["over", "mtl", "catch"]]


def country_series() -> dict[str, pd.DataFrame]:
    cp = st.country_panel(st.eez_panel())
    return {c: cp[cp["country"] == c].set_index("year")[["over", "mtl", "catch"]] for c in SEA5}


def sau_world_catch() -> pd.Series:
    """Sea Around Us global reconstructed catch = landings + discards (t), 1950–2019."""
    d = pd.read_csv(VAL / "sau_regions" / "catch" / "global.csv").set_index("year")
    return d["landings"] + d["discards"]


def sau_world_mtl() -> pd.Series:
    d = pd.read_csv(VAL / "sau_regions" / "marine_trophic_index" / "global.csv").set_index("year")
    return d["mean_trophic_level"]


def fao_area(name: str) -> pd.Series:
    o = v.owid("share-of-fish-stocks-overexploited")
    return o[o["Entity"] == f"{name} (FAO)"].set_index("Year").iloc[:, -1].astype(float)


# =========================================================================== #
# Statistical building blocks
# =========================================================================== #
def adf_p(x, regression: str) -> float:
    """Augmented Dickey-Fuller p-value (H0: unit root = non-stationary), lag length by AIC."""
    return float(adfuller(np.asarray(x, float), regression=regression, autolag="AIC")[1])


def kpss_p(x, regression: str) -> float:
    """KPSS p-value (H0: stationary). Table-bounded: reported as 0.01 / 0.10 at the edges."""
    return float(kpss(np.asarray(x, float), regression=regression, nlags="auto")[1])


def detrend(x: pd.Series) -> pd.Series:
    b, a = np.polyfit(x.index, x.values, 1)
    return x - (a + b * x.index.to_numpy())


def make_stationary(s: pd.Series, log: bool) -> tuple[str, pd.Series, float, float]:
    """
    Transform a trending series until ADF rejects a unit root:
    1) first difference (log-difference = growth rate for catch),
    2) if still non-stationary, first difference minus its linear trend.
    Returns (label, transformed series, ADF p, KPSS p).
    """
    x = np.log(s) if log else s
    d = x.diff().dropna()
    label = "Δlog (growth rate)" if log else "first difference"
    a, k = adf_p(d, "c"), kpss_p(d, "c")
    if a >= ALPHA:
        d = detrend(d)
        label = f"detrended {label}"
        a, k = adf_p(d, "c"), kpss_p(d, "c")
    return label, d, a, k


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


def nw_lags(n: int) -> int:
    """Newey-West rule of thumb for HAC lag length: floor(4·(n/100)^(2/9))."""
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def ols_hac(y: np.ndarray, X: np.ndarray, label: str):
    """OLS with Newey-West (HAC) standard errors; residual checks are logged to DIAGNOSTICS."""
    fit = sm.OLS(y, X).fit()
    hac = fit.get_robustcov_results(cov_type="HAC", maxlags=nw_lags(len(y)))
    r = fit.resid
    lb_lag = max(1, min(10, len(r) // 5))
    DIAGNOSTICS.append({
        "model": label, "n": len(y), "r2": fit.rsquared,
        "shapiro_p": stats.shapiro(r).pvalue if len(r) >= 3 else np.nan,
        "durbin_watson": durbin_watson(r),
        "ljung_box_lag": lb_lag, "ljung_box_p": float(acorr_ljungbox(r, lags=[lb_lag])["lb_pvalue"].iloc[0]),
        "breusch_pagan_p": het_breuschpagan(r, X)[1] if X.shape[1] > 1 else np.nan,
        "hac_maxlags": nw_lags(len(y)),
    })
    return fit, hac


def ccf_table(x: pd.Series, y: pd.Series, label: str) -> pd.DataFrame:
    """
    Cross-correlation r_k = corr(x_t, y_{t+k}) for k = 0..MAX_LAG (x leads y by k years),
    Pearson p per lag, Holm-adjusted across the 11 lags, and the ±1.96/√n white-noise band.
    """
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    rows = []
    for k in range(MAX_LAG + 1):
        a, b = d["x"].iloc[: len(d) - k].to_numpy(), d["y"].iloc[k:].to_numpy()
        r, p = stats.pearsonr(a, b)
        rho = stats.spearmanr(a, b)[0]
        rows.append({"pair": label, "lag": k, "n": len(a), "r": r, "p": p, "spearman_rho": rho,
                     "band_95": 1.96 / np.sqrt(len(a))})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    out["significant_holm"] = out["p_holm"] < ALPHA
    return out


def prewhiten(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    """
    Box-Jenkins prewhitening: fit an AR(p) model to the leading series x (p by AIC, ≤ 4),
    then pass BOTH series through the same filter. Removes x's own autocorrelation so
    that CCF spikes are not artefacts of persistence.
    """
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    sel = ar_select_order(d["x"].to_numpy(), maxlag=4, ic="aic", trend="c")
    p = max(sel.ar_lags) if sel.ar_lags else 0
    if p == 0:
        return d["x"] - d["x"].mean(), d["y"] - d["y"].mean(), 0
    phi = AutoReg(d["x"].to_numpy(), lags=p, trend="c").fit().params[1:]

    def filt(s: pd.Series) -> pd.Series:
        vals = s.to_numpy()
        out = vals[p:] - sum(phi[i] * vals[p - 1 - i: len(vals) - 1 - i] for i in range(p))
        return pd.Series(out, index=s.index[p:])

    return filt(d["x"]), filt(d["y"]), p


def granger(x: pd.Series, y: pd.Series, maxlag: int = 4) -> pd.DataFrame:
    """Does past x improve the prediction of y beyond y's own past? F-test per lag order, Holm-adjusted."""
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().to_numpy()
    res = grangercausalitytests(d, maxlag=maxlag)
    out = pd.DataFrame([{"lag_order": L, "F": res[L][0]["ssr_ftest"][0], "p": res[L][0]["ssr_ftest"][1]}
                        for L in range(1, maxlag + 1)])
    out["p_holm"] = holm(out["p"])
    return out


def chow(y: pd.Series, brk: int) -> tuple[float, float]:
    """
    Chow test for a structural break after `brk`: compares one line fitted to all years
    with separate lines before (≤ brk) and after (> brk).
    F = ((SSR_all − SSR_1 − SSR_2)/k) / ((SSR_1 + SSR_2)/(n − 2k)), k = 2 parameters.
    """
    t = y.index.to_numpy(float)

    def ssr(mask):
        X = sm.add_constant(t[mask])
        return sm.OLS(y.values[mask], X).fit().ssr

    pre, post = t <= brk, t > brk
    k, n = 2, len(y)
    s_all, s1, s2 = ssr(np.ones_like(t, bool)), ssr(pre), ssr(post)
    F = ((s_all - s1 - s2) / k) / ((s1 + s2) / (n - 2 * k))
    return float(F), float(stats.f.sf(F, k, n - 2 * k))


# =========================================================================== #
# Stationarity
# =========================================================================== #
def stationarity_specs() -> list[tuple[str, pd.Series, bool]]:
    """(name, series, use log) for every series that enters a correlation test."""
    sea = sea_series()
    cs = country_series()
    specs = [
        ("World % overexploited (FAO)", st.world_fao(), False),
        ("World % overexploited (SAU)", st.world_sau(), False),
        ("World capture, t (FAO)", st.capture("World").loc[1960:], True),
        ("World catch, t (SAU)", sau_world_catch(), True),
        ("SEA % overexploited (SAU)", sea["over"], False),
        ("SEA mean trophic level (SAU)", sea["mtl"], False),
        ("SEA catch, t (SAU)", sea["catch"], True),
    ]
    for c in SEA5:
        specs += [(f"{c} mean trophic level (SAU)", cs[c]["mtl"], False),
                  (f"{c} catch, t (SAU)", cs[c]["catch"], True)]
    return specs


TRANSFORMED: dict[str, pd.Series] = {}


def section_stationarity() -> None:
    print("Stationarity: ADF + KPSS")
    rows = []
    for name, s, log in stationarity_specs():
        x = np.log(s) if log else s
        label, d, a, k = make_stationary(s, log)
        TRANSFORMED[name] = d
        rows.append({"series": name, "first": int(s.index.min()), "last": int(s.index.max()), "n": len(s),
                     "level_adf_p": adf_p(x, "ct"), "level_kpss_p": kpss_p(x, "ct"),
                     "transformation_used": label, "transformed_adf_p": a, "transformed_kpss_p": k,
                     "stationary_after": "yes" if a < ALPHA and k >= ALPHA else ("ADF yes, KPSS no" if a < ALPHA else "no")})
    save_table(pd.DataFrame(rows), "0_stationarity_adf_kpss")


def ensure_transformed() -> None:
    if not TRANSFORMED:
        for name, s, log in stationarity_specs():
            TRANSFORMED[name] = make_stationary(s, log)[1]


# =========================================================================== #
# H1a — global trend
# =========================================================================== #
def section_h1a() -> None:
    sec = "1_H1a_global_trend"
    print("H1a: global trend")
    wf = st.world_fao()
    interp = st.interpolated_years(wf)
    ws = st.world_sau()
    variants = [("FAO world, all years 1974–2021", wf),
                ("FAO world, assessment years only (interpolated years removed)", wf.drop(interp)),
                ("FAO world, before SDGs 1974–2015", wf.loc[:BREAK]),
                ("FAO world, after SDGs 2015–2021", wf.loc[BREAK:]),
                ("SAU world, 1974–2019 (same window as FAO)", ws.loc[1974:]),
                ("SAU world, 1950–2019", ws)]
    mkt = pd.DataFrame([{"series": n, **mann_kendall(s)} for n, s in variants])
    save_table(mkt, "1_H1a_mann_kendall_sen")

    # OLS linear trend + segmented (continuous, slope change at 2015) with HAC errors
    t = wf.index.to_numpy(float)
    X_lin = sm.add_constant(t - t.min())
    _, lin = ols_hac(wf.values, X_lin, "H1a linear trend, FAO world")
    hinge = np.clip(t - BREAK, 0, None)
    X_seg = np.column_stack([X_lin, hinge])
    seg_fit, seg = ols_hac(wf.values, X_seg, "H1a segmented trend (break 2015), FAO world")
    F, p_chow = chow(wf, BREAK)
    reg = pd.DataFrame([
        {"model": "Linear trend", "term": "slope (pp/yr)", "coef": lin.params[1], "hac_se": lin.bse[1],
         "p": lin.pvalues[1], "r2": lin.rsquared},
        {"model": "Segmented (break 2015)", "term": "slope before 2015 (pp/yr)", "coef": seg.params[1],
         "hac_se": seg.bse[1], "p": seg.pvalues[1], "r2": seg.rsquared},
        {"model": "Segmented (break 2015)", "term": "change in slope after 2015 (pp/yr)", "coef": seg.params[2],
         "hac_se": seg.bse[2], "p": seg.pvalues[2], "r2": seg.rsquared},
        {"model": "Segmented (break 2015)", "term": "slope after 2015 (pp/yr)", "coef": seg.params[1] + seg.params[2],
         "hac_se": np.nan, "p": np.nan, "r2": seg.rsquared},
        {"model": "Chow test at 2015", "term": "F", "coef": F, "hac_se": np.nan, "p": p_chow, "r2": np.nan},
    ])
    save_table(reg, "1_H1a_regression_segmented_chow")

    main = mkt.iloc[0]
    robust = mkt.iloc[1]
    decide("H1a", "Is there a significant upward trend in world % stocks overexploited?",
           f"Hamed-Rao MK p {fmt_p(main['p'])}, Sen +{main['sen_slope']:.2f} pp/yr "
           f"[{main['sen_ci_low']:.2f}, {main['sen_ci_high']:.2f}]; assessment years only p {fmt_p(robust['p'])}; "
           f"SAU world p {fmt_p(mkt.iloc[4]['p'])}",
           main["p"] < ALPHA and main["trend"] == "increasing",
           f"No slowdown after 2015: slope change {seg.params[2]:+.2f} pp/yr (p {fmt_p(seg.pvalues[2])}), "
           f"Chow p {fmt_p(p_chow)}")

    # Figure
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
    a1.plot(wf.index, wf.values, color=v.SERIES[0], lw=2, zorder=2, label="FAO world")
    a1.scatter(interp, wf.loc[interp], s=40, facecolor=v.SURFACE, edgecolor=v.SERIES[0], lw=1.5, zorder=4,
               label="Looks interpolated (dropped in robustness run)")
    sen0 = main["sen_slope"]
    icpt = np.median(wf.values - sen0 * t)
    a1.plot(t, icpt + sen0 * t, color=v.INK_2, lw=1.2, ls=(0, (4, 3)), label=f"Sen's slope +{sen0:.2f} pp/yr")
    a1.plot(t, seg_fit.fittedvalues, color=v.SERIES[1], lw=1.6, label=(
        f"Segmented fit: {seg.params[1]:+.2f} → {seg.params[1] + seg.params[2]:+.2f} pp/yr after 2015"))
    a1.axvline(BREAK, color=v.AXIS, lw=1)
    a1.text(BREAK - 0.5, 8, "SDGs\nadopted", ha="right", fontsize=8.5, color=v.MUTED)
    a1.set_ylabel("% of stocks overexploited")
    a1.legend(loc="upper left", fontsize=8.5)
    a1.set_title(f"Mann-Kendall (Hamed-Rao) p {fmt_p(main['p'])}, τ = {main['tau']:.2f}; "
                 f"slope change at 2015 p {fmt_p(seg.pvalues[2])}, Chow p {fmt_p(p_chow)}", fontsize=9.5)
    # Sen's slope forest
    labels = ["FAO all years", "FAO assessment yrs", "FAO 1974–2015", "FAO 2015–2021", "SAU 1974–2019",
              "SAU 1950–2019"]
    for i, r in mkt.iterrows():
        y = len(mkt) - 1 - i
        sig = r["p"] < ALPHA
        a2.plot([r["sen_ci_low"], r["sen_ci_high"]], [y, y], color=v.SERIES[0], lw=2)
        a2.scatter(r["sen_slope"], y, s=60, color=v.SERIES[0] if sig else v.SURFACE, edgecolor=v.SERIES[0],
                   lw=1.5, zorder=3)
        a2.text(r["sen_ci_high"] + 0.03, y, f"p {fmt_p(r['p'])}", va="center", fontsize=8.5, color=v.INK_2)
    a2.axvline(0, color=v.AXIS, lw=0.8)
    a2.set_yticks(range(len(mkt)), labels[::-1])
    a2.set_xlabel("Sen's slope, pp per year (95% CI)")
    a2.grid(axis="y", visible=False)
    a2.set_title("Robustness: filled = significant trend", fontsize=9.5)
    v.suptitle(fig, "H1a — World % of stocks overexploited has a significant upward trend" if main["p"] < ALPHA
               else "H1a — No significant trend in world % of stocks overexploited",
               "Trend holds without interpolated years and in the independent Sea Around Us series; "
               "no measurable slowdown after 2015")
    v.layout(fig, 0)
    save_fig(fig, sec, "H1a.01_world_trend_mann_kendall_segmented", "FAO via OWID; Sea Around Us")


# =========================================================================== #
# H1b — global catch -> % overexploited
# =========================================================================== #
def section_h1b() -> None:
    sec = "2_H1b_global_ccf"
    print("H1b: global catch -> % overexploited")
    ensure_transformed()
    x = TRANSFORMED["World capture, t (FAO)"].loc[1975:2021]
    y = TRANSFORMED["World % overexploited (FAO)"]
    raw = ccf_table(x, y, "FAO capture growth -> Δ FAO % overexploited")
    pwx, pwy, p_ar = prewhiten(x, y)
    pw = ccf_table(pwx, pwy, f"same, prewhitened (AR({p_ar}))")
    x2 = TRANSFORMED["World catch, t (SAU)"].loc[1975:2019]
    alt = ccf_table(x2, y, "SAU catch growth -> Δ FAO % overexploited (independent sources)")
    ccf = pd.concat([raw, pw, alt], ignore_index=True)
    save_table(ccf, "2_H1b_ccf")
    g = granger(x, y)
    g.insert(0, "pair", "FAO capture growth -> Δ FAO % overexploited")
    save_table(g, "2_H1b_granger")
    lv = pd.concat([st.world_fao(), np.log(st.capture("World"))], axis=1).dropna()
    t_c, p_c, _ = coint(lv.iloc[:, 0], lv.iloc[:, 1], trend="ct")
    save_table(pd.DataFrame([{"test": "Engle-Granger cointegration (levels, with trend)",
                              "series": "FAO % overexploited ~ log FAO capture", "n": len(lv),
                              "t_stat": t_c, "p": p_c}]), "2_H1b_cointegration")
    any_sig = raw["significant_holm"].any() or pw["significant_holm"].any()
    best = raw.loc[raw["p"].idxmin()]
    decide("H1b", "Does global catch cross-correlate with % stocks overexploited at some lag?",
           f"Largest |r| on stationary series: r = {best['r']:+.2f} at lag {int(best['lag'])} "
           f"(Holm p {fmt_p(best['p_holm'])}); prewhitened min Holm p {fmt_p(pw['p_holm'].min())}; "
           f"Granger min Holm p {fmt_p(g['p_holm'].min())}; cointegration p {fmt_p(p_c)}",
           bool(any_sig),
           "Catch plateaued since the 1990s while overexploitation kept rising: no short-run or long-run "
           "link in this data. Catch is a weak proxy for fishing pressure (caveat 1).")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    pw_title = (f"Prewhitened (AR({p_ar}) filter)" if p_ar else
                "Prewhitened: AIC picks AR(0) — catch growth is\nalready white noise, so same as left")
    for ax, tab, title in zip(axes, [raw, pw, alt],
                              ["FAO capture → FAO stocks", pw_title,
                               "SAU catch → FAO stocks (robustness)"]):
        cols = [v.SERIES[0] if s else v.NEUTRAL for s in tab["significant_holm"]]
        ax.bar(tab["lag"], tab["r"], color=cols, width=0.6)
        ax.plot(tab["lag"], tab["band_95"], color=v.INK_2, lw=1, ls=(0, (3, 3)))
        ax.plot(tab["lag"], -tab["band_95"], color=v.INK_2, lw=1, ls=(0, (3, 3)))
        ax.axhline(0, color=v.AXIS, lw=0.8)
        ax.set_xticks(range(0, MAX_LAG + 1, 2))
        ax.set_xlabel("Lag k (years catch leads)")
        ax.set_title(f"{title}\nmin Holm p {fmt_p(tab['p_holm'].min())}", fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Correlation r")
    v.suptitle(fig, "H1b — " + ("Significant link between world catch and later overexploitation" if any_sig
                                else "No significant link between world catch and later overexploitation"),
               "Bars: correlation of catch growth with the change in % overexploited k years later; dashed = "
               "±1.96/√n. Blue bar = significant after Holm correction")
    v.layout(fig, 0)
    save_fig(fig, sec, "H1b.01_ccf_catch_vs_overexploited", "FAO via OWID; Sea Around Us")


# =========================================================================== #
# H2a — SEA trends
# =========================================================================== #
def section_h2a() -> None:
    sec = "3_H2a_sea_trends"
    print("H2a: SEA trends")
    sea = sea_series()
    cs = country_series()
    p = st.eez_panel()
    units = [("SEA (5 countries)", "SEA", sea)] + [(c, "country", cs[c]) for c in SEA5]
    # EEZ rows ordered by country; single-EEZ countries get an "(EEZ)" suffix so names stay unique
    eezs = sorted(p["eez"].unique(), key=lambda e: (SEA5.index(v.EEZ_COUNTRY[e]), st.eez_name(e)))
    for eez in eezs:
        name = st.eez_name(eez)
        name = f"{name} (EEZ)" if name in SEA5 else name
        units.append((name, "EEZ", p[p["eez"] == eez].set_index("year")[["over", "mtl"]]))
    rows = []
    for var, vname in [("over", "% overexploited"), ("mtl", "mean trophic level")]:
        for lo in (1950, 1990):
            for name, level, df in units:
                rows.append({"variable": vname, "window": f"{lo}–2019", "unit": name, "level": level,
                             **mann_kendall(df[var].loc[lo:2019])})
    res = pd.DataFrame(rows)
    # Holm within each family: variable × window × level (EEZs and countries adjusted separately)
    res["p_holm"] = res.groupby(["variable", "window", "level"])["p"].transform(lambda s: holm(s))
    res["significant_holm"] = res["p_holm"] < ALPHA
    fao = pd.DataFrame([{"unit": f"FAO {n}", **mann_kendall(fao_area(n))}
                        for n in ["Western Central Pacific", "Eastern Indian Ocean"]])
    save_table(res, "3_H2a_mann_kendall_sen_sea")
    save_table(fao, "3_H2a_mann_kendall_fao_areas")

    so = res[(res["variable"] == "% overexploited") & (res["window"] == "1950–2019") & (res["unit"] == "SEA (5 countries)")].iloc[0]
    sm_ = res[(res["variable"] == "mean trophic level") & (res["window"] == "1990–2019") & (res["level"] == "country")]
    n_up = int(((res["variable"] == "% overexploited") & (res["level"] != "SEA") & (res["window"] == "1950–2019")
                & res["significant_holm"] & (res["trend"] == "increasing")).sum())
    down = sm_[sm_["significant_holm"] & (sm_["trend"] == "decreasing")]["unit"].tolist()
    decide("H2a", "Is there a significant worsening trend in SEA (rising % overexploited or falling MTL)?",
           f"SEA % overexploited MK p {fmt_p(so['p'])}, Sen +{so['sen_slope'] * 10:.1f} pp/decade; "
           f"{n_up}/16 countries+EEZs rising after Holm; FAO Area 71 p {fmt_p(fao.iloc[0]['p'])}",
           so["p"] < ALPHA and so["trend"] == "increasing",
           "MTL ('fishing down') falls significantly 1990–2019 only in: " + (", ".join(down) or "none")
           + "; SEA-wide MTL rises, so the worsening shows in stock status rather than trophic level")

    # Figure: forest of Sen slopes per decade
    order = [u[0] for u in units]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.4), sharey=True)
    for ax, (vname, scale, unit_lbl) in zip(axes, [("% overexploited", 10, "pp per decade"),
                                                     ("mean trophic level", 10, "trophic level per decade")]):
        for k, (lo, colour, off) in enumerate([(1950, v.SERIES[0], 0.17), (1990, v.SERIES[1], -0.17)]):
            sub = res[(res["variable"] == vname) & (res["window"] == f"{lo}–2019")].set_index("unit").loc[order]
            ys = np.arange(len(order))[::-1] + off
            for y_, (_, r) in zip(ys, sub.iterrows()):
                ax.plot([r["sen_ci_low"] * scale, r["sen_ci_high"] * scale], [y_, y_], color=colour, lw=1.6)
                ax.scatter(r["sen_slope"] * scale, y_, s=42, zorder=3, edgecolor=colour, lw=1.4,
                           color=colour if r["significant_holm"] else v.SURFACE,
                           label=f"{lo}–2019" if y_ == ys[0] else None)
        ax.axvline(0, color=v.AXIS, lw=0.8)
        for y_ in (len(order) - 1.5, len(order) - 6.5):
            ax.axhline(y_, color=v.GRID, lw=1)
        ax.set_xlabel(f"Sen's slope, {unit_lbl} (95% CI)")
        ax.set_title(vname.capitalize(), fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order))[::-1], order, fontsize=9)
    axes[0].legend(loc="lower right", title="Window (filled = significant\nafter Holm)", title_fontsize=8.5)
    v.suptitle(fig, "H2a — Overexploitation rises across Southeast Asia; 'fishing down' only in places",
               "Sen's slope per decade, Hamed-Rao Mann-Kendall test. Rows: SEA, 5 countries, 11 EEZs")
    v.layout(fig, 0)
    save_fig(fig, sec, "H2a.01_sen_slopes_sea_countries_eez", "Sea Around Us (11 SEA EEZs)")


# =========================================================================== #
# H2b — SEA catch -> MTL
# =========================================================================== #
def section_h2b() -> None:
    sec = "4_H2b_sea_ccf"
    print("H2b: SEA catch -> MTL")
    ensure_transformed()
    tabs = []
    names = ["SEA"] + SEA5
    for n in names:
        key = "SEA" if n == "SEA" else n
        x = TRANSFORMED[f"{key} catch, t (SAU)"]
        y = TRANSFORMED[f"{key} mean trophic level (SAU)"]
        tabs.append(ccf_table(x, y, f"{n}: catch growth -> Δ MTL"))
    ccf = pd.concat(tabs, ignore_index=True)
    ccf["p_holm_all_66"] = holm(ccf["p"])            # stricter: all 6 units × 11 lags
    save_table(ccf, "4_H2b_ccf_catch_mtl")
    sig = ccf[ccf["significant_holm"]]
    best = ccf.loc[ccf["p"].idxmin()]
    decide("H2b", "Does SEA catch cross-correlate with mean trophic level at some lag?",
           f"Smallest raw p: {best['pair'].split(':')[0]} lag {int(best['lag'])}, r = {best['r']:+.2f} "
           f"(p {fmt_p(best['p'])}, Holm within unit p {fmt_p(best['p_holm'])}); "
           f"{len(sig)} of 66 lag tests significant after Holm",
           len(sig) > 0,
           "Stock status is not used as the response: Sea Around Us derives it from catch (circular)")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), sharex=True, sharey=True)
    for ax, tab, n in zip(axes.flat, tabs, names):
        cols = [v.SERIES[0] if s else (v.COUNTRY_COLOR.get(n, v.INK_2) if n != "SEA" else v.INK_2)
                for s in tab["significant_holm"]]
        ax.bar(tab["lag"], tab["r"], color=cols, width=0.6, alpha=0.85)
        ax.plot(tab["lag"], tab["band_95"], color=v.INK_2, lw=1, ls=(0, (3, 3)))
        ax.plot(tab["lag"], -tab["band_95"], color=v.INK_2, lw=1, ls=(0, (3, 3)))
        ax.axhline(0, color=v.AXIS, lw=0.8)
        ax.set_title(f"{n} — min Holm p {fmt_p(tab['p_holm'].min())}", fontsize=9.5)
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Lag k (years catch leads)")
        ax.set_xticks(range(0, MAX_LAG + 1, 2))
    for ax in axes[:, 0]:
        ax.set_ylabel("Correlation r")
    v.suptitle(fig, "H2b — " + ("Catch changes are followed by mean-trophic-level changes in: "
                                + ", ".join(sorted({p.split(':')[0] for p in sig['pair']})) if len(sig)
                                else "Catch changes do not predict later mean-trophic-level changes in Southeast Asia"),
               "Correlation of catch growth with the change in MTL k years later (stationary series); "
               "dashed = ±1.96/√n; blue bar = significant after Holm correction within each panel")
    v.layout(fig, 0)
    save_fig(fig, sec, "H2b.01_ccf_catch_vs_mtl", "Sea Around Us (11 SEA EEZs)")


# =========================================================================== #
# H2c — SEA vs world slope (interaction regression)
# =========================================================================== #
def gap_test(sea: pd.Series, world: pd.Series, label: str, lo: int, hi: int) -> dict:
    """
    Interaction regression y = b0 + b1·t + b2·SEA + b3·t·SEA on stacked SEA/world data.
    With both series on the same years, b3 equals the slope of the gap series
    (SEA − World) on t, so b3 is tested on the gap with HAC errors — this keeps each
    year as one observation and handles autocorrelation correctly.
    """
    d = pd.concat([sea.rename("sea"), world.rename("world")], axis=1).loc[lo:hi].dropna()
    t = d.index.to_numpy(float) - lo
    stacked_y = np.concatenate([d["world"], d["sea"]])
    flag = np.r_[np.zeros(len(d)), np.ones(len(d))]
    Xs = sm.add_constant(np.column_stack([np.r_[t, t], flag, np.r_[t, t] * flag]))
    b3_stacked = sm.OLS(stacked_y, Xs).fit().params[3]
    _, hac = ols_hac((d["sea"] - d["world"]).to_numpy(), sm.add_constant(t), f"H2c gap: {label}")
    ws = sm.OLS(d["world"].to_numpy(), sm.add_constant(t)).fit().params[1]
    ss = sm.OLS(d["sea"].to_numpy(), sm.add_constant(t)).fit().params[1]
    ci = hac.conf_int()[1]
    return {"comparison": label, "window": f"{d.index.min()}–{d.index.max()}", "n_years": len(d),
            "world_slope": ws, "sea_slope": ss, "b3_interaction_stacked": b3_stacked,
            "b3_gap_slope": hac.params[1], "hac_se": hac.bse[1], "ci_low": ci[0], "ci_high": ci[1],
            "p": hac.pvalues[1]}


def section_h2c() -> None:
    sec = "5_H2c_sea_vs_world"
    print("H2c: SEA slope vs world slope")
    sea = sea_series()
    cs = country_series()
    ws, wm = st.world_sau(), sau_world_mtl()
    wf = st.world_fao()
    rows = [gap_test(sea["over"], ws, "SEA vs world, % overexploited (SAU)", 1990, 2019),
            gap_test(sea["over"], ws, "SEA vs world, % overexploited (SAU)", 1950, 2019)]
    rows += [gap_test(cs[c]["over"], ws, f"{c} vs world, % overexploited (SAU)", 1990, 2019) for c in SEA5]
    rows += [gap_test(sea["mtl"], wm, "SEA vs world, mean trophic level (SAU)", 1990, 2019)]
    a71, a57 = fao_area("Western Central Pacific"), fao_area("Eastern Indian Ocean")
    rows += [gap_test(a71, wf.loc[a71.index], "Area 71 vs world, % overexploited (FAO)", 2004, 2021),
             gap_test(a57, wf.loc[a57.index], "Area 57 vs world, % overexploited (FAO)", 2004, 2021),
             gap_test(a71.loc[:2019], wf.loc[a71.index].loc[:2019], "Area 71 vs world, without 2021 (FAO)",
                      2004, 2019)]
    res = pd.DataFrame(rows)
    res["unit"] = np.where(res["comparison"].str.contains("trophic"), "trophic level per year", "pp per year")
    # "Worse" = faster rise in % overexploited, or faster fall in MTL -> one-sided direction check
    res["direction"] = np.where(res["comparison"].str.contains("trophic"),
                                np.where(res["b3_gap_slope"] < 0, "SEA worse", "SEA better"),
                                np.where(res["b3_gap_slope"] > 0, "SEA worse", "SEA better"))
    is_country = res["comparison"].str.split(" vs").str[0].isin(SEA5)
    res["p_holm_countries"] = np.nan
    res.loc[is_country, "p_holm_countries"] = holm(res.loc[is_country, "p"])
    save_table(res, "5_H2c_interaction_gap_regression")
    main, full, mtl = res.iloc[0], res.iloc[1], res.iloc[7]
    a71_all, a71_no21 = res.iloc[8], res.iloc[10]
    ctry = res[is_country]
    worse = ctry[(ctry["p_holm_countries"] < ALPHA) & (ctry["direction"] == "SEA worse")]["comparison"].str.split(" vs").str[0].tolist()
    better = ctry[(ctry["p_holm_countries"] < ALPHA) & (ctry["direction"] == "SEA better")]["comparison"].str.split(" vs").str[0].tolist()
    sea_reject = bool(main["p"] < ALPHA and main["b3_gap_slope"] > 0)
    verdict = ("Reject H0" if sea_reject else
               f"Fail to reject H0 for SEA as a whole; reject for {', '.join(worse)}" if worse else None)
    decide("H2c", "Is the SEA trend in % overexploited significantly worse than the world trend?",
           f"SAU 1990–2019: SEA {main['sea_slope']:+.2f} vs world {main['world_slope']:+.2f} pp/yr, "
           f"difference {main['b3_gap_slope']:+.2f} [{main['ci_low']:+.2f}, {main['ci_high']:+.2f}], "
           f"p {fmt_p(main['p'])}; countries worse (Holm): {', '.join(worse) or 'none'}; "
           f"better: {', '.join(better) or 'none'}",
           sea_reject,
           f"1950–2019 difference {full['b3_gap_slope']:+.2f} pp/yr (p {fmt_p(full['p'])}). "
           f"FAO Area 71 vs world {a71_all['b3_gap_slope']:+.2f} pp/yr (p {fmt_p(a71_all['p'])}), but without "
           f"the 2021 jump p {fmt_p(a71_no21['p'])}. SEA MTL rises {abs(mtl['b3_gap_slope']):.4f}/yr "
           f"{'slower' if mtl['b3_gap_slope'] < 0 else 'faster'} than the world's (p {fmt_p(mtl['p'])})",
           verdict)

    # Figure: series + forest of slope differences
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1, 1.15]})
    for s, lbl, colour in [(ws, "World (SAU)", v.INK_2), (sea["over"], "SEA, 5 countries (SAU)", v.SERIES[0])]:
        a1.plot(s.index, s.values, color=colour, lw=2, label=lbl)
        seg = s.loc[1990:2019]
        b, a = np.polyfit(seg.index, seg.values, 1)
        a1.plot(seg.index, a + b * seg.index, color=colour, lw=1.2, ls=(0, (4, 3)))
        v.end_label(a1, seg.index[-1], a + b * seg.index[-1], f"{b:+.2f} pp/yr")
    a1.axvspan(1990, 2019, color=v.GRID, alpha=0.4, lw=0)
    a1.set_ylabel("% of stocks collapsed or over-exploited")
    a1.legend(loc="upper left")
    a1.set_title("Dashed: linear fit 1990–2019 (shaded)", fontsize=9.5)
    fr = res[res["unit"] == "pp per year"].reset_index(drop=True)   # MTL row has other units: table only
    lbls = [f"{r['comparison'].replace(', % overexploited', '').replace(' vs world', '')} {r['window']}"
            for _, r in fr.iterrows()]
    for i, (_, r) in enumerate(fr.iterrows()):
        y = len(fr) - 1 - i
        sig = r["p"] < ALPHA
        colour = v.SERIES[7] if r["direction"] == "SEA worse" else v.SERIES[5]
        a2.plot([r["ci_low"], r["ci_high"]], [y, y], color=colour, lw=2)
        a2.scatter(r["b3_gap_slope"], y, s=55, color=colour if sig else v.SURFACE, edgecolor=colour, lw=1.5, zorder=3)
    a2.axvline(0, color=v.AXIS, lw=0.8)
    a2.set_yticks(range(len(fr)), lbls[::-1], fontsize=8.5)
    a2.set_xlabel("Slope difference SEA − world (b3), pp per year, 95% HAC CI")
    a2.grid(axis="y", visible=False)
    a2.set_title("% overexploited: red = rising faster than world, green = slower; filled = p < 0.05",
                 fontsize=9)
    v.suptitle(fig, ("H2c — Since 1990 Southeast Asia overall tracks the world trend; "
                     + (f"{', '.join(worse)} worsen faster" if worse else "no country worsens faster"))
               if not sea_reject else "H2c — Since 1990, overexploitation has risen faster in Southeast Asia",
               "Interaction regression y = b0 + b1·year + b2·SEA + b3·year×SEA; b3 tested on the SEA − world gap "
               "with Newey-West errors")
    v.layout(fig, 0)
    save_fig(fig, sec, "H2c.01_sea_vs_world_slopes", "Sea Around Us; FAO via OWID")


# =========================================================================== #
# Diagnostics + decisions
# =========================================================================== #
def section_diagnostics() -> None:
    sec = "6_diagnostics"
    print("Diagnostics")
    wf = st.world_fao()
    t = wf.index.to_numpy(float)
    fit = sm.OLS(wf.values, sm.add_constant(t - t.min())).fit()
    r = pd.Series(fit.resid, index=wf.index)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].plot(r.index, r.values, color=v.SERIES[0], marker="o", markersize=4, lw=1.2)
    axes[0].axhline(0, color=v.AXIS, lw=0.8)
    axes[0].set_title(f"Residuals over time — Durbin-Watson {durbin_watson(r):.2f}", fontsize=9.5)
    axes[0].set_ylabel("Residual (pp)")
    ac = acf(r, nlags=12)
    axes[1].bar(range(13), ac, color=v.SERIES[0], width=0.6)
    band = 1.96 / np.sqrt(len(r))
    axes[1].axhspan(-band, band, color=v.GRID, alpha=0.6, lw=0, zorder=0)
    axes[1].axhline(0, color=v.AXIS, lw=0.8)
    lb = acorr_ljungbox(r, lags=[10])["lb_pvalue"].iloc[0]
    axes[1].set_title(f"Residual ACF — Ljung-Box(10) p {fmt_p(lb)}", fontsize=9.5)
    axes[1].set_xlabel("Lag (years)")
    (osm, osr), (sl, ic, _) = stats.probplot(r.values)
    axes[2].plot(osm, ic + sl * osm, color=v.AXIS, lw=1.2)
    axes[2].scatter(osm, osr, s=20, color=v.SERIES[0], edgecolor=v.SURFACE, lw=1)
    axes[2].set_title(f"Normal Q-Q — Shapiro-Wilk p {fmt_p(stats.shapiro(r).pvalue)}", fontsize=9.5)
    axes[2].set_xlabel("Theoretical quantile")
    v.suptitle(fig, "Diagnostics — residuals of the H1a linear trend model",
               "Residuals are autocorrelated (low Durbin-Watson, Ljung-Box p < 0.05) → Newey-West (HAC) standard "
               "errors and the Hamed-Rao Mann-Kendall test are used")
    v.layout(fig, 0)
    save_fig(fig, sec, "D.01_h1a_residual_diagnostics", "FAO via OWID")


SECTIONS = {"stationarity": section_stationarity, "h1a": section_h1a, "h1b": section_h1b,
            "h2a": section_h2a, "h2b": section_h2b, "h2c": section_h2c, "diagnostics": section_diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal hypothesis tests H1a–H2c.")
    parser.add_argument("--only", nargs="+", choices=SECTIONS)
    args = parser.parse_args()
    v.setup_style()
    for name in args.only or SECTIONS:
        SECTIONS[name]()
    if not args.only:
        save_table(pd.DataFrame(DIAGNOSTICS), "6_regression_diagnostics")
        dec = pd.DataFrame(DECISIONS)
        h1 = dec[dec["hypothesis"].isin(["H1a", "H1b"])]["decision"].eq("Reject H0").all()
        dec.loc[len(dec)] = {"hypothesis": "Stage rule", "question": "Reject H0 for both H1a and H1b?",
                             "evidence": "; ".join(f"{h}: {d}" for h, d in zip(dec["hypothesis"], dec["decision"])
                                                   if h in ("H1a", "H1b")),
                             "decision": "Proceed to Stage 2" if h1 else "Rule not met",
                             "note": "" if h1 else "Stage 2 results are reported as exploratory/conditional"}
        save_table(dec, "decisions")
        print(dec[["hypothesis", "decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
