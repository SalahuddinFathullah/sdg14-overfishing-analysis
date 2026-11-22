"""
hypothesis_tests.py — Formal tests of hypotheses H1a–H2c (+ H3), in 3 iterations
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

  Iteration 2 (fishing effort instead of catch as the pressure variable; iteration-1
  results above are kept so the analysis history stays visible):
    i2_stationarity  ADF + KPSS on log effort and effort growth
    i2_cpue          catch per unit effort (CPUE) trends, Mann-Kendall on log CPUE
    i2_h1b           H1b repeated with world effort: CCF, prewhitening, Granger,
                     cointegration (constant and constant + trend)
    i2_h2b           H2b repeated with SEA EEZ effort, responses % overexploited
                     (no longer circular) and mean trophic level

  Iteration 3 (ocean warming as an alternative driver, new hypothesis H3):
    i3_stationarity  ADF + KPSS on world SST anomaly, ocean heat content, SEA EEZ SST
    i3_h3a           global: levels vs detrended correlation, cointegration (with and
                     without trend, and jointly with effort), CCF, Granger
    i3_h3b           SEA: SST trends (Mann-Kendall), CCF of Δ SST vs Δ % overexploited,
                     Δ MTL and Δlog catch per EEZ group, Holm-corrected

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
# ITERATION 2 — fishing effort as the pressure variable
# =========================================================================== #
# Iteration 1 (above) used catch as the measure of fishing pressure and found no
# link (H1b, H2b). Catch is a weak proxy: it can stay flat while pressure rises,
# because each unit of effort catches less as stocks are depleted. Iteration 2
# repeats H1b and H2b with fishing EFFORT (IMAS Global Fishing Effort database:
# active vessels x engine power x days at sea), and adds catch per unit effort
# (CPUE) as an explanatory check. Iteration-1 results are kept unchanged.
# Effort is also independent of Sea Around Us stock status, so — unlike catch —
# stock status can be used as a response here without circularity.

EFFORT = VAL / "effort"
BREAK_DLOG = 0.5   # |Δlog effort| > 0.5 (a >65% jump in one year) = break in the source data


def world_effort() -> pd.DataFrame:
    """World fishing effort 1950–2017 (kW × days at sea; nominal and with technological creep)."""
    return pd.read_csv(EFFORT / "world_effort.csv").set_index("year")


def sea_effort() -> dict[str, pd.DataFrame]:
    """Effort by all fleets inside each SEA5 EEZ, 1950–2017; 'SEA' = sum of the five."""
    d = pd.read_csv(EFFORT / "sea_eez_effort.csv")
    d["eez"] = d["eez"].replace({"Viet Nam": "Vietnam"})
    out = {c: d[d["eez"] == c].set_index("year").drop(columns="eez") for c in SEA5}
    out["SEA"] = sum(out[c] for c in SEA5)
    return out


def effort_growth(e: pd.Series) -> tuple[pd.Series, list[int], str]:
    """
    Stationary effort series = Δlog effort (growth rate), with single-year jumps above
    BREAK_DLOG removed: those are changes of source/method in the national statistics
    behind the database (e.g. Thailand 2002 and 2011), not real changes in fishing.
    Detrended if ADF still finds a unit root.
    """
    d = np.log(e).diff().dropna()
    breaks = [int(y) for y in d.index[d.abs() > BREAK_DLOG]]
    d = d.drop(breaks)
    label = "Δlog (growth rate)" + (", source breaks removed" if breaks else "")
    if adf_p(d, "c") >= ALPHA:
        d = detrend(d)
        label = "detrended " + label
    return d, breaks, label


def section_i2_stationarity() -> None:
    print("Iteration 2: effort stationarity")
    rows = []
    we, se = world_effort(), sea_effort()
    specs = [("World nominal effort", we["nom_kwdays"]), ("World effective effort", we["eff_kwdays"])]
    specs += [(f"{n} EEZ nominal effort", se[n]["nom_kwdays"]) for n in ["SEA"] + SEA5]
    for name, s in specs:
        d, breaks, label = effort_growth(s)
        a, k = adf_p(d, "c"), kpss_p(d, "c")
        rows.append({"series": name, "first": int(s.index.min()), "last": int(s.index.max()),
                     "level_adf_p": adf_p(np.log(s), "ct"), "level_kpss_p": kpss_p(np.log(s), "ct"),
                     "transformation_used": label, "breaks_removed": " ".join(map(str, breaks)) or "none",
                     "transformed_adf_p": a, "transformed_kpss_p": k,
                     "stationary_after": "yes" if a < ALPHA and k >= ALPHA else
                     ("ADF yes, KPSS no" if a < ALPHA else "no — CCF results for this series are unreliable")})
    save_table(pd.DataFrame(rows), "7_I2_stationarity_effort")


def section_i2_h1b() -> None:
    sec = "7_iteration2_effort"
    print("Iteration 2: H1b with fishing effort")
    we = world_effort()
    x = effort_growth(we["nom_kwdays"])[0]
    xe = effort_growth(we["eff_kwdays"])[0]
    yf = st.world_fao().diff().dropna()
    ys = st.world_sau().diff().dropna().loc[:2017]
    raw = ccf_table(x.loc[1975:], yf, "World effort growth -> Δ FAO % overexploited")
    pwx, pwy, p_ar = prewhiten(x.loc[1975:], yf)
    pw = ccf_table(pwx, pwy, f"same, prewhitened (AR({p_ar}))")
    sau = ccf_table(x, ys, "World effort growth -> Δ SAU % overexploited (1951–2017)")
    eff = ccf_table(xe.loc[1975:], yf, "Effective effort (with technological creep) -> Δ FAO %")
    ccf = pd.concat([raw, pw, sau, eff], ignore_index=True)
    save_table(ccf, "7_I2_H1b_ccf_effort")
    g = granger(x.loc[1975:], yf)
    g.insert(0, "pair", "World effort growth -> Δ FAO % overexploited")
    save_table(g, "7_I2_H1b_granger_effort")
    rows = []
    for name, y in [("FAO % overexploited", st.world_fao()), ("SAU % overexploited", st.world_sau())]:
        lv = pd.concat([y, np.log(we["nom_kwdays"])], axis=1).dropna()
        for tr in ("c", "ct"):
            t_c, p_c, _ = coint(lv.iloc[:, 0], lv.iloc[:, 1], trend=tr)
            rows.append({"response": name, "driver": "log world nominal effort", "trend_term": tr,
                         "years": f"{lv.index.min()}–{lv.index.max()}", "t_stat": t_c, "p": p_c})
    co = pd.DataFrame(rows)
    save_table(co, "7_I2_H1b_cointegration_effort")
    sig = bool(raw["significant_holm"].any() or pw["significant_holm"].any() or sau["significant_holm"].any())
    best = raw.loc[raw["p"].idxmin()]
    c_fao = co[(co["response"] == "FAO % overexploited") & (co["trend_term"] == "c")].iloc[0]
    decide("H1b (iteration 2: effort)", "Does world fishing effort cross-correlate with % stocks overexploited at some lag?",
           f"Largest effect r = {best['r']:+.2f} at lag {int(best['lag'])} (raw p {fmt_p(best['p'])}, "
           f"Holm p {fmt_p(best['p_holm'])}); SAU min Holm p {fmt_p(sau['p_holm'].min())}; "
           f"Granger min Holm p {fmt_p(g['p_holm'].min())}; cointegration with FAO p {fmt_p(c_fao['p'])}",
           sig,
           "Short-run (year-to-year) changes in effort do not line up with changes in stock status; the long-run "
           f"level link is weak (cointegration p {fmt_p(c_fao['p'])}, 10% level only). See CPUE for the long-run story.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, tab, title in zip(axes, [raw, sau, eff], ["Effort → FAO stocks (1975–2017)",
                                                      "Effort → SAU stocks (1951–2017)",
                                                      "Effective effort → FAO stocks\n(non-stationary: indicative only)"]):
        cols = [v.SERIES[0] if s else v.NEUTRAL for s in tab["significant_holm"]]
        ax.bar(tab["lag"], tab["r"], color=cols, width=0.6)
        ax.plot(tab["lag"], tab["band_95"], color=v.INK_2, lw=1, ls=(0, (3, 3)))
        ax.plot(tab["lag"], -tab["band_95"], color=v.INK_2, lw=1, ls=(0, (3, 3)))
        ax.axhline(0, color=v.AXIS, lw=0.8)
        ax.set_xticks(range(0, MAX_LAG + 1, 2))
        ax.set_xlabel("Lag k (years effort leads)")
        ax.set_title(f"{title}\nmin Holm p {fmt_p(tab['p_holm'].min())}", fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Correlation r")
    v.suptitle(fig, "Iteration 2 · H1b with fishing effort — " + (
        "significant lag found" if sig else "still no significant short-run link"),
               "Effort growth vs change in % overexploited k years later; dashed = ±1.96/√n; blue = significant "
               f"after Holm. Long-run (cointegration) p {fmt_p(c_fao['p'])} for FAO")
    v.layout(fig, 0)
    save_fig(fig, sec, "I2.02_h1b_ccf_effort", "IMAS Global Fishing Effort (Rousseau et al.); FAO via OWID; Sea Around Us")


def section_i2_h2b() -> None:
    sec = "7_iteration2_effort"
    print("Iteration 2: H2b with fishing effort")
    se = sea_effort()
    sea, cs = sea_series(), country_series()
    tabs = []
    for n in ["SEA"] + SEA5:
        x = effort_growth(se[n]["nom_kwdays"])[0]
        src = sea if n == "SEA" else cs[n]
        for resp, lbl in [("over", "Δ % overexploited"), ("mtl", "Δ MTL")]:
            t = ccf_table(x, src[resp].diff().dropna().loc[:2017], f"{n}: effort growth -> {lbl}")
            t.insert(0, "unit", n)
            t.insert(1, "response", lbl)
            tabs.append(t)
    ccf = pd.concat(tabs, ignore_index=True)
    ccf["p_holm_all"] = holm(ccf["p"])
    save_table(ccf, "7_I2_H2b_ccf_effort")
    sig = ccf[ccf["significant_holm"]]
    decide("H2b (iteration 2: effort)",
           "Does SEA fishing effort cross-correlate with % overexploited or mean trophic level at some lag?",
           f"{len(sig)} of {len(ccf)} lag tests significant after Holm within unit"
           + (": " + "; ".join(f"{r['unit']} {r['response']} lag {int(r['lag'])} r = {r['r']:+.2f} "
                                 f"(Holm p {fmt_p(r['p_holm'])})" for _, r in sig.iterrows()) if len(sig) else "")
           + f"; across all {len(ccf)} tests min Holm p {fmt_p(ccf['p_holm_all'].min())}",
           len(sig) > 0,
           "Stock status can be tested here (effort is independent of SAU's catch-based status). "
           "Thailand's effort series stays non-stationary (source breaks), so its lags are unreliable.",
           verdict=None if len(sig) != 1 else
           (f"Weak: reject H0 only for {sig.iloc[0]['unit']} {sig.iloc[0]['response']} at lag "
            f"{int(sig.iloc[0]['lag'])} (r = {sig.iloc[0]['r']:+.2f}); not significant once all "
            f"{len(ccf)} tests are corrected together"))

    # Heatmap: rows = unit × response, columns = lag, colour = r, ring = significant after Holm
    rows = [(u, r) for u in ["SEA"] + SEA5 for r in ["Δ % overexploited", "Δ MTL"]]
    mat = np.array([[ccf[(ccf["unit"] == u) & (ccf["response"] == r) & (ccf["lag"] == k)]["r"].iloc[0]
                     for k in range(MAX_LAG + 1)] for u, r in rows])
    sigm = np.array([[ccf[(ccf["unit"] == u) & (ccf["response"] == r) & (ccf["lag"] == k)]["significant_holm"].iloc[0]
                      for k in range(MAX_LAG + 1)] for u, r in rows])
    fig, ax = plt.subplots(figsize=(13, 6.6))
    im = ax.imshow(mat, cmap=v.DIVERGING, vmin=-0.5, vmax=0.5, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(mat[i, j]) > 0.35 else v.INK,
                    fontweight="bold" if sigm[i, j] else "normal")
            if sigm[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec=v.INK, lw=1.6))
    for b in range(2, len(rows), 2):
        ax.axhline(b - 0.5, color=v.SURFACE, lw=3)
    ax.set_xticks(range(MAX_LAG + 1), range(MAX_LAG + 1))
    ax.set_yticks(range(len(rows)), [f"{u} · {r}" for u, r in rows], fontsize=9)
    ax.set_xlabel("Lag k (years effort leads)")
    ax.grid(False)
    for spn in ax.spines.values():
        spn.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Correlation r", color=v.INK_2)
    cb.outline.set_visible(False)
    v.suptitle(fig, "Iteration 2 · H2b with fishing effort — " + (
        f"{len(sig)} of {len(ccf)} lag tests significant after Holm (outlined)" if len(sig)
        else "no significant lag after Holm correction"),
               "Correlation of effort growth with the change in each response k years later, per EEZ group; "
               "red = positive, blue = negative")
    v.layout(fig, 0)
    save_fig(fig, sec, "I2.03_h2b_ccf_effort_heatmap", "IMAS Global Fishing Effort (Rousseau et al.); Sea Around Us")


def section_i2_cpue() -> None:
    """
    Catch per unit effort (CPUE) = catch ÷ nominal effort. A falling CPUE means each
    kW-day of fishing lands less fish — the classic sign of a shrinking stock. It
    explains how catch can stay flat while overexploitation rises (iteration-1 H1b).
    """
    sec = "7_iteration2_effort"
    print("Iteration 2: catch per unit effort")
    we, se = world_effort(), sea_effort()
    sea, cs = sea_series(), country_series()
    units = {"World": (sau_world_catch().loc[:2017], we["nom_kwdays"])}
    units["SEA"] = (sea["catch"].loc[:2017], se["SEA"]["nom_kwdays"])
    units.update({c: (cs[c]["catch"].loc[:2017], se[c]["nom_kwdays"]) for c in SEA5})
    rows, cpue = [], {}
    for n, (c, e) in units.items():
        cp = (c / e * 1000).dropna()                      # kg of catch per kW-day... t × 1000 = kg
        cpue[n] = cp
        r = mann_kendall(np.log(cp.loc[1990:2017]))
        def chg(x: pd.Series) -> float:
            """% change between 5-year means 1990–94 and 2013–17 (robust to single noisy years)."""
            return (x.loc[2013:2017].mean() / x.loc[1990:1994].mean() - 1) * 100

        rows.append({"unit": n, "cpue_1990_94_kg_per_kwday": cp.loc[1990:1994].mean(),
                     "cpue_2013_17_kg_per_kwday": cp.loc[2013:2017].mean(),
                     "change_1990_2017_pct": chg(cp), "catch_change_1990_2017_pct": chg(c),
                     "effort_change_1990_2017_pct": chg(e),
                     "mk_p_log_cpue_1990_2017": r["p"], "trend": r["trend"],
                     "sen_pct_per_year": (np.exp(r["sen_slope"]) - 1) * 100,
                     "sen_ci_low_pct": (np.exp(r["sen_ci_low"]) - 1) * 100,
                     "sen_ci_high_pct": (np.exp(r["sen_ci_high"]) - 1) * 100})
    res = pd.DataFrame(rows)
    res["p_holm_countries"] = np.nan
    m = res["unit"].isin(SEA5)
    res.loc[m, "p_holm_countries"] = holm(res.loc[m, "mk_p_log_cpue_1990_2017"])
    save_table(res, "7_I2_cpue_trends")
    w, s_ = res.iloc[0], res.iloc[1]
    falling = res[m & (res["trend"] == "decreasing") & (res["p_holm_countries"] < ALPHA)]["unit"].tolist()
    decide("Explanation (iteration 2: CPUE)", "Is catch per unit effort falling (more effort for the same fish)?",
           f"World CPUE {w['change_1990_2017_pct']:+.0f}% 1990–94 → 2013–17 (catch {w['catch_change_1990_2017_pct']:+.0f}%, "
           f"effort {w['effort_change_1990_2017_pct']:+.0f}%), Sen {w['sen_pct_per_year']:+.1f}%/yr, p "
           f"{fmt_p(w['mk_p_log_cpue_1990_2017'])}; SEA {s_['change_1990_2017_pct']:+.0f}% (Sen "
           f"{s_['sen_pct_per_year']:+.1f}%/yr, p {fmt_p(s_['mk_p_log_cpue_1990_2017'])}); falling significantly in "
           f"{', '.join(falling) or 'none'}",
           bool(w["mk_p_log_cpue_1990_2017"] < ALPHA and w["trend"] == "decreasing"),
           "Not a hypothesis from the plan; supports the depletion explanation for flat catch + rising overexploitation",
           verdict=("CPUE falling" if w["trend"] == "decreasing" and w["mk_p_log_cpue_1990_2017"] < ALPHA
                    else "No clear CPUE trend"))

    # Figure: (a) world index 1990 = 100, (b) CPUE by SEA unit, index 1990 = 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.4))
    wf = st.world_fao()
    lines = [("Fishing effort", we["nom_kwdays"], v.SERIES[1]),
             ("Catch (Sea Around Us)", sau_world_catch(), v.SERIES[0]),
             ("Catch per unit effort", cpue["World"], v.SERIES[3]),
             ("% stocks overexploited (FAO)", wf, v.SERIES[7])]
    items = []
    for lbl, s, colour in lines:
        s = s.loc[1974:2017]
        idx = s / s.loc[1990:1994].mean() * 100
        a1.plot(idx.index, idx.values, color=colour, lw=2, label=lbl)
        items.append((idx.index[-1], idx.values[-1], f"{idx.values[-1]:.0f}"))
    v.end_labels(a1, items)
    a1.axhline(100, color=v.AXIS, lw=0.8)
    a1.axvspan(1990, 1994, color=v.GRID, alpha=0.5, lw=0)
    a1.set_ylabel("Index, 1990–94 average = 100")
    a1.legend(loc="upper left", fontsize=8.5)
    a1.set_title("World: effort and overexploitation up, catch flat → catch per unit effort down", fontsize=9.5)
    items = []
    for n in ["SEA"] + SEA5:
        s = cpue[n].loc[1974:2017]
        idx = s / s.loc[1990:1994].mean() * 100
        colour = v.INK_2 if n == "SEA" else v.COUNTRY_COLOR[n]
        a2.plot(idx.index, idx.values, color=colour, lw=2.4 if n == "SEA" else 1.6, label=n)
    a2.axhline(100, color=v.AXIS, lw=0.8)
    a2.axvspan(1990, 1994, color=v.GRID, alpha=0.5, lw=0)
    a2.legend(loc="upper right", fontsize=8.5, ncol=2)
    a2.set_ylabel("Catch per unit effort, index 1990–94 = 100")
    a2.set_title(f"Southeast Asia: CPUE falls in {', '.join(falling) or 'no country'} (Holm-corrected Mann-Kendall)",
                 fontsize=9.5)
    v.suptitle(fig, "Iteration 2 · Why overexploitation rises while catch stays flat: each unit of effort catches less",
               f"Catch per kW-day, 1990–94 → 2013–17: world {w['change_1990_2017_pct']:+.0f}%, SEA "
               f"{s_['change_1990_2017_pct']:+.0f}% — more fishing for the same catch, the signature of depleted stocks")
    v.layout(fig, 0)
    save_fig(fig, sec, "I2.01_cpue_effort_catch_index", "IMAS Global Fishing Effort (Rousseau et al.); Sea Around Us; FAO via OWID")


# =========================================================================== #
# ITERATION 3 — ocean warming as an alternative driver
# =========================================================================== #
# Iterations 1–2 found no short-run link between fishing pressure (catch, effort)
# and % stocks overexploited. Iteration 3 asks whether ocean warming lines up with
# overexploitation instead (new hypothesis H3):
#   H3a (global) world sea-surface temperature anomaly (HadSST, via OWID) and upper-
#       700 m ocean heat content (NOAA, via OWID) vs world % overexploited (FAO, SAU)
#   H3b (SEA)    annual SST inside each SEA5 EEZ (NOAA ERSST v5, area-weighted over the
#       EEZ's grid cells) vs SAU % overexploited, mean trophic level and catch
# Both warming and overexploitation trend upward, so a level correlation is expected
# whatever the cause. The tests therefore separate (1) a shared trend, (2) a long-run
# link beyond the trend (cointegration with and without a trend term, and jointly
# with effort), and (3) a year-to-year link (CCF of the stationary series).

CLIMATE = VAL / "climate"


def world_sst() -> pd.Series:
    """World sea-surface temperature anomaly (°C vs 1961–1990), HadSST via OWID; 1950–2025 (2026 is partial)."""
    d = v.owid("sea-surface-temperature-anomaly")
    return d[d["Entity"] == "World"].set_index("Year")["Average"].loc[1950:2025]


def world_ohc() -> pd.Series:
    """Upper-700 m ocean heat content anomaly (10^22 J), NOAA series via OWID; 1955–2025."""
    d = v.owid("ocean-heat-content-upper")
    return d[d["Entity"] == "World"].set_index("Year")["NOAA"].dropna()


def sea_sst() -> dict[str, pd.DataFrame]:
    """Annual mean SST (°C), anomaly vs 1971–2000 and hot-month count per SEA5 EEZ and 'SEA', 1950–2025."""
    d = pd.read_csv(CLIMATE / "sea_eez_sst_ersst.csv")
    d["region"] = d["region"].replace({"Viet Nam": "Vietnam"})
    return {r: g.set_index("year").drop(columns="region") for r, g in d.groupby("region")}


def detrended_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float, int]:
    """Spearman ρ of the levels and of the linear-trend residuals (common years)."""
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    rx, ry = detrend(d["x"]), detrend(d["y"])
    r1 = stats.spearmanr(d["x"], d["y"])
    r2 = stats.spearmanr(rx, ry)
    return r1[0], r1[1], r2[0], r2[1], len(d)


def section_i3_stationarity() -> None:
    print("Iteration 3: warming series stationarity")
    rows = []
    specs = [("World SST anomaly (HadSST)", world_sst()), ("World ocean heat content 0–700 m (NOAA)", world_ohc())]
    specs += [(f"{r} EEZ SST (ERSST v5)", s["sst_mean_c"]) for r, s in sea_sst().items()]
    for name, s in specs:
        label, d, a, k = make_stationary(s, log=False)
        rows.append({"series": name, "first": int(s.index.min()), "last": int(s.index.max()),
                     "level_adf_p": adf_p(s, "ct"), "level_kpss_p": kpss_p(s, "ct"),
                     "transformation_used": label, "transformed_adf_p": a, "transformed_kpss_p": k,
                     "stationary_after": "yes" if a < ALPHA and k >= ALPHA else ("ADF yes, KPSS no" if a < ALPHA else "no"),
                     "trend_per_decade": np.polyfit(s.index, s.values, 1)[0] * 10})
    save_table(pd.DataFrame(rows), "8_I3_stationarity_warming")


def section_i3_h3a() -> None:
    sec = "8_iteration3_warming"
    print("Iteration 3: H3a — global warming vs % overexploited")
    sst, ohc = world_sst(), world_ohc()
    wf, ws = st.world_fao(), st.world_sau()
    we = world_effort()
    responses = [("FAO % overexploited", wf), ("SAU % overexploited", ws)]
    drivers = [("SST anomaly", sst), ("Ocean heat content", ohc)]

    # (1) shared trend vs beyond-trend association
    rows = []
    for dn, x in drivers:
        for rn, y in responses:
            r_lv, p_lv, r_dt, p_dt, n = detrended_corr(x, y)
            rows.append({"driver": dn, "response": rn, "n_years": n, "spearman_levels": r_lv, "p_levels": p_lv,
                         "spearman_detrended": r_dt, "p_detrended": p_dt})
    trend = pd.DataFrame(rows)
    save_table(trend, "8_I3_H3a_levels_vs_detrended")

    # (2) long run: cointegration, alone and jointly with effort
    rows = []
    for dn, x in drivers:
        for rn, y in responses:
            lv = pd.concat([y, x], axis=1).dropna()
            for tr in ("c", "ct"):
                t_c, p_c, _ = coint(lv.iloc[:, 0], lv.iloc[:, 1], trend=tr)
                rows.append({"response": rn, "drivers": dn, "trend_term": tr,
                             "years": f"{lv.index.min()}–{lv.index.max()}", "t_stat": t_c, "p": p_c})
    for rn, y in responses:
        lv = pd.concat([y, sst, np.log(we["nom_kwdays"])], axis=1).dropna()
        t_c, p_c, _ = coint(lv.iloc[:, 0], lv.iloc[:, 1:], trend="c")
        rows.append({"response": rn, "drivers": "SST anomaly + log effort", "trend_term": "c",
                     "years": f"{lv.index.min()}–{lv.index.max()}", "t_stat": t_c, "p": p_c})
    co = pd.DataFrame(rows)
    save_table(co, "8_I3_H3a_cointegration")

    # (3) short run: CCF of first differences, Holm across lags
    tabs = []
    for dn, x in drivers:
        for rn, y in responses:
            tabs.append(ccf_table(x.diff().dropna(), y.diff().dropna(), f"Δ {dn} -> Δ {rn}"))
    ccf = pd.concat(tabs, ignore_index=True)
    save_table(ccf, "8_I3_H3a_ccf")
    g = granger(sst.diff().dropna(), wf.diff().dropna())
    g.insert(0, "pair", "Δ SST anomaly -> Δ FAO % overexploited")
    save_table(g, "8_I3_H3a_granger")

    sig = bool(ccf["significant_holm"].any())
    main = trend[(trend["driver"] == "SST anomaly") & (trend["response"] == "FAO % overexploited")].iloc[0]
    c_c = co[(co["response"] == "FAO % overexploited") & (co["drivers"] == "SST anomaly") & (co["trend_term"] == "c")].iloc[0]
    c_ct = co[(co["response"] == "FAO % overexploited") & (co["drivers"] == "SST anomaly") & (co["trend_term"] == "ct")].iloc[0]
    c_sau = co[(co["response"] == "SAU % overexploited") & (co["drivers"] == "SST anomaly") & (co["trend_term"] == "c")].iloc[0]
    c_joint = co[(co["response"] == "FAO % overexploited") & (co["drivers"] == "SST anomaly + log effort")].iloc[0]
    sau_dt = trend[(trend["driver"] == "SST anomaly") & (trend["response"] == "SAU % overexploited")].iloc[0]
    ohc_ct = co[(co["response"] == "FAO % overexploited") & (co["drivers"] == "Ocean heat content")
                & (co["trend_term"] == "ct")].iloc[0]
    decide("H3a (iteration 3: warming, global)",
           "Does ocean warming correlate with world % stocks overexploited beyond their shared upward trend?",
           f"Levels ρ = {main['spearman_levels']:+.2f}; after removing linear trends ρ = "
           f"{main['spearman_detrended']:+.2f} with FAO (p {fmt_p(main['p_detrended'])}) but "
           f"{sau_dt['spearman_detrended']:+.2f} with SAU (p {fmt_p(sau_dt['p_detrended'])}); cointegration with "
           f"constant p {fmt_p(c_c['p'])} (FAO), {fmt_p(c_sau['p'])} (SAU), with trend p {fmt_p(c_ct['p'])} (FAO; "
           f"ocean heat content {fmt_p(ohc_ct['p'])}), jointly with effort p {fmt_p(c_joint['p'])}; CCF min Holm p "
           f"{fmt_p(ccf['p_holm'].min())}; Granger min Holm p {fmt_p(g['p_holm'].min())}",
           sig,
           "Warming and overexploitation share an upward trend; beyond it the two sources disagree in sign (the SAU "
           "series is S-shaped, so a straight-line detrend leaves curvature that matches accelerating warming) and no "
           "year-to-year link exists. The data cannot separate warming from other steadily rising drivers.",
           verdict=None if sig else "Fail to reject H0 (shared trend only; no consistent link beyond it)")

    # Figure: levels vs detrended vs CCF
    d = pd.concat([sst.rename("sst"), wf.rename("over")], axis=1).dropna()
    rx, ry = detrend(d["sst"]), detrend(d["over"])
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 5), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    sc = a1.scatter(d["sst"], d["over"], c=d.index,
                    cmap=matplotlib.colors.LinearSegmentedColormap.from_list("years", ["#cde2fb", "#184f95"]),
                    s=34, edgecolor=v.SURFACE, linewidth=1, zorder=3)
    cb = fig.colorbar(sc, ax=a1, fraction=0.05, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Year", color=v.INK_2)
    a1.set_xlabel("World SST anomaly (°C vs 1961–90)")
    a1.set_ylabel("% of stocks overexploited (FAO)")
    a1.set_title(f"1 · Levels: ρ = {main['spearman_levels']:+.2f} — both simply rise over time", fontsize=9.5)
    a2.scatter(rx, ry, s=34, color=v.SERIES[0], edgecolor=v.SURFACE, linewidth=1, zorder=3)
    a2.axhline(0, color=v.AXIS, lw=0.8)
    a2.axvline(0, color=v.AXIS, lw=0.8)
    a2.set_xlabel("SST anomaly, deviation from its trend (°C)")
    a2.set_ylabel("% overexploited, deviation from its trend (pp)")
    a2.set_title(f"2 · Trend removed: ρ = {main['spearman_detrended']:+.2f} (p {fmt_p(main['p_detrended'])})",
                 fontsize=9.5)
    t = ccf[ccf["pair"] == "Δ SST anomaly -> Δ FAO % overexploited"]
    a3.bar(t["lag"], t["r"], color=[v.SERIES[0] if s else v.NEUTRAL for s in t["significant_holm"]], width=0.6)
    a3.plot(t["lag"], t["band_95"], color=v.INK_2, lw=1, ls=(0, (3, 3)))
    a3.plot(t["lag"], -t["band_95"], color=v.INK_2, lw=1, ls=(0, (3, 3)))
    a3.axhline(0, color=v.AXIS, lw=0.8)
    a3.set_xticks(range(0, MAX_LAG + 1, 2))
    a3.set_xlabel("Lag k (years warming leads)")
    a3.set_ylabel("Correlation r")
    a3.grid(axis="x", visible=False)
    a3.set_title(f"3 · Year-to-year changes: min Holm p {fmt_p(t['p_holm'].min())}", fontsize=9.5)
    v.suptitle(fig, "Iteration 3 · H3a — Warming and overexploitation rise together, but only through a shared trend",
               f"Long run: cointegrated with a constant (p {fmt_p(c_c['p'])}) but not once a trend is allowed "
               f"(p {fmt_p(c_ct['p'])}) or effort is added (p {fmt_p(c_joint['p'])}). Sea Around Us detrended "
               f"ρ = {sau_dt['spearman_detrended']:+.2f} has the opposite sign (curvature, see README)")
    v.layout(fig, 0)
    save_fig(fig, sec, "I3.01_h3a_warming_levels_detrended_ccf", "HadSST via OWID; FAO via OWID")


def section_i3_h3b() -> None:
    sec = "8_iteration3_warming"
    print("Iteration 3: H3b — SEA warming vs overexploitation, MTL, catch")
    ss = sea_sst()
    sea, cs = sea_series(), country_series()
    tabs = []
    for n in ["SEA"] + SEA5:
        x = ss[n]["sst_mean_c"].diff().dropna()
        src = sea if n == "SEA" else cs[n]
        for resp, lbl in [("over", "Δ % overexploited"), ("mtl", "Δ MTL"), ("catch", "Δlog catch")]:
            y = (np.log(src[resp]) if resp == "catch" else src[resp]).diff().dropna()
            t = ccf_table(x, y, f"{n}: Δ SST -> {lbl}")
            t.insert(0, "unit", n)
            t.insert(1, "response", lbl)
            tabs.append(t)
    ccf = pd.concat(tabs, ignore_index=True)
    ccf["p_holm_all"] = holm(ccf["p"])
    save_table(ccf, "8_I3_H3b_ccf_sea")
    trend = pd.DataFrame([{"unit": n, **{k: val for k, val in mann_kendall(ss[n]["sst_mean_c"]).items()
                                          if k in ("n", "p", "trend", "sen_slope", "sen_ci_low", "sen_ci_high")},
                           "hot_months_1950_1989": ss[n]["hot_months"].loc[:1989].mean(),
                           "hot_months_1990_2025": ss[n]["hot_months"].loc[1990:].mean()} for n in ["SEA"] + SEA5])
    save_table(trend, "8_I3_H3b_sst_trends")
    sig = ccf[ccf["significant_holm"]]
    w = trend.iloc[0]
    decide("H3b (iteration 3: warming, SEA)",
           "Do changes in SEA sea temperature cross-correlate with % overexploited, mean trophic level or catch?",
           f"SEA SST +{w['sen_slope'] * 10:.2f} °C/decade (MK p {fmt_p(w['p'])}); {len(sig)} of {len(ccf)} lag tests "
           "significant after Holm within unit and response"
           + (": " + "; ".join(f"{r['unit']} {r['response']} lag {int(r['lag'])} r = {r['r']:+.2f} "
                                 f"(Holm p {fmt_p(r['p_holm'])})" for _, r in sig.iterrows()) if len(sig) else "")
           + f"; across all {len(ccf)} tests min Holm p {fmt_p(ccf['p_holm_all'].min())}",
           len(sig) > 0,
           "Significant lags sit at the edge of the tested range or in a single country and vanish under the "
           "all-test correction, so they are treated as chance findings unless replicated.",
           verdict=None if len(sig) == 0 else
           f"Weak: {len(sig)} isolated lag(s) significant within unit; none across all {len(ccf)} tests")

    # Figure: (a) SST anomaly per unit with Sen slope, (b) CCF heatmap
    fig = plt.figure(figsize=(16, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.45], wspace=0.28)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for n in ["SEA"] + SEA5:
        s = ss[n]["sst_anom_c"].rolling(5, center=True).mean()
        r = trend[trend["unit"] == n].iloc[0]
        colour = v.INK_2 if n == "SEA" else v.COUNTRY_COLOR[n]
        a1.plot(s.index, s.values, color=colour, lw=2.6 if n == "SEA" else 1.5,
                label=f"{n} (+{r['sen_slope'] * 10:.2f} °C/decade)")
    a1.axhline(0, color=v.AXIS, lw=0.8)
    a1.set_ylabel("SST anomaly vs 1971–2000 (°C), 5-yr mean")
    a1.legend(loc="upper left", fontsize=8.5)
    a1.set_title("EEZ sea-surface temperature (ERSST v5, area-weighted)", fontsize=9.5)
    rows = [(u, r) for u in ["SEA"] + SEA5 for r in ["Δ % overexploited", "Δ MTL", "Δlog catch"]]
    mat = np.array([[ccf[(ccf["unit"] == u) & (ccf["response"] == r) & (ccf["lag"] == k)]["r"].iloc[0]
                     for k in range(MAX_LAG + 1)] for u, r in rows])
    sigm = np.array([[bool(ccf[(ccf["unit"] == u) & (ccf["response"] == r) & (ccf["lag"] == k)]["significant_holm"].iloc[0])
                      for k in range(MAX_LAG + 1)] for u, r in rows])
    im = a2.imshow(mat, cmap=v.DIVERGING, vmin=-0.5, vmax=0.5, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            a2.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if abs(mat[i, j]) > 0.35 else v.INK, fontweight="bold" if sigm[i, j] else "normal")
            if sigm[i, j]:
                a2.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec=v.INK, lw=1.6))
    for b in range(3, len(rows), 3):
        a2.axhline(b - 0.5, color=v.SURFACE, lw=3)
    a2.set_xticks(range(MAX_LAG + 1), range(MAX_LAG + 1))
    a2.set_yticks(range(len(rows)), [f"{u} · {r}" for u, r in rows], fontsize=8.5)
    a2.set_xlabel("Lag k (years warming leads)")
    a2.grid(False)
    for spn in a2.spines.values():
        spn.set_visible(False)
    cb = fig.colorbar(im, ax=a2, fraction=0.03, pad=0.02)
    cb.set_label("Correlation r", color=v.INK_2)
    cb.outline.set_visible(False)
    a2.set_title("Δ SST vs later change in each response (outlined = significant after Holm within row)", fontsize=9.5)
    v.suptitle(fig, "Iteration 3 · H3b — Southeast Asian seas are warming, but warming does not track the fisheries "
               "indicators year to year",
               f"SEA +{w['sen_slope'] * 10:.2f} °C/decade; warm-anomaly months per year "
               f"{w['hot_months_1950_1989']:.1f} (1950–89) → {w['hot_months_1990_2025']:.1f} (1990–2025). "
               f"{len(sig)} of {len(ccf)} lag tests significant within row, none across all tests")
    fig.subplots_adjust(top=0.8, bottom=0.08, left=0.05, right=0.97)
    save_fig(fig, sec, "I3.02_h3b_sea_sst_ccf", "NOAA ERSST v5 (PSL); Sea Around Us")


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
            "h2a": section_h2a, "h2b": section_h2b, "h2c": section_h2c,
            # iteration 2: fishing effort replaces catch as the pressure variable
            "i2_stationarity": section_i2_stationarity, "i2_cpue": section_i2_cpue,
            "i2_h1b": section_i2_h1b, "i2_h2b": section_i2_h2b,
            # iteration 3: ocean warming as an alternative driver (new hypothesis H3)
            "i3_stationarity": section_i3_stationarity, "i3_h3a": section_i3_h3a, "i3_h3b": section_i3_h3b,
            "diagnostics": section_diagnostics}


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

        def stage(label: str, h1b_name: str) -> None:
            ok = dec[dec["hypothesis"].isin(["H1a", h1b_name])]["decision"].eq("Reject H0").all()
            dec.loc[len(dec)] = {"hypothesis": label, "question": f"Reject H0 for both H1a and {h1b_name}?",
                                 "evidence": "; ".join(f"{h}: {d}" for h, d in zip(dec["hypothesis"], dec["decision"])
                                                       if h in ("H1a", h1b_name)),
                                 "decision": "Proceed to Stage 2" if ok else "Rule not met",
                                 "note": "" if ok else "Stage 2 results are reported as exploratory/conditional"}

        stage("Stage rule (iteration 1: catch)", "H1b")
        stage("Stage rule (iteration 2: effort)", "H1b (iteration 2: effort)")
        save_table(dec, "decisions")
        print(dec[["hypothesis", "decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
