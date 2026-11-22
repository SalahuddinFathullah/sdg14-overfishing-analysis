"""
hypothesis_tests.py — Formal tests of hypotheses H1a–H2c (+ H3, H4, H5), in 5 iterations
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

  Iteration 4 (shipping and built structures, new hypothesis H4):
    i4_data          driver coverage, container-series splice check, ADF + KPSS
    i4_h4a           global: container traffic, seaborne trade, dams, aquaculture vs world
                     % overexploited — levels vs linear- and quadratic-detrended correlation,
                     cointegration (≥ 30 years), CCF of growth rates (lags 0 to n/3)
    i4_h4b           SEA: the same drivers + liner shipping connectivity, CCF vs Δ% overexploited
                     and Δ MTL per country, Holm within row and across all tests
    i4_h4c           artificial islands: difference-in-differences, Spratly-facing EEZs vs other
                     SEA EEZs around China's 2014 reclamation, randomization inference, placebo
    i4_descriptive   offshore oil & gas platforms (2017–23) and wind turbines (2016–21): too
                     short to test, summarised and plotted

  Iteration 5 (more power and the "restore stocks" half of SDG 14.4, new hypothesis H5):
    i5_h5a           pooled panel: 11 EEZs (5 countries for effort), EEZ + year fixed effects,
                     with and without unit trends, wild cluster bootstrap p-values
    i5_h5b           share of rebuilding stocks 2015–19 vs 2010–14 (paired Wilcoxon across EEZs)
                     and segmented trend at 2015 for SEA, countries and the world
    i5_h5c           Fishing-in-Balance index: Mann-Kendall since 1990, decomposition into
                     catch and trophic-level parts
    i5_h5d           forecast of % overexploited to 2030 (linear trend, damped ETS, ARIMA on the
                     logit scale; model chosen by a 5-year back-test against a no-change forecast)

  Summary (report figure):
    summary          all 8 world drivers from iterations 1–4 on one common method:
                     levels vs linear/quadratic-detrended ρ and CCF, one figure (S.01)

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


def ccf_table(x: pd.Series, y: pd.Series, label: str, max_lag: int = MAX_LAG) -> pd.DataFrame:
    """
    Cross-correlation r_k = corr(x_t, y_{t+k}) for k = 0..max_lag (x leads y by k years),
    Pearson p per lag, Holm-adjusted across the lags, and the ±1.96/√n white-noise band.
    `max_lag` is lowered for short series (iteration 4) so every lag keeps ≥ 2/3 of the years.
    """
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    rows = []
    for k in range(max_lag + 1):
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
# Iteration 4 — human use of the sea: shipping and built structures (new hypothesis H4)
# =========================================================================== #
# Iterations 1–3 found no driver whose changes line up with overexploitation beyond a
# shared upward trend. Iteration 4 tests two families of human activity that grew over
# the same decades and could degrade fish stocks through habitat loss rather than catch:
#   shipping   container port throughput (TEU), seaborne trade volume, liner connectivity
#   structures dams/reservoirs (river flow, sediment and nutrient supply to the coast),
#              aquaculture (habitat conversion, wild-caught feed), offshore oil & gas
#              platforms, offshore wind farms, artificial islands / military outposts
# Same three-step logic as iteration 3: (1) levels vs detrended association,
# (2) cointegration where the series are long enough (n ≥ 30), (3) CCF of growth rates.
# Artificial islands have no annual series, only a dated event (China's Spratly
# reclamation, 2013–2016), so they are tested with a difference-in-differences design.
# Offshore platforms (2017–2023) and wind turbines (2016–2021) overlap the fisheries
# series by 3–4 years, too few for any test; they are shown descriptively.

HUMAN = VAL / "human_use"
ISO3 = {"MYS": "Malaysia", "IDN": "Indonesia", "THA": "Thailand", "VNM": "Vietnam", "PHL": "Philippines",
        "WLD": "World"}
SPLICE: list[dict] = []        # container-series splice check -> 9_I4_container_splice.csv
EVENT_YEAR = 2014              # China's Spratly reclamation: Dec 2013 – 2016 (bulk in 2014–15)
SCS_EEZ = [704, 608, 461, 463]           # Vietnam, Philippines, Sabah, Sarawak: face the Spratlys
OTHER_EEZ = [459, 938, 361, 956, 957]    # Malacca Strait, Indonesian Indian Ocean & east, Thai seas
EDGE_EEZ = [460, 937]                    # Peninsula East, Indonesia Central: SCS edge, not Spratlys


def container_teu() -> pd.DataFrame:
    """
    Container port throughput (TEU) per region, 2000–2024. The current WDI series starts in
    2010; the WDI December-2020 archive covers 2000–2019. Archive years 2000–2009 are rescaled
    by the median current/archive ratio over the 2010–2019 overlap so the joined series has no
    level jump (the match is logged to SPLICE). SEA = sum of the 5 countries.
    """
    d = pd.read_csv(HUMAN / "wb_shipping.csv")
    d = d[d["indicator"] == "IS.SHP.GOOD.TU"]
    out = {}
    SPLICE.clear()
    for iso, g in d.groupby("iso3"):
        a = g[g["source"] == "WDI_archive_202012"].set_index("year")["value"]
        c = g[g["source"] == "WDI_current"].set_index("year")["value"]
        ratio = (c / a).dropna()
        SPLICE.append({"region": ISO3[iso], "overlap_years": f"{ratio.index.min()}–{ratio.index.max()}",
                       "median_ratio_current_to_archive": ratio.median(), "min_ratio": ratio.min(),
                       "max_ratio": ratio.max(),
                       "growth_corr_overlap": np.corrcoef(np.log(a).diff().loc[2011:2019],
                                                          np.log(c).diff().loc[2011:2019])[0, 1]})
        out[ISO3[iso]] = pd.concat([a.loc[:2009] * ratio.median(), c]).sort_index()
    df = pd.DataFrame(out)
    df["SEA"] = df[SEA5].sum(axis=1, min_count=5)
    return df


def lsci() -> pd.DataFrame:
    """UNCTAD liner shipping connectivity index (current WDI, 2006–2021; no world value). SEA = mean."""
    d = pd.read_csv(HUMAN / "wb_shipping.csv")
    d = d[(d["indicator"] == "IS.SHP.GCNW.XQ") & (d["source"] == "WDI_current")]
    df = d.pivot(index="year", columns="iso3", values="value").rename(columns=ISO3)
    df["SEA"] = df[SEA5].mean(axis=1)
    return df


def seaborne() -> pd.DataFrame:
    """
    UNCTAD seaborne trade volume (thousand t), 2000–2024. Countries: goods loaded + discharged
    (total handled). World: goods loaded only (world loaded = world discharged). SEA = sum.
    """
    d = pd.read_csv(HUMAN / "unctad_seaborne_trade.csv")
    tot = d.groupby(["iso3", "year"])["thousand_tonnes"].sum().unstack(0).rename(columns=ISO3)
    tot["World"] = d[(d["iso3"] == "WLD") & (d["flow"] == "loaded")].set_index("year")["thousand_tonnes"]
    tot["SEA"] = tot[SEA5].sum(axis=1, min_count=5)
    return tot


def dams(last: int = 2019) -> pd.DataFrame:
    """
    Cumulative number of dams with a known completion year (GDAT v1), 1900–`last`.
    Decade-only years (e.g. '195') sit at the decade start. Undated dams (34% of GDAT, and
    more than half in Indonesia, Thailand and Vietnam) cannot be placed and are excluded,
    so the series is a lower bound; completions also thin out after ~2005 because the
    database lags construction.
    """
    d = pd.read_csv(HUMAN / "gdat_dams_by_year.csv").dropna(subset=["year"])
    d["year"] = d["year"].astype(int)
    yrs = range(1900, last + 1)
    out = {}
    for reg, g in d.groupby("region"):
        per_year = g.groupby("year")["n_dams"].sum()
        before = per_year.loc[:1899].sum()
        out[ISO3[reg]] = per_year.reindex(yrs, fill_value=0).cumsum() + before
    df = pd.DataFrame(out)
    df["SEA"] = df[SEA5].sum(axis=1)
    return df


def aquaculture() -> pd.DataFrame:
    """Aquaculture production (t, all farmed aquatic animals), FAO via OWID, 1960–2024. SEA = sum."""
    o = v.owid("aquaculture-farmed-fish-production")
    df = o[o["Entity"].isin(SEA5 + ["World"])].pivot(index="Year", columns="Entity",
                                                     values="Aquaculture production (metric tons)")
    df["SEA"] = df[SEA5].sum(axis=1, min_count=5)
    return df


def i4_drivers(unit: str) -> list[tuple[str, pd.Series, bool]]:
    """(name, level series, log?) for one unit ('World', 'SEA' or a country). LSCI has no world value."""
    out = [("Container port traffic", container_teu()[unit].dropna(), True),
           ("Seaborne trade volume", seaborne()[unit].dropna(), True)]
    if unit != "World":
        out.append(("Liner shipping connectivity", lsci()[unit].dropna(), False))
    out += [("Dams completed (cumulative)", dams()[unit], True),
            ("Aquaculture production", aquaculture()[unit].dropna(), True)]
    return out


def quad_detrended_corr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """
    Spearman ρ of residuals after removing a quadratic trend from each series. Robustness check
    for curved (S-shaped or saturating) series, where a straight-line detrend leaves curvature
    that two unrelated series can share; a real link should keep its sign under both detrends.
    """
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    t = d.index.to_numpy(float)
    rx = d["x"] - np.polyval(np.polyfit(t, d["x"], 2), t)
    ry = d["y"] - np.polyval(np.polyfit(t, d["y"], 2), t)
    r = stats.spearmanr(rx, ry)
    return float(r[0]), float(r[1])


def i4_consistent(df: pd.DataFrame) -> pd.Series:
    """Row flag: linear- and quadratic-detrended ρ both Holm-significant and of the same sign."""
    return ((df["p_detrended_holm"] < ALPHA) & (df["p_quad_holm"] < ALPHA)
            & (np.sign(df["spearman_detrended"]) == np.sign(df["spearman_quad_detrended"])))


def i4_change(s: pd.Series, log: bool) -> pd.Series:
    """Growth rate (Δlog) for volumes, first difference for indices."""
    return (np.log(s) if log else s).diff().dropna()


def i4_max_lag(n: int) -> int:
    """Short series: test lags up to n/3 so each lag keeps at least 2/3 of the years."""
    return int(min(MAX_LAG, max(2, n // 3)))


def section_i4_data() -> None:
    print("Iteration 4: driver coverage, container splice, stationarity")
    container_teu()
    save_table(pd.DataFrame(SPLICE), "9_I4_container_splice")
    rows, st_rows = [], []
    for unit in ["World", "SEA"] + SEA5:
        for name, s, log in i4_drivers(unit):
            s = s[s > 0] if log else s
            rows.append({"unit": unit, "driver": name, "first": int(s.index.min()), "last": int(s.index.max()),
                         "n_years": len(s), "growth_per_year_pct": (np.exp(np.polyfit(s.index, np.log(s), 1)[0]) - 1) * 100
                         if log else np.nan,
                         "trend_per_year": np.nan if log else np.polyfit(s.index, s.values, 1)[0]})
            if unit in ("World", "SEA"):
                x = np.log(s) if log else s
                label, d, a, k = make_stationary(s, log)
                st_rows.append({"series": f"{unit}: {name}", "n": len(s), "level_adf_p": adf_p(x, "ct"),
                                "level_kpss_p": kpss_p(x, "ct"), "transformation_used": label,
                                "transformed_adf_p": a, "transformed_kpss_p": k,
                                "stationary_after": "yes" if a < ALPHA and k >= ALPHA
                                else ("ADF yes, KPSS no" if a < ALPHA else "no")})
    save_table(pd.DataFrame(rows), "9_I4_driver_coverage")
    save_table(pd.DataFrame(st_rows), "9_I4_stationarity")


def section_i4_h4a() -> None:
    sec = "9_iteration4_human_use"
    print("Iteration 4: H4a — global shipping and structures vs % overexploited")
    responses = [("FAO % overexploited", st.world_fao()), ("SAU % overexploited", st.world_sau())]
    drivers = i4_drivers("World")

    # (1) shared trend vs beyond-trend association (Spearman on levels and on linear-trend residuals)
    rows = []
    for dn, x, log in drivers:
        xl = np.log(x) if log else x
        for rn, y in responses:
            r_lv, p_lv, r_dt, p_dt, n = detrended_corr(xl, y)
            r_q, p_q = quad_detrended_corr(xl, y)
            rows.append({"driver": dn, "response": rn, "n_years": n, "spearman_levels": r_lv, "p_levels": p_lv,
                         "spearman_detrended": r_dt, "p_detrended": p_dt,
                         "spearman_quad_detrended": r_q, "p_quad_detrended": p_q})
    trend = pd.DataFrame(rows)
    trend["p_detrended_holm"] = holm(trend["p_detrended"])
    trend["p_quad_holm"] = holm(trend["p_quad_detrended"])
    trend["consistent_link"] = i4_consistent(trend)
    save_table(trend, "9_I4_H4a_levels_vs_detrended")

    # (2) long run: Engle-Granger cointegration, only where ≥ 30 common years
    rows = []
    for dn, x, log in drivers:
        xl = np.log(x) if log else x
        for rn, y in responses:
            lv = pd.concat([y, xl], axis=1).dropna()
            if len(lv) < 30:
                rows.append({"response": rn, "driver": dn, "trend_term": "–", "years": f"{lv.index.min()}–{lv.index.max()}",
                             "n": len(lv), "t_stat": np.nan, "p": np.nan, "note": "too short (< 30 years)"})
                continue
            for tr in ("c", "ct"):
                t_c, p_c, _ = coint(lv.iloc[:, 0], lv.iloc[:, 1], trend=tr)
                rows.append({"response": rn, "driver": dn, "trend_term": tr,
                             "years": f"{lv.index.min()}–{lv.index.max()}", "n": len(lv), "t_stat": t_c, "p": p_c,
                             "note": ""})
    co = pd.DataFrame(rows)
    save_table(co, "9_I4_H4a_cointegration")

    # (3) short run: CCF of driver growth vs Δ response, Holm within pair and across all pairs
    tabs = []
    for dn, x, log in drivers:
        dx = i4_change(x, log)
        for rn, y in responses:
            n = len(pd.concat([dx, y.diff()], axis=1).dropna())
            t = ccf_table(dx, y.diff().dropna(), f"{'Δlog' if log else 'Δ'} {dn} -> Δ {rn}", i4_max_lag(n))
            t.insert(0, "driver", dn)
            t.insert(1, "response", rn)
            tabs.append(t)
    ccf = pd.concat(tabs, ignore_index=True)
    ccf["p_holm_all"] = holm(ccf["p"])
    save_table(ccf, "9_I4_H4a_ccf")

    # Decision: a driver counts only if the beyond-trend link is Holm-significant with one sign under BOTH
    # linear and quadratic detrending, in BOTH sources (FAO and SAU), or if a year-to-year (CCF) link
    # survives the all-test correction. Iteration 3 showed linear-detrended residuals of curved series
    # can correlate by accident, so one source or one detrend order is not enough.
    cons = trend[trend["consistent_link"]]
    both = [d for d, g in cons.groupby("driver") if len(g) == 2 and g["spearman_detrended"].apply(np.sign).nunique() == 1]
    sig_ccf = ccf[ccf["p_holm_all"] < ALPHA]
    one = trend[trend["p_detrended_holm"] < ALPHA]
    flips = one[~one["driver"].isin(both)]   # significant once, but not in both sources and both detrends
    reject = bool(both) or len(sig_ccf) > 0
    co_ok = co.dropna(subset=["p"])
    co_txt = "; ".join(f"{r['driver']} × {r['response'].split()[0]} {r['trend_term']} p {fmt_p(r['p'])}"
                       for _, r in co_ok.iterrows())
    decide("H4a (iteration 4: shipping & structures, global)",
           "Do shipping volume, dam building or aquaculture growth track world % stocks overexploited beyond "
           "their shared upward trend?",
           f"Levels ρ {trend['spearman_levels'].min():+.2f} to {trend['spearman_levels'].max():+.2f}; linear-detrended "
           "ρ significant after Holm in "
           + (", ".join(f"{r['driver']} × {r['response'].split()[0]} ({r['spearman_detrended']:+.2f}, quadratic "
                        f"{r['spearman_quad_detrended']:+.2f})" for _, r in one.iterrows()) or "none")
           + f"; consistent in both sources and both detrends: {', '.join(both) or 'none'}; cointegration: {co_txt}; "
           f"CCF {len(sig_ccf)} of {len(ccf)} lag tests significant across all (min Holm p {fmt_p(ccf['p_holm_all'].min())})",
           reject,
           f"{len(flips)} of {len(one)} linear-detrended links either lose significance under a quadratic detrend, have "
           "no significant counterpart in the other source, or flip sign between FAO and SAU (dams): they reflect "
           "curvature, not co-movement. Shipping starts in 2000 (≤ 22 overlapping years), so its "
           "power is low; GDAT completions are a lower bound that thins after ~2005 (database lag).",
           verdict=None if reject else "Fail to reject H0 (shared trend only; no consistent link beyond it)")

    # Figure: (1) drivers indexed, (2) levels vs detrended ρ dumbbell, (3) min Holm p per pair
    fig = plt.figure(figsize=(16, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 0.8], wspace=0.42)
    a1, a2, a3 = (fig.add_subplot(gs[i]) for i in range(3))
    for i, (dn, x, log) in enumerate(drivers):
        s = x.loc[1990:]
        base = s.loc[2000] if 2000 in s.index else s.iloc[0]
        a1.plot(s.index, s / base * 100, color=v.SERIES[i], lw=1.8, label=dn)
    a1.set_yscale("log")
    a1.axhline(100, color=v.AXIS, lw=0.8)
    a1.set_ylabel("Index, 2000 = 100 (log scale)")
    a1.legend(loc="upper left", fontsize=8.5)
    a1.set_title("1 · World drivers all rise, like % overexploited", fontsize=9.5)
    labels = [f"{dn}\n× {rn.split()[0]}" for dn, _, _ in drivers for rn, _ in responses]
    ypos = np.arange(len(trend))[::-1]
    for yp, (_, r) in zip(ypos, trend.iterrows()):
        a2.plot([r["spearman_levels"], r["spearman_detrended"]], [yp, yp], color=v.GRID, lw=2.5, zorder=1)
        a2.scatter(r["spearman_levels"], yp, s=46, facecolor=v.SURFACE, edgecolor=v.INK_2, lw=1.4, zorder=3)
        a2.scatter(r["spearman_detrended"], yp, s=52, zorder=4,
                   color=v.SERIES[0] if r["p_detrended_holm"] < ALPHA else v.NEUTRAL, edgecolor=v.SURFACE, lw=1)
        a2.scatter(r["spearman_quad_detrended"], yp, s=46, marker="D", zorder=4,
                   color=v.SERIES[1] if r["p_quad_holm"] < ALPHA else v.NEUTRAL, edgecolor=v.SURFACE, lw=1)
    a2.axvline(0, color=v.AXIS, lw=0.8)
    a2.set_yticks(ypos, labels, fontsize=7.8)
    a2.set_xlim(-1.05, 1.05)
    a2.set_xlabel("Spearman ρ: ○ levels, ● linear detrend, ◆ quadratic detrend\n(coloured = Holm p < 0.05, grey = not)")
    a2.grid(axis="y", visible=False)
    a2.set_title("2 · Beyond the trend, signs flip with the detrend or the source", fontsize=9.5)
    mins = ccf.groupby(["driver", "response"], sort=False)["p_holm"].min().reset_index()
    a3.barh(np.arange(len(mins))[::-1], -np.log10(mins["p_holm"]),
            color=[v.SERIES[0] if p < ALPHA else v.NEUTRAL for p in mins["p_holm"]], height=0.6)
    a3.axvline(-np.log10(ALPHA), color=v.INK_2, lw=1, ls=(0, (3, 3)))
    a3.set_yticks(np.arange(len(mins))[::-1], [""] * len(mins))
    a3.set_xlabel("−log10(min Holm p) across lags")
    a3.grid(axis="y", visible=False)
    a3.text(-np.log10(ALPHA), len(mins) - 0.4, " p = 0.05", color=v.INK_2, fontsize=8, va="bottom")
    a3.set_title("3 · Year-to-year changes (same rows; Holm within pair)", fontsize=9.5)
    v.suptitle(fig, "Iteration 4 · H4a — Shipping, dams and aquaculture rise with overexploitation, "
               "but no link survives beyond the shared trend" if not reject else
               "Iteration 4 · H4a — A human-use driver tracks overexploitation beyond the shared trend",
               f"Link consistent across both sources and both detrends: {', '.join(both) or 'none'}; CCF tests "
               f"significant across all {len(ccf)}: {len(sig_ccf)}. Shipping series cover 2000–2024 only")
    fig.subplots_adjust(top=0.8, bottom=0.17, left=0.05, right=0.98)
    save_fig(fig, sec, "I4.01_h4a_global_drivers",
             "World Bank WDI (UNCTAD port data); UNCTADstat; GDAT v1 (Zenodo); FAO via OWID; Sea Around Us")


def section_i4_h4b() -> None:
    sec = "9_iteration4_human_use"
    print("Iteration 4: H4b — SEA shipping and structures vs % overexploited and MTL")
    sea, cs = sea_series(), country_series()
    tabs, lv_rows = [], []
    for unit in ["SEA"] + SEA5:
        src = sea if unit == "SEA" else cs[unit]
        for dn, x, log in i4_drivers(unit):
            dx = i4_change(x[x > 0] if log else x, log)
            for resp, lbl in [("over", "Δ % overexploited"), ("mtl", "Δ MTL")]:
                dy = src[resp].diff().dropna()
                n = len(pd.concat([dx, dy], axis=1).dropna())
                if n < 12:
                    continue
                t = ccf_table(dx, dy, f"{unit}: {dn} -> {lbl}", i4_max_lag(n))
                t.insert(0, "unit", unit)
                t.insert(1, "driver", dn)
                t.insert(2, "response", lbl)
                tabs.append(t)
            if unit == "SEA":
                xl = np.log(x) if log else x
                r_lv, p_lv, r_dt, p_dt, n = detrended_corr(xl, src["over"])
                r_q, p_q = quad_detrended_corr(xl, src["over"])
                lv_rows.append({"driver": dn, "response": "SEA % overexploited", "n_years": n,
                                "spearman_levels": r_lv, "p_levels": p_lv, "spearman_detrended": r_dt,
                                "p_detrended": p_dt, "spearman_quad_detrended": r_q, "p_quad_detrended": p_q})
    ccf = pd.concat(tabs, ignore_index=True)
    ccf["p_holm_all"] = holm(ccf["p"])
    save_table(ccf, "9_I4_H4b_ccf_sea")
    lv = pd.DataFrame(lv_rows)
    lv["p_detrended_holm"] = holm(lv["p_detrended"])
    lv["p_quad_holm"] = holm(lv["p_quad_detrended"])
    lv["consistent_link"] = i4_consistent(lv)
    save_table(lv, "9_I4_H4b_levels_vs_detrended")

    sig = ccf[ccf["significant_holm"]]
    sig_all = ccf[ccf["p_holm_all"] < ALPHA]
    n_pairs = ccf.groupby(["unit", "driver", "response"]).ngroups
    decide("H4b (iteration 4: shipping & structures, SEA)",
           "Do changes in SEA port traffic, seaborne trade, liner connectivity, dam building or aquaculture "
           "cross-correlate with % overexploited or mean trophic level?",
           f"{len(sig)} of {len(ccf)} lag tests ({n_pairs} unit × driver × response rows) significant after Holm "
           "within row"
           + (": " + "; ".join(f"{r['unit']} {r['driver']} -> {r['response']} lag {int(r['lag'])} r = {r['r']:+.2f}"
                               for _, r in sig.iterrows()) if len(sig) else "")
           + f"; across all tests {len(sig_all)} (min Holm p {fmt_p(ccf['p_holm_all'].min())}); SEA linear-detrended ρ "
           "with % overexploited significant after Holm for "
           + (", ".join(f"{r['driver']} ({r['spearman_detrended']:+.2f}, quadratic {r['spearman_quad_detrended']:+.2f})"
                        for _, r in lv[lv["p_detrended_holm"] < ALPHA].iterrows()) or "none")
           + f"; consistent under both detrends: {', '.join(lv[lv['consistent_link']]['driver']) or 'none'}",
           len(sig_all) > 0 or bool(lv["consistent_link"].any()),
           "Shipping overlaps the SAU series for 19 changes only (2001–2019), so only strong links could be detected. "
           "Linear-detrended links that flip sign under a quadratic detrend are curvature artefacts.",
           verdict=None if (len(sig_all) > 0 or lv["consistent_link"].any()) else
           ("Fail to reject H0" if len(sig) == 0 else
            f"Weak: {len(sig)} isolated lag(s) significant within row; none across all {len(ccf)} tests"))

    # Figure: heatmap, rows = driver × response, columns = units; cell = r at the lag with the smallest p
    units = ["SEA"] + SEA5
    combos = [(d, r) for d in ccf["driver"].unique() for r in ["Δ % overexploited", "Δ MTL"]]
    mat = np.full((len(combos), len(units)), np.nan)
    lag = np.full_like(mat, np.nan)
    sigm = np.zeros_like(mat, bool)
    for i, (d, r) in enumerate(combos):
        for j, u in enumerate(units):
            t = ccf[(ccf["unit"] == u) & (ccf["driver"] == d) & (ccf["response"] == r)]
            if len(t):
                b = t.loc[t["p"].idxmin()]
                mat[i, j], lag[i, j], sigm[i, j] = b["r"], b["lag"], bool(t["significant_holm"].any())
    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=v.DIVERGING, vmin=-0.7, vmax=0.7, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5, color=v.MUTED)
                continue
            ax.text(j, i, f"{mat[i, j]:+.2f}\nlag {int(lag[i, j])}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if abs(mat[i, j]) > 0.45 else v.INK, fontweight="bold" if sigm[i, j] else "normal")
            if sigm[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec=v.INK, lw=1.6))
    for b in range(2, len(combos), 2):
        ax.axhline(b - 0.5, color=v.SURFACE, lw=3)
    ax.set_xticks(range(len(units)), units)
    ax.set_yticks(range(len(combos)), [f"{d} · {r}" for d, r in combos], fontsize=8.5)
    ax.grid(False)
    for spn in ax.spines.values():
        spn.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("r at the lag with the smallest p", color=v.INK_2)
    cb.outline.set_visible(False)
    v.suptitle(fig, "Iteration 4 · H4b — Southeast Asian shipping and structures do not track the fisheries "
               "indicators year to year" if len(sig_all) == 0 else
               "Iteration 4 · H4b — Some SEA human-use drivers track the fisheries indicators",
               f"{len(sig)} of {len(ccf)} lag tests significant within row (outlined), {len(sig_all)} across all "
               "tests. Liner connectivity has no world value; 'n/a' = fewer than 12 overlapping years")
    fig.subplots_adjust(top=0.86, bottom=0.06, left=0.3, right=0.97)
    save_fig(fig, sec, "I4.02_h4b_sea_ccf_heatmap",
             "World Bank WDI; UNCTADstat; GDAT v1; FAO via OWID; Sea Around Us")


def did_estimate(p: pd.DataFrame, y: str, treated: list[int], event: int) -> tuple[float, float, float]:
    """
    Two-way fixed-effects difference-in-differences: y_it = EEZ_i + year_t + β·(treated_i × post_t).
    Returns β, its EEZ-clustered SE and p. With a balanced panel β equals the classic 2×2 DiD of means.
    """
    d = p.assign(D=(p["eez"].isin(treated) & (p["year"] >= event)).astype(float))
    X = pd.get_dummies(d[["eez", "year"]].astype(str), drop_first=True).astype(float)
    X.insert(0, "D", d["D"])
    X = sm.add_constant(X)
    fit = sm.OLS(d[y].to_numpy(), X.to_numpy()).fit(cov_type="cluster", cov_kwds={"groups": d["eez"].to_numpy()})
    return float(fit.params[1]), float(fit.bse[1]), float(fit.pvalues[1])


def randomization_p(p: pd.DataFrame, y: str, treated: list[int], event: int) -> tuple[float, np.ndarray]:
    """
    Exact randomization inference: re-estimate β for every way of choosing len(treated) EEZs
    as 'treated'; p = share of assignments with |β| ≥ |β_observed|. Valid with few clusters,
    where clustered standard errors are unreliable.
    """
    from itertools import combinations
    eezs = sorted(p["eez"].unique())
    obs = did_estimate(p, y, treated, event)[0]
    draws = np.array([did_estimate(p, y, list(c), event)[0] for c in combinations(eezs, len(treated))])
    return float(np.mean(np.abs(draws) >= abs(obs) - 1e-12)), draws


def section_i4_h4c() -> None:
    sec = "9_iteration4_human_use"
    print("Iteration 4: H4c — South China Sea island building (difference-in-differences)")
    panel = st.eez_panel()
    panel = panel.assign(log_catch=np.log(panel["catch"]))
    specs = [("main", SCS_EEZ, OTHER_EEZ, 2004, 2019, EVENT_YEAR),
             ("SCS edge EEZs counted as treated", SCS_EEZ + EDGE_EEZ, OTHER_EEZ, 2004, 2019, EVENT_YEAR),
             ("placebo: fake event 2009, pre-period only", SCS_EEZ, OTHER_EEZ, 2004, 2013, 2009)]
    rows, draws_main = [], {}
    for name, tr, ctl, lo, hi, ev in specs:
        p = panel[panel["eez"].isin(tr + ctl) & panel["year"].between(lo, hi)]
        for y, lbl in [("over", "% overexploited"), ("mtl", "mean trophic level"), ("log_catch", "log catch")]:
            b, se, pc = did_estimate(p, y, tr, ev)
            pr, draws = randomization_p(p, y, tr, ev)
            if name == "main":
                draws_main[y] = (b, draws)
            rows.append({"spec": name, "response": lbl, "years": f"{lo}–{hi}", "event_year": ev,
                         "n_treated": len(tr), "n_control": len(ctl), "did_beta": b, "cluster_se": se,
                         "p_cluster": pc, "p_randomization": pr, "n_assignments": len(draws)})
    did = pd.DataFrame(rows)
    did["p_rand_holm"] = np.nan
    for s in did["spec"].unique():
        m = did["spec"] == s
        did.loc[m, "p_rand_holm"] = holm(did.loc[m, "p_randomization"])
    save_table(did, "9_I4_H4c_did")

    # Event study for the figure: coefficient per year relative to 2013 (the last pre-event year)
    p = panel[panel["eez"].isin(SCS_EEZ + OTHER_EEZ) & panel["year"].between(2004, 2019)].copy()
    rel_years = [r for r in range(2004 - EVENT_YEAR, 2019 - EVENT_YEAR + 1) if r != -1]
    X = pd.get_dummies(p[["eez", "year"]].astype(str), drop_first=True).astype(float)
    for r in rel_years:
        X.insert(0, f"rel{r}", ((p["year"] - EVENT_YEAR == r) & p["eez"].isin(SCS_EEZ)).astype(float).to_numpy())
    X = sm.add_constant(X)
    fit = sm.OLS(p["over"].to_numpy(), X.to_numpy()).fit(cov_type="cluster", cov_kwds={"groups": p["eez"].to_numpy()})
    names = list(X.columns)
    es = pd.DataFrame([{"rel_year": r, "year": EVENT_YEAR + r, "coef": fit.params[names.index(f"rel{r}")],
                        "se": fit.bse[names.index(f"rel{r}")]} for r in rel_years])
    es = pd.concat([es, pd.DataFrame([{"rel_year": -1, "year": EVENT_YEAR - 1, "coef": 0.0, "se": 0.0}])]
                   ).sort_values("rel_year")
    save_table(es, "9_I4_H4c_event_study")

    main = did[did["spec"] == "main"]
    m_over = main[main["response"] == "% overexploited"].iloc[0]
    plac = did[(did["spec"].str.startswith("placebo")) & (did["response"] == "% overexploited")].iloc[0]
    edge = did[(did["spec"].str.startswith("SCS edge")) & (did["response"] == "% overexploited")].iloc[0]
    reject = bool((main["p_rand_holm"] < ALPHA).any()) and plac["p_randomization"] >= ALPHA
    decide("H4c (iteration 4: artificial islands, SEA)",
           "Did % overexploited, mean trophic level or catch in the EEZs facing the Spratly Islands change "
           "after China's 2013–2016 island building, relative to other SEA EEZs?",
           "DiD 2004–2019, event 2014, 4 treated vs 5 control EEZs: "
           + "; ".join(f"{r['response']} β = {r['did_beta']:+.3f} (randomization p {fmt_p(r['p_randomization'])}, "
                       f"Holm {fmt_p(r['p_rand_holm'])})" for _, r in main.iterrows())
           + f"; edge EEZs as treated: % overexploited β = {edge['did_beta']:+.2f} (p {fmt_p(edge['p_randomization'])})"
           f"; placebo 2009: β = {plac['did_beta']:+.2f} (p {fmt_p(plac['p_randomization'])})",
           reject,
           "Island building is a single dated event; military outposts were built on the same features in "
           "2014–2018, so the two cannot be separated. Vietnam's own reclamation (2022–2025) is after the "
           "fisheries data end (2019). Only 9 EEZs → randomization inference (126 assignments) instead of "
           "clustered errors.",
           verdict=None if reject else "Fail to reject H0 (no post-2014 change specific to Spratly-facing EEZs)")

    # Figure: (1) group means, (2) event study, (3) randomization distribution
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.1, 1.1, 0.9]})
    pp = panel[panel["year"].between(1990, 2019)]
    for grp, eezs, col in [("Spratly-facing EEZs (VN, PH, Sabah, Sarawak)", SCS_EEZ, v.SERIES[0]),
                           ("Other SEA EEZs", OTHER_EEZ, v.NEUTRAL)]:
        s = pp[pp["eez"].isin(eezs)].groupby("year")["over"].mean()
        a1.plot(s.index, s.values, color=col, lw=2.2, label=grp)
    a1.axvspan(2013.9, 2016, color=v.GRID, alpha=0.7, lw=0)
    a1.text(2014.1, a1.get_ylim()[1], "China's Spratly\nreclamation", fontsize=8, color=v.INK_2, va="top")
    a1.set_ylabel("% of stocks overexploited (mean of EEZs)")
    a1.legend(loc="upper left", fontsize=8.5)
    a1.set_title("1 · The gap moves before 2014 too; no break at the event", fontsize=9.5)
    a2.errorbar(es["year"], es["coef"], yerr=1.96 * es["se"], fmt="o", color=v.SERIES[0], ecolor=v.NEUTRAL,
                elinewidth=1.5, capsize=0, markersize=5)
    a2.axhline(0, color=v.AXIS, lw=0.8)
    a2.axvline(EVENT_YEAR - 0.5, color=v.INK_2, lw=1, ls=(0, (3, 3)))
    a2.set_ylabel("Spratly-facing minus other, vs 2013 (pp)")
    a2.set_title("2 · Event study (95% CI, EEZ-clustered)", fontsize=9.5)
    b_obs, draws = draws_main["over"]
    a3.hist(draws, bins=20, color=v.NEUTRAL, edgecolor=v.SURFACE)
    a3.axvline(b_obs, color=v.SERIES[0], lw=2.2)
    a3.text(0.02, 0.97, f"observed β = {b_obs:+.2f} (blue line)\nrandomization p = {m_over['p_randomization']:.2f}",
            transform=a3.transAxes, color=v.INK, fontsize=8.5, va="top")
    a3.set_xlabel("DiD β for every 4-of-9 EEZ assignment (pp)")
    a3.set_ylabel("Assignments")
    a3.grid(axis="x", visible=False)
    a3.set_title("3 · Randomization inference", fontsize=9.5)
    v.suptitle(fig, "Iteration 4 · H4c — Island building in the Spratlys left no measurable mark on "
               "overexploitation in neighbouring EEZs" if not reject else
               "Iteration 4 · H4c — Spratly-facing EEZs changed after the island building",
               f"DiD β = {m_over['did_beta']:+.2f} pp (randomization p {fmt_p(m_over['p_randomization'])}); "
               f"placebo 2009 β = {plac['did_beta']:+.2f} pp. Data end in 2019, before Vietnam's 2022–25 reclamation")
    v.layout(fig, 0)
    save_fig(fig, sec, "I4.03_h4c_island_building_did", "Sea Around Us; event dates: CSIS AMTI, US DoD (2016)")


def section_i4_descriptive() -> None:
    sec = "9_iteration4_human_use"
    print("Iteration 4: offshore platforms and wind farms (descriptive)")
    w = pd.read_csv(HUMAN / "deepowt_offshore_wind.csv")
    # year-end (Q4) status; 2021 ends in Q2 in DeepOWT
    w = pd.concat([w[w["quarter"] == "Q4"], w[(w["year"] == 2021) & (w["quarter"] == "Q2")]])
    pl = pd.read_csv(HUMAN / "oogp_platforms.csv")
    sea = sea_series()["over"]
    rows = []
    for reg in ["WLD", "VNM"]:
        s = w[w["region"] == reg].set_index("year")["n_class2"]
        rows.append({"structure": "Offshore wind turbines (DeepOWT)", "region": ISO3[reg], "first": int(s.index.min()),
                     "last": int(s.index.max()), "start": s.iloc[0], "end": s.iloc[-1],
                     "years_overlapping_fisheries_data": len(set(s.index) & set(range(1950, 2020)))})
    for reg in ["ALL6", "THA", "MYS", "VNM", "IDN", "KHM"]:
        s = pl[pl["region"] == reg].set_index("year")["n_platforms_detected"]
        rows.append({"structure": "Offshore oil & gas platforms (OOGP)",
                     "region": "6 basins" if reg == "ALL6" else ISO3.get(reg, "Cambodia"),
                     "first": int(s.index.min()), "last": int(s.index.max()), "start": s.iloc[0], "end": s.iloc[-1],
                     "years_overlapping_fisheries_data": len(set(s.index) & set(range(1950, 2020)))})
    save_table(pd.DataFrame(rows), "9_I4_structures_descriptive")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.8))
    for reg, col, lbl in [("THA", v.COUNTRY_COLOR["Thailand"], "Thailand"),
                          ("MYS", v.COUNTRY_COLOR["Malaysia"], "Malaysia"),
                          ("VNM", v.COUNTRY_COLOR["Vietnam"], "Vietnam")]:
        s = pl[pl["region"] == reg].set_index("year")["n_platforms_detected"]
        a1.plot(s.index, s.values, color=col, lw=2, marker="o", markersize=4)
        v.end_label(a1, s.index[-1], s.iloc[-1], lbl)
    a1.set_ylabel("Platforms detected")
    a1.set_title("1 · Offshore oil & gas platforms, SEA (Sentinel-1)", fontsize=9.5)
    a1.set_xlim(2016.5, 2023.9)
    for reg, col, lbl in [("WLD", v.WORLD_COLOR, "World"), ("VNM", v.COUNTRY_COLOR["Vietnam"], "Vietnam")]:
        s = w[w["region"] == reg].set_index("year")["n_class2"]
        a2.plot(s.index, s.values, color=col, lw=2, marker="o", markersize=4)
        v.end_label(a2, s.index[-1], s.iloc[-1], lbl)
    a2.set_yscale("log")
    a2.set_ylabel("Offshore wind turbines (log scale)")
    a2.set_title("2 · Offshore wind turbines (DeepOWT; 2021 = Q2)", fontsize=9.5)
    a2.set_xlim(2015.6, 2021.9)
    s = sea.loc[2010:]
    a3.plot(s.index, s.values, color=v.INK_2, lw=2)
    a3.axvspan(2016.5, 2019.5, color=v.GRID, alpha=0.7, lw=0)
    a3.text(2016.6, a3.get_ylim()[0], " overlap\n with structures", fontsize=8, color=v.INK_2, va="bottom")
    a3.set_ylabel("SEA % of stocks overexploited")
    a3.set_title("3 · Fisheries series end in 2019", fontsize=9.5)
    v.suptitle(fig, "Iteration 4 · Offshore platforms and wind farms — satellite records start too late to test",
               "Platform counts are flat in SEA (Thailand ~420–450, Malaysia ~130–150); offshore wind in SEA is a "
               "single nearshore cluster in Vietnam. Overlap with the fisheries data: 3–4 years")
    v.layout(fig, 0)
    save_fig(fig, sec, "I4.04_structures_descriptive", "Si et al. 2026 (OOGP, Zenodo); Hoeser et al. 2022 "
             "(DeepOWT, Zenodo); Sea Around Us")


# =========================================================================== #
# ITERATION 5 — pooled panel test, stock rebuilding, Fishing-in-Balance (H5)
# =========================================================================== #
# Iterations 1–4 tested each driver on single yearly series of 20–70 points, so a
# common objection is low statistical power. Iteration 5 adds three tests:
#   H5a  pooled panel: all 11 EEZs (or 5 countries for effort) in one regression with
#        EEZ and year fixed effects. Year effects absorb the common upward trend, so the
#        driver's effect is estimated from differences BETWEEN EEZs in how fast pressure
#        and overexploitation moved — exactly the variation the single-series tests lacked.
#        Few clusters (11 / 5) → p-values from a wild cluster bootstrap (Webb weights).
#   H5b  SDG 14.4 also asks to RESTORE stocks: did the share of rebuilding stocks rise
#        after 2015? (EEZ-level paired test + segmented trend for SEA and the world)
#   H5c  Fishing-in-Balance index (Pauly et al. 2000): does catch still compensate for
#        the change in trophic level? Explains why SEA mean trophic level rises while
#        stocks worsen.
# Iterations 1–4 above are unchanged.

PROC = ROOT / "data" / "processed"
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])  # Webb 6-point weights
N_BOOT = 9999
TE = 0.1   # trophic transfer efficiency used in the FiB index (Pauly et al. 2000)


def eez_full_panel() -> pd.DataFrame:
    """Cleaned EEZ-year table (3-data-quality/clean_data.py output): adds industrial and rebuilding shares."""
    p = pd.read_csv(PROC / "sea_eez_annual.csv")
    return p.rename(columns={"eez_id": "unit", "pct_overexploited": "over", "mean_trophic_level": "mtl",
                             "catch_t": "catch"})


def effort_panel() -> pd.DataFrame:
    """
    Country-year panel 1950–2017: % overexploited, MTL and break-adjusted log effort.
    Effort has single-year source breaks (Thailand 2002/2011, Malaysia 1965); in a LEVEL
    regression a break would act as a permanent jump, so log effort is rebuilt by
    cumulating the growth series with the break years set to zero growth.
    """
    se, cs = sea_effort(), country_series()
    rows = []
    for c in SEA5:
        e = se[c]["nom_kwdays"]
        d = np.log(e).diff()
        d[d.abs() > BREAK_DLOG] = 0.0
        loge = np.log(e.iloc[0]) + d.fillna(0).cumsum()
        df = cs[c].loc[:2017].join(loge.rename("log_effort"), how="inner")
        rows.append(df.assign(unit=c, country=c).reset_index())
    return pd.concat(rows, ignore_index=True)


def fwl_residuals(df: pd.DataFrame, cols: list[str], unit_trends: bool) -> pd.DataFrame:
    """
    Residualise `cols` on unit and year dummies (two-way fixed effects), plus a linear trend
    per unit if `unit_trends` (Frisch-Waugh-Lovell). Regressing residual y on residual x
    gives exactly the fixed-effects β.
    """
    W = pd.get_dummies(df[["unit", "year"]].astype(str), drop_first=True).astype(float)
    W.insert(0, "const", 1.0)
    if unit_trends:
        t = df["year"].to_numpy(float) - df["year"].mean()
        for u in df["unit"].unique()[1:]:
            W[f"trend_{u}"] = t * (df["unit"] == u).to_numpy(float)
        W["trend"] = t
    Wm = W.to_numpy()
    coef, *_ = np.linalg.lstsq(Wm, df[cols].to_numpy(float), rcond=None)
    return pd.DataFrame(df[cols].to_numpy(float) - Wm @ coef, columns=cols, index=df.index)


def wild_cluster_p(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 4033) -> tuple[float, float, float]:
    """
    β, CR1 cluster-robust SE and the wild cluster bootstrap-t p-value (restricted: H0 β = 0
    imposed, Webb 6-point weights; all 6^G draws enumerated when G ≤ 5, else N_BOOT draws).
    With few clusters, plain clustered p-values reject far too often; this bootstrap does not.
    """
    ug = np.unique(groups)
    G, n = len(ug), len(y)
    idx = [np.where(groups == g)[0] for g in ug]

    def t_stat(yy: np.ndarray) -> tuple[float, float, float]:
        b = (x @ yy) / (x @ x)
        e = yy - b * x
        meat = sum((x[i] @ e[i]) ** 2 for i in idx)
        se = np.sqrt(G / (G - 1) * (n - 1) / (n - 1 - 1) * meat) / (x @ x)
        return b, se, b / se

    b, se, t0 = t_stat(y)
    if G <= 5:
        from itertools import product
        draws = np.array(list(product(WEBB, repeat=G)))
    else:
        draws = np.random.default_rng(seed).choice(WEBB, size=(N_BOOT, G))
    ts = np.empty(len(draws))
    for k, w in enumerate(draws):
        yy = np.empty(n)
        for g, i in enumerate(idx):
            yy[i] = y[i] * w[g]          # null model residual (y already purged of FE) × cluster weight
        ts[k] = t_stat(yy)[2]
    return float(b), float(se), float(np.mean(np.abs(ts) >= abs(t0)))


def section_i5_h5a() -> None:
    sec = "11_iteration5"
    print("Iteration 5: H5a — pooled panel, two-way fixed effects")
    eez = eez_full_panel().assign(log_catch=lambda d: np.log(d["catch"]))
    eff = effort_panel()
    # (driver column, label, panel, responses, expected sign on % overexploited)
    specs = [("log_catch", "log catch", eez, ["mtl"]),            # catch -> stock status is circular in SAU
             ("industrial_share_pct", "industrial share of catch (%)", eez, ["over", "mtl"]),
             ("unreported_share_pct", "unreported share of catch (%)", eez, ["over", "mtl"]),
             ("foreign_share_pct", "foreign-fleet share of catch (%)", eez, ["over", "mtl"]),
             ("log_effort", "log fishing effort", eff, ["over", "mtl"])]
    rows = []
    for col, lbl, panel, resps in specs:
        p = panel.sort_values(["unit", "year"]).copy()
        # cumulative pressure: mean of the previous 5 years (t−5 … t−1)
        p[f"{col}_ma5"] = p.groupby("unit")[col].transform(lambda s: s.shift(1).rolling(5).mean())
        for lag_lbl, xcol in [("same year", col), ("mean of previous 5 years", f"{col}_ma5")]:
            for resp in resps:
                d = p.dropna(subset=[xcol, resp]).reset_index(drop=True)
                for trends in (False, True):
                    r = fwl_residuals(d, [xcol, resp], trends)
                    b, se, p_boot = wild_cluster_p(r[xcol].to_numpy(), r[resp].to_numpy(), d["unit"].to_numpy())
                    sd_x, sd_y = r[xcol].std(), r[resp].std()
                    rows.append({"driver": lbl, "response": "% overexploited" if resp == "over" else "mean trophic level",
                                 "timing": lag_lbl, "spec": "EEZ/country + year FE" + (" + unit trends" if trends else ""),
                                 "units": d["unit"].nunique(), "years": f"{d['year'].min()}–{d['year'].max()}",
                                 "n_obs": len(d), "beta": b, "cluster_se": se, "p_wild_bootstrap": p_boot,
                                 "beta_standardised": b * sd_x / sd_y,
                                 "worse_direction": (b > 0) if resp == "over" else (b < 0)})
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res["p_wild_bootstrap"])
    res["significant_holm"] = res["p_holm"] < ALPHA
    save_table(res, "11_I5_H5a_panel_fixed_effects")

    sig = res[res["significant_holm"]]
    robust = []
    for (dn, rn, tm), g in sig.groupby(["driver", "response", "timing"]):
        if g["spec"].nunique() == 2 and g["worse_direction"].all():
            robust.append(f"{dn} → {rn} ({tm})")
    reject = bool(robust)
    decide("H5a (iteration 5: pooled panel, SEA)",
           "Pooling all EEZs (fixed effects absorb the common trend): does a pressure measure raise % overexploited "
           "or lower mean trophic level?",
           f"{len(sig)} of {len(res)} coefficients significant after Holm (wild cluster bootstrap)"
           + (": " + "; ".join(f"{r['driver']} → {r['response']}, {r['timing']}, {r['spec']}: β = {r['beta']:+.3g} "
                                f"(Holm p {fmt_p(r['p_holm'])})" for _, r in sig.iterrows()) if len(sig) else "")
           + f"; robust to unit trends and in the 'worse' direction: {', '.join(robust) or 'none'}; smallest raw p: "
           + (lambda r: f"{r['driver']} → {r['response']} ({r['timing']}, {r['spec']}), standardised β = "
              f"{r['beta_standardised']:+.2f}, raw p {fmt_p(r['p_wild_bootstrap'])}, Holm p {fmt_p(r['p_holm'])}")(
               res.loc[res["p_wild_bootstrap"].idxmin()]),
           reject,
           "Catch is tested only against mean trophic level (SAU stock status is derived from catch). Effort uses the "
           "5-country panel (1950–2017; break-adjusted), the rest the 11-EEZ panel (1950–2019). Few clusters → "
           "wild cluster bootstrap p-values (Webb weights).",
           verdict=None if reject or len(sig) == 0 else
           f"Weak: {len(sig)} coefficient(s) significant, but not robust to unit trends or not in the 'worse' direction")

    # Figure: standardised β (within-SD units), one row per driver × response, markers per timing/spec
    combos = list(dict.fromkeys(zip(res["driver"], res["response"])))
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    offs = {("same year", False): 0.27, ("same year", True): 0.09,
            ("mean of previous 5 years", False): -0.09, ("mean of previous 5 years", True): -0.27}
    marks = {False: "o", True: "D"}
    cols = {"same year": v.SERIES[0], "mean of previous 5 years": v.SERIES[1]}
    for i, (dn, rn) in enumerate(combos):
        y0 = len(combos) - 1 - i
        for _, r in res[(res["driver"] == dn) & (res["response"] == rn)].iterrows():
            tr = "unit trends" in r["spec"]
            yp = y0 + offs[(r["timing"], tr)]
            ci = 1.96 * r["cluster_se"] * r["beta_standardised"] / r["beta"] if r["beta"] else 0
            ax.plot([r["beta_standardised"] - abs(ci), r["beta_standardised"] + abs(ci)], [yp, yp],
                    color=v.GRID, lw=2, zorder=1)
            ax.scatter(r["beta_standardised"], yp, marker=marks[tr], s=48, zorder=3, lw=1.4,
                       edgecolor=cols[r["timing"]],
                       color=cols[r["timing"]] if r["significant_holm"] else v.SURFACE)
    for b in range(1, len(combos)):
        ax.axhline(b - 0.5, color=v.GRID, lw=0.8)
    ax.axvline(0, color=v.AXIS, lw=0.8)
    ax.set_yticks(range(len(combos))[::-1], [f"{d} → {r}" for d, r in combos], fontsize=8.8)
    ax.set_xlabel("Standardised effect (within-panel SD of response per SD of driver), 95% clustered CI")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=v.SERIES[0], label="same year, EEZ + year FE"),
               plt.Line2D([], [], marker="D", ls="", color=v.SERIES[0], label="same year, + unit trends"),
               plt.Line2D([], [], marker="o", ls="", color=v.SERIES[1], label="previous 5-year mean, EEZ + year FE"),
               plt.Line2D([], [], marker="D", ls="", color=v.SERIES[1], label="previous 5-year mean, + unit trends")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=4, fontsize=8.3,
              title="Filled = significant after Holm (wild cluster bootstrap); hollow = not", title_fontsize=8.3)
    v.suptitle(fig, "Iteration 5 · H5a — Pooling all EEZs " + (
        f"finds a robust pressure effect: {', '.join(robust)}" if reject else
        "still finds no robust pressure effect on overexploitation"),
               f"Two-way fixed effects (EEZ + year): the common trend is removed, the effect is estimated from "
               f"differences between EEZs. {len(sig)} of {len(res)} coefficients significant after Holm")
    fig.subplots_adjust(top=0.86, bottom=0.2, left=0.27, right=0.98)
    save_fig(fig, sec, "I5.01_h5a_panel_fixed_effects", "Sea Around Us (11 SEA EEZs); IMAS Global Fishing Effort")


def section_i5_h5b() -> None:
    sec = "11_iteration5"
    print("Iteration 5: H5b — stock rebuilding before vs after 2015")
    p = eez_full_panel()
    g = pd.read_csv(VAL / "sau_regions" / "stock_status" / "global.csv")
    world = g[g["Data_Set"] == "css"].set_index("Year")["Rebuilding"].astype(float)
    country = p.groupby(["country", "year"])["pct_rebuilding"].mean().unstack(0)
    sea = country[SEA5].mean(axis=1)

    # (1) EEZ-level paired test: 2015–19 mean vs 2010–14 mean, one-sided (did rebuilding increase?)
    pre = p[p["year"].between(2010, 2014)].groupby("unit")["pct_rebuilding"].mean()
    post = p[p["year"].between(2015, 2019)].groupby("unit")["pct_rebuilding"].mean()
    diff = (post - pre)
    w = stats.wilcoxon(post, pre, alternative="greater", zero_method="zsplit")
    pair = pd.DataFrame({"eez": [st.eez_name(u) for u in pre.index], "rebuilding_2010_14": pre.values,
                         "rebuilding_2015_19": post.values, "change_pp": diff.values})
    save_table(pair, "11_I5_H5b_rebuilding_by_eez")

    # (2) Trend + segmented slope change at 2015 (HAC errors) for SEA and the world, 1990–2019
    rows = []
    for name, s in [("World (SAU)", world.loc[1990:2019]), ("SEA, 5 countries (SAU)", sea.loc[1990:2019])] + \
                   [(c, country[c].loc[1990:2019]) for c in SEA5]:
        t = s.index.to_numpy(float)
        X = np.column_stack([np.ones_like(t), t - t.min(), np.clip(t - BREAK, 0, None)])
        _, hac = ols_hac(s.values, X, f"H5b rebuilding segmented: {name}")
        mk_ = mann_kendall(s)
        rows.append({"unit": name, "mean_2010_14": s.loc[2010:2014].mean(), "mean_2015_19": s.loc[2015:2019].mean(),
                     "slope_before_2015_pp_yr": hac.params[1], "slope_change_after_2015": hac.params[2],
                     "p_slope_change": hac.pvalues[2], "mk_p_1990_2019": mk_["p"], "mk_trend": mk_["trend"],
                     "sen_pp_per_decade": mk_["sen_slope"] * 10})
    seg = pd.DataFrame(rows)
    m = seg["unit"].isin(SEA5)
    seg["p_slope_change_holm_countries"] = np.nan
    seg.loc[m, "p_slope_change_holm_countries"] = holm(seg.loc[m, "p_slope_change"])
    save_table(seg, "11_I5_H5b_rebuilding_trends")
    w_row, s_row = seg.iloc[0], seg.iloc[1]
    up = int((diff > 0).sum())
    reject = bool(w.pvalue < ALPHA and s_row["slope_change_after_2015"] > 0 and s_row["p_slope_change"] < ALPHA)
    decide("H5b (iteration 5: stock rebuilding, SDG 14.4)",
           "Has the share of rebuilding stocks increased since the SDGs were adopted (2015)?",
           f"EEZ-level 2015–19 vs 2010–14: {up} of {len(diff)} EEZs higher, median change {diff.median():+.2f} pp, "
           f"one-sided Wilcoxon p {fmt_p(w.pvalue)}; SEA slope change after 2015 {s_row['slope_change_after_2015']:+.2f} "
           f"pp/yr (p {fmt_p(s_row['p_slope_change'])}); world {w_row['slope_change_after_2015']:+.2f} pp/yr "
           f"(p {fmt_p(w_row['p_slope_change'])}); SEA level {s_row['mean_2010_14']:.1f}% → {s_row['mean_2015_19']:.1f}%",
           reject,
           "Sea Around Us labels a stock 'rebuilding' when its catch recovers after a collapse (catch-based, Kleisner "
           "et al. 2013). The 'restore stocks' half of SDG 14.4 is judged on this measure only. Country slope changes "
           "after 2015 (Holm): " + "; ".join(f"{r['unit']} {r['slope_change_after_2015']:+.2f} pp/yr "
                                              f"(p {fmt_p(r['p_slope_change_holm_countries'])})"
                                              for _, r in seg[m].iterrows()),
           verdict=None if reject else (
               "Mixed: world rebuilding accelerated after 2015, SEA's did not" if (
                   w_row["slope_change_after_2015"] > 0 and w_row["p_slope_change"] < ALPHA) else None))

    # Figure: (1) rebuilding share over time, (2) per-EEZ before/after dumbbell
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.15, 1]})
    for lbl, s, col, lw in [("World (SAU)", world, v.INK_2, 2.2), ("SEA, 5 countries", sea, v.SERIES[0], 2.4)]:
        s = s.loc[1990:2019]
        a1.plot(s.index, s.values, color=col, lw=lw, label=lbl)
    a1.axvline(BREAK, color=v.AXIS, lw=1)
    a1.text(BREAK - 0.3, a1.get_ylim()[0] + 0.3, "SDGs adopted", fontsize=8.5, color=v.MUTED, ha="right", va="bottom")
    a1.set_ylabel("% of stocks rebuilding (SAU)")
    a1.legend(loc="upper left", fontsize=8.5)
    a1.set_title(f"1 · Share of rebuilding stocks, 1990–2019 — SEA slope change after 2015 "
                 f"{s_row['slope_change_after_2015']:+.2f} pp/yr (p {fmt_p(s_row['p_slope_change'])})", fontsize=9.5)
    order = pair.sort_values("change_pp").reset_index(drop=True)
    for i, r in order.iterrows():
        col = v.SERIES[5] if r["change_pp"] > 0 else v.SERIES[7]
        a2.plot([r["rebuilding_2010_14"], r["rebuilding_2015_19"]], [i, i], color=v.GRID, lw=2.5, zorder=1)
        a2.scatter(r["rebuilding_2010_14"], i, s=44, facecolor=v.SURFACE, edgecolor=v.INK_2, lw=1.4, zorder=3)
        a2.scatter(r["rebuilding_2015_19"], i, s=50, color=col, edgecolor=v.SURFACE, lw=1, zorder=4)
    a2.set_yticks(range(len(order)), order["eez"], fontsize=8.5)
    a2.set_xlabel("% of stocks rebuilding: ○ 2010–14 mean → ● 2015–19 mean (green = up, red = down)")
    a2.grid(axis="y", visible=False)
    a2.set_title(f"2 · Per EEZ: {up} of {len(diff)} up; one-sided Wilcoxon p {fmt_p(w.pvalue)}", fontsize=9.5)
    world_up = bool(w_row["slope_change_after_2015"] > 0 and w_row["p_slope_change"] < ALPHA)
    v.suptitle(fig, "Iteration 5 · H5b — " + ("Stock rebuilding has increased since 2015" if reject else
                                              "Rebuilding sped up worldwide after 2015, but not in Southeast Asia"
                                              if world_up else "No sign that stock rebuilding increased after 2015"),
               f"World slope change after 2015 {w_row['slope_change_after_2015']:+.2f} pp/yr (p "
               f"{fmt_p(w_row['p_slope_change'])}). The 'restore stocks' half of SDG 14.4, on Sea Around Us stock "
               "status (catch-based)")
    v.layout(fig, 0)
    save_fig(fig, sec, "I5.02_h5b_rebuilding_before_after_2015", "Sea Around Us (World; 11 SEA EEZs)")


def fib(catch: pd.Series, mtl: pd.Series, base: int) -> pd.Series:
    """
    Fishing-in-Balance index (Pauly, Christensen & Walters 2000):
        FiB_t = ln(Y_t · (1/TE)^TL_t) − ln(Y_0 · (1/TE)^TL_0),  TE = 0.1
    = 0 when a fall in trophic level is exactly offset by the ecologically expected rise in
    catch; > 0 = catch grows faster (often geographic expansion); < 0 = catch fails to
    compensate (fishing beyond balance / ecosystem impaired). Natural log; the base only
    rescales the index and does not change trend tests.
    """
    d = pd.concat([catch.rename("y"), mtl.rename("tl")], axis=1).dropna()
    v_ = np.log(d["y"]) + d["tl"] * np.log(1 / TE)
    return v_ - v_.loc[base]


def section_i5_h5c() -> None:
    sec = "11_iteration5"
    print("Iteration 5: H5c — Fishing-in-Balance index")
    sea, cs = sea_series(), country_series()
    p = st.eez_panel()
    units = [("World (SAU)", sau_world_catch(), sau_world_mtl()), ("SEA", sea["catch"], sea["mtl"])]
    units += [(c, cs[c]["catch"], cs[c]["mtl"]) for c in SEA5]
    eezs = sorted(p["eez"].unique(), key=lambda e: (SEA5.index(v.EEZ_COUNTRY[e]), st.eez_name(e)))
    for e in eezs:
        name = st.eez_name(e)
        g = p[p["eez"] == e].set_index("year")
        units.append((f"{name} (EEZ)" if name in SEA5 else name, g["catch"], g["mtl"]))
    rows, series = [], {}
    for name, y, tl in units:
        f = fib(y, tl, 1950)
        series[name] = f
        r = mann_kendall(f.loc[1990:2019])
        d_catch = np.log(y.loc[2015:2019].mean() / y.loc[1990:1994].mean())
        d_tl = (tl.loc[2015:2019].mean() - tl.loc[1990:1994].mean()) * np.log(1 / TE)
        level = "World" if name.startswith("World") else ("SEA" if name == "SEA" else ("country" if name in SEA5 else "EEZ"))
        rows.append({"unit": name, "level": level, "fib_2019": f.loc[2019], "mk_p_1990_2019": r["p"], "trend": r["trend"],
                     "sen_per_decade": r["sen_slope"] * 10, "sen_ci_low": r["sen_ci_low"] * 10,
                     "sen_ci_high": r["sen_ci_high"] * 10,
                     "change_1990_94_to_2015_19": d_catch + d_tl, "from_catch": d_catch, "from_trophic_level": d_tl})
    res = pd.DataFrame(rows)
    res["p_holm"] = res.groupby("level")["mk_p_1990_2019"].transform(lambda s: holm(s))
    res["significant_holm"] = res["p_holm"] < ALPHA
    save_table(res, "11_I5_H5c_fishing_in_balance")

    s_row = res[res["unit"] == "SEA"].iloc[0]
    w_row = res[res["unit"] == "World (SAU)"].iloc[0]
    falling = res[(res["level"].isin(["country", "EEZ"])) & res["significant_holm"] & (res["trend"] == "decreasing")]["unit"].tolist()
    rising = res[(res["level"].isin(["country", "EEZ"])) & res["significant_holm"] & (res["trend"] == "increasing")]["unit"].tolist()
    reject = bool(s_row["trend"] == "decreasing" and s_row["mk_p_1990_2019"] < ALPHA)
    decide("H5c (iteration 5: Fishing-in-Balance, SEA)",
           "Since 1990, has the Fishing-in-Balance index fallen (catch no longer compensating for trophic-level change)?",
           f"SEA FiB Sen {s_row['sen_per_decade']:+.2f}/decade (MK p {fmt_p(s_row['mk_p_1990_2019'])}); world "
           f"{w_row['sen_per_decade']:+.2f}/decade (p {fmt_p(w_row['mk_p_1990_2019'])}); 1990–94 → 2015–19 SEA change "
           f"{s_row['change_1990_94_to_2015_19']:+.2f} = catch {s_row['from_catch']:+.2f} + trophic level "
           f"{s_row['from_trophic_level']:+.2f}; falling after Holm in: {', '.join(falling) or 'none'}; rising in: "
           f"{', '.join(rising) or 'none'}",
           reject,
           "FiB > 0 and rising means catches grew faster than a trophic-level change alone would allow, usually through "
           "expansion (new grounds, offshore/pelagic stocks); FiB falling means the ecosystem no longer delivers. "
           "TE = 0.1 by convention.",
           verdict=None if reject else
           ("Fail to reject H0 for SEA (FiB not falling)" + (f"; falling in {', '.join(falling)}" if falling else "")))

    # Figure: (1) FiB over time (World, SEA, countries), (2) decomposition 1990–94 → 2015–19
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 5.8), gridspec_kw={"width_ratios": [1.2, 1]})
    for name in ["World (SAU)", "SEA"] + SEA5:
        f = series[name].loc[1950:2019]
        col = v.INK_2 if name.startswith("World") else (v.INK if name == "SEA" else v.COUNTRY_COLOR[name])
        a1.plot(f.index, f.values, color=col, lw=2.4 if name in ("SEA", "World (SAU)") else 1.5,
                ls=(0, (4, 2)) if name.startswith("World") else "-", label=name)
    a1.axhline(0, color=v.AXIS, lw=0.8)
    a1.axvspan(1990, 2019, color=v.GRID, alpha=0.35, lw=0)
    a1.set_ylabel("Fishing-in-Balance index (0 = 1950 balance)")
    a1.legend(loc="upper left", fontsize=8.3, ncol=2)
    a1.set_title("1 · FiB since 1950 (shaded = window tested)", fontsize=9.5)
    sub = res[res["level"].isin(["World", "SEA", "country"])].reset_index(drop=True)
    yy = np.arange(len(sub))[::-1]
    a2.barh(yy + 0.18, sub["from_catch"], height=0.34, color=v.SERIES[0], label="from catch (Δ ln catch)")
    a2.barh(yy - 0.18, sub["from_trophic_level"], height=0.34, color=v.SERIES[1],
            label="from trophic level (ΔTL × ln 10)")
    a2.scatter(sub["change_1990_94_to_2015_19"], yy, marker="|", s=260, color=v.INK, lw=2.2, zorder=4,
               label="total FiB change")
    a2.axvline(0, color=v.AXIS, lw=0.8)
    a2.set_yticks(yy, sub["unit"], fontsize=8.8)
    a2.grid(axis="y", visible=False)
    a2.set_xlabel("Change in FiB, 1990–94 → 2015–19 (5-year means)")
    a2.legend(loc="lower right", fontsize=8.3)
    a2.set_title("2 · What moved the index: catch vs trophic level", fontsize=9.5)
    v.suptitle(fig, "Iteration 5 · H5c — " + ("Southeast Asian fisheries have fallen out of balance since 1990" if reject
                                              else "Southeast Asian catches still outgrow the trophic-level change: "
                                              "expansion, not balance"),
               f"SEA FiB {s_row['sen_per_decade']:+.2f}/decade since 1990 (p {fmt_p(s_row['mk_p_1990_2019'])}); "
               f"falling significantly in: {', '.join(falling) or 'no country or EEZ'}")
    v.layout(fig, 0)
    save_fig(fig, sec, "I5.03_h5c_fishing_in_balance", "Sea Around Us (World; 11 SEA EEZs)")


FORECAST_END = 2030   # SDG deadline
FORECAST_FROM = 1990  # fit window start: the post-development phase (earlier SAU years are mostly 0%)
BACKTEST_H = 5        # hold-out years for model selection


def logit_pct(s: pd.Series) -> pd.Series:
    """% share → logit, so forecasts and intervals stay inside 0–100%. Clipped at 0.5% / 99.5%."""
    p = (s / 100).clip(0.005, 0.995)
    return np.log(p / (1 - p))


def inv_logit_pct(x) -> np.ndarray:
    return 100 / (1 + np.exp(-np.asarray(x, float)))


def fit_forecast(s: pd.Series, model: str, horizon: int) -> pd.DataFrame:
    """
    Forecast `horizon` years of a % series with one model, on the logit scale, back-transformed.
    Returns mean, lower, upper (95% prediction interval) indexed by year.
      linear  OLS trend on logit(y) — the supervised-regression baseline; its interval ignores
              autocorrelation, so it is too narrow (reported, not preferred)
      ets     damped additive-trend exponential smoothing (ETS(A,Ad,N)); damping lets the trend
              flatten, the cautious choice for a bounded share
      arima   ARIMA(p,1,q) with drift, p, q ≤ 2 chosen by AIC
    """
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    y = logit_pct(s).astype(float)
    years = np.arange(s.index.max() + 1, s.index.max() + 1 + horizon)
    if model == "linear":
        t = y.index.to_numpy(float)
        fit = sm.OLS(y.to_numpy(), sm.add_constant(t)).fit()
        fr = fit.get_prediction(sm.add_constant(years.astype(float))).summary_frame(alpha=0.05)
        mean, lo, hi = fr["mean"], fr["obs_ci_lower"], fr["obs_ci_upper"]
    elif model == "ets":
        fit = ETSModel(y.reset_index(drop=True), error="add", trend="add", damped_trend=True).fit(disp=False)
        fr = fit.get_prediction(start=len(y), end=len(y) + horizon - 1).summary_frame(alpha=0.05)
        mean, lo, hi = fr["mean"], fr["pi_lower"], fr["pi_upper"]
    else:
        best = None
        for p_ in range(3):
            for q_ in range(3):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f = ARIMA(y.reset_index(drop=True), order=(p_, 1, q_), trend="t").fit()
                except Exception:   # some orders fail to converge on short series
                    continue
                if best is None or f.aic < best.aic:
                    best = f
        fr = best.get_forecast(horizon).summary_frame(alpha=0.05)
        mean, lo, hi = fr["mean"], fr["mean_ci_lower"], fr["mean_ci_upper"]
        model = f"arima{best.model.order}"
    return pd.DataFrame({"model": model, "mean": inv_logit_pct(mean), "lower": inv_logit_pct(lo),
                         "upper": inv_logit_pct(hi)}, index=years)


def section_i5_h5d() -> None:
    sec = "11_iteration5"
    print("Iteration 5: H5d — forecast of % overexploited to 2030")
    sea, cs = sea_series(), country_series()
    units = [("World (FAO)", st.world_fao()), ("World (SAU)", st.world_sau()), ("SEA (SAU)", sea["over"])]
    units += [(f"{c} (SAU)", cs[c]["over"]) for c in SEA5]
    bt_rows, fc_rows, paths, all_rows = [], [], [], []
    for name, s in units:
        s = s.loc[FORECAST_FROM:].dropna()
        # (1) back-test: fit on all but the last BACKTEST_H years, forecast them, score MAE in pp
        train, test = s.iloc[:-BACKTEST_H], s.iloc[-BACKTEST_H:]
        scores = {"naive (last value)": float(np.mean(np.abs(test.values - train.iloc[-1])))}
        for m in ("linear", "ets", "arima"):
            f = fit_forecast(train, m, BACKTEST_H)
            scores[m] = float(np.mean(np.abs(test.values - f["mean"].values)))
            bt_rows.append({"unit": name, "model": f["model"].iloc[0], "train": f"{train.index.min()}–{train.index.max()}",
                            "test": f"{test.index.min()}–{test.index.max()}", "mae_pp": scores[m],
                            "test_inside_95pi": int(((test.values >= f["lower"].values) &
                                                     (test.values <= f["upper"].values)).sum())})
        bt_rows.append({"unit": name, "model": "naive (last value)", "train": f"{train.index.min()}–{train.index.max()}",
                        "test": f"{test.index.min()}–{test.index.max()}", "mae_pp": scores["naive (last value)"],
                        "test_inside_95pi": np.nan})
        # (2) choose the model with the lowest back-test error (naive excluded: it gives no trend or interval)
        chosen = min(("linear", "ets", "arima"), key=lambda m: scores[m])
        horizon = FORECAST_END - int(s.index.max())
        for m in ("linear", "ets", "arima"):          # robustness: every model's 2030 value
            fm = fit_forecast(s, m, horizon).loc[FORECAST_END]
            all_rows.append({"unit": name, "model": fm["model"], "forecast_2030": fm["mean"],
                             "pi95_low_2030": fm["lower"], "pi95_high_2030": fm["upper"],
                             "low_above_last_value": bool(fm["lower"] > float(s.iloc[-1]))})
        f = fit_forecast(s, chosen, horizon)
        last = float(s.iloc[-1])
        r = f.loc[FORECAST_END]
        fc_rows.append({"unit": name, "fit_window": f"{s.index.min()}–{s.index.max()}", "chosen_model": r["model"],
                        "backtest_mae_pp": scores[chosen], "naive_mae_pp": scores["naive (last value)"],
                        "last_year": int(s.index.max()), "last_value": last, "forecast_2030": r["mean"],
                        "pi95_low_2030": r["lower"], "pi95_high_2030": r["upper"],
                        "beats_naive_in_backtest": scores[chosen] < scores["naive (last value)"],
                        "rise_with_95pct_confidence": bool(r["lower"] > last),
                        "fall_with_95pct_confidence": bool(r["upper"] < last)})
        paths.append(f.assign(unit=name).reset_index(names="year"))
    bt = pd.DataFrame(bt_rows)
    fc = pd.DataFrame(fc_rows)
    save_table(bt, "11_I5_H5d_forecast_backtest")
    save_table(fc, "11_I5_H5d_forecast_2030")
    save_table(pd.concat(paths, ignore_index=True), "11_I5_H5d_forecast_paths")
    allm = pd.DataFrame(all_rows)
    save_table(allm, "11_I5_H5d_forecast_all_models")
    wa = allm[allm["unit"] == "World (FAO)"]

    # A projected rise counts only if the trend model also beat a no-change forecast in the back-test;
    # otherwise the recent years have flattened and extending the old trend is not supported.
    fc["rise_credible"] = fc["rise_with_95pct_confidence"] & fc["beats_naive_in_backtest"]
    fc.to_csv(RES / "11_I5_H5d_forecast_2030.csv", index=False, float_format="%.6g")
    w, s_ = fc.iloc[0], fc.iloc[2]
    rising = fc[fc["rise_credible"]]["unit"].tolist()
    flat = fc[~fc["beats_naive_in_backtest"]]["unit"].tolist()
    reject = bool(w["rise_credible"])
    decide("H5d (iteration 5: forecast to 2030)",
           "On current trends, will the share of overexploited stocks in 2030 be higher than its latest value?",
           f"World (FAO, {w['chosen_model']}): {w['last_value']:.1f}% in {w['last_year']} → {w['forecast_2030']:.1f}% in "
           f"2030 [95% PI {w['pi95_low_2030']:.1f}–{w['pi95_high_2030']:.1f}]; SEA (SAU, {s_['chosen_model']}): "
           f"{s_['last_value']:.1f}% → {s_['forecast_2030']:.1f}% [{s_['pi95_low_2030']:.1f}–{s_['pi95_high_2030']:.1f}]; "
           f"world 2030 lower bound above {w['last_value']:.1f}% in {int(wa['low_above_last_value'].sum())} of 3 models ("
           + ", ".join(f"{r['model']} {r['pi95_low_2030']:.1f}" for _, r in wa.iterrows()) + "); "
           f"rise with 95% confidence AND trend validated in the back-test: {', '.join(rising) or 'none'}; a no-change "
           f"forecast beat every trend model in: {', '.join(flat) or 'none'}",
           reject,
           "Model per series chosen by a 5-year back-test (linear trend on logit, damped-trend ETS, ARIMA with drift); "
           "the logit keeps forecasts within 0–100%. A forecast extends past trends, so it assumes no change in "
           "management; it shows the distance from the SDG target, not a prediction of policy. Where the no-change "
           "forecast wins the back-test (most Sea Around Us series flattened in 2015–19), the projected rise is not "
           "supported and the honest reading is 'no improvement', not 'further rise'.",
           verdict=None if reject else
           ("Fail to reject H0 for the world (2030 interval includes today's level)" +
            (f"; rise projected with 95% confidence in {', '.join(rising)}" if rising else "")))

    # Figure: small multiples, history + forecast mean + 95% PI to 2030
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), sharex=True)
    for ax, (name, s) in zip(axes.flat, units):
        s = s.loc[FORECAST_FROM:].dropna()
        f = pd.concat(paths)[lambda d: d["unit"] == name].set_index("year")
        r = fc[fc["unit"] == name].iloc[0]
        c = name.split(" (")[0]
        col = v.COUNTRY_COLOR.get(c, v.SERIES[0] if c == "SEA" else v.INK_2)
        ax.plot(s.index, s.values, color=v.INK, lw=1.6)
        xs = np.r_[s.index[-1], f.index]
        ax.fill_between(xs, np.r_[s.iloc[-1], f["lower"]], np.r_[s.iloc[-1], f["upper"]], color=col, alpha=0.18, lw=0)
        ax.plot(xs, np.r_[s.iloc[-1], f["mean"]], color=col, lw=2, ls=(0, (4, 2)))
        ax.scatter([FORECAST_END], [r["forecast_2030"]], s=36, color=col, edgecolor=v.SURFACE, lw=1, zorder=4)
        ax.axhline(r["last_value"], color=v.AXIS, lw=0.8, ls=(0, (2, 2)))
        ax.set_title(f"{name}: {r['last_value']:.0f}% → {r['forecast_2030']:.0f}% "
                     f"[{r['pi95_low_2030']:.0f}–{r['pi95_high_2030']:.0f}]", fontsize=9.2)
        if not r["beats_naive_in_backtest"]:
            ax.text(0.02, 0.9, "trend not validated:\nno-change forecast won back-test", transform=ax.transAxes,
                    fontsize=7.8, color=v.INK_2, va="top")
        ax.text(0.02, 0.04, r["chosen_model"].replace("arima", "ARIMA").replace("ets", "damped ETS")
                .replace("linear", "linear trend"), transform=ax.transAxes, fontsize=7.8, color=v.MUTED)
        ax.set_ylim(0, max(80, float(f["upper"].max()) + 5))
    for ax in axes[:, 0]:
        ax.set_ylabel("% of stocks overexploited")
    for ax in axes[1]:
        ax.set_xticks([1990, 2000, 2010, 2020, 2030])
    v.suptitle(fig, "Iteration 5 · H5d — " + ("On current trends, overexploitation will be higher in 2030"
                                              if reject else "Forecast to the SDG deadline (2030): no reversal in sight"),
               f"Solid = data since 1990; dashed = forecast, shaded = 95% prediction interval; dotted = latest value. "
               f"A rise is supported (interval above today and trend validated) for: {', '.join(rising) or 'none'}. "
               "Linear-trend forecasts follow the fitted line, so they can start away from the last value (Thailand)")
    v.layout(fig, 0)
    save_fig(fig, sec, "I5.04_h5d_forecast_2030", "FAO via OWID; Sea Around Us")


# =========================================================================== #
# Summary across iterations — every driver, one common method (report figure)
# =========================================================================== #
# The per-iteration sections each use the tests that suit their driver. For the report,
# one figure compares ALL drivers on the same footing, so the reader sees in one place
# that every driver shares the upward trend and none is linked beyond it:
#   levels ρ · linear-detrended ρ · quadratic-detrended ρ (Holm across all driver ×
#   source pairs) and the year-to-year CCF (growth / first difference vs Δ response,
#   lags 0 to n/3, Holm within pair and across all lag tests).
# Values can differ slightly from the per-iteration tables, which use driver-specific
# transformations (e.g. source-break removal for effort).

def summary_drivers() -> list[tuple[str, str, pd.Series, bool]]:
    """(iteration label, driver name, world level series, log?) for every driver tested on world data."""
    c, s, d, a = container_teu(), seaborne(), dams(), aquaculture()
    return [("1", "Catch (FAO capture)", st.capture("World").loc[1960:], True),
            ("2", "Fishing effort (nominal)", world_effort()["nom_kwdays"], True),
            ("3", "Sea-surface temperature", world_sst(), False),
            ("3", "Ocean heat content", world_ohc(), False),
            ("4", "Container port traffic", c["World"].dropna(), True),
            ("4", "Seaborne trade volume", s["World"].dropna(), True),
            ("4", "Dams completed (cumulative)", d["World"], True),
            ("4", "Aquaculture production", a["World"].dropna(), True)]


def section_summary() -> None:
    sec = "10_summary_all_drivers"
    print("Summary: every driver, levels vs beyond-trend, one common method")
    responses = [("FAO", st.world_fao()), ("SAU", st.world_sau())]
    rows, tabs = [], []
    for it, dn, x, log in summary_drivers():
        xl = np.log(x) if log else x
        dx = i4_change(x, log)
        for rn, y in responses:
            r_lv, p_lv, r_dt, p_dt, n = detrended_corr(xl, y)
            r_q, p_q = quad_detrended_corr(xl, y)
            n_ch = len(pd.concat([dx, y.diff()], axis=1).dropna())
            t = ccf_table(dx, y.diff().dropna(), f"{dn} -> Δ {rn} % overexploited", i4_max_lag(n_ch))
            t.insert(0, "iteration", it)
            t.insert(1, "driver", dn)
            t.insert(2, "response", rn)
            tabs.append(t)
            rows.append({"iteration": it, "driver": dn, "response": f"{rn} % overexploited",
                         "years": f"{pd.concat([xl, y], axis=1).dropna().index.min()}–"
                                  f"{pd.concat([xl, y], axis=1).dropna().index.max()}", "n_years": n,
                         "spearman_levels": r_lv, "p_levels": p_lv, "spearman_detrended": r_dt, "p_detrended": p_dt,
                         "spearman_quad_detrended": r_q, "p_quad_detrended": p_q,
                         "ccf_max_lag": int(t["lag"].max()), "ccf_min_p_holm_within_pair": t["p_holm"].min()})
    res = pd.DataFrame(rows)
    res["p_detrended_holm"] = holm(res["p_detrended"])
    res["p_quad_holm"] = holm(res["p_quad_detrended"])
    res["consistent_beyond_trend"] = i4_consistent(res)
    ccf = pd.concat(tabs, ignore_index=True)
    ccf["p_holm_all"] = holm(ccf["p"])
    res["ccf_min_p_holm_all"] = res.apply(lambda r: ccf[(ccf["driver"] == r["driver"]) & (
        ccf["response"] == r["response"].split()[0])]["p_holm_all"].min(), axis=1)
    save_table(res, "10_summary_all_drivers")
    save_table(ccf, "10_summary_all_drivers_ccf")

    both = [d for d, g in res[res["consistent_beyond_trend"]].groupby("driver")
            if len(g) == 2 and g["spearman_detrended"].apply(np.sign).nunique() == 1]
    n_ccf_sig = int((ccf["p_holm_all"] < ALPHA).sum())

    # Figure: rows = driver × source grouped by iteration; (1) ρ in levels vs after detrending,
    # (2) strongest year-to-year link (−log10 of the smallest Holm p across lags, within pair).
    fig = plt.figure(figsize=(15, 9.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 0.75], wspace=0.08)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ypos, labels, y = [], [], 0.0
    group_edges = []
    for it in ["1", "2", "3", "4"]:
        sub = res[res["iteration"] == it]
        if y:
            y += 0.7
            group_edges.append(y - 0.85)
        for _, r in sub.iterrows():
            ypos.append(y)
            labels.append(f"{r['driver']} × {r['response'].split()[0]}")
            y += 1
    ypos = np.array(ypos)
    yplot = ypos.max() - ypos
    for yp, (_, r) in zip(yplot, res.iterrows()):
        a1.plot([r["spearman_levels"], r["spearman_detrended"]], [yp, yp], color=v.GRID, lw=2.5, zorder=1)
        a1.scatter(r["spearman_levels"], yp, s=46, facecolor=v.SURFACE, edgecolor=v.INK_2, lw=1.4, zorder=3)
        a1.scatter(r["spearman_detrended"], yp, s=52, zorder=4, edgecolor=v.SURFACE, lw=1,
                   color=v.SERIES[0] if r["p_detrended_holm"] < ALPHA else v.NEUTRAL)
        a1.scatter(r["spearman_quad_detrended"], yp, s=46, marker="D", zorder=4, edgecolor=v.SURFACE, lw=1,
                   color=v.SERIES[1] if r["p_quad_holm"] < ALPHA else v.NEUTRAL)
    for e in group_edges:
        a1.axhline(ypos.max() - e, color=v.AXIS, lw=0.8)
        a2.axhline(ypos.max() - e, color=v.AXIS, lw=0.8)
    a1.axvline(0, color=v.AXIS, lw=0.8)
    a1.set_yticks(yplot, labels, fontsize=8.5)
    a1.set_xlim(-1.05, 1.05)
    a1.set_xlabel("Spearman ρ with world % overexploited: ○ levels, ● linear detrend, ◆ quadratic detrend\n"
                  "(coloured = Holm p < 0.05 across all 16 pairs, grey = not)")
    a1.grid(axis="y", visible=False)
    a1.set_title("1 · Levels: every driver ρ ≥ 0.7 — the shared upward trend", fontsize=9.5)
    for it in ["1", "2", "3", "4"]:
        top = yplot[res["iteration"].to_numpy() == it].max()
        a1.text(-1.03, top + 0.45, f"Iteration {it}", fontsize=8.5, color=v.INK_2, fontweight="bold", va="bottom")
    best = -np.log10(res["ccf_min_p_holm_within_pair"].clip(lower=1e-6))
    a2.barh(yplot, best, height=0.6,
            color=[v.SERIES[0] if p < ALPHA else v.NEUTRAL for p in res["ccf_min_p_holm_all"]])
    a2.axvline(-np.log10(ALPHA), color=v.INK_2, lw=1, ls=(0, (3, 3)))
    a2.text(-np.log10(ALPHA), yplot.max() + 0.6, " p = 0.05", color=v.INK_2, fontsize=8, va="bottom")
    a2.set_yticks(yplot, [""] * len(yplot))
    a2.set_ylim(a1.get_ylim())
    a2.set_xlabel("−log10(smallest Holm p across lags, within pair)\n(blue = significant across all lag tests)")
    a2.grid(axis="y", visible=False)
    a2.set_title("2 · Year-to-year changes: no driver leads", fontsize=9.5)
    v.suptitle(fig, "Every driver tested rises with overexploitation, but none is linked to it beyond the shared trend"
               if not both and n_ccf_sig == 0 else
               "Summary — a driver is linked to overexploitation beyond the shared trend",
               f"8 drivers from 4 iterations, world data. Linked beyond the trend in both sources and both detrends: "
               f"{', '.join(both) or 'none'}. Year-to-year lag tests significant across all {len(ccf)}: {n_ccf_sig}")
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.19, right=0.98)
    save_fig(fig, sec, "S.01_all_drivers_levels_vs_beyond_trend",
             "FAO via OWID; Sea Around Us; IMAS effort; HadSST & NOAA via OWID; World Bank WDI; UNCTADstat; GDAT")


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
            # iteration 4: shipping and built structures (new hypothesis H4)
            "i4_data": section_i4_data, "i4_h4a": section_i4_h4a, "i4_h4b": section_i4_h4b,
            "i4_h4c": section_i4_h4c, "i4_descriptive": section_i4_descriptive,
            # iteration 5: pooled panel, stock rebuilding, Fishing-in-Balance (new hypothesis H5)
            "i5_h5a": section_i5_h5a, "i5_h5b": section_i5_h5b, "i5_h5c": section_i5_h5c, "i5_h5d": section_i5_h5d,
            # summary across iterations (report figure S.01)
            "summary": section_summary,
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
