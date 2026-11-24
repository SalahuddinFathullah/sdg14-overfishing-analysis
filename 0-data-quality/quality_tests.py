"""
quality_tests.py — Formal validation and data-integrity tests
=============================================================

KIE4033 Assignment 1 · SDG 14 overfishing analysis (Global -> Southeast Asia)

Turns the visual checks of ``1-visualisation`` sections 4 (validation) and 5
(integrity) into formal tests, following the validation table in the main README.

    python 0-data-quality/quality_tests.py              # both sections
    python 0-data-quality/quality_tests.py --only validation

Section V — validation: does each main series agree with an alternative source?
    For every pair of sources on their common years:
      systematic difference   Shapiro-Wilk on the differences decides between the
                              paired t-test and the Wilcoxon signed-rank test
      size of disagreement    Bland-Altman bias and 95% limits of agreement
                              (in % of the pair mean when the unit is tonnes)
      point-for-point         Lin's concordance correlation coefficient (CCC)
      moving together         Spearman ρ on levels and on first differences
      same trend              Sen's slope + 95% CI for each source; overlapping CIs
                              = same trend (tonnes: % per year, from log values)
    Verdict: "agree on level" if CCC ≥ 0.90; "agree on trend" if Sen CIs overlap;
    "same trend direction, different rate" if both CIs exclude 0 on the same side.

Section I — integrity: is officially reported catch plausible?
    I1 Benford        MAD of leading digits per country × reported/unreported,
                      Nigrini conformity band
    I2 Mirror trade   one-sample Wilcoxon of the yearly gap (partner imports vs own
                      exports): median > 10% (above the CIF/FOB band → under-reported
                      exports?) and median < 0% (re-exports?), Holm across countries;
                      Mann-Kendall trend of the gap
    I3 Peer outliers  log-log regression of fisheries value added on catch across all
                      countries (2021 and 2023); SEA5 z-scores and 95% prediction intervals
    I4 Unreported     Mann-Kendall + Sen's slope of the unreported share 1950–2019
                      and 1990–2019, Holm across countries
    I5 Agreement      do the integrity signals rank countries the same way? (Spearman, n = 5)

Outputs: results/*.csv and figures/*.png in this folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))   # step 0 imports shared code only, never a later step
import loaders as ld  # noqa: E402  (data loaders, reading the checked copies)
import stats_utils as su  # noqa: E402  (Holm, Mann-Kendall, Benford MAD, p formatting)
import style as v  # noqa: E402  (chart style + header helpers)

VAL = ld.CLEAN_DIR / "validation"   # checked copies written by clean_data.py
FIG = HERE / "figures"
RES = HERE / "results"
ALPHA = 0.05
CCC_GOOD = 0.90
fmt_p, holm, mann_kendall = su.fmt_p, su.holm, su.mann_kendall


def save_table(df: pd.DataFrame, name: str) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(RES / f"{name}.csv", index=False)
    print(f"  saved 0-data-quality/results/{name}.csv")


def save_fig(fig, name: str, source: str) -> None:
    fig.text(0.01, -0.01, f"Source: {source}", ha="left", va="top", fontsize=8, color=v.MUTED)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"  saved 0-data-quality/figures/{name}.png")


# =========================================================================== #
# Section V — validation against alternative sources
# =========================================================================== #
def lins_ccc(a: np.ndarray, b: np.ndarray) -> float:
    """Lin's CCC = 2·cov(a,b) / (var a + var b + (mean a − mean b)²); population moments."""
    return float(2 * np.cov(a, b, ddof=0)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


def compare(name: str, a: pd.Series, b: pd.Series, unit: str, a_lbl: str, b_lbl: str) -> dict:
    """All validation statistics for one pair of series (common years only)."""
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    x, y = d["a"].to_numpy(float), d["b"].to_numpy(float)
    tonnes = unit == "t"
    diff = (x - y) / ((x + y) / 2) * 100 if tonnes else x - y          # % of pair mean for tonnes
    out = {"pair": name, "source_a": a_lbl, "source_b": b_lbl, "unit": unit,
           "years": f"{d.index.min()}–{d.index.max()}", "n": len(d)}
    if np.allclose(x, y, atol=1e-6, rtol=1e-6):
        out.update({"identical": True, "test": "—", "p_difference": np.nan, "bias": 0.0, "loa_low": 0.0,
                    "loa_high": 0.0, "ccc": 1.0, "spearman_levels": 1.0, "spearman_differences": 1.0,
                    "sen_a": np.nan, "sen_b": np.nan, "sen_ci_overlap": True,
                    "verdict": "identical (same data, two publishers)"})
        return out
    sw = stats.shapiro(diff).pvalue if len(diff) >= 3 else np.nan
    if sw >= ALPHA:
        test, p = "paired t-test", stats.ttest_rel(x, y).pvalue
    else:
        test, p = "Wilcoxon signed-rank", stats.wilcoxon(x, y).pvalue
    bias, sd = diff.mean(), diff.std(ddof=1)
    ta, tb = (np.log(d["a"]), np.log(d["b"])) if tonnes else (d["a"], d["b"])
    sa = stats.theilslopes(ta.values, ta.index.to_numpy(float))
    sb = stats.theilslopes(tb.values, tb.index.to_numpy(float))
    k = 100 if tonnes else 1                                             # log slope ≈ % per year
    overlap = not (sa[3] < sb[2] or sb[3] < sa[2])
    ccc = lins_ccc(x, y)
    dx, dy = np.diff(x), np.diff(y)
    out.update({"identical": False, "shapiro_p_diff": sw, "test": test, "p_difference": p,
                "bias": bias, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
                "bias_unit": "% of pair mean" if tonnes else unit, "ccc": ccc,
                "spearman_levels": stats.spearmanr(x, y)[0],
                "spearman_differences": stats.spearmanr(dx, dy)[0] if len(dx) > 2 else np.nan,
                "p_spearman_differences": stats.spearmanr(dx, dy)[1] if len(dx) > 2 else np.nan,
                "sen_a": sa[0] * k, "sen_a_low": sa[2] * k, "sen_a_high": sa[3] * k,
                "sen_b": sb[0] * k, "sen_b_low": sb[2] * k, "sen_b_high": sb[3] * k,
                "sen_unit": "% per year" if tonnes else f"{unit} per year", "sen_ci_overlap": overlap})
    level = "agree on level" if ccc >= CCC_GOOD else "differ in level"
    same_dir = (sa[2] > 0 and sb[2] > 0) or (sa[3] < 0 and sb[3] < 0)
    trend = ("agree on trend" if overlap else
             "same trend direction, different rate" if same_dir else "differ in trend")
    out["trend_agreement"] = trend
    out["verdict"] = f"{level}; {trend}"
    return out


def section_validation() -> None:
    print("Section V: validation")
    rows = []
    fao = ld.world_fao()
    sdg = pd.read_csv(VAL / "un_sdg" / "14_4_1_ER_H2O_FWTL.csv")
    sdg_over = 100 - sdg[sdg["geo_area_name"] == "World"].set_index("year")["value_pct_sustainable"].loc[:2021]
    rows.append(compare("World % overexploited: OWID vs UN SDG database", fao, sdg_over, "pp",
                        "FAO via OWID", "UN SDG 14.4.1 (100 − sustainable)"))
    rows.append(compare("World % overexploited: FAO vs Sea Around Us", fao, ld.world_sau(), "pp",
                        "FAO via OWID", "Sea Around Us"))
    for ent, f, lbl in [("Western Central Pacific", "fao_71_pacific_western_central", "Area 71"),
                        ("Eastern Indian Ocean", "fao_57_indian_ocean_eastern", "Area 57")]:
        a = ld.fao_area(ent)
        b = ld.sau_overexploited(VAL / "sau_regions" / "stock_status" / f"{f}.csv")
        rows.append(compare(f"{lbl} % overexploited: FAO vs Sea Around Us", a, b, "pp", "FAO via OWID", "Sea Around Us"))
    wb = pd.read_csv(VAL / "world_bank" / "ER.FSH.CAPT.MT.csv")
    wb["country"] = wb["country"].replace({"Viet Nam": "Vietnam"})
    d = ld.load_sau_eez()["catch"]
    rep = d[d["reporting_status"] == "Reported"].groupby(["country", "year"])["tonnes"].sum()
    glob = pd.read_csv(VAL / "sau_regions" / "catch" / "global.csv").set_index("year")["reported"]
    for c in ["World"] + ld.SEA5:
        owid_cap = ld.capture(c)
        wbc = wb[wb["country"] == c].set_index("year")["capture_fisheries_production_t"]
        rows.append(compare(f"{c} capture: OWID vs World Bank WDI", owid_cap, wbc, "t", "FAO via OWID", "World Bank WDI"))
        sau_rep = glob if c == "World" else rep.xs(c, level="country")
        rows.append(compare(f"{c} catch: FAO capture vs Sea Around Us reported", owid_cap, sau_rep, "t",
                            "FAO capture (incl. inland)", "Sea Around Us reported marine"))
    stock = ld.load_sau_eez()["stock"]
    stock["over"] = stock["Collapsed"] + stock["Over-exploited"]
    for f, title, ids in [("lme_35_gulf_of_thailand", "Gulf of Thailand", [957]),
                          ("lme_36_south_china_sea", "South China Sea", [704, 460, 461, 463]),
                          ("lme_37_sulu_celebes_sea", "Sulu-Celebes Sea", [608]),
                          ("lme_38_indonesian_sea", "Indonesian Sea", [937, 361]),
                          ("lme_34_bay_of_bengal", "Bay of Bengal", [956, 459])]:
        lme = ld.sau_overexploited(VAL / "sau_regions" / "stock_status" / f"{f}.csv")
        eez = stock[stock["eez"].isin(ids)].groupby("year")["over"].mean()
        rows.append(compare(f"{title}: EEZ mean vs Large Marine Ecosystem (% overexploited)", eez, lme, "pp",
                            "Sea Around Us EEZs", "Sea Around Us LME"))
    res = pd.DataFrame(rows)
    save_table(res, "V1_validation_tests")

    # Figure: CCC and Sen slopes side by side for every non-identical pair
    r = res[~res["identical"]].reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.8), gridspec_kw={"width_ratios": [1, 1.25]}, sharey=True)
    ys = np.arange(len(r))[::-1]
    a1.barh(ys, r["ccc"], color=[v.SERIES[0] if c >= CCC_GOOD else v.NEUTRAL for c in r["ccc"]], height=0.6)
    a1.axvline(CCC_GOOD, color=v.INK_2, lw=1, ls=(0, (3, 3)))
    for y_, c in zip(ys, r["ccc"]):
        a1.text(max(c, 0) + 0.02, y_, f"{c:.2f}", va="center", fontsize=8.5, color=v.INK_2)
    a1.set_xlim(min(0, r["ccc"].min() - 0.05), 1.12)
    a1.set_yticks(ys, [p.replace(" (% overexploited)", "") for p in r["pair"]], fontsize=8.5)
    a1.set_xlabel("Lin's CCC (1 = identical; dashed = 0.90)")
    a1.set_title("Point-for-point agreement (blue = agree on level)", fontsize=9.5)
    a1.grid(axis="y", visible=False)
    for y_, (_, row) in zip(ys, r.iterrows()):
        for off, key, colour in [(0.15, "a", v.SERIES[0]), (-0.15, "b", v.SERIES[1])]:
            a2.plot([row[f"sen_{key}_low"], row[f"sen_{key}_high"]], [y_ + off] * 2, color=colour, lw=2)
            a2.scatter(row[f"sen_{key}"], y_ + off, s=36, color=colour, zorder=3, edgecolor=v.SURFACE, lw=1,
                       label=("Source A (first named)" if key == "a" else "Source B") if y_ == ys[0] else None)
        tag = {"agree on trend": "✓ same trend", "same trend direction, different rate": "≈ same direction"}.get(
            row["trend_agreement"], "✗ differ")
        a2.text(1.01, y_, tag, transform=a2.get_yaxis_transform(), ha="left", va="center", fontsize=8.5, color=v.INK_2)
    a2.axvline(0, color=v.AXIS, lw=0.8)
    a2.set_xlabel("Sen's slope with 95% CI (pp per year, or % per year for tonnes)")
    a2.set_title("Trend agreement: overlapping CIs = same trend", fontsize=9.5)
    a2.legend(loc="lower right", fontsize=8.5)
    a2.grid(axis="y", visible=False)
    n_lvl = int((r["ccc"] >= CCC_GOOD).sum())
    n_tr = int(r["sen_ci_overlap"].sum())
    n_dir = int((r["trend_agreement"] == "same trend direction, different rate").sum())
    v.suptitle(fig, f"V.01 — Validation: {n_tr} of {len(r)} source pairs agree on trend ({n_dir} more on direction only), "
               f"{n_lvl} agree on level",
               f"{int(res['identical'].sum())} more pairs are identical (same data, two publishers). Level gaps have known "
               "causes: FAO counts inland catch, Sea Around Us adds unreported catch and uses a catch-based stock method")
    fig.subplots_adjust(top=0.84, left=0.27, right=0.9, wspace=0.08, bottom=0.08)
    save_fig(fig, "V.01_validation_summary", "FAO via OWID; UN SDG; World Bank WDI; Sea Around Us")


# =========================================================================== #
# Section I — data integrity
# =========================================================================== #
def section_integrity() -> None:
    print("Section I: data integrity")
    iso = {"MYS": "Malaysia", "IDN": "Indonesia", "THA": "Thailand", "VNM": "Vietnam", "PHL": "Philippines"}

    # I1 Benford ------------------------------------------------------------------
    bf = pd.read_csv(VAL / "benford" / "sau_catch_leading_digits.csv")
    dcols = [f"d{i}" for i in range(1, 10)]
    g = bf.groupby(["country", "reporting_status"])[dcols].sum()
    b_rows = []
    for (c, s), cnt in g.iterrows():
        mad = su.benford_mad(cnt.to_numpy(float))
        b_rows.append({"country": c, "reporting_status": s, "n_values": int(cnt.sum()), "mad": mad,
                       "nigrini": su.nigrini_label(mad)})
    ben = pd.DataFrame(b_rows)
    save_table(ben, "I1_benford_mad")

    # I2 Mirror trade -------------------------------------------------------------
    ct = pd.read_csv(VAL / "comtrade" / "fish_hs03_mirror_trade.csv")
    ct["gap"] = (ct["partner_imports_usd"] / ct["reported_exports_usd"] - 1) * 100
    m_rows = []
    for c in ld.SEA5:
        gp = ct[ct["country"] == c].set_index("year")["gap"].dropna()
        up = stats.wilcoxon(gp - 10, alternative="greater")
        dn = stats.wilcoxon(gp, alternative="less")
        mk_ = mann_kendall(gp)
        m_rows.append({"country": c, "years": f"{gp.index.min()}–{gp.index.max()}", "n": len(gp),
                       "median_gap_pct": gp.median(), "p_median_above_10pct": up.pvalue,
                       "p_median_below_0pct": dn.pvalue, "trend": mk_["trend"], "trend_p": mk_["p"],
                       "sen_pp_per_year": mk_["sen_slope"]})
    mir = pd.DataFrame(m_rows)
    mir["p_above_holm"] = holm(mir["p_median_above_10pct"])
    mir["p_below_holm"] = holm(mir["p_median_below_0pct"])
    mir["reading"] = np.select([mir["p_above_holm"] < ALPHA, mir["p_below_holm"] < ALPHA],
                               ["partners report clearly more: possible under-reported exports",
                                "partners report less: re-exports / transshipment"],
                               "median within the normal 0–10% CIF/FOB band")
    mir["trend_p_holm"] = holm(mir["trend_p"])
    mir.loc[mir["trend_p_holm"] < ALPHA, "reading"] += np.where(
        mir.loc[mir["trend_p_holm"] < ALPHA, "sen_pp_per_year"] > 0, "; gap widening over time", "; gap narrowing over time")
    save_table(mir, "I2_mirror_trade_tests")

    # I3 Peer outliers with prediction intervals ---------------------------------
    econ = pd.read_csv(VAL / "economic" / "fisheries_gdp_capture.csv")
    econ["va"] = econ["sustainable_fisheries_pct_gdp"] / 100 * econ["gdp_current_usd"]
    p_rows = []
    for year in (2021, 2023):
        x = econ[(econ["year"] == year) & (econ["iso3"] != "WLD") & (econ["va"] > 0) & (econ["capture_t"] > 0)]
        lx, ly = np.log10(x["capture_t"].to_numpy()), np.log10(x["va"].to_numpy())
        fit = stats.linregress(lx, ly)
        resid = ly - (fit.intercept + fit.slope * lx)
        s = resid.std(ddof=2)
        n = len(lx)
        tcrit = stats.t.ppf(0.975, n - 2)
        for i3, c in iso.items():
            m = (x["iso3"] == i3).to_numpy()
            if not m.any():
                continue
            xi = lx[m][0]
            se_pred = s * np.sqrt(1 + 1 / n + (xi - lx.mean()) ** 2 / ((lx - lx.mean()) ** 2).sum())
            pred = fit.intercept + fit.slope * xi
            p_rows.append({"year": year, "country": c, "n_countries": n, "r2": fit.rvalue ** 2,
                           "z": (resid[m][0] - resid.mean()) / resid.std(ddof=1),
                           "observed_log10_va": ly[m][0], "pi95_low": pred - tcrit * se_pred,
                           "pi95_high": pred + tcrit * se_pred,
                           "inside_pi95": bool(pred - tcrit * se_pred <= ly[m][0] <= pred + tcrit * se_pred)})
    peer = pd.DataFrame(p_rows)
    save_table(peer, "I3_peer_outliers")

    # I4 Unreported share trends ---------------------------------------------------
    e = pd.read_csv(ROOT / "data" / "processed" / "sea_eez_annual.csv")
    u_rows = []
    for c in ld.SEA5:
        sub = e[e["country"] == c]
        sh = (sub["unreported_share_pct"] * sub["catch_t"]).groupby(sub["year"]).sum() / sub.groupby("year")["catch_t"].sum()
        for lo in (1950, 1990):
            r = mann_kendall(sh.loc[lo:2019])
            u_rows.append({"country": c, "window": f"{lo}–2019", "share_start_5yr": sh.loc[lo:lo + 4].mean(),
                           "share_2015_19": sh.loc[2015:2019].mean(), "trend": r["trend"], "p": r["p"],
                           "sen_pp_per_decade": r["sen_slope"] * 10})
    unr = pd.DataFrame(u_rows)
    unr["p_holm"] = unr.groupby("window")["p"].transform(lambda s_: holm(s_))
    save_table(unr, "I4_unreported_share_trends")

    # I5 Do the signals agree? ------------------------------------------------------
    rec = unr[unr["window"] == "1990–2019"].set_index("country")["share_2015_19"]
    gap = mir.set_index("country")["median_gap_pct"]
    z = peer[peer["year"] == 2021].set_index("country")["z"]
    mad = ben[ben["reporting_status"] == "Reported"].set_index("country")["mad"]
    pairs = [("unreported share", rec, "median mirror gap", gap), ("unreported share", rec, "peer z (2021)", z),
             ("median mirror gap", gap, "peer z (2021)", z), ("Benford MAD", mad, "unreported share", rec)]
    a_rows = []
    for an, a, bn, b in pairs:
        d = pd.concat([a, b], axis=1).dropna()
        r_, p_ = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
        a_rows.append({"signal_a": an, "signal_b": bn, "n": len(d), "spearman": r_, "p_exact_n5": p_})
    agree = pd.DataFrame(a_rows)
    save_table(agree, "I5_integrity_signal_agreement")

    # Figure: mirror gaps with band + unreported share trends ------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1, 1.1]})
    a1.axhspan(0, 10, color=v.GRID, alpha=0.7, lw=0, label="Expected 0–10% (CIF/FOB)")
    rng = np.random.default_rng(0)
    for k, c in enumerate(ld.SEA5):
        gp = ct[ct["country"] == c]["gap"].dropna()
        a1.scatter(k + rng.uniform(-0.12, 0.12, len(gp)), gp, s=22, color=v.COUNTRY_COLOR[c], edgecolor=v.SURFACE,
                   lw=0.8, zorder=3)
        a1.plot([k - 0.25, k + 0.25], [gp.median()] * 2, color=v.INK, lw=2, zorder=4)
        rr = mir[mir["country"] == c].iloc[0]
        tag = ("above band" if rr["p_above_holm"] < ALPHA else "below 0" if rr["p_below_holm"] < ALPHA else "in band")
        if rr["trend_p_holm"] < ALPHA:
            tag += f"\n{'rising' if rr['sen_pp_per_year'] > 0 else 'falling'} {rr['sen_pp_per_year']:+.1f} pp/yr"
        a1.text(k, gp.max() + 4, tag, ha="center", va="bottom", fontsize=8.5, color=v.INK_2)
    a1.axhline(0, color=v.AXIS, lw=0.8)
    a1.set_xticks(range(5), ld.SEA5)
    a1.set_ylabel("Partner imports vs own exports (% gap), 2011–2023")
    a1.legend(loc="upper left", fontsize=8.5)
    a1.set_title("I2 · Mirror trade gap per year (bar = median; label = Holm-corrected Wilcoxon result)", fontsize=9.5)
    a1.grid(axis="x", visible=False)
    for c in ld.SEA5:
        sub = e[e["country"] == c]
        sh = (sub["unreported_share_pct"] * sub["catch_t"]).groupby(sub["year"]).sum() / sub.groupby("year")["catch_t"].sum()
        sh = sh.rolling(5, center=True).mean()
        rr = unr[(unr["country"] == c) & (unr["window"] == "1990–2019")].iloc[0]
        a2.plot(sh.index, sh.values, color=v.COUNTRY_COLOR[c], lw=1.8,
                label=f"{c} ({rr['sen_pp_per_decade']:+.1f} pp/decade since 1990)")
    a2.set_ylabel("Unreported share of catch (%), 5-yr mean")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8.5)
    falling = unr[(unr["window"] == "1990–2019") & (unr["p_holm"] < ALPHA) & (unr["trend"] == "decreasing")]["country"]
    a2.set_title(f"I4 · Unreported share falling since 1990 in {', '.join(falling) or 'no country'} (Holm)", fontsize=9.5)
    above = mir[mir["p_above_holm"] < ALPHA]["country"].tolist()
    below = mir[mir["p_below_holm"] < ALPHA]["country"].tolist()
    v.suptitle(fig, "I.01 — Integrity: digits look genuine; the problem is unreported catch and trade gaps",
               f"Benford MAD {ben['mad'].min():.4f}–{ben['mad'].max():.4f} (all close conformity); peers: all SEA5 inside the "
               f"95% prediction interval. Mirror gap above band: {', '.join(above) or 'none'}; below 0: {', '.join(below) or 'none'}")
    v.layout(fig, 0)
    save_fig(fig, "I.01_integrity_summary", "Sea Around Us; UN Comtrade; FAO/World Bank via OWID")


SECTIONS = {"validation": section_validation, "integrity": section_integrity}


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal validation and integrity tests.")
    parser.add_argument("--only", nargs="+", choices=SECTIONS)
    args = parser.parse_args()
    ld.require_checked_data()   # clean_data.py (stage A + B) must have passed first
    v.setup_style()
    for name in args.only or SECTIONS:
        SECTIONS[name]()


if __name__ == "__main__":
    main()
