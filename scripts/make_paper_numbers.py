"""Turn the result CSVs into the tables and the numbers the paper prints.

Every figure in the paper's prose is emitted here as a LaTeX macro, so a number cannot be
typed into the text by hand and drift away from the data. Writes outputs/paper/*.tex and
prints a plain-text summary for checking.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "outputs"))
DEST = os.path.join(OUT, "paper")
CITIES = ["london", "bayarea", "tokyo"]
CITY_NAME = {"london": "London", "bayarea": "San Francisco", "tokyo": "Tokyo"}
RNG = np.random.default_rng(2026)
os.makedirs(DEST, exist_ok=True)
# LaTeX macro names cannot contain digits or underscores
RULE_TAG = {"variance_opt": "Var", "ede_opt": "Ede", "min_range": "Minrange",
            "min_max": "Minmax", "min_sum": "Minsum", "random_100": "Random",
            "iso_intersection": "Iso", "centroid": "Centroid",
            "weighted_centroid": "Wcentroid", "geometric_median": "Gmedian"}

macros: dict[str, str] = {}


def mac(name, value, fmt=None):
    if fmt is not None:
        value = format(value, fmt)
    macros[name] = str(value)


def load(kind):
    frames = []
    for c in CITIES:
        p = os.path.join(OUT, "eval_%s_%s.csv" % (kind, c))
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        sys.exit("missing outputs/eval_%s_*.csv" % kind)
    return pd.concat(frames, ignore_index=True)


def boot_ci(x, stat=np.mean, n=4000):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    vals = stat(x[idx], axis=1)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def fmt_int(x):
    return "{:,}".format(int(round(x))).replace(",", "{,}")


def pct(x, nd=1):
    return format(100 * x, ".%df" % nd)


# ---------------------------------------------------------------- instances
search = load("search")
equity = load("equity")
lip = load("lipschitz")
sound = load("soundness")

inst = search.drop_duplicates(["city", "seed"])
mac("NInst", len(inst))
for c in CITIES:
    mac("NInst" + c.capitalize(), int((inst.city == c).sum()))
mac("NSurfaces", len(lip))
mac("MeanCands", fmt_int(inst.n_fine.mean()))
mac("MedianCands", fmt_int(inst.n_fine.median()))
mac("MinCands", fmt_int(inst.n_fine.min()))
mac("MaxCands", fmt_int(inst.n_fine.max()))
mac("MeanFeasible", fmt_int(inst.n_feasible.mean()))
mac("MedianRegion", inst.region_km2.median(), ".0f")
unreach = 1 - inst.n_feasible.sum() / inst.n_fine.sum()
mac("PctUnreachable", pct(unreach, 0))
mac("NUsers", int(inst.n_users.iloc[0]))
mac("Departures", "; ".join("%s %s" % (CITY_NAME[c], d.split(" ")[0]) for c, d in
                            inst.groupby("city").departure.first().items()))

# ---------------------------------------------------------------- search and cost
order = [
    ("exhaustive", "Exhaustive, no screen"),
    ("exhaustive_screened", "Exhaustive after screen"),
    ("geometric", "Straight-line bound"),
    ("alt", "Landmarks (ALT)"),
    ("lipschitz_assumed", "Lipschitz, as submitted"),
    ("lipschitz_assumed_screened", "Lipschitz + screen"),
    ("lipschitz_recalibrated_screened", "Lipschitz recal.\\ + screen"),
    ("bands_K12_noscreen", "Level sets, $K{=}12$, no screen"),
    ("bands_K4", "Level sets, $K{=}4$"),
    ("bands_K12", "Level sets, $K{=}12$"),
    ("bands_K60", "Level sets, $K{=}60$"),
    ("polygon_contour", "Polygon bands, $K{=}12$"),
]
var = search[search.objective == "variance"]
g = var.groupby("method")
summary = pd.DataFrame({
    "pairs": g.pairs.mean(),
    "pairs_median": g.pairs.median(),
    "sweeps": g.sweeps.mean(),
    "iso": g.iso_requests.mean(),
    "pre": g.pre_sweeps.mean(),
    "optimal": g.optimal.mean(),
    "certified": g.certified.mean(),
    "n": g.size(),
})
summary["billed"] = summary.pairs + summary.iso
print("\nSearch variants, variance objective, mean per instance")
print(summary.round(2).to_string())

sound_by = {"lipschitz_assumed": "no", "lipschitz_assumed_screened": "no",
            "lipschitz_recalibrated_screened": "no", "polygon_contour": "no",
            "alt": "yes$^{a}$"}
lines = []
for key, label in order:
    if key not in summary.index:
        continue
    r = summary.loc[key]
    iso = "--" if r.iso == 0 else fmt_int(r.iso)
    pre = " (+%d)" % round(r.pre) if r.pre else ""
    sweeps = ("%d%s" % (round(r.sweeps), pre)) if (r.sweeps or r.pre) else "0"
    premise = sound_by.get(key, "yes")
    lines.append("%s & %s & %s & %s & %s & %s \\\\" % (
        label, fmt_int(r.pairs), sweeps, iso, pct(r.optimal, 0), premise))
with open(os.path.join(DEST, "tab_search.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")

for key, name in [("exhaustive", "Exh"), ("exhaustive_screened", "ExhScr"),
                  ("lipschitz_assumed", "LipSub"), ("lipschitz_assumed_screened", "LipScr"),
                  ("lipschitz_recalibrated_screened", "LipRecScr"),
                  ("bands_K4", "BandFour"), ("bands_K12", "BandTwelve"), ("bands_K60", "BandSixty"),
                  ("bands_K12_noscreen", "BandTwelveNoScr"), ("alt", "Alt"),
                  ("geometric", "Geo"), ("polygon_contour", "PolyContour"),
                  ("polygon_nearest", "PolyNearest"), ("polygon_contour_widened", "PolyWide"),
                  ("bands_submitted", "BandSub")]:
    if key not in summary.index:
        continue
    r = summary.loc[key]
    mac("Pairs" + name, fmt_int(r.pairs))
    mac("Billed" + name, fmt_int(r.billed))
    mac("Iso" + name, fmt_int(r.iso))
    mac("Opt" + name, pct(r.optimal, 0))
    mac("NotOpt" + name, int(round((1 - r.optimal) * r.n)))
    mac("Cert" + name, pct(r.certified, 1))
    ev = var[var.method == key].evaluated
    if ev.notna().any():
        mac("Eval" + name, ev.mean(), ".1f")

# ratios the text quotes
def ratio(a, b, col="billed"):
    return summary.loc[a, col] / summary.loc[b, col]

mac("RatioLipSubBandTwelve", ratio("lipschitz_assumed", "bands_K12"), ".0f")
mac("RatioLipScrBandTwelve", ratio("lipschitz_assumed_screened", "bands_K12"), ".1f")
mac("RatioExhExhScr", ratio("exhaustive", "exhaustive_screened", "pairs"), ".1f")
mac("RatioExhScrBandTwelve", ratio("exhaustive_screened", "bands_K12"), ".0f")
mac("RatioLipSubLipScr", ratio("lipschitz_assumed", "lipschitz_assumed_screened", "pairs"), ".1f")
mac("RatioNoScrScr", ratio("bands_K12_noscreen", "bands_K12", "pairs"), ".1f")

# the submitted configuration reproduces? compare with the CSVs of the submitted version
old = []
for c in CITIES:
    p = os.path.join(OUT, "certificate_soundness_%s.csv" % c)
    if os.path.exists(p):
        d = pd.read_csv(p)
        d["city"] = c
        old.append(d[["city", "seed", "truth"]])
if old:
    old = pd.concat(old)
    m = inst.merge(old, on=["city", "seed"], how="inner")
    same = np.isclose(m.truth, m.truth_variance, rtol=1e-6, atol=1e-6)
    mac("NReproduced", int(same.sum()))
    mac("NReproducible", len(m))
    print("\nsubmitted instances present again: %d, same optimum on %d" % (len(m), same.sum()))

# min-max: where a one-sided bound does help
mm = search[search.objective == "min_max"].groupby("method")
mmsum = pd.DataFrame({"pairs": mm.pairs.mean(), "optimal": mm.optimal.mean()})
print("\nmin-max objective\n", mmsum)
if "geometric_minmax" in mmsum.index:
    mac("PairsGeoMinmax", fmt_int(mmsum.loc["geometric_minmax", "pairs"]))
    mac("PairsBandMinmax", fmt_int(mmsum.loc["bands_K12_minmax", "pairs"]))

# ---------------------------------------------------------------- soundness of other boxes
sd = sound.groupby("construction").agg(pairs=("pairs_checked", "sum"),
                                       viol=("violations", "sum"),
                                       mis=("misbanded", "sum"),
                                       fexcl=("feasible_excluded", "sum"),
                                       optexcl=("optimum_excluded", "sum"),
                                       n=("seed", "size"))
print("\nbox soundness\n", sd)
feas_total = inst.n_feasible.sum()
for key, name in [("polygon_nearest", "PolyNearest"), ("polygon_contour", "PolyContour"),
                  ("polygon_contour_widened", "PolyWide"), ("alt", "Alt")]:
    if key not in sd.index:
        continue
    r = sd.loc[key]
    mac("Viol" + name, pct(r.viol / r.pairs, 2))
    mac("ViolCount" + name, fmt_int(r.viol))
    if np.isfinite(r.mis):
        mac("Mis" + name, pct(r.mis / r.pairs, 1))
    mac("FeasExcl" + name, pct(r.fexcl / feas_total, 1))
    mac("OptExcl" + name, int(r.optexcl))

# ---------------------------------------------------------------- Lipschitz audit
lip["city_name"] = lip.city.map(CITY_NAME)
modes = ["transit", "cycling", "driving", "walking"]
lines = []
for mode in modes:
    d = lip[lip["mode"] == mode]
    if d.empty:
        continue
    L = d.L_assumed.iloc[0]
    def wmean(col, ncol):
        w = d[ncol].fillna(0)
        v = d[col]
        ok = v.notna() & (w > 0)
        return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
    far_raw = wmean("b1.2_2_viol", "b1.2_2_n")
    far_rob = wmean("b1.2_2_viol_robust", "b1.2_2_n")
    near_rob = wmean("b0_0.45_viol_robust", "b0_0.45_n")
    rec = d.emp_over_assumed.median()
    rec_rob = wmean("emp_viol_close_robust", "close_n")
    lines.append("%s & %d & %.2f & %.1f & %.1f & %.1f & %.1f & %.1f \\\\" % (
        mode.capitalize(), len(d), L, near_rob, far_raw, far_rob, rec, rec_rob))
    mac("LipFarRob" + mode.capitalize(), far_rob, ".1f")
    mac("LipRecRob" + mode.capitalize(), rec_rob, ".1f")
    mac("LipRecFactor" + mode.capitalize(), rec, ".1f")
    mac("LipSurf" + mode.capitalize(), len(d))
with open(os.path.join(DEST, "tab_lipschitz.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")

surf_viol = lip["b1.2_2_viol_robust"]
mac("SurfAnyViolFar", int((surf_viol > 0).sum()))
mac("SurfWithFar", int(surf_viol.notna().sum()))
mac("SurfViolFarMedian", surf_viol.median(), ".1f")
mac("SurfViolFarQOne", surf_viol.quantile(0.25), ".1f")
mac("SurfViolFarQThree", surf_viol.quantile(0.75), ".1f")
mac("SurfTenPctFar", int((surf_viol >= 10).sum()))
for c in CITIES:
    d = lip[lip.city == c]
    if len(d):
        mac("LipFarRobCity" + c.capitalize(),
            float((d["b1.2_2_viol_robust"] * d["b1.2_2_n"]).sum() / d["b1.2_2_n"].sum()), ".1f")
mac("CloseDtMax", lip.close_dt_max.max(), ".0f")
jump = (lip.close_jump_pct * lip.close_n).sum() / lip.close_n.sum()
mac("CloseJumpPct", jump, ".0f")
mac("CloseDtMedian", lip.close_dt_median.median(), ".0f")
road = lip[lip["mode"] != "transit"]
mac("LipFarRobRoad", float((road["b1.2_2_viol_robust"] * road["b1.2_2_n"]).sum()
                           / road["b1.2_2_n"].sum()), ".1f")
print("\nLipschitz audit per mode (near robust, far raw, far robust, recal factor, recal robust)")
print("\n".join(lines))

# ---------------------------------------------------------------- wall clock
t = search.drop_duplicates(["city", "seed"])
if "t_full_sweep_s" in t:
    tb, tl, ts = t.t_full_sweep_s.mean(), t.t_lip_touched_s.mean(), t.t_lip_screened_touched_s.mean()
    mac("TimeSweep", tb, ".2f")
    mac("TimeLip", tl, ".2f")
    mac("TimeLipScr", ts, ".2f")
    mac("TimeRatio", tb / tl, ".2f")
    mac("TimeRatioScr", tb / ts, ".2f")
    per_city = t.groupby("city").apply(lambda d: d.t_full_sweep_s.mean() / d.t_lip_touched_s.mean())
    mac("TimeRatioMin", per_city.min(), ".2f")
    mac("TimeRatioMax", per_city.max(), ".2f")
    print("\nwall clock: sweep %.2fs  lipschitz touched %.2fs  screened %.2fs" % (tb, tl, ts))
    print(per_city)

# ---------------------------------------------------------------- equity
rules = [("variance_opt", "Variance (ours)"), ("ede_opt", "Kolm-Pollak EDE"),
         ("min_range", "Min-range"), ("min_max", "Min-max"), ("min_sum", "Min-sum"),
         ("random_100", "Random, 100 samples"), ("iso_intersection", "Isochrone intersection"),
         ("centroid", "Centroid"), ("weighted_centroid", "Weighted centroid"),
         ("geometric_median", "Geometric median")]
eq = equity.copy()
eq["key"] = eq.city + ":" + eq.seed.astype(str)
feas = eq.pivot_table(index="key", columns="method", values="feasible", aggfunc="first")
common = feas.index[feas.fillna(False).astype(bool).all(axis=1)]
mac("NEquityCommon", len(common))
lines = []
metrics_cols = ["variance", "range", "gini", "jain", "ede", "max", "mean"]
for key, label in rules:
    d = eq[eq.method == key]
    infeas = 1 - d.feasible.astype(bool).mean()
    dc = d[d.key.isin(common)]
    vals = [dc[c].mean() for c in metrics_cols]
    lines.append("%s & %.0f & %.1f & %.3f & %.3f & %.1f & %.1f & %.1f & %s \\\\" % (
        label, *vals, pct(infeas, 1)))
    mac("Infeas" + RULE_TAG[key], pct(infeas, 1))
with open(os.path.join(DEST, "tab_equity.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nequity on %d instances where every rule is feasible" % len(common))
print("\n".join(lines))

piv = eq[eq.key.isin(common)].pivot_table(index="key", columns="method", values="variance")
tests = []
for key, _label in rules[1:]:
    a, b = piv["variance_opt"], piv[key]
    diff = b - a
    rel = (diff / b.replace(0, np.nan)).dropna()
    if (diff != 0).sum() == 0:
        p = 1.0
    else:
        p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="less").pvalue
    tests.append((key, p, float(np.median(rel)), boot_ci(rel, np.median), float((diff > 0).mean())))
ps = sorted([(p, k) for k, p, *_ in tests])
holm = {}
m = len(ps)
running = 0.0
for i, (p, k) in enumerate(ps):
    running = max(running, min(1.0, (m - i) * p))
    holm[k] = running
print("\nvariance-optimal against each rule, paired, %d instances" % len(common))
for k, p, med, ci, frac in tests:
    print("  %-18s median reduction %5.1f%% [%5.1f, %5.1f]  lower on %5.1f%%  Holm p=%.2e"
          % (k, 100 * med, 100 * ci[0], 100 * ci[1], 100 * frac, holm[k]))
    name = RULE_TAG[k]
    mac("Red" + name, pct(med, 0))
    mac("RedLo" + name, pct(ci[0], 0))
    mac("RedHi" + name, pct(ci[1], 0))
    mac("Holm" + name, ("%.0e" % holm[k]).replace("e-0", "e-").replace("e-", "\\times10^{-") + "}"
        if holm[k] < 0.001 else "%.3f" % holm[k])
mac("MaxHolmNonRandom", max(holm[k] for k in holm if k != "random_100"), ".1e")

# ---------------------------------------------------------------- discretisation (Euclidean)
tess = pd.read_csv(os.path.join(OUT, "tessellation.csv"))
tess["inst"] = tess.city + ":" + tess.n.astype(str) + ":" + tess.seed.astype(str)
names = {"poisson": "Poisson-disk (control)", "trunc_square": "Truncated square",
         "h3": "Hexagonal", "triangle": "Triangular", "octile": "Octile", "square": "Square"}
agg = tess.groupby("pattern").agg(total=("total_gap_pct", "mean"), disc=("quant_gap_pct", "mean"),
                                  heur=("seam_gap_pct", "mean"),
                                  miss=("seam_gap_pct", lambda s: 100 * (s > 1e-6).mean()),
                                  cert=("certified_gap_pct", "mean"),
                                  cert_calls=("certified_calls", "mean"),
                                  exh_calls=("exhaustive_calls", "mean"),
                                  heur_calls=("ours_calls", "mean"),
                                  n=("inst", "nunique")).sort_values("disc")
lines = []
for pat, r in agg.iterrows():
    lo, hi = boot_ci(tess[tess.pattern == pat].quant_gap_pct)
    lines.append("%s & %.2f\\,[%.2f, %.2f] & %.2f & %.1f \\\\" % (
        names[pat], r.disc, lo, hi, r.heur, r.miss))
with open(os.path.join(DEST, "tab_tess.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
wide = tess.pivot_table(index="inst", columns="pattern", values="quant_gap_pct").dropna()
fr = stats.friedmanchisquare(*[wide[c] for c in wide.columns])
mac("TessN", len(wide))
mac("TessFriedmanP", fr.pvalue, ".2g")
mac("TessFriedmanChi", fr.statistic, ".1f")
spread = agg.disc.max() - agg.disc.min()
mac("TessSpread", spread, ".2f")
best, worst = agg.disc.idxmin(), agg.disc.idxmax()
dd = (wide[worst] - wide[best]).values
lo, hi = boot_ci(dd)
mac("TessSpreadLo", lo, ".2f")
mac("TessSpreadHi", hi, ".2f")
mac("TessBest", names[best])
mac("TessWorst", names[worst].lower())
mac("TessCertCalls", fmt_int(agg.cert_calls.mean()))
mac("TessExhCalls", fmt_int(agg.exh_calls.mean()))
mac("TessHeurCalls", fmt_int(agg.heur_calls.mean()))
mac("TessMissMin", agg.miss.min(), ".1f")
mac("TessMissMax", agg.miss.max(), ".1f")
mac("TessCertEqDisc", "yes" if np.allclose(tess.certified_gap_pct, tess.quant_gap_pct, atol=1e-3)
    else "no")
print("\ndiscretisation\n", agg.round(3))
print("Friedman chi2=%.2f p=%.3g on %d instances; best-worst spread %.3f [%.3f, %.3f]"
      % (fr.statistic, fr.pvalue, len(wide), spread, lo, hi))

# ---------------------------------------------------------------- write macros
bad = [k for k in macros if not k.isalpha()]
if bad:
    sys.exit("macro names must be letters only: %s" % bad)
with open(os.path.join(DEST, "numbers.tex"), "w") as f:
    f.write("% generated by scripts/make_paper_numbers.py; do not edit by hand\n")
    for k in sorted(macros):
        f.write("\\newcommand{\\%s}{%s}\n" % (k, macros[k]))
print("\nwrote %d macros and tables to %s" % (len(macros), DEST))
