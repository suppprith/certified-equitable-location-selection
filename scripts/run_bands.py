
from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fairmp import metrics
from fairmp.algorithm import Params, fair_meeting_point
from fairmp.bands import (FieldOracle, band_certificate_report, iso_int, make_thresholds)
from fairmp.baselines import exhaustive_variance, min_max
from fairmp.candidates import region_polygon
from fairmp.certificate import certificate_report, euclidean_lipschitz
from fairmp.geo import centroid
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.tessellation import make_tessellation
from fairmp.travel_time import CachedEvaluator, EuclideanBackend

COARSE, FINE = 8, 9
KS = [10, 20, 40, 80]
CITIES = ["london", "bengaluru"]
NS = [5, 6, 7, 8]
SEEDS = range(20)


def instance(city, n, seed, backend):
    origins = sample_origins(city, n, seed=seed, spread="clustered",
                             clusters=2, cluster_sd_deg=0.04)
    modes = assign_modes(n, "mixed", seed=seed)
    return origins, modes


def run_one(origins, modes, backend):
    n = len(origins)
    p = Params(coarse_res=COARSE, fine_res=FINE, k_c=400, k_refine=10, t_max=240.0)
    tess = make_tessellation("h3", centroid(origins))
    lip = euclidean_lipschitz(modes)

    ev = CachedEvaluator(backend)
    best, _r, scored, _d = fair_meeting_point(origins, modes, ev, p, tess=tess)
    if best is None:
        return []
    ours_v = metrics.variance(best.times)

    ev_x = CachedEvaluator(backend)
    xpt = exhaustive_variance(origins, modes, ev_x, res=FINE, tess=tess)
    xtimes = [ev_x.effective(o, xpt, m) for o, m in zip(origins, modes)]
    xvar = metrics.variance(xtimes)

    # the Lipschitz certificate as it stands today
    ev_l = CachedEvaluator(backend)
    ev_l._cache = dict(ev._cache)
    rep_lip = certificate_report(origins, modes, ev_l, p, best, scored, lip, tess=tess)

    pts = [pt for _c, pt in tess.cells_in(region_polygon(origins), FINE)]
    oracle = FieldOracle(backend)
    fields = oracle.fields(origins, modes, pts)
    t_hi = max(max(f for f in fi if math.isfinite(f)) for fi in fields)

    rows = []
    base = {"n": n, "ours_variance": round(ours_v, 4), "exhaustive_variance": round(xvar, 4),
            "opt_gap_pct": round(100 * (ours_v - xvar) / xvar, 4) if xvar > 1e-12 else 0.0,
            "lip_certified": rep_lip["certified"], "t_hi": round(t_hi, 2)}

    for k in KS:
        thr = make_thresholds(0.0, t_hi, k)
        for mode in ("band", "hybrid"):
            ev_b = CachedEvaluator(backend)
            ev_b._cache = dict(ev._cache)
            rep = band_certificate_report(origins, modes, ev_b, p, best, scored, thr,
                                          backend, lipschitz=lip, mode=mode, tess=tess)
            rows.append(dict(base, k=k, mode=mode,
                             certified=bool(rep.get("certified")),
                             failed=rep.get("failed", -1),
                             checked=rep.get("checked", -1),
                             sweeps=rep.get("sweeps", -1),
                             band_width=round(t_hi / k, 3)))

    # ISO-INT head to head, at the coarsest band resolution industry would use
    iso_rows = []
    for k in KS:
        thr = make_thresholds(0.0, t_hi, k)
        r = iso_int(origins, modes, backend, thr, pts, oracle=FieldOracle(backend))
        if r is None:
            continue
        ev_m = CachedEvaluator(backend)
        mpt = min_max(origins, modes, ev_m, res=FINE, tess=tess)
        mtimes = [ev_m.effective(o, mpt, m) for o, m in zip(origins, modes)]
        iso_rows.append({
            "n": n, "k": k,
            "iso_centroid_var": round(metrics.variance(r["centroid_times"]), 4),
            "iso_oracle_var": round(metrics.variance(r["oracle_times"]), 4),
            "iso_centroid_max": round(metrics.max_time(r["centroid_times"]), 4),
            "iso_oracle_max": round(metrics.max_time(r["oracle_times"]), 4),
            "iso_centroid_jain": round(metrics.jain(r["centroid_times"]), 4),
            "minmax_var": round(metrics.variance(mtimes), 4),
            "minmax_max": round(metrics.max_time(mtimes), 4),
            "ours_var": round(ours_v, 4),
            "ours_max": round(metrics.max_time(best.times), 4),
            "ours_jain": round(metrics.jain(best.times), 4),
            "region_size": r["region_size"],
            "t_star": round(r["t_star"], 3),
            "sweeps": r["sweeps"],
        })
    return rows, iso_rows


def main():
    os.makedirs("outputs", exist_ok=True)
    backend = EuclideanBackend()
    cert_rows, iso_rows = [], []
    t0 = time.time()
    done = 0
    for city in CITIES:
        for n in NS:
            for seed in SEEDS:
                origins, modes = instance(city, n, seed, backend)
                out = run_one(origins, modes, backend)
                if not out:
                    continue
                cr, ir = out
                for r in cr:
                    r.update(city=city, seed=seed)
                for r in ir:
                    r.update(city=city, seed=seed)
                cert_rows += cr
                iso_rows += ir
                done += 1
                if done % 10 == 0:
                    print("  %d instances, %.0fs" % (done, time.time() - t0), flush=True)

    cdf = pd.DataFrame(cert_rows)
    idf = pd.DataFrame(iso_rows)
    cdf.to_csv("outputs/bands_certificate.csv", index=False)
    idf.to_csv("outputs/bands_isoint.csv", index=False)

    lip_rate = 100 * cdf.drop_duplicates(["city", "seed", "n"]).lip_certified.mean()
    print("\nCertificate bound comparison, Euclidean backend")
    print("(Euclidean is the case most favourable to Lipschitz: the constant is exactly")
    print(" tight there, so any band result at parity is a lower bound on its real-network value)")
    print("\n  Lipschitz box, fixed budget: %.1f%% certified" % lip_rate)
    agg = cdf.groupby(["mode", "k"]).agg(
        certified_pct=("certified", lambda s: 100.0 * s.mean()),
        mean_failed=("failed", "mean"),
        parents=("checked", "mean"),
        sweeps=("sweeps", "mean"),
        band_width_min=("band_width", "mean"),
    ).round(2)
    print("\n%s" % agg.to_string())

    print("\nISO-INT vs min_max vs ours (mean over instances)")
    iagg = idf.groupby("k").agg(
        iso_centroid_var=("iso_centroid_var", "mean"),
        iso_oracle_var=("iso_oracle_var", "mean"),
        minmax_var=("minmax_var", "mean"),
        ours_var=("ours_var", "mean"),
        iso_oracle_max=("iso_oracle_max", "mean"),
        minmax_max=("minmax_max", "mean"),
        ours_max=("ours_max", "mean"),
        region=("region_size", "mean"),
    ).round(2)
    print("\n%s" % iagg.to_string())

    beat_c = 100 * (idf.iso_centroid_var < idf.ours_var - 1e-9).mean()
    beat_o = 100 * (idf.iso_oracle_var < idf.ours_var - 1e-9).mean()
    excl = 100 * (idf.iso_oracle_var > idf.ours_var + 1e-9).mean()
    infl = (idf.iso_centroid_var / idf.ours_var.clip(lower=1e-9)).mean()
    print("\n  iso_centroid = the only selection an isochrone product actually supports:")
    print("                 the polygon intersection returns a region, not per-point times")
    print("  iso_oracle   = same region, ranked with our per-point times (not purchasable)")
    print("\n  ISO-INT centroid beats our variance on %.1f%% of instances" % beat_c)
    print("  ISO-INT centroid variance is %.2fx ours on average" % infl)
    print("  ISO-INT oracle-assisted beats our variance on %.1f%% of instances" % beat_o)
    print("  the intersection region excludes our point on %.1f%% of instances" % excl)
    print("\nwrote outputs/bands_certificate.csv and outputs/bands_isoint.csv")


if __name__ == "__main__":
    main()
