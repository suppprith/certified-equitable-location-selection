
from __future__ import annotations

import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import pandas as pd

from fairmp import metrics
from fairmp import objectives as ob
from fairmp.algorithm import Params
from fairmp.bands import band_certified_search, make_thresholds
from fairmp.candidates import polyfill_centroids, region_polygon
from fairmp.certificate import certified_search, euclidean_lipschitz
from fairmp.geo import centroid
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.tessellation import make_tessellation
from fairmp.travel_time import CachedEvaluator, R5Backend

from run_real_bands import (COARSE, DEPARTURE, FINE, GTFS, OSM, empirical_lipschitz,
                            precompute)

K_BANDS = int(os.environ.get("K_BANDS", "60"))
SEEDS = range(int(os.environ.get("N_INSTANCES", "12")))
N_USERS = int(os.environ.get("N_USERS", "5"))
PARAMS = Params(coarse_res=COARSE, fine_res=FINE, k_c=400, k_refine=10, ring=1, t_max=240.0)


def main():
    departure = pd.Timestamp(DEPARTURE)
    print("loading London network...", flush=True)
    t0 = time.time()
    r5 = R5Backend(OSM, GTFS)
    print("  ready in %.0fs" % (time.time() - t0), flush=True)

    rows = []
    for seed in SEEDS:
        origins = sample_origins("london", N_USERS, seed=seed, spread="clustered",
                                 clusters=2, cluster_sd_deg=0.04)
        modes = assign_modes(N_USERS, "mixed", seed=seed)
        tess = make_tessellation("h3", centroid(origins))
        poly = region_polygon(origins)

        pts = {}
        for res in (COARSE, FINE):
            for _c, p in polyfill_centroids(poly, res):
                pts[(round(p.lat, 6), round(p.lng, 6))] = p
        cands = list(pts.values())
        if len(cands) < 80:
            continue

        pre, surfaces = precompute(r5, origins, modes, cands, departure)
        all_t = [v for s in surfaces.values() for v in s.values() if math.isfinite(v)]
        if not all_t:
            continue
        thr = make_thresholds(0.0, max(all_t), K_BANDS)

        # ground truth over the fine candidate set
        ev0 = CachedEvaluator(pre)
        truth = math.inf
        for _c, pt in tess.cells_in(poly, FINE):
            ts = [ev0.effective(o, pt, m) for o, m in zip(origins, modes)]
            if metrics.all_reachable(ts, len(origins)):
                truth = min(truth, metrics.variance(ts))
        if not math.isfinite(truth):
            continue

        row = {"seed": seed, "truth": round(truth, 4), "candidates": len(cands)}

        # adaptive certified search under each Lipschitz constant. This is where an
        # unsound bound does damage: it prunes a parent that actually contains the optimum
        # and then reports the run as certified.
        for name, lip in (("assumed", euclidean_lipschitz(modes)),
                          ("empirical", empirical_lipschitz(surfaces, cands, len(origins)))):
            ev = CachedEvaluator(pre)
            _best, cert, _d = certified_search(origins, modes, ev, PARAMS, lip, tess=tess)
            v = cert.get("v_star", math.inf)
            row["lip_%s_value" % name] = round(v, 4)
            row["lip_%s_certified" % name] = bool(cert.get("certified"))
            row["lip_%s_excess_pct" % name] = (round(100 * (v - truth) / truth, 4)
                                               if truth > 1e-12 else 0.0)
            # certified, yet strictly worse than the true optimum over the same candidates
            row["lip_%s_UNSOUND" % name] = bool(cert.get("certified") and v > truth + 1e-9)
            row["lip_%s_calls" % name] = cert.get("routing_calls", -1)

        ev_b = CachedEvaluator(pre)
        inc, info = band_certified_search(origins, modes, ev_b, PARAMS, thr, pre,
                                          objective="variance", tess=tess)
        v = info.get("value", math.inf)
        row["band_value"] = round(v, 4) if math.isfinite(v) else float("nan")
        row["band_certified"] = bool(info.get("certified"))
        row["band_excess_pct"] = round(100 * (v - truth) / truth, 4) if truth > 1e-12 else 0.0
        row["band_UNSOUND"] = bool(info.get("certified") and v > truth + 1e-9)
        row["band_evaluated"] = info.get("evaluated", -1)
        row["band_eval_frac"] = round(info.get("eval_fraction", float("nan")), 4)
        row["band_sweeps"] = info.get("sweeps", -1)
        row["band_point_queries"] = info.get("point_queries", -1)

        rows.append(row)
        print("  seed %2d truth=%9.3f | assumed %9.3f %+8.3f%% unsound=%-5s | "
              "empirical %9.3f %+8.3f%% unsound=%-5s | band %9.3f %+8.3f%% unsound=%-5s "
              "eval=%5.1f%%"
              % (seed, truth,
                 row["lip_assumed_value"], row["lip_assumed_excess_pct"], row["lip_assumed_UNSOUND"],
                 row["lip_empirical_value"], row["lip_empirical_excess_pct"], row["lip_empirical_UNSOUND"],
                 row["band_value"], row["band_excess_pct"], row["band_UNSOUND"],
                 100 * row["band_eval_frac"]), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/certificate_soundness.csv", index=False)

    print("\nAdaptive certified search on real London multimodal surfaces, %d instances" % len(df))
    print("UNSOUND = the run reported itself certified while returning a point strictly")
    print("worse than the true optimum over the same candidate set.\n")
    for name in ("lip_assumed", "lip_empirical", "band"):
        c = df["%s_certified" % name]
        u = df["%s_UNSOUND" % name]
        e = df["%s_excess_pct" % name]
        print("  %-14s certified %5.1f%%   UNSOUND %5.1f%% (%d)   mean excess over optimum %+7.3f%%"
              % (name, 100 * c.mean(), 100 * u.mean(), int(u.sum()), e.mean()))
    if "band_eval_frac" in df:
        print("\n  band search evaluated %.1f%% of candidates on average (%.0f sweeps, %.0f point queries)"
              % (100 * df.band_eval_frac.mean(), df.band_sweeps.mean(),
                 df.band_point_queries.mean()))
    print("\nwrote outputs/certificate_soundness.csv")


if __name__ == "__main__":
    main()
