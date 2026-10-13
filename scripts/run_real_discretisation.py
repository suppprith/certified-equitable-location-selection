"""Discretisation gap on the real networks.

The grid ablation measures how far the optimum over the candidate set lies from the optimum
on a grid seven times finer, but on the straight-line backend. This measures the same gap on
the real router, for the instances of run_real_evaluation.py: the variance optimum over the
H3 resolution-9 candidates against the optimum over resolution 10, whose cells are a seventh
the area and whose centre children sit on the resolution-9 centres. Writes
outputs/eval_discretisation_<city>.csv.
"""
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
from fairmp.candidates import polyfill_centroids, region_polygon
from fairmp.geo import centroid, haversine_km
from fairmp.scenarios import ROAD_MODES, assign_modes, sample_origins
from fairmp.tessellation import make_tessellation
from fairmp.travel_time import R5Backend

from cities import apply_sample_bbox
from run_real_evaluation import (CITY, COARSE, DEPARTURE, FINE, GTFS, N_USERS, OSM, SEEDS,
                                 SPREAD, matrix)

REF = FINE + 1


def optimum(S, n):
    feas = np.all(np.isfinite(S), axis=0)
    if not feas.any():
        return math.inf, None
    vals = np.full(S.shape[1], np.inf)
    for j in np.flatnonzero(feas):
        vals[j] = metrics.variance(list(S[:, j]))
    j = int(np.argmin(vals))
    return float(vals[j]), j


def run_instance(r5, seed, departure, rows):
    # same instance generator and filters as run_real_evaluation.py
    origins = sample_origins(CITY, N_USERS, seed=seed, spread="clustered",
                             clusters=2, cluster_sd_deg=SPREAD)
    modes = assign_modes(N_USERS, "mixed", seed=seed, pool=None if GTFS else ROAD_MODES)
    tess = make_tessellation("h3", centroid(origins))
    poly = region_polygon(origins)
    pts = {}
    for res in (COARSE, FINE):
        for _c, p in polyfill_centroids(poly, res):
            pts[(round(p.lat, 6), round(p.lng, 6))] = p
    if len(pts) < 80:
        return
    fine = [p for _c, p in tess.cells_in(poly, FINE)]
    ref = [p for _c, p in polyfill_centroids(poly, REF)]
    dests = fine + ref
    S = np.full((N_USERS, len(dests)), np.inf)
    by_mode = {}
    for i, m in enumerate(modes):
        by_mode.setdefault(m[0], []).append(i)
    for mode, users in by_mode.items():
        M = matrix(r5, [origins[i] for i in users], dests, mode, departure)
        for row, i in enumerate(users):
            S[i] = M[row]
    v9, j9 = optimum(S[:, :len(fine)], N_USERS)
    if not math.isfinite(v9):
        return
    v10, j10 = optimum(S[:, len(fine):], N_USERS)
    rows.append({
        "city": CITY, "seed": seed, "n_fine": len(fine), "n_ref": len(ref),
        "opt_res9": v9, "opt_res10": v10,
        "gap_pct": 100 * (v9 - v10) / v10 if v10 > 1e-12 else 0.0,
        "km_between_optima": haversine_km(fine[j9], ref[j10]) if j10 is not None else math.nan,
    })
    print("  seed %2d res9 %8.2f res10 %8.2f gap %6.2f%%" % (seed, v9, v10, rows[-1]["gap_pct"]),
          flush=True)


def main():
    apply_sample_bbox(CITY)
    departure = pd.Timestamp(DEPARTURE)
    print("loading %s network (departure %s)..." % (CITY, DEPARTURE), flush=True)
    t0 = time.time()
    r5 = R5Backend(OSM, GTFS)
    print("  ready in %.0fs" % (time.time() - t0), flush=True)
    rows = []
    for seed in SEEDS:
        run_instance(r5, seed, departure, rows)
    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/eval_discretisation_%s.csv" % CITY, index=False)
    if len(df):
        print("\n%s: %d instances, gap median %.2f%%, p90 %.2f%%, max %.2f%%"
              % (CITY, len(df), df.gap_pct.median(), df.gap_pct.quantile(0.9), df.gap_pct.max()))


if __name__ == "__main__":
    main()
