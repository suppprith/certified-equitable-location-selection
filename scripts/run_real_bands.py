
from __future__ import annotations

import math
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from fairmp import metrics
from fairmp.algorithm import Params, fair_meeting_point
from fairmp.bands import band_certificate_report, make_thresholds
from fairmp.baselines import exhaustive_variance
from fairmp.candidates import polyfill_centroids, region_polygon
from fairmp.certificate import certificate_report, euclidean_lipschitz
from fairmp.geo import centroid, haversine_km
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.tessellation import make_tessellation
from fairmp.travel_time import R5_MODE, CachedEvaluator, PrecomputedBackend, R5Backend

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from cities import paths as city_paths, pick_departure

CITY = os.environ.get("CITY", "london")
OSM, GTFS = city_paths(CITY)
# Detected from the feeds rather than hard-coded: a departure outside the validity
# window silently yields no transit service and falls back to walking.
DEPARTURE = pick_departure(CITY)

COARSE, FINE = 8, 9
K_BANDS = 40
SEEDS = range(10)
N_USERS = 5

# Default parameters find the grid optimum on nearly every instance, which makes any
# certificate vacuously correct. The starved setting deliberately makes the search miss,
# which is the only regime where a certificate can be caught certifying a wrong answer.
DEFAULT = Params(coarse_res=COARSE, fine_res=FINE, k_c=400, k_refine=10, ring=1, t_max=240.0)
STARVED = Params(coarse_res=6, fine_res=FINE, k_c=400, k_refine=1, ring=1, t_max=240.0)


def precompute(r5, origins, modes, cands, departure):
    r5py = r5._r5py
    pre = PrecomputedBackend()
    surfaces = {}
    dest = gpd.GeoDataFrame({"id": [f"c{i}" for i in range(len(cands))]},
                            geometry=[Point(p.lng, p.lat) for p in cands], crs="EPSG:4326")
    by_mode = defaultdict(list)
    for i, (o, m) in enumerate(zip(origins, modes)):
        by_mode[m[0]].append((i, o))
    for mode, users in by_mode.items():
        tmodes = [getattr(r5py.TransportMode, x) for x in R5_MODE[mode]]
        src = gpd.GeoDataFrame({"id": [f"o{i}" for i, _ in users]},
                               geometry=[Point(o.lng, o.lat) for _, o in users], crs="EPSG:4326")
        ttm = r5py.TravelTimeMatrix(r5.network, origins=src, destinations=dest,
                                    departure=departure, transport_modes=tmodes)
        df = ttm.compute_travel_times() if hasattr(ttm, "compute_travel_times") else ttm
        omap = {f"o{i}": (i, o) for i, o in users}
        for r in df.itertuples(index=False):
            ent = omap.get(r.from_id)
            if ent is None:
                continue
            i, o = ent
            try:
                j = int(str(r.to_id)[1:])
            except ValueError:
                continue
            t = r.travel_time
            v = float(t) if t == t else math.inf
            pre.put(mode, o, cands[j], v)
            surfaces.setdefault(i, {})[j] = v
    return pre, surfaces


def empirical_lipschitz(surfaces, cands, n_users, quantile=99.0):
    """Per-user constant estimated from the surface itself: the high quantile of the
    observed slope over candidate pairs. This is the honest version of the constant the
    certificate needs, and the question is whether it is enough."""
    lat = np.array([p.lat for p in cands])
    lng = np.array([p.lng for p in cands])
    lat0 = float(lat.mean())
    x = (lng - float(lng.mean())) * 111.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 111.0
    D = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(D, np.inf)
    knn = np.argsort(D, axis=1)[:, :8]

    out = []
    for i in range(n_users):
        s = surfaces.get(i, {})
        slopes = []
        for a in range(len(cands)):
            ta = s.get(a, math.inf)
            if not math.isfinite(ta):
                continue
            for b in knn[a]:
                b = int(b)
                tb = s.get(b, math.inf)
                if not math.isfinite(tb):
                    continue
                d = float(D[a, b])
                if d > 1e-6:
                    slopes.append(abs(ta - tb) / d)
        out.append(float(np.percentile(slopes, quantile)) if slopes else 1e6)
    return out


def run_instance(r5, seed, departure, rows):
    origins = sample_origins(CITY, N_USERS, seed=seed, spread="clustered",
                             clusters=2, cluster_sd_deg=0.04)
    modes = assign_modes(N_USERS, "mixed", seed=seed)
    tess = make_tessellation("h3", centroid(origins))
    poly = region_polygon(origins)

    pts = {}
    for res in (6, COARSE, FINE):
        for _c, p in polyfill_centroids(poly, res):
            pts[(round(p.lat, 6), round(p.lng, 6))] = p
    cands = list(pts.values())
    if len(cands) < 80:
        return

    pre, surfaces = precompute(r5, origins, modes, cands, departure)
    L_assumed = euclidean_lipschitz(modes)
    L_emp = empirical_lipschitz(surfaces, cands, len(origins))

    all_t = [v for s in surfaces.values() for v in s.values() if math.isfinite(v)]
    if not all_t:
        return
    thr = make_thresholds(0.0, max(all_t), K_BANDS)

    for label, params in (("default", DEFAULT), ("starved", STARVED)):
        ev = CachedEvaluator(pre)
        best, _r, scored, _d = fair_meeting_point(origins, modes, ev, params, tess=tess)
        if best is None:
            continue
        ours = metrics.variance(best.times)

        ev_x = CachedEvaluator(pre)
        xpt = exhaustive_variance(origins, modes, ev_x, res=FINE, tess=tess)
        if xpt is None:
            continue
        xvar = metrics.variance([ev_x.effective(o, xpt, m) for o, m in zip(origins, modes)])
        optimal = ours <= xvar + 1e-9

        out = {"seed": seed, "params": label, "ours": round(ours, 4),
               "grid_opt": round(xvar, 4), "found_optimum": bool(optimal),
               "gap_pct": round(100 * (ours - xvar) / xvar, 4) if xvar > 1e-12 else 0.0}

        for cname, lip in (("lip_assumed", L_assumed), ("lip_empirical", L_emp)):
            ev_c = CachedEvaluator(pre)
            rep = certificate_report(origins, modes, ev_c, params, best, scored, lip, tess=tess)
            out[cname] = bool(rep.get("certified"))
            # the number that matters: certified, but the incumbent is not the grid optimum
            out[cname + "_wrong"] = bool(rep.get("certified") and not optimal)

        ev_b = CachedEvaluator(pre)
        rep_b = band_certificate_report(origins, modes, ev_b, params, best, scored, thr,
                                        pre, lipschitz=None, mode="band", tess=tess)
        out["band"] = bool(rep_b.get("certified"))
        out["band_wrong"] = bool(rep_b.get("certified") and not optimal)
        out["band_sweeps"] = rep_b.get("sweeps", -1)
        out["L_assumed_mean"] = round(float(np.mean(L_assumed)), 3)
        out["L_emp_mean"] = round(float(np.mean(L_emp)), 3)
        rows.append(out)
        print("  seed %2d %-8s opt=%-5s gap=%7.3f%% | assumed=%-5s emp=%-5s band=%-5s | "
              "WRONG assumed=%-5s emp=%-5s band=%-5s"
              % (seed, label, optimal, out["gap_pct"], out["lip_assumed"],
                 out["lip_empirical"], out["band"], out["lip_assumed_wrong"],
                 out["lip_empirical_wrong"], out["band_wrong"]), flush=True)


def main():
    departure = pd.Timestamp(DEPARTURE)
    print("loading London network...", flush=True)
    t0 = time.time()
    r5 = R5Backend(OSM, GTFS)
    print("  ready in %.0fs" % (time.time() - t0), flush=True)

    rows = []
    for seed in SEEDS:
        run_instance(r5, seed, departure, rows)

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/real_bands.csv", index=False)

    print("\nCertificates on real London multimodal surfaces")
    print("'wrong' means the certificate certified while the incumbent was NOT the grid")
    print("optimum. That is a demonstrated soundness failure, not a theoretical one.\n")
    for label in ("default", "starved"):
        d = df[df.params == label]
        if not len(d):
            continue
        print("%s parameters, %d instances, search found the optimum on %d" %
              (label, len(d), int(d.found_optimum.sum())))
        for c in ("lip_assumed", "lip_empirical", "band"):
            print("   %-14s certified %5.1f%%   certified-but-wrong %5.1f%%  (%d instances)"
                  % (c, 100 * d[c].mean(), 100 * d[c + "_wrong"].mean(),
                     int(d[c + "_wrong"].sum())))
        print()
    print("mean assumed L %.2f, mean empirical L %.2f"
          % (df.L_assumed_mean.mean(), df.L_emp_mean.mean()))
    print("\nwrote outputs/real_bands.csv")


if __name__ == "__main__":
    main()
