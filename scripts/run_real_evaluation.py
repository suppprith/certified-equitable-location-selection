"""Every real-network result in the paper, on one instance set.

Per instance the origins' one-to-all sweeps over all candidates are computed once and serve
as the exact oracle. On top of them the script runs every search variant with its query
count, the equity baselines on every metric, the Lipschitz audit on the same surfaces, the
soundness of polygon-derived and landmark boxes against the exact times, and the wall-clock
cost of routing on the engine itself. Four CSVs per city:

  eval_search_<city>.csv     one row per instance and search variant
  eval_equity_<city>.csv     one row per instance and location rule
  eval_lipschitz_<city>.csv  one row per instance and user surface
  eval_soundness_<city>.csv  one row per instance and box construction

Instances are generated exactly as in run_certificate_soundness.py, so seeds 0-29 reproduce
the instance set of the submitted version.
"""
from __future__ import annotations

import datetime
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from fairmp import metrics
from fairmp import objectives as ob
from fairmp.algorithm import Params
from fairmp.bands import band_certified_search, box_search, iso_int, make_thresholds
from fairmp.baselines import geometric_centroid, geometric_median, weighted_centroid
from fairmp.bounds import (Grid, alt_boxes, band_index_box, geometric_boxes, pick_landmarks,
                           polygon_band_index)
from fairmp.candidates import polyfill_centroids, region_polygon
from fairmp.certificate import certified_search, euclidean_lipschitz, screened_lipschitz_search
from fairmp.geo import centroid, haversine_km
from fairmp.scenarios import ROAD_MODES, assign_modes, sample_origins
from fairmp.tessellation import make_tessellation
from fairmp.travel_time import R5_MODE, CachedEvaluator, PrecomputedBackend, R5Backend, cell_key

from cities import apply_sample_bbox, paths as city_paths, pick_departure

CITY = os.environ.get("CITY", "london")
OSM, GTFS = city_paths(CITY)
DEPARTURE = pick_departure(CITY)

N_USERS = 5
SEEDS = range(int(os.environ.get("N_INSTANCES", "60")))
SPREAD = float(os.environ.get("SPREAD", "0.03"))
COARSE, FINE = 8, 9
CAP_MIN = 120                 # r5py's default max_time, passed explicitly
K_LIST = (4, 6, 12, 24, 60)   # thresholds evenly spaced over [0, CAP_MIN]
K_POLY = 12                   # 10-minute isochrones for the polygon experiment
GRID_KM = 0.2                 # raster the polygons are built on
CONTOURS_PER_REQUEST = 4      # Mapbox Isochrone API limit per request
PARAMS = Params(coarse_res=COARSE, fine_res=FINE, k_c=400, k_refine=10, ring=1, t_max=240.0)

# The speed-derived Lipschitz constant uses the router's own walking and cycling speeds
# (r5py defaults) so that a violation cannot come from a speed mismatch; driving and
# transit have no single router speed and keep typical urban averages.
ROUTER_SPEEDS = {"walking": 3.6, "cycling": 12.0, "driving": 25.0, "transit": 18.0}

# Lipschitz audit, same pair construction as run_lipschitz_audit.py
BANDS = [(0.0, 0.45), (0.45, 0.8), (0.8, 1.2), (1.2, 2.0)]
NEAR_K, FAR_PAIRS, CLOSE_KM, JUMP_MIN = 6, 6000, 0.45, 3.0


def gdf(pts, prefix):
    return gpd.GeoDataFrame({"id": [f"{prefix}{i}" for i in range(len(pts))]},
                            geometry=[Point(p.lng, p.lat) for p in pts], crs="EPSG:4326")


def matrix(r5, sources, dests, mode, departure):
    """Minutes from each source to each destination, inf beyond the cap."""
    r5py = r5._r5py
    tmodes = [getattr(r5py.TransportMode, x) for x in R5_MODE[mode]]
    ttm = r5py.TravelTimeMatrix(r5.network, origins=gdf(sources, "s"),
                                destinations=gdf(dests, "d"), departure=departure,
                                transport_modes=tmodes,
                                max_time=datetime.timedelta(minutes=CAP_MIN))
    df = ttm.compute_travel_times() if hasattr(ttm, "compute_travel_times") else ttm
    out = np.full((len(sources), len(dests)), np.inf)
    for r in df.itertuples(index=False):
        t = r.travel_time
        if t == t:
            out[int(str(r.from_id)[1:]), int(str(r.to_id)[1:])] = float(t)
    return out


def timed_sweeps(r5, origins, modes, dests, departure):
    t0 = time.time()
    for o, m in zip(origins, modes):
        matrix(r5, [o], dests, m[0], departure)
    return time.time() - t0


def empirical_lipschitz(F, fine, pairs_far, d_far):
    """Per-user constant at the 99th percentile of the slope over 1.2-2 km pairs."""
    out = []
    for row in F:
        a, b = pairs_far[:, 0], pairs_far[:, 1]
        ok = np.isfinite(row[a]) & np.isfinite(row[b])
        s = np.abs(row[a][ok] - row[b][ok]) / d_far[ok]
        out.append(float(np.percentile(s, 99)) if len(s) else 1e6)
    return out


def build_pairs(fine, seed):
    lat = np.array([p.lat for p in fine])
    lng = np.array([p.lng for p in fine])
    lat0 = float(lat.mean())
    x = (lng - float(lng.mean())) * 111.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 111.0
    D = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(D, np.inf)
    pairs = set()
    knn = np.argsort(D, axis=1)[:, :NEAR_K]
    for a in range(len(fine)):
        for b in knn[a]:
            pairs.add((min(a, int(b)), max(a, int(b))))
    ok = np.argwhere((D >= 0.3) & (D <= 2.0))
    if len(ok):
        rs = np.random.default_rng(seed)
        take = ok[rs.choice(len(ok), size=min(len(ok), FAR_PAIRS), replace=False)]
        for a, b in take:
            pairs.add((min(int(a), int(b)), max(int(a), int(b))))
    pairs = np.array(sorted(pairs))
    d = np.array([haversine_km(fine[a], fine[b]) for a, b in pairs])
    return pairs, d


def lipschitz_rows(F, modes, pairs, d, base):
    rows = []
    far = (d >= 1.2) & (d < 2.0)
    close = d < CLOSE_KM
    for i, row in enumerate(F):
        mode = modes[i][0]
        L = euclidean_lipschitz([[mode]], speeds=ROUTER_SPEEDS)[0]
        a, b = pairs[:, 0], pairs[:, 1]
        ok = np.isfinite(row[a]) & np.isfinite(row[b])
        if ok.sum() == 0:
            continue
        dt = np.abs(row[a] - row[b])[ok]
        dd = d[ok]
        s = dt / dd
        # Whole-minute reporting moves |dt| by at most one minute, so a pair whose gap
        # exceeds L*d even after one minute is removed cannot be a rounding artefact.
        s_rob = np.maximum(dt - 1.0, 0.0) / dd
        r = dict(base, user=i, mode=mode, L_assumed=round(L, 3), pairs=int(ok.sum()),
                 reachable_frac=round(float(np.isfinite(row).mean()), 4))
        for lo, hi in BANDS:
            m = (dd >= lo) & (dd < hi)
            key = "b%g_%g" % (lo, hi)
            r[key + "_n"] = int(m.sum())
            r[key + "_viol"] = float(100 * (s[m] > L).mean()) if m.sum() else np.nan
            r[key + "_viol_robust"] = float(100 * (s_rob[m] > L).mean()) if m.sum() else np.nan
        fm, cm = far[ok], close[ok]
        r["close_n"] = int(cm.sum())
        r["close_dt_max"] = float(dt[cm].max()) if cm.sum() else np.nan
        r["close_jump_pct"] = float(100 * (dt[cm] >= JUMP_MIN).mean()) if cm.sum() else np.nan
        r["close_dt_median"] = float(np.median(dt[cm])) if cm.sum() else np.nan
        if fm.sum() >= 20 and cm.sum() >= 20:
            L_emp = float(np.percentile(s[fm], 99))
            r["L_empirical"] = L_emp
            r["emp_over_assumed"] = L_emp / L
            r["emp_viol_close"] = float(100 * (s[cm] > L_emp).mean())
            r["emp_viol_close_robust"] = float(100 * (s_rob[cm] > L_emp).mean())
        rows.append(r)
    return rows


def exact_band(t, thresholds):
    return next((k for k, th in enumerate(thresholds[1:], start=1) if t <= th), None)


def run_instance(r5, seed, departure, out):
    origins = sample_origins(CITY, N_USERS, seed=seed, spread="clustered",
                             clusters=2, cluster_sd_deg=SPREAD)
    modes = assign_modes(N_USERS, "mixed", seed=seed, pool=None if GTFS else ROAD_MODES)
    n = N_USERS
    tess = make_tessellation("h3", centroid(origins))
    poly = region_polygon(origins)

    pts = {}
    for res in (COARSE, FINE):
        for _c, p in polyfill_centroids(poly, res):
            pts[(round(p.lat, 6), round(p.lng, 6))] = p
    cands = list(pts.values())
    if len(cands) < 80:
        return
    fine_cells = tess.cells_in(poly, FINE)
    fine = [p for _c, p in fine_cells]
    nf = len(fine)
    geo = {"centroid": geometric_centroid(origins),
           "weighted_centroid": weighted_centroid(origins, modes),
           "geometric_median": geometric_median(origins)}
    grid = Grid(poly, GRID_KM)
    nodes = grid.nodes()
    dests = cands + list(geo.values()) + nodes
    g0 = len(cands) + len(geo)

    # origin sweeps: the exact oracle, also the source of every band
    S = np.full((n, len(dests)), np.inf)
    by_mode = {}
    for i, m in enumerate(modes):
        by_mode.setdefault(m[0], []).append(i)
    for mode, users in by_mode.items():
        M = matrix(r5, [origins[i] for i in users], dests, mode, departure)
        for row, i in enumerate(users):
            S[i] = M[row]

    pre = PrecomputedBackend()
    for i, (o, m) in enumerate(zip(origins, modes)):
        for j, p in enumerate(dests[:g0]):
            pre.put(m[0], o, p, float(S[i, j]))
    idx = {(round(p.lat, 6), round(p.lng, 6)): j for j, p in enumerate(cands)}
    fidx = np.array([idx[(round(p.lat, 6), round(p.lng, 6))] for p in fine])
    F = S[:, fidx]                                   # users x fine candidates
    feas = np.all(np.isfinite(F), axis=0)
    if not feas.any():
        return
    var_all = np.array([metrics.variance(list(F[:, j])) if feas[j] else np.inf
                        for j in range(nf)])
    truth = float(var_all.min())
    feasible_key = {(round(fine[j].lat, 6), round(fine[j].lng, 6)) for j in range(nf) if feas[j]}
    for c in (pt for _c, pt in tess.cells_in(poly, COARSE)):
        j = idx.get((round(c.lat, 6), round(c.lng, 6)))
        if j is not None and np.all(np.isfinite(S[:, j])):
            feasible_key.add((round(c.lat, 6), round(c.lng, 6)))

    def feasible(pt):
        return (round(pt.lat, 6), round(pt.lng, 6)) in feasible_key

    base = {"city": CITY, "seed": seed, "departure": DEPARTURE, "n_users": n,
            "modes": "+".join(m[0] for m in modes), "n_fine": nf,
            "n_feasible": int(feas.sum()), "n_coarse": len(tess.cells_in(poly, COARSE)),
            "region_km2": round(poly.area * 111.0 * 111.0 *
                                math.cos(math.radians(centroid(origins).lat)), 3),
            "truth_variance": truth}

    def search_row(method, value, pairs, sweeps=0, iso_requests=0, pre_sweeps=0,
                   evaluated=None, certified=True, objective="variance", ref=truth, **extra):
        r = dict(base, method=method, objective=objective, value=value, pairs=int(pairs),
                 sweeps=sweeps, iso_requests=iso_requests, pre_sweeps=pre_sweeps,
                 evaluated=evaluated, certified=bool(certified),
                 optimal=bool(math.isfinite(value) and value <= ref + 1e-9),
                 excess_pct=(100 * (value - ref) / ref if ref > 1e-12 and math.isfinite(value)
                             else (0.0 if value == ref else np.nan)))
        r.update(extra)
        out["search"].append(r)

    # exhaustive, with and without the reachability screen
    search_row("exhaustive", truth, n * nf, evaluated=nf)
    # on an API the screen is each origin's outermost isochrone: one request per origin
    search_row("exhaustive_screened", truth, n * int(feas.sum()), sweeps=n, iso_requests=n,
               evaluated=int(feas.sum()))

    # Lipschitz: the adaptive search as submitted, and with the same screen
    lip_sets = {"assumed": euclidean_lipschitz(modes, speeds=ROUTER_SPEEDS)}
    pairs_ab, d_ab = build_pairs(fine, seed)
    farm = (d_ab >= 1.2) & (d_ab < 2.0)
    lip_sets["recalibrated"] = empirical_lipschitz(F, fine, pairs_ab[farm], d_ab[farm])
    touched = {}
    for name, lip in lip_sets.items():
        ev = CachedEvaluator(pre)
        _b, cert, _d = certified_search(origins, modes, ev, PARAMS, lip, tess=tess)
        v = cert.get("v_star", math.inf)
        search_row("lipschitz_%s" % name, v, ev.calls, certified=cert.get("certified", False),
                   reason=cert.get("reason", ""))
        touched["lipschitz_%s" % name] = {k[2] for k in ev._cache}
        ev = CachedEvaluator(pre)
        _c, cert = screened_lipschitz_search(origins, modes, ev, PARAMS, lip, feasible, tess=tess)
        search_row("lipschitz_%s_screened" % name, cert["v_star"], ev.calls, sweeps=n,
                   iso_requests=n, evaluated=cert["evaluated"], certified=cert["certified"],
                   anchors=cert["anchors"], anchor_pairs=cert["anchor_calls"])
        touched["lipschitz_%s_screened" % name] = {k[2] for k in ev._cache}

    # level sets at fixed thresholds over [0, cap], with and without the screen
    for K in K_LIST:
        thr = make_thresholds(0.0, float(CAP_MIN), K)
        for screen in ((True, False) if K == 12 else (True,)):
            ev = CachedEvaluator(pre)
            _inc, info = band_certified_search(origins, modes, ev, PARAMS, thr, pre, tess=tess,
                                               screen=screen)
            search_row("bands_K%d%s" % (K, "" if screen else "_noscreen"), info["value"],
                       ev.calls, sweeps=n, iso_requests=n * math.ceil(K / CONTOURS_PER_REQUEST),
                       evaluated=info["evaluated"], certified=info["certified"], K=K)
    # the submitted configuration: 60 thresholds up to the largest observed time
    all_t = S[:, :len(cands)][np.isfinite(S[:, :len(cands)])]
    thr_sub = make_thresholds(0.0, float(all_t.max()), 60)
    ev = CachedEvaluator(pre)
    _inc, info = band_certified_search(origins, modes, ev, PARAMS, thr_sub, pre, tess=tess)
    search_row("bands_submitted", info["value"], ev.calls, sweeps=n, evaluated=info["evaluated"],
               certified=info["certified"], K=60)

    def run_boxes(boxes, objective="variance"):
        ev = CachedEvaluator(pre)

        def evaluate(j):
            return [ev.effective(o, fine[j], m) for o, m in zip(origins, modes)]

        _j, val, info = box_search(boxes, evaluate, n, objective)
        return ev, val, info

    # zero-query straight-line bound: one-sided, so it cannot prune for variance
    gboxes = geometric_boxes(origins, modes, fine)
    ev, val, info = run_boxes(gboxes)
    search_row("geometric", val, ev.calls, evaluated=info["evaluated"])
    minmax_truth = float(min(F[:, j].max() for j in range(nf) if feas[j]))
    ev, val, info = run_boxes(gboxes, "min_max")
    search_row("geometric_minmax", val, ev.calls, evaluated=info["evaluated"],
               objective="min_max", ref=minmax_truth)
    ev3 = CachedEvaluator(pre)
    thr12 = make_thresholds(0.0, float(CAP_MIN), 12)
    _inc, info3 = band_certified_search(origins, modes, ev3, PARAMS, thr12, pre, tess=tess,
                                        objective="min_max")
    search_row("bands_K12_minmax", info3["value"], ev3.calls, sweeps=n,
               iso_requests=n * math.ceil(12 / CONTOURS_PER_REQUEST),
               evaluated=info3["evaluated"], objective="min_max", ref=minmax_truth, K=12)

    # landmarks: sweeps from four region corners per street mode, reusable across queries
    lm = pick_landmarks(fine)
    street = sorted({m[0] for m in modes if m[0] != "transit"})
    from_l, l_to_o = {}, {}
    for mode in street:
        M = matrix(r5, [fine[k] for k in lm], fine + origins, mode, departure)
        from_l[mode] = [list(M[k, :nf]) for k in range(len(lm))]
        l_to_o[mode] = [list(M[k, nf:]) for k in range(len(lm))]
    o_to_l = [[float(F[i, k]) for k in lm] for i in range(n)]
    aboxes = alt_boxes(origins, modes, fine, from_l, l_to_o, o_to_l)
    ev, val, info = run_boxes(aboxes)
    search_row("alt", val, ev.calls + n * len(lm), pre_sweeps=len(lm) * len(street),
               evaluated=info["evaluated"], landmarks=len(lm))
    viol = sum(1 for j in range(nf) if feas[j] for i in range(n)
               if not (aboxes[j][0][i] <= F[i, j] <= aboxes[j][1][i]))
    out["soundness"].append(dict(base, construction="alt", pairs_checked=int(n * feas.sum()),
                                 violations=viol, misbanded=np.nan,
                                 feasible_excluded=0, optimum_excluded=False))

    # polygons built on a grid, then read back as bands
    G = S[:, g0:]
    for method, widen in (("nearest", 0), ("contour", 0), ("contour", 1)):
        kidx = [polygon_band_index(grid, G[i], fine, thr12, method) for i in range(n)]
        boxes, excluded = [], set()
        for j in range(nf):
            ks = [kidx[i][j] for i in range(n)]
            if any(k is None for k in ks):
                boxes.append(None)
                excluded.add(j)
                continue
            b = [band_index_box(k, thr12, widen) for k in ks]
            boxes.append(([x[0] for x in b], [x[1] for x in b]))
        label = "polygon_%s%s" % (method, "_widened" if widen else "")
        ev, val, info = run_boxes(boxes)
        search_row(label, val, ev.calls, sweeps=n,
                   iso_requests=n * math.ceil(12 / CONTOURS_PER_REQUEST),
                   evaluated=info["evaluated"], K=12)
        mis = sum(1 for j in range(nf) if feas[j] for i in range(n)
                  if kidx[i][j] != exact_band(F[i, j], thr12))
        outside = sum(1 for j in range(nf) if feas[j] and j not in excluded for i in range(n)
                      if not (boxes[j][0][i] <= F[i, j] <= boxes[j][1][i]))
        fe = [j for j in range(nf) if feas[j] and j in excluded]
        out["soundness"].append(dict(
            base, construction=label, pairs_checked=int(n * feas.sum()),
            violations=outside, misbanded=mis, feasible_excluded=len(fe),
            optimum_excluded=bool(any(var_all[j] <= truth + 1e-9 for j in fe))))

    # equity: every rule scored on every metric, with kappa fixed per instance
    kappa = ob.calibrate_kappa(list(F[:, feas].ravel()))
    choice = {}
    fj = [j for j in range(nf) if feas[j]]
    choice["variance_opt"] = list(F[:, min(fj, key=lambda j: var_all[j])])
    choice["ede_opt"] = list(F[:, min(fj, key=lambda j: ob.ede_of(list(F[:, j]), kappa=kappa))])
    choice["min_sum"] = list(F[:, min(fj, key=lambda j: F[:, j].sum())])
    choice["min_max"] = list(F[:, min(fj, key=lambda j: F[:, j].max())])
    choice["min_range"] = list(F[:, min(fj, key=lambda j: F[:, j].max() - F[:, j].min())])
    rng = random.Random(seed)
    sample = rng.sample(range(nf), min(100, nf))
    rs = [j for j in sample if feas[j]]
    choice["random_100"] = list(F[:, min(rs, key=lambda j: var_all[j])]) if rs else None
    iso = iso_int(origins, modes, pre, thr12, fine)
    choice["iso_intersection"] = list(iso["centroid_times"]) if iso else None
    for name, p in geo.items():
        j = len(cands) + list(geo).index(name)
        choice[name] = list(S[:, j])
    for name, times in choice.items():
        r = dict(base, method=name, kappa=kappa)
        ok = times is not None and metrics.all_reachable(times, n)
        r["feasible"] = bool(ok)
        if ok:
            r.update(variance=metrics.variance(times), range=metrics.spread(times),
                     gini=metrics.gini(times), jain=metrics.jain(times),
                     ede=ob.ede_of(times, kappa=kappa), max=metrics.max_time(times),
                     mean=metrics.mean_time(times))
        out["equity"].append(r)

    # Lipschitz audit on the same surfaces
    out["lipschitz"].extend(lipschitz_rows(F, modes, pairs_ab, d_ab, base))

    # wall clock on the engine; alternate the order so JIT warm-up favours neither side
    by_key = {cell_key(p, 12): p for p in cands}
    sets = {"full_sweep": fine}
    for name in ("lipschitz_assumed", "lipschitz_assumed_screened"):
        sets[name] = [by_key[k] for k in touched[name] if k in by_key]
    order = list(sets) if seed % 2 == 0 else list(reversed(list(sets)))
    timing = {}
    for name in order:
        timing[name] = timed_sweeps(r5, origins, modes, sets[name], departure) if sets[name] else 0.0
    for row in out["search"]:
        if row["seed"] == seed and row["city"] == CITY:
            row["t_full_sweep_s"] = timing["full_sweep"]
            row["t_lip_touched_s"] = timing["lipschitz_assumed"]
            row["t_lip_screened_touched_s"] = timing["lipschitz_assumed_screened"]
            row["n_lip_touched"] = len(sets["lipschitz_assumed"])
            row["n_lip_screened_touched"] = len(sets["lipschitz_assumed_screened"])

    s = [r for r in out["search"] if r["seed"] == seed]
    pick = {r["method"]: r for r in s}
    print("  seed %2d fine=%4d feasible=%4d truth=%8.2f | exh %5d | lip %5d (%s) | lip+scr %4d (%s)"
          " | K12 %3d (%s) | poly-contour %s"
          % (seed, nf, int(feas.sum()), truth, pick["exhaustive"]["pairs"],
             pick["lipschitz_assumed"]["pairs"], pick["lipschitz_assumed"]["optimal"],
             pick["lipschitz_assumed_screened"]["pairs"],
             pick["lipschitz_assumed_screened"]["optimal"],
             pick["bands_K12"]["pairs"], pick["bands_K12"]["optimal"],
             pick["polygon_contour"]["optimal"]), flush=True)


def main():
    nb = apply_sample_bbox(CITY)
    if nb:
        print("sampling origins inside the loaded network extent %s" % (nb,), flush=True)
    departure = pd.Timestamp(DEPARTURE)
    print("loading %s network (departure %s)..." % (CITY, DEPARTURE), flush=True)
    t0 = time.time()
    r5 = R5Backend(OSM, GTFS)
    print("  ready in %.0fs" % (time.time() - t0), flush=True)

    out = {"search": [], "equity": [], "lipschitz": [], "soundness": []}
    for seed in SEEDS:
        run_instance(r5, seed, departure, out)

    os.makedirs("outputs", exist_ok=True)
    for k, rows in out.items():
        pd.DataFrame(rows).to_csv("outputs/eval_%s_%s.csv" % (k, CITY), index=False)
    if not out["search"]:
        raise SystemExit("no solvable instances for %s" % CITY)
    df = pd.DataFrame(out["search"])
    print("\n%s: %d instances" % (CITY, df.seed.nunique()))
    print(df.groupby("method")[["pairs", "sweeps", "iso_requests", "optimal", "certified"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
