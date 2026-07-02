
from __future__ import annotations

import datetime as dt
import glob
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_jdk = sorted(glob.glob(os.path.join(ROOT, "tools", "jdk21", "jdk-*")))
if _jdk:
    os.environ.setdefault("JAVA_HOME", _jdk[0])
    os.environ["PATH"] = os.path.join(_jdk[0], "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Xmx4g")

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from fairmp import baselines, metrics
from fairmp.algorithm import Params, fair_meeting_point
from fairmp.candidates import polyfill_centroids, region_polygon
from fairmp.geo import haversine_km
from fairmp.runner import run_instance
from fairmp.scenarios import assign_modes_with_choice, sample_origins
from fairmp.travel_time import R5_MODE, CachedEvaluator, PrecomputedBackend, R5Backend

OSM = os.path.join(ROOT, "data", "london", "network.osm.pbf")
GTFS = [os.path.join(ROOT, "data", "london", "gtfs", "london_bus.zip")]

def candidates_for(origins, modes, coarse, fine):
    region = region_polygon(origins)
    pts = {}
    for res in (coarse, fine):
        for _c, p in polyfill_centroids(region, res):
            pts[(round(p.lat, 6), round(p.lng, 6))] = p
    for bp in (baselines.geometric_centroid(origins, modes),
               baselines.weighted_centroid(origins, modes),
               baselines.geometric_median(origins, modes)):
        pts[(round(bp.lat, 6), round(bp.lng, 6))] = bp
    return list(pts.values())

def precompute(r5, origins, modes, cands, departure):

    r5py = r5._r5py
    net = r5.network
    pre = PrecomputedBackend()
    dest_gdf = gpd.GeoDataFrame(
        {"id": [f"c{i}" for i in range(len(cands))]},
        geometry=[Point(p.lng, p.lat) for p in cands], crs="EPSG:4326")

    by_mode = defaultdict(list)
    for i, (o, ms) in enumerate(zip(origins, modes)):
        for mode in ms:
            by_mode[mode].append((i, o))

    for mode, users in by_mode.items():
        tmodes = [getattr(r5py.TransportMode, x) for x in R5_MODE[mode]]
        src_gdf = gpd.GeoDataFrame(
            {"id": [f"o{i}" for i, _ in users]},
            geometry=[Point(o.lng, o.lat) for _, o in users], crs="EPSG:4326")
        ttm = r5py.TravelTimeMatrix(net, origins=src_gdf, destinations=dest_gdf,
                                    departure=departure, transport_modes=tmodes)
        df = ttm.compute_travel_times() if hasattr(ttm, "compute_travel_times") else ttm
        omap = {f"o{i}": o for i, o in users}
        cmap = {f"c{i}": cands[i] for i in range(len(cands))}
        for r in df.itertuples(index=False):
            o, c = omap.get(r.from_id), cmap.get(r.to_id)
            if o is None or c is None:
                continue
            t = r.travel_time
            pre.put(mode, o, c, float(t) if t == t else float("inf"))
    return pre

def secondary_mode_usage(pre, origins, modes, point):

    used_secondary = 0
    multimodal = 0
    for o, ms in zip(origins, modes):
        if len(ms) < 2:
            continue
        multimodal += 1
        times = [pre.minutes(o, point, m) for m in ms]
        if min(range(len(ms)), key=lambda k: times[k]) != 0:
            used_secondary += 1
    return multimodal, used_secondary

def main():
    print("loading London network (cached .dat if present)...")
    r5 = R5Backend(OSM, GTFS)
    print("network ready")
    today = dt.date.today()
    wed = today + dt.timedelta((2 - today.weekday()) % 7 + 7)
    departure = dt.datetime(wed.year, wed.month, wed.day, 8, 30)
    params = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0, gamma=0.0)

    n_instances = int(os.environ.get("N_INSTANCES", "100"))
    n_users = int(os.environ.get("N_USERS", "5"))
    rows, cmp_rows = [], []

    for seed in range(n_instances):
        origins = sample_origins("london", n_users, seed=seed, spread="clustered",
                                 clusters=1, cluster_sd_deg=0.03)
        modes = assign_modes_with_choice(n_users, seed=seed, frac_choice=0.4)
        primary = [[ms[0]] for ms in modes]
        cands = candidates_for(origins, modes, 8, 9)
        print(f"seed {seed}: {len(cands)} candidates, "
              f"mode sets {[len(m) for m in modes]} -> r5 matrices...")
        pre = precompute(r5, origins, modes, cands, departure)

        for r in run_instance(origins, modes, pre, params, fine_res=9, variants=("ede",)):
            r.update(seed=seed, config="choice")
            rows.append(r)

        best_c, _r, _s, _d = fair_meeting_point(origins, modes, CachedEvaluator(pre), params)
        best_p, _r, _s, _d = fair_meeting_point(origins, primary, CachedEvaluator(pre), params)
        if best_c is None or best_p is None:
            continue
        var_c = metrics.variance(best_c.times)
        var_p = metrics.variance(best_p.times)
        multimodal, used_secondary = secondary_mode_usage(pre, origins, modes, best_c.point)
        cmp_rows.append({
            "seed": seed,
            "n_multimodal": multimodal,
            "users_using_secondary": used_secondary,
            "var_choice": round(var_c, 3),
            "var_primary_only": round(var_p, 3),
            "var_reduction_pct": round(100 * (var_p - var_c) / var_p, 2) if var_p > 0 else 0.0,
            "point_moved_m": round(1000 * haversine_km(best_c.point, best_p.point), 1),
        })

    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/multimode_london.csv", index=False)
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv("outputs/multimode_choice_effect.csv", index=False)

    cols = [c for c in ["variance", "jain", "gini", "ede", "mean", "max", "feasible", "opt_gap"]
            if c in df.columns]
    print("\nREAL London multi-mode social meetup (mean over instances):")
    print(df.groupby("method")[cols].mean(numeric_only=True).round(3)
          .sort_values("variance").to_string())

    if not cmp.empty:
        moved = (cmp["point_moved_m"] > 1.0).mean()
        switched = (cmp["users_using_secondary"] > 0).mean()
        print("\nDoes genuine mode choice change anything?")
        print(f"  instances where a multi-mode user takes their secondary mode "
              f"at the fair point: {100 * switched:.0f}%")
        print(f"  instances where mode choice moves the fair point vs primary-only: "
              f"{100 * moved:.0f}%")
        print(f"  mean travel-time variance reduction from allowing choice: "
              f"{cmp['var_reduction_pct'].mean():.1f}%")

if __name__ == "__main__":
    main()
