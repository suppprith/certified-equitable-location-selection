
from __future__ import annotations

import math
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from fairmp.candidates import polyfill_centroids, region_polygon
from fairmp.geo import haversine_km
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.travel_time import EUCLIDEAN_SPEED_KMH, R5_MODE, R5Backend

OSM = os.path.join(ROOT, "data", "london", "network.osm.pbf")
GTFS = [os.path.join(ROOT, "data", "london", "gtfs", "london_bus.zip")]

COARSE, FINE = 8, 9
PAIR_MAX_KM = 2.0      # the scale the parent-cell bound actually operates at
DETOUR = 1.3
SEEDS = range(int(os.environ.get("N_INSTANCES", "8")))
N_USERS = 6

# The GTFS feed is valid 2026-06-21 to 2027-03-21. A departure outside that window
# silently yields no transit service and r5 falls back to walking, which invalidates
# every transit number. Keep this inside the window and on a weekday.
DEPARTURE = "2026-09-16 08:30:00"

# r5 reports whole minutes, so |dt| between two nearby candidates carries up to a minute
# of pure rounding. That inflates |dt|/d by 1/d, which at 0.35 km is 2.9 min/km against a
# driving constant of 3.12. Slope statistics are therefore reported per distance band, and
# the discontinuity evidence is the size of |dt| itself at short range: rounding can only
# ever produce one minute, so a multi-minute jump across a short distance is real.
BANDS = [(0.0, 0.45), (0.45, 0.8), (0.8, 1.2), (1.2, 2.0)]

# Adjacent H3 res-9 centres sit sqrt(3) * 0.2008 = 0.348 km apart, so the shortest pair the
# candidate set can produce is ~0.35 km. The close-pair threshold has to clear that or the
# bucket is empty by construction. At 0.35 km a whole minute of rounding is 2.9 min/km, so
# requiring a gap of 3 minutes keeps the jump test well clear of the artefact.
JUMP_MAX_KM = 0.45
JUMP_MIN = 3.0
NEAR_K = 6           # nearest neighbours per candidate, to populate the close bands
FAR_PAIRS = 6000     # random long-range pairs, to populate the far bands


def euclidean_constant(mode: str) -> float:
    """L used today: derived from mode speed with a detour factor. Exactly tight on the
    Euclidean backend, assumed to transfer to real networks."""
    return 60.0 * DETOUR / EUCLIDEAN_SPEED_KMH.get(mode, 4.8)


def surface(r5, origin, mode, cands, departure):
    r5py = r5._r5py
    tmodes = [getattr(r5py.TransportMode, x) for x in R5_MODE[mode]]
    src = gpd.GeoDataFrame({"id": ["o0"]},
                           geometry=[Point(origin.lng, origin.lat)], crs="EPSG:4326")
    dest = gpd.GeoDataFrame({"id": [f"c{i}" for i in range(len(cands))]},
                            geometry=[Point(p.lng, p.lat) for p in cands], crs="EPSG:4326")
    ttm = r5py.TravelTimeMatrix(r5.network, origins=src, destinations=dest,
                                departure=departure, transport_modes=tmodes)
    df = ttm.compute_travel_times() if hasattr(ttm, "compute_travel_times") else ttm
    out = {}
    for r in df.itertuples(index=False):
        try:
            j = int(str(r.to_id)[1:])
        except ValueError:
            continue
        t = r.travel_time
        out[j] = float(t) if t == t else math.inf
    return [out.get(j, math.inf) for j in range(len(cands))]


def audit_pairs(times, cands, rng_pairs):
    """Empirical slope |dt|/d over candidate pairs, kept with the distance and the raw |dt|
    so rounding artefacts can be separated from genuine discontinuities."""
    out = []
    for a, b in rng_pairs:
        ta, tb = times[a], times[b]
        if not (math.isfinite(ta) and math.isfinite(tb)):
            continue
        d = haversine_km(cands[a], cands[b])
        if d < 1e-6:
            continue
        out.append((abs(ta - tb) / d, d, abs(ta - tb)))
    return out


def main():
    import numpy as np

    departure = pd.Timestamp(DEPARTURE)
    print("loading London network...", flush=True)
    t0 = time.time()
    r5 = R5Backend(OSM, GTFS)
    print("  network ready in %.0fs" % (time.time() - t0), flush=True)

    rows = []
    for seed in SEEDS:
        origins = sample_origins("london", N_USERS, seed=seed, spread="clustered",
                                 clusters=2, cluster_sd_deg=0.04)
        modes = assign_modes(N_USERS, "mixed", seed=seed)
        poly = region_polygon(origins)
        cands = [p for _c, p in polyfill_centroids(poly, FINE)]
        if len(cands) < 50:
            continue

        # Two pair populations, because they answer different questions.
        # Near-neighbour pairs are where a discontinuity shows up as a multi-minute jump
        # over a few hundred metres; random pairs from a wide region essentially never
        # land that close, so they have to be constructed deliberately.
        rs = np.random.default_rng(seed)
        lat = np.array([p.lat for p in cands])
        lng = np.array([p.lng for p in cands])
        lat0 = float(lat.mean())
        x = (lng - float(lng.mean())) * 111.0 * math.cos(math.radians(lat0))
        y = (lat - lat0) * 111.0
        D = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        np.fill_diagonal(D, np.inf)

        pairs = set()
        # (a) each candidate against its nearest neighbours -> the close bands
        knn = np.argsort(D, axis=1)[:, :NEAR_K]
        for a in range(len(cands)):
            for b in knn[a]:
                pairs.add((min(a, int(b)), max(a, int(b))))
        # (b) random pairs across the region -> the far bands
        ok = np.argwhere((D >= 0.3) & (D <= PAIR_MAX_KM))
        if len(ok):
            take = ok[rs.choice(len(ok), size=min(len(ok), FAR_PAIRS), replace=False)]
            for a, b in take:
                pairs.add((min(int(a), int(b)), max(int(a), int(b))))
        pairs = sorted(pairs)
        if not pairs:
            continue

        for i, m in enumerate(modes):
            mode = m[0]
            ts = surface(r5, origins[i], mode, cands, departure)
            reach = sum(1 for t in ts if math.isfinite(t))
            slopes = audit_pairs(ts, cands, pairs)
            if not slopes:
                continue
            L_euc = euclidean_constant(mode)
            arr = np.array(slopes)
            s, d, dt = arr[:, 0], arr[:, 1], arr[:, 2]

            row = {
                "seed": seed, "user": i, "mode": mode,
                "pairs": len(s),
                "reachable_frac": round(reach / len(cands), 4),
                "L_euclidean": round(L_euc, 3),
                "slope_p99": round(float(np.percentile(s, 99)), 3),
                "violation_all_pct": round(100.0 * float((s > L_euc).mean()), 3),
            }
            # slope per distance band: rounding noise falls off as 1/d, a true jump does not
            for lo_d, hi_d in BANDS:
                m = (d >= lo_d) & (d < hi_d)
                key = "b%g_%g" % (lo_d, hi_d)
                row[key + "_n"] = int(m.sum())
                row[key + "_viol"] = (round(100.0 * float((s[m] > L_euc).mean()), 3)
                                      if m.sum() else float("nan"))
            # discontinuity evidence: rounding can only ever move |dt| by one minute, so a
            # multi-minute gap over a short distance cannot be an artefact
            close = d < JUMP_MAX_KM
            row["close_n"] = int(close.sum())
            row["close_dt_max"] = round(float(dt[close].max()), 2) if close.sum() else float("nan")
            row["close_jump_pct"] = (round(100.0 * float((dt[close] >= JUMP_MIN).mean()), 3)
                                     if close.sum() else float("nan"))

            # Does an empirically calibrated constant transfer across scales? Calibrate on
            # the long-range band, which is where estimation is cleanest, then test it at
            # short range. A constant that fails there cannot bound the local gradient,
            # which is exactly what the parent-cell bound needs it to do.
            far = d >= 1.2
            if far.sum() >= 20 and close.sum() >= 20:
                L_emp = float(np.percentile(s[far], 99))
                row["L_empirical"] = round(L_emp, 3)
                row["emp_viol_close_pct"] = round(100.0 * float((s[close] > L_emp).mean()), 3)
                row["emp_over_euc"] = round(L_emp / L_euc, 3)
            else:
                row["L_empirical"] = float("nan")
                row["emp_viol_close_pct"] = float("nan")
                row["emp_over_euc"] = float("nan")
            rows.append(row)
            print("  seed %2d u%d %-8s reach=%.2f p99=%6.2f L=%5.2f viol_all=%5.1f%% "
                  "far(1-2km)=%5s%% closeMaxDt=%5s min jump>=3min=%5s%%"
                  % (seed, i, mode, row["reachable_frac"], row["slope_p99"], L_euc,
                     row["violation_all_pct"], row.get("b1.2_2_viol"),
                     row["close_dt_max"], row["close_jump_pct"]), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/lipschitz_audit.csv", index=False)

    print("\nLipschitz validity on the real London multimodal network")
    print("departure %s, inside the GTFS validity window" % DEPARTURE)
    print("\nr5 reports whole minutes, so |dt|/d carries up to 1/d of pure rounding.")
    print("Violation rates are therefore shown per distance band; the 1-2 km band is the")
    print("one where rounding (<= 0.5-1.0 min/km) is small next to L.\n")
    agg = df.groupby("mode").agg(
        users=("L_euclidean", "size"),
        reach=("reachable_frac", "mean"),
        L=("L_euclidean", "mean"),
        p99=("slope_p99", "mean"),
        viol_0_045=("b0_0.45_viol", "mean"),
        viol_045_08=("b0.45_0.8_viol", "mean"),
        viol_08_12=("b0.8_1.2_viol", "mean"),
        viol_12_2=("b1.2_2_viol", "mean"),
    ).round(3)
    print(agg.to_string())

    print("\nDiscontinuity evidence: pairs closer than %.1f km. Rounding can move |dt| by at"
          % JUMP_MAX_KM)
    print("most one minute, so a gap of >= %.0f minutes over that distance is a real jump." % JUMP_MIN)
    jag = df.groupby("mode").agg(
        close_pairs=("close_n", "sum"),
        max_dt_min=("close_dt_max", "max"),
        pct_jumps=("close_jump_pct", "mean"),
        L_emp=("L_empirical", "mean"),
        emp_over_euc=("emp_over_euc", "mean"),
        emp_fails_close_pct=("emp_viol_close_pct", "mean"),
    ).round(3)
    print(jag.to_string())

    print("\n  users violating the assumed constant in the 1.2-2 km band: %d / %d"
          % (int((df["b1.2_2_viol"] > 0).sum()), len(df)))
    print("\nwrote outputs/lipschitz_audit.csv")


if __name__ == "__main__":
    main()
