
from __future__ import annotations

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import pandas as pd

from fairmp.candidates import polyfill_centroids, region_polygon
from fairmp.geo import haversine_km
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.travel_time import R5Backend

from run_lipschitz_audit import DEPARTURE, GTFS, JUMP_MAX_KM, NEAR_K, OSM, FINE, surface

SEED = 0
N_USERS = 6
TOP = 15


def main():
    departure = pd.Timestamp(DEPARTURE)
    r5 = R5Backend(OSM, GTFS)

    origins = sample_origins("london", N_USERS, seed=SEED, spread="clustered",
                             clusters=2, cluster_sd_deg=0.04)
    modes = assign_modes(N_USERS, "mixed", seed=SEED)
    cands = [p for _c, p in polyfill_centroids(region_polygon(origins), FINE)]

    lat = np.array([p.lat for p in cands])
    lng = np.array([p.lng for p in cands])
    lat0 = float(lat.mean())
    x = (lng - float(lng.mean())) * 111.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 111.0
    D = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(D, np.inf)
    knn = np.argsort(D, axis=1)[:, :NEAR_K]

    rows = []
    for i, m in enumerate(modes):
        mode = m[0]
        ts = surface(r5, origins[i], mode, cands, departure)
        n_reach = sum(1 for t in ts if math.isfinite(t))
        t_max_reach = max((t for t in ts if math.isfinite(t)), default=0.0)
        for a in range(len(cands)):
            for b in knn[a]:
                b = int(b)
                if b <= a:
                    continue
                ta, tb = ts[a], ts[b]
                if not (math.isfinite(ta) and math.isfinite(tb)):
                    continue
                d = haversine_km(cands[a], cands[b])
                if d > JUMP_MAX_KM:
                    continue
                rows.append({
                    "user": i, "mode": mode, "dt": abs(ta - tb), "d_km": round(d, 3),
                    "slope": round(abs(ta - tb) / d, 1),
                    "t_a": ta, "t_b": tb,
                    "lat_a": round(cands[a].lat, 5), "lng_a": round(cands[a].lng, 5),
                    "lat_b": round(cands[b].lat, 5), "lng_b": round(cands[b].lng, 5),
                    # is either endpoint close to the reachability frontier? if so the jump
                    # may be a horizon effect rather than a topological one
                    "near_frontier": (max(ta, tb) > 0.9 * t_max_reach),
                    "reach_frac": round(n_reach / len(cands), 3),
                })

    df = pd.DataFrame(rows).sort_values("dt", ascending=False)
    df.to_csv("outputs/jump_inspection.csv", index=False)

    print("Largest travel-time jumps between candidates under %.2f km apart" % JUMP_MAX_KM)
    print("London, seed %d, departure %s\n" % (SEED, DEPARTURE))
    cols = ["mode", "dt", "d_km", "slope", "t_a", "t_b", "lat_a", "lng_a", "lat_b", "lng_b",
            "near_frontier"]
    print(df.head(TOP)[cols].to_string(index=False))

    print("\nHow many of the big jumps sit near the reachability frontier?")
    big = df[df.dt >= 3.0]
    print("  jumps >= 3 min: %d, of which near frontier: %d (%.1f%%)"
          % (len(big), int(big.near_frontier.sum()),
             100.0 * big.near_frontier.mean() if len(big) else 0.0))
    interior = big[~big.near_frontier]
    print("  interior jumps >= 3 min: %d, max %.0f min" % (len(interior),
          interior.dt.max() if len(interior) else 0.0))
    print("\n  by mode (interior only):")
    if len(interior):
        print(interior.groupby("mode").agg(n=("dt", "size"), max_dt=("dt", "max"),
                                           median_dt=("dt", "median")).round(2).to_string())
    print("\nwrote outputs/jump_inspection.csv")
    print("Paste a lat/lng pair into a map to check it against a river, rail cutting or")
    print("park boundary before any of this is described as a topological discontinuity.")


if __name__ == "__main__":
    main()
