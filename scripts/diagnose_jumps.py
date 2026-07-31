
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
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.travel_time import R5Backend

from run_lipschitz_audit import DEPARTURE, FINE, GTFS, OSM, surface

SEED = 0
N_USERS = 6
MODES_TO_PROBE = ["driving", "cycling", "transit", "walking"]


def main():
    departure = pd.Timestamp(DEPARTURE)
    r5 = R5Backend(OSM, GTFS)

    origins = sample_origins("london", N_USERS, seed=SEED, spread="clustered",
                             clusters=2, cluster_sd_deg=0.04)
    cands = [p for _c, p in polyfill_centroids(region_polygon(origins), FINE)]

    lat = np.array([p.lat for p in cands])
    lng = np.array([p.lng for p in cands])
    lat0 = float(lat.mean())
    x = (lng - float(lng.mean())) * 111.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 111.0
    D = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(D, np.inf)
    knn = np.argsort(D, axis=1)[:, :6]

    # Probe every candidate from the SAME origin under several modes, and from several
    # origins under the same mode. A candidate that is anomalous under every origin and
    # every mode is a property of the point (network attachment). One that is anomalous
    # only for particular origins is a property of the route (a real barrier).
    probes = {}
    for mode in MODES_TO_PROBE:
        probes[(mode, 0)] = surface(r5, origins[0], mode, cands, departure)
    for oi in (1, 2, 3):
        probes[("driving", oi)] = surface(r5, origins[oi], "driving", cands, departure)

    def excess(ts, a):
        """How far this candidate sits above its own neighbours."""
        ta = ts[a]
        nb = [ts[int(b)] for b in knn[a] if math.isfinite(ts[int(b)])]
        if not math.isfinite(ta) or len(nb) < 3:
            return float("nan")
        return ta - float(np.median(nb))

    rows = []
    for a in range(len(cands)):
        r = {"idx": a, "lat": round(cands[a].lat, 5), "lng": round(cands[a].lng, 5)}
        for key, ts in probes.items():
            r["%s_o%d" % key] = round(excess(ts, a), 1)
        rows.append(r)
    df = pd.DataFrame(rows)

    cols_mode = [c for c in df.columns if c.endswith("_o0")]
    cols_orig = [c for c in df.columns if c.startswith("driving_o")]
    df["n_modes_anom"] = (df[cols_mode] > 10).sum(axis=1)
    df["n_origins_anom"] = (df[cols_orig] > 10).sum(axis=1)
    df.to_csv("outputs/jump_diagnosis.csv", index=False)

    print("Are the extreme candidates a property of the POINT or of the ROUTE?")
    print("excess = candidate time minus the median of its immediate neighbours, minutes.")
    print("A point anomalous under every mode and every origin is a network-attachment")
    print("(snapping) effect. One anomalous only for some origins is a real barrier.\n")

    worst = df.sort_values("cycling_o0", ascending=False).head(12)
    print(worst[["lat", "lng"] + cols_mode + ["n_modes_anom"]].to_string(index=False))

    print("\nSame candidates, driving from four different origins:")
    print(worst[["lat", "lng"] + cols_orig + ["n_origins_anom"]].to_string(index=False))

    anom = df[df[cols_mode].max(axis=1) > 10]
    print("\ncandidates anomalous (>10 min above neighbours) under at least one mode: %d / %d"
          % (len(anom), len(df)))
    if len(anom):
        allmode = int((anom.n_modes_anom == len(cols_mode)).sum())
        allorig = int((anom.n_origins_anom == len(cols_orig)).sum())
        print("  of those, anomalous under ALL %d modes:   %d (%.0f%%)  -> snapping"
              % (len(cols_mode), allmode, 100.0 * allmode / len(anom)))
        print("  of those, anomalous from ALL %d origins:  %d (%.0f%%)  -> snapping"
              % (len(cols_orig), allorig, 100.0 * allorig / len(anom)))
        print("  anomalous under exactly one mode:        %d (%.0f%%)  -> mode-specific barrier"
              % (int((anom.n_modes_anom == 1).sum()),
                 100.0 * int((anom.n_modes_anom == 1).sum()) / len(anom)))
    print("\nwrote outputs/jump_diagnosis.csv")


if __name__ == "__main__":
    main()
