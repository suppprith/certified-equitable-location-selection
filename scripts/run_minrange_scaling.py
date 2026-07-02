
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fairmp import metrics
from fairmp.candidates import polyfill_centroids, region_polygon
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.travel_time import CachedEvaluator, EuclideanBackend

NS = [5, 10, 20, 50]
N_SEEDS = int(os.environ.get("N_INSTANCES", "100"))
FINE_RES = 9

def main():
    b = EuclideanBackend()
    rows = []
    for n in NS:
        for seed in range(N_SEEDS):
            origins = sample_origins("london", n, seed=seed, spread="clustered",
                                     clusters=1, cluster_sd_deg=0.03)
            modes = assign_modes(n, "mixed", seed=seed)
            ev = CachedEvaluator(b)
            grid = polyfill_centroids(region_polygon(origins), FINE_RES)

            best_var, best_var_times = float("inf"), None
            best_rng, best_rng_times = float("inf"), None
            for _c, pt in grid:
                times = [ev.effective(o, pt, m) for o, m in zip(origins, modes)]
                if not metrics.all_reachable(times, n):
                    continue
                v, r = metrics.variance(times), metrics.spread(times)
                if v < best_var:
                    best_var, best_var_times = v, times
                if r < best_rng:
                    best_rng, best_rng_times = r, times

            v_at_rng = metrics.variance(best_rng_times)
            rows.append({
                "n": n, "seed": seed,
                "var_at_var_opt": round(best_var, 3),
                "var_at_range_opt": round(v_at_rng, 3),
                "rel_gap": round((v_at_rng - best_var) / v_at_rng, 4) if v_at_rng > 0 else 0.0,
                "gini_var_opt": round(metrics.gini(best_var_times), 4),
                "gini_range_opt": round(metrics.gini(best_rng_times), 4),
                "spread_var_opt": round(metrics.spread(best_var_times), 2),
                "spread_range_opt": round(best_rng, 2),
            })

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/minrange_scaling.csv", index=False)

    print("Variance at the range optimum vs the variance optimum, fine-grid enumeration")
    print(f"({N_SEEDS} instances per group size, synthetic Euclidean model)\n")
    g = df.groupby("n")["rel_gap"]
    summary = pd.DataFrame({"median_gap": g.median(), "mean_gap": g.mean(),
                            "p90_gap": g.quantile(0.9), "max_gap": g.max()}).round(4)
    print(summary.to_string())

if __name__ == "__main__":
    main()
