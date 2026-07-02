
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fairmp import darkstore
from fairmp.algorithm import Params
from fairmp.travel_time import EuclideanBackend

N_CELLS = [20, 40, 100, 150]
N_SEEDS = int(os.environ.get("N_INSTANCES", "40"))
PARAMS = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=30.0)

def main():
    backend = EuclideanBackend()
    rows = []
    for n_cells in N_CELLS:
        for seed in range(N_SEEDS):
            demand, weights = darkstore.sample_demand("london", n_cells, seed=seed)
            res = {r["method"]: r for r in
                   darkstore.run_darkstore_instance(demand, weights, backend, PARAMS, sla_min=10.0)}
            if "ours" not in res or "min_range" not in res:
                continue
            v_ours = res["ours"]["w_variance"]
            v_range = res["min_range"]["w_variance"]
            rows.append({
                "n_cells": n_cells, "seed": seed,
                "wvar_variance_site": round(v_ours, 3),
                "wvar_range_site": round(v_range, 3),
                "range_penalty_pct": round(100 * (v_range - v_ours) / v_range, 2) if v_range > 0 else 0.0,
            })

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/siting_scaling.csv", index=False)

    print("Dark-store siting: weighted-variance penalty of the range site vs the variance site")
    print(f"({N_SEEDS} instances per demand size, synthetic Euclidean model)\n")
    g = df.groupby("n_cells")["range_penalty_pct"]
    summary = pd.DataFrame({"median_pct": g.median(), "mean_pct": g.mean(),
                            "p90_pct": g.quantile(0.9), "max_pct": g.max()}).round(2)
    print(summary.to_string())

if __name__ == "__main__":
    main()
