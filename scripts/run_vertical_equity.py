
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fairmp.algorithm import Params, fair_meeting_point
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.travel_time import CachedEvaluator, EuclideanBackend

N_SEEDS = int(os.environ.get("N_INSTANCES", "24"))

def main():
    b = EuclideanBackend()
    p = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0)
    print(f"{'seed':>4} {'worst-off time':>14} {'after up-weight x4':>19} {'drop':>6}")
    rows = []
    for seed in range(N_SEEDS):
        o = sample_origins("london", 5, seed=seed, spread="clustered", clusters=1, cluster_sd_deg=0.03)
        m = assign_modes(5, "mixed", seed=seed)
        ev = CachedEvaluator(b)
        best, _r, _s, _d = fair_meeting_point(o, m, ev, p)
        i = max(range(5), key=lambda k: best.times[k])
        w = [1, 1, 1, 1, 1]
        w[i] = 4
        ev2 = CachedEvaluator(b)
        best2, _r, _s, _d = fair_meeting_point(o, m, ev2, p, weights=w)
        t, tw = best.times[i], best2.times[i]
        rows.append({"seed": seed, "worst_user_time": round(t, 2),
                     "worst_user_time_upweighted": round(tw, 2), "drop_min": round(t - tw, 2)})
        print(f"{seed:>4} {t:>14.1f} {tw:>19.1f} {t - tw:>6.1f}")
    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/vertical_equity.csv", index=False)
    print(f"\nmean drop in the up-weighted (worst-off) user's travel time: {df['drop_min'].mean():.1f} min")
    print(f"max drop: {df['drop_min'].max():.1f} min over {len(df)} instances")

if __name__ == "__main__":
    main()
