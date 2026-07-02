
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fairmp import metrics
from fairmp.algorithm import Params, fair_meeting_point
from fairmp.baselines import exhaustive_variance
from fairmp.certificate import certificate_report, certified_search, euclidean_lipschitz
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.travel_time import CachedEvaluator, EuclideanBackend

PARAMS = Params(coarse_res=8, fine_res=9, k_c=400, k_refine=10, ring=1, t_max=240.0)
STARVED = Params(coarse_res=6, fine_res=9, k_c=400, k_refine=1, ring=1, t_max=240.0)

def exhaustive_variance_value(origins, modes, backend, res):
    ev = CachedEvaluator(backend)
    xpt = exhaustive_variance(origins, modes, ev, res=res)
    xtimes = [ev.effective(o, xpt, m) for o, m in zip(origins, modes)]
    return metrics.variance(xtimes), ev.calls

def run_one(origins, modes, backend, params):
    lip = euclidean_lipschitz(modes)

    ev = CachedEvaluator(backend)
    best, _r, scored, _d = fair_meeting_point(origins, modes, ev, params)
    plain_calls = ev.calls
    rep = certificate_report(origins, modes, ev, params, best, scored, lip)

    ev2 = CachedEvaluator(backend)
    cbest, cert, _d2 = certified_search(origins, modes, ev2, params, lip)

    xv, x_calls = exhaustive_variance_value(origins, modes, backend, params.fine_res)
    ours_v = metrics.variance(best.times)
    return {
        "ours_variance": round(ours_v, 3),
        "exhaustive_variance": round(xv, 3),
        "opt_gap_pct": round(100 * (ours_v - xv) / xv, 3) if xv > 0 else 0.0,
        "fixed_certified": rep["certified"],
        "fixed_extra_evals": rep.get("extra_evals", 0),
        "adaptive_variance": round(cert["v_star"], 3),
        "adaptive_gap_pct": round(100 * (cert["v_star"] - xv) / xv, 3) if xv > 0 else 0.0,
        "adaptive_rounds": cert["rounds"],
        "plain_calls": plain_calls,
        "adaptive_calls": cert["routing_calls"],
        "exhaustive_calls": x_calls,
    }

def main():
    os.makedirs("outputs", exist_ok=True)
    backend = EuclideanBackend()

    rows = []
    for city in ["london", "bengaluru"]:
        for n in [5, 6, 7, 8]:
            for seed in range(40):
                origins = sample_origins(city, n, seed=seed, spread="clustered",
                                         clusters=2, cluster_sd_deg=0.04)
                modes = assign_modes(n, "mixed", seed=seed)
                r = run_one(origins, modes, backend, PARAMS)
                r.update(city=city, n=n, seed=seed)
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/certificate.csv", index=False)

    zero = df[df.opt_gap_pct <= 1e-6]
    pos = df[df.opt_gap_pct > 1e-6]
    unsound = df[(df.fixed_certified) & (df.opt_gap_pct > 1e-6)]
    adaptive_bad = df[df.adaptive_gap_pct.abs() > 1e-6]
    print("Certificate validation, %d synthetic instances (default parameters):" % len(df))
    print("  fixed-budget certification rate: %.1f%% overall, %.1f%% of zero-gap instances"
          % (100 * df.fixed_certified.mean(),
             100 * zero.fixed_certified.mean() if len(zero) else 0.0))
    print("  soundness violations (certified but gap > 0): %d  [must be 0]" % len(unsound))
    print("  positive-gap instances: %d; certificate refused all: %s"
          % (len(pos), "yes" if len(pos) and not pos.fixed_certified.any() else
             ("n/a" if not len(pos) else "NO")))
    print("  adaptive mode: nonzero final gap on %d instances  [must be 0]" % len(adaptive_bad))
    print("  mean routing calls: plain %.0f, adaptive-certified %.0f, exhaustive %.0f"
          % (df.plain_calls.mean(), df.adaptive_calls.mean(), df.exhaustive_calls.mean()))
    print("  adaptive / exhaustive call ratio: %.2f"
          % (df.adaptive_calls.mean() / df.exhaustive_calls.mean()))

    origins = sample_origins("london", 5, seed=7, spread="clustered",
                             clusters=2, cluster_sd_deg=0.04)
    modes = assign_modes(5, "mixed", seed=7)
    seam = run_one(origins, modes, backend, STARVED)
    seam.update(city="london", n=5, seed=7)
    pd.DataFrame([seam]).to_csv("outputs/certificate_seam.csv", index=False)
    print("\nSeam instance (london N=5 seed=7, starved coarse_res=6, k_refine=1):")
    print("  starved gap %.1f%%, fixed certificate certified: %s  [must be False]"
          % (seam["opt_gap_pct"], seam["fixed_certified"]))
    print("  adaptive-certified gap %.3f%% with %d calls (exhaustive %d)"
          % (seam["adaptive_gap_pct"], seam["adaptive_calls"], seam["exhaustive_calls"]))

if __name__ == "__main__":
    main()
