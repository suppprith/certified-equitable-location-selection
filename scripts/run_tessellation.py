
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fairmp import metrics
from fairmp.algorithm import Params, fair_meeting_point
from fairmp.baselines import exhaustive_variance
from fairmp.candidates import region_polygon
from fairmp.certificate import certified_search, euclidean_lipschitz
from fairmp.geo import centroid, haversine_km
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.tessellation import ALL_TESSELLATIONS, make_tessellation
from fairmp.travel_time import CachedEvaluator, EuclideanBackend

COARSE, FINE, REF = 8, 9, 10
CITIES = ["london", "bengaluru"]
NS = [5, 6, 7]
SEEDS = range(int(os.environ.get("N_INSTANCES", "30")))


def reference_optimum(origins, modes, backend):
    """Best variance over a grid 7x finer than the working grid, as a proxy for the
    continuous optimum. Pattern-neutral so every tessellation is scored against it."""
    ev = CachedEvaluator(backend)
    ref = make_tessellation("h3", centroid(origins))
    pt = exhaustive_variance(origins, modes, ev, res=REF, tess=ref)
    if pt is None:
        return None, None
    times = [ev.effective(o, pt, m) for o, m in zip(origins, modes)]
    return metrics.variance(times), pt


def run_pattern(name, origins, modes, backend, ref_var, ref_pt):
    t = make_tessellation(name, centroid(origins), seed=1)
    p = Params(coarse_res=COARSE, fine_res=FINE, k_c=400, k_refine=10, t_max=240.0,
               tessellation=name)
    poly = region_polygon(origins)
    n_coarse = len(t.cells_in(poly, COARSE))
    fine_cells = t.cells_in(poly, FINE)
    n_fine = len(fine_cells)

    ev = CachedEvaluator(backend)
    best, _r, _s, _d = fair_meeting_point(origins, modes, ev, p, tess=t)
    if best is None:
        return None
    ours = metrics.variance(best.times)
    ours_calls = ev.calls

    ev2 = CachedEvaluator(backend)
    own_pt = exhaustive_variance(origins, modes, ev2, res=FINE, tess=t)
    own_times = [ev2.effective(o, own_pt, m) for o, m in zip(origins, modes)]
    own_opt = metrics.variance(own_times)
    own_calls = ev2.calls

    ev3 = CachedEvaluator(backend)
    _cb, cert, _d3 = certified_search(origins, modes, ev3, p, euclidean_lipschitz(modes), tess=t)

    # distance from the pattern's best cell to the fine reference optimum
    quant_km = haversine_km(own_pt, ref_pt) if (own_pt and ref_pt) else float("nan")

    def pct(x, base):
        return 100.0 * (x - base) / base if base > 1e-12 else 0.0

    return {
        "pattern": name,
        "n_coarse": n_coarse,
        "n_fine": n_fine,
        "ours_variance": round(ours, 4),
        "own_grid_opt": round(own_opt, 4),
        "ref_opt": round(ref_var, 4),
        # search failure: how far the coarse-to-fine search lands from the best cell
        # this pattern could have offered
        "seam_gap_pct": round(pct(ours, own_opt), 4),
        # discretization penalty: how much the pattern itself costs against a 7x finer grid
        "quant_gap_pct": round(pct(own_opt, ref_var), 4),
        "total_gap_pct": round(pct(ours, ref_var), 4),
        "quant_km": round(quant_km, 4),
        "ours_calls": ours_calls,
        "exhaustive_calls": own_calls,
        "call_ratio": round(ours_calls / own_calls, 4) if own_calls else float("nan"),
        "certified": bool(cert.get("certified")),
        "certified_rounds": cert.get("rounds", -1),
        "certified_calls": cert.get("routing_calls", -1),
        "certified_gap_pct": round(pct(cert.get("v_star", float("nan")), ref_var), 4),
    }


def main():
    os.makedirs("outputs", exist_ok=True)
    backend = EuclideanBackend()
    rows = []
    t0 = time.time()
    total = len(CITIES) * len(NS) * len(SEEDS)
    done = 0

    for city in CITIES:
        for n in NS:
            for seed in SEEDS:
                origins = sample_origins(city, n, seed=seed, spread="clustered",
                                         clusters=2, cluster_sd_deg=0.04)
                modes = assign_modes(n, "mixed", seed=seed)
                ref_var, ref_pt = reference_optimum(origins, modes, backend)
                if ref_var is None:
                    continue
                for name in ALL_TESSELLATIONS:
                    r = run_pattern(name, origins, modes, backend, ref_var, ref_pt)
                    if r is None:
                        continue
                    r.update(city=city, n=n, seed=seed)
                    rows.append(r)
                done += 1
                if done % 10 == 0:
                    print("  %d/%d instances, %.0fs" % (done, total, time.time() - t0), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("outputs/tessellation.csv", index=False)

    agg = df.groupby("pattern").agg(
        instances=("seam_gap_pct", "size"),
        cand_fine=("n_fine", "mean"),
        seam_gap=("seam_gap_pct", "mean"),
        seam_fail=("seam_gap_pct", lambda s: 100.0 * (s > 1e-6).mean()),
        quant_gap=("quant_gap_pct", "mean"),
        total_gap=("total_gap_pct", "mean"),
        quant_km=("quant_km", "mean"),
        calls=("ours_calls", "mean"),
        call_ratio=("call_ratio", "mean"),
        cert_rate=("certified", lambda s: 100.0 * s.mean()),
        cert_calls=("certified_calls", "mean"),
    ).sort_values("total_gap")

    pd.set_option("display.width", 200)
    print("\nTessellation ablation, matched candidate budget, Euclidean backend")
    print("\n%s" % agg.round(3).to_string())
    print("\nseam_gap  = ours vs the best cell this pattern offers      (search failure)")
    print("quant_gap = that best cell vs a 7x finer reference grid     (discretization cost)")
    print("total_gap = ours vs the reference grid                      (what the user gets)")
    print("seam_fail = share of instances where the search missed its own grid optimum")

    agg.round(4).to_csv("outputs/tessellation_summary.csv")
    print("\nwrote outputs/tessellation.csv and outputs/tessellation_summary.csv")

    # The certificate claims optimality over the pattern's OWN fine grid, never over the
    # finer reference grid. Comparing against ref_opt conflates the discretization gap with
    # a soundness failure and reports false alarms.
    v_star = df.ref_opt * (1 + df.certified_gap_pct / 100.0)
    bad = df[(df.certified) & (v_star > df.own_grid_opt + 1e-6)]
    print("certificate soundness violations vs own grid: %d  [must be 0]" % len(bad))
    print("mean certified gap vs the finer reference grid: %.3f%%  [discretization, not a failure]"
          % df[df.certified].certified_gap_pct.mean())


if __name__ == "__main__":
    main()
