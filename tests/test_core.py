
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fairmp import metrics
from fairmp.algorithm import Params, fair_meeting_point
from fairmp.baselines import exhaustive_variance, geometric_centroid, min_range
from fairmp.certificate import (certificate_report, certified_search, euclidean_lipschitz,
                                min_variance_over_box)
from fairmp.runner import run_instance
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.travel_time import CachedEvaluator, EuclideanBackend

def test_metrics_basic():
    assert metrics.variance([10, 10, 10]) == 0
    assert abs(metrics.jain([10, 10, 10]) - 1.0) < 1e-9
    assert abs(metrics.gini([10, 10, 10])) < 1e-9
    assert not metrics.all_reachable([10, math.inf, 5], 3)
    assert metrics.feasible([10, 12], 2, t_max=60)
    assert not metrics.feasible([10, 80], 2, t_max=60)

def test_ede_properties():

    assert abs(metrics.kolm_pollak_ede([12, 12, 12]) - 12.0) < 1e-6

    a = [10, 20, 30]
    b = [5, 20, 35]
    assert metrics.kolm_pollak_ede(a) > metrics.mean_time(a)
    assert metrics.kolm_pollak_ede(b) > metrics.kolm_pollak_ede(a)

    assert abs(metrics.wkolm_pollak_ede([12, 12, 12], [1, 2, 3]) - 12.0) < 1e-6

def test_eq8_two_user_variance_is_range_squared_over_four():

    t1, t2 = 10.0, 20.0
    assert abs(metrics.variance([t1, t2]) - ((t1 - t2) / 2) ** 2) < 1e-9

def test_algorithm_is_fair_and_near_optimal():
    n = 5
    origins = sample_origins("london", n, seed=7, spread="clustered", clusters=1, cluster_sd_deg=0.025)
    modes = assign_modes(n, mix="mixed", seed=7)
    backend = EuclideanBackend()
    p = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0)

    ev = CachedEvaluator(backend)
    best, _r, _s, _d = fair_meeting_point(origins, modes, ev, p)
    assert best is not None and best.feasible

    ev2 = CachedEvaluator(backend)
    cpt = geometric_centroid(origins, modes, ev2)
    ctimes = [ev2.effective(o, cpt, m) for o, m in zip(origins, modes)]
    cvar = metrics.variance(ctimes)

    ev3 = CachedEvaluator(backend)
    xpt = exhaustive_variance(origins, modes, ev3, res=9)
    xtimes = [ev3.effective(o, xpt, m) for o, m in zip(origins, modes)]
    xvar = metrics.variance(xtimes)

    our_var = metrics.variance(best.times)

    assert xvar <= cvar + 1e-9

    assert our_var <= xvar * 1.5 + 1e-6

def test_min_range_minimises_spread():
    n = 5
    origins = sample_origins("london", n, seed=3, spread="clustered", clusters=1, cluster_sd_deg=0.025)
    modes = assign_modes(n, mix="mixed", seed=3)
    backend = EuclideanBackend()
    ev = CachedEvaluator(backend)
    rpt = min_range(origins, modes, ev, res=9)
    rtimes = [ev.effective(o, rpt, m) for o, m in zip(origins, modes)]
    ev2 = CachedEvaluator(backend)
    cpt = geometric_centroid(origins, modes, ev2)
    ctimes = [ev2.effective(o, cpt, m) for o, m in zip(origins, modes)]

    assert metrics.spread(rtimes) <= metrics.spread(ctimes) + 1e-9

def test_ede_objective_variant_beats_centroid_on_ede():
    n = 5
    origins = sample_origins("london", n, seed=11, spread="clustered", clusters=1, cluster_sd_deg=0.03)
    modes = assign_modes(n, mix="mixed", seed=11)
    backend = EuclideanBackend()
    p = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0)
    rows = {r["method"]: r for r in run_instance(origins, modes, backend, p, fine_res=9, variants=("ede",))}
    assert "ours_ede" in rows and "exhaustive_ede" in rows

    assert rows["ours_ede"]["ede"] <= rows["centroid"]["ede"] + 1e-6
    assert rows["ours_ede"]["ede"] <= rows["exhaustive_ede"]["ede"] * 1.5 + 1e-6

def test_assign_modes_with_choice_structure():
    from fairmp.scenarios import assign_modes_with_choice
    m = assign_modes_with_choice(5, seed=1, frac_choice=0.4)
    assert len(m) == 5
    multi = [x for x in m if len(x) > 1]
    assert len(multi) >= 1
    for x in m:
        assert 1 <= len(x) <= 2 and len(set(x)) == len(x)

def test_effective_takes_min_over_mode_set():
    from fairmp.geo import LatLng
    from fairmp.travel_time import CachedEvaluator, PrecomputedBackend
    o = LatLng(51.5, -0.1)
    a = LatLng(51.51, -0.12)
    b = LatLng(51.49, -0.08)
    pre = PrecomputedBackend()
    pre.put("walking", o, a, 10.0)
    pre.put("cycling", o, a, 14.0)
    pre.put("walking", o, b, 20.0)
    pre.put("cycling", o, b, 12.0)
    ev = CachedEvaluator(pre)

    assert ev.effective(o, a, ["walking", "cycling"]) == 10.0
    assert ev.effective(o, b, ["walking", "cycling"]) == 12.0

    assert ev.effective(o, b, ["walking"]) == 20.0

def test_min_variance_over_box():

    assert min_variance_over_box([10, 12], [14, 16]) == 0.0

    assert abs(min_variance_over_box([0, 3], [1, 4]) - 1.0) < 1e-6

    assert abs(min_variance_over_box([0, 10, 20], [0, 10, 20]) - 200.0 / 3) < 1e-4

def test_certificate_sound_and_adaptive_exact():
    n = 5
    backend = EuclideanBackend()
    p = Params(coarse_res=8, fine_res=9, k_c=400, k_refine=10, t_max=240.0)
    for seed in (0, 1, 7):
        origins = sample_origins("london", n, seed=seed, spread="clustered",
                                 clusters=2, cluster_sd_deg=0.04)
        modes = assign_modes(n, "mixed", seed=seed)
        lip = euclidean_lipschitz(modes)

        ev = CachedEvaluator(backend)
        xpt = exhaustive_variance(origins, modes, ev, res=9)
        xvar = metrics.variance([ev.effective(o, xpt, m) for o, m in zip(origins, modes)])

        ev2 = CachedEvaluator(backend)
        best, _r, scored, _d = fair_meeting_point(origins, modes, ev2, p)
        rep = certificate_report(origins, modes, ev2, p, best, scored, lip)
        if rep["certified"]:
            assert metrics.variance(best.times) <= xvar + 1e-9

        ev3 = CachedEvaluator(backend)
        _cbest, cert, _diag = certified_search(origins, modes, ev3, p, lip)
        assert cert["certified"]
        assert cert["v_star"] <= xvar + 1e-9

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, "PASS")
    print("all tests passed")
