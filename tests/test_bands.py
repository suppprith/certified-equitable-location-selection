
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fairmp import metrics
from fairmp import objectives as ob
from fairmp.algorithm import Params
from fairmp.bands import (band_box, band_certified_search, band_interval, iso_int,
                          lipschitz_box, make_thresholds)
from fairmp.candidates import region_polygon
from fairmp.geo import centroid
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.tessellation import make_tessellation
from fairmp.travel_time import CachedEvaluator, EuclideanBackend

P = Params(coarse_res=8, fine_res=9, k_c=400, k_refine=10, t_max=240.0)


def _setup(seed, n=5, k=60):
    backend = EuclideanBackend()
    origins = sample_origins("london", n, seed=seed, spread="clustered",
                             clusters=2, cluster_sd_deg=0.04)
    modes = assign_modes(n, "mixed", seed=seed)
    tess = make_tessellation("h3", centroid(origins))
    cells = tess.cells_in(region_polygon(origins), 9)
    ev = CachedEvaluator(backend)
    times = []
    for _c, pt in cells:
        times += [ev.effective(o, pt, m) for o, m in zip(origins, modes)]
    finite = [t for t in times if math.isfinite(t)]
    return backend, origins, modes, tess, cells, make_thresholds(0.0, max(finite), k), finite


def test_band_interval_brackets_the_value():
    thr = make_thresholds(0.0, 100.0, 10)
    for t in (0.0, 3.2, 10.0, 55.5, 99.9):
        lo, hi = band_interval(t, thr)
        assert lo <= t <= hi, (t, lo, hi)
    lo, hi = band_interval(math.inf, thr)
    assert lo == thr[-1] and hi == math.inf


def test_band_box_is_sound_for_the_true_times():
    thr = make_thresholds(0.0, 120.0, 40)
    times = [12.3, 44.9, 71.0, 5.5]
    lo, hi = band_box(times, thr)
    for t, a, b in zip(times, lo, hi):
        assert a <= t <= b


def test_band_bound_never_exceeds_the_true_objective():
    # soundness: the bound from a band box must not exceed what the true times achieve
    for seed in (0, 3, 5):
        backend, origins, modes, tess, cells, thr, _f = _setup(seed)
        ev = CachedEvaluator(backend)
        for _c, pt in cells[:150]:
            times = [ev.effective(o, pt, m) for o, m in zip(origins, modes)]
            if not metrics.all_reachable(times, len(origins)):
                continue
            lo, hi = band_box(times, thr)
            assert ob.variance_min(lo, hi) <= metrics.variance(times) + 1e-9


def test_band_certified_search_returns_the_exhaustive_optimum():
    for objective in ("variance", "min_max", "min_sum", "range", "ede"):
        for seed in (0, 4):
            backend, origins, modes, tess, cells, thr, finite = _setup(seed)
            kappa = ob.calibrate_kappa(finite) if objective == "ede" else None

            ev1 = CachedEvaluator(backend)
            best = math.inf
            for _c, pt in cells:
                ts = [ev1.effective(o, pt, m) for o, m in zip(origins, modes)]
                if not metrics.all_reachable(ts, len(origins)):
                    continue
                v = (ob.EVAL[objective](ts, None, kappa) if objective == "ede"
                     else ob.EVAL[objective](ts))
                best = min(best, v)

            ev2 = CachedEvaluator(backend)
            inc, info = band_certified_search(origins, modes, ev2, P, thr, backend,
                                              objective=objective, tess=tess, kappa=kappa)
            assert inc is not None, (objective, seed)
            assert info["value"] <= best + 1e-6, (objective, seed, info["value"], best)


def test_band_certified_search_evaluates_a_small_fraction():
    backend, origins, modes, tess, cells, thr, _f = _setup(0)
    ev = CachedEvaluator(backend)
    _inc, info = band_certified_search(origins, modes, ev, P, thr, backend, tess=tess)
    assert info["sweeps"] == len(origins)
    assert info["eval_fraction"] < 0.5, info["eval_fraction"]


def test_finer_bands_do_not_loosen_the_bound():
    backend, origins, modes, tess, cells, _thr, finite = _setup(2)
    ev = CachedEvaluator(backend)
    pt = cells[len(cells) // 2][1]
    times = [ev.effective(o, pt, m) for o, m in zip(origins, modes)]
    prev = -1.0
    for k in (10, 20, 40, 80, 160):
        thr = make_thresholds(0.0, max(finite), k)
        lo, hi = band_box(times, thr)
        v = ob.variance_min(lo, hi)
        assert v <= metrics.variance(times) + 1e-9
        assert v >= prev - 1e-9, (k, v, prev)
        prev = v


def test_iso_int_reports_both_variants_and_a_region():
    backend, origins, modes, tess, cells, thr, _f = _setup(1)
    pts = [pt for _c, pt in cells]
    r = iso_int(origins, modes, backend, thr, pts)
    assert r is not None
    assert r["region_size"] >= 1
    assert len(r["centroid_times"]) == len(origins)
    # the oracle variant ranks the same region with our times, so it cannot be worse
    assert (metrics.variance(r["oracle_times"])
            <= metrics.variance(r["centroid_times"]) + 1e-9)


def test_lipschitz_box_brackets_and_widens_with_distance():
    times = [10.0, 20.0]
    lip = [3.0, 5.0]
    lo1, hi1 = lipschitz_box(times, 0.5, lip)
    lo2, hi2 = lipschitz_box(times, 1.5, lip)
    for t, a, b in zip(times, lo1, hi1):
        assert a <= t <= b
    assert lo2[0] < lo1[0] and hi2[0] > hi1[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, "PASS")
    print("all band tests passed")
