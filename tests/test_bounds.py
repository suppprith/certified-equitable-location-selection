from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fairmp import metrics
from fairmp import objectives as ob
from fairmp.algorithm import Params
from fairmp.bands import band_box, band_certified_search, box_search, make_thresholds
from fairmp.bounds import (Grid, alt_boxes, band_index_box, geometric_boxes, pick_landmarks,
                           polygon_band_index)
from fairmp.candidates import region_polygon
from fairmp.certificate import euclidean_lipschitz, screened_lipschitz_search
from fairmp.geo import centroid
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.tessellation import make_tessellation
from fairmp.travel_time import EUCLIDEAN_SPEED_KMH, CachedEvaluator, EuclideanBackend

P = Params(coarse_res=8, fine_res=9, k_c=400, k_refine=10, t_max=240.0)


def _setup(seed, n=5):
    backend = EuclideanBackend()
    origins = sample_origins("london", n, seed=seed, spread="clustered",
                             clusters=2, cluster_sd_deg=0.03)
    modes = assign_modes(n, "mixed", seed=seed)
    tess = make_tessellation("h3", centroid(origins))
    cells = tess.cells_in(region_polygon(origins), 9)
    pts = [pt for _c, pt in cells]
    truth = [[backend.minutes(o, p, m[0]) for o, m in zip(origins, modes)] for p in pts]
    return backend, origins, modes, tess, cells, pts, truth


def _optimum(truth):
    return min(metrics.variance(t) for t in truth if all(math.isfinite(x) for x in t))


def test_box_search_with_exact_boxes_evaluates_only_the_optimum_and_ties():
    _b, _o, _m, _t, _c, _p, truth = _setup(1)
    boxes = [(t, t) for t in truth]
    j, val, info = box_search(boxes, lambda j: truth[j], len(truth[0]))
    assert abs(val - _optimum(truth)) < 1e-9
    # a box of zero width has bound equal to the value, so only ties can be evaluated
    assert info["evaluated"] <= sum(1 for t in truth if metrics.variance(t) <= val + 1e-9)


def test_band_search_without_the_screen_reaches_the_same_optimum():
    backend, origins, modes, tess, _c, _p, truth = _setup(2)
    thr = make_thresholds(0.0, 120.0, 12)
    for screen in (True, False):
        ev = CachedEvaluator(backend)
        _inc, info = band_certified_search(origins, modes, ev, P, thr, backend, tess=tess,
                                           screen=screen)
        assert abs(info["value"] - _optimum(truth)) < 1e-9, (screen, info)


def test_screened_lipschitz_is_exact_where_the_constant_holds():
    # on the Euclidean backend the speed-derived constant is exactly tight
    for seed in (0, 3):
        backend, origins, modes, tess, _c, _p, truth = _setup(seed)
        ev = CachedEvaluator(backend)
        lip = euclidean_lipschitz(modes)
        _cell, cert = screened_lipschitz_search(origins, modes, ev, P, lip,
                                                feasible=lambda pt: True, tess=tess)
        assert abs(cert["v_star"] - _optimum(truth)) < 1e-9, (seed, cert)


def test_geometric_boxes_contain_the_truth_and_prune_nothing_for_variance():
    _b, origins, modes, _t, _c, pts, truth = _setup(4)
    boxes = geometric_boxes(origins, modes, pts, vmax=EUCLIDEAN_SPEED_KMH)
    for (lo, hi), t in zip(boxes, truth):
        assert all(a <= x <= b for a, x, b in zip(lo, t, hi))
        assert ob.variance_min(lo, hi) == 0.0
        assert ob.min_max_min(lo, hi) <= max(t)


def test_alt_boxes_contain_the_truth_on_a_metric_backend():
    backend, origins, modes, _t, _c, pts, truth = _setup(5)
    lm = pick_landmarks(pts)
    used = {m[0] for m in modes if m[0] != "transit"}
    from_l = {m: [[backend.minutes(pts[k], p, m) for p in pts] for k in lm] for m in used}
    l_to_o = {m: [[backend.minutes(pts[k], o, m) for o in origins] for k in lm] for m in used}
    o_to_l = [[backend.minutes(o, pts[k], m[0]) for k in lm] for o, m in zip(origins, modes)]
    boxes = alt_boxes(origins, modes, pts, from_l, l_to_o, o_to_l, vmax=EUCLIDEAN_SPEED_KMH)
    for (lo, hi), t in zip(boxes, truth):
        assert all(a - 1e-9 <= x <= b + 1e-9 for a, x, b in zip(lo, t, hi))


def test_band_index_box_brackets_the_band():
    thr = make_thresholds(0.0, 120.0, 12)
    assert band_index_box(3, thr) == (20.0, 30.0)
    assert band_index_box(3, thr, widen=1) == (10.0, 40.0)
    assert band_index_box(12, thr, widen=1)[1] == math.inf
    assert band_index_box(1, thr, widen=1)[0] == 0.0


def test_polygon_bands_match_on_a_smooth_field_and_denoising_drops_islands():
    from shapely.geometry import box
    poly = box(-0.10, 51.50, -0.05, 51.53)
    grid = Grid(poly, 0.2)
    ny, nx = grid.shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    smooth = (0.5 * (xx + yy)).ravel()            # a ramp, minutes
    thr = make_thresholds(0.0, float(smooth.max()) + 1, 6)
    rng = random.Random(0)
    pts = []
    for _ in range(300):
        r, c = rng.uniform(1, ny - 2), rng.uniform(1, nx - 2)
        pts.append((r, c))
    from fairmp.geo import LatLng

    def at(r, c):
        y = grid.ys[0] + r * grid.s
        x = grid.xs[0] + c * grid.s
        return LatLng(grid.lat0 + y / grid.ky, grid.lng0 + x / grid.kx)

    cands = [at(r, c) for r, c in pts]
    exact = [band_box([0.5 * (c + r)], thr) for r, c in pts]
    got = polygon_band_index(grid, smooth, cands, thr, "contour")
    agree = sum(1 for k, (lo, hi) in zip(got, exact) if k is not None
                and thr[k - 1] <= lo[0] and hi[0] <= thr[k])
    assert agree >= 0.95 * len(pts), agree

    # an island of fast access far from the main reachable area is dropped by denoising
    T = np.full(grid.shape, 1000.0)
    T[0:3, 0:3] = 1.0
    T[ny - 2:, nx - 2:] = 1.0
    thr2 = make_thresholds(0.0, 10.0, 2)
    island = at(ny - 1.5, nx - 1.5)
    main = at(1.0, 1.0)
    got2 = polygon_band_index(grid, T.ravel(), [main, island], thr2, "contour")
    assert got2[0] == 1 and got2[1] is None, got2
    near = polygon_band_index(grid, T.ravel(), [main, island], thr2, "nearest")
    assert near == [1, 1], near


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, "PASS")
    print("all bound tests passed")
