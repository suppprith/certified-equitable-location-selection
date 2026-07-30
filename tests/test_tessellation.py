
from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shapely.geometry import Point

from fairmp import metrics
from fairmp.algorithm import Params, fair_meeting_point
from fairmp.baselines import exhaustive_variance
from fairmp.candidates import polyfill_centroids, refine_cells, region_polygon
from fairmp.certificate import certified_search, euclidean_lipschitz
from fairmp.geo import LatLng, centroid, haversine_km
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.tessellation import (ALL_TESSELLATIONS, SQRT2, TruncatedSquareTessellation,
                                 hex_area_km2, make_tessellation)
from fairmp.travel_time import CachedEvaluator, EuclideanBackend

ORIGINS = sample_origins("london", 6, seed=3, spread="clustered", clusters=2, cluster_sd_deg=0.04)
POLY = region_polygon(ORIGINS)
ANCHOR = centroid(ORIGINS)

def test_h3_tessellation_matches_legacy_helpers():

    t = make_tessellation("h3", ANCHOR)
    assert dict(t.cells_in(POLY, 8)).keys() == dict(polyfill_centroids(POLY, 8)).keys()
    cell = sorted(dict(polyfill_centroids(POLY, 8)))[0]
    assert {c for c, _ in t.refine(cell, 9, 1)} == {c for c, _ in refine_cells(cell, 9, 1)}

def test_truncated_square_geometry_is_regular():

    a = 1.0
    s = a * (SQRT2 - 1.0)
    v = [(a / 2, s / 2), (s / 2, a / 2), (-s / 2, a / 2), (-a / 2, s / 2),
         (-a / 2, -s / 2), (-s / 2, -a / 2), (s / 2, -a / 2), (a / 2, -s / 2)]
    edges = [math.dist(v[i], v[(i + 1) % 8]) for i in range(8)]
    assert max(edges) - min(edges) < 1e-12
    assert abs(edges[0] - s) < 1e-12

    oct_area = 2 * (1 + SQRT2) * s * s
    assert abs(oct_area + s * s - a * a) < 1e-12

def test_truncated_square_partitions_the_plane():

    t = TruncatedSquareTessellation(ANCHOR)
    a = t.pitch(9)
    rng = random.Random(0)
    kinds = {"o": 0, "q": 0}
    for _ in range(20000):
        x, y = rng.uniform(-20, 20), rng.uniform(-20, 20)
        cell = t.cell_at(t.proj.to_latlng(x, y), 9)
        cx, cy = t._center_xy(cell)
        dx, dy = x - cx, y - cy
        if cell.startswith("o"):
            assert abs(dx) <= a / 2 + 1e-9 and abs(dy) <= a / 2 + 1e-9
            assert abs(dx) + abs(dy) <= a / SQRT2 + 1e-9
        else:
            assert abs(dx) + abs(dy) <= a * (SQRT2 - 1.0) / SQRT2 + 1e-9
        kinds[cell[0]] += 1
    share = kinds["o"] / (kinds["o"] + kinds["q"])
    assert abs(share - 2 * (SQRT2 - 1.0)) < 0.01

def test_truncated_square_neighbour_degrees():

    t = TruncatedSquareTessellation(ANCHOR)
    assert len(t.neighbors("o:9:3:4")) == 8
    assert len(t.neighbors("q:9:3:4")) == 4

    for oc in t.neighbors("q:9:3:4"):
        assert "q:9:3:4" in t.neighbors(oc)

def test_all_tessellations_match_candidate_density():

    ref = len(make_tessellation("h3", ANCHOR).cells_in(POLY, 9))
    for name in ALL_TESSELLATIONS:
        n = len(make_tessellation(name, ANCHOR, seed=1).cells_in(POLY, 9))
        assert 0.75 * ref <= n <= 1.25 * ref, (name, n, ref)

def test_parent_child_round_trip():

    for name in ALL_TESSELLATIONS:
        t = make_tessellation(name, ANCHOR, seed=1)
        t.cells_in(POLY, 8)
        for cell, _pt in t.cells_in(POLY, 9)[:60]:
            parent = t.parent(cell, 8)
            assert t.level(parent) == 8
            assert cell in {c for c, _ in t.children(parent, 9)}, (name, cell)

def test_cell_at_agrees_with_center():

    for name in ALL_TESSELLATIONS:
        t = make_tessellation(name, ANCHOR, seed=1)
        cells = t.cells_in(POLY, 9)
        for cell, pt in cells[:80]:
            assert t.cell_at(pt, 9) == cell, (name, cell)

def test_circumradius_bounds_distance_to_cell_centre():

    # the Lipschitz box in certificate.py is only sound if every point of a cell lies
    # within circumradius of that cell's centre
    rng = random.Random(4)
    lng0, lat0, lng1, lat1 = POLY.bounds
    probes = []
    while len(probes) < 3000:
        p = LatLng(rng.uniform(lat0, lat1), rng.uniform(lng0, lng1))
        if POLY.contains(Point(p.lng, p.lat)):
            probes.append(p)

    for name in ALL_TESSELLATIONS:
        t = make_tessellation(name, ANCHOR, seed=1)
        t.cells_in(POLY, 9)
        worst = 0.0
        for p in probes:
            cell = t.cell_at(p, 9)
            r = t.circumradius_km(cell)
            worst = max(worst, haversine_km(p, t.center(cell)) / r)
        assert worst <= 1.0 + 1e-3, (name, worst)

def test_search_runs_on_every_tessellation():

    backend = EuclideanBackend()
    modes = assign_modes(len(ORIGINS), "mixed", seed=3)
    for name in ALL_TESSELLATIONS:
        p = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=240.0, tessellation=name)
        ev = CachedEvaluator(backend)
        best, _r, _s, _d = fair_meeting_point(ORIGINS, modes, ev, p)
        assert best is not None, name
        assert math.isfinite(metrics.variance(best.times))

def test_certified_search_is_sound_on_every_tessellation():

    backend = EuclideanBackend()
    n = 5
    for name in ALL_TESSELLATIONS:
        for seed in (0, 7):
            origins = sample_origins("london", n, seed=seed, spread="clustered",
                                     clusters=2, cluster_sd_deg=0.04)
            modes = assign_modes(n, "mixed", seed=seed)
            p = Params(coarse_res=8, fine_res=9, k_c=400, k_refine=10, t_max=240.0,
                       tessellation=name)
            t = make_tessellation(name, centroid(origins), seed=1)
            lip = euclidean_lipschitz(modes)

            ev = CachedEvaluator(backend)
            xpt = exhaustive_variance(origins, modes, ev, res=9, tess=t)
            xvar = metrics.variance([ev.effective(o, xpt, m) for o, m in zip(origins, modes)])

            ev2 = CachedEvaluator(backend)
            _best, cert, _d = certified_search(origins, modes, ev2, p, lip, tess=t)
            assert cert["certified"], (name, seed)
            assert cert["v_star"] <= xvar + 1e-9, (name, seed, cert["v_star"], xvar)

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, "PASS")
    print("all tessellation tests passed")
