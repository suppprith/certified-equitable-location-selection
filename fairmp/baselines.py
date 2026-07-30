
from __future__ import annotations

import math
import random

from . import metrics
from .candidates import polyfill_centroids, region_polygon
from .geo import LatLng, centroid
from .travel_time import EUCLIDEAN_SPEED_KMH

def geometric_centroid(origins, modes_list=None, evaluator=None, **kw):
    return centroid(origins)

def weighted_centroid(origins, modes_list, evaluator=None, **kw):

    ws = [1.0 / max((EUCLIDEAN_SPEED_KMH.get(m, 1.0) for m in modes), default=1.0) for modes in modes_list]
    sw = sum(ws)
    lat = sum(o.lat * w for o, w in zip(origins, ws)) / sw
    lng = sum(o.lng * w for o, w in zip(origins, ws)) / sw
    return LatLng(lat, lng)

def geometric_median(origins, modes_list=None, evaluator=None, iters=200, eps=1e-9, **kw):

    x = sum(o.lng for o in origins) / len(origins)
    y = sum(o.lat for o in origins) / len(origins)
    for _ in range(iters):
        nx = ny = den = 0.0
        for o in origins:
            d = math.hypot(o.lng - x, o.lat - y) or eps
            w = 1.0 / d
            nx += o.lng * w
            ny += o.lat * w
            den += w
        ux, uy = nx / den, ny / den
        if math.hypot(ux - x, uy - y) < eps:
            break
        x, y = ux, uy
    return LatLng(y, x)

def _grid_search(origins, modes_list, evaluator, res, key, bucket="static", tess=None):
    n = len(origins)
    best, best_val = None, math.inf
    grid = (tess.cells_in(region_polygon(origins), res) if tess
            else polyfill_centroids(region_polygon(origins), res))
    for _c, pt in grid:
        times = [evaluator.effective(o, pt, modes, bucket) for o, modes in zip(origins, modes_list)]
        if not metrics.all_reachable(times, n):
            continue
        v = key(times)
        if v < best_val:
            best_val, best = v, pt
    return best

def min_sum(origins, modes_list, evaluator, res=9, bucket="static", **kw):
    return _grid_search(origins, modes_list, evaluator, res, metrics.total_time, bucket, kw.get("tess"))

def min_max(origins, modes_list, evaluator, res=9, bucket="static", **kw):
    return _grid_search(origins, modes_list, evaluator, res, metrics.max_time, bucket, kw.get("tess"))

def exhaustive_variance(origins, modes_list, evaluator, res=9, bucket="static", **kw):

    return _grid_search(origins, modes_list, evaluator, res, metrics.variance, bucket, kw.get("tess"))

def exhaustive_ede(origins, modes_list, evaluator, res=9, bucket="static", **kw):

    return _grid_search(origins, modes_list, evaluator, res, metrics.kolm_pollak_ede, bucket, kw.get("tess"))

def min_range(origins, modes_list, evaluator, res=9, bucket="static", **kw):

    return _grid_search(origins, modes_list, evaluator, res, metrics.spread, bucket, kw.get("tess"))

def random_best(origins, modes_list, evaluator, res=9, samples=100, seed=0, bucket="static", **kw):
    rng = random.Random(seed)
    tess = kw.get("tess")
    grid = (tess.cells_in(region_polygon(origins), res) if tess
            else polyfill_centroids(region_polygon(origins), res))
    if not grid:
        return None
    n = len(origins)
    best, best_val = None, math.inf
    for _c, pt in rng.sample(grid, min(samples, len(grid))):
        times = [evaluator.effective(o, pt, modes, bucket) for o, modes in zip(origins, modes_list)]
        if not metrics.all_reachable(times, n):
            continue
        v = metrics.variance(times)
        if v < best_val:
            best_val, best = v, pt
    return best
