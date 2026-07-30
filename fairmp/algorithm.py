
from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import Point

from . import metrics
from .candidates import prefilter, region_polygon
from .geo import LatLng, centroid, haversine_km
from .tessellation import make_tessellation

@dataclass
class Params:
    coarse_res: int = 7
    fine_res: int = 9
    k_c: int = 200
    k_refine: int = 10
    ring: int = 1
    t_max: float = 60.0
    gamma: float = 0.0
    objective: str = "variance"
    ede_epsilon: float = metrics.EDE_EPSILON
    runners_up: int = 2
    min_sep_km: float = 1.0
    tessellation: str = "h3"

def resolve_tessellation(origins, params, tess=None):
    return tess or make_tessellation(params.tessellation, centroid(origins))

@dataclass
class Area:
    cell: str
    point: LatLng
    times: list
    objective: float
    feasible: bool

def _score(cell, point, origins, modes_list, ev, p, bucket, weights=None) -> Area:
    times = [ev.effective(o, point, modes, bucket) for o, modes in zip(origins, modes_list)]
    n = len(origins)
    if not metrics.all_reachable(times, n):
        obj = math.inf
    elif p.objective == "ede":
        ede = (metrics.wkolm_pollak_ede(times, weights, p.ede_epsilon) if weights is not None
               else metrics.kolm_pollak_ede(times, p.ede_epsilon))
        obj = ede + p.gamma * (metrics.wmean(times, weights) if weights is not None else metrics.mean_time(times))
    elif weights is None:
        obj = metrics.objective(times, n, p.gamma)
    else:
        obj = metrics.wvariance(times, weights) + p.gamma * metrics.wmean(times, weights)
    return Area(cell, point, times, obj, metrics.feasible(times, n, p.t_max))

def fair_meeting_point(origins, modes_list, evaluator, params: Params | None = None,
                       is_dead=None, bucket: str = "static", weights=None, tess=None):

    p = params or Params()
    t = resolve_tessellation(origins, p, tess)

    poly = region_polygon(origins)
    coarse = t.cells_in(poly, p.coarse_res)
    if is_dead:
        coarse = [(c, pt) for c, pt in coarse if not is_dead(c)]
    coarse = prefilter(coarse, origins, p.k_c)

    coarse_scored = [_score(c, pt, origins, modes_list, evaluator, p, bucket, weights) for c, pt in coarse]
    seen = {a.cell for a in coarse_scored}

    fine_scored: list[Area] = []
    finite = sorted((a for a in coarse_scored if math.isfinite(a.objective)), key=lambda a: a.objective)
    for a in finite[: p.k_refine]:
        for c, pt in t.refine(a.cell, p.fine_res, p.ring):
            if c in seen:
                continue
            seen.add(c)
            if is_dead and is_dead(c):
                continue

            if not poly.contains(Point(pt.lng, pt.lat)):
                continue
            fine_scored.append(_score(c, pt, origins, modes_list, evaluator, p, bucket, weights))

    all_scored = coarse_scored + fine_scored

    src = fine_scored if fine_scored else coarse_scored
    feasible = [a for a in src if a.feasible]
    pool = feasible if feasible else [a for a in src if math.isfinite(a.objective)]
    diag = {"coarse": len(coarse), "evaluated": len(all_scored), "routing_calls": evaluator.calls}
    if not pool:
        return None, [], all_scored, diag

    pool.sort(key=lambda a: a.objective)
    best = pool[0]
    runners: list[Area] = []
    for a in pool[1:]:
        if len(runners) >= p.runners_up:
            break
        if all(haversine_km(a.point, b.point) >= p.min_sep_km for b in [best, *runners]):
            runners.append(a)
    return best, runners, all_scored, diag
