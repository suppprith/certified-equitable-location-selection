
from __future__ import annotations

import math
from collections import defaultdict

from . import metrics
from .algorithm import Area, Params, fair_meeting_point, resolve_tessellation
from .candidates import region_polygon
from .geo import haversine_km
from .travel_time import EUCLIDEAN_SPEED_KMH


def euclidean_lipschitz(modes_list, detour: float = 1.3) -> list[float]:

    out = []
    for modes in modes_list:
        speed = min(EUCLIDEAN_SPEED_KMH.get(m, 4.8) for m in modes)
        out.append(60.0 * detour / speed)
    return out


def min_variance_over_box(lo: list[float], hi: list[float]) -> float:

    if max(lo) <= min(hi):
        return 0.0

    def phi(mu):
        s = 0.0
        for a, b in zip(lo, hi):
            if mu < a:
                s += (a - mu) ** 2
            elif mu > b:
                s += (mu - b) ** 2
        return s

    left, right = min(lo), max(hi)
    for _ in range(80):
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        if phi(m1) <= phi(m2):
            right = m2
        else:
            left = m1
    return phi((left + right) / 2) / len(lo)


def _uncovered_by_parent(origins, params: Params, evaluated_fine: set[str], tess):

    poly = region_polygon(origins)
    uncovered = defaultdict(list)
    for cell, pt in tess.cells_in(poly, params.fine_res):
        if cell not in evaluated_fine:
            uncovered[tess.parent(cell, params.coarse_res)].append((cell, pt))
    return uncovered


def _parent_bound(parent_pt, fines, times, lipschitz) -> float:

    if not all(math.isfinite(t) for t in times):
        return 0.0
    bound = math.inf
    for _cell, pt in fines:
        d = haversine_km(parent_pt, pt)
        lo = [t - L * d for t, L in zip(times, lipschitz)]
        hi = [t + L * d for t, L in zip(times, lipschitz)]
        bound = min(bound, min_variance_over_box(lo, hi))
        if bound == 0.0:
            break
    return bound


def _split_scored(all_scored, params: Params, n: int, tess):

    coarse_times, evaluated_fine = {}, set()
    v_reachable_min = math.inf
    for a in all_scored:
        res = tess.level(a.cell)
        if res == params.coarse_res:
            coarse_times[a.cell] = a.times
        elif res == params.fine_res:
            evaluated_fine.add(a.cell)
            if metrics.all_reachable(a.times, n):
                v_reachable_min = min(v_reachable_min, metrics.variance(a.times))
    return coarse_times, evaluated_fine, v_reachable_min


def _parent_times(parent, coarse_times, origins, modes_list, evaluator, bucket, tess):

    times = coarse_times.get(parent)
    if times is None:
        pt = tess.center(parent)
        times = [evaluator.effective(o, pt, m, bucket) for o, m in zip(origins, modes_list)]
        coarse_times[parent] = times
    return times


def certificate_report(origins, modes_list, evaluator, params: Params, best, all_scored,
                       lipschitz, bucket: str = "static", tess=None):

    if params.gamma != 0:
        return {"certified": False, "reason": "gamma != 0", "failed": -1, "checked": -1}
    t = resolve_tessellation(origins, params, tess)
    n = len(origins)
    v_star = metrics.variance(best.times)
    coarse_times, evaluated_fine, v_reachable_min = _split_scored(all_scored, params, n, t)
    if v_reachable_min < v_star - 1e-9:
        return {"certified": False, "reason": "t_max excluded a lower-variance candidate",
                "failed": -1, "checked": -1, "v_star": v_star}

    uncovered = _uncovered_by_parent(origins, params, evaluated_fine, t)
    known = len(coarse_times)
    checked, failed = 0, 0
    for parent, fines in uncovered.items():
        checked += 1
        times = _parent_times(parent, coarse_times, origins, modes_list, evaluator, bucket, t)
        if _parent_bound(t.center(parent), fines, times, lipschitz) < v_star - 1e-9:
            failed += 1
    return {"certified": failed == 0, "checked": checked, "failed": failed,
            "extra_evals": len(coarse_times) - known, "v_star": v_star}


def certified_search(origins, modes_list, evaluator, params: Params, lipschitz,
                     bucket: str = "static", tess=None):

    t = resolve_tessellation(origins, params, tess)
    best, _runners, all_scored, diag = fair_meeting_point(
        origins, modes_list, evaluator, params, bucket=bucket, tess=t)
    if best is None:
        return None, {"certified": False, "reason": "no feasible point"}, diag

    n = len(origins)
    incumbent, v_star = best, metrics.variance(best.times)
    coarse_times, evaluated_fine, v_reachable_min = _split_scored(all_scored, params, n, t)
    v_star = min(v_star, v_reachable_min)
    uncovered = _uncovered_by_parent(origins, params, evaluated_fine, t)

    bounds = {}
    for parent, fines in uncovered.items():
        times = _parent_times(parent, coarse_times, origins, modes_list, evaluator, bucket, t)
        bounds[parent] = _parent_bound(t.center(parent), fines, times, lipschitz)

    rounds = 0
    while True:
        alive = [(b, p) for p, b in bounds.items() if b < v_star - 1e-9]
        if not alive:
            break
        rounds += 1
        _b, parent = min(alive)
        del bounds[parent]
        for cell, pt in uncovered.pop(parent):
            times = [evaluator.effective(o, pt, m, bucket) for o, m in zip(origins, modes_list)]
            if not metrics.all_reachable(times, n):
                continue
            v = metrics.variance(times)
            if v < v_star - 1e-12:
                v_star = v
                if metrics.feasible(times, n, params.t_max):
                    incumbent = Area(cell, pt, times, v, True)

    cert = {"certified": True, "rounds": rounds, "v_star": v_star,
            "routing_calls": evaluator.calls}
    return incumbent, cert, diag
