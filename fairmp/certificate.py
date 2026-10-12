
from __future__ import annotations

import math
from collections import defaultdict

from . import metrics
from .algorithm import Area, Params, fair_meeting_point, resolve_tessellation
from .candidates import region_polygon
from .geo import haversine_km
from .objectives import variance_min
from .travel_time import EUCLIDEAN_SPEED_KMH


def euclidean_lipschitz(modes_list, detour: float = 1.3, speeds=None) -> list[float]:
    """Speed-derived constant 60 * detour / speed, in minutes per km. Exactly tight on the
    Euclidean backend with the same speeds and detour."""
    spd = speeds or EUCLIDEAN_SPEED_KMH
    out = []
    for modes in modes_list:
        speed = min(spd.get(m, 4.8) for m in modes)
        out.append(60.0 * detour / speed)
    return out


def min_variance_over_box(lo: list[float], hi: list[float]) -> float:
    """Unweighted exact box minimum; see objectives.variance_min."""
    return variance_min(lo, hi)


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


def screened_lipschitz_search(origins, modes_list, evaluator, params: Params, lipschitz,
                              feasible, tess=None, bucket: str = "static"):
    """The Lipschitz bound given the same reachability screen as the level-set search.

    feasible(pt) says whether every user reaches pt within the cap; it comes from the same
    one-to-all sweeps the level-set search pays for, so neither search queries an
    unreachable candidate. Anchors are the screened coarse centres, each bought with N
    point queries. Every screened fine candidate is boxed from its nearest anchor by the
    Lipschitz constant, and the candidates are then evaluated in order of their box minimum
    exactly as in the level-set search. The result is certified only if the constant holds,
    which is the premise the audit tests."""
    from .bands import box_search, lipschitz_box

    t = resolve_tessellation(origins, params, tess)
    n = len(origins)
    poly = region_polygon(origins)
    anchors = []
    for _c, pt in t.cells_in(poly, params.coarse_res):
        if feasible(pt):
            times = [evaluator.effective(o, pt, m, bucket) for o, m in zip(origins, modes_list)]
            if metrics.all_reachable(times, n):
                anchors.append((pt, times))
    anchor_calls = evaluator.calls

    cells = t.cells_in(poly, params.fine_res)
    pts = [pt for _c, pt in cells]
    boxes = []
    for pt in pts:
        if not feasible(pt):
            boxes.append(None)
            continue
        if not anchors:
            boxes.append(([0.0] * n, [math.inf] * n))
            continue
        a_pt, a_times = min(anchors, key=lambda a: haversine_km(a[0], pt))
        boxes.append(lipschitz_box(a_times, haversine_km(a_pt, pt), lipschitz))

    def evaluate(j):
        return [evaluator.effective(o, pts[j], m, bucket) for o, m in zip(origins, modes_list)]

    best_j, best_val, info = box_search(boxes, evaluate, n, "variance")
    return (None if best_j is None else cells[best_j]), {
        "certified": best_j is not None, "v_star": best_val,
        "routing_calls": evaluator.calls, "anchor_calls": anchor_calls,
        "anchors": len(anchors), "evaluated": info["evaluated"]}


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
