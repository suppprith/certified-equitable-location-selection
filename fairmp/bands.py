
from __future__ import annotations

import math
from collections import defaultdict

from . import metrics
from .algorithm import Area, Params, resolve_tessellation
from .candidates import region_polygon
from .certificate import min_variance_over_box
from .geo import haversine_km


def make_thresholds(t_lo: float, t_hi: float, k: int) -> list[float]:
    if k < 1:
        raise ValueError("need at least one band")
    step = (t_hi - t_lo) / k
    return [t_lo + i * step for i in range(k + 1)]


def band_interval(t: float, thresholds: list[float]) -> tuple[float, float]:
    """The level-set band containing t. Exact by construction: no smoothness assumed."""
    if not math.isfinite(t):
        return (thresholds[-1], math.inf)
    if t <= thresholds[0]:
        return (0.0, thresholds[0])
    lo, hi = 0, len(thresholds) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if thresholds[mid] <= t:
            lo = mid
        else:
            hi = mid
    if t > thresholds[-1]:
        return (thresholds[-1], math.inf)
    return (thresholds[lo], thresholds[hi])


def band_box(times: list[float], thresholds: list[float]):
    lo, hi = [], []
    for t in times:
        a, b = band_interval(t, thresholds)
        lo.append(a)
        hi.append(b)
    return lo, hi


def lipschitz_box(times: list[float], dist_km: float, lipschitz: list[float]):
    lo = [t - L * dist_km for t, L in zip(times, lipschitz)]
    hi = [t + L * dist_km for t, L in zip(times, lipschitz)]
    return lo, hi


def intersect_box(a, b):
    """Tighter than either input, but soundness does not survive the intersection: if one
    box is unsound the result is too. Measured on real London surfaces, the Lipschitz box
    is unsound even with an empirically calibrated constant, so the hybrid mode below is a
    tightness experiment only and must not be used to certify on a real network."""
    lo = [max(x, y) for x, y in zip(a[0], b[0])]
    hi = [min(x, y) for x, y in zip(a[1], b[1])]
    return lo, hi


class FieldOracle:
    """One-to-all sweep model. A single sweep per origin yields that origin's whole
    travel-time field, which is what an isochrone or an r5 travel-time surface gives.
    Counted separately from per-point queries so the two cost models stay distinguishable."""

    def __init__(self, backend):
        self.backend = backend
        self.sweeps = 0

    def field(self, origin, modes, points):
        self.sweeps += 1
        out = []
        for p in points:
            best = math.inf
            for m in modes:
                t = self.backend.minutes(origin, p, m)
                if t < best:
                    best = t
            out.append(best)
        return out

    def fields(self, origins, modes_list, points):
        return [self.field(o, m, points) for o, m in zip(origins, modes_list)]


def iso_int(origins, modes_list, backend, thresholds, points, oracle=None):
    """ISO-INT: smallest band threshold whose isochrone intersection over all origins is
    non-empty, then the best candidate inside that intersection. This is the strongest
    decision extractable from the isochrone primitive alone, and it is analytically a
    quantized min-max, so it should track min_max and not the variance optimum."""
    oracle = oracle or FieldOracle(backend)
    fields = oracle.fields(origins, modes_list, points)
    n = len(origins)
    worst = [max(fields[i][j] for i in range(n)) for j in range(len(points))]

    for t in thresholds:
        inside = [j for j, w in enumerate(worst) if w <= t]
        if not inside:
            continue
        times_at = {j: [fields[i][j] for i in range(n)] for j in inside}

        # What the isochrone primitive actually supports: the polygon intersection is a
        # region, so the only selections available are geometric. Per-point travel times
        # inside the region are NOT returned by an isochrone API.
        mx = sum(points[j].lat for j in inside) / len(inside)
        my = sum(points[j].lng for j in inside) / len(inside)
        centroid_j = min(inside, key=lambda j: (points[j].lat - mx) ** 2 + (points[j].lng - my) ** 2)

        # Oracle-assisted upper bound: hand ISO-INT our per-point times for free and let
        # it rank the region by variance. Not purchasable from any isochrone product;
        # reported only to show the region itself is the binding limitation.
        oracle_j = min(inside, key=lambda j: metrics.variance(times_at[j]))
        return {
            "t_star": t,
            "region_size": len(inside),
            "centroid": points[centroid_j],
            "centroid_times": times_at[centroid_j],
            "oracle": points[oracle_j],
            "oracle_times": times_at[oracle_j],
            "sweeps": oracle.sweeps,
        }
    return None


def band_certified_search(origins, modes_list, evaluator, params: Params, thresholds,
                          band_source, objective="variance", tess=None, bucket: str = "static",
                          kappa=None):
    """Certified search whose bound comes from level sets rather than from a smoothness
    assumption.

    The cost model this targets is the paid-API one. A one-to-all sweep per origin is cheap
    and returns band membership, not per-point times: an isochrone call yields a polygon.
    So the band gives a sound lower bound on the objective for *every* candidate at once,
    for N sweeps, and exact per-point times still have to be bought one query at a time.

    The search therefore evaluates candidates in increasing order of their band lower bound
    and stops as soon as the next bound cannot beat the incumbent. Every candidate left
    unevaluated has a bound at or above the incumbent, so the incumbent is optimal over the
    candidate set. Unlike the Lipschitz box this needs no modulus of continuity, so it stays
    valid across the discontinuities that a real multimodal field actually contains.
    """
    from . import objectives as ob

    t = resolve_tessellation(origins, params, tess)
    n = len(origins)
    poly = region_polygon(origins)
    cells = t.cells_in(poly, params.fine_res)
    if not cells:
        return None, {"certified": False, "reason": "no candidates"}

    pts = [pt for _c, pt in cells]
    oracle = FieldOracle(band_source)
    fields = oracle.fields(origins, modes_list, pts)

    box_min = ob.BOX_MIN[objective]
    kw = {"kappa": kappa} if objective == "ede" else {}

    bounds = []
    unreachable = 0
    for j in range(len(pts)):
        times = [fields[i][j] for i in range(n)]
        # A candidate one user cannot reach scores infinity, so it is not a competitor and
        # must not be queried. Without this it lands in the open band [t_last, inf), whose
        # unbounded ceiling drives the box minimum to zero and makes the least promising
        # candidates look like the most promising ones. On real networks, where a large
        # share of the grid is unreachable on foot, that alone forces a near-exhaustive
        # search.
        if not metrics.all_reachable(times, n):
            unreachable += 1
            continue
        lo, hi = band_box(times, thresholds)
        bounds.append((box_min(lo, hi, **kw), j))
    bounds.sort()

    evaluate = ob.EVAL[objective]
    incumbent, best_val, evaluated = None, math.inf, 0
    for bound, j in bounds:
        if bound >= best_val - 1e-12:
            break
        cell, pt = cells[j]
        times = [evaluator.effective(o, pt, m, bucket) for o, m in zip(origins, modes_list)]
        evaluated += 1
        if not metrics.all_reachable(times, n):
            continue
        val = evaluate(times, None, kappa) if objective == "ede" else evaluate(times)
        if val < best_val:
            best_val, incumbent = val, Area(cell, pt, times, val,
                                            metrics.feasible(times, n, params.t_max))

    reachable = len(bounds)
    return incumbent, {
        "certified": incumbent is not None,
        "objective": objective,
        "value": best_val,
        "candidates": len(pts),
        "reachable": reachable,
        "unreachable": unreachable,
        "evaluated": evaluated,
        # against the reachable set, since unreachable candidates are excluded by the
        # sweep itself and never cost a point query
        "eval_fraction": evaluated / reachable if reachable else float("nan"),
        "eval_fraction_all": evaluated / len(pts) if pts else float("nan"),
        "sweeps": oracle.sweeps,
        "point_queries": evaluator.calls,
        "n_thresholds": len(thresholds) - 1,
    }


def band_certificate_report(origins, modes_list, evaluator, params: Params, best, all_scored,
                            thresholds, backend, lipschitz=None, mode: str = "band",
                            tess=None, bucket: str = "static"):
    """A posteriori certificate using level-set bands instead of (or intersected with) the
    Lipschitz box. Bands are sound by construction and need no smoothness assumption."""
    if params.gamma != 0:
        return {"certified": False, "reason": "gamma != 0"}
    t = resolve_tessellation(origins, params, tess)
    n = len(origins)
    v_star = metrics.variance(best.times)

    evaluated_fine = set()
    v_reachable_min = math.inf
    for a in all_scored:
        if t.level(a.cell) == params.fine_res:
            evaluated_fine.add(a.cell)
            if metrics.all_reachable(a.times, n):
                v_reachable_min = min(v_reachable_min, metrics.variance(a.times))
    if v_reachable_min < v_star - 1e-9:
        return {"certified": False, "reason": "t_max excluded a lower-variance candidate"}

    poly = region_polygon(origins)
    uncovered = defaultdict(list)
    for cell, pt in t.cells_in(poly, params.fine_res):
        if cell not in evaluated_fine:
            uncovered[t.parent(cell, params.coarse_res)].append((cell, pt))

    all_pts, index = [], {}
    for parent, fines in uncovered.items():
        for cell, pt in fines:
            index[cell] = len(all_pts)
            all_pts.append(pt)
    if not all_pts:
        return {"certified": True, "checked": 0, "failed": 0, "sweeps": 0, "v_star": v_star}

    oracle = FieldOracle(backend)
    fields = oracle.fields(origins, modes_list, all_pts)

    checked, failed = 0, 0
    for parent, fines in uncovered.items():
        checked += 1
        parent_pt = t.center(parent)
        parent_times = None
        bound = math.inf
        for cell, pt in fines:
            j = index[cell]
            times = [fields[i][j] for i in range(n)]
            box = band_box(times, thresholds)
            if mode in ("hybrid", "lipschitz") and lipschitz is not None:
                if parent_times is None:
                    parent_times = [evaluator.effective(o, parent_pt, m, bucket)
                                    for o, m in zip(origins, modes_list)]
                lb = lipschitz_box(parent_times, haversine_km(parent_pt, pt), lipschitz)
                box = lb if mode == "lipschitz" else intersect_box(box, lb)
            bound = min(bound, min_variance_over_box(box[0], box[1]))
            if bound == 0.0:
                break
        if bound < v_star - 1e-9:
            failed += 1

    return {"certified": failed == 0, "checked": checked, "failed": failed,
            "sweeps": oracle.sweeps, "v_star": v_star, "mode": mode,
            "n_thresholds": len(thresholds) - 1}
