
from __future__ import annotations

import math
from collections import defaultdict

from . import metrics
from .algorithm import Params, resolve_tessellation
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
