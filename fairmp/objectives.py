
from __future__ import annotations

import math

from . import metrics

# An objective is box-computable if the minimum of F over a box of time vectors is
# computable. The certificate only ever needs that quantity, so any objective supplying it
# plugs into the same search with the same guarantee. Five of the six below are exact; the
# coefficient of variation admits a lower bound only.


def _finite(lo, hi):
    return all(math.isfinite(x) for x in lo) and all(math.isfinite(x) for x in hi)


def variance_min(lo, hi, weights=None):
    """Exact. Variance is min over mu of the mean squared deviation, so exchanging the two
    minimizations leaves an independent clamp per coordinate and a convex problem in mu."""
    if not _finite(lo, hi):
        return 0.0
    if max(lo) <= min(hi):
        return 0.0
    w = weights or [1.0] * len(lo)
    tot = sum(w)

    def phi(mu):
        s = 0.0
        for a, b, wi in zip(lo, hi, w):
            if mu < a:
                s += wi * (a - mu) ** 2
            elif mu > b:
                s += wi * (mu - b) ** 2
        return s

    left, right = min(lo), max(hi)
    for _ in range(100):
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        if phi(m1) <= phi(m2):
            right = m2
        else:
            left = m1
    return phi((left + right) / 2) / tot


def min_sum_min(lo, hi, weights=None):
    """Exact. Monotone increasing in every coordinate, so the minimum is the lower corner."""
    w = weights or [1.0] * len(lo)
    return sum(wi * a for a, wi in zip(lo, w))


def min_max_min(lo, hi, weights=None):
    """Exact. Monotone, lower corner."""
    return max(lo) if lo else math.inf


def range_min(lo, hi, weights=None):
    """Exact. The spread can only be driven to zero when the intervals share a point;
    otherwise the tightest achievable spread is the gap between the highest floor and the
    lowest ceiling."""
    return max(0.0, max(lo) - min(hi))


def ede_min(lo, hi, weights=None, kappa=None, epsilon=metrics.EDE_EPSILON):
    """Exact for a FIXED kappa. Kolm-Pollak EDE is strictly increasing in each coordinate
    once kappa is held constant, so the box minimum is the lower corner. kappa must not be
    recalibrated from the box, or monotonicity is no longer guaranteed."""
    if kappa is None:
        raise ValueError("ede_min needs a fixed kappa; calibrate it once per instance")
    r = [t for t in lo if math.isfinite(t)]
    if not r:
        return math.inf
    if kappa == 0:
        return sum(r) / len(r)
    m = max(kappa * t for t in r)
    lse = m + math.log(math.fsum(math.exp(kappa * t - m) for t in r))
    return (lse - math.log(len(r))) / kappa


def cv_min(lo, hi, weights=None):
    """Lower bound only, not exact. Standard deviation and mean cannot in general be
    minimized and maximized by the same point of the box, so pairing the smallest possible
    dispersion with the largest possible mean can be loose."""
    sd = math.sqrt(max(0.0, variance_min(lo, hi, weights)))
    w = weights or [1.0] * len(hi)
    tot = sum(w)
    mean_hi = sum(wi * b for b, wi in zip(hi, w)) / tot if tot else math.inf
    return sd / mean_hi if mean_hi > 0 else 0.0


def calibrate_kappa(times, epsilon=metrics.EDE_EPSILON):
    """Fix kappa once, on a reference distribution, so the EDE bound stays monotone."""
    r = [t for t in times if math.isfinite(t)]
    if not r:
        return 0.0
    sum_x = math.fsum(r)
    sum_x2 = math.fsum(t * t for t in r)
    return epsilon * sum_x / sum_x2 if sum_x2 > 0 else 0.0


def variance_of(times, weights=None):
    return metrics.wvariance(times, weights) if weights else metrics.variance(times)


def ede_of(times, weights=None, kappa=None, epsilon=metrics.EDE_EPSILON):
    if kappa is None:
        return (metrics.wkolm_pollak_ede(times, weights, epsilon) if weights
                else metrics.kolm_pollak_ede(times, epsilon))
    r = [t for t in times if math.isfinite(t)]
    if not r:
        return math.inf
    if kappa == 0:
        return sum(r) / len(r)
    m = max(kappa * t for t in r)
    lse = m + math.log(math.fsum(math.exp(kappa * t - m) for t in r))
    return (lse - math.log(len(r))) / kappa


BOX_MIN = {
    "variance": variance_min,
    "min_sum": min_sum_min,
    "min_max": min_max_min,
    "range": range_min,
    "ede": ede_min,
    "cv": cv_min,
}

EVAL = {
    "variance": lambda t, w=None: variance_of(t, w),
    "min_sum": lambda t, w=None: metrics.total_time(t),
    "min_max": lambda t, w=None: metrics.max_time(t),
    "range": lambda t, w=None: metrics.spread(t),
    "ede": lambda t, w=None, kappa=None: ede_of(t, w, kappa),
    "cv": lambda t, w=None: (math.sqrt(variance_of(t, w)) / metrics.mean_time(t)
                             if metrics.mean_time(t) > 0 else 0.0),
}

EXACT = {"variance": True, "min_sum": True, "min_max": True, "range": True,
         "ede": True, "cv": False}
