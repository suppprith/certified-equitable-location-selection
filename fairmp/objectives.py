
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from itertools import accumulate

from . import metrics

# An objective is box-computable if the minimum of F over a box of time vectors is
# computable. The certificate only ever needs that quantity, so any objective supplying it
# plugs into the same search with the same guarantee. Five of the six below are exact; the
# coefficient of variation admits a lower bound only.


# Relative slack subtracted from the variance bound. The minimizer is found in closed form,
# but the value is still computed in floating point; without the slack, rounding could put
# the bound a few ulps above the true minimum and it would no longer be a lower bound.
BOUND_SLACK = 1e-9


def _phi(mu, lo, hi, w):
    s = 0.0
    for a, b, wi in zip(lo, hi, w):
        if mu < a:
            s += wi * (a - mu) ** 2
        elif mu > b:
            s += wi * (mu - b) ** 2
    return s


def variance_min(lo, hi, weights=None):
    """Exact, with weights. Weighted variance is min over mu of sum w_i (t_i - mu)^2 / W, so
    exchanging the two minimizations leaves phi(mu) = sum w_i dist(mu, [l_i, h_i])^2 / W.

    phi is convex and piecewise quadratic with breakpoints at the interval ends. Between two
    consecutive breakpoints the sets {i: l_i > mu} and {i: h_i < mu} are fixed, so phi is a
    single quadratic there and its minimizer is the weighted mean of the active ends,
    clamped to the piece. The minimum of phi is the smallest of these per-piece minima, so
    no iterative search is involved. An infinite upper end (an open band) never becomes
    active from above and is handled exactly rather than by giving up."""
    n = len(lo)
    if n == 0:
        return 0.0
    w = list(weights) if weights else [1.0] * n
    tot = sum(w)
    if tot <= 0:
        return 0.0
    if max(lo) <= min(hi):
        return 0.0

    lo_sorted = sorted(zip(lo, w))
    hi_sorted = sorted((b, wi) for b, wi in zip(hi, w) if math.isfinite(b))
    ls = [a for a, _ in lo_sorted]
    hs = [b for b, _ in hi_sorted]
    # suffix sums over lower ends (active when l_i > mu), prefix sums over upper ends
    # (active when h_i < mu)
    lw = list(accumulate(wi for _, wi in reversed(lo_sorted)))[::-1] + [0.0]
    lwx = list(accumulate(wi * a for a, wi in reversed(lo_sorted)))[::-1] + [0.0]
    hw = [0.0] + list(accumulate(wi for _, wi in hi_sorted))
    hwx = [0.0] + list(accumulate(wi * b for b, wi in hi_sorted))

    # the minimizer lies between the smallest lower end and the largest finite end
    left = ls[0]
    right = max([ls[-1]] + hs)
    knots = sorted({x for x in ls + hs if left <= x <= right})

    best = math.inf
    for a, b in zip(knots, knots[1:] + [knots[-1]]):
        ia = bisect_left(ls, b) if b > a else bisect_right(ls, a)
        ib = bisect_right(hs, a)
        den = lw[ia] + hw[ib]
        mu = (lwx[ia] + hwx[ib]) / den if den > 0 else a
        mu = min(max(mu, a), b)
        best = min(best, _phi(mu, lo, hi, w))
    return max(0.0, best / tot * (1.0 - BOUND_SLACK))


def min_sum_min(lo, hi, weights=None):
    """Exact. Monotone increasing in every coordinate, so the minimum is the lower corner."""
    w = weights or [1.0] * len(lo)
    return sum(wi * a for a, wi in zip(lo, w))


def min_max_min(lo, hi, weights=None):
    """Exact. Monotone, lower corner."""
    return max(lo) if lo else math.inf


def range_min(lo, hi, weights=None):
    """Exact, although range is not monotone. For any t in the box, max t >= max l and
    min t <= min h, so range >= max l - min h. The bound is attained: with a = min h and
    b = max l, every [l_i, h_i] meets [a, b], so each t_i can be put inside [a, b]. When the
    intervals share a point the range is zero."""
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
