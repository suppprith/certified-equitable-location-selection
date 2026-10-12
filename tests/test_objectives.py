
from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fairmp import objectives as ob

RNG = random.Random(11)


def _box(n):
    lo = [RNG.uniform(1.0, 60.0) for _ in range(n)]
    hi = [a + RNG.uniform(0.0, 40.0) for a in lo]
    return lo, hi


def _sample(lo, hi, k=600):
    return [[RNG.uniform(a, b) for a, b in zip(lo, hi)] for _ in range(k)]


def test_box_minima_are_lower_bounds():
    # every objective's claimed box minimum must not exceed any point in the box
    for name in ob.BOX_MIN:
        for _ in range(250):
            n = RNG.randint(2, 6)
            lo, hi = _box(n)
            kw = {}
            if name == "ede":
                kw["kappa"] = ob.calibrate_kappa(lo)
            claimed = ob.BOX_MIN[name](lo, hi, **kw)
            for t in _sample(lo, hi, 120):
                got = (ob.EVAL[name](t, None, kw["kappa"]) if name == "ede"
                       else ob.EVAL[name](t))
                assert got >= claimed - 1e-7, (name, claimed, got)


def test_exact_objectives_are_attained():
    """Random sampling cannot reach a corner in n dimensions, so exactness is checked
    against the family that actually contains the minimizer. For variance and range the
    optimal point clamps every coordinate toward a common value c, so scanning c densely
    reaches the true optimum; the monotone objectives are attained at the lower corner and
    are checked directly in test_monotone_objectives_use_the_lower_corner."""
    for name in ("variance", "range"):
        for _ in range(300):
            n = RNG.randint(2, 6)
            lo, hi = _box(n)
            claimed = ob.BOX_MIN[name](lo, hi)
            best = math.inf
            span_lo, span_hi = min(lo), max(hi)
            for k in range(4001):
                c = span_lo + (span_hi - span_lo) * k / 4000
                t = [min(max(c, a), b) for a, b in zip(lo, hi)]
                best = min(best, ob.EVAL[name](t))
            assert best >= claimed - 1e-6, (name, claimed, best)
            assert best - claimed <= 1e-4 * max(1.0, abs(claimed)) + 1e-6, (name, claimed, best)


def test_monotone_objectives_use_the_lower_corner():
    for _ in range(300):
        n = RNG.randint(2, 6)
        lo, hi = _box(n)
        k = ob.calibrate_kappa(lo)
        assert abs(ob.min_sum_min(lo, hi) - sum(lo)) < 1e-9
        assert abs(ob.min_max_min(lo, hi) - max(lo)) < 1e-9
        assert abs(ob.ede_min(lo, hi, kappa=k) - ob.ede_of(lo, kappa=k)) < 1e-9


def test_range_bound_is_zero_when_intervals_overlap():
    assert ob.range_min([10, 12], [14, 16]) == 0.0
    assert abs(ob.range_min([0, 5], [1, 9]) - 4.0) < 1e-9


def _wvar(t, w):
    tot = sum(w)
    m = sum(wi * x for x, wi in zip(t, w)) / tot
    return sum(wi * (x - m) ** 2 for x, wi in zip(t, w)) / tot


def _phi_scan(lo, hi, w, steps=20000):
    """Reference minimum of phi by a dense scan of mu, refined by ternary search on the
    best bracket. Used only to check the closed form, never by the search itself."""
    tot = sum(w)

    def phi(mu):
        return ob._phi(mu, lo, hi, w) / tot

    left = min(lo)
    right = max([max(lo)] + [b for b in hi if math.isfinite(b)])
    if right <= left:
        return phi(left)
    xs = [left + (right - left) * k / steps for k in range(steps + 1)]
    k = min(range(len(xs)), key=lambda i: phi(xs[i]))
    a, b = xs[max(0, k - 1)], xs[min(steps, k + 1)]
    for _ in range(200):
        m1, m2 = a + (b - a) / 3, b - (b - a) / 3
        if phi(m1) <= phi(m2):
            b = m2
        else:
            a = m1
    return phi((a + b) / 2)


def test_weighted_variance_bound_is_exact_and_below_the_minimum():
    # the closed form must never exceed the true minimum (soundness) and must reach it
    # (exactness); the scan reference is itself an upper estimate of the minimum
    for _ in range(400):
        n = RNG.randint(2, 7)
        lo, hi = _box(n)
        w = [RNG.uniform(0.1, 5.0) for _ in range(n)]
        claimed = ob.variance_min(lo, hi, w)
        ref = _phi_scan(lo, hi, w)
        assert claimed <= ref + 1e-12, (claimed, ref)
        assert ref - claimed <= 1e-7 * max(1.0, ref), (claimed, ref)
        for t in _sample(lo, hi, 60):
            assert _wvar(t, w) >= claimed, (claimed, _wvar(t, w))


def test_variance_bound_attained_by_clamping_to_the_optimal_mean():
    # the minimizing time vector clamps every coordinate to mu*; its variance must equal
    # the bound up to the stated slack
    for _ in range(300):
        n = RNG.randint(2, 6)
        lo, hi = _box(n)
        claimed = ob.variance_min(lo, hi)
        best = math.inf
        for k in range(20001):
            mu = min(lo) + (max(hi) - min(lo)) * k / 20000
            t = [min(max(mu, a), b) for a, b in zip(lo, hi)]
            best = min(best, ob.EVAL["variance"](t))
        assert claimed <= best + 1e-12
        assert best - claimed <= 1e-5 * max(1.0, best), (claimed, best)


def test_open_band_is_bounded_not_zeroed():
    # a user known only to need at least 120 minutes, against users near 20 minutes,
    # cannot reach zero spread; the bound must stay positive and below any realization
    lo, hi = [18.0, 20.0, 120.0], [22.0, 25.0, math.inf]
    claimed = ob.variance_min(lo, hi)
    assert claimed > 0.0
    for t in ([22.0, 25.0, 120.0], [20.0, 21.0, 300.0]):
        assert ob.EVAL["variance"](t) >= claimed


def test_certificate_uses_the_same_bound():
    from fairmp.certificate import min_variance_over_box
    for _ in range(100):
        lo, hi = _box(RNG.randint(2, 6))
        assert min_variance_over_box(lo, hi) == ob.variance_min(lo, hi)


def test_range_bound_is_attained_inside_the_box():
    # with a = min h < b = max l, t_i = max(l_i, a) lies in [l_i, h_i] and in [a, b], and
    # its range is exactly b - a
    seen = 0
    for _ in range(600):
        lo, hi = _box(RNG.randint(2, 6))
        a, b = min(hi), max(lo)
        if a >= b:
            continue
        seen += 1
        t = [max(x, a) for x in lo]
        assert all(l_ <= x <= h_ for x, l_, h_ in zip(t, lo, hi))
        assert abs(ob.EVAL["range"](t) - ob.range_min(lo, hi)) < 1e-9
    assert seen > 50


def test_ede_requires_a_fixed_kappa():
    lo, hi = _box(4)
    try:
        ob.ede_min(lo, hi)
    except ValueError:
        return
    raise AssertionError("ede_min must refuse to run without a fixed kappa")


def test_cv_is_a_lower_bound_but_not_claimed_exact():
    assert ob.EXACT["cv"] is False
    for _ in range(200):
        n = RNG.randint(2, 6)
        lo, hi = _box(n)
        claimed = ob.cv_min(lo, hi)
        for t in _sample(lo, hi, 150):
            assert ob.EVAL["cv"](t) >= claimed - 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, "PASS")
    print("all objective tests passed")
