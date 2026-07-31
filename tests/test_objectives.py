
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


def test_variance_matches_the_legacy_implementation():
    from fairmp.certificate import min_variance_over_box
    for _ in range(400):
        n = RNG.randint(2, 6)
        lo, hi = _box(n)
        a = ob.variance_min(lo, hi)
        b = min_variance_over_box(lo, hi)
        assert abs(a - b) < 1e-6 * max(1.0, abs(b)), (a, b)


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
