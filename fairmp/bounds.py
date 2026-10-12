"""Travel-time boxes other than level sets, used as baselines for the certified search.

Each function returns, per candidate, a box (lo, hi) per user that is meant to contain the
true travel times. Which of them is actually sound on a given router is an empirical
question, so every box here is also checked against the exact times in the evaluation.
"""
from __future__ import annotations

import math

import numpy as np

from .geo import LatLng, haversine_km

# Upper limits on the speed of each mode in the routers we use. Walking and cycling are
# r5py's fixed routing speeds (3.6 and 12 km/h), so network travel can never beat the
# straight line at that speed. Driving and transit vary by link, so the limit is set above
# any legal road speed or scheduled service speed in the three study cities.
VMAX_KMH = {"walking": 3.6, "cycling": 12.0, "driving": 150.0, "transit": 150.0}

# The router reports whole minutes, so a reported time can sit up to one minute below the
# true one. Every box built from reported times is widened by this much per reported term.
ROUNDING_MIN = 1.0


def geometric_lower(origin: LatLng, pt: LatLng, mode: str, vmax=None) -> float:
    """Zero-query lower bound: straight-line distance at the mode's top speed."""
    v = (vmax or VMAX_KMH)[mode]
    return max(0.0, 60.0 * haversine_km(origin, pt) / v - ROUNDING_MIN)


def geometric_boxes(origins, modes_list, pts, vmax=None):
    """One-sided boxes [straight-line time at top speed, inf). They need no query, but
    nothing caps a travel time from above without one, so the upper end is open. For a
    monotone objective that is enough; for variance or range a box open above can always
    be collapsed to zero spread, so these boxes prune nothing."""
    out = []
    for pt in pts:
        lo = [geometric_lower(o, pt, m[0], vmax) for o, m in zip(origins, modes_list)]
        out.append((lo, [math.inf] * len(origins)))
    return out


def alt_boxes(origins, modes_list, pts, from_landmark, landmark_to_origin, origin_to_landmark,
              vmax=None):
    """Landmark (ALT) boxes from the triangle inequality on shortest-path times.

    from_landmark[m][k][j]   time from landmark k to candidate j under mode m
    landmark_to_origin[m][k][i]  time from landmark k to origin i under mode m
    origin_to_landmark[i][k]     time from origin i to landmark k under user i's mode

    Lower: d(L,p) <= d(L,u) + d(u,p), so d(u,p) >= d(L,p) - d(L,u).
    Upper: d(u,p) <= d(u,L) + d(L,p).
    Both need a fixed, metric travel time, which holds for street modes. A timetabled
    journey does not satisfy the triangle inequality at a fixed departure time, because
    reaching L first changes when the onward leg departs, so transit users get only the
    straight-line lower bound and an open upper end. Each reported term carries up to one
    minute of rounding, hence the widening."""
    n = len(origins)
    out = []
    for j, pt in enumerate(pts):
        lo, hi = [], []
        for i in range(n):
            mode = modes_list[i][0]
            g = geometric_lower(origins[i], pt, mode, vmax)
            if mode not in from_landmark:
                lo.append(g)
                hi.append(math.inf)
                continue
            best_lo, best_hi = g, math.inf
            for k in range(len(from_landmark[mode])):
                d_lp = from_landmark[mode][k][j]
                d_lu = landmark_to_origin[mode][k][i]
                d_ul = origin_to_landmark[i][k]
                if math.isfinite(d_lp) and math.isfinite(d_lu):
                    best_lo = max(best_lo, d_lp - d_lu - 2 * ROUNDING_MIN)
                if math.isfinite(d_lp) and math.isfinite(d_ul):
                    best_hi = min(best_hi, d_ul + d_lp + 2 * ROUNDING_MIN)
            lo.append(max(0.0, best_lo))
            hi.append(max(best_hi, best_lo))
        out.append((lo, hi))
    return out


def pick_landmarks(pts, k: int = 4):
    """Candidates at the extremes of the region along the two diagonals, so that most
    origin-candidate pairs have a landmark roughly behind one end. Chosen from geometry
    alone, since choosing by reachability would need the origin sweeps ALT is meant to
    avoid."""
    lat = np.array([p.lat for p in pts])
    lng = np.array([p.lng for p in pts])
    x = (lng - lng.mean()) * math.cos(math.radians(float(lat.mean())))
    y = lat - lat.mean()
    picks = []
    for score in (x + y, x - y, -x + y, -x - y)[:k]:
        j = int(np.argmax(score))
        if j not in picks:
            picks.append(j)
    return picks


class Grid:
    """A regular grid in a local metric projection, the raster an isochrone service
    computes reachability on before it turns the result into polygons."""

    def __init__(self, poly, spacing_km: float):
        minx, miny, maxx, maxy = poly.bounds
        self.lat0 = (miny + maxy) / 2
        self.lng0 = (minx + maxx) / 2
        self.kx = 111.32 * math.cos(math.radians(self.lat0))
        self.ky = 110.57
        x0, x1 = (minx - self.lng0) * self.kx, (maxx - self.lng0) * self.kx
        y0, y1 = (miny - self.lat0) * self.ky, (maxy - self.lat0) * self.ky
        self.s = spacing_km
        self.xs = np.arange(x0 - spacing_km, x1 + 2 * spacing_km, spacing_km)
        self.ys = np.arange(y0 - spacing_km, y1 + 2 * spacing_km, spacing_km)
        self.shape = (len(self.ys), len(self.xs))

    def nodes(self) -> list[LatLng]:
        return [LatLng(self.lat0 + y / self.ky, self.lng0 + x / self.kx)
                for y in self.ys for x in self.xs]

    def frac_index(self, pt: LatLng):
        x = (pt.lng - self.lng0) * self.kx
        y = (pt.lat - self.lat0) * self.ky
        return (y - self.ys[0]) / self.s, (x - self.xs[0]) / self.s


def polygon_band_index(grid: Grid, node_times, pts, thresholds, method: str):
    """Band index each candidate would get from isochrone polygons built on the grid.

    node_times is the travel time at every grid node (row-major, inf where unreachable).
    Returns, per candidate, the smallest k >= 1 with the candidate inside polygon k, or None
    when it is outside the outermost one.

    method 'nearest': a candidate takes the reachability of the grid cell it falls in, as
      with unsmoothed cell-based output. Error comes only from sub-cell variation.
    method 'contour': the polygon boundary is the contour of the bilinearly interpolated
      grid, as marching-squares isolines are, and only the largest connected piece of each
      isochrone is kept, as a provider's default denoising does. Interpolation assumes the
      field varies smoothly between nodes; denoising drops genuinely reachable islands.
    """
    from scipy import ndimage

    cap = thresholds[-1]
    T = np.asarray(node_times, dtype=float).reshape(grid.shape)
    T_fill = np.where(np.isfinite(T), T, 2.0 * cap)
    ny, nx = grid.shape

    keep = None
    if method == "contour":
        keep = []
        for t in thresholds[1:]:
            mask = T <= t
            lab, count = ndimage.label(mask)
            if count == 0:
                keep.append(np.zeros_like(mask))
                continue
            sizes = ndimage.sum(mask, lab, index=range(1, count + 1))
            keep.append(lab == (1 + int(np.argmax(sizes))))

    out = []
    for pt in pts:
        fy, fx = grid.frac_index(pt)
        if method == "nearest":
            r, c = int(round(fy)), int(round(fx))
            if not (0 <= r < ny and 0 <= c < nx):
                out.append(None)
                continue
            v = T_fill[r, c]
            k = next((k for k, t in enumerate(thresholds[1:], start=1) if v <= t), None)
            out.append(k)
            continue

        r0, c0 = int(math.floor(fy)), int(math.floor(fx))
        if not (0 <= r0 < ny - 1 and 0 <= c0 < nx - 1):
            out.append(None)
            continue
        dy, dx = fy - r0, fx - c0
        corners = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
        wts = [(1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx]
        v = sum(w * T_fill[rc] for w, rc in zip(wts, corners))
        k_hit = None
        for k, t in enumerate(thresholds[1:], start=1):
            if v <= t and any(keep[k - 1][rc] for rc in corners):
                k_hit = k
                break
        out.append(k_hit)
    return out


def band_index_box(k, thresholds, widen: int = 0):
    """Box for a candidate placed in band k, i.e. inside polygon k and outside k-1. With
    widen > 0 the box is stretched by that many bands on each side, a hedge against a
    polygon that is off by one contour."""
    lo_k = max(0, k - 1 - widen)
    hi_k = min(len(thresholds) - 1, k + widen)
    hi = thresholds[hi_k] if k + widen <= len(thresholds) - 1 else math.inf
    return thresholds[lo_k], hi
