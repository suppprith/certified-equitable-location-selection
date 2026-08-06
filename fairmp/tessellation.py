
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import h3
from shapely.geometry import Point

from .geo import LatLng

KM_PER_DEG = 111.0
SQRT2 = math.sqrt(2.0)
S_OVER_A = SQRT2 - 1.0

# Bridson sampling at min-distance r packs looser than a lattice of pitch r. This factor
# is calibrated so the Poisson control lands at the same candidate density as the lattices.
POISSON_R_FACTOR = 0.783

# Bridson terminates after k failed attempts, so the sample set is only stochastically
# maximal and the covering radius exceeds r. Unlike the lattices, whose circumradius is
# exact geometry, the Poisson control needs a measured pad. Worst observed ratio 1.07.
POISSON_COVER_PAD = 1.25


def hex_area_km2(level: int) -> float:
    return h3.average_hexagon_area(level, unit="km^2")


class Projection:

    def __init__(self, anchor: LatLng):
        self.lat0 = anchor.lat
        self.lng0 = anchor.lng
        self.kx = KM_PER_DEG * math.cos(math.radians(anchor.lat))
        self.ky = KM_PER_DEG

    def to_xy(self, p: LatLng) -> tuple[float, float]:
        return ((p.lng - self.lng0) * self.kx, (p.lat - self.lat0) * self.ky)

    def to_latlng(self, x: float, y: float) -> LatLng:
        return LatLng(self.lat0 + y / self.ky, self.lng0 + x / self.kx)


def _bbox_xy(proj: Projection, poly):
    lng0, lat0, lng1, lat1 = poly.bounds
    xs, ys = [], []
    for lng in (lng0, lng1):
        for lat in (lat0, lat1):
            x, y = proj.to_xy(LatLng(lat, lng))
            xs.append(x)
            ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


class Tessellation:

    name = "abstract"

    # Whether the pattern admits an exact parent/child hierarchy at *some* aperture.
    # Square and triangular lattices do at integer aperture; H3 and 4.8.8 do at none.
    # Levels here are pinned to the H3 area ladder so candidate budgets are comparable
    # across patterns, which makes the level ratio sqrt(7) and leaves every pattern
    # without exact nesting in practice. children() is therefore centre-containment and
    # refine() adds a neighbour ring to cover the seam. See 16-octagon-indexing.md.
    admits_exact_refinement = False

    def cells_in(self, poly, level: int) -> list[tuple[str, LatLng]]:
        raise NotImplementedError

    def cell_at(self, p: LatLng, level: int) -> str:
        raise NotImplementedError

    def center(self, cell: str) -> LatLng:
        raise NotImplementedError

    def level(self, cell: str) -> int:
        raise NotImplementedError

    def parent(self, cell: str, level: int) -> str:
        return self.cell_at(self.center(cell), level)

    def children(self, cell: str, level: int) -> list[tuple[str, LatLng]]:
        raise NotImplementedError

    def neighbors(self, cell: str) -> list[str]:
        raise NotImplementedError

    def circumradius_km(self, cell: str) -> float:
        raise NotImplementedError

    def refine(self, cell: str, fine_level: int, ring: int = 1) -> list[tuple[str, LatLng]]:
        out = {}
        for c, pt in self.children(cell, fine_level):
            out[c] = pt
            if ring:
                for nb in self._ring(c, ring):
                    out.setdefault(nb, self.center(nb))
        return list(out.items())

    def _ring(self, cell: str, ring: int) -> list[str]:
        seen = {cell}
        frontier = [cell]
        for _ in range(ring):
            nxt = []
            for c in frontier:
                for nb in self.neighbors(c):
                    if nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
            frontier = nxt
        seen.discard(cell)
        return list(seen)


class H3Tessellation(Tessellation):

    name = "h3"
    admits_exact_refinement = False

    def cells_in(self, poly, level):
        outer = [(lat, lng) for lng, lat in poly.exterior.coords]
        holes = [[(lat, lng) for lng, lat in r.coords] for r in poly.interiors]
        out = []
        for c in h3.polygon_to_cells(h3.LatLngPoly(outer, *holes), level):
            lat, lng = h3.cell_to_latlng(c)
            out.append((c, LatLng(lat, lng)))
        return out

    def cell_at(self, p, level):
        return h3.latlng_to_cell(p.lat, p.lng, level)

    def center(self, cell):
        lat, lng = h3.cell_to_latlng(cell)
        return LatLng(lat, lng)

    def level(self, cell):
        return h3.get_resolution(cell)

    def parent(self, cell, level):
        return h3.cell_to_parent(cell, level)

    def children(self, cell, level):
        fine = max(level, h3.get_resolution(cell))
        out = []
        for c in h3.cell_to_children(cell, fine):
            lat, lng = h3.cell_to_latlng(c)
            out.append((c, LatLng(lat, lng)))
        return out

    def neighbors(self, cell):
        return [c for c in h3.grid_disk(cell, 1) if c != cell]

    def circumradius_km(self, cell):
        return h3.average_hexagon_edge_length(h3.get_resolution(cell), unit="km")

    def _ring(self, cell, ring):
        return [c for c in h3.grid_disk(cell, ring) if c != cell]


class _Lattice(Tessellation):

    def __init__(self, anchor: LatLng):
        self.proj = Projection(anchor)

    def pitch(self, level: int) -> float:
        raise NotImplementedError

    def _decode(self, cell: str):
        parts = cell.split(":")
        return parts[0], int(parts[1]), int(parts[2]), int(parts[3])

    def level(self, cell):
        return self._decode(cell)[1]

    def center(self, cell):
        x, y = self._center_xy(cell)
        return self.proj.to_latlng(x, y)

    def _child_reach_km(self, cell) -> float:
        return self.circumradius_km(cell)

    def children(self, cell, level):
        lv = self.level(cell)
        fine = max(level, lv)
        if fine == lv:
            return [(cell, self.center(cell))]
        cx, cy = self._center_xy(cell)
        r = self._child_reach_km(cell)
        step = self.pitch(fine)
        out = []
        reach = int(math.ceil(r / step)) + 2
        for c, (x, y) in self._lattice_block(cx, cy, fine, reach):
            if math.hypot(x - cx, y - cy) <= r and self.parent(c, lv) == cell:
                out.append((c, self.proj.to_latlng(x, y)))
        return out


class SquareTessellation(_Lattice):

    name = "square"
    admits_exact_refinement = True
    n_neighbors = 4

    def __init__(self, anchor, diagonal: bool = False):
        super().__init__(anchor)
        self.diagonal = diagonal
        if diagonal:
            self.name = "octile"
            self.n_neighbors = 8

    def pitch(self, level):
        return math.sqrt(hex_area_km2(level))

    def _center_xy(self, cell):
        _k, lv, i, j = self._decode(cell)
        a = self.pitch(lv)
        return ((i + 0.5) * a, (j + 0.5) * a)

    def cell_at(self, p, level):
        x, y = self.proj.to_xy(p)
        a = self.pitch(level)
        return "s:%d:%d:%d" % (level, math.floor(x / a), math.floor(y / a))

    def cells_in(self, poly, level):
        a = self.pitch(level)
        x0, y0, x1, y1 = _bbox_xy(self.proj, poly)
        out = []
        for i in range(math.floor(x0 / a) - 1, math.floor(x1 / a) + 2):
            for j in range(math.floor(y0 / a) - 1, math.floor(y1 / a) + 2):
                pt = self.proj.to_latlng((i + 0.5) * a, (j + 0.5) * a)
                if poly.contains(Point(pt.lng, pt.lat)):
                    out.append(("s:%d:%d:%d" % (level, i, j), pt))
        return out

    def _lattice_block(self, cx, cy, level, reach):
        a = self.pitch(level)
        i0, j0 = math.floor(cx / a), math.floor(cy / a)
        out = []
        for i in range(i0 - reach, i0 + reach + 1):
            for j in range(j0 - reach, j0 + reach + 1):
                out.append(("s:%d:%d:%d" % (level, i, j), ((i + 0.5) * a, (j + 0.5) * a)))
        return out

    def neighbors(self, cell):
        _k, lv, i, j = self._decode(cell)
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if self.diagonal:
            steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        return ["s:%d:%d:%d" % (lv, i + di, j + dj) for di, dj in steps]

    def circumradius_km(self, cell):
        return self.pitch(self.level(cell)) * SQRT2 / 2.0


class TruncatedSquareTessellation(_Lattice):

    name = "trunc_square"
    admits_exact_refinement = False

    def __init__(self, anchor):
        super().__init__(anchor)

    def pitch(self, level):
        return math.sqrt(2.0 * hex_area_km2(level))

    def _center_xy(self, cell):
        k, lv, i, j = self._decode(cell)
        a = self.pitch(lv)
        if k == "o":
            return (i * a, j * a)
        return ((i + 0.5) * a, (j + 0.5) * a)

    def cell_at(self, p, level):
        x, y = self.proj.to_xy(p)
        a = self.pitch(level)
        i, j = round(x / a), round(y / a)
        if abs(x - i * a) + abs(y - j * a) <= a / SQRT2:
            return "o:%d:%d:%d" % (level, i, j)
        return "q:%d:%d:%d" % (level, round(x / a - 0.5), round(y / a - 0.5))

    def cells_in(self, poly, level):
        a = self.pitch(level)
        x0, y0, x1, y1 = _bbox_xy(self.proj, poly)
        out = []
        for i in range(math.floor(x0 / a) - 1, math.floor(x1 / a) + 2):
            for j in range(math.floor(y0 / a) - 1, math.floor(y1 / a) + 2):
                for k, (cx, cy) in (("o", (i * a, j * a)), ("q", ((i + 0.5) * a, (j + 0.5) * a))):
                    pt = self.proj.to_latlng(cx, cy)
                    if poly.contains(Point(pt.lng, pt.lat)):
                        out.append(("%s:%d:%d:%d" % (k, level, i, j), pt))
        return out

    def _lattice_block(self, cx, cy, level, reach):
        a = self.pitch(level)
        i0, j0 = round(cx / a), round(cy / a)
        out = []
        for i in range(i0 - reach, i0 + reach + 1):
            for j in range(j0 - reach, j0 + reach + 1):
                out.append(("o:%d:%d:%d" % (level, i, j), (i * a, j * a)))
                out.append(("q:%d:%d:%d" % (level, i, j), ((i + 0.5) * a, (j + 0.5) * a)))
        return out

    def neighbors(self, cell):
        k, lv, i, j = self._decode(cell)
        if k == "q":
            return ["o:%d:%d:%d" % (lv, i + di, j + dj)
                    for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1))]
        out = ["o:%d:%d:%d" % (lv, i + di, j + dj)
               for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        out += ["q:%d:%d:%d" % (lv, i + di, j + dj)
                for di, dj in ((0, 0), (-1, 0), (0, -1), (-1, -1))]
        return out

    def circumradius_km(self, cell):
        a = self.pitch(self.level(cell))
        if self._decode(cell)[0] == "o":
            return a * math.hypot(0.5, S_OVER_A / 2.0)
        return a * S_OVER_A / SQRT2


class TriangularTessellation(_Lattice):

    name = "triangle"
    admits_exact_refinement = True
    n_neighbors = 3

    def pitch(self, level):
        return math.sqrt(4.0 * hex_area_km2(level) / math.sqrt(3.0))

    def _height(self, level):
        return self.pitch(level) * math.sqrt(3.0) / 2.0

    def _center_xy(self, cell):
        k, lv, i, j = self._decode(cell)
        t, h = self.pitch(lv), self._height(lv)
        if k == "u":
            return ((i + 0.5) * t, j * h + h / 3.0)
        return ((i + 1.0) * t, j * h + 2.0 * h / 3.0)

    def cell_at(self, p, level):
        x, y = self.proj.to_xy(p)
        t, h = self.pitch(level), self._height(level)
        j = math.floor(y / h)
        v = (y - j * h) / h
        m = math.floor(x / t)
        u = x / t - m
        if u < v / 2.0:
            return "d:%d:%d:%d" % (level, m - 1, j)
        if u > 1.0 - v / 2.0:
            return "d:%d:%d:%d" % (level, m, j)
        return "u:%d:%d:%d" % (level, m, j)

    def cells_in(self, poly, level):
        t, h = self.pitch(level), self._height(level)
        x0, y0, x1, y1 = _bbox_xy(self.proj, poly)
        out = []
        for j in range(math.floor(y0 / h) - 1, math.floor(y1 / h) + 2):
            for m in range(math.floor(x0 / t) - 1, math.floor(x1 / t) + 2):
                for k in ("u", "d"):
                    cid = "%s:%d:%d:%d" % (k, level, m, j)
                    x, y = self._center_xy(cid)
                    pt = self.proj.to_latlng(x, y)
                    if poly.contains(Point(pt.lng, pt.lat)):
                        out.append((cid, pt))
        return out

    def _lattice_block(self, cx, cy, level, reach):
        t, h = self.pitch(level), self._height(level)
        j0, m0 = math.floor(cy / h), math.floor(cx / t)
        out = []
        for j in range(j0 - reach, j0 + reach + 1):
            for m in range(m0 - reach, m0 + reach + 1):
                for k in ("u", "d"):
                    cid = "%s:%d:%d:%d" % (k, level, m, j)
                    out.append((cid, self._center_xy(cid)))
        return out

    def neighbors(self, cell):
        k, lv, m, j = self._decode(cell)
        if k == "u":
            return ["d:%d:%d:%d" % (lv, m - 1, j), "d:%d:%d:%d" % (lv, m, j),
                    "d:%d:%d:%d" % (lv, m, j - 1)]
        return ["u:%d:%d:%d" % (lv, m, j), "u:%d:%d:%d" % (lv, m + 1, j),
                "u:%d:%d:%d" % (lv, m, j + 1)]

    def circumradius_km(self, cell):
        return self.pitch(self.level(cell)) / math.sqrt(3.0)


class PoissonDiskTessellation(_Lattice):

    name = "poisson"
    admits_exact_refinement = False

    def __init__(self, anchor, seed: int = 0):
        super().__init__(anchor)
        self.seed = seed
        self._pts: dict[tuple, list] = {}
        self._index: dict[tuple, dict] = {}
        self._domain: tuple | None = None

    def pitch(self, level):
        return math.sqrt(hex_area_km2(level))

    def _sample(self, level, x0, y0, x1, y1):
        key = (level, round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))
        if key in self._pts:
            return key
        r = self.pitch(level) * POISSON_R_FACTOR
        rng = random.Random((self.seed, key).__hash__() & 0xFFFFFFFF)
        cell = r / SQRT2
        nx = max(1, int((x1 - x0) / cell) + 1)
        ny = max(1, int((y1 - y0) / cell) + 1)
        grid: dict[tuple[int, int], int] = {}
        pts: list[tuple[float, float]] = []

        def fits(x, y):
            gi, gj = int((x - x0) / cell), int((y - y0) / cell)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    q = grid.get((gi + di, gj + dj))
                    if q is not None and math.hypot(pts[q][0] - x, pts[q][1] - y) < r:
                        return False
            return True

        def add(x, y):
            pts.append((x, y))
            grid[(int((x - x0) / cell), int((y - y0) / cell))] = len(pts) - 1

        add(rng.uniform(x0, x1), rng.uniform(y0, y1))
        active = [0]
        while active:
            idx = active[rng.randrange(len(active))]
            px, py = pts[idx]
            placed = False
            for _ in range(24):
                ang = rng.uniform(0, 2 * math.pi)
                rad = rng.uniform(r, 2 * r)
                x, y = px + rad * math.cos(ang), py + rad * math.sin(ang)
                if x0 <= x <= x1 and y0 <= y <= y1 and fits(x, y):
                    add(x, y)
                    active.append(len(pts) - 1)
                    placed = True
                    break
            if not placed:
                active.remove(idx)
        self._pts[key] = pts
        self._index[key] = {"cell": cell, "x0": x0, "y0": y0, "grid": {}}
        gmap: dict[tuple[int, int], list[int]] = {}
        for n, (x, y) in enumerate(pts):
            gmap.setdefault((int((x - x0) / cell), int((y - y0) / cell)), []).append(n)
        self._index[key]["grid"] = gmap
        return key

    def _nearest(self, key, x, y):
        info = self._index[key]
        pts = self._pts[key]
        cell, x0, y0, gmap = info["cell"], info["x0"], info["y0"], info["grid"]
        gi, gj = int((x - x0) / cell), int((y - y0) / cell)
        best, bd = None, math.inf
        rad = 1
        while best is None or rad <= 3:
            for di in range(-rad, rad + 1):
                for dj in range(-rad, rad + 1):
                    for n in gmap.get((gi + di, gj + dj), ()):
                        d = math.hypot(pts[n][0] - x, pts[n][1] - y)
                        if d < bd:
                            best, bd = n, d
            rad += 1
            if rad > 8:
                break
        if best is None:
            best = min(range(len(pts)), key=lambda n: math.hypot(pts[n][0] - x, pts[n][1] - y))
        return best

    def _key_for(self, level):
        for k in self._pts:
            if k[0] == level:
                return k
        if self._domain is None:
            raise KeyError("call cells_in before any other query on the poisson tessellation")
        return self._sample(level, *self._domain)

    def cells_in(self, poly, level):
        x0, y0, x1, y1 = _bbox_xy(self.proj, poly)
        if self._domain is None:
            self._domain = (x0, y0, x1, y1)
        key = self._sample(level, x0, y0, x1, y1)
        out = []
        for n, (x, y) in enumerate(self._pts[key]):
            pt = self.proj.to_latlng(x, y)
            if poly.contains(Point(pt.lng, pt.lat)):
                out.append(("p:%d:%d:0" % (level, n), pt))
        return out

    def _center_xy(self, cell):
        _k, lv, n, _z = self._decode(cell)
        return self._pts[self._key_for(lv)][n]

    def cell_at(self, p, level):
        key = self._key_for(level)
        x, y = self.proj.to_xy(p)
        return "p:%d:%d:0" % (level, self._nearest(key, x, y))

    def _lattice_block(self, cx, cy, level, reach):
        key = self._key_for(level)
        rad = self.pitch(level) * (reach + 1)
        out = []
        for n, (x, y) in enumerate(self._pts[key]):
            if abs(x - cx) <= rad and abs(y - cy) <= rad:
                out.append(("p:%d:%d:0" % (level, n), (x, y)))
        return out

    def neighbors(self, cell):
        _k, lv, n, _z = self._decode(cell)
        key = self._key_for(lv)
        pts = self._pts[key]
        x, y = pts[n]
        r = self.pitch(lv) * 2.2
        near = [(math.hypot(px - x, py - y), m) for m, (px, py) in enumerate(pts)
                if m != n and abs(px - x) <= r and abs(py - y) <= r]
        near.sort()
        return ["p:%d:%d:0" % (lv, m) for _d, m in near[:6]]

    def circumradius_km(self, cell):
        return self.pitch(self.level(cell)) * POISSON_R_FACTOR * POISSON_COVER_PAD

    def _child_reach_km(self, cell):
        return self.pitch(self.level(cell)) * 1.6


def make_tessellation(name: str, anchor: LatLng, seed: int = 0) -> Tessellation:
    if name == "h3":
        return H3Tessellation()
    if name == "square":
        return SquareTessellation(anchor)
    if name == "octile":
        return SquareTessellation(anchor, diagonal=True)
    if name == "triangle":
        return TriangularTessellation(anchor)
    if name == "trunc_square":
        return TruncatedSquareTessellation(anchor)
    if name == "poisson":
        return PoissonDiskTessellation(anchor, seed=seed)
    raise ValueError("unknown tessellation %r" % name)


ALL_TESSELLATIONS = ["h3", "square", "octile", "triangle", "trunc_square", "poisson"]
