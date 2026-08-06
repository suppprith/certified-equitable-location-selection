
from __future__ import annotations

import random

from .geo import LatLng

CITY_BBOX = {
    "london": (51.40, -0.30, 51.62, 0.10),
    "bengaluru": (12.85, 77.45, 13.10, 77.75),

    "tokyo": (35.55, 139.60, 35.82, 139.92),

    "bayarea": (37.70, -122.52, 37.88, -122.20),
}

MODES = ["transit", "driving", "walking", "cycling"]

CHOICE_PAIRS = [["driving", "transit"], ["walking", "cycling"],
                ["cycling", "transit"], ["driving", "cycling"]]

def sample_origins(city, n, seed=0, spread="uniform", clusters=2, cluster_sd_deg=0.02):
    rng = random.Random(seed)
    lat0, lng0, lat1, lng1 = CITY_BBOX[city]
    pts = []
    if spread == "uniform":
        for _ in range(n):
            pts.append(LatLng(rng.uniform(lat0, lat1), rng.uniform(lng0, lng1)))
    else:
        centers = [(rng.uniform(lat0, lat1), rng.uniform(lng0, lng1)) for _ in range(clusters)]
        for _ in range(n):
            clat, clng = rng.choice(centers)
            lat = min(max(rng.gauss(clat, cluster_sd_deg), lat0), lat1)
            lng = min(max(rng.gauss(clng, cluster_sd_deg), lng0), lng1)
            pts.append(LatLng(lat, lng))
    return pts

def assign_modes(n, mix="mixed", seed=0, pool=None):
    """pool restricts which modes may be drawn. A city with no schedule feed must not be
    given transit users: the router silently falls back to walking, their reachable set
    collapses, and the instance is discarded for having no commonly reachable candidate.
    That is what reduced a 25-instance Bay Area run to 2 usable instances."""
    rng = random.Random(seed)
    options = pool or MODES
    chosen = [rng.choice(options) for _ in range(n)] if mix == "mixed" else [mix] * n
    return [[m] for m in chosen]


ROAD_MODES = ["driving", "walking", "cycling"]

def assign_modes_with_choice(n, seed=0, frac_choice=0.4):

    rng = random.Random(seed)
    modes = [[rng.choice(MODES)] for _ in range(n)]
    k = max(1, round(frac_choice * n))
    for i in rng.sample(range(n), min(k, n)):
        modes[i] = list(rng.choice(CHOICE_PAIRS))
    return modes
