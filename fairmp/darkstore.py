
from __future__ import annotations

import math
import random
from dataclasses import replace

from . import metrics
from .algorithm import Params, fair_meeting_point
from .candidates import polyfill_centroids, region_polygon
from .geo import LatLng
from .scenarios import CITY_BBOX
from .travel_time import CachedEvaluator

COURIER = ["cycling"]

def sample_demand(city, n_cells, seed=0, clusters=2, sd=0.008, service_radius_deg=0.025):

    rng = random.Random(seed)
    lat0, lng0, lat1, lng1 = CITY_BBOX[city]
    cx = rng.uniform(lat0 + 0.05, lat1 - 0.05)
    cy = rng.uniform(lng0 + 0.05, lng1 - 0.05)
    centers = [(cx + rng.uniform(-service_radius_deg, service_radius_deg),
                cy + rng.uniform(-service_radius_deg, service_radius_deg),
                rng.uniform(0.5, 2.0)) for _ in range(clusters)]
    demand, weights = [], []
    for _ in range(n_cells):
        clat, clng, inten = rng.choice(centers)
        lat = min(max(rng.gauss(clat, sd), lat0), lat1)
        lng = min(max(rng.gauss(clng, sd), lng0), lng1)
        demand.append(LatLng(lat, lng))
        weights.append(max(0.1, rng.gauss(inten, 0.4)))
    return demand, weights

def sample_demand_worldpop(raster_path, bbox, n_cells, seed=0):

    import rasterio
    import numpy as np
    import rasterio

    rng = np.random.default_rng(seed)
    lat0, lng0, lat1, lng1 = bbox
    with rasterio.open(raster_path) as src:
        demand, weights = [], []
        tries = 0
        while len(demand) < n_cells and tries < n_cells * 50:
            tries += 1
            lat = rng.uniform(lat0, lat1)
            lng = rng.uniform(lng0, lng1)
            try:
                val = next(src.sample([(lng, lat)]))[0]
            except StopIteration:
                continue
            if val is None or val <= 0 or math.isnan(val):
                continue

            if rng.random() < min(1.0, float(val) / 50.0):
                demand.append(LatLng(lat, lng))
                weights.append(float(val))
        return demand, weights

def _weighted_centroid(demand, weights):
    w = sum(weights)
    lat = sum(d.lat * x for d, x in zip(demand, weights)) / w
    lng = sum(d.lng * x for d, x in zip(demand, weights)) / w
    return LatLng(lat, lng)

def _coverage_max_point(demand, weights, ev, sla_min, res, courier):

    best, best_cov = None, -1.0
    for _c, pt in polyfill_centroids(region_polygon(demand), res):
        cov = sum(w for o, w in zip(demand, weights)
                  if (t := ev.effective(o, pt, courier)) is not None and math.isfinite(t) and t <= sla_min)
        if cov > best_cov:
            best_cov, best = cov, pt
    return best

def _min_sum_point(demand, ev, res, courier):
    best, best_val = None, math.inf
    for _c, pt in polyfill_centroids(region_polygon(demand), res):
        times = [ev.effective(o, pt, courier) for o in demand]
        if not all(math.isfinite(t) for t in times):
            continue
        s = sum(times)
        if s < best_val:
            best_val, best = s, pt
    return best

def _min_range_point(demand, ev, res, courier):

    best, best_val = None, math.inf
    for _c, pt in polyfill_centroids(region_polygon(demand), res):
        times = [ev.effective(o, pt, courier) for o in demand]
        if not all(math.isfinite(t) for t in times):
            continue
        r = max(times) - min(times)
        if r < best_val:
            best_val, best = r, pt
    return best

def summarize_site(times, weights, sla_min) -> dict:
    return {
        "w_variance": metrics.wvariance(times, weights),
        "w_mean": metrics.wmean(times, weights),
        "w_ede": metrics.wkolm_pollak_ede(times, weights),
        "p90": metrics.percentile(times, 90),
        "p95": metrics.percentile(times, 95),
        "pct_within_sla": metrics.wshare_within(times, weights, sla_min) * 100,
        "courier_gini": metrics.gini(times),
        "max": metrics.max_time(times),
    }

def place_darkstore(demand, weights, backend, params=None, courier=None):
    courier = courier or COURIER
    ev = CachedEvaluator(backend)
    best, runners, scored, diag = fair_meeting_point(
        demand, [courier] * len(demand), ev, params, weights=weights)
    return best, runners, ev, diag

def run_darkstore_instance(demand, weights, backend, params=None, sla_min=10.0, courier=None):

    courier = courier or COURIER
    p = params or Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=30.0)
    rows = []

    best, _r, ev, _d = place_darkstore(demand, weights, backend, p, courier)
    if best is not None:
        rows.append({**summarize_site(best.times, weights, sla_min), "method": "ours", "routing_calls": ev.calls})

    best_e, _r, ev_e, _d = place_darkstore(demand, weights, backend, replace(p, objective="ede"), courier)
    if best_e is not None:
        rows.append({**summarize_site(best_e.times, weights, sla_min), "method": "ours_ede", "routing_calls": ev_e.calls})

    for name, fn in (
        ("coverage_max", lambda ev: _coverage_max_point(demand, weights, ev, sla_min, p.fine_res, courier)),
        ("weighted_centroid", lambda ev: _weighted_centroid(demand, weights)),
        ("min_sum", lambda ev: _min_sum_point(demand, ev, p.fine_res, courier)),
        ("min_range", lambda ev: _min_range_point(demand, ev, p.fine_res, courier)),
    ):
        ev = CachedEvaluator(backend)
        pt = fn(ev)
        if pt is None:
            continue
        times = [ev.effective(o, pt, courier) for o in demand]
        rows.append({**summarize_site(times, weights, sla_min), "method": name, "routing_calls": ev.calls})

    return rows
