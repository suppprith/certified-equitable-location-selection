
from __future__ import annotations

import datetime as dt
import io
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# osm is the network file; gtfs may be empty, in which case the city is road-only and
# transit is simply unavailable rather than silently degraded.
CITIES = {
    "london": {
        "osm": "london/network.osm.pbf",
        "gtfs": ["london/gtfs/london_bus.zip"],
        "multimodal": True,
    },
    "bayarea": {
        "osm": "bayarea/sf_city.osm.pbf",   # produced by crop_bayarea.py
        "gtfs": ["bayarea/gtfs/bart.zip", "bayarea/gtfs/muni.zip"],
        "multimodal": True,
    },
    # Bengaluru's OSM extract is a manual step (karnataka.osm.pbf, then crop), so it is
    # not reachable from CI. Kept for local runs only.
    "bengaluru": {
        "osm": "bengaluru/blr_city.osm.pbf",
        "gtfs": [],
        "multimodal": False,
    },
    "tokyo": {
        "osm": "tokyo/network.osm.pbf",
        "gtfs": [],
        "multimodal": False,
    },
}


# Origins must be sampled inside the network the router actually loaded. The module-level
# CITY_BBOX entries are wider than the cropped extracts, so sampling from them places
# origins where there is no road data and the instance is discarded as unsolvable. The Bay
# Area box in particular reaches ~15 km east across the bay while the cropped network stops
# at -122.37, which reduced a 30-instance run to 3. These match the crop scripts.
SAMPLE_BBOX = {
    "bayarea": (37.73, -122.51, 37.81, -122.39),    # crop_bayarea.py: -122.53..-122.37
    "bengaluru": (12.90, 77.52, 13.06, 77.68),      # crop_bengaluru.py
}


def sample_bbox(city: str):
    return SAMPLE_BBOX.get(city)


def apply_sample_bbox(city: str):
    """Narrow scenarios.CITY_BBOX to the loaded network before sampling origins."""
    b = SAMPLE_BBOX.get(city)
    if b:
        from fairmp.scenarios import CITY_BBOX
        CITY_BBOX[city] = b
    return b


def paths(city: str):
    cfg = CITIES[city]
    osm = os.path.join(ROOT, "data", *cfg["osm"].split("/"))
    gtfs = [os.path.join(ROOT, "data", *g.split("/")) for g in cfg["gtfs"]]
    return osm, [g for g in gtfs if os.path.exists(g)]


def _service_window(gtfs_path: str):
    """Earliest start and latest end across calendar.txt, plus the weekday flags."""
    with zipfile.ZipFile(gtfs_path) as z:
        names = z.namelist()
        if "calendar.txt" not in names:
            return None
        rows = z.read("calendar.txt").decode("utf-8-sig").splitlines()
    if len(rows) < 2:
        return None
    head = [h.strip() for h in rows[0].split(",")]
    idx = {h: i for i, h in enumerate(head)}
    need = ("start_date", "end_date", "monday", "tuesday", "wednesday", "thursday", "friday")
    if any(k not in idx for k in need):
        return None
    lo, hi, weekdays = None, None, set()
    for r in rows[1:]:
        f = r.split(",")
        if len(f) < len(head):
            continue
        try:
            s = dt.datetime.strptime(f[idx["start_date"]].strip(), "%Y%m%d").date()
            e = dt.datetime.strptime(f[idx["end_date"]].strip(), "%Y%m%d").date()
        except ValueError:
            continue
        lo = s if lo is None or s < lo else lo
        hi = e if hi is None or e > hi else hi
        for d, name in enumerate(("monday", "tuesday", "wednesday", "thursday", "friday")):
            if f[idx[name]].strip() == "1":
                weekdays.add(d)
    if lo is None:
        return None
    return lo, hi, weekdays


def pick_departure(city: str, hour: int = 8, minute: int = 30) -> str:
    """A weekday inside every loaded feed's validity window.

    A departure outside that window silently yields no transit service and the router
    falls back to walking, which invalidates every transit figure while raising no error.
    Detecting the window beats hard-coding a date that goes stale when a feed is refreshed.
    """
    _osm, gtfs = paths(city)
    windows = [w for w in (_service_window(g) for g in gtfs) if w]
    if not windows:
        # road-only city: any weekday will do
        d = dt.date.today() + dt.timedelta(days=30)
        while d.weekday() > 4:
            d += dt.timedelta(days=1)
        return "%s %02d:%02d:00" % (d.isoformat(), hour, minute)

    lo = max(w[0] for w in windows)
    hi = min(w[1] for w in windows)
    common = set.intersection(*[w[2] for w in windows]) or {2}
    if lo > hi:
        raise ValueError("feeds for %s have no overlapping validity window" % city)

    # a quarter of the way in, so a feed that tapers at its edges is still dense
    d = lo + dt.timedelta(days=max(1, (hi - lo).days // 4))
    for _ in range(14):
        if d.weekday() in common and lo <= d <= hi:
            return "%s %02d:%02d:00" % (d.isoformat(), hour, minute)
        d += dt.timedelta(days=1)
    raise ValueError("no weekday with service found for %s in %s..%s" % (city, lo, hi))


if __name__ == "__main__":
    for c in CITIES:
        osm, gtfs = paths(c)
        ok = os.path.exists(osm)
        try:
            dep = pick_departure(c) if ok else "n/a"
        except Exception as e:
            dep = "ERROR: %s" % e
        print("%-10s osm=%-5s gtfs=%d  departure=%s" % (c, ok, len(gtfs), dep))
