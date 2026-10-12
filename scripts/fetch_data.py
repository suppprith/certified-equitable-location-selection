
from __future__ import annotations

import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

AUTO = [
    ("london/network.osm.pbf", "https://download.bbbike.org/osm/bbbike/London/London.osm.pbf"),
    ("london/gtfs/london_bus.zip", "https://data.bus-data.dft.gov.uk/timetable/download/gtfs-file/london/"),
    ("bengaluru/gtfs/bmtc.zip", "https://raw.githubusercontent.com/Vonter/bmtc-gtfs/main/gtfs/bmtc.zip"),

    ("tokyo/network.osm.pbf", "https://download.bbbike.org/osm/bbbike/Tokyo/Tokyo.osm.pbf"),

    ("bayarea/network.osm.pbf", "https://download.bbbike.org/osm/bbbike/SanFrancisco/SanFrancisco.osm.pbf"),
    ("bayarea/gtfs/bart.zip", "https://www.bart.gov/dev/schedules/google_transit.zip"),
    # SFMTA retired gtfs.sfmta.com; this is the producer URL listed on the Mobility
    # Database (feed mdb-2886) as of October 2026
    ("bayarea/gtfs/muni.zip", "https://muni-gtfs.apps.sfmta.com/data/muni_gtfs-current.zip"),
]

MANUAL = """
Manual steps (large + crop, account, or browser-only):
  - Bengaluru OSM: download karnataka.osm.pbf from osm.fr (extracts/asia/india), then
    run scripts/crop_bengaluru.py -> data/bengaluru/blr_city.osm.pbf
  - London tube/rail GTFS: already bundled in the London region feed above; no separate
    download needed.
  - Bengaluru metro GTFS (synthetic): built by scripts/build_metro_gtfs.py.
  - Tokyo transit GTFS: Open Data Platform for Transportation (ODPT,
    https://www.odpt.org) needs a free API token; Tokyo Metro / Toei / JR-East GTFS-JP
    feeds -> data/tokyo/gtfs/ . Mobility Database (https://mobilitydatabase.org) mirrors
    some Tokyo feeds without a token; verify currency before use.
  - Bay Area regional GTFS: 511.org all-agency feed (https://511.org/open-data/transit)
    needs a free API token; the BART + Muni feeds above cover rail + SF core without one.
  - WorldPop rasters (siting scenario): worldpop.org -> data/<city>/worldpop.tif
  - London Fire incidents (emergency, real): London Datastore -> data/london/incidents.csv
"""

def fetch(rel: str, url: str):
    dest = os.path.join(DATA, rel)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("exists  ", rel)
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("download", rel)
    # some feed hosts refuse urllib's default user agent with a 403
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research data fetch)"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print("        ", round(os.path.getsize(dest) / 1e6, 1), "MB")

def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    items = [(rel, url) for rel, url in AUTO if only is None or rel.startswith(only + "/")]
    failed = []
    for rel, url in items:
        # mirrors time out now and then; a missing network must stop the run here rather
        # than surface later as a missing-file error inside the router
        for attempt in range(1, 5):
            try:
                fetch(rel, url)
                break
            except Exception as e:
                print("FAILED  ", rel, "attempt", attempt, "->", e)
                partial = os.path.join(DATA, rel)
                if os.path.exists(partial):
                    os.remove(partial)
                time.sleep(20 * attempt)
        else:
            failed.append(rel)
    if only is None:
        print(MANUAL)
    if failed:
        sys.exit("could not download: " + ", ".join(failed))

if __name__ == "__main__":
    main()
