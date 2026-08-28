# Certified Equitable Location Selection

Reference implementation and reproducible experiments for the paper
*Certified Search for Equitable Location Selection on Multimodal Networks*.

Given `N` users who each travel by a possibly different mode (walking, cycling, driving,
public transit), the algorithm selects a single location that minimises the **variance of
their multimodal travel times**, so no one carries a much longer trip than the rest. The
search discretises the region with the [H3](https://h3geo.org/) hexagonal grid and runs
coarse-to-fine over candidate cells with memoised travel times, reaching the full
fine-grid optimum while issuing far fewer routing queries.

The same search is metric-agnostic: it also optimises the **Kolm-Pollak equally-distributed-
equivalent (EDE)** travel time, and a weight on the mean term traces the variance-vs-mean
Pareto front. The method is applied to three settings: social meeting points, ride-share
pickup, and demand-weighted micro-fulfilment (dark-store) siting.

## Install

```
pip install -r requirements.txt
```

`numpy pandas shapely h3 scipy matplotlib` are enough for the data-free backend.
`geopandas rasterio r5py` plus a **JDK 21** are needed only for the real-network runs.

A `Dockerfile` (Java 21 + Python) is included for a reproducible environment:

```
docker build -t fairmp . && docker run --rm fairmp   # runs the unit tests
```

## Quickstart

Compute a fair meeting point for a synthetic group on the data-free backend:

```python
from fairmp.scenarios import sample_origins, assign_modes
from fairmp.algorithm import Params, fair_meeting_point
from fairmp.travel_time import EuclideanBackend, CachedEvaluator

origins = sample_origins("london", 5, seed=0)   # five users in a local area
modes = assign_modes(5, "mixed", seed=0)         # one mode each
ev = CachedEvaluator(EuclideanBackend())
best, runners_up, _, _ = fair_meeting_point(origins, modes, ev, Params())
print(best.point, [round(t, 1) for t in best.times])
```

Swap `EuclideanBackend()` for `R5Backend(osm, gtfs)` to use real travel times.

## Run without data (Euclidean backend)

Everything runs immediately on a straight-line travel-time backend, which is meant for
testing and development, not for the reported numbers:

```
python tests/test_core.py        # unit tests (metrics, EDE, range baseline, optimality)
python scripts/smoke_test.py     # end-to-end on a synthetic instance
python scripts/run_significance.py   # 30 instances/scenario, 95% CIs + paired Wilcoxon
```

## Reproduce the real-network results

1. Fetch the road network and transit feeds (see `scripts/DATA.md`; `scripts/fetch_data.py`
   automates the open, no-auth downloads). Real runs use OpenStreetMap + GTFS via
   [r5py](https://r5py.readthedocs.io/) and need a JDK 21.
2. Run the real-network experiments (each builds the routing network once):

```
python scripts/run_london.py        # London social meetup (multimodal) + gamma/Pareto
python scripts/run_bengaluru.py     # Bengaluru social meetup (road-only)
python scripts/run_bayarea.py       # San Francisco social meetup (multimodal, BART rail)
python scripts/run_tokyo.py         # Tokyo social meetup (road-only)
python scripts/run_rideshare.py     # ride-share walk-access pickup
python scripts/run_darkstore.py     # demand-weighted dark-store siting
python scripts/run_adversarial.py   # river-crossing / mode-mismatch / linear stress tests
```

The bbbike OSM extracts for San Francisco and Tokyo are re-written and cropped to the
routable city region with `scripts/crop_bayarea.py` and `scripts/crop_tokyo.py` before use.

Set `N_INSTANCES` to change the number of instances (default 100 per city/scenario, the
count reported in the paper; set e.g. `N_INSTANCES=3` for a quick smoke run).

3. Regenerate the paper tables from the result CSVs:

```
python scripts/make_paper_tables.py      # writes outputs/paper_tables.tex
```

## Results (`outputs/`)

| File | What it contains |
| --- | --- |
| `london.csv` | Social meetup, London multimodal network: every method's variance, Jain, Gini, EDE, mean, max, optimality gap. |
| `london_pareto.csv` | Gamma sweep: the operating point matching min-sum's mean travel time and the variance reduction there. |
| `bengaluru.csv`, `bayarea.csv`, `tokyo.csv` | Social meetup on the Bengaluru (road-only), San Francisco (multimodal), and Tokyo (road-only) networks, each with a `_pareto` variant. |
| `rideshare.csv` | Ride-share pickup, walk-access times. |
| `darkstore.csv` | Dark-store siting, cycling times, demand-weighted metrics (w-variance, courier Gini, within-SLA share, w-EDE). |
| `adversarial.csv` | River-crossing, mode-mismatch, and linear topologies. |
| `significance.csv`, `significance_tests.csv` | 100-instance synthetic sweep with paired Wilcoxon tests. |
| `sweep.csv`, `dev_darkstore.csv`, `dev_rideshare.csv` | Euclidean development runs. |
| `paper_tables.tex` | LaTeX tables generated from the CSVs above. |

## Layout

```
fairmp/
  geo.py          haversine, centroid, spread
  metrics.py      variance, Jain, Gini, Kolm-Pollak EDE, weighted variants, feasibility
  travel_time.py  Backend ABC; EuclideanBackend; R5Backend (r5py); cached evaluator
  candidates.py   region bound, H3 polyfill, geometric prefilter, refinement
  algorithm.py    coarse-to-fine fair meeting point (variance or EDE objective)
  baselines.py    centroid, weighted centroid, Weiszfeld, min-sum, min-max, min-range,
                  random, exhaustive (variance and EDE references)
  scenarios.py    synthetic instance generator
  darkstore.py    demand-weighted siting + coverage-max baseline
  runner.py       harness: run every method, collect metrics, optimality gap
  sweep.py        multi-instance sweeps, gamma/Pareto, resolution and size scaling
scripts/          runnable experiments + data fetch + table generation
tests/test_core.py
```

The travel-time backend is abstract. `EuclideanBackend` needs no data; `R5Backend` and
`PrecomputedBackend` supply real OSM + GTFS times via r5py. The algorithm, baselines, and
metrics are identical across backends, so the optimality gap and routing-query counts are
backend-independent and the real backend only changes the travel-time values.

## License

MIT, see `LICENSE`.
