# Certified Equitable Location Selection

Reference implementation and reproducible experiments for the paper
*Certified Search for Equitable Location Selection on Multimodal Networks*.

Given `N` weighted origins whose occupants travel by different modes (walking, cycling,
driving, public transit), the search selects the single location that minimises an equity
functional of the resulting travel times. The travel-time field is treated as an oracle:
it can only be sampled, and every sample is billed. The search runs coarse to fine over a
tessellation of the region with memoised travel times, and returns with the chosen point a
**certificate** that no unevaluated candidate can beat it.

Three properties follow from how that certificate is built.

**Objective-agnostic.** The certificate needs one quantity, the minimum of the objective
over a box of travel-time vectors. Any objective supplying it plugs into the same search
with the same guarantee. `objectives.py` gives exact box minima for variance, min-sum,
min-max, range and the Kolm-Pollak EDE, and a lower bound for the coefficient of variation.

**Tessellation-agnostic.** The argument refers to a cell only through its circumradius, so
the discretisation is a parameter rather than a commitment. `tessellation.py` implements
six patterns (`h3`, `square`, `octile`, `triangle`, `trunc_square`, `poisson`), which at
matched candidate density are indistinguishable in optimality gap.

**Two bounds, one of which does not hold.** Bounding a cell by a per-mode Lipschitz
constant is the obvious choice and is exactly tight on a Euclidean backend. On real
multimodal networks it is not valid, and recalibrating the constant from data does not
repair it; `run_lipschitz_audit.py` measures this. `bands.py` bounds by level sets
instead, which assumes no smoothness, reaches the same optimum, and certifies every
instance where the Lipschitz bound certifies 96.8%.

The query saving from level sets is a property of the cost model, not of the algorithm.
Where a one-to-all sweep is cheap, as on a self-hosted router, the sweep already returns
exact times and the level-set search is a few percent slower in wall-clock time. The
saving is real under per-element billing, where band membership costs one call and a
matrix bills per element. Both sides are measured here.

A group meeting in a city is the smallest instance and serves as the running example;
weighting the origins by population turns the same problem into facility siting.

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

Select a location for a synthetic group on the data-free backend:

```python
from fairmp.scenarios import sample_origins, assign_modes
from fairmp.algorithm import Params, fair_meeting_point
from fairmp.travel_time import EuclideanBackend, CachedEvaluator

origins = sample_origins("london", 5, seed=0)   # five origins in a local area
modes = assign_modes(5, "mixed", seed=0)         # a mode set each
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
# the certificate results
python scripts/run_lipschitz_audit.py    # does the Lipschitz premise hold? (it does not)
python scripts/run_certificate.py        # certified search, both bounds, against exhaustive
python scripts/run_certificate_soundness.py
python scripts/run_real_bands.py         # level-set bounds on real surfaces
python scripts/run_tessellation.py       # six patterns at matched candidate density

# the fairness comparisons
python scripts/run_london.py        # London, multimodal, + gamma/Pareto
python scripts/run_bengaluru.py     # Bengaluru, road-only
python scripts/run_bayarea.py       # San Francisco, multimodal, BART rail
python scripts/run_tokyo.py         # Tokyo, road-only
python scripts/run_rideshare.py     # walk-access pickup
python scripts/run_darkstore.py     # demand-weighted siting
python scripts/run_adversarial.py   # river-crossing / mode-mismatch / linear stress tests
```

`scripts/fig_lipschitz.py` draws the violation-rate figure from `lipschitz_audit.csv`.
`scripts/inspect_jumps.py` and `scripts/diagnose_jumps.py` produce the discontinuity
evidence: whether a short-range jump is a real multimodal barrier or a snapping artefact.

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
| `lipschitz_audit.csv` | Per-mode Lipschitz violation rates by candidate separation, over 72 real travel-time surfaces, plus the recalibrated constant and whether it fares better. |
| `certificate.csv`, `certificate_soundness*.csv` | Certified search against an exhaustive optimum: certified share, suboptimal share, point queries and sweeps, for the Lipschitz and level-set bounds. |
| `real_bands.csv` | Level-set band structure on real surfaces. |
| `tessellation.csv`, `tessellation_summary.csv` | Six candidate patterns at matched density: optimality gap split into discretisation cost and search failure. |
| `jump_inspection.csv`, `jump_diagnosis.csv` | Short-range travel-time jumps, and whether each is mode-specific (a real barrier) or common to all modes (a snapping artefact). |
| `london.csv` | Location selection, London multimodal network: every method's variance, Jain, Gini, EDE, mean, max, optimality gap. |
| `london_pareto.csv` | Gamma sweep: the operating point matching min-sum's mean travel time and the variance reduction there. |
| `bengaluru.csv`, `bayarea.csv`, `tokyo.csv` | Same comparison on the Bengaluru (road-only), San Francisco (multimodal), and Tokyo (road-only) networks, each with a `_pareto` variant. |
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
  candidates.py   region bound, polyfill, geometric prefilter, refinement
  tessellation.py h3, square, octile, triangle, truncated square, Poisson-disk
  objectives.py   box-computable objective family and its exact box minima
  certificate.py  Lipschitz bounds, certified coarse-to-fine search
  bands.py        level-set bounds: thresholds, band membership, certified search
  algorithm.py    coarse-to-fine location selection (variance or EDE objective)
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
