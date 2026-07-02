
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

OUT = "outputs"

DISPLAY = {
    "ours": "Ours (variance)", "exhaustive": "Exhaustive",
    "ours_ede": "Ours (EDE)", "exhaustive_ede": "Exhaustive (EDE)",
    "min_range": "Min-range", "min_max": "Min-max", "min_sum": "Min-sum",
    "random": "Random", "weighted_centroid": "Weighted centroid",
    "geometric_median": "Geometric median", "centroid": "Geometric centroid",
    "coverage_max": "Coverage-max",
}

def _grouped(path, metrics):
    df = pd.read_csv(path)
    return df.groupby("method")[metrics].mean(numeric_only=True)

def _table(rows, colspec, header, caption, label):
    out = [r"\begin{table}[t]", r"\centering",
           rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
           header + r" \\", r"\midrule"]
    out += [r + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)

def fairness_table():
    df = pd.read_csv(f"{OUT}/london.csv")
    solvable = set(df.loc[df["method"] == "ours", "seed"])
    df = df[df["seed"].isin(solvable)]
    g = df.groupby("method")[["variance", "jain", "gini", "ede", "max"]].mean(numeric_only=True)
    g = g.drop(index=[m for m in ["exhaustive_ede"] if m in g.index])
    g = g.sort_values("variance")
    rows = [f"{DISPLAY.get(m, m)} & {r.variance:.1f} & {r.jain:.2f} & {r.gini:.2f} & "
            f"{r.ede:.1f} & {r['max']:.1f}" for m, r in g.iterrows()]
    return _table(
        rows, "lrrrrr",
        "Method & Variance & Jain & Gini & EDE & Max (min)",
        f"London social meetup, mean over the {len(solvable)} solvable instances. On "
        "instances where a method's point is unreachable for some user, its variance "
        "is computed over the reachable users only, which favours that method. Lower "
        "variance, Gini, EDE, and max are fairer; higher Jain is fairer. The variance "
        "objective minimises variance and the EDE objective minimises EDE; both match "
        "their exhaustive references.",
        "tab:fairness")

def adversarial_table():
    df = pd.read_csv(f"{OUT}/adversarial.csv")
    kinds = [("river", "River-crossing"), ("linear", "Linear arterial"), ("mismatch", "Mode mismatch")]
    rows = []
    for key, name in kinds:
        sub = df[df["kind"] == key].groupby("method")["variance"].mean(numeric_only=True)
        ours, cen = sub.get("ours", float("nan")), sub.get("centroid", float("nan"))
        ratio = cen / ours if ours else float("nan")
        rows.append(f"{name} & {ours:.1f} & {cen:.1f} & {ratio:.1f}$\\times$")
    return _table(
        rows, "lrrr",
        "Topology & Ours (= exhaustive) & Geometric centroid & Centroid / ours",
        "Travel-time variance on adversarial London topologies, mean over instances. "
        "Ours equals the exhaustive optimum; the last column is the centroid's variance "
        "as a multiple of ours.",
        "tab:adversarial")

def darkstore_table():
    df = pd.read_csv(f"{OUT}/darkstore.csv")
    n_seeds = df["seed"].nunique()
    g = df.groupby("method")[["w_variance", "courier_gini", "pct_within_sla"]].mean(numeric_only=True)
    order = [m for m in ["ours", "min_range", "ours_ede", "min_sum", "weighted_centroid", "coverage_max"] if m in g.index]
    g = g.reindex(order)
    rows = [f"{DISPLAY.get(m, m)} & {r.w_variance:.1f} & {r.courier_gini:.2f} & {r.pct_within_sla:.0f}"
            for m, r in g.iterrows()]
    return _table(
        rows, "lrrr",
        "Method & W-variance & Courier Gini & Within-SLA (\\%)",
        f"Dark-store siting on the real London network (real cycling times, synthetic "
        f"demand, {n_seeds} instances, SLA 10 min). Lower w-variance and Gini are fairer.",
        "tab:darkstore")

def rideshare_table():
    df = pd.read_csv(f"{OUT}/rideshare_sf.csv")
    n_seeds = df["seed"].nunique()
    g = df.groupby("method")[["variance", "spread", "max", "jain"]].mean(numeric_only=True)
    order = [m for m in ["ours", "min_range", "random", "min_max", "min_sum"] if m in g.index]
    g = g.reindex(order)
    rows = [f"{DISPLAY.get(m, m)} & {r.variance:.1f} & {r.spread:.1f} & {r['max']:.1f} & {r.jain:.2f}"
            for m, r in g.iterrows()]
    return _table(
        rows, "lrrrr",
        "Method & Variance & Spread & Max (min) & Jain",
        f"Ride-share pickup on the real San Francisco walk network: rider walk (access) "
        f"time, mean over {n_seeds} instances. The geometric centroid, geometric median, "
        "and weighted centroid are infeasible on some instances (off-network) and are omitted.",
        "tab:rideshare")

def main():
    builders = [
        ("london.csv", fairness_table),
        ("adversarial.csv", adversarial_table),
        ("darkstore.csv", darkstore_table),
        ("rideshare_sf.csv", rideshare_table),
    ]
    blocks = []
    for fname, fn in builders:
        if os.path.exists(os.path.join(OUT, fname)):
            blocks.append(fn())
        else:
            print(f"skip {fn.__name__}: {fname} missing", file=sys.stderr)
    tex = ("% Auto-generated by scripts/make_paper_tables.py from outputs/*.csv\n"
           "% Requires \\usepackage{booktabs}\n\n" + "\n\n".join(blocks) + "\n")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "paper_tables.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    print(tex)
    print(f"wrote {OUT}/paper_tables.tex", file=sys.stderr)

if __name__ == "__main__":
    main()
