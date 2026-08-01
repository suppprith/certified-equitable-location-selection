
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BANDS = [("b0_0.45_viol", 0.225), ("b0.45_0.8_viol", 0.625),
         ("b0.8_1.2_viol", 1.0), ("b1.2_2_viol", 1.6)]
MODES = ["transit", "cycling", "driving", "walking"]
MARK = {"transit": "o", "cycling": "s", "driving": "^", "walking": "D"}


def main():
    df = pd.read_csv(os.path.join(ROOT, "outputs", "lipschitz_audit.csv"))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    xs = [d for _c, d in BANDS]
    for mode in MODES:
        sub = df[df["mode"] == mode]
        if not len(sub):
            continue
        ys = [sub[c].mean() for c, _d in BANDS]
        ax.plot(xs, ys, marker=MARK[mode], ms=4, lw=1.2, label=mode)

    # The reporting granularity floor: whole-minute travel times inflate the observed
    # slope by up to 1/d, so anything above this line cannot be a rounding artefact.
    L = df.groupby("mode").L_euclidean.mean().min()
    dd = np.linspace(0.2, 2.0, 100)
    ax.axvspan(0.0, 0.45, color="0.9", zorder=0)
    ax.text(0.23, 6, "rounding\nfloor high", ha="center", va="bottom", fontsize=6, color="0.35")

    ax.set_xlabel("candidate separation (km)")
    ax.set_ylabel("violation rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 1.85)
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper right")
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    fig.tight_layout(pad=0.2)

    out = os.path.join(ROOT, "outputs", "figures", "fig_lipschitz_violation.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    for mode in MODES:
        sub = df[df["mode"] == mode]
        if len(sub):
            print("  %-8s far-band violation %.1f%%" % (mode, sub["b1.2_2_viol"].mean()))


if __name__ == "__main__":
    main()
