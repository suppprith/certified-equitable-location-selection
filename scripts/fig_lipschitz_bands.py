"""Violation rate of the speed-derived Lipschitz constant against candidate separation,
pooled over every surface of the real-network evaluation in all three cities.

Solid lines count a pair only if its gap exceeds L*d after one minute of rounding is
removed; dashed lines are the raw rates. The gap between them is what whole-minute
reporting can explain, and it narrows with distance as 1/d.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "figures", "fig_lipschitz_bands.pdf")

BANDS = [("0", "0.45"), ("0.45", "0.8"), ("0.8", "1.2"), ("1.2", "2")]
MID = [0.225, 0.625, 1.0, 1.6]
# categorical slots 1-4 of the validated default palette, in fixed order; aqua and yellow
# sit below 3:1 on white, so every line also carries a marker shape and a direct label
MODES = [("transit", "#2a78d6", "o"), ("cycling", "#eb6834", "s"),
         ("driving", "#1baf7a", "^"), ("walking", "#eda100", "D")]
INK, MUTED = "#1a1a19", "#6b6a63"


def pooled(df, col, ncol):
    w = df[ncol].fillna(0)
    ok = df[col].notna() & (w > 0)
    return float((df.loc[ok, col] * w[ok]).sum() / w[ok].sum())


def main():
    lip = pd.concat([pd.read_csv(os.path.join(ROOT, "outputs", "eval_lipschitz_%s.csv" % c))
                     for c in ("london", "bayarea", "tokyo")], ignore_index=True)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "text.color": INK,
    })
    fig, ax = plt.subplots(figsize=(3.45, 1.5))
    ends = []
    for mode, color, marker in MODES:
        d = lip[lip["mode"] == mode]
        key = ["b%s_%s" % b for b in BANDS]
        raw = [pooled(d, k + "_viol", k + "_n") for k in key]
        rob = [pooled(d, k + "_viol_robust", k + "_n") for k in key]
        ax.plot(MID, raw, color=color, lw=0.8, ls=(0, (3, 2)), alpha=0.9)
        ax.plot(MID, rob, color=color, lw=1.2, marker=marker, ms=3.2,
                markeredgecolor="white", markeredgewidth=0.5, label=mode.capitalize())
        ends.append([rob[-1], mode.capitalize()])

    # end labels, nudged apart where two lines finish close together
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 11)
    for y, name in ends:
        ax.annotate(name, (MID[-1], y), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=6.5, color=INK)

    ax.set_xticks(MID)
    ax.set_xticklabels(["%s–%s" % b for b in BANDS])
    ax.set_xlim(0.1, 2.0)
    ax.set_xlabel("Separation between candidates (km)")
    ax.set_ylabel("Pairs violating $L$ (%)")
    ax.set_ylim(0, 85)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.grid(axis="y", color="#e4e3dc", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    style = [plt.Line2D([], [], color=MUTED, lw=1.2), plt.Line2D([], [], color=MUTED, lw=0.8,
                                                                  ls=(0, (3, 2)))]
    ax.legend(handles + style, labels + ["rounding removed", "raw"], ncol=3, frameon=False,
              fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.8,
              columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
