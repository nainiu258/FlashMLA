#!/usr/bin/env python3
"""Draw the code-derived SM90 sparse-prefill baseline KV schedule.

The figure follows csrc/sm90/prefill/sparse/phase1.cuh.  It deliberately uses
program order and barrier dependencies, not measured execution time.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch


OUT = Path("docs/images/sm90_sparse_prefill_baseline_double_buffer_pipeline.png")
COLORS = {
    "load": "#277da1",
    "qk": "#4c9a66",
    "softmax": "#bf5b54",
    "pv": "#7e62a8",
    "barrier": "#d98c34",
    "wait": "#d7d7d7",
    "data": "#6aaed6",
}


def box(ax, x, y, w, h, color, text, *, fontsize=8, text_color="white"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color, edgecolor="white", linewidth=1.1, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, zorder=4, wrap=True)
    return (x + w / 2, y + h / 2)


def arrow(ax, start, end, text="", *, color="#555", text_offset=(0, 0), fontsize=7.4):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.15,
                    "shrinkA": 4, "shrinkB": 4}, zorder=2,
    )
    if text:
        ax.text((start[0] + end[0]) / 2 + text_offset[0],
                (start[1] + end[1]) / 2 + text_offset[1], text,
                fontsize=fontsize, color=color, ha="center", va="center",
                zorder=5)


def setup(ax):
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")


def data_flow(ax):
    """Panel 1: which warpgroup computes which data, directly from the code."""
    setup(ax)
    ax.set_title("1. One CTA's data flow for one pair of 64-token blocks", loc="left",
                 fontsize=13, fontweight="bold", pad=9)
    ax.text(0.05, 9.45,
            "One CTA owns B_H = 64 query heads.  K0 = selected tokens [block_idx], "
            "K1 = selected tokens [block_idx + 1].",
            fontsize=9.2, va="top")

    q = box(ax, 0.35, 5.75, 2.15, 1.15, COLORS["data"], "Q\n64 query heads", fontsize=9)
    k0 = box(ax, 3.15, 7.15, 2.2, 1.0, COLORS["data"], "KV slot 0\nK0 / V0: 64 tokens")
    k1 = box(ax, 3.15, 3.45, 2.2, 1.0, COLORS["data"], "KV slot 1\nK1 / V1: 64 tokens")
    qk0 = box(ax, 6.0, 7.05, 2.55, 1.2, COLORS["qk"],
              "WG0: Q × K0ᵀ\nall D_K tiles → logits P0")
    qk1 = box(ax, 6.0, 3.35, 2.55, 1.2, COLORS["qk"],
              "WG1: Q × K1ᵀ\nall D_K tiles → logits P1")
    sm0 = box(ax, 9.3, 6.7, 2.1, 1.05, COLORS["softmax"],
              "WG0 online\nsoftmax for P0")
    sm1 = box(ax, 9.3, 3.8, 2.1, 1.05, COLORS["softmax"],
              "WG1 updates\nsoftmax with P1")
    out_l = box(ax, 12.5, 6.55, 2.75, 1.25, COLORS["pv"],
                "WG0 output cols [0,256)\nS0·V0L + S1·V1L")
    out_r = box(ax, 12.5, 3.65, 2.75, 1.25, COLORS["pv"],
                "WG1 output cols [256,512)\nS1·V1R + S0·V0R")

    arrow(ax, (q[0] + 1.05, q[1] + 0.30), (qk0[0] - 1.25, qk0[1]), "same Q heads", text_offset=(0, 0.25))
    arrow(ax, (q[0] + 1.05, q[1] - 0.30), (qk1[0] - 1.25, qk1[1]), "same Q heads", text_offset=(0, -0.25))
    arrow(ax, (k0[0] + 1.1, k0[1]), (qk0[0] - 1.25, qk0[1]), "K0", text_offset=(0, 0.28))
    arrow(ax, (k1[0] + 1.1, k1[1]), (qk1[0] - 1.25, qk1[1]), "K1", text_offset=(0, -0.28))
    arrow(ax, (qk0[0] + 1.28, qk0[1]), (sm0[0] - 1.05, sm0[1]), "P0")
    arrow(ax, (qk1[0] + 1.28, qk1[1]), (sm1[0] - 1.05, sm1[1]), "P1")
    arrow(ax, (sm0[0], sm0[1] - 0.5), (sm1[0], sm1[1] + 0.5),
          "running max / sum", color="#8a3d39", text_offset=(1.35, 0))
    arrow(ax, (sm1[0] + 1.05, sm1[1] + 0.25), (sm0[0] + 1.05, sm0[1] - 0.25),
          "new scale for S0", color="#8a3d39", text_offset=(1.45, 0))
    arrow(ax, (sm0[0] + 1.05, sm0[1] + 0.1), (out_l[0] - 1.35, out_l[1]), "S0 / S1")
    arrow(ax, (sm1[0] + 1.05, sm1[1] - 0.1), (out_r[0] - 1.35, out_r[1]), "S1 / S0")
    arrow(ax, (k0[0] + 1.1, k0[1] + 0.25), (out_l[0] - 1.35, out_l[1] + 0.25), "V0L", color="#7155a0")
    arrow(ax, (k1[0] + 1.1, k1[1] + 0.25), (out_l[0] - 1.35, out_l[1] - 0.25), "V1L", color="#7155a0")
    arrow(ax, (k1[0] + 1.1, k1[1] - 0.25), (out_r[0] - 1.35, out_r[1] + 0.25), "V1R", color="#7155a0")
    arrow(ax, (k0[0] + 1.1, k0[1] - 0.25), (out_r[0] - 1.35, out_r[1] - 0.25), "V0R", color="#7155a0")
    ax.text(0.35, 1.05,
            "Code evidence: WG0 calls QK tiles 0–3 then 4–end on plan.k[0]; "
            "WG1 calls 4–end then 0–3 on plan.k[1].  Thus both QK products use the full K dimension.",
            fontsize=8.6, va="bottom", bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f6f6f6", "edgecolor": "#c4c4c4"})


def dependency_flow(ax):
    """Panel 2: exact producer order and the relevant readiness/free barriers."""
    setup(ax)
    ax.set_title("2. Double-buffer dependency schedule (program order; not a timing trace)", loc="left",
                 fontsize=13, fontweight="bold", pad=9)
    ys = {"prod": 8.0, "wg0": 5.0, "wg1": 2.0}
    for label, y in (("producer WG2", ys["prod"]), ("consumer WG0", ys["wg0"]), ("consumer WG1", ys["wg1"])):
        ax.text(0.15, y + 0.42, label, fontsize=9.2, fontweight="bold", va="center")
        ax.hlines(y, 1.75, 15.7, color="#d9d9d9", linewidth=1.0, zorder=0)

    # Producer: exact sequence at phase1.cuh:525–543.
    p0 = box(ax, 2.0, 7.45, 2.05, 1.1, COLORS["load"], "wait F0L\ncp.async V0L\n→ R0L")
    p1 = box(ax, 4.35, 7.45, 2.05, 1.1, COLORS["load"], "wait F1R\ncp.async V1R\n→ R1R")
    p2 = box(ax, 6.70, 7.45, 2.05, 1.1, COLORS["load"], "wait F0R\ncp.async V0R\n→ R0R")
    p3 = box(ax, 9.05, 7.45, 2.05, 1.1, COLORS["load"], "wait F1L\ncp.async V1L\n→ R1L")
    pn = box(ax, 12.3, 7.45, 2.8, 1.1, COLORS["wait"], "next pair: same order\nafter matching F barriers", text_color="#333")
    arrow(ax, p0, p1); arrow(ax, p1, p2); arrow(ax, p2, p3); arrow(ax, p3, pn)

    # QK and softmax sequence in the consumer warp groups.
    w0q = box(ax, 2.0, 4.45, 2.65, 1.1, COLORS["qk"], "wait R0L → QK0L\nwait R0R → QK0R\n(full K0 → P0)")
    w0s = box(ax, 4.95, 4.45, 1.7, 1.1, COLORS["softmax"], "mask +\nsoftmax P0\n→ B0")
    w0pv0 = box(ax, 6.95, 4.45, 1.6, 1.1, COLORS["pv"], "P0 × V0L\n→ F0L")
    w0x = box(ax, 8.85, 4.45, 1.65, 1.1, COLORS["barrier"], "wait B1\nrescale S0\nwrite S0")
    w0pv1 = box(ax, 10.8, 4.45, 1.6, 1.1, COLORS["pv"], "wait S1\nP1 × V1L")
    w0f = box(ax, 12.7, 4.45, 2.4, 1.1, COLORS["barrier"], "launch next QK0\nthen F1L")
    for a, b in ((w0q, w0s), (w0s, w0pv0), (w0pv0, w0x), (w0x, w0pv1), (w0pv1, w0f)):
        arrow(ax, a, b)

    w1q = box(ax, 2.0, 1.45, 2.65, 1.1, COLORS["qk"], "wait R1R → QK1R\nwait R1L → QK1L\n(full K1 → P1)")
    w1s = box(ax, 4.95, 1.45, 1.7, 1.1, COLORS["softmax"], "wait B0\nsoftmax P1\n→ B1")
    w1pv1 = box(ax, 6.95, 1.45, 1.6, 1.1, COLORS["pv"], "P1 × V1R\nwrite S1")
    w1x = box(ax, 8.85, 1.45, 1.65, 1.1, COLORS["barrier"], "wait S0\nP0 × V0R")
    w1f = box(ax, 10.8, 1.45, 2.4, 1.1, COLORS["barrier"], "GMMA done\n→ F1R, then F0R")
    for a, b in ((w1q, w1s), (w1s, w1pv1), (w1pv1, w1x), (w1x, w1f)):
        arrow(ax, a, b)

    # Only the following arrows represent cross-warpgroup / reuse dependencies.
    arrow(ax, (p0[0], p0[1] - 0.55), (w0q[0] - 1.05, w0q[1] + 0.15), "R0L", color="#1f629c")
    arrow(ax, (p2[0], p2[1] - 0.55), (w0q[0] + 1.05, w0q[1] + 0.15), "R0R", color="#1f629c")
    arrow(ax, (p1[0], p1[1] - 0.55), (w1q[0] - 1.05, w1q[1] + 0.15), "R1R", color="#1f629c")
    arrow(ax, (p3[0], p3[1] - 0.55), (w1q[0] + 1.05, w1q[1] + 0.15), "R1L", color="#1f629c")
    arrow(ax, w0s, w1s, "B0", color="#8a3d39", text_offset=(0.25, 0))
    arrow(ax, w1s, w0x, "B1", color="#8a3d39", text_offset=(0.25, 0))
    arrow(ax, w0x, w1x, "S0", color="#8a5218", text_offset=(0.2, 0))
    arrow(ax, w1pv1, w0pv1, "S1", color="#8a5218", text_offset=(0.2, 0))
    arrow(ax, w0pv0, pn, "F0L → next V0L", color="#8a5218", text_offset=(0.1, 0.25))
    arrow(ax, w1f, pn, "F1R / F0R → next V1R / V0R", color="#8a5218", text_offset=(0.3, -0.2))
    arrow(ax, w0f, pn, "F1L → next V1L", color="#8a5218", text_offset=(0.1, 0.25))
    ax.text(2.0, 9.35,
            "Rxy = bar_kx_ready[y]; Fxy = bar_kx_free[y].  The producer's four loads are ordered V0L → V1R → V0R → V1L.\n"
            "A free signal allows the matching part of the next pair to be loaded; it does not mean a third KV buffer exists.",
            fontsize=8.6, va="top", bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f6f6f6", "edgecolor": "#c4c4c4"})


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(18, 13), constrained_layout=False)
    data_flow(axes[0])
    dependency_flow(axes[1])
    legend = [
        Patch(facecolor=COLORS["load"], label="GMEM → local SMEM: cp.async"),
        Patch(facecolor=COLORS["qk"], label="QK WGMMA"),
        Patch(facecolor=COLORS["softmax"], label="online softmax"),
        Patch(facecolor=COLORS["pv"], label="PV WGMMA"),
        Patch(facecolor=COLORS["barrier"], label="CTA barrier / shared P"),
        Patch(facecolor=COLORS["wait"], label="wait / next-pair placeholder"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.012), fontsize=9)
    fig.suptitle("SM90 BF16 sparse prefill baseline: code-derived KV double-buffer pipeline",
                 fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.985), h_pad=3.0)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
