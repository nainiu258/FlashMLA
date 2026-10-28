#!/usr/bin/env python3
"""Draw the conceptual KV-load schedules for the three SM90 BF16 variants.

The horizontal axis is a normalized scheduling timeline, not measured time.  It
shows where the implementation permits overlap; it must not be read as a
latency or bandwidth measurement.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


OUT = Path("docs/images/sm90_sparse_prefill_kv_loading_timeline.png")

COLORS = {
    "gmem": "#2a6fbb",       # global -> registers
    "cpasync": "#277da1",    # baseline: global -> local shared memory
    "smem": "#6aaed6",       # registers -> local shared memory
    "dsmem": "#da8a35",      # register -> peer DSMEM
    "qk": "#4c9a66",         # QK WGMMA
    "pv": "#7e62a8",         # PV WGMMA
    "wait": "#b9b9b9",
}


def block(ax, row, start, width, color, text, *, height=0.58):
    """Draw a labelled timeline block."""
    rect = ax.barh(row, width, left=start, height=height, color=color,
                   edgecolor="white", linewidth=1.0, zorder=3)[0]
    if width >= 0.48:
        ax.text(start + width / 2, row, text, ha="center", va="center",
                fontsize=8, color="white" if color != COLORS["wait"] else "#333")
    return rect


def setup_axis(ax, title, labels):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("normalized scheduling time (not a performance scale)", fontsize=9)
    ax.set_xticks(range(13))
    ax.set_xticklabels([])
    ax.grid(axis="x", color="#dedede", linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(16, 13), constrained_layout=False)

    # Baseline: both CTA halves independently fetch the same selected KV.
    ax = axes[0]
    setup_axis(ax, "Baseline BF16 sparse prefill — two independent CTA halves", [
        "CTA 0: heads 0–63", "CTA 1: heads 64–127", "memory traffic"
    ])
    for row in (0, 1):
        block(ax, row, 0.0, 2.8, COLORS["cpasync"], "GMEM→local SMEM (cp.async)")
        block(ax, row, 2.8, 2.0, COLORS["qk"], "QK")
        block(ax, row, 4.8, 4.0, COLORS["pv"], "PV")
    ax.annotate("both CTA halves fetch all selected KV\n→ 2 × 288 MiB = 576 MiB global requests",
                xy=(1.35, 1.05), xytext=(9.2, 1.4), fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#555"}, va="center")
    block(ax, 2, 0.0, 2.8, COLORS["cpasync"], "CTA 0: KV copy")
    block(ax, 2, 2.85, 2.8, COLORS["cpasync"], "CTA 1: KV copy")

    # Fine-grained pipeline.  Each CTA fetches its own 32-token half once and
    # st.async sends that half to the peer CTA's DSMEM (bidirectionally).
    ax = axes[1]
    setup_axis(ax, "Fine-grained pipelined DSMEM crossover — pair n preloaded; load pair n+1 while consuming n", [
        "CTA 0 · WG2 producer: GMEM→reg, rows 0–31 (n+1)",
        "CTA 1 · WG2 producer: GMEM→reg, rows 32–63 (n+1)",
        "CTA 0 · WG2 producer: reg→local SMEM, rows 0–31",
        "CTA 1 · WG2 producer: reg→local SMEM, rows 32–63",
        "DSMEM: CTA 0 → CTA 1, rows 0–31",
        "DSMEM: CTA 1 → CTA 0, rows 32–63",
        "CTA 0 · WG0 consumer: K0, heads 0–63 (n)",
        "CTA 0 · WG1 consumer: K1, heads 0–63 (n)",
        "CTA 1 · WG0 consumer: K0, heads 64–127 (n)",
        "CTA 1 · WG1 consumer: K1, heads 64–127 (n)",
        "CTA 0 · WG0: next QK K0L (n+1)",
        "CTA 1 · WG0: next QK K0L (n+1)"
    ])
    # Existing pair n.  K0L is consumed first and frees first.
    for row in (6, 8):  # WG0 in CTA0 / CTA1: same K0 schedule, different Q heads.
        block(ax, row, 0.0, 1.55, COLORS["qk"], "QK K0L")
        block(ax, row, 1.55, 1.35, COLORS["pv"], "PV K0L")
        block(ax, row, 2.9, 1.65, COLORS["qk"], "QK K0R")
        block(ax, row, 4.55, 2.2, COLORS["pv"], "PV K0R")
    for row in (7, 9):  # WG1 in CTA0 / CTA1: same K1 schedule, different Q heads.
        block(ax, row, 0.0, 1.55, COLORS["qk"], "QK K1R")
        block(ax, row, 1.55, 1.35, COLORS["pv"], "PV K1R")
        block(ax, row, 2.9, 1.65, COLORS["qk"], "QK K1L")
        block(ax, row, 4.55, 2.2, COLORS["pv"], "PV K1L")
    # Local register values are stored locally and asynchronously mirrored to
    # the peer.  Orange bars are the actual cluster DSMEM transactions.
    for start, label in ((2.9, "K0L"), (4.15, "K1R"), (5.4, "K0R"), (6.65, "K1L")):
        block(ax, 0, start, 0.76, COLORS["gmem"], f"load {label}")
        block(ax, 1, start, 0.76, COLORS["gmem"], f"load {label}")
        block(ax, 2, start + 0.76, 0.24, COLORS["smem"], f"K {label}")
        block(ax, 3, start + 0.76, 0.24, COLORS["smem"], f"K {label}")
        block(ax, 4, start + 1.0, 0.49, COLORS["dsmem"], f"st {label}")
        block(ax, 5, start + 1.0, 0.49, COLORS["dsmem"], f"st {label}")
    block(ax, 10, 4.45, 1.45, COLORS["qk"], "QK K0L")
    block(ax, 11, 4.45, 1.45, COLORS["qk"], "QK K0L")
    ax.annotate("Prologue before this window: pair n has completed local load + peer DSMEM exchange.\nTherefore both CTAs may start its independent query-head computation at t=0.",
                xy=(0.2, 6.0), xytext=(0.15, 5.25), fontsize=8.5,
                arrowprops={"arrowstyle": "->", "color": "#555"}, va="center")
    ax.annotate("Each orange transaction is `st.async`: its source register value is written\ninto the peer CTA's DSMEM; both directions carry the complementary 32-token half.",
                xy=(4.1, 4.0), xytext=(8.0, 4.4), fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#555"}, va="center")
    ax.annotate("Both CTAs issue QK(K0L,n+1) after local-ready + incoming DSMEM half. This QK\noverlaps PV(n); PV(n+1) is deliberately not shown as a second concurrent compute stream.",
                xy=(4.45, 10.0), xytext=(8.0, 10.0), fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#555"}, va="center")

    legend = [
        Patch(facecolor=COLORS["cpasync"], label="baseline: GMEM → local SMEM (cp.async)"),
        Patch(facecolor=COLORS["gmem"], label="crossover: GMEM → registers"),
        Patch(facecolor=COLORS["smem"], label="registers → local SMEM"),
        Patch(facecolor=COLORS["dsmem"], label="registers → peer DSMEM"),
        Patch(facecolor=COLORS["qk"], label="QK compute"),
        Patch(facecolor=COLORS["pv"], label="PV compute"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.012), fontsize=9,
               columnspacing=2.0, handletextpad=0.65)
    fig.suptitle("SM90 BF16 sparse prefill: selected KV loading schedules\n"
                 "B=128, top-k=2048, d_qk=576; 288 MiB per BF16 KV copy", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.105, 1.0, 0.945), h_pad=2.3)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
