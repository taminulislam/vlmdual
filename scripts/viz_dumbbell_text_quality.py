"""Dumbbell / slope plot — LLaVA-1.5 vs VLMDual on the 6 textual-quality
metrics from paper Table tab:text_quality.

Each metric is one row.  Two markers — LLaVA (left, muted red) and
VLMDual (right, green) — connected by a thick grey bar, with both
endpoint values labelled and the signed Δ printed above the bar.
Three facets are used because the metrics live on three different
x-scales:

    Panel 1 : presence indicators (%)              x ∈ [0, 100]
    Panel 2 : per-narrative counts                 x ∈ [0,   4]
    Panel 3 : narrative length (words)             x ∈ [80, 100]

Output: paper/overleaf/figures/seaborn_viz/dumbbell_text_quality.{png,pdf}
"""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = Path("/work/nvme/bgte/tislam6/ACID_Journal/paper/overleaf/figures/seaborn_viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (metric, llava, vlmdual, lower_is_better)
PANEL_PCT = [
    ("Contains explicit\nprobability",         0.0, 100.0, False),
    ("References predicted\nmask geometry",    8.4,  96.1, False),
    ("References diagnostic\npH threshold",    4.2,  88.7, False),
]
PANEL_COUNT = [
    ("Hedging verbs ↓\n(per narrative)",       3.6,  0.4,  True),
    ("Verifiable claims\n(per narrative)",     0.7,  3.4,  False),
]
PANEL_LEN = [
    ("Narrative length\n(words)",             84.5, 96.2,  False),
]

C_LLAVA   = "#c0392b"
C_VLMDUAL = "#27ae60"
C_LINE    = "#9aa0a6"

def draw_panel(ax, rows, xlim, xlabel, *, title="", delta_unit="",
               n_slots=3, fmt="{:.1f}"):
    """Draw a dumbbell panel.  `n_slots` keeps row-heights consistent
    across panels with different metric counts."""
    ys = list(range(len(rows)))
    pad = 0.05 * (xlim[1] - xlim[0])     # extra horizontal padding to keep
                                          # endpoint value labels off the markers

    for y, (metric, llava, vlm, lib) in zip(ys, rows):
        x0, x1 = sorted([llava, vlm])
        # connecting bar
        ax.hlines(y, x0, x1, color=C_LINE, lw=4.0, alpha=0.55, zorder=1)
        # endpoint markers
        ax.scatter([llava], [y], s=170, color=C_LLAVA,   zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.scatter([vlm],   [y], s=190, color=C_VLMDUAL, zorder=3,
                   edgecolor="white", linewidth=1.5)
        # endpoint value labels — placed on the OUTER side of each marker
        ll_outer = "right" if llava < vlm else "left"
        vlm_outer = "left" if vlm > llava else "right"
        ax.text(llava + (-pad if ll_outer == "right" else pad), y,
                fmt.format(llava), ha=ll_outer, va="center",
                color=C_LLAVA, fontsize=10.5, fontweight="bold")
        ax.text(vlm + (pad if vlm_outer == "left" else -pad), y,
                fmt.format(vlm), ha=vlm_outer, va="center",
                color=C_VLMDUAL, fontsize=10.5, fontweight="bold")
        # delta — ABOVE the bar (smaller y in inverted axes = higher visually)
        delta = vlm - llava
        good  = (delta < 0) if lib else (delta > 0)
        sign  = "+" if delta > 0 else ""
        ax.text((x0 + x1) / 2, y - 0.32,
                f"Δ {sign}{delta:.1f}{delta_unit}",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=("#27ae60" if good else "#c0392b"))

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5, fontweight="bold")
    ax.set_ylim(n_slots - 0.5, -0.7)         # inverted, leave room for Δ
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.tick_params(axis="x", labelsize=10)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    ax.grid(axis="x", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


# ── plot ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(
    1, 3, figsize=(15.5, 4.6),
    gridspec_kw=dict(width_ratios=[3.0, 2.3, 1.7]),
)

draw_panel(
    axes[0], PANEL_PCT, xlim=(-10, 115),
    xlabel="% of held-out narratives",
    delta_unit=" pp",
    title="Presence indicators (% of narratives)",
)
draw_panel(
    axes[1], PANEL_COUNT, xlim=(-0.5, 4.4),
    xlabel="Mean count per narrative",
    title="Per-narrative counts",
)
draw_panel(
    axes[2], PANEL_LEN, xlim=(78, 102),
    xlabel="Words",
    title="Narrative length",
)

# Single shared legend at the top
handles = [
    plt.Line2D([0], [0], marker="o", linestyle="", color=C_LLAVA,
               markersize=13, markeredgecolor="white", markeredgewidth=1.5,
               label="LLaVA-1.5 (vanilla, image-only prompt)"),
    plt.Line2D([0], [0], marker="o", linestyle="", color=C_VLMDUAL,
               markersize=13, markeredgecolor="white", markeredgewidth=1.5,
               label="VLMDual (ours, structured-grounded prompt)"),
]
leg = fig.legend(handles=handles, loc="upper center",
                 bbox_to_anchor=(0.5, 1.04),
                 ncol=2, frameon=False, fontsize=11)
for text in leg.get_texts():
    text.set_fontweight("bold")

fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.14, wspace=0.85)

# single visible border around the WHOLE figure (not per-panel) so the
# image reads as a self-contained block on the printed page
fig.patch.set_edgecolor("#2c3e50")
fig.patch.set_linewidth(2.0)

OUT = OUT_DIR / "dumbbell_text_quality"
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight",
            edgecolor=fig.get_edgecolor(), facecolor="white")
fig.savefig(OUT.with_suffix(".pdf"),                bbox_inches="tight",
            edgecolor=fig.get_edgecolor(), facecolor="white")
print(f"saved {OUT}.{{png,pdf}}")
