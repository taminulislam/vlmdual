"""Compute-vs-performance Pareto scatter — TRACE-style compact paper figure.

A small, square-ish figure with discrete pastel markers, a red-ringed
``VLMDual (ours)'' highlight, and dashed connectors from Ours to a
curated set of strong baselines, each annotated with the relative
Δ% improvement on segmentation mIoU.

Mapping for our paper:
    x   : Parameters (M)         (linear)
    y   : mIoU                   (large dynamic range — VLMDual stands out)

Only a curated set of comparison baselines is plotted to keep the
figure clean.  Δ% is computed against each baseline as
    (mIoU_ours - mIoU_baseline) / mIoU_baseline * 100.

Output: paper/overleaf/figures/seaborn_viz/pareto_scatter_compute_vs_perf.{png,pdf}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = Path("/work/nvme/bgte/tislam6/ACID_Journal/paper/overleaf/figures/seaborn_viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (model, params_M, miou)  — values from paper Tables 5 & 8
ROWS = [
    ("VLMDual (Ours)",     173.1, 0.836),
    ("ConvNeXt-B",         189.9, 0.713),
    ("CLIP-FT",             87.8, 0.432),
    ("UNet-ConvNeXtV2-B",   93.3, 0.726),
    ("ViT-B/16",           187.9, 0.738),
    ("UNet-R50",            33.6, 0.738),
    ("DeepLabV3+",          27.7, 0.730),
    ("SegFormer-B2",        27.6, 0.712),
    ("PSPNet",              24.6, 0.728),
]
df = pd.DataFrame(ROWS, columns=["model", "params_M", "miou"])

# Discrete pastel palette per model — VLMDual will be overridden to red below.
pastel = sns.color_palette("Set2", n_colors=len(df)).as_hex()
df["color"] = pastel
df.loc[df["model"] == "VLMDual (Ours)", "color"] = "#e63946"

# Connectors: VLMDual → these baselines, with Δ% labels
# (ViT-B/16 omitted — its Δ% is identical to UNet-R50's and shares the
#  crowded right-edge zone with ConvNeXt-B)
COMPARISONS = ["UNet-R50", "UNet-ConvNeXtV2-B", "DeepLabV3+",
               "PSPNet", "CLIP-FT", "ConvNeXt-B"]

# ── plot ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})

fig, ax = plt.subplots(figsize=(8.0, 6.6))
fig.patch.set_facecolor("white")
# very faint pinkish wash — barely visible, lets gridlines and markers dominate
ax.set_facecolor("#FDF2F5")
ax.patch.set_alpha(0.30)
ax.grid(True, which="major", linewidth=0.45, alpha=0.40, color="#6a6a6a")

ours = df[df["model"] == "VLMDual (Ours)"].iloc[0]

# 1) Connectors with Δ%  (drawn before scatter so they sit beneath the markers).
# Per-connector label position: (frac_ours_to_base, x_off_data, y_off_data)
# where frac=0 is at baseline, frac=1 is at Ours.
LABEL_POS = {
    "UNet-R50":          (0.55,  -3,  0.011),
    "ViT-B/16":          (0.55,  -2, -0.005),
    "UNet-ConvNeXtV2-B": (0.45,  -2, -0.014),
    "DeepLabV3+":        (0.30,   0, -0.004),
    "PSPNet":            (0.40,   0,  0.020),
    "CLIP-FT":           (0.50,   0,  0.020),
    "ConvNeXt-B":        (0.55,   0, -0.005),
}
for cname in COMPARISONS:
    c = df[df["model"] == cname].iloc[0]
    pct = (ours["miou"] - c["miou"]) / c["miou"] * 100
    ax.plot([ours["params_M"], c["params_M"]],
            [ours["miou"],     c["miou"]],
            linestyle=(0, (4, 3)), color="#5d6d7e", lw=1.1, alpha=0.7,
            zorder=2)
    f, dx, dy = LABEL_POS[cname]
    mx = c["params_M"] * (1 - f) + ours["params_M"] * f + dx
    my = c["miou"]     * (1 - f) + ours["miou"]     * f + dy
    ax.text(mx, my, f"+{pct:.2f}%",
            color="#e63946", fontsize=9, fontweight="bold", style="italic",
            ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="none", alpha=0.92))

# 2) All markers — pastel filled with thin dark edge
for _, r in df.iterrows():
    is_ours = (r["model"] == "VLMDual (Ours)")
    ax.scatter(r["params_M"], r["miou"],
               s=240 if is_ours else 150,
               facecolor=r["color"], edgecolor="#212121",
               linewidth=0.9, alpha=0.95, zorder=5)

# 3) Red ring around Ours
ax.scatter([ours["params_M"]], [ours["miou"]],
           s=520, marker="o", facecolor="none",
           edgecolor="#e63946", linewidth=2.4, zorder=6)

# 4) Model labels — italic small, with cream halo for legibility
LABEL_OFFSETS = {
    "VLMDual (Ours)":      (-10, 14),
    "ConvNeXt-B":          ( -8,  -3),  # left/below — clear of ViT-B/16 above
    "CLIP-FT":             (  9,   3),  # right/up
    "UNet-ConvNeXtV2-B":   ( 10,   2),
    "ViT-B/16":            ( -8,   6),  # left/up — clear of right edge
    "UNet-R50":            (  9,  -1),  # right of marker (line passes upper-right)
    "DeepLabV3+":          ( -9,   6),  # left/up
    "SegFormer-B2":        (  9,  -3),  # right/below
    "PSPNet":              ( -9,  -3),  # left/below
}
for _, r in df.iterrows():
    is_ours = (r["model"] == "VLMDual (Ours)")
    dx, dy = LABEL_OFFSETS.get(r["model"], (8, 6))
    ha = "right" if dx < 0 else "left"
    ax.annotate(
        r["model"],
        xy=(r["params_M"], r["miou"]),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=10.5 if is_ours else 9,
        fontweight="bold" if is_ours else "normal",
        style="italic" if not is_ours else "normal",
        color="#e63946" if is_ours else "#212121",
        ha=ha, zorder=7,
    )

# Axes
ax.set_xlabel("Parameters (M)", fontsize=12)
ax.set_ylabel("mIoU",           fontsize=12)
ax.set_xlim(0, 210)
ax.set_ylim(0.40, 0.92)
ax.tick_params(labelsize=10)
for sp in ["top", "right", "left", "bottom"]:
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_color("#5d6d7e")
    ax.spines[sp].set_linewidth(1.0)

fig.tight_layout()

OUT = OUT_DIR / "pareto_scatter_compute_vs_perf"
fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight",
            facecolor=fig.get_facecolor())
fig.savefig(OUT.with_suffix(".pdf"),                bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"saved {OUT}.{{png,pdf}}")
