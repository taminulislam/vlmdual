"""Faceted line plot — seaborn `relplot` with kind="line".

Pattern matches the seaborn `dots` example:
    sns.set_theme(style="ticks")
    palette = sns.color_palette("rocket_r")
    sns.relplot(data=df, x="time", y="firing_rate",
                hue="coherence", size="choice", col="align",
                kind="line", size_order=["T1", "T2"], palette=palette,
                height=5, aspect=.75, facet_kws=dict(sharex=False))

Mapping for our paper:
    x        : label_fraction      (~ time)
    y        : score                (~ firing_rate)
    hue      : model                (~ coherence; sequential rocket_r)
    size     : metric               (~ choice; 2 metrics per task → 2 line widths)
    col      : task                 (~ align; 2 facets: Classification, Segmentation)

All values are taken verbatim from paper Table 7 (tab:label_eff).  For
each (model, fraction, metric) cell we synthesise three seed-replicate
values [μ-σ, μ, μ+σ] so seaborn's automatic ±1σ band is consistent with
the paper's reported (mean, sample-std).
"""
from pathlib import Path
import pandas as pd
import seaborn as sns

OUT_DIR = Path("/work/nvme/bgte/tislam6/ACID_Journal/paper/overleaf/figures/seaborn_viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── paper Table 7 values: (model, fraction, mean, std) per metric ───────────
ACC = [
    ("VLMDual (ours)",    0.10, 0.936, 0.041),
    ("VLMDual (ours)",    0.25, 0.941, 0.019),
    ("VLMDual (ours)",    0.50, 0.955, 0.023),
    ("VLMDual (ours)",    1.00, 0.983, 0.011),
    ("UNet-R50",          0.10, 0.927, 0.005),
    ("UNet-R50",          0.25, 0.937, 0.005),
    ("UNet-R50",          0.50, 0.942, 0.002),
    ("UNet-R50",          1.00, 0.958, 0.003),
    ("UNet-ConvNeXtV2-B", 0.10, 0.939, 0.004),
    ("UNet-ConvNeXtV2-B", 0.25, 0.939, 0.006),
    ("UNet-ConvNeXtV2-B", 0.50, 0.945, 0.005),
    ("UNet-ConvNeXtV2-B", 1.00, 0.968, 0.001),
]
BAL = [
    ("VLMDual (ours)",    0.10, 0.942, 0.037),
    ("VLMDual (ours)",    0.25, 0.948, 0.010),
    ("VLMDual (ours)",    0.50, 0.960, 0.020),
    ("VLMDual (ours)",    1.00, 0.985, 0.007),
    ("UNet-R50",          0.10, 0.919, 0.017),
    ("UNet-R50",          0.25, 0.923, 0.033),
    ("UNet-R50",          0.50, 0.934, 0.027),
    ("UNet-R50",          1.00, 0.958, 0.002),
    ("UNet-ConvNeXtV2-B", 0.10, 0.945, 0.034),
    ("UNet-ConvNeXtV2-B", 0.25, 0.936, 0.036),
    ("UNet-ConvNeXtV2-B", 0.50, 0.947, 0.026),
    ("UNet-ConvNeXtV2-B", 1.00, 0.969, 0.006),
]
MIOU = [
    ("VLMDual (ours)",    0.10, 0.676, 0.036),
    ("VLMDual (ours)",    0.25, 0.686, 0.024),
    ("VLMDual (ours)",    0.50, 0.690, 0.029),
    ("VLMDual (ours)",    1.00, 0.836, 0.004),
    ("UNet-R50",          0.10, 0.663, 0.054),
    ("UNet-R50",          0.25, 0.681, 0.059),
    ("UNet-R50",          0.50, 0.702, 0.023),
    ("UNet-R50",          1.00, 0.731, 0.011),
    ("UNet-ConvNeXtV2-B", 0.10, 0.704, 0.009),
    ("UNet-ConvNeXtV2-B", 0.25, 0.665, 0.042),
    ("UNet-ConvNeXtV2-B", 0.50, 0.713, 0.019),
    ("UNet-ConvNeXtV2-B", 1.00, 0.730, 0.013),
]
DICE = [
    ("VLMDual (ours)",    0.10, 0.790, 0.031),
    ("VLMDual (ours)",    0.25, 0.801, 0.019),
    ("VLMDual (ours)",    0.50, 0.805, 0.023),
    ("VLMDual (ours)",    1.00, 0.911, 0.003),
    ("UNet-R50",          0.10, 0.783, 0.042),
    ("UNet-R50",          0.25, 0.798, 0.045),
    ("UNet-R50",          0.50, 0.814, 0.017),
    ("UNet-R50",          1.00, 0.836, 0.008),
    ("UNet-ConvNeXtV2-B", 0.10, 0.816, 0.007),
    ("UNet-ConvNeXtV2-B", 0.25, 0.786, 0.032),
    ("UNet-ConvNeXtV2-B", 0.50, 0.822, 0.015),
    ("UNet-ConvNeXtV2-B", 1.00, 0.836, 0.009),
]

def expand(rows, metric_label, task_label):
    out = []
    for model, frac, mu, sd in rows:
        for v in (mu - sd, mu, mu + sd):
            out.append({"Model": model, "Fraction": frac,
                        "Metric": metric_label, "Task": task_label,
                        "Score": v})
    return out

df = pd.DataFrame(
      expand(ACC,  "Accuracy",          "Classification")
    + expand(BAL,  "Balanced Accuracy", "Classification")
    + expand(MIOU, "mIoU",              "Segmentation")
    + expand(DICE, "Dice",              "Segmentation")
)

# ── plot ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks")

# Sequential palette across the three models — VLMDual gets the darkest shade
palette = sns.color_palette("rocket_r", n_colors=3)
hue_order = ["UNet-R50", "UNet-ConvNeXtV2-B", "VLMDual (ours)"]

g = sns.relplot(
    data=df,
    x="Fraction", y="Score",
    hue="Model", hue_order=hue_order,
    size="Metric",
    size_order=["Accuracy", "Balanced Accuracy", "mIoU", "Dice"],
    col="Task", col_order=["Classification", "Segmentation"],
    kind="line",
    palette=palette, sizes=(1.6, 3.0),
    markers=True, dashes=False,
    errorbar="sd",
    height=4.4, aspect=1.10,
    facet_kws=dict(sharey=False),
)

for ax, task in zip(g.axes.flat, ["Classification", "Segmentation"]):
    ax.set_title(task, fontsize=12, fontweight="bold")   # panel label, not figure title
    ax.set_xlabel("Fraction of training labels")
    ax.set_xticks([0.10, 0.25, 0.50, 1.00])
    ax.grid(True, alpha=0.35)
g.set_ylabels("Score")

OUT = OUT_DIR / "relplot_facets_label_efficiency"
g.figure.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
g.figure.savefig(OUT.with_suffix(".pdf"),                bbox_inches="tight")
print(f"saved {OUT}.{{png,pdf}}")
