"""Timeseries-style line plot with error bands.

Pattern (matches the seaborn fmri example):
    sns.set_theme(style="darkgrid")
    sns.lineplot(x="timepoint", y="signal",
                 hue="region",     # 3 models
                 style="event",    # 2 metrics
                 data=df)

Data values are taken verbatim from paper Table 7 (tab:label_eff).
For each (model, fraction, metric) cell we synthesise three seed-replicate
values [μ-σ, μ, μ+σ] that *exactly* reproduce the paper's reported
(mean, sample-std) pair, so seaborn's `errorbar="sd"` shading agrees with
the table.

Output: paper/overleaf/figures/seaborn_viz/lineplot_bands_label_efficiency.{png,pdf}
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = Path("/work/nvme/bgte/tislam6/ACID_Journal/paper/overleaf/figures/seaborn_viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (model, fraction, mean, std) for two metrics — values from paper Table 7
ACC = [
    # VLMDual (ours)
    ("VLMDual (ours)",       0.10, 0.936, 0.041),
    ("VLMDual (ours)",       0.25, 0.941, 0.019),
    ("VLMDual (ours)",       0.50, 0.955, 0.023),
    ("VLMDual (ours)",       1.00, 0.983, 0.011),
    # UNet-R50
    ("UNet-R50",             0.10, 0.927, 0.005),
    ("UNet-R50",             0.25, 0.937, 0.005),
    ("UNet-R50",             0.50, 0.942, 0.002),
    ("UNet-R50",             1.00, 0.958, 0.003),
    # UNet-ConvNeXtV2-B
    ("UNet-ConvNeXtV2-B",    0.10, 0.939, 0.004),
    ("UNet-ConvNeXtV2-B",    0.25, 0.939, 0.006),
    ("UNet-ConvNeXtV2-B",    0.50, 0.945, 0.005),
    ("UNet-ConvNeXtV2-B",    1.00, 0.968, 0.001),
]
BAL = [
    ("VLMDual (ours)",       0.10, 0.942, 0.037),
    ("VLMDual (ours)",       0.25, 0.948, 0.010),
    ("VLMDual (ours)",       0.50, 0.960, 0.020),
    ("VLMDual (ours)",       1.00, 0.985, 0.007),
    ("UNet-R50",             0.10, 0.919, 0.017),
    ("UNet-R50",             0.25, 0.923, 0.033),
    ("UNet-R50",             0.50, 0.934, 0.027),
    ("UNet-R50",             1.00, 0.958, 0.002),
    ("UNet-ConvNeXtV2-B",    0.10, 0.945, 0.034),
    ("UNet-ConvNeXtV2-B",    0.25, 0.936, 0.036),
    ("UNet-ConvNeXtV2-B",    0.50, 0.947, 0.026),
    ("UNet-ConvNeXtV2-B",    1.00, 0.969, 0.006),
]

def expand(rows, metric_label):
    """For each (model, fraction, mean, std) row, emit three seed
    replicates {mean−std, mean, mean+std} — these reproduce the paper's
    (mean, sample-std) exactly with n=3."""
    out = []
    for model, frac, mu, sd in rows:
        for v in (mu - sd, mu, mu + sd):
            out.append({"model": model, "fraction": frac,
                        "metric": metric_label, "score": v})
    return out

df = pd.DataFrame(expand(ACC, "Accuracy") + expand(BAL, "Balanced Accuracy"))

# ── plot ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(8.0, 5.0))
sns.lineplot(
    data=df, x="fraction", y="score",
    hue="model",
    hue_order=["VLMDual (ours)", "UNet-ConvNeXtV2-B", "UNet-R50"],
    style="metric",
    style_order=["Accuracy", "Balanced Accuracy"],
    errorbar="sd",
    markers=True, dashes=True, linewidth=2.0, markersize=7,
    palette={"VLMDual (ours)":    "#27ae60",
             "UNet-ConvNeXtV2-B": "#2980b9",
             "UNet-R50":          "#7f7f7f"},
    ax=ax,
)
ax.set_xlabel("Fraction of training labels")
ax.set_ylabel("Score")
ax.set_xticks([0.10, 0.25, 0.50, 1.00])
ax.set_ylim(0.85, 1.005)
ax.legend(title=None, loc="lower right", ncol=2, fontsize=9)

OUT = OUT_DIR / "lineplot_bands_label_efficiency"
fig.tight_layout()
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".pdf"),                bbox_inches="tight")
print(f"saved {OUT}.{{png,pdf}}")
