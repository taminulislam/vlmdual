"""Grouped split-violin plot of per-class F1 scores.

Data values are taken verbatim from paper Table 4 (tab:main_cls).
Two hue groups (split violins):
    'Baselines' = 17 supervised / VLM-foundation baselines from tab:main_cls
                  (CLIP-ZS dropped as outlier).
    'VLMDual'   = 4 \\ours{} variants:
                  Dual-Base, VLMDual-v1,
                  VLMDual (visual_c4_c5), VLMDual (alpha_text_1.0).

Output: paper/overleaf/figures/seaborn_viz/violin_grouped_per_class_f1.{png,pdf}
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = Path("/work/nvme/bgte/tislam6/ACID_Journal/paper/overleaf/figures/seaborn_viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (model, group, F1_Healthy, F1_Transitional, F1_Acidotic) — paper Table 4
ROWS = [
    # Baselines
    ("Res50-Single",        "Baselines", 0.944, 0.885, 0.964),
    ("Res50-Dual",          "Baselines", 0.964, 0.851, 0.970),
    ("Res101-Dual",         "Baselines", 0.964, 0.891, 0.978),
    ("EffNet-B3",           "Baselines", 0.962, 0.908, 0.968),
    ("ConvNeXt-B",          "Baselines", 0.975, 0.945, 0.976),
    ("ViT-B/16",            "Baselines", 0.983, 0.945, 0.983),
    ("Swin-B",              "Baselines", 0.980, 0.950, 0.980),
    ("UNet-R50",            "Baselines", 0.978, 0.941, 0.977),
    ("DeepLabV3+",          "Baselines", 0.983, 0.944, 0.983),
    ("PSPNet",              "Baselines", 0.979, 0.861, 0.995),
    ("SegFormer-B2",        "Baselines", 0.976, 0.946, 0.976),
    ("HRNet-W48",           "Baselines", 0.991, 0.977, 0.994),
    ("UNet-SwinV2-B",       "Baselines", 0.976, 0.955, 0.974),
    ("UNet-ConvNeXtV2-B",   "Baselines", 0.978, 0.948, 0.974),
    ("CLIP-LP",             "Baselines", 0.930, 0.877, 0.953),
    ("CLIP-FT",             "Baselines", 0.981, 0.925, 0.984),
    ("DINOv2-LP",           "Baselines", 0.968, 0.948, 0.967),
    # VLMDual variants (ours)
    ("Dual-Base",                     "VLMDual", 0.948, 0.932, 0.957),
    ("VLMDual-v1",                    "VLMDual", 0.966, 0.945, 0.969),
    ("VLMDual (visual_c4_c5)",        "VLMDual", 0.980, 0.905, 0.981),
    ("VLMDual (alpha_text_1.0)",      "VLMDual", 0.984, 0.950, 0.984),
]
df = (
    pd.DataFrame(ROWS, columns=["model", "group", "Healthy",
                                 "Transitional", "Acidotic"])
      .melt(id_vars=["model", "group"],
            value_vars=["Healthy", "Transitional", "Acidotic"],
            var_name="Class", value_name="F1")
)

# ── plot ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="dark")
fig, ax = plt.subplots(figsize=(7.6, 4.6))
sns.violinplot(
    data=df, x="Class", y="F1", hue="group",
    hue_order=["Baselines", "VLMDual"],
    split=True, inner="quart", fill=False,
    palette={"VLMDual": "g", "Baselines": "k"},
    ax=ax,
)
ax.set_xlabel("Diagnostic class")
ax.set_ylabel("Per-class F$_1$")
ax.set_ylim(0.83, 1.005)
ax.legend(title="Method group", loc="lower right")

OUT = OUT_DIR / "violin_grouped_per_class_f1"
fig.tight_layout()
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".pdf"),                bbox_inches="tight")
print(f"saved {OUT}.{{png,pdf}}")
