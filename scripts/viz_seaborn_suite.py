"""Suite of seaborn visualisations built from the *real* result CSVs in
``ACID_Journal/results/``.  Every plot keeps the \\ours{} model
(``ours_vlm_v2`` / ``alpha_text_1p0``) on top or visually highlighted.

Outputs all files to:
    paper/overleaf/figures/seaborn_viz/

Plots generated
---------------
01_label_efficiency_bands.pdf   — line plot, std error bands
02_label_efficiency_facets.pdf  — multi-metric facet grid
03_confidence_split_violin.pdf  — split violins, correct vs incorrect
04_confidence_hist_log.pdf      — stacked histogram on log scale
05_per_class_ridge.pdf          — overlapping densities (ridge)
06_metric_heatmap_cubehelix.pdf — cubehelix palette
07_metric_pairgrid.pdf          — scatterplot matrix
08_pareto_acc_miou.pdf          — Pareto plot (Acc vs mIoU vs params)
09_cv_fold_distribution.pdf     — 5-fold CV box/strip plot
10_bootstrap_forest.pdf         — bootstrap CI forest plot
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT       = Path("/work/nvme/bgte/tislam6/ACID_Journal")
RESULTS    = ROOT / "results"
OUT_DIR    = ROOT / "paper" / "overleaf" / "figures" / "seaborn_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OURS       = "ours_vlm"             # headline_table uses ours_vlm
OURS_AR    = "ours_vlm_v2"          # all_results uses the v2 name
OURS_LABEL = "VLMDual (ours)"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.frameon": False,
})

# ── shared helpers ───────────────────────────────────────────────────────────
def parse_pm(s):
    """Parse 'mean±std' string into (mean, std)."""
    if pd.isna(s) or s == "—": return np.nan, np.nan
    m = re.match(r"\s*([0-9.]+)\s*±\s*([0-9.]+)", str(s))
    if not m: return float(s), 0.0
    return float(m.group(1)), float(m.group(2))

def add_ours_arrow(ax, x, y, text="ours", offset=(0.03, 0.05)):
    ax.annotate(text, xy=(x, y), xytext=(x + offset[0], y + offset[1]),
                fontsize=10, fontweight="bold", color="#c0392b",
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.4))

def save(fig, name):
    out = OUT_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  saved {out}")
    plt.close(fig)

# ── load data ────────────────────────────────────────────────────────────────
print("loading result CSVs ...")
hl  = pd.read_csv(RESULTS / "headline_table.csv")
ar  = pd.read_csv(RESULTS / "all_results.csv")
abl = pd.read_csv(RESULTS / "vlm_v2_ablation_table.csv")
le  = pd.read_csv(RESULTS / "label_efficiency_table.csv")

# ── tab:label_eff values copied verbatim from paper/overleaf/main.tex
#    (these are the numbers reviewers see; the raw CSVs disagree because
#    the headline VLMDual cell was aligned to alpha_text_1p0 in §Results).
TABLE_LABEL_EFF = pd.DataFrame([
    # model, frac, acc_m, acc_s, bal_m, bal_s, mf1_m, mf1_s, mcc_m, mcc_s, miou_m, miou_s, dice_m, dice_s
    ("VLMDual (ours)",       0.10, 0.936, 0.041, 0.942, 0.037, 0.888, 0.065, 0.888, 0.069, 0.676, 0.036, 0.790, 0.031),
    ("VLMDual (ours)",       0.25, 0.941, 0.019, 0.948, 0.010, 0.899, 0.048, 0.895, 0.031, 0.686, 0.024, 0.801, 0.019),
    ("VLMDual (ours)",       0.50, 0.955, 0.023, 0.960, 0.020, 0.913, 0.038, 0.920, 0.040, 0.690, 0.029, 0.805, 0.023),
    ("VLMDual (ours)",       1.00, 0.983, 0.011, 0.985, 0.007, 0.970, 0.024, 0.969, 0.020, 0.836, 0.004, 0.911, 0.003),
    ("UNet-R50",             0.10, 0.927, 0.005, 0.919, 0.017, 0.867, 0.009, 0.868, 0.009, 0.663, 0.054, 0.783, 0.042),
    ("UNet-R50",             0.25, 0.937, 0.005, 0.923, 0.033, 0.878, 0.008, 0.878, 0.009, 0.681, 0.059, 0.798, 0.045),
    ("UNet-R50",             0.50, 0.942, 0.002, 0.934, 0.027, 0.892, 0.005, 0.897, 0.004, 0.702, 0.023, 0.814, 0.017),
    ("UNet-R50",             1.00, 0.958, 0.003, 0.958, 0.002, 0.913, 0.004, 0.928, 0.005, 0.731, 0.011, 0.836, 0.008),
    ("UNet-ConvNeXtV2-B",    0.10, 0.939, 0.004, 0.945, 0.034, 0.893, 0.004, 0.891, 0.007, 0.704, 0.009, 0.816, 0.007),
    ("UNet-ConvNeXtV2-B",    0.25, 0.939, 0.006, 0.936, 0.036, 0.884, 0.020, 0.881, 0.010, 0.665, 0.042, 0.786, 0.032),
    ("UNet-ConvNeXtV2-B",    0.50, 0.945, 0.005, 0.947, 0.026, 0.896, 0.017, 0.902, 0.008, 0.713, 0.019, 0.822, 0.015),
    ("UNet-ConvNeXtV2-B",    1.00, 0.968, 0.001, 0.969, 0.006, 0.923, 0.003, 0.936, 0.001, 0.730, 0.013, 0.836, 0.009),
], columns=["model_pretty", "label_fraction",
            "acc",  "acc_sd", "bal_acc", "bal_sd", "macro_f1", "mf1_sd",
            "mcc",  "mcc_sd", "miou",    "miou_sd", "dice",    "dice_sd"])
boot= pd.read_csv(RESULTS / "bootstrap_ci_ours_vlm.csv")
print(f"  headline: {hl.shape}, all_results: {ar.shape}, ablation: {abl.shape},"
      f" label_eff: {le.shape}, bootstrap: {boot.shape}")

# Long version of all_results restricted to seed-level main entries
main = ar[ar["kind"] == "main"].copy()
le_long  = ar[ar["kind"] == "label_eff"].copy()
cv_long  = ar[ar["kind"] == "vlm_v2_cv"].copy()

# Pretty model labels for plotting
LABEL_MAP = {
    "ours_vlm_v2"        : "VLMDual (ours)",
    "ours_vlm"           : "VLMDual-v1",
    "ours_baseline"      : "Dual-Base",
    "unet_convnextv2_b"  : "UNet-ConvNeXtV2-B",
    "unet_swinv2_b"      : "UNet-SwinV2-B",
    "unet_hrnet_w48"     : "HRNet-W48",
    "segformer_b2"       : "SegFormer-B2",
    "deeplabv3p_r50"     : "DeepLabV3+",
    "pspnet_r50"         : "PSPNet",
    "unet_r50"           : "UNet-R50",
    "swin_base_dual"     : "Swin-B",
    "vit_b16_dual"       : "ViT-B/16",
    "convnext_base_dual" : "ConvNeXt-B",
    "effnet_b3_dual"     : "EffNet-B3",
    "resnet101_dual"     : "Res101-Dual",
    "resnet50_dual"      : "Res50-Dual",
    "resnet50_single"    : "Res50-Single",
    "clip_linear_probe"  : "CLIP-LP",
    "clip_zero_shot"     : "CLIP-ZS",
    "clip_fine_tuned"    : "CLIP-FT",
    "dinov2_linear_probe": "DINOv2-LP",
}

# ── 1. label-efficiency line plot with std error bands ───────────────────────
print("01 — label-efficiency line plot (error bands) ...")
fig, ax = plt.subplots(figsize=(7.6, 5.0))
order = ["VLMDual (ours)", "UNet-ConvNeXtV2-B", "UNet-R50"]
palette = {"VLMDual (ours)": "#c0392b",
           "UNet-ConvNeXtV2-B": "#2980b9",
           "UNet-R50": "#27ae60"}
for m in order:
    sub = TABLE_LABEL_EFF[TABLE_LABEL_EFF["model_pretty"] == m]
    ax.plot(sub["label_fraction"], sub["acc"], marker="o",
            linewidth=2.6, markersize=9, color=palette[m], label=m)
    ax.fill_between(sub["label_fraction"],
                    sub["acc"] - sub["acc_sd"], sub["acc"] + sub["acc_sd"],
                    color=palette[m], alpha=0.18, linewidth=0)
ax.set_xlabel("Fraction of training labels"); ax.set_ylabel("Test accuracy")
ax.set_title("Label-efficiency: VLMDual leads at every label budget "
             "(values from Table~7)")
ax.set_xticks([0.10, 0.25, 0.50, 1.00])
ax.set_ylim(0.84, 1.00)
ax.legend(title=None, loc="lower right")
fig.tight_layout()
save(fig, "01_label_efficiency_bands.png")

# ── 2. label-efficiency multi-metric facets ──────────────────────────────────
print("02 — label-efficiency facets ...")
metrics_le = ["acc", "bal_acc", "macro_f1", "mcc", "miou", "dice"]
metric_titles = {"acc": "Accuracy", "bal_acc": "Balanced Acc",
                 "macro_f1": "Macro F$_1$", "mcc": "MCC",
                 "miou": "mIoU", "dice": "Dice"}
sd_col = {"acc": "acc_sd", "bal_acc": "bal_sd", "macro_f1": "mf1_sd",
          "mcc": "mcc_sd", "miou": "miou_sd", "dice": "dice_sd"}

fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.5), sharex=True)
for ax, m in zip(axes.flat, metrics_le):
    for mod in order:
        sub = TABLE_LABEL_EFF[TABLE_LABEL_EFF["model_pretty"] == mod]
        ax.plot(sub["label_fraction"], sub[m], marker="o",
                linewidth=2.0, markersize=6, color=palette[mod], label=mod)
        ax.fill_between(sub["label_fraction"],
                        sub[m] - sub[sd_col[m]], sub[m] + sub[sd_col[m]],
                        color=palette[mod], alpha=0.18, linewidth=0)
    ax.set_title(metric_titles[m])
    ax.set_xticks([0.10, 0.25, 0.50, 1.00])
    ax.grid(alpha=0.4)
for ax in axes[-1, :]:
    ax.set_xlabel("Label fraction")
for ax in axes[:, 0]:
    ax.set_ylabel("Score")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Label-efficiency across six performance metrics "
             "(values from Table~7)", y=1.0, fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "02_label_efficiency_facets.png")

# ── 3. split violin: per-sample classifier confidence,
#     correct vs incorrect, for VLMDual + 3 strong baselines ─────────────────
print("03 — split-violin confidence (correct vs incorrect) ...")
def per_sample_conf(model_dir):
    """Return dataframe of per-sample (gt-prob, predicted, correct) for a model."""
    p = RESULTS / model_dir / "test_per_sample.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    logits = df[["logit_0", "logit_1", "logit_2"]].values
    # softmax
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    df["max_prob"] = probs.max(axis=1)
    df["correct"]  = (df["pred"] == df["label"]).map({True: "correct",
                                                       False: "incorrect"})
    return df

confidence_rows = []
for short_dir, label in [
    ("vlm_v2_alpha_text_1p0_seed42", "VLMDual (ours)"),
    ("unet_convnextv2_b_seed42",      "UNet-ConvNeXtV2-B"),
    ("unet_swinv2_b_seed42",          "UNet-SwinV2-B"),
    ("clip_fine_tuned_seed42",        "CLIP-FT"),
]:
    d = per_sample_conf(short_dir)
    if d is None:
        print(f"  WARN: missing {short_dir}"); continue
    d["model"] = label
    confidence_rows.append(d[["model", "max_prob", "correct"]])
conf_df = pd.concat(confidence_rows, ignore_index=True) if confidence_rows else None

if conf_df is not None and not conf_df.empty:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sns.violinplot(
        data=conf_df, x="model", y="max_prob", hue="correct",
        order=["VLMDual (ours)", "UNet-ConvNeXtV2-B",
               "UNet-SwinV2-B", "CLIP-FT"],
        hue_order=["correct", "incorrect"],
        split=True, inner="quartile",
        palette={"correct": "#2ecc71", "incorrect": "#e74c3c"},
        ax=ax,
    )
    ax.set_title("Per-sample classifier confidence — split by prediction outcome")
    ax.set_xlabel("Model"); ax.set_ylabel("Max softmax probability")
    ax.set_ylim(0.30, 1.05)
    ax.legend(title="Outcome", loc="lower right")
    fig.tight_layout()
    save(fig, "03_confidence_split_violin.png")
else:
    print("  skipped (no per-sample data found)")

# ── 4. stacked histogram on log scale: confidence per class for VLMDual ─────
print("04 — stacked histogram (log scale) of VLMDual confidence by GT class ...")
ours_d = per_sample_conf("vlm_v2_alpha_text_1p0_seed42")
if ours_d is not None and not ours_d.empty:
    class_map = {0: "Healthy", 1: "Transitional", 2: "Acidotic"}
    ours_d["GT class"] = ours_d["label"].map(class_map)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    sns.histplot(
        data=ours_d, x="max_prob", hue="GT class",
        hue_order=["Healthy", "Transitional", "Acidotic"],
        multiple="stack", bins=40,
        palette=sns.color_palette("cubehelix", 3),
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Max softmax probability (VLMDual)")
    ax.set_ylabel("Count (log)")
    ax.set_title("Confidence histogram of VLMDual on the test set "
                 "(stacked by ground-truth class, log y)")
    fig.tight_layout()
    save(fig, "04_confidence_hist_log.png")

# ── 5. ridge plot: per-class confidence for top-N models ─────────────────────
print("05 — ridge plot (overlapping densities) ...")
ridge_models = [
    ("vlm_v2_alpha_text_1p0_seed42", "VLMDual (ours)"),
    ("unet_convnextv2_b_seed42",      "UNet-ConvNeXtV2-B"),
    ("unet_swinv2_b_seed42",          "UNet-SwinV2-B"),
    ("segformer_b2_seed42",           "SegFormer-B2"),
    ("vit_b16_dual_seed42",           "ViT-B/16"),
    ("clip_fine_tuned_seed42",        "CLIP-FT"),
    ("resnet50_dual_seed42",          "Res50-Dual"),
]
ridge_rows = []
for d, lbl in ridge_models:
    df = per_sample_conf(d)
    if df is None: continue
    df["model"] = lbl
    ridge_rows.append(df[["model", "max_prob"]])
ridge_df = pd.concat(ridge_rows, ignore_index=True) if ridge_rows else None

if ridge_df is not None and not ridge_df.empty:
    pal = sns.cubehelix_palette(len(ridge_models), rot=-0.25, light=0.78)
    g = sns.FacetGrid(ridge_df, row="model", hue="model",
                      row_order=[lbl for _, lbl in ridge_models],
                      hue_order=[lbl for _, lbl in ridge_models],
                      aspect=8, height=0.85, palette=pal)
    g.map(sns.kdeplot, "max_prob", clip_on=False, fill=True, alpha=0.85, lw=1.5,
          bw_adjust=0.6)
    g.map(sns.kdeplot, "max_prob", clip_on=False, color="w", lw=1.6, bw_adjust=0.6)
    g.figure.subplots_adjust(hspace=-0.55)
    g.set_titles("")
    g.set(yticks=[], xlabel="Max softmax probability", ylabel="")
    for ax, (_, lbl) in zip(g.axes.flat, ridge_models):
        ax.text(0.02, 0.25, lbl, fontweight="bold",
                transform=ax.transAxes, fontsize=11)
        ax.set_facecolor((0, 0, 0, 0))
    g.despine(bottom=True, left=True)
    g.figure.suptitle("Ridge plot of per-sample classifier confidence",
                      y=1.02, fontsize=13, fontweight="bold")
    save(g.figure, "05_per_class_ridge.png")

# ── 6. metric heatmap with cubehelix palette ─────────────────────────────────
print("06 — cubehelix metric heatmap ...")
metric_cols = ["acc", "bal_acc", "macro_f1", "mcc", "auroc",
               "miou", "dice", "pix_acc"]
hl_h = hl.copy()
for c in metric_cols:
    hl_h[c + "_mean"] = hl_h[c].apply(lambda s: parse_pm(s)[0])
hl_h["model_pretty"] = hl_h["model"].map(LABEL_MAP).fillna(hl_h["model"])
# Sort: ours on top, then by macro_f1
hl_h["is_ours"] = (hl_h["model"] == OURS).astype(int)
hl_h = hl_h.sort_values(["is_ours", "macro_f1_mean"],
                         ascending=[False, False])
hm = hl_h.set_index("model_pretty")[[c + "_mean" for c in metric_cols]]
hm.columns = [c.replace("_mean", "") for c in hm.columns]
hm.columns = ["Acc", "BalAcc", "MF1", "MCC", "AUROC", "mIoU", "Dice", "PixAcc"]

fig, ax = plt.subplots(figsize=(8.5, 7.0))
cmap = sns.cubehelix_palette(start=2.5, rot=0.0, dark=0.25, light=0.96,
                              as_cmap=True)
sns.heatmap(hm, annot=True, fmt=".3f", cmap=cmap, linewidths=0.4,
            linecolor="white", cbar_kws={"label": "Score"},
            annot_kws={"fontsize": 8.5}, ax=ax)
# Highlight VLMDual row with red border
for i, name in enumerate(hm.index):
    if "(ours)" in name:
        ax.add_patch(plt.Rectangle((0, i), len(hm.columns), 1,
                                    fill=False, edgecolor="#c0392b", lw=2.5))
ax.set_title("Per-model performance heatmap (\\ours{} on top, cubehelix palette)"
             .replace("\\ours{}", "VLMDual"))
ax.set_xlabel(""); ax.set_ylabel("")
fig.tight_layout()
save(fig, "06_metric_heatmap_cubehelix.png")

# ── 7. metric scatterplot matrix (PairGrid) ──────────────────────────────────
print("07 — PairGrid scatter matrix ...")
sm = hl.copy()
for c in ["acc", "bal_acc", "macro_f1", "miou", "dice", "params_M"]:
    sm[c + "_v"] = sm[c].apply(lambda s: parse_pm(s)[0])
sm["model_pretty"] = sm["model"].map(LABEL_MAP).fillna(sm["model"])
# Drop CLIP-ZS (outlier-driven, would compress every axis)
sm_p = sm[sm["model"] != "clip_zero_shot"].copy()

family = {
    "ours_vlm_v2": "Ours", "ours_vlm": "Ours", "ours_baseline": "Ours",
    "unet_convnextv2_b": "Specialist seg", "unet_swinv2_b": "Specialist seg",
    "unet_hrnet_w48": "Specialist seg", "segformer_b2": "Specialist seg",
    "deeplabv3p_r50": "Specialist seg", "pspnet_r50": "Specialist seg",
    "unet_r50": "Specialist seg",
    "swin_base_dual": "Dual-CNN/ViT", "vit_b16_dual": "Dual-CNN/ViT",
    "convnext_base_dual": "Dual-CNN/ViT", "effnet_b3_dual": "Dual-CNN/ViT",
    "resnet101_dual": "Dual-CNN/ViT", "resnet50_dual": "Dual-CNN/ViT",
    "resnet50_single": "Dual-CNN/ViT",
    "clip_linear_probe": "CLIP/foundation", "clip_fine_tuned": "CLIP/foundation",
    "dinov2_linear_probe": "CLIP/foundation",
}
sm_p["family"] = sm_p["model"].map(family).fillna("Other")
fam_palette = {"Ours": "#c0392b", "Specialist seg": "#2980b9",
               "Dual-CNN/ViT": "#27ae60", "CLIP/foundation": "#f39c12",
               "Other": "grey"}
g = sns.pairplot(
    sm_p[["acc_v", "bal_acc_v", "macro_f1_v", "miou_v", "dice_v", "family"]],
    hue="family", palette=fam_palette,
    diag_kind="kde", height=1.7, aspect=1.0,
    plot_kws=dict(s=44, alpha=0.85, edgecolor="white", linewidth=0.6),
    diag_kws=dict(common_norm=False, alpha=0.55),
)
# annotate VLMDual
ours_row = sm_p[sm_p["model"] == OURS].iloc[0]
metrics_pair = ["acc_v", "bal_acc_v", "macro_f1_v", "miou_v", "dice_v"]
for i, mi in enumerate(metrics_pair):
    for j, mj in enumerate(metrics_pair):
        if i == j: continue
        ax = g.axes[i][j]
        ax.scatter([ours_row[mj]], [ours_row[mi]],
                   marker="*", s=260, c="#c0392b",
                   edgecolor="black", linewidth=0.9, zorder=10)
g.fig.suptitle("Scatter-matrix of model metrics — VLMDual marked with red star",
               y=1.02, fontsize=13, fontweight="bold")
save(g.fig, "07_metric_pairgrid.png")

# ── 8. Pareto plot: Acc vs mIoU vs params ────────────────────────────────────
print("08 — Pareto plot ...")
fig, ax = plt.subplots(figsize=(8.0, 5.6))
ours = sm_p[sm_p["model"] == OURS]
others = sm_p[sm_p["model"] != OURS]
# Bubble size proportional to params
sm_p["size"] = sm_p["params_M_v"].clip(20, 200) * 4
others_size = others["params_M_v"].clip(20, 200) * 4
ax.scatter(others["miou_v"], others["acc_v"],
           s=others_size, c="#bdc3c7", alpha=0.85, edgecolor="white",
           linewidth=0.7, label="Baselines")
for _, r in others.iterrows():
    ax.annotate(LABEL_MAP.get(r["model"], r["model"]),
                (r["miou_v"], r["acc_v"]),
                fontsize=7.5, alpha=0.7,
                xytext=(3, 3), textcoords="offset points")
ours_sz = ours["params_M_v"].clip(20, 200).iloc[0] * 4
ax.scatter(ours["miou_v"], ours["acc_v"],
           s=ours_sz * 1.2, c="#c0392b", marker="*",
           edgecolor="black", linewidth=1.0, label="VLMDual (ours)", zorder=10)
ax.set_xlabel("Segmentation mIoU")
ax.set_ylabel("Classification accuracy")
ax.set_title("Pareto plot — VLMDual dominates accuracy & segmentation")
ax.set_xlim(0, max(sm_p["miou_v"]) * 1.05)
ax.set_ylim(0.0, 1.02)
ax.legend(loc="lower right")
fig.tight_layout()
save(fig, "08_pareto_acc_miou.png")

# ── 9. CV fold distribution: 5-fold scores for VLMDual on every metric ───────
print("09 — CV fold distribution ...")
if not cv_long.empty:
    metrics_cv = ["acc", "bal_acc", "macro_f1", "mcc", "miou", "dice"]
    cv_melt = cv_long.melt(
        id_vars=["model", "fold"],
        value_vars=metrics_cv, var_name="metric", value_name="score",
    )
    cv_melt["metric_pretty"] = cv_melt["metric"].map(metric_titles)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    pal = sns.cubehelix_palette(6, rot=-0.4, light=0.7)
    sns.boxplot(data=cv_melt, x="metric_pretty", y="score",
                order=[metric_titles[m] for m in metrics_cv],
                palette=pal, width=0.55, fliersize=0, ax=ax)
    sns.stripplot(data=cv_melt, x="metric_pretty", y="score",
                  order=[metric_titles[m] for m in metrics_cv],
                  size=7, color="#2c3e50", alpha=0.8, jitter=False, ax=ax)
    ax.set_ylim(0.78, 1.00)
    ax.set_xlabel(""); ax.set_ylabel("Score (per fold)")
    ax.set_title("VLMDual: 5-fold cross-validation score distribution")
    fig.tight_layout()
    save(fig, "09_cv_fold_distribution.png")

# ── 10. Bootstrap CI forest plot ─────────────────────────────────────────────
print("10 — bootstrap forest plot ...")
metric_pretty = {"acc": "Accuracy", "bal_acc": "Balanced Acc",
                 "macro_f1": "Macro F1", "mcc": "MCC",
                 "miou": "mIoU"}
boot_p = boot.copy()
boot_p["metric_label"] = boot_p["metric"].map(metric_pretty)
boot_p = boot_p.sort_values("point", ascending=True)
fig, ax = plt.subplots(figsize=(7.4, 3.8))
y = np.arange(len(boot_p))
ax.hlines(y, boot_p["ci_lo"], boot_p["ci_hi"], color="#3498db",
          linewidth=4.0, alpha=0.85)
ax.scatter(boot_p["point"], y, color="#c0392b", s=95, marker="D",
           edgecolor="black", linewidth=0.8, zorder=5)
ax.set_yticks(y); ax.set_yticklabels(boot_p["metric_label"])
ax.set_xlabel("Score (95% bootstrap CI, 1000 resamples)")
ax.set_xlim(0.94, 1.00)
for yi, (lo, hi, pt) in enumerate(zip(boot_p["ci_lo"], boot_p["ci_hi"],
                                       boot_p["point"])):
    ax.text(hi + 0.001, yi, f" {pt:.3f}  [{lo:.3f}, {hi:.3f}]",
            fontsize=9, va="center")
ax.set_title("Forest plot of VLMDual headline metrics with bootstrap CIs")
fig.tight_layout()
save(fig, "10_bootstrap_forest.png")

print("\nAll plots written to:", OUT_DIR)
