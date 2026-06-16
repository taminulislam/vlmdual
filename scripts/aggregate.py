"""
Aggregate all `results/<exp>/test_metrics.json` + `test_per_sample.csv` files
into a single master CSV, a LaTeX headline table, an ablation table, a
McNemar p-value matrix, and all figures for the Pattern Recognition paper.

Runs in seconds on the login node (pure CSV/JSON I/O, no model forward pass).

Usage:
    python scripts/aggregate.py
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stats import (  # noqa: E402
    bootstrap_ci,
    mcnemar,
    acc_fn,
    macro_f1_fn,
    bal_acc_fn,
    mcc_fn,
    kappa_fn,
)

PROJECT_ROOT = "/work/nvme/bgte/tislam6/ACID_Journal"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Which metrics to pull from each test_metrics.json — flat key -> column.
HEADLINE_COLS = {
    "cls_acc": "acc",
    "cls_top2_acc": "top2",
    "cls_bal_acc": "bal_acc",
    "cls_macro_f1": "macro_f1",
    "cls_mcc": "mcc",
    "cls_kappa": "kappa",
    "cls_auroc_macro": "auroc",
    "cls_auprc_macro": "auprc",
    "cls_ece": "ece",
    "cls_brier": "brier",
    "seg_mean_iou_fg": "miou",
    "seg_mean_dice_fg": "dice",
    "seg_pixel_acc": "pix_acc",
    "seg_freq_weighted_iou": "fwiou",
    "seg_hd95": "hd95",
    "seg_assd": "assd",
    "seg_bf_score": "bf",
    "eff_params_M": "params_M",
    "eff_flops_G": "flops_G",
    "eff_latency_ms": "latency_ms",
    "eff_throughput_imgs_per_sec": "throughput",
    "eff_peak_mem_GB": "peak_mem_GB",
}

# Pretty display names, in column order.
DISPLAY_ORDER = list(HEADLINE_COLS.values())


def _parse_exp_name(exp: str) -> Dict[str, str]:
    """Parse exp_name → {model, seed, fold, kind, extra...}."""
    out = {"model": exp, "seed": "", "fold": "", "kind": "main"}

    # VLM v2 CV:  vlm_v2_cv_fold{k}
    m = re.match(r"vlm_v2_cv_fold(?P<fold>\d+)$", exp)
    if m:
        out["model"] = "ours_vlm_v2"
        out["fold"] = m.group("fold")
        out["kind"] = "vlm_v2_cv"
        return out

    # VLM v2 ablation:  vlm_v2_<cell>_seed{s}
    m = re.match(r"vlm_v2_(?P<cell>[a-zA-Z0-9_]+?)_seed(?P<seed>\d+)$", exp)
    if m:
        out["model"] = f"vlm_v2_{m.group('cell')}"
        out["cell"] = m.group("cell")
        out["seed"] = m.group("seed")
        out["kind"] = "vlm_v2_abl"
        return out

    # Label-efficiency:  label_eff_<model>_frac{p}_seed{s}
    m = re.match(
        r"label_eff_(?P<model>.+?)_frac(?P<frac>[0-9p]+)_seed(?P<seed>\d+)$", exp
    )
    if m:
        frac_str = m.group("frac").replace("p", ".")
        out["model"] = m.group("model")
        out["seed"] = m.group("seed")
        out["label_fraction"] = frac_str
        out["kind"] = "label_eff"
        return out

    # Legacy v1 alpha sweep
    m = re.match(r"alpha_(?P<alpha>[\d.]+)_seed(?P<seed>\d+)$", exp)
    if m:
        out["model"] = f"ours_vlm_alpha{m.group('alpha')}"
        out["seed"] = m.group("seed")
        out["kind"] = "vlm_alpha"
        return out

    # Legacy CV:  <model>_cv_fold{k}
    m = re.match(r"(?P<model>.+?)_cv_fold(?P<fold>\d+)$", exp)
    if m:
        out["model"] = m.group("model")
        out["fold"] = m.group("fold")
        out["kind"] = "cv"
        return out

    # Generic <model>_seed{s}
    m = re.match(r"(?P<model>.+?)_seed(?P<seed>\d+)$", exp)
    if m:
        out["model"] = m.group("model")
        out["seed"] = m.group("seed")
        return out
    return out


def load_all_results() -> pd.DataFrame:
    rows: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*/test_metrics.json"))):
        exp = os.path.basename(os.path.dirname(path))
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        meta = _parse_exp_name(exp)
        row = {"exp": exp, **meta}
        for key, short in HEADLINE_COLS.items():
            val = data.get(key)
            row[short] = float(val) if val is not None else float("nan")
        row["n_test"] = int(data.get("n_test_samples", 0))
        # per-class F1
        pcf1 = data.get("cls_per_class_f1", [float("nan")] * 3)
        row["f1_healthy"] = float(pcf1[0])
        row["f1_transitional"] = float(pcf1[1])
        row["f1_acidotic"] = float(pcf1[2])
        rows.append(row)
    return pd.DataFrame(rows)


def write_master_csv(df: pd.DataFrame) -> str:
    path = os.path.join(RESULTS_DIR, "all_results.csv")
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")
    return path


def _fmt_mean_std(values: np.ndarray, fmt: str = "{:.3f}") -> str:
    if len(values) == 0 or np.all(np.isnan(values)):
        return "—"
    m = np.nanmean(values)
    s = np.nanstd(values)
    if len(values) == 1:
        return fmt.format(m)
    return f"{fmt.format(m)}±{fmt.format(s)}"


def _write_latex_table(table: "pd.DataFrame", path: str) -> None:
    """Emit a LaTeX-safe tabular. pandas to_latex(escape=True) escapes
    special chars (_, %, &, ...) in headers and cells; then we convert the
    Unicode ± (used by _fmt_mean_std) into \\,$\\pm$\\, so the math mode is
    rendered correctly instead of appearing as a raw byte LaTeX can't parse.
    """
    tex = table.to_latex(index=False, escape=True)
    tex = tex.replace("±", r"\,$\pm$\,")
    # pandas escapes '—' via text-escape but safest: use an em-dash that
    # always survives
    tex = tex.replace("—", "---")
    with open(path, "w") as f:
        f.write(tex)


def build_headline_table(df: pd.DataFrame) -> str:
    """Mean ± std over seeds for each (model) on the main split (kind='main')."""
    main = df[df["kind"] == "main"].copy()
    if main.empty:
        return "(no main-split results yet)"

    rows = []
    for model in sorted(main["model"].unique()):
        sub = main[main["model"] == model]
        row = {"model": model, "n_seeds": len(sub)}
        for col in DISPLAY_ORDER:
            row[col] = _fmt_mean_std(sub[col].to_numpy())
        rows.append(row)
    table = pd.DataFrame(rows)

    # Write as CSV (for easy eyeballing) + LaTeX
    csv_path = os.path.join(RESULTS_DIR, "headline_table.csv")
    table.to_csv(csv_path, index=False)
    tex_path = os.path.join(RESULTS_DIR, "headline_table.tex")
    _write_latex_table(table, tex_path)
    print(f"Wrote {csv_path} and {tex_path}")
    return csv_path


def build_cv_table(df: pd.DataFrame) -> str:
    cv = df[df["kind"] == "cv"].copy()
    if cv.empty:
        return "(no CV results yet)"
    rows = []
    for model in sorted(cv["model"].unique()):
        sub = cv[cv["model"] == model]
        row = {"model": model, "n_folds": len(sub)}
        for col in DISPLAY_ORDER:
            row[col] = _fmt_mean_std(sub[col].to_numpy())
        rows.append(row)
    table = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "cv_table.csv")
    table.to_csv(csv_path, index=False)
    tex_path = os.path.join(RESULTS_DIR, "cv_table.tex")
    _write_latex_table(table, tex_path)
    print(f"Wrote {csv_path} and {tex_path}")
    return csv_path


def build_vlm_ablation_tables(df: pd.DataFrame) -> None:
    # α sweep
    alpha_rows = df[df["kind"] == "vlm_alpha"].copy()
    if not alpha_rows.empty:
        alpha_vals = alpha_rows["model"].str.extract(r"alpha([\d.]+)").astype(float)
        alpha_rows["alpha"] = alpha_vals
        g = alpha_rows.groupby("alpha")
        rows = []
        for a, sub in g:
            row = {"alpha": a, "n_seeds": len(sub)}
            for col in DISPLAY_ORDER:
                row[col] = _fmt_mean_std(sub[col].to_numpy())
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "ablation_alpha.csv"), index=False)

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        try:
            means = g["bal_acc"].mean()
            stds = g["bal_acc"].std()
            ax[0].errorbar(means.index, means.values, yerr=stds.values, marker="o")
            ax[0].set_xlabel("α (alignment weight)")
            ax[0].set_ylabel("val_bal_acc")
            ax[0].set_title("VLM α sweep — classification")
            ax[0].grid(True, alpha=0.3)

            means = g["miou"].mean()
            stds = g["miou"].std()
            ax[1].errorbar(means.index, means.values, yerr=stds.values, marker="o", color="C1")
            ax[1].set_xlabel("α (alignment weight)")
            ax[1].set_ylabel("mean IoU (fg)")
            ax[1].set_title("VLM α sweep — segmentation")
            ax[1].grid(True, alpha=0.3)
        except Exception as e:
            print(f"alpha plot failed: {e}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "ablation_alpha.png"), dpi=150)
        plt.close(fig)
        print("Wrote results/ablation_alpha.{csv,png}")

    # CLIP arch sweep
    clip_rows = df[df["kind"] == "vlm_clip"].copy()
    if not clip_rows.empty:
        rows = []
        for arch in clip_rows["model"].unique():
            sub = clip_rows[clip_rows["model"] == arch]
            row = {"clip_arch": arch, "n_seeds": len(sub)}
            for col in DISPLAY_ORDER:
                row[col] = _fmt_mean_std(sub[col].to_numpy())
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "ablation_clip_arch.csv"), index=False)
        print("Wrote results/ablation_clip_arch.csv")


def build_mcnemar_matrix(df: pd.DataFrame) -> None:
    """Paired McNemar tests of ours_vlm (seed=42) against every baseline (seed=42).

    Requires per-sample prediction CSVs in results/<exp>/test_per_sample.csv.
    """
    main = df[(df["kind"] == "main") & (df["seed"] == "42")]
    if main.empty:
        print("No seed-42 per-sample preds for McNemar")
        return

    # Find ours_vlm predictions
    vlm_exp = "ours_vlm_seed42"
    vlm_psv = os.path.join(RESULTS_DIR, vlm_exp, "test_per_sample.csv")
    if not os.path.exists(vlm_psv):
        print("ours_vlm_seed42/test_per_sample.csv not found — skipping McNemar")
        return
    vlm_df = pd.read_csv(vlm_psv)
    labels = vlm_df["label"].to_numpy()
    ours = vlm_df["pred"].to_numpy()

    rows = []
    for _, row in main.iterrows():
        exp = row["exp"]
        if exp == vlm_exp:
            continue
        psv = os.path.join(RESULTS_DIR, exp, "test_per_sample.csv")
        if not os.path.exists(psv):
            continue
        bdf = pd.read_csv(psv)
        if len(bdf) != len(vlm_df):
            # different test set size — skip
            continue
        bp = bdf["pred"].to_numpy()
        try:
            b, c, p = mcnemar(labels, ours, bp)
        except Exception as e:
            print(f"mcnemar fail {exp}: {e}")
            continue
        rows.append(
            {
                "baseline": exp,
                "ours_correct_baseline_wrong": b,
                "baseline_correct_ours_wrong": c,
                "p_value": p,
                "sig_p<0.05": p < 0.05,
                "sig_p<0.01": p < 0.01,
            }
        )
    if not rows:
        return
    out = pd.DataFrame(rows).sort_values("p_value")
    out.to_csv(os.path.join(RESULTS_DIR, "mcnemar_matrix.csv"), index=False)
    print(f"Wrote results/mcnemar_matrix.csv  ({len(out)} comparisons)")


def build_bootstrap_ci(df: pd.DataFrame) -> None:
    """Bootstrap 95% CI for ours_vlm on the main split (seed=42)."""
    psv = os.path.join(RESULTS_DIR, "ours_vlm_seed42", "test_per_sample.csv")
    if not os.path.exists(psv):
        print("ours_vlm_seed42/test_per_sample.csv not found — skipping bootstrap CI")
        return
    pdf = pd.read_csv(psv)
    labels = pdf["label"].to_numpy()
    preds = pdf["pred"].to_numpy()
    rows = []
    for name, fn in [
        ("acc", acc_fn),
        ("bal_acc", bal_acc_fn),
        ("macro_f1", macro_f1_fn),
        ("mcc", mcc_fn),
        ("kappa", kappa_fn),
    ]:
        pt, lo, hi = bootstrap_ci(fn, labels, preds, n=1000, seed=42)
        rows.append({"metric": name, "point": pt, "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "bootstrap_ci_ours_vlm.csv"), index=False)
    print(f"Wrote results/bootstrap_ci_ours_vlm.csv")


def build_vlm_v2_ablation_table(df: pd.DataFrame) -> None:
    sub = df[df["kind"] == "vlm_v2_abl"].copy()
    if sub.empty:
        return
    # cell is stored in `model` = "vlm_v2_<cell>" by the parser. Extract it back.
    sub["cell"] = sub["model"].str.replace("vlm_v2_", "", regex=False)
    rows = []
    for cell in sorted(sub["cell"].unique()):
        g = sub[sub["cell"] == cell]
        row = {"cell": cell, "n_seeds": len(g)}
        for col in DISPLAY_ORDER:
            row[col] = _fmt_mean_std(g[col].to_numpy())
        rows.append(row)
    table = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "vlm_v2_ablation_table.csv")
    table.to_csv(csv_path, index=False)
    _write_latex_table(table, os.path.join(RESULTS_DIR, "vlm_v2_ablation_table.tex"))
    print(f"Wrote {csv_path} and .tex")

    # Bar chart of mIoU + bal_acc per cell
    try:
        cells = sorted(sub["cell"].unique())
        miou_m = [sub[sub["cell"] == c]["miou"].mean() for c in cells]
        miou_s = [sub[sub["cell"] == c]["miou"].std() for c in cells]
        bacc_m = [sub[sub["cell"] == c]["bal_acc"].mean() for c in cells]
        bacc_s = [sub[sub["cell"] == c]["bal_acc"].std() for c in cells]

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        x = np.arange(len(cells))
        ax[0].bar(x, miou_m, yerr=miou_s, color="C0", capsize=4)
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(cells, rotation=35, ha="right")
        ax[0].set_ylabel("mean IoU")
        ax[0].set_title("VLM v2 ablation — segmentation")
        ax[0].grid(True, axis="y", alpha=0.3)

        ax[1].bar(x, bacc_m, yerr=bacc_s, color="C1", capsize=4)
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(cells, rotation=35, ha="right")
        ax[1].set_ylabel("balanced accuracy")
        ax[1].set_title("VLM v2 ablation — classification")
        ax[1].grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "vlm_v2_ablation_bars.png"), dpi=150)
        plt.close(fig)
        print("Wrote results/vlm_v2_ablation_bars.png")
    except Exception as e:
        print(f"vlm_v2 ablation plot failed: {e}")


def build_label_efficiency_table(df: pd.DataFrame) -> None:
    sub = df[df["kind"] == "label_eff"].copy()
    if sub.empty:
        return

    # Grid: rows = models, cols = fractions. Values = mean ± std of bal_acc / miou.
    fractions = sorted(sub["label_fraction"].unique(), key=float)
    models = sorted(sub["model"].unique())

    rows = []
    for model in models:
        for frac in fractions:
            g = sub[(sub["model"] == model) & (sub["label_fraction"] == frac)]
            if g.empty:
                continue
            row = {"model": model, "frac": frac, "n_seeds": len(g)}
            for col in DISPLAY_ORDER:
                row[col] = _fmt_mean_std(g[col].to_numpy())
            rows.append(row)
    table = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "label_efficiency_table.csv")
    table.to_csv(csv_path, index=False)
    _write_latex_table(table, os.path.join(RESULTS_DIR, "label_efficiency_table.tex"))
    print(f"Wrote {csv_path} and .tex")

    # Line curves
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        colors = {"ours_vlm_v2": "C0", "unet_r50": "C1", "unet_convnextv2_b": "C2"}
        markers = {"ours_vlm_v2": "o", "unet_r50": "s", "unet_convnextv2_b": "^"}
        for metric, ax in zip(("bal_acc", "miou"), axes):
            for model in models:
                m_sub = sub[sub["model"] == model]
                xs = sorted(m_sub["label_fraction"].unique(), key=float)
                means, stds = [], []
                for f_ in xs:
                    g = m_sub[m_sub["label_fraction"] == f_][metric]
                    means.append(g.mean())
                    stds.append(g.std())
                ax.errorbar(
                    [float(x) for x in xs],
                    means,
                    yerr=stds,
                    label=model,
                    color=colors.get(model, None),
                    marker=markers.get(model, "o"),
                    linewidth=2,
                    capsize=4,
                )
            ax.set_xlabel("label fraction")
            ax.set_ylabel(metric)
            ax.set_title(f"Label efficiency — {metric}")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "label_efficiency_curves.png"), dpi=150)
        plt.close(fig)
        print("Wrote results/label_efficiency_curves.png")
    except Exception as e:
        print(f"label efficiency plot failed: {e}")


def build_vlm_v2_cv_table(df: pd.DataFrame) -> None:
    sub = df[df["kind"] == "vlm_v2_cv"].copy()
    if sub.empty:
        return
    row = {"model": "ours_vlm_v2", "n_folds": len(sub)}
    for col in DISPLAY_ORDER:
        row[col] = _fmt_mean_std(sub[col].to_numpy())
    table = pd.DataFrame([row])
    csv_path = os.path.join(RESULTS_DIR, "vlm_v2_cv_table.csv")
    table.to_csv(csv_path, index=False)
    _write_latex_table(table, os.path.join(RESULTS_DIR, "vlm_v2_cv_table.tex"))
    print(f"Wrote {csv_path} and .tex")


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_all_results()
    if df.empty:
        print("No results yet — nothing to aggregate.")
        return 0
    print(f"Loaded {len(df)} result entries")

    write_master_csv(df)
    build_headline_table(df)
    build_cv_table(df)
    build_vlm_ablation_tables(df)
    build_mcnemar_matrix(df)
    build_bootstrap_ci(df)

    # VLM v2 new artefacts
    build_vlm_v2_ablation_table(df)
    build_label_efficiency_table(df)
    build_vlm_v2_cv_table(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
