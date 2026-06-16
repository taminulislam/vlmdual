"""
Hero figure for the paper — single-animal end-to-end visualisation.

Picks one representative test sample per class (Healthy / Transitional /
Acidotic) and renders a single-page figure showing the complete inference
pipeline:

    CO2 frame  |  CH4 frame  |  Predicted seg mask overlay on CO2  |
    softmax bars over 3 classes  |  LLaVA explanation text box

Loads `ours_vlm_v2` (or v1 as fallback) and the pre-generated
`results/explanations_diagnostic.json`.

Usage:
    python scripts/viz_hero_figure.py \\
        --checkpoint checkpoints/vlm_v2/full_seed42/best_acc.pth \\
        --config configs/vlm_v2.yaml \\
        --explanations results/explanations_diagnostic.json \\
        --out results/figures/hero_figure.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dual_gas_dataset import DualGasDataset, CLASS_TO_ID
from train_utils import load_yaml, set_seed

CLASS_NAMES = ["Healthy", "Transitional", "Acidotic"]
CLASS_COLORS = ["#2ca02c", "#ff7f0e", "#d62728"]
PH_RANGE = {
    "Healthy":      "pH 6.2–6.5",
    "Transitional": "pH 5.9",
    "Acidotic":     "pH 5.0–5.6",
}


def _select_hero_samples(
    ds: DualGasDataset,
    explanations: dict[str, dict],
    prefer_correct: bool = True,
) -> dict[str, int]:
    """Return dict class_name → dataset index.

    Priority: correctly-predicted sample with the highest confidence for the
    class (so the narrative in the figure is clean)."""
    df = ds.df
    chosen: dict[str, int] = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        # Candidates of this class
        cands = df[df["class_name"] == cls_name].index.tolist()
        if not cands:
            continue

        scored: list[tuple[float, int]] = []
        for idx in cands:
            sample_id = str(df.iloc[idx]["sample_id"])
            exp = explanations.get(sample_id)
            if exp is None:
                continue
            correct = bool(exp.get("correct", False))
            conf = float(exp.get("confidence", 0.0))
            if prefer_correct and not correct:
                conf -= 1.0  # penalise but still keep as fallback
            scored.append((conf, idx))
        if not scored:
            chosen[cls_name] = cands[0]
        else:
            scored.sort(reverse=True)
            chosen[cls_name] = scored[0][1]
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--explanations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", default="ours_vlm_v2")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset (test split)
    data = cfg["data"]
    test_csv = os.path.join(data["ext_dir"], data["test_csv"])
    ds = DualGasDataset(
        csv_path=test_csv,
        dataset_root=data["dataset_root"],
        img_size=data["img_size"],
        seed=cfg["seed"] + 10,
        merge_tube=bool(data.get("merge_tube", False)),
    )
    print(f"Test samples: {len(ds)}")

    # Load pre-generated explanations keyed by sample_id
    with open(args.explanations) as f:
        exp_list = json.load(f)
    exp_by_id = {str(r["sample_id"]): r for r in exp_list}
    print(f"Loaded {len(exp_by_id)} explanations")

    chosen = _select_hero_samples(ds, exp_by_id)
    print("Chosen samples:")
    for cls, idx in chosen.items():
        sid = ds.df.iloc[idx]["sample_id"]
        print(f"  {cls:14s} idx={idx}  sample_id={sid}")

    # Load model
    from models import build_model

    cfg_eval = {**cfg, "model": {**cfg["model"], "pretrained": False}}
    model = build_model(args.model_name, cfg_eval)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=False)
    model.to(device).eval()

    # Build the figure: 3 rows × 4 cols
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(
        3, 4,
        width_ratios=[1.0, 1.0, 1.0, 1.6],
        hspace=0.35, wspace=0.12,
    )

    softmax = torch.nn.Softmax(dim=1)

    for row, cls_name in enumerate(CLASS_NAMES):
        if cls_name not in chosen:
            continue
        idx = chosen[cls_name]
        item = ds[idx]
        co2 = item["co2"].unsqueeze(0).to(device)
        ch4 = item["ch4"].unsqueeze(0).to(device)
        has_ch4 = item["has_ch4"].unsqueeze(0).to(device)
        mask = item["mask"].cpu().numpy()
        label = int(item["label"].item())

        with torch.no_grad():
            out = model(co2, ch4, has_ch4)
            probs = softmax(out["cls_logits"].float())[0].cpu().numpy()
            seg_pred = out["seg_logits"].argmax(dim=1)[0].cpu().numpy()
        pred_id = int(probs.argmax())
        conf = float(probs[pred_id])

        co2_np = (item["co2"][0].numpy() * 255).astype(np.uint8)
        ch4_np = (item["ch4"][0].numpy() * 255).astype(np.uint8)

        # Col 0 — CO2 frame
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(co2_np, cmap="gray", vmin=0, vmax=255)
        ax0.set_title(f"{cls_name}\n({PH_RANGE[cls_name]}) — CO₂ input", fontsize=11)
        ax0.axis("off")

        # Col 1 — CH4 frame
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(ch4_np, cmap="gray", vmin=0, vmax=255)
        ax1.set_title("CH₄ input", fontsize=11)
        ax1.axis("off")

        # Col 2 — CO2 with predicted gas-plume overlay
        ax2 = fig.add_subplot(gs[row, 2])
        rgb = np.stack([co2_np] * 3, axis=-1).astype(np.float32)
        # overlay gas-plume class (==1 in merged masks) in red
        overlay = np.zeros_like(rgb)
        overlay[..., 0] = np.where(seg_pred == 1, 255, 0)
        blended = np.clip(0.65 * rgb + 0.35 * overlay, 0, 255).astype(np.uint8)
        ax2.imshow(blended)
        pred_name = CLASS_NAMES[pred_id]
        correct = pred_id == label
        title_color = "green" if correct else "red"
        ax2.set_title(
            f"Pred: {pred_name}  ({conf:.1%})\nGT: {cls_name}",
            fontsize=11,
            color=title_color,
        )
        ax2.axis("off")

        # Col 3 — text panel with softmax bars + LLaVA explanation
        ax3 = fig.add_subplot(gs[row, 3])
        ax3.axis("off")

        # Bar chart inset
        bar_ax = ax3.inset_axes([0.05, 0.65, 0.9, 0.28])
        bar_ax.bar(CLASS_NAMES, probs, color=CLASS_COLORS)
        bar_ax.set_ylim(0, 1.05)
        bar_ax.set_ylabel("softmax", fontsize=9)
        bar_ax.tick_params(axis="x", labelsize=9)
        bar_ax.tick_params(axis="y", labelsize=8)
        bar_ax.grid(True, axis="y", alpha=0.3)

        # Explanation text
        sid = str(ds.df.iloc[idx]["sample_id"])
        exp_text = exp_by_id.get(sid, {}).get(
            "explanation", "(no LLaVA explanation available)"
        )
        # Truncate for figure readability
        if len(exp_text) > 600:
            exp_text = exp_text[:597] + "..."
        ax3.text(
            0.02, 0.02, exp_text,
            transform=ax3.transAxes,
            fontsize=8.5,
            wrap=True,
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#f7f7f7",
                edgecolor="#cccccc",
            ),
        )

    fig.suptitle(
        "End-to-end VLMDual inference: input → segmentation → classification → explanation",
        fontsize=14,
        y=0.995,
    )

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
