"""
visualize_predictions.py — Per-sample prediction images for paper.

Folder structure produced:
  results/figures/
    <model_name>/
      correct/   — 50 correctly predicted images
      incorrect/ — 10 incorrectly predicted images
    vlm_explanations/
      correct/   — 100 images with LLaVA explanation
      incorrect/ — 30 images with LLaVA explanation

Each image shows:
  [CO2 frame] [CH4 frame] [GT mask overlay] [Pred mask overlay]
  Title: GT class | Pred class | Confidence | Sample ID

VLM images additionally show the LLaVA explanation text below the panels.

Requires GPU (runs seg inference from checkpoint).

Usage:
  python scripts/visualize_predictions.py \\
      --checkpoint checkpoints/ours_vlm_seed42/best_acc.pth \\
      --config     configs/vlm.yaml \\
      --model-name ours_vlm

  # Run all models:
  python scripts/visualize_predictions.py --all-models
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

PROJECT_ROOT = "/work/nvme/bgte/tislam6/ACID_Journal"
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")
TEST_CSV     = os.path.join(DATASET_ROOT, "extended_dataset_50x", "test_annotations.csv")
ANN_CSV      = os.path.join(DATASET_ROOT, "extended_dataset_50x", "extended_annotations.csv")
EXPL_JSON    = os.path.join(RESULTS_DIR, "explanations_diagnostic.json")

CLASS_NAMES  = ["Healthy", "Transitional", "Acidotic"]
CLASS_COLORS = [
    (52,  168,  83),   # Healthy     — green
    (251, 188,   4),   # Transitional — yellow
    (234,  67,  53),   # Acidotic    — red
]
SEG_PALETTE = np.array([
    [ 30,  30,  30],   # 0 background
    [255, 100,   0],   # 1 gas plume — orange
], dtype=np.uint8)


# ── image helpers ─────────────────────────────────────────────────────────────

def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(SEG_PALETTE):
        rgb[mask == i] = color
    return rgb


def overlay_mask(frame_gray: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    frame_rgb = gray_to_rgb(frame_gray)
    mask_rgb  = mask_to_rgb(mask)
    return (frame_rgb * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)


def merge_mask(raw_mask: np.ndarray) -> np.ndarray:
    """Remap 3-class mask (0=bg,1=tube,2=gas) → 2-class (0=bg,1=gas)."""
    m = np.zeros_like(raw_mask)
    m[raw_mask == 2] = 1
    return m


# ── model / dataset helpers ───────────────────────────────────────────────────

def load_model(checkpoint: str, cfg_path: str, model_name: str,
               device: str = "cuda") -> torch.nn.Module:
    from train_utils import load_yaml
    from eval_metrics import build_model_from_type

    cfg = load_yaml(cfg_path)
    # override model name in config
    cfg = {**cfg, "model": {**cfg.get("model", {}), "name": model_name}}
    model = build_model_from_type("zoo", cfg, model_name_override=model_name)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Checkpoint layout written by train.py is {"epoch", "model", "optimizer", ...}.
    # Fall back to legacy "model_state_dict" or a bare state dict for compatibility.
    if isinstance(state, dict) and "model" in state:
        ms = state["model"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        ms = state["model_state_dict"]
    else:
        ms = state
    result = model.load_state_dict(ms, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f"  load_state_dict: missing={len(result.missing_keys)} "
              f"unexpected={len(result.unexpected_keys)}")
    return model.to(device).eval()


def find_partner(sample_row: pd.Series, full_ann: pd.DataFrame,
                 partner_gas: str) -> Optional[pd.Series]:
    """Find a partner-gas frame for sample_row by (ph, class, aug_id).
    Mirrors DualGasDataset._partner_index so viz gets the same pairing
    the model was trained with."""
    mask = (
        (full_ann["ph_value"]        == sample_row["ph_value"])   &
        (full_ann["class_name"]      == sample_row["class_name"]) &
        (full_ann["augmentation_id"] == sample_row["augmentation_id"]) &
        (full_ann["gas_type"]        == partner_gas)
    )
    matches = full_ann[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def load_frame_mask(row: pd.Series, size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    fp = os.path.join(DATASET_ROOT, row["frame_path"])
    mp = os.path.join(DATASET_ROOT, row["mask_path"])
    frame = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    mask  = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if frame is None: frame = np.zeros((size, size), dtype=np.uint8)
    if mask  is None: mask  = np.zeros((size, size), dtype=np.uint8)
    if frame.shape[0] != size:
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    if mask.shape[0] != size:
        mask  = cv2.resize(mask,  (size, size), interpolation=cv2.INTER_NEAREST)
    return frame, merge_mask(mask)


# ── inference loop ────────────────────────────────────────────────────────────

def run_inference(model: torch.nn.Module, cfg_path: str,
                  full_ann: pd.DataFrame, device: str = "cuda") -> List[dict]:
    """
    Run model on every test sample. Returns list of dicts with:
      sample_id, co2_frame, ch4_frame, gt_mask, pred_mask,
      label, pred, logits, is_correct
    """
    from train_utils import load_yaml
    cfg   = load_yaml(cfg_path)
    size  = cfg["data"].get("img_size", 256)
    test_df = pd.read_csv(TEST_CSV)
    ann_idx = full_ann.set_index("sample_id")

    records = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            sid   = row["sample_id"]
            label = int(row["class_id"])
            primary_gas = row["gas_type"]

            # Load the primary frame + its GT mask from disk.
            primary_frame, gt_mask = load_frame_mask(row, size)

            # Find the opposite-gas partner (same (ph, class, aug_id) bucket).
            partner_gas = "ch4" if primary_gas == "co2" else "co2"
            partner_row = find_partner(row, full_ann, partner_gas)
            if partner_row is not None:
                partner_frame, _ = load_frame_mask(partner_row, size)
                has_partner = 1.0
            else:
                partner_frame = np.zeros_like(primary_frame)
                has_partner   = 0.0

            # Slot-assign by gas to match DualGasDataset.__getitem__:
            #   co2 always in the co2 slot, ch4 always in the ch4 slot.
            if primary_gas == "co2":
                co2_frame = primary_frame
                ch4_frame = partner_frame
                has_ch4   = has_partner
            else:
                co2_frame = partner_frame if partner_row is not None \
                            else np.zeros_like(primary_frame)
                ch4_frame = primary_frame
                has_ch4   = 1.0  # primary is real CH4

            # tensors
            co2_t = torch.from_numpy(co2_frame).float().unsqueeze(0).unsqueeze(0) / 255.0
            ch4_t = torch.from_numpy(ch4_frame).float().unsqueeze(0).unsqueeze(0) / 255.0
            hc4_t = torch.tensor([[has_ch4]], dtype=torch.float32)

            co2_t = co2_t.to(device)
            ch4_t = ch4_t.to(device)
            hc4_t = hc4_t.to(device)

            out      = model(co2_t, ch4_t, hc4_t)
            logits   = out["cls_logits"][0].cpu().numpy().astype(np.float32)
            seg_pred = out["seg_logits"][0].argmax(0).cpu().numpy().astype(np.uint8)
            pred     = int(np.argmax(logits))

            records.append({
                "sample_id":  sid,
                "co2_frame":  co2_frame,
                "ch4_frame":  ch4_frame,
                "gt_mask":    gt_mask,
                "pred_mask":  seg_pred,
                "label":      label,
                "pred":       pred,
                "logits":     logits,
                "is_correct": (pred == label),
            })

    return records


# ── single-image renderer ─────────────────────────────────────────────────────

def save_sample_image(rec: dict, out_path: str, explanation: Optional[str] = None) -> None:
    """
    Save one image showing:
      [CO2 frame] [CH4 frame] [GT mask overlay] [Pred mask overlay]
    Optionally adds LLaVA explanation text below (for VLM panel).
    """
    co2   = rec["co2_frame"]
    ch4   = rec["ch4_frame"]
    gt    = overlay_mask(co2, rec["gt_mask"])
    pred_m= overlay_mask(co2, rec["pred_mask"])
    label = rec["label"]
    pred  = rec["pred"]
    logits= rec["logits"]
    sid   = rec["sample_id"]
    correct = rec["is_correct"]

    probs = np.exp(logits) / np.exp(logits).sum()
    conf  = probs[pred]
    mark  = "✓" if correct else "✗"
    edge  = "#27ae60" if correct else "#e74c3c"

    has_expl = explanation is not None
    fig_h = 5.5 if not has_expl else 8.5
    fig = plt.figure(figsize=(14, fig_h))

    if has_expl:
        gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[3, 1.5],
                               hspace=0.35, wspace=0.08)
    else:
        gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.08)

    panels = [
        (co2,   "CO₂ Frame",       "gray"),
        (ch4,   "CH₄ Frame",       "gray"),
        (gt,    "Ground Truth Mask", None),
        (pred_m,"Predicted Mask",   None),
    ]

    axes_img = []
    for c, (img, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(edge)
            sp.set_linewidth(2.5)
        axes_img.append(ax)

    # super-title
    gt_color   = [c/255 for c in CLASS_COLORS[label]]
    pred_color = [c/255 for c in CLASS_COLORS[pred]]
    fig.suptitle(
        f"GT: {CLASS_NAMES[label]}   |   Pred: {CLASS_NAMES[pred]} {mark} ({conf:.3f})   |   {sid}",
        fontsize=10, fontweight="bold", color=edge, y=0.98
    )

    # confidence bar (bottom of image panels area)
    bar_ax = fig.add_axes([0.08, 0.38 if has_expl else 0.08, 0.84, 0.04])
    x = 0
    bar_w = 840
    bar_img = np.ones((40, bar_w, 3), dtype=np.uint8) * 240
    for i, p in enumerate(probs):
        w = int(round(p * bar_w))
        bar_img[:, x:x+w] = CLASS_COLORS[i]
        x += w
    bar_ax.imshow(bar_img, aspect="auto")
    bar_ax.set_xticks([]); bar_ax.set_yticks([])
    bar_ax.set_title(
        "  ".join([f"{CLASS_NAMES[i]}: {probs[i]:.3f}" for i in range(3)]),
        fontsize=8, pad=3
    )

    # VLM explanation
    if has_expl:
        expl_ax = fig.add_subplot(gs[1, :])
        expl_ax.axis("off")
        wrapped = "\n".join(textwrap.wrap(explanation, width=120))
        expl_ax.text(
            0.0, 1.0, "LLaVA-1.5 Explanation:\n" + wrapped,
            transform=expl_ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                      edgecolor=edge, linewidth=1.5),
            wrap=True,
        )

    # legend
    patches = [mpatches.Patch(color=[c/255 for c in CLASS_COLORS[i]],
                               label=CLASS_NAMES[i]) for i in range(3)]
    seg_patches = [
        mpatches.Patch(color=[c/255 for c in SEG_PALETTE[0]], label="Seg: background"),
        mpatches.Patch(color=[c/255 for c in SEG_PALETTE[1]], label="Seg: gas plume"),
    ]
    fig.legend(handles=patches + seg_patches, loc="lower center", ncol=5,
               fontsize=7, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0 if not has_expl else -0.01))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

MODELS = {
    "ours_vlm":         ("checkpoints/ours_vlm_seed42/best_acc.pth",         "configs/vlm.yaml"),
    "ours_baseline":    ("checkpoints/ours_baseline_seed42/best_acc.pth",     "configs/baseline.yaml"),
    "segformer_b2":     ("checkpoints/zoo/segformer_b2_seed42/best_acc.pth",  "configs/baselines/_base.yaml"),
    "unet_convnextv2_b":("checkpoints/zoo/unet_convnextv2_b_seed42/best_acc.pth","configs/baselines/_base.yaml"),
}


def run_model(model_name: str, ckpt: str, cfg_path: str,
              full_ann: pd.DataFrame, device: str,
              n_correct: int = 50, n_incorrect: int = 10,
              explanations: Optional[dict] = None) -> None:

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Checkpoint: {ckpt}")
    print(f"{'='*60}")

    model = load_model(ckpt, cfg_path, model_name, device)
    records = run_inference(model, cfg_path, full_ann, device)

    correct   = [r for r in records if r["is_correct"]]
    incorrect = [r for r in records if not r["is_correct"]]

    # sample deterministically
    rng = np.random.default_rng(42)
    correct_sel   = rng.choice(len(correct),   min(n_correct,   len(correct)),   replace=False)
    incorrect_sel = rng.choice(len(incorrect), min(n_incorrect, len(incorrect)), replace=False)

    model_dir = os.path.join(FIGURES_DIR, model_name)
    correct_dir   = os.path.join(model_dir, "correct")
    incorrect_dir = os.path.join(model_dir, "incorrect")

    print(f"  Saving {len(correct_sel)} correct → {correct_dir}")
    for i, idx in enumerate(correct_sel):
        rec = correct[idx]
        expl = explanations.get(rec["sample_id"], {}).get("explanation") if explanations else None
        out = os.path.join(correct_dir, f"{rec['sample_id']}.png")
        save_sample_image(rec, out, explanation=expl)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(correct_sel)} correct done")

    print(f"  Saving {len(incorrect_sel)} incorrect → {incorrect_dir}")
    for i, idx in enumerate(incorrect_sel):
        rec = incorrect[idx]
        expl = explanations.get(rec["sample_id"], {}).get("explanation") if explanations else None
        out = os.path.join(incorrect_dir, f"{rec['sample_id']}.png")
        save_sample_image(rec, out, explanation=expl)
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(incorrect_sel)} incorrect done")

    print(f"  Done: {model_name}")


def run_vlm_explanations(full_ann: pd.DataFrame, device: str,
                         n_correct: int = 100, n_incorrect: int = 30) -> None:
    """VLM model inference + explanation panel."""
    print(f"\n{'='*60}")
    print(f"  VLM Explanations (ours_vlm_seed42)")
    print(f"{'='*60}")

    if not os.path.exists(EXPL_JSON):
        print("  explanations_diagnostic.json not found — skipping VLM panel")
        return

    explanations = {e["sample_id"]: e for e in json.load(open(EXPL_JSON))}

    ckpt     = "checkpoints/ours_vlm_seed42/best_acc.pth"
    cfg_path = "configs/vlm.yaml"
    model    = load_model(ckpt, cfg_path, "ours_vlm", device)
    records  = run_inference(model, cfg_path, full_ann, device)

    correct   = [r for r in records if r["is_correct"]]
    incorrect = [r for r in records if not r["is_correct"]]

    rng = np.random.default_rng(42)
    correct_sel   = rng.choice(len(correct),   min(n_correct,   len(correct)),   replace=False)
    incorrect_sel = rng.choice(len(incorrect), min(n_incorrect, len(incorrect)), replace=False)

    vlm_dir       = os.path.join(FIGURES_DIR, "vlm_explanations")
    correct_dir   = os.path.join(vlm_dir, "correct")
    incorrect_dir = os.path.join(vlm_dir, "incorrect")

    print(f"  Saving {len(correct_sel)} correct → {correct_dir}")
    for i, idx in enumerate(correct_sel):
        rec  = correct[idx]
        expl = explanations.get(rec["sample_id"], {}).get("explanation")
        out  = os.path.join(correct_dir, f"{rec['sample_id']}.png")
        save_sample_image(rec, out, explanation=expl)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(correct_sel)} done")

    print(f"  Saving {len(incorrect_sel)} incorrect → {incorrect_dir}")
    for i, idx in enumerate(incorrect_sel):
        rec  = incorrect[idx]
        expl = explanations.get(rec["sample_id"], {}).get("explanation")
        out  = os.path.join(incorrect_dir, f"{rec['sample_id']}.png")
        save_sample_image(rec, out, explanation=expl)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(incorrect_sel)} done")

    print("  VLM explanation panels done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=None,
                        help="Single model to run (default: all models)")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--vlm-only",   action="store_true")
    parser.add_argument("--n-correct",   type=int, default=50)
    parser.add_argument("--n-incorrect", type=int, default=10)
    parser.add_argument("--n-correct-vlm",   type=int, default=100)
    parser.add_argument("--n-incorrect-vlm", type=int, default=30)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    full_ann = pd.read_csv(ANN_CSV)

    # Load explanations once (used for both ours_vlm panels and vlm_explanations)
    explanations = None
    if os.path.exists(EXPL_JSON):
        explanations = {e["sample_id"]: e for e in json.load(open(EXPL_JSON))}
        print(f"Loaded {len(explanations)} LLaVA explanations")

    if args.vlm_only:
        run_vlm_explanations(full_ann, device,
                             n_correct=args.n_correct_vlm,
                             n_incorrect=args.n_incorrect_vlm)
        return

    models_to_run = MODELS if (args.all_models or args.model_name is None) \
                            else {args.model_name: MODELS[args.model_name]}

    for model_name, (ckpt, cfg_path) in models_to_run.items():
        # For ours_vlm, attach explanations
        expl = explanations if model_name == "ours_vlm" else None
        run_model(model_name, ckpt, cfg_path, full_ann, device,
                  n_correct=args.n_correct,
                  n_incorrect=args.n_incorrect,
                  explanations=expl)

    # VLM explanation folder (ours_vlm with full explanation text, more samples)
    run_vlm_explanations(full_ann, device,
                         n_correct=args.n_correct_vlm,
                         n_incorrect=args.n_incorrect_vlm)

    print(f"\nAll figures written to {FIGURES_DIR}")
    print("Structure:")
    for p in sorted(Path(FIGURES_DIR).rglob("*.png")):
        print(f"  {p.relative_to(FIGURES_DIR)}")


if __name__ == "__main__":
    main()
