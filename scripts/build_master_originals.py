"""Run all 20 paper-benchmark models over every ORIGINAL frame and
save the predicted segmentation mask per (model, pH, gas, filename).

Output layout:
  master_originals/
    originals/      <-- already populated by hand
    ground_truth/   <-- already populated by hand
    predictions/<model_name>/ph_X.X/{co2,ch4}/<orig_filename>.png

Each predicted mask is a PNG with values {0, 255}: 0 = background,
255 = gas plume.  Filename matches the original frame (so master/originals
and master/predictions can be joined by basename).

Run as a SLURM GPU job — see scripts/build_master_originals.slurm.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path("/work/nvme/bgte/tislam6/ACID_Journal")
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
MASTER       = PROJECT_ROOT / "master_originals"
ORIG_ROOT    = PROJECT_ROOT / "dataset" / "original_annotated"

# Make scripts/ importable for train_utils + eval_metrics.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from visualize_predictions import load_model  # noqa: E402  (already validated)

# ── 20 paper-benchmark models with seed-42 checkpoints ────────────────────────
MODELS: List[Tuple[str, str, str]] = [
    # name, checkpoint, config
    ("resnet50_single",     "checkpoints/zoo/resnet50_single_seed42/best_acc.pth",     "configs/baselines/_base.yaml"),
    ("resnet50_dual",       "checkpoints/zoo/resnet50_dual_seed42/best_acc.pth",       "configs/baselines/_base.yaml"),
    ("resnet101_dual",      "checkpoints/zoo/resnet101_dual_seed42/best_acc.pth",      "configs/baselines/_base.yaml"),
    ("effnet_b3_dual",      "checkpoints/zoo/effnet_b3_dual_seed42/best_acc.pth",      "configs/baselines/_base.yaml"),
    ("convnext_base_dual",  "checkpoints/zoo/convnext_base_dual_seed42/best_acc.pth",  "configs/baselines/_base.yaml"),
    ("vit_b16_dual",        "checkpoints/zoo/vit_b16_dual_seed42/best_acc.pth",        "configs/baselines/_base.yaml"),
    ("swin_base_dual",      "checkpoints/zoo/swin_base_dual_seed42/best_acc.pth",      "configs/baselines/_base.yaml"),
    ("unet_r50",            "checkpoints/zoo/unet_r50_seed42/best_acc.pth",            "configs/baselines/_base.yaml"),
    ("deeplabv3p_r50",      "checkpoints/zoo/deeplabv3p_r50_seed42/best_acc.pth",      "configs/baselines/_base.yaml"),
    ("pspnet_r50",          "checkpoints/zoo/pspnet_r50_seed42/best_acc.pth",          "configs/baselines/_base.yaml"),
    ("segformer_b2",        "checkpoints/zoo/segformer_b2_seed42/best_acc.pth",        "configs/baselines/_base.yaml"),
    ("unet_hrnet_w48",      "checkpoints/zoo/unet_hrnet_w48_seed42/best_acc.pth",      "configs/baselines/_base.yaml"),
    ("unet_swinv2_b",       "checkpoints/zoo/unet_swinv2_b_seed42/best_acc.pth",       "configs/baselines/_base.yaml"),
    ("unet_convnextv2_b",   "checkpoints/zoo/unet_convnextv2_b_seed42/best_acc.pth",   "configs/baselines/_base.yaml"),
    ("clip_linear_probe",   "checkpoints/zoo/clip_linear_probe_seed42/best_acc.pth",   "configs/baselines/_base.yaml"),
    ("clip_zero_shot",      "checkpoints/vlm_baselines/clip_zero_shot_seed42/best_acc.pth",   "configs/baselines/_base.yaml"),
    ("clip_fine_tuned",     "checkpoints/vlm_baselines/clip_fine_tuned_seed42/best_acc.pth",  "configs/baselines/_base.yaml"),
    ("dinov2_linear_probe", "checkpoints/vlm_baselines/dinov2_linear_probe_seed42/best_acc.pth", "configs/baselines/_base.yaml"),
    ("ours_baseline",       "checkpoints/ours_baseline_seed42/best_acc.pth",           "configs/baseline.yaml"),
    ("ours_vlm",            "checkpoints/ours_vlm_seed42/best_acc.pth",                "configs/vlm.yaml"),
]

PH_BINS = ["5.0", "5.3", "5.6", "5.9", "6.2", "6.5"]
GASES   = ["co2", "ch4"]
IMG_SIZE = 256


# ── data prep ─────────────────────────────────────────────────────────────────

def load_gray_resized(path: Path, size: int = IMG_SIZE) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((size, size), dtype=np.uint8)
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def enumerate_originals():
    """Yield (ph, gas, filename, abs_path) for every original frame."""
    items = []
    for ph in PH_BINS:
        for gas in GASES:
            d = ORIG_ROOT / f"ph_{ph}" / f"{gas}_frame"
            if not d.exists():
                continue
            for p in sorted(d.glob("*.png")):
                items.append((ph, gas, p.name, p))
    return items


def build_partner_pool():
    """{ph: {gas: [resized np.uint8 arrays]}} — used as CH4 (or CO2) partners
    when the original has the opposite gas missing or just to provide the
    second stream's input."""
    pool = {ph: {g: [] for g in GASES} for ph in PH_BINS}
    for ph, gas, _, p in enumerate_originals():
        pool[ph][gas].append(load_gray_resized(p))
    return pool


# ── inference ─────────────────────────────────────────────────────────────────

def run_one_model(name: str, ckpt: str, cfg_path: str,
                  originals: List[Tuple[str, str, str, Path]],
                  partner_pool: dict, device: str) -> None:

    out_root = MASTER / "predictions" / name
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] loading {ckpt}", flush=True)
    try:
        model = load_model(str(PROJECT_ROOT / ckpt),
                           str(PROJECT_ROOT / cfg_path),
                           name, device)
    except Exception as e:
        print(f"[{name}] ERROR loading model: {e}", flush=True)
        traceback.print_exc()
        return

    saved = 0
    skipped = 0
    with torch.no_grad():
        for ph, gas, fname, abs_path in originals:
            primary = load_gray_resized(abs_path)

            partner_gas = "ch4" if gas == "co2" else "co2"
            partner_list = partner_pool[ph][partner_gas]
            if partner_list:
                partner = partner_list[saved % len(partner_list)]
                has_ch4 = 1.0
            else:
                partner = np.zeros_like(primary)
                has_ch4 = 0.0

            if gas == "co2":
                co2 = primary
                ch4 = partner
                has_ch4_val = has_ch4 if partner_list else 0.0
            else:
                co2 = partner if partner_list else np.zeros_like(primary)
                ch4 = primary
                has_ch4_val = 1.0  # primary is real CH4

            co2_t = torch.from_numpy(co2).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            ch4_t = torch.from_numpy(ch4).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            hc4_t = torch.tensor([[has_ch4_val]], dtype=torch.float32, device=device)

            try:
                out = model(co2_t, ch4_t, hc4_t)
            except Exception as e:
                print(f"[{name}] forward FAIL on {fname}: {e}", flush=True)
                skipped += 1
                continue

            seg_logits = out.get("seg_logits") if isinstance(out, dict) else None
            if seg_logits is None:
                print(f"[{name}] no seg_logits in output — stopping (model is "
                      f"classification-only).", flush=True)
                break

            pred = seg_logits[0].argmax(0).detach().cpu().numpy().astype(np.uint8)
            pred_png = (pred * 255).astype(np.uint8)

            out_dir = out_root / f"ph_{ph}" / gas
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / fname), pred_png)
            saved += 1

    del model
    torch.cuda.empty_cache()
    print(f"[{name}] saved {saved}, skipped {skipped}", flush=True)


def main():
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device — run this on a GPU node.", flush=True)
        sys.exit(1)
    device = "cuda"

    originals = enumerate_originals()
    print(f"Total originals: {len(originals)} "
          f"(across {len(PH_BINS)} pH bins x 2 gases)", flush=True)

    partner_pool = build_partner_pool()

    only = os.environ.get("MASTER_ONLY", "").strip()
    todo = MODELS if not only else [m for m in MODELS if m[0] in set(only.split(","))]
    print(f"Will run {len(todo)} model(s): "
          f"{', '.join(m[0] for m in todo)}", flush=True)

    for name, ckpt, cfg in todo:
        try:
            run_one_model(name, ckpt, cfg, originals, partner_pool, device)
        except Exception as e:
            print(f"[{name}] FATAL: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
