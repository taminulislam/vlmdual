"""
Grad-CAM visualisation for the dual-stream CNN classifier.

Hooks on the CO2-branch `fused_c5` feature map, computes gradient-weighted
class activations, and overlays the heatmap on the CO2 input. Additionally
reports a quantitative agreement metric: cosine similarity between the
Grad-CAM activation map and the ground-truth gas-plume mask (after both
are normalised). Higher = model attends to the right region.

Usage:
    python scripts/viz_gradcam.py \\
        --checkpoint checkpoints/vlm_v2/full_seed42/best_acc.pth \\
        --config configs/vlm_v2.yaml \\
        --model-name ours_vlm_v2 \\
        --n-per-class 3 \\
        --out-dir results/figures/gradcam
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dual_gas_dataset import DualGasDataset
from train_utils import load_yaml, set_seed

CLASS_NAMES = ["Healthy", "Transitional", "Acidotic"]


def _get_fused_c5_module(model: torch.nn.Module) -> torch.nn.Module:
    """Return the module whose output we hook (fused_c5 location).

    For ours_baseline / ours_vlm / ours_vlm_v2 this is model.cnn.fuse.out_norm
    (the final layer of the CrossAttentionFusion). For GenericDualStream it
    is the fuse_proj output. We pick the first hook point that works.
    """
    # Walk a few known paths
    if hasattr(model, "cnn") and hasattr(model.cnn, "fuse"):
        return model.cnn.fuse  # VLMGuidedDualGasNet / v2
    if hasattr(model, "fuse"):
        return model.fuse  # ours_baseline directly
    if hasattr(model, "fuse_proj"):
        return model.fuse_proj  # GenericDualStream
    raise RuntimeError("cannot locate fused-feature module on the model")


def _gradcam_heatmap(
    model, co2, ch4, has_ch4, target_class: int
) -> np.ndarray:
    """Compute a [H, W] Grad-CAM heatmap in [0,1] for the given target class."""
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    target_module = _get_fused_c5_module(model)

    def fwd_hook(_, __, output):
        # For VLMGuidedDualGasNet v2 the fuse module returns a tensor, not a dict
        if isinstance(output, torch.Tensor):
            activations.append(output)
        elif isinstance(output, (tuple, list)):
            activations.append(output[0])

    def bwd_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = target_module.register_forward_hook(fwd_hook)
    bh = target_module.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    out = model(co2, ch4, has_ch4)
    score = out["cls_logits"][0, target_class]
    score.backward()

    fh.remove()
    bh.remove()

    if not activations or not gradients:
        return np.zeros((co2.shape[-2], co2.shape[-1]), dtype=np.float32)

    act = activations[0][0]    # [C, H, W]
    grad = gradients[0][0]     # [C, H, W]
    weights = grad.mean(dim=(1, 2))  # [C]
    cam = (weights[:, None, None] * act).sum(dim=0)  # [H, W]
    cam = torch.clamp(cam, min=0)
    cam = cam - cam.min()
    maxv = cam.max()
    if float(maxv) > 0:
        cam = cam / maxv
    cam_np = cam.detach().cpu().numpy()
    # Resize to input resolution
    cam_np = cv2.resize(cam_np, (co2.shape[-1], co2.shape[-2]), interpolation=cv2.INTER_LINEAR)
    return cam_np.astype(np.float32)


def _cam_mask_agreement(cam: np.ndarray, mask: np.ndarray) -> float:
    """Cosine similarity between flat CAM and binary foreground mask."""
    m = (mask > 0).astype(np.float32).flatten()
    c = cam.flatten()
    m = m - m.mean()
    c = c - c.mean()
    denom = np.linalg.norm(m) * np.linalg.norm(c)
    if denom < 1e-9:
        return 0.0
    return float((m @ c) / denom)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name", default="ours_vlm_v2")
    parser.add_argument("--n-per-class", type=int, default=3)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = cfg["data"]
    ds = DualGasDataset(
        csv_path=os.path.join(data["ext_dir"], data["test_csv"]),
        dataset_root=data["dataset_root"],
        img_size=data["img_size"],
        seed=cfg["seed"] + 20,
        merge_tube=bool(data.get("merge_tube", False)),
    )

    from models import build_model

    cfg_eval = {**cfg, "model": {**cfg["model"], "pretrained": False}}
    model = build_model(args.model_name, cfg_eval)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)  # required for Grad-CAM

    os.makedirs(args.out_dir, exist_ok=True)
    agreements = {c: [] for c in CLASS_NAMES}

    # Pick first N samples of each class
    picks: dict[str, List[int]] = {c: [] for c in CLASS_NAMES}
    for idx in range(len(ds)):
        cls = ds.df.iloc[idx]["class_name"]
        if len(picks.get(cls, [])) < args.n_per_class:
            picks[cls].append(idx)
        if all(len(v) >= args.n_per_class for v in picks.values()):
            break

    n_rows = sum(len(v) for v in picks.values())
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    row = 0
    for cls_name in CLASS_NAMES:
        for idx in picks[cls_name]:
            item = ds[idx]
            co2 = item["co2"].unsqueeze(0).to(device)
            ch4 = item["ch4"].unsqueeze(0).to(device)
            has_ch4 = item["has_ch4"].unsqueeze(0).to(device)
            mask_np = item["mask"].cpu().numpy()
            label = int(item["label"].item())

            cam = _gradcam_heatmap(model, co2, ch4, has_ch4, target_class=label)
            agreement = _cam_mask_agreement(cam, mask_np)
            agreements[cls_name].append(agreement)

            co2_np = (item["co2"][0].cpu().numpy() * 255).astype(np.uint8)

            axes[row, 0].imshow(co2_np, cmap="gray")
            axes[row, 0].set_title(f"{cls_name}  CO₂", fontsize=10)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title("GT plume mask", fontsize=10)
            axes[row, 1].axis("off")

            axes[row, 2].imshow(co2_np, cmap="gray")
            axes[row, 2].imshow(cam, cmap="jet", alpha=0.5)
            axes[row, 2].set_title(f"Grad-CAM (class {label})", fontsize=10)
            axes[row, 2].axis("off")

            axes[row, 3].imshow(cam, cmap="jet")
            axes[row, 3].set_title(f"agreement={agreement:.3f}", fontsize=10)
            axes[row, 3].axis("off")
            row += 1

    plt.tight_layout()
    out = os.path.join(args.out_dir, "gradcam_grid.png")
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # Summary numbers
    print("Grad-CAM ↔ GT mask agreement (cosine, higher = better):")
    for cls in CLASS_NAMES:
        if agreements[cls]:
            arr = np.array(agreements[cls])
            print(f"  {cls:14s}  mean={arr.mean():.3f}  ±{arr.std():.3f}  (n={len(arr)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
