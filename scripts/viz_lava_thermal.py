"""Render lava-style thermal overlays for the two test_pred originals.

Strategy
--------
Gas absorbs IR, so plume regions appear *darker* than the background in
the raw grayscale.  Inside the GT plume mask we invert the intensity
(so denser plume -> higher value), normalize per-image, and apply the
`inferno` LUT (FLIR-Lava equivalent: black -> purple -> red -> orange
-> yellow).  Outside the mask we keep the original grayscale, so the
animal/scene context is preserved.

Two outputs per frame:
  *_lava_overlay.png       hard-edged overlay
  *_lava_overlay_soft.png  feathered overlay (Gaussian alpha at edges)
"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/work/nvme/bgte/tislam6/ACID_Journal/test_pred/acidosis_explain")

PAIRS = [
    "MOV0429_frame_0331",
    "FLIR0286_frame_0719",
]

cmap = plt.get_cmap("inferno")


def colorize_plume(orig: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply the inferno LUT to (255 - orig) inside the mask."""
    plume_vals = orig[mask].astype(np.float32)
    inv = 255.0 - plume_vals                                  # invert: dense plume -> high
    inv_norm = (inv - inv.min()) / max(inv.max() - inv.min(), 1e-6)
    rgba = cmap(inv_norm)
    rgb_plume = (rgba[..., :3] * 255).astype(np.uint8)        # (N, 3) RGB

    out = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR).copy()
    out[mask] = rgb_plume[:, [2, 1, 0]]                       # RGB -> BGR
    return out


def colorize_plume_soft(orig: np.ndarray, mask: np.ndarray, blur_px: int = 9) -> np.ndarray:
    """Same but blend the colored plume on top of grayscale with a
    feathered alpha so the mask boundary isn't a hard cut."""
    if not mask.any():
        return cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)

    inv_full = 255.0 - orig.astype(np.float32)
    plume_vals = inv_full[mask]
    lo, hi = plume_vals.min(), plume_vals.max()
    inv_norm = np.clip((inv_full - lo) / max(hi - lo, 1e-6), 0, 1)

    rgba = cmap(inv_norm)
    rgb_full = (rgba[..., :3] * 255).astype(np.uint8)
    bgr_full = rgb_full[..., [2, 1, 0]]                       # (H,W,3) BGR

    alpha = mask.astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (blur_px * 2 + 1, blur_px * 2 + 1), 0)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    bg = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR).astype(np.float32)
    fg = bgr_full.astype(np.float32)
    out = (alpha * fg + (1.0 - alpha) * bg).astype(np.uint8)
    return out


for name in PAIRS:
    orig = cv2.imread(str(ROOT / f"{name}_original.png"),     cv2.IMREAD_GRAYSCALE)
    gt   = cv2.imread(str(ROOT / f"{name}_ground_truth.png"), cv2.IMREAD_GRAYSCALE)
    if orig is None or gt is None:
        raise FileNotFoundError(name)
    if gt.shape != orig.shape:
        gt = cv2.resize(gt, (orig.shape[1], orig.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    mask = gt >= 128

    hard = colorize_plume(orig, mask)
    soft = colorize_plume_soft(orig, mask, blur_px=9)

    cv2.imwrite(str(ROOT / f"{name}_lava_overlay.png"),       hard)
    cv2.imwrite(str(ROOT / f"{name}_lava_overlay_soft.png"),  soft)
    print(f"saved {name}_lava_overlay.png  and  _soft.png")
