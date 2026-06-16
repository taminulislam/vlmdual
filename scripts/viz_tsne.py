"""
t-SNE visualisation of CNN feature space before / after VLM distillation.

Compares ``ours_baseline`` (no VLM) vs ``ours_vlm_v2`` (cross-modal
distillation). Extracts fused_c5 global average-pooled features on the test
set, runs t-SNE, and plots side-by-side scatter plots colour-coded by
ground-truth class. The VLM v2 plot additionally projects the frozen CLIP
text prototypes into the same 2-D space (via a shared t-SNE fit) so the
reader can see that CNN features cluster *near* the semantic class
prototypes.

Usage:
    python scripts/viz_tsne.py \\
        --baseline-ckpt checkpoints/ours_baseline_seed42/best_acc.pth \\
        --vlm-ckpt      checkpoints/vlm_v2/full_seed42/best_acc.pth \\
        --baseline-config configs/baseline.yaml \\
        --vlm-config      configs/vlm_v2.yaml \\
        --n-samples 1000 \\
        --out results/figures/tsne_vlm_comparison.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.manifold import TSNE

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dual_gas_dataset import DualGasDataset
from train_utils import load_yaml, set_seed

CLASS_NAMES = ["Healthy", "Transitional", "Acidotic"]
CLASS_COLORS = ["#2ca02c", "#ff7f0e", "#d62728"]


def _extract_features(
    model, loader, device, n_max: int, use_vlm: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (feats [N, D], labels [N], text_proto [C, D] or None)."""
    feats_list = []
    labels_list = []
    text_proto = None
    collected = 0

    with torch.no_grad():
        for batch in loader:
            co2 = batch["co2"].to(device)
            ch4 = batch["ch4"].to(device)
            has_ch4 = batch["has_ch4"].to(device)
            out = model(co2, ch4, has_ch4)

            if use_vlm and "cnn_proj_c5" in out:
                # Use the CLIP-aligned projection
                f = out["cnn_proj_c5"].float().cpu().numpy()
                if text_proto is None and "clip_text_emb" in out:
                    text_proto = out["clip_text_emb"].float().cpu().numpy()
            elif "fused_c5" in out:
                # Fall back to raw CNN fused_c5 GAP
                pooled = torch.nn.functional.adaptive_avg_pool2d(out["fused_c5"], 1)
                f = pooled.flatten(1).float().cpu().numpy()
            else:
                # Use classifier logits as last resort
                f = out["cls_logits"].float().cpu().numpy()

            feats_list.append(f)
            labels_list.append(batch["label"].numpy())
            collected += f.shape[0]
            if collected >= n_max:
                break

    feats = np.concatenate(feats_list, axis=0)[:n_max]
    labels = np.concatenate(labels_list, axis=0)[:n_max]
    return feats, labels, text_proto


def _build_model_and_loader(cfg_path: str, ckpt_path: str, model_name: str, device):
    from models import build_model
    from torch.utils.data import DataLoader

    cfg = load_yaml(cfg_path)
    set_seed(cfg["seed"])
    data = cfg["data"]
    ds = DualGasDataset(
        csv_path=os.path.join(data["ext_dir"], data["test_csv"]),
        dataset_root=data["dataset_root"],
        img_size=data["img_size"],
        seed=cfg["seed"] + 30,
        merge_tube=bool(data.get("merge_tube", False)),
    )
    loader = DataLoader(
        ds,
        batch_size=data["batch_size"],
        shuffle=False,
        num_workers=data.get("num_workers", 4),
        pin_memory=True,
    )

    cfg_eval = {**cfg, "model": {**cfg["model"], "pretrained": False}}
    model = build_model(model_name, cfg_eval)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=False)
    model.to(device).eval()
    return model, loader


def _plot_tsne(ax, coords, labels, text_coords=None, title=""):
    for c in range(3):
        mask = labels == c
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=CLASS_COLORS[c],
            label=CLASS_NAMES[c],
            s=18,
            alpha=0.75,
            edgecolors="none",
        )
    if text_coords is not None:
        for i, c in enumerate(range(3)):
            ax.scatter(
                text_coords[i, 0],
                text_coords[i, 1],
                c=CLASS_COLORS[c],
                marker="*",
                s=380,
                edgecolors="black",
                linewidth=1.5,
                label=f"{CLASS_NAMES[c]} prototype",
            )
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(True, alpha=0.3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--vlm-ckpt", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--vlm-config", required=True)
    parser.add_argument("--baseline-name", default="ours_baseline")
    parser.add_argument("--vlm-name", default="ours_vlm_v2")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading baseline model ...")
    m1, l1 = _build_model_and_loader(
        args.baseline_config, args.baseline_ckpt, args.baseline_name, device
    )
    f1, y1, _ = _extract_features(m1, l1, device, args.n_samples, use_vlm=False)
    print(f"  baseline features: {f1.shape}")
    del m1
    torch.cuda.empty_cache()

    print("Loading VLM v2 model ...")
    m2, l2 = _build_model_and_loader(
        args.vlm_config, args.vlm_ckpt, args.vlm_name, device
    )
    f2, y2, text_proto = _extract_features(m2, l2, device, args.n_samples, use_vlm=True)
    print(f"  VLM features: {f2.shape}")
    del m2
    torch.cuda.empty_cache()

    # Fit t-SNE independently on each feature set (matched sample indices
    # but different embedding dims).
    print("Fitting t-SNE on baseline features ...")
    tsne1 = TSNE(
        n_components=2, perplexity=30, n_iter=1000, init="pca",
        random_state=42, verbose=1,
    )
    coords1 = tsne1.fit_transform(f1)

    print("Fitting t-SNE on VLM features (including text prototypes) ...")
    if text_proto is not None:
        # Concatenate CNN features + text prototypes so t-SNE places them in
        # the same 2-D space, then split the result.
        combined = np.concatenate([f2, text_proto], axis=0)
        tsne2 = TSNE(
            n_components=2, perplexity=30, n_iter=1000, init="pca",
            random_state=42, verbose=1,
        )
        coords_all = tsne2.fit_transform(combined)
        coords2 = coords_all[: f2.shape[0]]
        text_coords = coords_all[f2.shape[0] :]
    else:
        tsne2 = TSNE(
            n_components=2, perplexity=30, n_iter=1000, init="pca",
            random_state=42, verbose=1,
        )
        coords2 = tsne2.fit_transform(f2)
        text_coords = None

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _plot_tsne(
        axes[0], coords1, y1,
        title="(a) Dual-Base (no VLM distillation)",
    )
    _plot_tsne(
        axes[1], coords2, y2, text_coords=text_coords,
        title="(b) VLMDual v2 with cross-modal distillation",
    )
    handles, labels_ = axes[1].get_legend_handles_labels()
    # Deduplicate legend entries while preserving order
    seen = set()
    uniq = []
    for h, l in zip(handles, labels_):
        if l not in seen:
            seen.add(l)
            uniq.append((h, l))
    fig.legend(
        [h for h, _ in uniq],
        [l for _, l in uniq],
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=10,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
