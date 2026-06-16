"""
Lightweight smoke test for VLMGuidedDualGasNetV2:

- build model with pretrained=False (CPU-cheap, no internet)
- forward a 2-sample dummy batch
- verify all expected output keys
- compute CombinedLoss with v2 multi-scale + text terms
- one backward pass + confirm gradients exist on CNN projections

Login-safe (<60s, <500MB RAM).
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def main() -> int:
    from losses_metrics import CombinedLoss
    from models import build_model

    cfg = {
        "seed": 42,
        "model": {
            "name": "ours_vlm_v2",
            "num_classes": 3,
            "seg_classes": 2,
            "pretrained": False,
            "dropout": 0.3,
            "attn_heads": 8,
            "clip_model": "ViT-B-16",
            "clip_pretrained": "openai",
            "cls_feat_dim": 512,
        },
        "data": {"img_size": 256},
    }

    print("Building ours_vlm_v2 (pretrained=False) ...")
    model = build_model("ours_vlm_v2", cfg)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  params: total={total:.1f}M  trainable={trainable:.1f}M")

    # Dummy batch (2 samples, 256x256, both gases present)
    co2 = torch.zeros(2, 1, 256, 256)
    ch4 = torch.zeros(2, 1, 256, 256)
    has_ch4 = torch.tensor([[1.0], [0.0]])  # one with, one without
    mask = torch.zeros(2, 256, 256, dtype=torch.long)
    mask[0, 100:150, 100:150] = 1  # fake foreground for sample 0
    label = torch.tensor([0, 2], dtype=torch.long)  # Healthy, Acidotic

    print("Forward pass ...")
    out = model(co2, ch4, has_ch4)

    expected = {
        "cls_logits",
        "seg_logits",
        "cnn_proj_c5",
        "cnn_proj_c4",
        "cnn_proj_text",
        "clip_img_emb",
        "clip_text_emb",
    }
    missing = expected - set(out.keys())
    assert not missing, f"missing output keys: {missing}"
    print("  all expected keys present")
    for k in sorted(out.keys()):
        v = out[k]
        print(f"  {k:15s}  {tuple(v.shape)}")

    assert out["cls_logits"].shape == (2, 3), out["cls_logits"].shape
    assert out["seg_logits"].shape == (2, 2, 256, 256), out["seg_logits"].shape
    assert out["cnn_proj_c5"].shape[0] == 2
    assert out["cnn_proj_c4"].shape == out["cnn_proj_c5"].shape
    assert out["cnn_proj_text"].shape == out["cnn_proj_c5"].shape
    assert out["clip_img_emb"].shape == out["cnn_proj_c5"].shape
    assert out["clip_text_emb"].shape[0] == 3  # one per class

    print("\nBuilding CombinedLoss (v2 mode) ...")
    loss_fn = CombinedLoss(
        seg_classes=2,
        cls_class_weights=[1.0, 2.0, 1.0],
        seg_w=1.0,
        cls_w=1.0,
        align_w=0.0,
        align_dist_w=0.1,
        align_text_w=0.5,
        text_temperature=0.07,
        use_multiscale_dist=True,
        dice_include_background=False,
    )
    batch = {"mask": mask, "label": label}
    losses = loss_fn(out, batch)
    print(
        f"  total={losses['total'].item():.4f}  seg={losses['seg'].item():.4f}  "
        f"cls={losses['cls'].item():.4f}  dist={losses['align_dist'].item():.4f}  "
        f"text={losses['align_text'].item():.4f}"
    )
    assert torch.isfinite(losses["total"]), "non-finite loss"
    assert losses["align_dist"].item() > 0, "multi-scale distill loss should be nonzero"
    assert losses["align_text"].item() > 0, "text InfoNCE loss should be nonzero"

    print("\nBackward pass ...")
    losses["total"].backward()
    grads_ok = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.cnn.parameters()
    )
    assert grads_ok, "no CNN gradients after backward"
    # CLIP should have no gradients (frozen)
    clip_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.clip_visual.parameters()
    )
    assert not clip_grad, "CLIP received gradients (should be frozen)"

    print("\nVLM v2 SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
