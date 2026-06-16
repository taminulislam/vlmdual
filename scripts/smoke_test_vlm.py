"""
CPU smoke test for VLMGuidedDualGasNet. Takes 30–90s on login node.
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dual_gas_dataset import DualGasDataset
from losses_metrics import CombinedLoss
from train_utils import load_yaml, set_seed
from vlm_model import VLMGuidedDualGasNet


def main() -> int:
    cfg = load_yaml(os.path.join(os.path.dirname(_HERE), "configs", "vlm.yaml"))
    set_seed(cfg["seed"])

    data = cfg["data"]
    ds = DualGasDataset(
        csv_path=os.path.join(data["ext_dir"], data["train_csv"]),
        dataset_root=data["dataset_root"],
        img_size=data["img_size"],
        seed=cfg["seed"],
    )
    from torch.utils.data import DataLoader

    batch = next(iter(DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:10s} {tuple(v.shape)}  {v.dtype}")

    print("Building VLMGuidedDualGasNet (pretrained_resnet=False for speed)...")
    model = VLMGuidedDualGasNet(
        num_classes=cfg["model"]["num_classes"],
        seg_classes=cfg["model"]["seg_classes"],
        pretrained_resnet=False,    # still loads the frozen CLIP; that is the point
        dropout=cfg["model"]["dropout"],
        attn_heads=cfg["model"]["attn_heads"],
        clip_model=cfg["model"]["clip_model"],
        clip_pretrained=cfg["model"]["clip_pretrained"],
        clip_dim=cfg["model"]["clip_dim"],
    )
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Params: {total:.1f}M total, {trainable:.1f}M trainable (CLIP frozen)")

    model.eval()
    with torch.no_grad():
        out = model(batch["co2"], batch["ch4"], batch["has_ch4"])
    for k, v in out.items():
        print(f"  {k:12s} {tuple(v.shape)}")
    assert out["cls_logits"].shape == (2, 3)
    assert out["seg_logits"].shape == (2, 3, 256, 256)
    assert out["cnn_proj"].shape == (2, cfg["model"]["clip_dim"])
    assert out["clip_emb"].shape == (2, cfg["model"]["clip_dim"])

    loss_fn = CombinedLoss(
        seg_classes=cfg["model"]["seg_classes"],
        cls_class_weights=cfg["loss"]["cls_class_weights"],
        seg_w=cfg["loss"]["seg_w"],
        cls_w=cfg["loss"]["cls_w"],
        align_w=cfg["loss"]["align_w"],
        dice_include_background=cfg["loss"]["dice_include_background"],
    )
    model.train()
    out = model(batch["co2"], batch["ch4"], batch["has_ch4"])
    losses = loss_fn(out, batch)
    print(
        f"Loss: total={losses['total'].item():.4f}  "
        f"seg={losses['seg'].item():.4f}  "
        f"cls={losses['cls'].item():.4f}  "
        f"align={losses['align'].item():.4f}"
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    # Verify CNN path got grads; CLIP path did not.
    cnn_has_grad = any(
        (p.grad is not None and p.grad.abs().sum().item() > 0)
        for p in model.cnn.parameters()
    )
    clip_no_grad = all(
        (p.grad is None or p.grad.abs().sum().item() == 0)
        for p in model.clip_visual.parameters()
    )
    assert cnn_has_grad, "CNN received no gradients"
    assert clip_no_grad, "CLIP received gradients (should be frozen)"
    print("VLM SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
