"""
CPU smoke test — runs in <1 minute on the login node.

Verifies:
- DualGasDataset loads a batch without errors
- Shapes and dtypes are as expected
- DualStreamBaselineNet forward produces correct output shapes
- CombinedLoss returns a finite scalar and backprop runs

Exits 0 on success, non-zero on any assertion failure.
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from baseline_model import DualStreamBaselineNet
from dual_gas_dataset import DualGasDataset
from losses_metrics import CombinedLoss
from train_utils import load_yaml, set_seed


def main() -> int:
    cfg_path = os.path.join(
        os.path.dirname(_HERE), "configs", "baseline.yaml"
    )
    cfg = load_yaml(cfg_path)
    set_seed(cfg["seed"])

    data = cfg["data"]
    ds = DualGasDataset(
        csv_path=os.path.join(data["ext_dir"], data["train_csv"]),
        dataset_root=data["dataset_root"],
        img_size=data["img_size"],
        seed=cfg["seed"],
    )
    print(f"Dataset len: {len(ds)}")
    assert len(ds) > 1000, "train set much smaller than expected"

    # Build a tiny batch via a DataLoader with a small batch
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    print("Batch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:10s} {tuple(v.shape)}  {v.dtype}")
        else:
            print(f"  {k:10s} {type(v).__name__}")

    assert batch["co2"].shape == (4, 1, 256, 256), batch["co2"].shape
    assert batch["ch4"].shape == (4, 1, 256, 256), batch["ch4"].shape
    assert batch["mask"].shape == (4, 256, 256), batch["mask"].shape
    assert batch["mask"].dtype == torch.long
    assert batch["label"].shape == (4,), batch["label"].shape
    assert batch["label"].dtype == torch.long
    assert batch["has_ch4"].shape == (4, 1), batch["has_ch4"].shape
    mu = int(batch["mask"].unique().max().item())
    ml = int(batch["mask"].unique().min().item())
    assert 0 <= ml and mu <= 2, f"mask out of range: [{ml},{mu}]"

    # Build model on CPU with pretrained weights disabled for speed
    print("Building DualStreamBaselineNet (pretrained=False for smoke speed)...")
    model = DualStreamBaselineNet(
        num_classes=cfg["model"]["num_classes"],
        seg_classes=cfg["model"]["seg_classes"],
        pretrained=False,
        dropout=cfg["model"]["dropout"],
        attn_heads=cfg["model"]["attn_heads"],
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.1f}M")

    model.eval()
    with torch.no_grad():
        out = model(batch["co2"], batch["ch4"], batch["has_ch4"])

    print("Output shapes:")
    for k, v in out.items():
        print(f"  {k:12s} {tuple(v.shape)}")

    assert out["cls_logits"].shape == (4, 3), out["cls_logits"].shape
    assert out["seg_logits"].shape == (4, 3, 256, 256), out["seg_logits"].shape

    # Loss + backward
    model.train()
    loss_fn = CombinedLoss(
        seg_classes=cfg["model"]["seg_classes"],
        cls_class_weights=cfg["loss"]["cls_class_weights"],
        seg_w=cfg["loss"]["seg_w"],
        cls_w=cfg["loss"]["cls_w"],
        dice_include_background=cfg["loss"]["dice_include_background"],
    )
    out = model(batch["co2"], batch["ch4"], batch["has_ch4"])
    losses = loss_fn(out, batch)
    print(f"Loss: total={losses['total'].item():.4f}  "
          f"seg={losses['seg'].item():.4f}  cls={losses['cls'].item():.4f}")
    assert torch.isfinite(losses["total"]), "non-finite loss"

    losses["total"].backward()
    # Check at least one gradient is non-zero
    has_grad = any(
        (p.grad is not None and p.grad.abs().sum().item() > 0)
        for p in model.parameters()
    )
    assert has_grad, "no non-zero gradients after backward"

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
