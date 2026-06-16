"""Construct every model in the zoo and forward a dummy batch to catch shape
/ import errors before burning GPU time."""

from __future__ import annotations

import os
import sys
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models import build_model
from train_utils import load_yaml

CFG_BASE = {
    "seed": 42,
    "data": {"img_size": 256},
    "model": {
        "num_classes": 3,
        "seg_classes": 3,
        "pretrained": False,  # keep smoke test offline
        "dropout": 0.3,
        "attn_heads": 8,
        "clip_model": "ViT-B-16",
        "clip_pretrained": "openai",
        "clip_dim": 512,
    },
}

MODELS = [
    "ours_baseline",
    "ours_vlm",
    "resnet50_single",
    "resnet50_dual",
    "resnet101_dual",
    "effnet_b3_dual",
    "convnext_base_dual",
    "vit_b16_dual",
    "swin_base_dual",
    "unet_r50",
    "deeplabv3p_r50",
    "pspnet_r50",
    "segformer_b2",
    "clip_linear_probe",
]


def main() -> int:
    co2 = torch.zeros(2, 1, 256, 256)
    ch4 = torch.zeros(2, 1, 256, 256)
    has_ch4 = torch.tensor([[1.0], [0.0]])

    failed = []
    for name in MODELS:
        cfg = {**CFG_BASE}
        cfg["model"] = {**CFG_BASE["model"], "name": name}
        try:
            print(f"\n--- {name} ---", flush=True)
            m = build_model(name, cfg)
            n = sum(p.numel() for p in m.parameters()) / 1e6
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
            print(f"  params: {n:.1f}M  trainable: {trainable:.1f}M")
            m.eval()
            with torch.no_grad():
                out = m(co2, ch4, has_ch4)
            cls_s = tuple(out["cls_logits"].shape)
            seg_s = tuple(out["seg_logits"].shape)
            print(f"  cls_logits: {cls_s}  seg_logits: {seg_s}")
            assert cls_s == (2, 3), f"bad cls shape {cls_s}"
            assert seg_s == (2, 3, 256, 256), f"bad seg shape {seg_s}"
            print(f"  [OK]")
        except Exception as e:
            print(f"  [FAIL] {e}")
            traceback.print_exc()
            failed.append((name, str(e)))

    print()
    if failed:
        print(f"FAILED {len(failed)}/{len(MODELS)}:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        return 1
    print(f"ALL {len(MODELS)} MODELS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
