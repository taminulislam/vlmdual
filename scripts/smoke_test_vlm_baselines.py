"""Login-safe smoke test for the four new VLM baselines."""

from __future__ import annotations

import os
import sys
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def main() -> int:
    from models import build_model

    cfg_base = {
        "seed": 42,
        "model": {
            "num_classes": 3,
            "seg_classes": 2,
            "pretrained": False,
            "dropout": 0.3,
            "attn_heads": 8,
            "clip_model": "ViT-B-16",
            "clip_pretrained": "openai",
            "cls_feat_dim": 512,
            "coop_n_ctx": 4,
        },
    }

    co2 = torch.zeros(2, 1, 256, 256)
    ch4 = torch.zeros(2, 1, 256, 256)
    has_ch4 = torch.tensor([[1.0], [0.0]])

    models_to_test = [
        "clip_zero_shot",
        "clip_fine_tuned",
        "dinov2_linear_probe",
        "coop",
    ]
    failed = []
    for name in models_to_test:
        cfg = {**cfg_base}
        cfg["model"] = {**cfg_base["model"], "name": name}
        try:
            print(f"\n--- {name} ---")
            m = build_model(name, cfg)
            n_total = sum(p.numel() for p in m.parameters()) / 1e6
            n_train = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
            print(f"  params: {n_total:.1f}M  trainable: {n_train:.1f}M")
            m.eval()
            with torch.no_grad():
                out = m(co2, ch4, has_ch4)
            cls_s = tuple(out["cls_logits"].shape)
            seg_s = tuple(out["seg_logits"].shape)
            print(f"  cls: {cls_s}  seg: {seg_s}")
            assert cls_s == (2, 3), f"{name}: wrong cls shape {cls_s}"
            assert seg_s == (2, 2, 256, 256), f"{name}: wrong seg shape {seg_s}"
            print(f"  [OK]")
        except Exception as e:
            print(f"  [FAIL] {e}")
            traceback.print_exc()
            failed.append((name, str(e)))

    if failed:
        print(f"\n{len(failed)}/{len(models_to_test)} FAILED:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        return 1
    print(f"\nALL {len(models_to_test)} VLM BASELINES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
