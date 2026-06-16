"""
Test-set evaluator. Loads a trained checkpoint, runs the full journal-grade
metric suite on `test_annotations.csv`, and writes:

    results/<exp>/test_metrics.json   — all scalar metrics
    results/<exp>/test_preds.npz      — raw cls_logits, labels, (optional) seg summary
    results/<exp>/test_per_sample.csv — per-sample pred/label for bootstrap & McNemar

Usage:
    python scripts/eval_metrics.py --config configs/baseline.yaml \
        --checkpoint checkpoints/baseline/best_acc.pth \
        --model-type baseline \
        --exp-name ours_baseline_seed42
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dual_gas_dataset import DualGasDataset, CLASS_TO_ID
from journal_metrics import (
    SegmentationAccumulator,
    classification_metrics,
    efficiency_metrics,
)
from train_utils import load_yaml, set_seed

CLASS_NAMES = ["Healthy", "Transitional", "Acidotic"]


def build_model_from_type(
    model_type: str, cfg: dict, model_name_override: str | None = None
) -> torch.nn.Module:
    """Dispatch to the correct factory.

    model_type:
      - "baseline" : legacy DualStreamBaselineNet (reads cfg.model.*)
      - "vlm"      : legacy VLMGuidedDualGasNet (reads cfg.model.*)
      - "zoo"      : generic dispatch via models.build_model(cfg.model.name)
    """
    if model_type == "zoo":
        from models import build_model

        if model_name_override:
            cfg = {**cfg, "model": {**cfg.get("model", {}), "name": model_name_override}}
        if "name" not in cfg.get("model", {}):
            raise ValueError("zoo model_type requires cfg.model.name or --model-name")
        # pretrained=False for eval rebuild — state dict will supply weights
        cfg_eval = {**cfg, "model": {**cfg["model"], "pretrained": False}}
        return build_model(cfg_eval["model"]["name"], cfg_eval)

    if model_type == "baseline":
        from baseline_model import DualStreamBaselineNet

        return DualStreamBaselineNet(
            num_classes=cfg["model"]["num_classes"],
            seg_classes=cfg["model"]["seg_classes"],
            pretrained=False,
            dropout=cfg["model"]["dropout"],
            attn_heads=cfg["model"]["attn_heads"],
        )
    elif model_type == "vlm":
        from vlm_model import VLMGuidedDualGasNet

        return VLMGuidedDualGasNet(
            num_classes=cfg["model"]["num_classes"],
            seg_classes=cfg["model"]["seg_classes"],
            pretrained_resnet=False,
            dropout=cfg["model"]["dropout"],
            attn_heads=cfg["model"]["attn_heads"],
            clip_model=cfg["model"]["clip_model"],
            clip_pretrained=cfg["model"]["clip_pretrained"],
            clip_dim=cfg["model"]["clip_dim"],
        )
    else:
        raise ValueError(f"unknown model_type {model_type!r}")


CH4_MODES = ("real", "gate_random", "gate_zero", "co2_only")


@torch.no_grad()
def run_forward_pass(model, loader, device, cfg, ch4_mode: str = "real"):
    model.eval()
    all_logits = []
    all_labels = []
    all_sample_ids = []
    seg_classes = cfg.get("model", {}).get("seg_classes", 3)
    seg = SegmentationAccumulator(num_classes=seg_classes)

    # Optional VLM v2 projection / CLIP embeddings, collected only if the
    # model output dict contains them.  Used post-hoc to compute zero-shot,
    # retrieval, and CKA metrics.
    vlm_keys = ("cnn_proj_c5", "cnn_proj_text", "clip_img_emb")
    vlm_buffers: Dict[str, list] = {k: [] for k in vlm_keys}
    clip_text_proto = None

    for batch in loader:
        co2 = batch["co2"].to(device, non_blocking=True)
        ch4 = batch["ch4"].to(device, non_blocking=True)
        has_ch4 = batch["has_ch4"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"]

        if ch4_mode == "gate_random":
            has_ch4 = torch.bernoulli(torch.full_like(has_ch4, 0.5))
        elif ch4_mode == "gate_zero":
            has_ch4 = torch.zeros_like(has_ch4)
        elif ch4_mode == "co2_only":
            has_ch4 = torch.zeros_like(has_ch4)
            ch4 = torch.zeros_like(ch4)
        elif ch4_mode != "real":
            raise ValueError(f"unknown ch4_mode={ch4_mode!r}; must be one of {CH4_MODES}")

        out = model(co2, ch4, has_ch4)
        cls_logits = out["cls_logits"].float().cpu().numpy()
        all_logits.append(cls_logits)
        all_labels.append(labels.numpy())
        all_sample_ids.extend(batch["sample_id"])

        seg_pred = out["seg_logits"].argmax(dim=1)
        seg.update(seg_pred, mask)

        for k in vlm_keys:
            if k in out:
                vlm_buffers[k].append(out[k].float().cpu().numpy())
        if "clip_text_emb" in out and clip_text_proto is None:
            clip_text_proto = out["clip_text_emb"].float().cpu().numpy()

    cls_logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    vlm_feats: Dict[str, np.ndarray] = {}
    for k, v_list in vlm_buffers.items():
        if v_list:
            vlm_feats[k] = np.concatenate(v_list, axis=0)
    if clip_text_proto is not None:
        vlm_feats["clip_text_proto"] = clip_text_proto

    return cls_logits, labels, all_sample_ids, seg.compute(), vlm_feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", required=True, choices=["baseline", "vlm", "zoo"])
    parser.add_argument(
        "--model-name",
        default=None,
        help="zoo model name (used with --model-type zoo)",
    )
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--results-dir", default="/work/nvme/bgte/tislam6/ACID_Journal/results")
    parser.add_argument("--test-csv-override", default=None, help="optional alternate test CSV")
    parser.add_argument(
        "--ch4-mode",
        type=str,
        default="real",
        choices=list(CH4_MODES),
        help="R2-Q2 CH4-gate shortcut audit: real|gate_random|gate_zero|co2_only",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test dataloader
    from torch.utils.data import DataLoader

    data = cfg["data"]
    test_csv = args.test_csv_override or os.path.join(data["ext_dir"], data["test_csv"])
    test_ds = DualGasDataset(
        csv_path=test_csv,
        dataset_root=data["dataset_root"],
        img_size=data["img_size"],
        seed=cfg["seed"] + 10,
        merge_tube=bool(data.get("merge_tube", False)),
    )
    loader = DataLoader(
        test_ds,
        batch_size=data["batch_size"],
        shuffle=False,
        num_workers=data.get("num_workers", 4),
        pin_memory=True,
    )
    print(f"Test samples: {len(test_ds)}")

    # If the checkpoint recorded its own model name (zoo runs do), prefer it.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = args.model_name or state.get("model_name")
    model = build_model_from_type(args.model_type, cfg, model_name_override=model_name)
    model_state = state["model"] if "model" in state else state
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    print(f"Loaded {args.checkpoint} (model={model_name or args.model_type})")

    # Inference
    print(f"CH4 mode: {args.ch4_mode}")
    cls_logits, labels, sample_ids, seg_dict, vlm_feats = run_forward_pass(
        model, loader, device, cfg, ch4_mode=args.ch4_mode
    )

    # Classification metrics
    cls_dict = classification_metrics(cls_logits, labels, n_classes=3)

    # Efficiency metrics
    input_shapes = {"co2": (1, 256, 256), "ch4": (1, 256, 256), "has_ch4": (1,)}
    try:
        eff_dict = efficiency_metrics(model, input_shapes, device)
    except Exception as e:
        print(f"Efficiency metrics failed: {e}")
        eff_dict = {}

    # VLM-specific metrics (only populated if the model produced the
    # corresponding projection / CLIP embedding tensors in its forward dict)
    vlm_dict: dict = {}
    if (
        "cnn_proj_text" in vlm_feats
        and "cnn_proj_c5" in vlm_feats
        and "clip_img_emb" in vlm_feats
        and "clip_text_proto" in vlm_feats
    ):
        try:
            from journal_metrics import vlm_alignment_metrics

            vlm_dict = vlm_alignment_metrics(
                vlm_feats["cnn_proj_c5"],
                vlm_feats["clip_img_emb"],
                vlm_feats["cnn_proj_text"],
                vlm_feats["clip_text_proto"],
                labels,
            )
        except Exception as e:
            print(f"VLM alignment metrics failed: {e}")
            vlm_dict = {}

    all_metrics = {
        "exp_name": args.exp_name,
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "n_test_samples": int(len(labels)),
        "class_names": CLASS_NAMES,
        **{f"cls_{k}": v for k, v in cls_dict.items()},
        **{f"seg_{k}": v for k, v in seg_dict.items()},
        **{f"eff_{k}": v for k, v in eff_dict.items()},
        **{f"vlm_{k}": v for k, v in vlm_dict.items()},
    }

    out_dir = os.path.join(args.results_dir, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=float)

    np.savez(
        os.path.join(out_dir, "test_preds.npz"),
        cls_logits=cls_logits,
        labels=labels,
    )
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": labels,
            "pred": cls_logits.argmax(axis=1),
            **{f"logit_{i}": cls_logits[:, i] for i in range(cls_logits.shape[1])},
        }
    ).to_csv(os.path.join(out_dir, "test_per_sample.csv"), index=False)

    # Pretty print a summary
    print()
    print("=" * 60)
    print(f"RESULTS — {args.exp_name}")
    print("=" * 60)
    print(f"  acc               {cls_dict['acc']:.4f}")
    print(f"  top-2 acc         {cls_dict['top2_acc']:.4f}")
    print(f"  balanced acc      {cls_dict['bal_acc']:.4f}")
    print(f"  macro F1          {cls_dict['macro_f1']:.4f}")
    print(f"  MCC               {cls_dict['mcc']:.4f}")
    print(f"  Cohen κ           {cls_dict['kappa']:.4f}")
    print(f"  AUROC macro       {cls_dict['auroc_macro']:.4f}")
    print(f"  AUPRC macro       {cls_dict['auprc_macro']:.4f}")
    print(f"  ECE               {cls_dict['ece']:.4f}")
    print(f"  Brier             {cls_dict['brier']:.4f}")
    print(
        "  per-class F1      "
        + ", ".join(f"{n}={cls_dict['per_class_f1'][i]:.3f}" for i, n in enumerate(CLASS_NAMES))
    )
    print("  mean IoU (fg)     %.4f" % seg_dict["mean_iou_fg"])
    print("  mean Dice (fg)    %.4f" % seg_dict["mean_dice_fg"])
    print("  pixel acc         %.4f" % seg_dict["pixel_acc"])
    print("  freq-weighted IoU %.4f" % seg_dict["freq_weighted_iou"])
    print("  HD95              %.2f" % seg_dict["hd95"])
    print("  ASSD              %.2f" % seg_dict["assd"])
    print("  BF-score          %.4f" % seg_dict["bf_score"])
    if eff_dict:
        print("  params (M)        %.2f" % eff_dict["params_M"])
        print("  FLOPs (G)         %.2f" % eff_dict["flops_G"])
        print("  latency (ms/img)  %.2f" % eff_dict["latency_ms"])
        print("  throughput (i/s)  %.1f" % eff_dict["throughput_imgs_per_sec"])
        print("  peak mem (GB)     %.2f" % eff_dict["peak_mem_GB"])
    if vlm_dict:
        print()
        print("  -- VLM alignment metrics --")
        print("  zero-shot top-1   %.4f" % vlm_dict.get("zs_top1_acc", float("nan")))
        print("  zero-shot bal-acc %.4f" % vlm_dict.get("zs_bal_acc", float("nan")))
        print("  retrieval MRR     %.4f" % vlm_dict.get("retrieval_mrr", float("nan")))
        print("  retrieval R@1     %.4f" % vlm_dict.get("retrieval_r1", float("nan")))
        print("  CKA visual        %.4f" % vlm_dict.get("cka_visual", float("nan")))
        print("  CKA text          %.4f" % vlm_dict.get("cka_text", float("nan")))
        print("  proto margin      %.4f" % vlm_dict.get("zs_proto_margin", float("nan")))
    print()
    print(f"Wrote {out_dir}/test_metrics.json and test_preds.npz")


if __name__ == "__main__":
    main()
