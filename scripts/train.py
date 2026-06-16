"""
Unified training script for the ACID_Journal baseline zoo, VLM ablations,
and CV folds. Reads a YAML config, builds a model via the factory in
``scripts/models/__init__.py``, and runs the same train/val loop used by
the original train_baseline.py / train_vlm.py.

Usage:
    python scripts/train.py --config configs/baselines/resnet50_dual.yaml --seed 42

The config's ``model.name`` field picks the model; every model in the zoo
conforms to the forward signature

    forward(co2, ch4, has_ch4) -> {"cls_logits", "seg_logits", ...}

so the loop is unchanged per model.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy
from typing import Any, Dict

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dual_gas_dataset import build_dataloaders
from losses_metrics import CombinedLoss, confusion_matrix_str
from models import build_model
from train_utils import (
    build_optimizer,
    load_yaml,
    save_checkpoint,
    set_seed,
    warmup_cosine_lr,
)

CLASS_NAMES = ["Healthy", "Transitional", "Acidotic"]


def _apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


CH4_MODES = ("real", "gate_random", "gate_zero", "co2_only")


def _apply_ch4_mode(batch: Dict[str, Any], mode: str) -> None:
    """R2-Q2 CH4-shortcut audit. Mutates ``batch["has_ch4"]`` (and ``ch4`` for
    ``co2_only``) in place so the model sees the audit-perturbed inputs.

    - ``real``        : no change. Headline behaviour.
    - ``gate_random`` : per-sample Bernoulli(0.5) gate. Tests whether the model
      learns any useful signal from the gate prior.
    - ``gate_zero``   : gate forced to 0 everywhere. CH4 frame still present in
      the slot but ignored by the gating; tests whether the model leans on
      gate=1 as a Healthy/Acidotic vs. Transitional shortcut.
    - ``co2_only``    : gate=0 AND CH4 slot zeroed. True single-gas baseline.
    """
    if mode == "real":
        return
    has_ch4 = batch["has_ch4"]
    if mode == "gate_random":
        rand = torch.bernoulli(torch.full_like(has_ch4, 0.5))
        batch["has_ch4"] = rand
    elif mode == "gate_zero":
        batch["has_ch4"] = torch.zeros_like(has_ch4)
    elif mode == "co2_only":
        batch["has_ch4"] = torch.zeros_like(has_ch4)
        batch["ch4"] = torch.zeros_like(batch["ch4"])
    else:
        raise ValueError(f"unknown ch4_mode={mode!r}; must be one of {CH4_MODES}")


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, epoch, cfg, log, ch4_mode="real"):
    model.train()
    running = {"total": 0.0, "seg": 0.0, "cls": 0.0, "align": 0.0, "align_dist": 0.0, "align_text": 0.0}
    n = 0
    t0 = time.time()
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    log_every = cfg["train"]["log_every"]
    grad_clip = cfg["train"]["grad_clip"]

    trainable = [p for p in model.parameters() if p.requires_grad]

    from torch.amp import autocast

    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)
        _apply_ch4_mode(batch, ch4_mode)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp):
            outputs = model(batch["co2"], batch["ch4"], batch["has_ch4"])
            losses = loss_fn(outputs, batch)
        total = losses["total"]
        if use_amp:
            scaler.scale(total).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()

        running["total"] += total.item()
        running["seg"] += losses["seg"].item()
        running["cls"] += losses["cls"].item()
        running["align"] += losses["align"].item()
        running["align_dist"] += losses.get("align_dist", torch.tensor(0.0)).item()
        running["align_text"] += losses.get("align_text", torch.tensor(0.0)).item()
        n += 1
        if step % log_every == 0:
            log(
                f"  [ep {epoch:02d} step {step:04d}] loss={total.item():.4f} "
                f"seg={losses['seg'].item():.4f} cls={losses['cls'].item():.4f} "
                f"align={losses['align'].item():.4f}"
            )

    return {
        "train_loss": running["total"] / max(1, n),
        "train_seg_loss": running["seg"] / max(1, n),
        "train_cls_loss": running["cls"] / max(1, n),
        "train_align_loss": running["align"] / max(1, n),
        "train_align_dist_loss": running["align_dist"] / max(1, n),
        "train_align_text_loss": running["align_text"] / max(1, n),
        "train_time_sec": time.time() - t0,
    }


@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg, ch4_mode="real"):
    model.eval()
    from torch.amp import autocast
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    total_loss = 0.0
    total_align = 0.0
    n = 0
    seg_classes = cfg["model"]["seg_classes"]
    inter_cls = torch.zeros(seg_classes)
    pred_area = torch.zeros(seg_classes)
    tgt_area = torch.zeros(seg_classes)
    all_logits = []
    all_labels = []
    for batch in loader:
        batch = _to_device(batch, device)
        _apply_ch4_mode(batch, ch4_mode)
        with autocast("cuda", enabled=use_amp):
            outputs = model(batch["co2"], batch["ch4"], batch["has_ch4"])
            losses = loss_fn(outputs, batch)
        total_loss += losses["total"].item()
        total_align += losses["align"].item()
        n += 1
        all_logits.append(outputs["cls_logits"].float().cpu())
        all_labels.append(batch["label"].cpu())
        seg_pred = outputs["seg_logits"].argmax(dim=1)
        for c in range(seg_classes):
            pc = seg_pred == c
            tc = batch["mask"] == c
            inter_cls[c] += (pc & tc).sum().item()
            pred_area[c] += pc.sum().item()
            tgt_area[c] += tc.sum().item()
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support

    preds = all_logits.argmax(dim=1).numpy()
    truth = all_labels.numpy()
    acc = float((preds == truth).mean())
    bal_acc = float(balanced_accuracy_score(truth, preds))
    macro_f1 = float(f1_score(truth, preds, average="macro", zero_division=0))
    p, r, f1, _ = precision_recall_fscore_support(
        truth, preds, labels=[0, 1, 2], zero_division=0
    )
    fg_iou, fg_dice = [], []
    for c in range(1, seg_classes):
        union = pred_area[c] + tgt_area[c] - inter_cls[c]
        if union > 0:
            fg_iou.append((inter_cls[c] / union).item())
        if (pred_area[c] + tgt_area[c]) > 0:
            fg_dice.append((2 * inter_cls[c] / (pred_area[c] + tgt_area[c])).item())
    import statistics as _st

    return {
        "val_loss": total_loss / max(1, n),
        "val_align_loss": total_align / max(1, n),
        "val_acc": acc,
        "val_bal_acc": bal_acc,
        "val_macro_f1": macro_f1,
        "val_per_class_f1": [float(x) for x in f1],
        "val_mean_iou": float(_st.mean(fg_iou)) if fg_iou else float("nan"),
        "val_mean_dice": float(_st.mean(fg_dice)) if fg_dice else float("nan"),
        "_cls_logits": all_logits,
        "_labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="override cfg.seed")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="override cfg.model.name (e.g. resnet50_dual)",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="override cfg.train.ckpt_dir (where checkpoints are written)",
    )
    parser.add_argument(
        "--exp-suffix",
        type=str,
        default=None,
        help="optional suffix appended to ckpt_dir/run_name (e.g. _seed42)",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default=None,
        help="override cfg.data.train_csv (e.g. cv_fold0/train_annotations.csv)",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=None,
        help="override cfg.data.val_csv",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="override cfg.data.test_csv",
    )
    parser.add_argument(
        "--align-w",
        type=float,
        default=None,
        help="override cfg.loss.align_w (VLM ablation)",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default=None,
        help="override cfg.model.clip_model (VLM ablation)",
    )
    parser.add_argument(
        "--align-dist-w",
        type=float,
        default=None,
        help="override cfg.loss.align_dist_w (VLM v2 multi-scale visual distill)",
    )
    parser.add_argument(
        "--align-text-w",
        type=float,
        default=None,
        help="override cfg.loss.align_text_w (VLM v2 text-prototype InfoNCE)",
    )
    parser.add_argument(
        "--label-fraction",
        type=float,
        default=None,
        help="override cfg.data.label_fraction (label-efficiency sweep)",
    )
    parser.add_argument(
        "--ch4-mode",
        type=str,
        default="real",
        choices=list(CH4_MODES),
        help="R2-Q2 CH4-gate shortcut audit: real|gate_random|gate_zero|co2_only",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.model_name is not None:
        cfg["model"]["name"] = args.model_name
    if args.ckpt_dir is not None:
        cfg["train"]["ckpt_dir"] = args.ckpt_dir
    if args.exp_suffix:
        cfg["train"]["ckpt_dir"] = cfg["train"]["ckpt_dir"] + args.exp_suffix
        if "wandb" in cfg:
            cfg["wandb"]["run_name"] = cfg["wandb"].get("run_name", "run") + args.exp_suffix
    if args.train_csv is not None:
        cfg["data"]["train_csv"] = args.train_csv
    if args.val_csv is not None:
        cfg["data"]["val_csv"] = args.val_csv
    if args.test_csv is not None:
        cfg["data"]["test_csv"] = args.test_csv
    if args.align_w is not None:
        cfg["loss"]["align_w"] = float(args.align_w)
    if args.align_dist_w is not None:
        cfg["loss"]["align_dist_w"] = float(args.align_dist_w)
    if args.align_text_w is not None:
        cfg["loss"]["align_text_w"] = float(args.align_text_w)
    if args.clip_model is not None:
        cfg["model"]["clip_model"] = args.clip_model
    if args.label_fraction is not None:
        cfg["data"]["label_fraction"] = float(args.label_fraction)

    set_seed(cfg["seed"])
    ckpt_dir = cfg["train"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    if device.type == "cuda":
        log(
            f"CUDA: {torch.cuda.get_device_name(0)}  "
            f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB"
        )

    log(f"Config: {args.config}")
    log(f"Seed:   {cfg['seed']}")
    log(f"Ckpt:   {ckpt_dir}")
    log(f"CH4 mode: {args.ch4_mode}")

    log("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(cfg)
    log(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    model_name = cfg["model"]["name"]
    log(f"Building model: {model_name}")
    model = build_model(model_name, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log(f"Model: {n_params:.1f}M total / {trainable:.1f}M trainable")

    loss_fn = CombinedLoss(
        seg_classes=cfg["model"]["seg_classes"],
        cls_class_weights=cfg["loss"]["cls_class_weights"],
        seg_w=cfg["loss"]["seg_w"],
        cls_w=cfg["loss"]["cls_w"],
        align_w=cfg["loss"].get("align_w", 0.0),
        align_dist_w=cfg["loss"].get("align_dist_w", 0.0),
        align_text_w=cfg["loss"].get("align_text_w", 0.0),
        text_temperature=cfg["loss"].get("text_temperature", 0.07),
        use_multiscale_dist=cfg["loss"].get("use_multiscale_dist", True),
        dice_include_background=cfg["loss"]["dice_include_background"],
    ).to(device)

    optimizer = build_optimizer(
        [p for p in model.parameters() if p.requires_grad], cfg["optim"]
    )
    from torch.amp import GradScaler
    scaler = GradScaler("cuda", enabled=(cfg["train"]["amp"] and device.type == "cuda"))

    base_lr = float(cfg["optim"]["lr"])
    warmup = cfg["sched"]["warmup_epochs"]
    total = cfg["sched"]["total_epochs"]
    min_lr_factor = cfg["sched"]["min_lr_factor"]

    best_bal_acc = -1.0
    best_iou = -1.0
    epochs_no_improve = 0
    patience = cfg["train"]["patience"]

    log("=" * 60)
    log(f"TRAIN START — {model_name}")
    log("=" * 60)

    for epoch in range(cfg["train"]["epochs"]):
        lr = warmup_cosine_lr(epoch, warmup, total, base_lr, min_lr_factor)
        _apply_lr(optimizer, lr)

        tr = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, epoch, cfg, log, ch4_mode=args.ch4_mode)
        val = validate(model, val_loader, loss_fn, device, cfg, ch4_mode=args.ch4_mode)
        log(
            f"[ep {epoch:02d}] lr={lr:.2e}  "
            f"train_loss={tr['train_loss']:.4f}  "
            f"val_loss={val['val_loss']:.4f}  "
            f"val_acc={val['val_acc']:.4f}  "
            f"val_bal_acc={val['val_bal_acc']:.4f}  "
            f"val_macro_f1={val['val_macro_f1']:.4f}  "
            f"val_mean_iou={val['val_mean_iou']:.4f}  "
            f"time={tr['train_time_sec']:.1f}s"
        )
        log("  per-class F1 : " + ", ".join(
            f"{CLASS_NAMES[i]}={val['val_per_class_f1'][i]:.3f}" for i in range(3)
        ))
        log("  confusion:\n" + confusion_matrix_str(val["_cls_logits"], val["_labels"], CLASS_NAMES))

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "val": {k: v for k, v in val.items() if not k.startswith("_")},
            "cfg": cfg,
            "model_name": model_name,
            "ch4_mode": args.ch4_mode,
        }
        save_checkpoint(os.path.join(ckpt_dir, "last.pth"), state)
        if val["val_bal_acc"] > best_bal_acc:
            best_bal_acc = val["val_bal_acc"]
            save_checkpoint(os.path.join(ckpt_dir, "best_acc.pth"), state)
            epochs_no_improve = 0
            log(f"  ** new best val_bal_acc = {best_bal_acc:.4f}")
        else:
            epochs_no_improve += 1
        if val["val_mean_iou"] > best_iou:
            best_iou = val["val_mean_iou"]
            save_checkpoint(os.path.join(ckpt_dir, "best_iou.pth"), state)
            log(f"  ** new best val_mean_iou = {best_iou:.4f}")
        if (epoch + 1) % 10 == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pth"), state)
        if epochs_no_improve >= patience:
            log(f"Early stopping: patience {patience} exceeded.")
            break

    log("DONE")
    log(f"best val_bal_acc = {best_bal_acc:.4f}")
    log(f"best val_mean_iou = {best_iou:.4f}")


if __name__ == "__main__":
    main()
