"""
Train the VLMGuidedDualGasNet on the ACID_Journal extended_dataset_50x.

Usage:
    python scripts/train_vlm.py --config configs/vlm.yaml

Identical structure to train_baseline.py except it builds the
VLMGuidedDualGasNet (baseline CNN + frozen CLIP ViT-B/16) and includes the
alignment loss in logging.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

import torch
from torch.cuda.amp import GradScaler, autocast

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dual_gas_dataset import build_dataloaders                     # noqa: E402
from losses_metrics import CombinedLoss, confusion_matrix_str       # noqa: E402
from train_utils import (                                            # noqa: E402
    build_optimizer,
    load_yaml,
    save_checkpoint,
    set_seed,
    warmup_cosine_lr,
)
from vlm_model import VLMGuidedDualGasNet                            # noqa: E402

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


def train_one_epoch(
    model, loader, optimizer, scaler, loss_fn, device, epoch, cfg, logger
) -> Dict[str, float]:
    model.train()
    running = {"total": 0.0, "seg": 0.0, "cls": 0.0, "align": 0.0}
    n_batches = 0
    t0 = time.time()
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    log_every = cfg["train"]["log_every"]
    grad_clip = cfg["train"]["grad_clip"]

    # Only optimize parameters that require grad (CLIP is frozen).
    trainable = [p for p in model.parameters() if p.requires_grad]

    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
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
        n_batches += 1

        if step % log_every == 0:
            logger(
                f"  [ep {epoch:02d} step {step:04d}] "
                f"loss={total.item():.4f}  seg={losses['seg'].item():.4f}  "
                f"cls={losses['cls'].item():.4f}  align={losses['align'].item():.4f}"
            )

    dt = time.time() - t0
    return {
        "train_loss": running["total"] / max(1, n_batches),
        "train_seg_loss": running["seg"] / max(1, n_batches),
        "train_cls_loss": running["cls"] / max(1, n_batches),
        "train_align_loss": running["align"] / max(1, n_batches),
        "train_time_sec": dt,
    }


@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg) -> Dict[str, Any]:
    model.eval()
    use_amp = cfg["train"]["amp"] and device.type == "cuda"

    total_loss = 0.0
    total_align = 0.0
    n = 0
    all_cls_logits = []
    all_labels = []
    seg_classes = cfg["model"]["seg_classes"]
    inter_cls = torch.zeros(seg_classes)
    pred_area = torch.zeros(seg_classes)
    tgt_area = torch.zeros(seg_classes)

    for batch in loader:
        batch = _to_device(batch, device)
        with autocast(enabled=use_amp):
            outputs = model(batch["co2"], batch["ch4"], batch["has_ch4"])
            losses = loss_fn(outputs, batch)
        total_loss += losses["total"].item()
        total_align += losses["align"].item()
        n += 1

        all_cls_logits.append(outputs["cls_logits"].float().cpu())
        all_labels.append(batch["label"].cpu())

        seg_pred = outputs["seg_logits"].argmax(dim=1)
        mask = batch["mask"]
        for c in range(seg_classes):
            pc = (seg_pred == c)
            tc = (mask == c)
            inter_cls[c] += (pc & tc).sum().item()
            pred_area[c] += pc.sum().item()
            tgt_area[c] += tc.sum().item()

    all_cls_logits = torch.cat(all_cls_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support

    preds = all_cls_logits.argmax(dim=1).numpy()
    truth = all_labels.numpy()
    acc = float((preds == truth).mean())
    bal_acc = float(balanced_accuracy_score(truth, preds))
    macro_f1 = float(f1_score(truth, preds, average="macro", zero_division=0))
    p, r, f1, _ = precision_recall_fscore_support(
        truth, preds, labels=[0, 1, 2], zero_division=0
    )

    fg_iou = []
    fg_dice = []
    for c in range(1, seg_classes):
        union = pred_area[c].item() + tgt_area[c].item() - inter_cls[c].item()
        if union > 0:
            fg_iou.append(inter_cls[c].item() / union)
        if (pred_area[c] + tgt_area[c]).item() > 0:
            fg_dice.append(2 * inter_cls[c].item() / (pred_area[c].item() + tgt_area[c].item()))
    import statistics as _st

    mean_iou = float(_st.mean(fg_iou)) if fg_iou else float("nan")
    mean_dice = float(_st.mean(fg_dice)) if fg_dice else float("nan")

    return {
        "val_loss": total_loss / max(1, n),
        "val_align_loss": total_align / max(1, n),
        "val_acc": acc,
        "val_bal_acc": bal_acc,
        "val_macro_f1": macro_f1,
        "val_per_class_precision": [float(x) for x in p],
        "val_per_class_recall": [float(x) for x in r],
        "val_per_class_f1": [float(x) for x in f1],
        "val_mean_iou": mean_iou,
        "val_mean_dice": mean_dice,
        "_cls_logits": all_cls_logits,
        "_labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    ckpt_dir = cfg["train"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)

    wandb_run = None
    try:
        if cfg.get("wandb", {}).get("mode", "disabled") != "disabled":
            import wandb

            os.environ.setdefault("WANDB_MODE", cfg["wandb"]["mode"])
            wandb_run = wandb.init(
                project=cfg["wandb"]["project"],
                name=cfg["wandb"]["run_name"],
                config=cfg,
                dir=ckpt_dir,
            )
    except Exception as e:  # pragma: no cover
        log(f"wandb init failed ({e}); continuing stdout-only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    if device.type == "cuda":
        log(f"CUDA: {torch.cuda.get_device_name(0)}  mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    log("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(cfg)
    log(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    log("Building VLMGuidedDualGasNet...")
    model = VLMGuidedDualGasNet(
        num_classes=cfg["model"]["num_classes"],
        seg_classes=cfg["model"]["seg_classes"],
        pretrained_resnet=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
        attn_heads=cfg["model"]["attn_heads"],
        clip_model=cfg["model"]["clip_model"],
        clip_pretrained=cfg["model"]["clip_pretrained"],
        clip_dim=cfg["model"]["clip_dim"],
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log(f"Model params: {total_params:.1f}M total, {trainable_params:.1f}M trainable (CLIP frozen)")

    loss_fn = CombinedLoss(
        seg_classes=cfg["model"]["seg_classes"],
        cls_class_weights=cfg["loss"]["cls_class_weights"],
        seg_w=cfg["loss"]["seg_w"],
        cls_w=cfg["loss"]["cls_w"],
        align_w=cfg["loss"]["align_w"],
        dice_include_background=cfg["loss"]["dice_include_background"],
    ).to(device)

    optimizer = build_optimizer(
        [p for p in model.parameters() if p.requires_grad], cfg["optim"]
    )
    scaler = GradScaler(enabled=(cfg["train"]["amp"] and device.type == "cuda"))

    base_lr = float(cfg["optim"]["lr"])
    warmup = cfg["sched"]["warmup_epochs"]
    total = cfg["sched"]["total_epochs"]
    min_lr_factor = cfg["sched"]["min_lr_factor"]

    best_bal_acc = -1.0
    best_iou = -1.0
    epochs_no_improve = 0
    patience = cfg["train"]["patience"]

    log("=" * 60)
    log("VLM-GUIDED TRAINING START")
    log("=" * 60)

    for epoch in range(cfg["train"]["epochs"]):
        lr = warmup_cosine_lr(epoch, warmup, total, base_lr, min_lr_factor)
        _apply_lr(optimizer, lr)

        tr = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, epoch, cfg, log)
        val = validate(model, val_loader, loss_fn, device, cfg)

        log(
            f"[ep {epoch:02d}] lr={lr:.2e}  "
            f"train_loss={tr['train_loss']:.4f} "
            f"(seg {tr['train_seg_loss']:.3f}, cls {tr['train_cls_loss']:.3f}, align {tr['train_align_loss']:.3f})  "
            f"val_loss={val['val_loss']:.4f}  "
            f"val_acc={val['val_acc']:.4f}  "
            f"val_bal_acc={val['val_bal_acc']:.4f}  "
            f"val_macro_f1={val['val_macro_f1']:.4f}  "
            f"val_mean_iou={val['val_mean_iou']:.4f}  "
            f"val_align={val['val_align_loss']:.4f}  "
            f"time={tr['train_time_sec']:.1f}s"
        )
        log("  per-class F1 : " + ", ".join(f"{CLASS_NAMES[i]}={val['val_per_class_f1'][i]:.3f}" for i in range(3)))
        log("  confusion:\n" + confusion_matrix_str(val["_cls_logits"], val["_labels"], CLASS_NAMES))

        if wandb_run is not None:
            payload = {
                "epoch": epoch,
                "lr": lr,
                **{k: v for k, v in tr.items() if not k.startswith("_")},
                **{k: v for k, v in val.items() if not k.startswith("_") and not isinstance(v, list)},
            }
            for i in range(3):
                payload[f"val_f1_{CLASS_NAMES[i]}"] = val["val_per_class_f1"][i]
            wandb_run.log(payload, step=epoch)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "val": {k: v for k, v in val.items() if not k.startswith("_")},
            "cfg": cfg,
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
            log(f"Early stopping: no val_bal_acc improvement for {patience} epochs.")
            break

    log("Training complete.")
    log(f"best val_bal_acc = {best_bal_acc:.4f}")
    log(f"best val_mean_iou = {best_iou:.4f}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
