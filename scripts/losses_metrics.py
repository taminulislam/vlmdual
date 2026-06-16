"""
Losses and metrics for the ACID_Journal baseline.

- MultiClassDiceLoss: mean soft-Dice over foreground classes (tube, gas).
- SegLoss: Dice + pixel CrossEntropy.
- ClsLoss: weighted CrossEntropy (class_id order 0=Healthy,1=Transitional,2=Acidotic).
- CombinedLoss: sum of SegLoss and ClsLoss with per-term weights.
- compute_metrics: classification accuracy / balanced accuracy / per-class F1
  + segmentation mean IoU / per-class Dice.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes: int, include_background: bool = False, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)                              # [B,C,H,W]
        target_oh = F.one_hot(target, num_classes=self.num_classes)   # [B,H,W,C]
        target_oh = target_oh.permute(0, 3, 1, 2).float()             # [B,C,H,W]

        start = 0 if self.include_background else 1
        probs = probs[:, start:]
        target_oh = target_oh[:, start:]

        dims = (0, 2, 3)
        intersect = (probs * target_oh).sum(dims)
        card = probs.sum(dims) + target_oh.sum(dims)
        dice = (2.0 * intersect + self.eps) / (card + self.eps)       # [C-start]
        return 1.0 - dice.mean()


class SegLoss(nn.Module):
    def __init__(self, num_classes: int, include_background: bool = False):
        super().__init__()
        self.dice = MultiClassDiceLoss(num_classes, include_background=include_background)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.dice(logits, target) + 0.5 * self.ce(logits, target)


class ClsLoss(nn.Module):
    def __init__(self, class_weights: List[float] | None = None):
        super().__init__()
        weight = None
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def to(self, *args, **kwargs):  # type: ignore[override]
        self.ce = self.ce.to(*args, **kwargs)
        if self.ce.weight is not None:
            self.ce.weight.data = self.ce.weight.data.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target)


class AlignmentLoss(nn.Module):
    """1 - mean cosine similarity between cnn_proj and clip_emb."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, cnn_proj: torch.Tensor, clip_emb: torch.Tensor) -> torch.Tensor:
        a = F.normalize(cnn_proj, p=2, dim=-1, eps=self.eps)
        b = F.normalize(clip_emb, p=2, dim=-1, eps=self.eps)
        cos = (a * b).sum(dim=-1)  # [B]
        return 1.0 - cos.mean()


class CombinedLoss(nn.Module):
    """Multi-task loss supporting (a) VLM v1 single-scale cosine alignment and
    (b) VLM v2 multi-scale visual distillation + cross-modal InfoNCE text
    alignment. The correct branch is auto-detected from the model output dict:

    - v1 outputs contain ``cnn_proj`` and ``clip_emb`` (single alignment)
    - v2 outputs contain ``cnn_proj_c5`` / ``cnn_proj_c4`` / ``cnn_proj_text``
      and ``clip_img_emb`` / ``clip_text_emb``
    """

    def __init__(
        self,
        seg_classes: int,
        cls_class_weights: List[float] | None = None,
        seg_w: float = 1.0,
        cls_w: float = 1.0,
        align_w: float = 0.0,          # v1 cosine alignment weight
        align_dist_w: float = 0.0,     # v2 multi-scale visual distillation weight
        align_text_w: float = 0.0,     # v2 cross-modal text InfoNCE weight
        text_temperature: float = 0.07,
        use_multiscale_dist: bool = True,  # if True, average c4 and c5 distill
        dice_include_background: bool = False,
    ):
        super().__init__()
        self.seg_loss = SegLoss(seg_classes, include_background=dice_include_background)
        self.cls_loss = ClsLoss(cls_class_weights)
        self.align_loss = AlignmentLoss()
        self.seg_w = seg_w
        self.cls_w = cls_w
        self.align_w = align_w
        self.align_dist_w = align_dist_w
        self.align_text_w = align_text_w
        self.text_temperature = text_temperature
        self.use_multiscale_dist = use_multiscale_dist

    def _distill_loss(
        self,
        proj_c5: torch.Tensor,
        proj_c4: torch.Tensor | None,
        clip_img: torch.Tensor,
    ) -> torch.Tensor:
        """1 - cos_sim averaged over one or two scales."""
        l5 = self.align_loss(proj_c5, clip_img)
        if not self.use_multiscale_dist or proj_c4 is None:
            return l5
        l4 = self.align_loss(proj_c4, clip_img)
        return 0.5 * (l4 + l5)

    def _text_infonce_loss(
        self,
        z_text: torch.Tensor,     # [B, D] cnn projection into text space
        text_emb: torch.Tensor,   # [C, D] frozen class prototypes (normalized)
        labels: torch.Tensor,     # [B]
    ) -> torch.Tensor:
        z = F.normalize(z_text, p=2, dim=-1)
        logits = z @ text_emb.T / self.text_temperature  # [B, C]
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        seg = self.seg_loss(outputs["seg_logits"], batch["mask"])
        cls = self.cls_loss(outputs["cls_logits"], batch["label"])
        total = self.seg_w * seg + self.cls_w * cls

        # v1 alignment (kept for backwards compatibility)
        align_val = torch.tensor(0.0, device=seg.device)
        if self.align_w > 0 and "cnn_proj" in outputs and "clip_emb" in outputs:
            align_val = self.align_loss(outputs["cnn_proj"], outputs["clip_emb"])
            total = total + self.align_w * align_val

        # v2 multi-scale visual distillation
        dist_val = torch.tensor(0.0, device=seg.device)
        if self.align_dist_w > 0 and "cnn_proj_c5" in outputs and "clip_img_emb" in outputs:
            dist_val = self._distill_loss(
                outputs["cnn_proj_c5"],
                outputs.get("cnn_proj_c4"),
                outputs["clip_img_emb"],
            )
            total = total + self.align_dist_w * dist_val

        # v2 cross-modal text InfoNCE
        text_val = torch.tensor(0.0, device=seg.device)
        if (
            self.align_text_w > 0
            and "cnn_proj_text" in outputs
            and "clip_text_emb" in outputs
        ):
            text_val = self._text_infonce_loss(
                outputs["cnn_proj_text"],
                outputs["clip_text_emb"],
                batch["label"],
            )
            total = total + self.align_text_w * text_val

        return {
            "total": total,
            "seg": seg.detach(),
            "cls": cls.detach(),
            "align": align_val.detach(),
            "align_dist": dist_val.detach(),
            "align_text": text_val.detach(),
        }


# ---------- metrics ----------


@torch.no_grad()
def compute_metrics(
    cls_logits: torch.Tensor,
    labels: torch.Tensor,
    seg_logits: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int = 3,
    seg_classes: int = 3,
) -> Dict[str, float]:
    preds = cls_logits.argmax(dim=1).cpu().numpy()
    truth = labels.cpu().numpy()
    acc = float((preds == truth).mean())
    bal_acc = float(balanced_accuracy_score(truth, preds))
    macro_f1 = float(f1_score(truth, preds, average="macro", zero_division=0))

    per_cls_prec, per_cls_rec, per_cls_f1, _ = precision_recall_fscore_support(
        truth, preds, labels=list(range(num_classes)), zero_division=0
    )

    # Segmentation IoU / Dice over foreground classes (1=tube, 2=gas).
    seg_pred = seg_logits.argmax(dim=1)
    ious: List[float] = []
    dices: List[float] = []
    for c in range(1, seg_classes):  # skip background
        pred_c = (seg_pred == c)
        tgt_c = (masks == c)
        inter = (pred_c & tgt_c).sum().item()
        union = (pred_c | tgt_c).sum().item()
        pred_area = pred_c.sum().item()
        tgt_area = tgt_c.sum().item()
        iou = inter / (union + 1e-6) if union > 0 else float("nan")
        dice = (2 * inter) / (pred_area + tgt_area + 1e-6) if (pred_area + tgt_area) > 0 else float("nan")
        ious.append(iou)
        dices.append(dice)
    mean_iou = float(np.nanmean(ious)) if ious else float("nan")
    mean_dice = float(np.nanmean(dices)) if dices else float("nan")

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "per_cls_prec": [float(x) for x in per_cls_prec],
        "per_cls_rec": [float(x) for x in per_cls_rec],
        "per_cls_f1": [float(x) for x in per_cls_f1],
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
    }


@torch.no_grad()
def confusion_matrix_str(
    cls_logits: torch.Tensor, labels: torch.Tensor, class_names: List[str]
) -> str:
    preds = cls_logits.argmax(dim=1).cpu().numpy()
    truth = labels.cpu().numpy()
    cm = confusion_matrix(truth, preds, labels=list(range(len(class_names))))
    header = "pred→  " + "  ".join(f"{n[:5]:>5s}" for n in class_names)
    lines = [header]
    for i, n in enumerate(class_names):
        row = "  ".join(f"{cm[i, j]:>5d}" for j in range(len(class_names)))
        lines.append(f"{n[:5]:>5s}  {row}")
    return "\n".join(lines)
