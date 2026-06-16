"""
Journal-grade metric suite for the ACID_Journal project.

Keeps the training-loop metrics (in losses_metrics.py) minimal and fast; this
module is test-time only and computes the full set the Pattern Recognition
paper needs:

Classification
  - top-1, top-2 accuracy
  - balanced accuracy, macro F1, macro precision, macro recall
  - Matthews correlation coefficient (MCC)
  - Cohen's kappa
  - AUROC (one-vs-rest, macro and per-class)
  - AUPRC (one-vs-rest, macro and per-class)
  - Expected Calibration Error (ECE, 15 bins)
  - Brier score (multiclass)
  - per-class precision/recall/F1
  - full confusion matrix

Segmentation
  - mean IoU, per-class IoU
  - mean Dice, per-class Dice
  - pixel accuracy (global + macro mean-class)
  - frequency-weighted IoU
  - Hausdorff Distance 95 (HD95) — tolerant to NaN when a class is absent
  - Average Symmetric Surface Distance (ASSD)
  - Boundary F1 (BF-score) at 2-pixel tolerance

Efficiency (model-level, called once per checkpoint)
  - params (M)
  - FLOPs (G) via fvcore
  - latency (ms/image, batch=1) — warm-up + median of 100 calls
  - throughput (imgs/sec, batch=32)
  - peak GPU memory (GB) during one batch=32 forward

Everything returns plain Python floats in a flat dict so `eval_metrics.py`
can dump it as JSON.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
)

# -------------------------------------------------------------- classification


def top_k_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float(np.mean([lbl in row for row, lbl in zip(topk, labels)]))


# ------------------- VLM-specific metrics -------------------


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def vlm_zero_shot_metrics(
    cnn_text_feat: np.ndarray,    # [N, D]
    clip_text_proto: np.ndarray,  # [C, D]
    labels: np.ndarray,           # [N]
) -> Dict[str, Any]:
    """Zero-shot classification via cosine similarity with frozen text prototypes.

    Returns top-1 accuracy, balanced accuracy, and per-class confusion for the
    zero-shot classifier that uses ``argmax(cnn_feat @ text_proto.T)`` — no
    trained classification head is used, so this measures the quality of the
    cross-modal alignment directly.
    """
    from sklearn.metrics import balanced_accuracy_score

    cnn = _l2_normalize(cnn_text_feat.astype(np.float32))
    proto = _l2_normalize(clip_text_proto.astype(np.float32))
    sims = cnn @ proto.T                        # [N, C]
    preds = sims.argmax(axis=1)
    acc = float((preds == labels).mean())
    bal = float(balanced_accuracy_score(labels, preds))

    # Prototype margin: mean (top1 sim - runner-up sim) per correct prediction
    sorted_sims = -np.sort(-sims, axis=1)
    margin = float((sorted_sims[:, 0] - sorted_sims[:, 1]).mean())
    return {
        "zs_top1_acc": acc,
        "zs_bal_acc": bal,
        "zs_proto_margin": margin,
    }


def vlm_retrieval_recall(
    cnn_text_feat: np.ndarray,    # [N, D]
    clip_text_proto: np.ndarray,  # [C, D]
    labels: np.ndarray,           # [N]
) -> Dict[str, Any]:
    """Image→text retrieval Recall@K.  With only C=3 classes, R@1 is equivalent
    to zero-shot top-1 and R@3 is trivially 1.  We also report mean
    reciprocal rank for a more granular signal.
    """
    cnn = _l2_normalize(cnn_text_feat.astype(np.float32))
    proto = _l2_normalize(clip_text_proto.astype(np.float32))
    sims = cnn @ proto.T                         # [N, C]
    ranked = np.argsort(-sims, axis=1)           # [N, C]
    ranks = np.where(ranked == labels[:, None])[1] + 1  # [N], 1-indexed
    mrr = float((1.0 / ranks).mean())
    r1 = float((ranks <= 1).mean())
    r2 = float((ranks <= 2).mean())
    return {"retrieval_mrr": mrr, "retrieval_r1": r1, "retrieval_r2": r2}


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment (linear kernel) between two feature matrices.

    Both arrays must have the same number of rows (samples); columns can
    differ.  Returns a value in [0, 1] measuring representational similarity
    in a rotation-invariant way.  CKA = 1 iff one is an invertible linear
    transform of the other.

    Reference: Kornblith et al., "Similarity of neural network
    representations revisited", ICML 2019.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # Linear HSIC = ||X^T Y||_F^2
    xty = X.T @ Y
    hsic_xy = (xty * xty).sum()
    xtx = X.T @ X
    yty = Y.T @ Y
    hsic_xx = (xtx * xtx).sum()
    hsic_yy = (yty * yty).sum()
    denom = np.sqrt(max(hsic_xx * hsic_yy, 1e-30))
    return float(hsic_xy / denom)


def vlm_alignment_metrics(
    cnn_proj_visual: np.ndarray,  # [N, D] CNN c5 projection (or c4)
    clip_img_emb: np.ndarray,     # [N, D] frozen CLIP image embedding
    cnn_proj_text: np.ndarray,    # [N, D] CNN classifier projection into text space
    clip_text_proto: np.ndarray,  # [C, D]
    labels: np.ndarray,           # [N]
) -> Dict[str, Any]:
    """Combine all VLM-specific measurements into one dict.

    Includes:
    - cosine similarity mean of CNN projection vs CLIP visual
    - CKA between CNN visual projection and CLIP visual (rotation-invariant)
    - CKA between CNN text projection and per-sample text prototype
    - zero-shot acc / balanced acc / prototype margin from text alignment
    - retrieval R@1, R@2, MRR
    """
    vis_cos = float(
        (_l2_normalize(cnn_proj_visual) * _l2_normalize(clip_img_emb)).sum(axis=-1).mean()
    )
    cka_visual = linear_cka(cnn_proj_visual, clip_img_emb)
    # Per-sample text target = text prototype of the ground-truth class
    text_target = clip_text_proto[labels]  # [N, D]
    cka_text = linear_cka(cnn_proj_text, text_target)

    out: Dict[str, Any] = {
        "cnn_clip_visual_cos": vis_cos,
        "cka_visual": cka_visual,
        "cka_text": cka_text,
    }
    out.update(vlm_zero_shot_metrics(cnn_proj_text, clip_text_proto, labels))
    out.update(vlm_retrieval_recall(cnn_proj_text, clip_text_proto, labels))
    return out


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Multi-class ECE using max probability as confidence."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if i == 0:
            in_bin = in_bin | (confidences == lo)
        count = in_bin.sum()
        if count > 0:
            acc = correct[in_bin].mean()
            conf = confidences[in_bin].mean()
            ece += (count / n) * abs(acc - conf)
    return float(ece)


def multiclass_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score: mean squared error between one-hot and probs."""
    n, c = probs.shape
    oh = np.zeros_like(probs)
    oh[np.arange(n), labels] = 1.0
    return float(((probs - oh) ** 2).sum(axis=1).mean())


def classification_metrics(
    cls_logits: np.ndarray, labels: np.ndarray, n_classes: int = 3
) -> Dict[str, Any]:
    """All classification metrics for one (seed, split) pair."""
    preds = cls_logits.argmax(axis=1)

    # Probabilities via softmax (numerically stable).
    shifted = cls_logits - cls_logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / exps.sum(axis=1, keepdims=True)

    acc = float((preds == labels).mean())
    top2 = top_k_accuracy(cls_logits, labels, k=2)
    bal_acc = float(balanced_accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(labels, preds))
    kappa = float(cohen_kappa_score(labels, preds))

    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(n_classes)), zero_division=0
    )
    macro_prec = float(p.mean())
    macro_rec = float(r.mean())

    # AUROC / AUPRC — only compute if every class present in labels.
    present = set(labels.tolist())
    if present.issuperset(set(range(n_classes))):
        try:
            auroc_macro = float(
                roc_auc_score(
                    labels, probs, multi_class="ovr", average="macro",
                    labels=list(range(n_classes)),
                )
            )
        except Exception:
            auroc_macro = float("nan")
        auprc_per = []
        auroc_per = []
        for c in range(n_classes):
            binary = (labels == c).astype(np.int32)
            try:
                auprc_per.append(float(average_precision_score(binary, probs[:, c])))
            except Exception:
                auprc_per.append(float("nan"))
            try:
                auroc_per.append(float(roc_auc_score(binary, probs[:, c])))
            except Exception:
                auroc_per.append(float("nan"))
        auprc_macro = float(np.nanmean(auprc_per))
    else:
        auroc_macro = float("nan")
        auprc_macro = float("nan")
        auprc_per = [float("nan")] * n_classes
        auroc_per = [float("nan")] * n_classes

    ece = expected_calibration_error(probs, labels, n_bins=15)
    brier = multiclass_brier(probs, labels)

    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))

    return {
        "acc": acc,
        "top2_acc": top2,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "mcc": mcc,
        "kappa": kappa,
        "auroc_macro": auroc_macro,
        "auprc_macro": auprc_macro,
        "auroc_per_class": auroc_per,
        "auprc_per_class": auprc_per,
        "per_class_precision": [float(x) for x in p],
        "per_class_recall": [float(x) for x in r],
        "per_class_f1": [float(x) for x in f1],
        "ece": ece,
        "brier": brier,
        "confusion_matrix": cm.tolist(),
    }


# -------------------------------------------------------------- segmentation


class SegmentationAccumulator:
    """Online accumulator for pixel-wise + boundary segmentation metrics.

    Intersection/union counts are accumulated incrementally to avoid keeping
    entire prediction volumes in memory. Boundary metrics (HD95, ASSD, BF)
    are computed per batch and averaged at the end to keep memory bounded.
    """

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.inter = np.zeros(num_classes, dtype=np.float64)
        self.pred_area = np.zeros(num_classes, dtype=np.float64)
        self.gt_area = np.zeros(num_classes, dtype=np.float64)
        self.total_px = 0
        self.correct_px = 0
        self._hd95: List[float] = []
        self._assd: List[float] = []
        self._bf: List[float] = []

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred, target: [B, H, W] LongTensors on any device."""
        pred = pred.detach().cpu()
        target = target.detach().cpu()
        pred_np = pred.numpy().astype(np.int64)
        tgt_np = target.numpy().astype(np.int64)

        self.total_px += pred_np.size
        self.correct_px += int((pred_np == tgt_np).sum())

        for c in range(self.num_classes):
            pc = pred_np == c
            tc = tgt_np == c
            self.inter[c] += float((pc & tc).sum())
            self.pred_area[c] += float(pc.sum())
            self.gt_area[c] += float(tc.sum())

        # Boundary metrics: one value per sample, averaged at end.
        try:
            from scipy.ndimage import distance_transform_edt
        except Exception:
            return

        for b in range(pred_np.shape[0]):
            p = pred_np[b]
            t = tgt_np[b]
            for c in range(1, self.num_classes):  # foreground only
                pm = (p == c)
                tm = (t == c)
                if pm.sum() == 0 or tm.sum() == 0:
                    continue
                dt_p = distance_transform_edt(~pm)
                dt_t = distance_transform_edt(~tm)
                d_p_to_t = dt_t[pm]
                d_t_to_p = dt_p[tm]
                if d_p_to_t.size == 0 or d_t_to_p.size == 0:
                    continue
                hd95 = float(
                    np.percentile(np.concatenate([d_p_to_t, d_t_to_p]), 95)
                )
                assd = float((d_p_to_t.mean() + d_t_to_p.mean()) / 2)
                self._hd95.append(hd95)
                self._assd.append(assd)

                # BF-score at 2px tolerance
                bp = _boundary_mask(pm)
                bt = _boundary_mask(tm)
                if bp.sum() == 0 or bt.sum() == 0:
                    continue
                dt_bp = distance_transform_edt(~bp)
                dt_bt = distance_transform_edt(~bt)
                tp_p = (dt_bt[bp] <= 2).sum()
                tp_t = (dt_bp[bt] <= 2).sum()
                prec = tp_p / max(1, bp.sum())
                rec = tp_t / max(1, bt.sum())
                if (prec + rec) > 0:
                    self._bf.append(2 * prec * rec / (prec + rec))

    def compute(self) -> Dict[str, Any]:
        num_classes = self.num_classes
        iou_per = []
        dice_per = []
        for c in range(num_classes):
            union = self.pred_area[c] + self.gt_area[c] - self.inter[c]
            iou_per.append(self.inter[c] / union if union > 0 else float("nan"))
            denom = self.pred_area[c] + self.gt_area[c]
            dice_per.append(
                2 * self.inter[c] / denom if denom > 0 else float("nan")
            )

        fg = list(range(1, num_classes))
        mean_iou_fg = float(np.nanmean([iou_per[c] for c in fg]))
        mean_dice_fg = float(np.nanmean([dice_per[c] for c in fg]))

        # Pixel accuracy
        pix_acc = float(self.correct_px / max(1, self.total_px))
        # Mean pixel accuracy (macro over classes where gt_area > 0)
        per_class_px_acc = []
        for c in range(num_classes):
            if self.gt_area[c] > 0:
                per_class_px_acc.append(float(self.inter[c] / self.gt_area[c]))
        mean_pix_acc = float(np.mean(per_class_px_acc)) if per_class_px_acc else float("nan")

        # Frequency-weighted IoU
        freq = self.gt_area / max(1.0, float(self.gt_area.sum()))
        fwiou = float(np.nansum(freq * np.array(iou_per)))

        hd95_mean = float(np.mean(self._hd95)) if self._hd95 else float("nan")
        assd_mean = float(np.mean(self._assd)) if self._assd else float("nan")
        bf_mean = float(np.mean(self._bf)) if self._bf else float("nan")

        return {
            "per_class_iou": [float(x) if not np.isnan(x) else float("nan") for x in iou_per],
            "per_class_dice": [float(x) if not np.isnan(x) else float("nan") for x in dice_per],
            "mean_iou_fg": mean_iou_fg,
            "mean_dice_fg": mean_dice_fg,
            "pixel_acc": pix_acc,
            "mean_pixel_acc": mean_pix_acc,
            "freq_weighted_iou": fwiou,
            "hd95": hd95_mean,
            "assd": assd_mean,
            "bf_score": bf_mean,
        }


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Compute a 1-pixel boundary of a binary mask via erosion difference."""
    from scipy.ndimage import binary_erosion

    return mask & ~binary_erosion(mask, iterations=1)


# -------------------------------------------------------------- efficiency


@torch.no_grad()
def efficiency_metrics(
    model: torch.nn.Module,
    input_shapes: Dict[str, tuple],
    device: torch.device,
    n_warmup: int = 20,
    n_iters: int = 100,
) -> Dict[str, Any]:
    """Params, FLOPs, latency (bs=1), throughput (bs=32), peak memory (bs=32).

    ``input_shapes``: dict of tensor_name → shape-excluding-batch, e.g.
      {"co2": (1, 256, 256), "ch4": (1, 256, 256), "has_ch4": (1,)}
    """
    was_training = model.training
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6

    # FLOPs via fvcore
    flops_g = float("nan")
    try:
        from fvcore.nn import FlopCountAnalysis

        dummy_1 = {
            k: torch.zeros((1, *shape), device=device) for k, shape in input_shapes.items()
        }
        # Some models take positional args; our models take (co2, ch4, has_ch4).
        fca = FlopCountAnalysis(model, tuple(dummy_1[k] for k in input_shapes))
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        flops_g = float(fca.total() / 1e9)
    except Exception as e:
        print(f"[efficiency] FLOPs count skipped: {e}")

    # Latency at batch=1
    dummy_1 = {
        k: torch.zeros((1, *shape), device=device) for k, shape in input_shapes.items()
    }
    for _ in range(n_warmup):
        _ = model(*[dummy_1[k] for k in input_shapes])
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        _ = model(*[dummy_1[k] for k in input_shapes])
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / n_iters * 1000.0

    # Throughput + peak memory at batch=32
    dummy_32 = {
        k: torch.zeros((32, *shape), device=device) for k, shape in input_shapes.items()
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        _ = model(*[dummy_32[k] for k in input_shapes])
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        _ = model(*[dummy_32[k] for k in input_shapes])
    if device.type == "cuda":
        torch.cuda.synchronize()
    throughput = (20 * 32) / (time.time() - t0)
    peak_mem_gb = float("nan")
    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    if was_training:
        model.train()

    return {
        "params_M": float(params),
        "flops_G": float(flops_g),
        "latency_ms": float(latency_ms),
        "throughput_imgs_per_sec": float(throughput),
        "peak_mem_GB": float(peak_mem_gb),
    }
