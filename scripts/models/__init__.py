"""
Model factory for the ACID_Journal baseline zoo.

Usage:
    from models import build_model
    model = build_model("resnet50_dual", cfg)

The model returned is a ``nn.Module`` whose forward signature is
    forward(co2, ch4, has_ch4) -> {"cls_logits": ..., "seg_logits": ...}

so every model in the zoo is a drop-in replacement in the unified training
loop. Single-gas models silently ignore ``ch4`` / ``has_ch4``.

Model names:

    Ours
        ours_baseline        DualStreamBaselineNet (cross-attn + gas-presence gate)
        ours_vlm             VLMGuidedDualGasNet (+ frozen CLIP alignment)

    Baselines (B1..B12)
        resnet50_single      B1  single-gas CO2 ResNet-50 + U-Net decoder
        resnet50_dual        B2  dual-stream ResNet-50 concat at c5
        resnet101_dual       B3
        effnet_b3_dual       B4
        convnext_base_dual   B5
        vit_b16_dual         B6
        swin_base_dual       B7
        unet_r50             B8  segmentation_models_pytorch U-Net
        deeplabv3p_r50       B9
        pspnet_r50           B10
        segformer_b2         B11
        clip_linear_probe    B12
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn


def build_model(name: str, cfg: Dict[str, Any]) -> nn.Module:
    m = cfg.get("model", {})
    num_classes = m.get("num_classes", 3)
    seg_classes = m.get("seg_classes", 3)
    pretrained = m.get("pretrained", True)
    dropout = m.get("dropout", 0.3)
    attn_heads = m.get("attn_heads", 8)

    if name == "ours_baseline":
        from baseline_model import DualStreamBaselineNet

        return DualStreamBaselineNet(
            num_classes=num_classes,
            seg_classes=seg_classes,
            pretrained=pretrained,
            dropout=dropout,
            attn_heads=attn_heads,
        )

    if name == "ours_vlm":
        from vlm_model import VLMGuidedDualGasNet

        return VLMGuidedDualGasNet(
            num_classes=num_classes,
            seg_classes=seg_classes,
            pretrained_resnet=pretrained,
            dropout=dropout,
            attn_heads=attn_heads,
            clip_model=m.get("clip_model", "ViT-B-16"),
            clip_pretrained=m.get("clip_pretrained", "openai"),
            clip_dim=m.get("clip_dim", 512),
        )

    if name == "ours_vlm_v2":
        from vlm_model_v2 import VLMGuidedDualGasNetV2

        return VLMGuidedDualGasNetV2(
            num_classes=num_classes,
            seg_classes=seg_classes,
            pretrained_resnet=pretrained,
            dropout=dropout,
            attn_heads=attn_heads,
            clip_model=m.get("clip_model", "ViT-B-16"),
            clip_pretrained=m.get("clip_pretrained", "openai"),
            cls_feat_dim=m.get("cls_feat_dim", 512),
        )

    # Extra VLM baselines (CLIP zero-shot, CLIP fine-tuned, DINOv2, CoOp)
    if name == "clip_zero_shot":
        from .vlm_baselines import CLIPZeroShot

        return CLIPZeroShot(
            num_classes=num_classes,
            seg_classes=seg_classes,
            clip_model=m.get("clip_model", "ViT-B-16"),
            clip_pretrained=m.get("clip_pretrained", "openai"),
        )

    if name == "clip_fine_tuned":
        from .vlm_baselines import CLIPFullFineTune

        return CLIPFullFineTune(
            num_classes=num_classes,
            seg_classes=seg_classes,
            clip_model=m.get("clip_model", "ViT-B-16"),
            clip_pretrained=m.get("clip_pretrained", "openai"),
            dropout=dropout,
        )

    if name == "dinov2_linear_probe":
        from .vlm_baselines import DINOv2LinearProbe

        return DINOv2LinearProbe(
            num_classes=num_classes,
            seg_classes=seg_classes,
            dropout=dropout,
        )

    if name == "coop":
        from .vlm_baselines import CoOp

        return CoOp(
            num_classes=num_classes,
            seg_classes=seg_classes,
            n_ctx=m.get("coop_n_ctx", 4),
            clip_model=m.get("clip_model", "ViT-B-16"),
            clip_pretrained=m.get("clip_pretrained", "openai"),
            dropout=dropout,
        )

    # Timm-backed dual-stream or single-stream baselines (B1..B7)
    if name in (
        "resnet50_single",
        "resnet50_dual",
        "resnet101_dual",
        "effnet_b3_dual",
        "convnext_base_dual",
        "vit_b16_dual",
        "swin_base_dual",
    ):
        from .generic_dualstream import GenericDualStream

        backbone_for = {
            "resnet50_single": ("resnet50", False),
            "resnet50_dual":   ("resnet50", True),
            "resnet101_dual":  ("resnet101", True),
            "effnet_b3_dual":  ("tf_efficientnet_b3.ns_jft_in1k", True),
            "convnext_base_dual": ("convnext_base.fb_in22k_ft_in1k", True),
            "vit_b16_dual":    ("vit_base_patch16_224.augreg_in21k_ft_in1k", True),
            "swin_base_dual":  ("swinv2_base_window8_256", True),
        }
        backbone, dual = backbone_for[name]
        return GenericDualStream(
            backbone=backbone,
            dual_stream=dual,
            pretrained=pretrained,
            num_classes=num_classes,
            seg_classes=seg_classes,
            dropout=dropout,
            img_size=cfg.get("data", {}).get("img_size", 256),
        )

    # Segmentation-models-pytorch backbones (B8..B10)
    smp_models = {
        "unet_r50":         ("Unet", "resnet50"),
        "deeplabv3p_r50":   ("DeepLabV3Plus", "resnet50"),
        "pspnet_r50":       ("PSPNet", "resnet50"),
        "unet_hrnet_w48":   ("Unet", "tu-hrnet_w48"),
        "unet_swinv2_b":    ("Unet", "tu-swinv2_base_window8_256"),
        "unet_convnextv2_b":("Unet", "tu-convnextv2_base.fcmae_ft_in22k_in1k"),
    }
    if name in smp_models:
        from .smp_wrappers import SMPDualInput

        arch, encoder = smp_models[name]
        return SMPDualInput(
            arch=arch,
            encoder=encoder,
            num_classes=num_classes,
            seg_classes=seg_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    # HuggingFace SegFormer (B11)
    if name == "segformer_b2":
        from .segformer_wrapper import SegFormerDualInput

        return SegFormerDualInput(
            model_name="nvidia/segformer-b2-finetuned-ade-512-512",
            num_classes=num_classes,
            seg_classes=seg_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    # CLIP linear probe (B12)
    if name == "clip_linear_probe":
        from .clip_linear_probe import CLIPLinearProbe

        return CLIPLinearProbe(
            num_classes=num_classes,
            seg_classes=seg_classes,
            clip_model=m.get("clip_model", "ViT-B-16"),
            clip_pretrained=m.get("clip_pretrained", "openai"),
            dropout=dropout,
        )

    raise ValueError(f"unknown model name: {name!r}")
