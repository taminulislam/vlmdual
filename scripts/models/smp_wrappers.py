"""
Segmentation-models-pytorch (smp) wrappers for the ACID_Journal baseline zoo.

Covers B8 (U-Net), B9 (DeepLabV3+), B10 (PSPNet).

Input is the two gases stacked channel-wise: [co2, ch4] → in_channels=2.
(This is an early-fusion dual-input strategy; it is the simplest way to let
an off-the-shelf segmentation model see both gases without architectural
changes.)

Classification head: global-avg-pool over the encoder's deepest feature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SMPDualInput(nn.Module):
    def __init__(
        self,
        arch: str,
        encoder: str = "resnet50",
        num_classes: int = 3,
        seg_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        import segmentation_models_pytorch as smp

        ArchCls = getattr(smp, arch)
        self.backbone = ArchCls(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=2,
            classes=seg_classes,
        )

        enc_channels = self.backbone.encoder.out_channels
        self.c5_channels = enc_channels[-1]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.c5_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _stack(co2: torch.Tensor, ch4: torch.Tensor, has_ch4: torch.Tensor) -> torch.Tensor:
        gate = has_ch4.view(-1, 1, 1, 1).to(co2.dtype)
        return torch.cat([co2, ch4 * gate], dim=1)  # [B, 2, H, W]

    def forward(
        self, co2: torch.Tensor, ch4: torch.Tensor, has_ch4: torch.Tensor
    ) -> dict:
        x = self._stack(co2, ch4, has_ch4)

        # Use encoder features for cls head and the full pipeline for seg.
        feats = self.backbone.encoder(x)
        c5 = feats[-1]
        seg_logits = self.backbone.decoder(feats)
        seg_logits = self.backbone.segmentation_head(seg_logits)
        if seg_logits.shape[-2:] != co2.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits, size=co2.shape[-2:], mode="bilinear", align_corners=False
            )

        cls_logits = self.cls_head(self.pool(c5))
        return {"cls_logits": cls_logits, "seg_logits": seg_logits, "fused_c5": c5}
