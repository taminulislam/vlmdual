"""
VLMGuidedDualGasNet — dual-stream baseline + frozen CLIP ViT-B/16 feature
alignment. Adds a CLIP visual encoder path that runs on a dual-channel RGB
synthesis of the (CO2, CH4) pair and a small MLP that projects the fused
CNN features into CLIP's 512-d embedding space. Training minimizes
(1 - cos_sim(cnn_proj, clip_emb)) alongside the seg+cls multi-task loss.

Design notes:
- The existing DualStreamBaselineNet is reused verbatim as the CNN path so
  head-to-head comparisons stay clean.
- CLIP is frozen (`requires_grad_=False`) and runs in eval() mode. We never
  back-prop into it.
- Grayscale→RGB for CLIP: place CO2 in R, CH4 in G, mean(CO2, CH4) in B —
  a simple dual-channel composite that gives CLIP both streams at once.
  For zero-imputed CH4, the G channel is zero and B becomes CO2 * 0.5.
- CLIP's standard preprocessing (224×224, ImageNet-ish normalize) is applied
  inside the model. Inputs from the DualGasDataset are 256×256 in [0,1].
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_model import DualStreamBaselineNet

# CLIP's standard ImageNet-ish normalization.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_open_clip(name: str = "ViT-B-16", pretrained: str = "openai"):
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    return model


class VLMGuidedDualGasNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        seg_classes: int = 3,
        pretrained_resnet: bool = True,
        dropout: float = 0.3,
        attn_heads: int = 8,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "openai",
        clip_dim: int = 512,
    ):
        super().__init__()
        self.cnn = DualStreamBaselineNet(
            num_classes=num_classes,
            seg_classes=seg_classes,
            pretrained=pretrained_resnet,
            dropout=dropout,
            attn_heads=attn_heads,
        )

        clip = _load_open_clip(clip_model, clip_pretrained)
        self.clip_visual = clip.visual
        for p in self.clip_visual.parameters():
            p.requires_grad_(False)
        self.clip_visual.eval()

        # Detect actual CLIP output dim (512 for ViT-B, 768 for ViT-L)
        actual_clip_dim = self.clip_visual.output_dim

        # Projection from fused_c5 global pool (2048-d) to CLIP space.
        self.cnn_proj = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, actual_clip_dim),
        )

        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))
        self._clip_input_size = 224

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # Keep CLIP always in eval to avoid BN/dropout drift.
        self.clip_visual.eval()
        return self

    @staticmethod
    def _dual_to_rgb(
        co2: torch.Tensor,     # [B,1,H,W]
        ch4: torch.Tensor,     # [B,1,H,W]
        has_ch4: torch.Tensor, # [B,1]
    ) -> torch.Tensor:
        """Compose a 3-channel image for CLIP: R=CO2, G=CH4, B=avg."""
        gate = has_ch4.view(-1, 1, 1, 1).to(co2.dtype)
        ch4_masked = ch4 * gate
        r = co2
        g = ch4_masked
        b = 0.5 * (co2 + ch4_masked)
        return torch.cat([r, g, b], dim=1)  # [B,3,H,W]

    def _prep_for_clip(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.shape[-1] != self._clip_input_size:
            rgb = F.interpolate(
                rgb,
                size=(self._clip_input_size, self._clip_input_size),
                mode="bilinear",
                align_corners=False,
            )
        return (rgb - self.clip_mean) / self.clip_std

    def forward(
        self,
        co2: torch.Tensor,
        ch4: torch.Tensor,
        has_ch4: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cnn_out = self.cnn(co2, ch4, has_ch4)
        # Project fused_c5 global pool into CLIP space.
        fused_gap = self.cnn.pool(cnn_out["fused_c5"])  # [B, 2048, 1, 1]
        cnn_proj = self.cnn_proj(fused_gap)             # [B, 512]

        with torch.no_grad():
            rgb = self._dual_to_rgb(co2, ch4, has_ch4)
            clip_in = self._prep_for_clip(rgb)
            clip_emb = self.clip_visual(clip_in).float()  # [B, 512]

        return {
            "cls_logits": cnn_out["cls_logits"],
            "seg_logits": cnn_out["seg_logits"],
            "cnn_proj": cnn_proj,
            "clip_emb": clip_emb,
        }
