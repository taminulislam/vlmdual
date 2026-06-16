"""
SegFormer wrapper for baseline B11 of the ACID_Journal zoo.

Uses the HuggingFace transformers SegFormer; modifies the first conv to accept
two input channels (co2, ch4 stacked) and adds a classification head on the
encoder's deepest feature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerDualInput(nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        num_classes: int = 3,
        seg_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
            model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=seg_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            cfg = SegformerConfig.from_pretrained(model_name)
            cfg.num_labels = seg_classes
            model = SegformerForSemanticSegmentation(cfg)

        # Rewire the patch embedding to accept 2 input channels.
        first = model.segformer.encoder.patch_embeddings[0]
        orig_proj = first.proj
        new_proj = nn.Conv2d(
            in_channels=2,
            out_channels=orig_proj.out_channels,
            kernel_size=orig_proj.kernel_size,
            stride=orig_proj.stride,
            padding=orig_proj.padding,
            bias=orig_proj.bias is not None,
        )
        # Initialize the 2-channel conv by averaging the original 3-channel weights
        with torch.no_grad():
            w = orig_proj.weight  # [C_out, 3, kh, kw]
            new_proj.weight.copy_(w.mean(dim=1, keepdim=True).expand(-1, 2, -1, -1) * (3.0 / 2.0))
            if orig_proj.bias is not None:
                new_proj.bias.copy_(orig_proj.bias)
        first.proj = new_proj

        self.backbone = model

        hidden_sizes = self.backbone.config.hidden_sizes
        c5 = hidden_sizes[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(c5, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _stack(co2: torch.Tensor, ch4: torch.Tensor, has_ch4: torch.Tensor) -> torch.Tensor:
        gate = has_ch4.view(-1, 1, 1, 1).to(co2.dtype)
        return torch.cat([co2, ch4 * gate], dim=1)

    def forward(
        self, co2: torch.Tensor, ch4: torch.Tensor, has_ch4: torch.Tensor
    ) -> dict:
        x = self._stack(co2, ch4, has_ch4)
        # Resize to SegFormer's expected input
        x_resized = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)

        out = self.backbone.segformer(
            x_resized, output_hidden_states=True, return_dict=True
        )
        c5_map = out.hidden_states[-1]  # [B, C5, H/32, W/32]

        seg_logits = self.backbone.decode_head(out.hidden_states)
        seg_logits = F.interpolate(
            seg_logits, size=co2.shape[-2:], mode="bilinear", align_corners=False
        )

        cls_logits = self.cls_head(self.pool(c5_map))
        return {"cls_logits": cls_logits, "seg_logits": seg_logits, "fused_c5": c5_map}
