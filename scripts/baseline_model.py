"""
DualStreamBaselineNet — two ResNet-50 encoders (CO2, CH4) fused by a
cross-attention block (queries from CO2, keys/values from CH4, blended with
``has_ch4`` so zero-imputed CH4 samples fall back to CO2-only features), a
U-Net style decoder sharing skip connections from the CO2 branch, and a
classification head on top of the fused c5 features.

Input  : co2 [B,1,256,256], ch4 [B,1,256,256], has_ch4 [B,1]
Outputs: cls_logits [B, num_classes], seg_logits [B, seg_classes, 256, 256]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_encoder(pretrained: bool):
    """Return a timm ResNet-50 feature extractor that yields 4 stages.

    Output feature shapes for a 256x256 input:
        c2: [B,  256, 64, 64]    (stride 4)
        c3: [B,  512, 32, 32]    (stride 8)
        c4: [B, 1024, 16, 16]    (stride 16)
        c5: [B, 2048,  8,  8]    (stride 32)
    """
    import timm

    return timm.create_model(
        "resnet50",
        pretrained=pretrained,
        features_only=True,
        out_indices=(1, 2, 3, 4),  # c2, c3, c4, c5
        in_chans=3,                # we repeat grayscale to 3 channels
    )


class CrossAttentionFusion(nn.Module):
    """Multi-head cross-attention on c5 (queries CO2, keys/values CH4).

    The fused output is blended with the CO2 feature using the per-sample
    ``has_ch4`` flag, so zero-imputed CH4 samples cleanly degrade to the
    CO2-only representation.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        co2_c5: torch.Tensor,   # [B, C, H, W]
        ch4_c5: torch.Tensor,   # [B, C, H, W]
        has_ch4: torch.Tensor,  # [B, 1]
    ) -> torch.Tensor:
        b, c, h, w = co2_c5.shape
        q = co2_c5.flatten(2).transpose(1, 2)       # [B, HW, C]
        kv = ch4_c5.flatten(2).transpose(1, 2)      # [B, HW, C]
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        fused_tokens = self.out_norm(q + attn_out)  # residual on the query side
        fused = fused_tokens.transpose(1, 2).reshape(b, c, h, w)

        # Per-sample blend: missing CH4 → pure CO2 features.
        gate = has_ch4.view(b, 1, 1, 1).to(fused.dtype)
        return gate * fused + (1.0 - gate) * co2_c5


class UpBlock(nn.Module):
    """Single U-Net decoder block: upsample + concat skip + 2 conv layers."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DualStreamBaselineNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        seg_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        attn_heads: int = 8,
    ):
        super().__init__()
        self.co2_enc = _make_encoder(pretrained)
        self.ch4_enc = _make_encoder(pretrained)

        # ResNet-50 feature-only channel widths: [256, 512, 1024, 2048]
        chs = self.co2_enc.feature_info.channels()  # list[int]
        c2, c3, c4, c5 = chs

        self.fuse = CrossAttentionFusion(dim=c5, num_heads=attn_heads)

        # U-Net decoder: takes fused c5, skips c4/c3/c2 from the CO2 branch,
        # then one more upsample to reach input resolution.
        self.up4 = UpBlock(c5, c4, 512)
        self.up3 = UpBlock(512, c3, 256)
        self.up2 = UpBlock(256, c2, 128)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.seg_head = nn.Conv2d(64, seg_classes, kernel_size=1)

        # Classification head on fused c5.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(c5, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _to_rgb(x: torch.Tensor) -> torch.Tensor:
        """[B,1,H,W] → [B,3,H,W] by channel repeat."""
        return x.expand(-1, 3, -1, -1)

    def forward(
        self,
        co2: torch.Tensor,
        ch4: torch.Tensor,
        has_ch4: torch.Tensor,
    ) -> dict:
        co2_feats = self.co2_enc(self._to_rgb(co2))  # list[c2, c3, c4, c5]
        ch4_feats = self.ch4_enc(self._to_rgb(ch4))

        c2, c3, c4, c5_co2 = co2_feats
        c5_ch4 = ch4_feats[-1]

        fused_c5 = self.fuse(c5_co2, c5_ch4, has_ch4)

        # Segmentation decoder (skip connections from CO2 branch).
        d4 = self.up4(fused_c5, c4)
        d3 = self.up3(d4, c3)
        d2 = self.up2(d3, c2)
        d1 = self.up1(d2)
        if d1.shape[-2:] != co2.shape[-2:]:
            d1 = F.interpolate(d1, size=co2.shape[-2:], mode="bilinear", align_corners=False)
        seg_logits = self.seg_head(d1)

        # Classification head on fused c5.
        cls_logits = self.cls_head(self.pool(fused_c5))

        return {
            "cls_logits": cls_logits,
            "seg_logits": seg_logits,
            "fused_c5": fused_c5,
            # Multi-scale feature exposed for VLM v2 alignment (stride 16 after
            # first decoder block that integrates the fused c5 with the c4 skip).
            "fused_c4": d4,
        }
