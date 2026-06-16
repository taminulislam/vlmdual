"""
GenericDualStream — a flexible dual- or single-stream backbone wrapper used by
baselines B1..B7 of the ACID_Journal zoo.

- Accepts any timm backbone via ``features_only=True`` when supported, or
  falls back to a classifier-less forward for ViT/Swin transformers.
- Dual-stream mode: two identical encoders (CO2, CH4), concatenation at the
  deepest feature, then a U-Net-style decoder + classification head.
- Single-stream mode: only the CO2 encoder is used; CH4 is ignored.
- Fusion is simple concat (no cross-attention) — this is what B2..B7 use so
  we can attribute improvement from cross-attention to our model.

forward(co2, ch4, has_ch4) -> {"cls_logits": [B, num_cls], "seg_logits": [B, seg, H, W]}
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_rgb(x: torch.Tensor) -> torch.Tensor:
    return x.expand(-1, 3, -1, -1)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


def _make_encoder(backbone: str, pretrained: bool):
    """Try timm features_only; fall back to a simple feature extractor for ViT/Swin."""
    import timm

    # features_only is supported for CNNs and Swin; falls back for plain ViTs.
    try:
        return (
            timm.create_model(
                backbone, pretrained=pretrained, features_only=True, in_chans=3
            ),
            True,
        )
    except Exception:
        base = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="", in_chans=3
        )
        return base, False


class _ViTFeatureAdapter(nn.Module):
    """Wraps a plain ViT/Swin to emit a single c5-like feature map [B,C,H',W'].

    For ViT with input 224 and patch 16: output shape is [B, 197/196, 768]
    after forward_features. We drop the CLS token (if present) and reshape
    the patch tokens into a 14x14 grid.
    """

    def __init__(self, backbone_name: str, model: nn.Module):
        super().__init__()
        self.backbone = model
        self.backbone_name = backbone_name

        # Determine feature dim by a dummy forward.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = model.forward_features(dummy)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            self._dim_trailing = False
            if out.dim() == 3:
                # [B, N, C]
                self._out_channels = out.shape[-1]
                self._dim_trailing = True
            elif out.dim() == 4:
                # Already [B, C, H, W]
                self._out_channels = out.shape[1]
            else:
                raise RuntimeError(f"unexpected feature rank {out.shape} for {backbone_name}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = self.backbone.forward_features(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        if self._dim_trailing:
            # Drop CLS token if num_tokens+1 is a perfect square+1.
            n = out.shape[1]
            side = int(n ** 0.5)
            if side * side == n - 1:
                out = out[:, 1:, :]
                n -= 1
            side = int(n ** 0.5)
            if side * side != n:
                raise RuntimeError(f"cannot reshape {n} tokens to grid")
            out = out.transpose(1, 2).reshape(out.shape[0], -1, side, side)
        # Return a single-element list so the rest of the pipeline can treat
        # this like a features_only backbone's last stage.
        return [out]

    @property
    def feature_channels(self) -> List[int]:
        return [self._out_channels]


class GenericDualStream(nn.Module):
    def __init__(
        self,
        backbone: str,
        dual_stream: bool = True,
        pretrained: bool = True,
        num_classes: int = 3,
        seg_classes: int = 3,
        dropout: float = 0.3,
        img_size: int = 256,
    ):
        super().__init__()
        self.dual_stream = dual_stream
        self.backbone_name = backbone
        self.img_size = img_size

        # Detect if the backbone needs a different input resolution
        self._needs_resize = None
        if "224" in backbone and img_size != 224:
            self._needs_resize = 224

        # Build CO2 encoder
        enc1, features_only = _make_encoder(backbone, pretrained)
        if features_only:
            self.co2_enc = enc1
            raw_chs = list(enc1.feature_info.channels())
            self._features_only = True
        else:
            self.co2_enc = _ViTFeatureAdapter(backbone, enc1)
            raw_chs = list(self.co2_enc.feature_channels)
            self._features_only = False

        # Store raw channel list for NHWC detection in forward
        self._raw_feature_channels = raw_chs

        # Pad to exactly 4 stage channels
        chs = list(raw_chs)
        while len(chs) < 4:
            chs = [chs[0]] + chs
        chs = chs[-4:]

        if dual_stream:
            enc2, fo2 = _make_encoder(backbone, pretrained)
            if fo2:
                self.ch4_enc = enc2
            else:
                self.ch4_enc = _ViTFeatureAdapter(backbone, enc2)
        else:
            self.ch4_enc = None

        c2, c3, c4, c5 = chs

        # Post-fusion channel count
        if dual_stream:
            fuse_c5 = 2 * c5
        else:
            fuse_c5 = c5

        # Seg decoder: always expects 4 skip-like features; if the backbone
        # doesn't expose them (ViT), we replicate c5 for skip connections.
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(fuse_c5, c5, 1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )

        self.up4 = UpBlock(c5, c4, 512)
        self.up3 = UpBlock(512, c3, 256)
        self.up2 = UpBlock(256, c2, 128)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.seg_head = nn.Conv2d(64, seg_classes, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(c5, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _last_dim(m: nn.Module) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = m.forward_features(dummy)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            if out.dim() == 3:
                return out.shape[-1]
            return out.shape[1]

    def _to_nchw(self, t: torch.Tensor, expected_c: int) -> torch.Tensor:
        """Convert NHWC → NCHW if needed (SwinV2 features_only gives NHWC).

        Uses the known expected channel dim to detect format unambiguously.
        """
        if t.dim() == 4:
            if t.shape[1] == expected_c:
                return t  # already NCHW
            if t.shape[-1] == expected_c:
                return t.permute(0, 3, 1, 2).contiguous()  # NHWC → NCHW
        return t

    def _encode(self, enc: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
        # Resize if the backbone expects a different input size (e.g. ViT 224)
        if self._needs_resize is not None and x.shape[-1] != self._needs_resize:
            x = F.interpolate(
                x, size=(self._needs_resize, self._needs_resize),
                mode="bilinear", align_corners=False,
            )
        feats = enc(x)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        # Convert NHWC → NCHW using known channel dims
        raw_chs = self._raw_feature_channels
        out = []
        for i, f in enumerate(feats):
            ch_idx = min(i, len(raw_chs) - 1)
            out.append(self._to_nchw(f, raw_chs[ch_idx]))
        feats = out[-4:]
        # Pad to 4 stages
        while len(feats) < 4:
            shallow = feats[0]
            feats = [F.avg_pool2d(shallow, kernel_size=2)] + feats
        return feats

    def forward(
        self, co2: torch.Tensor, ch4: torch.Tensor, has_ch4: torch.Tensor
    ) -> dict:
        co2_rgb = _to_rgb(co2)
        co2_feats = self._encode(self.co2_enc, co2_rgb)
        c2, c3, c4, c5_a = co2_feats

        if self.dual_stream:
            ch4_rgb = _to_rgb(ch4)
            ch4_feats = self._encode(self.ch4_enc, ch4_rgb)
            c5_b = ch4_feats[-1]
            # Masked blend: zero-imputed CH4 samples get 0 contribution from c5_b.
            gate = has_ch4.view(-1, 1, 1, 1).to(c5_b.dtype)
            c5_b = c5_b * gate
            fused = torch.cat([c5_a, c5_b], dim=1)
        else:
            fused = c5_a

        fused = self.fuse_proj(fused)

        d4 = self.up4(fused, c4)
        d3 = self.up3(d4, c3)
        d2 = self.up2(d3, c2)
        d1 = self.up1(d2)
        if d1.shape[-2:] != co2.shape[-2:]:
            d1 = F.interpolate(d1, size=co2.shape[-2:], mode="bilinear", align_corners=False)
        seg_logits = self.seg_head(d1)

        cls_logits = self.cls_head(self.pool(fused))
        return {"cls_logits": cls_logits, "seg_logits": seg_logits, "fused_c5": fused}
