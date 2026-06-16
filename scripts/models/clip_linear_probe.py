"""
CLIPLinearProbe — frozen OpenCLIP image encoder, linear classification head,
+ tiny ConvTranspose segmentation decoder driven by the patch-token grid.

Covers baseline B12. This is the VLM baseline *without* our alignment loss:
it shows what raw frozen-CLIP features buy for this task, so our model's
improvement over it quantifies the value of end-to-end training with CLIP
guidance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPLinearProbe(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        seg_classes: int = 3,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "openai",
        dropout: float = 0.3,
    ):
        super().__init__()
        import open_clip

        clip, _, _ = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.clip_visual = clip.visual
        for p in self.clip_visual.parameters():
            p.requires_grad_(False)
        self.clip_visual.eval()

        self.clip_dim = self.clip_visual.output_dim

        self.cls_head = nn.Sequential(
            nn.Linear(self.clip_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        # Segmentation decoder from the 14x14 patch token grid.
        # We use a small ConvTranspose tower.
        self.seg_decoder = nn.Sequential(
            nn.Conv2d(self.clip_dim, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 14 -> 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),   # 28 -> 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),    # 56 -> 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, seg_classes, 1),
        )

        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.clip_visual.eval()
        return self

    @staticmethod
    def _dual_to_rgb(co2, ch4, has_ch4):
        gate = has_ch4.view(-1, 1, 1, 1).to(co2.dtype)
        ch4_g = ch4 * gate
        return torch.cat([co2, ch4_g, 0.5 * (co2 + ch4_g)], dim=1)

    def _prep(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.shape[-1] != 224:
            rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        return (rgb - self.clip_mean) / self.clip_std

    def _tokens_to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens [B, N, C] with CLS at index 0 (openai ViT); drop CLS, reshape.
        if tokens.dim() == 2:
            # No patch tokens available — bail with a 1x1 spatial map.
            return tokens.unsqueeze(-1).unsqueeze(-1)
        patch = tokens[:, 1:, :] if tokens.shape[1] % 2 == 1 or tokens.shape[1] > 196 + 1 else tokens
        n = patch.shape[1]
        side = int(n ** 0.5)
        if side * side != n:
            # Fallback — mean-pool
            return patch.mean(dim=1, keepdim=True).permute(0, 2, 1).unsqueeze(-1)
        return patch.transpose(1, 2).reshape(patch.shape[0], -1, side, side)

    def forward(self, co2, ch4, has_ch4) -> dict:
        rgb = self._dual_to_rgb(co2, ch4, has_ch4)
        with torch.no_grad():
            x = self._prep(rgb)
            # Direct pooled embedding for classification
            emb = self.clip_visual(x).float()  # [B, clip_dim]
            # Also get patch tokens for segmentation. OpenCLIP's ViT has
            # `forward_intermediates` in recent versions; fall back to a
            # simple manual forward otherwise.
            try:
                tokens = self.clip_visual.forward_intermediates(x)[-1]
                if tokens.dim() == 3:
                    grid = self._tokens_to_grid(tokens.float())
                else:
                    grid = tokens.float()
            except Exception:
                # Fallback: tile the global embedding into a 14x14 map.
                grid = emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 14, 14)

        seg_logits = self.seg_decoder(grid)
        seg_logits = F.interpolate(
            seg_logits, size=co2.shape[-2:], mode="bilinear", align_corners=False
        )
        cls_logits = self.cls_head(emb)
        return {"cls_logits": cls_logits, "seg_logits": seg_logits, "fused_c5": grid}
