"""
VLMGuidedDualGasNetV2 — cross-modal vision-language distillation from frozen
CLIP (both visual and text encoders) into the dual-stream CNN.

Three upgrades over VLMGuidedDualGasNet (v1):

1. **CLIP text encoder + class prompts**
   We hand-craft one descriptive sentence per class (Healthy / Transitional /
   Acidotic) and pre-compute normalised CLIP text embeddings once at init.
   These serve as class prototypes in CLIP's joint vision-language space.

2. **Multi-scale visual distillation**
   The CNN exposes both ``fused_c5`` (stride 32) and ``fused_c4`` (stride 16,
   via the first U-Net decoder block) after cross-attention fusion.  Two
   separate projection MLPs map each to CLIP's visual embedding dim and
   cosine-align them with the frozen CLIP image embedding of the RGB gas
   composite.

3. **Cross-modal InfoNCE contrastive objective**
   A third projection head maps the classifier's penultimate 512-d feature
   into CLIP's text space and computes a supervised-contrastive (InfoNCE)
   loss against the frozen text prototypes.  Temperature τ = 0.07.

Total loss (added to the existing seg+cls losses):
    L_dist = 0.5 (1 - cos(z_c5, v_clip)) + 0.5 (1 - cos(z_c4, v_clip))
    L_text = CE(z_cnn @ T^T / τ, y)      with T = frozen text prototypes
    L = L_seg + L_cls + α_dist L_dist + α_text L_text
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_model import DualStreamBaselineNet

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Hand-crafted class-descriptive prompts for cross-modal alignment.
CLIP_CLASS_PROMPTS: List[str] = [
    # class_id 0 — Healthy
    "an optical gas image of a healthy dairy cow rumen "
    "showing normal carbon dioxide and methane emission patterns",
    # class_id 1 — Transitional
    "an optical gas image showing borderline sub-acute rumen acidosis "
    "with altered carbon dioxide and methane plume dispersion",
    # class_id 2 — Acidotic
    "an optical gas image of acidotic rumen showing dense carbon dioxide "
    "plumes and reduced methane emission from sub-acute rumen acidosis",
]


def _load_open_clip(name: str = "ViT-B-16", pretrained: str = "openai"):
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(name)
    return model, tokenizer


def _make_proj(in_dim: int, out_dim: int, hidden: int = 1024, dropout: float = 0.3) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class VLMGuidedDualGasNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        seg_classes: int = 2,
        pretrained_resnet: bool = True,
        dropout: float = 0.3,
        attn_heads: int = 8,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "openai",
        cls_feat_dim: int = 512,
    ):
        super().__init__()

        # CNN path (reuses the existing dual-stream baseline unchanged)
        self.cnn = DualStreamBaselineNet(
            num_classes=num_classes,
            seg_classes=seg_classes,
            pretrained=pretrained_resnet,
            dropout=dropout,
            attn_heads=attn_heads,
        )

        # Frozen CLIP image + text encoders
        clip, tokenizer = _load_open_clip(clip_model, clip_pretrained)
        self.clip_visual = clip.visual
        self.clip_text = clip
        for p in self.clip_visual.parameters():
            p.requires_grad_(False)
        self.clip_visual.eval()
        for p in self.clip_text.parameters():
            p.requires_grad_(False)
        self.clip_text.eval()

        clip_dim = int(self.clip_visual.output_dim)
        self.clip_dim = clip_dim

        # Pre-compute normalised text prototypes once; store as a buffer so they
        # move with the model and are saved in the checkpoint.
        with torch.no_grad():
            tokens = tokenizer(CLIP_CLASS_PROMPTS)  # [C, L]
            text_emb = clip.encode_text(tokens).float()  # [C, clip_dim]
            text_emb = F.normalize(text_emb, p=2, dim=-1)
        self.register_buffer("clip_text_emb", text_emb)

        # We don't need the text transformer after init. Free its memory.
        del self.clip_text

        # Projection heads: multi-scale visual + classification→text
        fused_c5_dim = 2048
        fused_c4_dim = 512  # from baseline_model.UpBlock(..., out_ch=512)
        self.proj_c5 = _make_proj(fused_c5_dim, clip_dim, hidden=1024, dropout=dropout)
        self.proj_c4 = _make_proj(fused_c4_dim, clip_dim, hidden=1024, dropout=dropout)
        self.proj_text = _make_proj(cls_feat_dim, clip_dim, hidden=1024, dropout=dropout)

        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))
        self._clip_input_size = 224

        self._cls_feat_dim = cls_feat_dim

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.clip_visual.eval()
        return self

    @staticmethod
    def _dual_to_rgb(
        co2: torch.Tensor, ch4: torch.Tensor, has_ch4: torch.Tensor
    ) -> torch.Tensor:
        """R=CO2, G=gated CH4, B=average."""
        gate = has_ch4.view(-1, 1, 1, 1).to(co2.dtype)
        ch4_masked = ch4 * gate
        r = co2
        g = ch4_masked
        b = 0.5 * (co2 + ch4_masked)
        return torch.cat([r, g, b], dim=1)

    def _prep_for_clip(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.shape[-1] != self._clip_input_size:
            rgb = F.interpolate(
                rgb,
                size=(self._clip_input_size, self._clip_input_size),
                mode="bilinear",
                align_corners=False,
            )
        return (rgb - self.clip_mean) / self.clip_std

    def _cls_penult(self, fused_c5: torch.Tensor) -> torch.Tensor:
        """Replay the first two layers of the classifier MLP to get the 512-d
        penultimate feature we'll project into CLIP text space."""
        # self.cnn.cls_head = Sequential(
        #     Flatten(1), Linear(2048,512), ReLU, Dropout, Linear(512,3)
        # )
        x = self.cnn.pool(fused_c5)          # [B, 2048, 1, 1]
        x = self.cnn.cls_head[0](x)          # Flatten
        x = self.cnn.cls_head[1](x)          # Linear 2048 -> 512
        x = self.cnn.cls_head[2](x)          # ReLU
        return x                              # [B, 512]

    def forward(
        self,
        co2: torch.Tensor,
        ch4: torch.Tensor,
        has_ch4: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cnn_out = self.cnn(co2, ch4, has_ch4)
        fused_c5 = cnn_out["fused_c5"]
        fused_c4 = cnn_out["fused_c4"]

        # Multi-scale visual projections (GAP then MLP)
        gap_c5 = self.cnn.pool(fused_c5).flatten(1)   # [B, 2048]
        gap_c4 = self.cnn.pool(fused_c4).flatten(1)   # [B, 512]
        z_c5 = self.proj_c5(gap_c5)                    # [B, clip_dim]
        z_c4 = self.proj_c4(gap_c4)                    # [B, clip_dim]

        # Classification-space projection into CLIP text space
        cls_feat = self._cls_penult(fused_c5)          # [B, 512]
        z_text = self.proj_text(cls_feat)              # [B, clip_dim]

        # Frozen CLIP image embedding of the RGB gas composite
        with torch.no_grad():
            rgb = self._dual_to_rgb(co2, ch4, has_ch4)
            clip_in = self._prep_for_clip(rgb)
            clip_img_emb = self.clip_visual(clip_in).float()  # [B, clip_dim]

        return {
            "cls_logits": cnn_out["cls_logits"],
            "seg_logits": cnn_out["seg_logits"],
            "cnn_proj_c5": z_c5,
            "cnn_proj_c4": z_c4,
            "cnn_proj_text": z_text,
            "clip_img_emb": clip_img_emb,
            "clip_text_emb": self.clip_text_emb,   # [C, clip_dim], pre-computed
        }
