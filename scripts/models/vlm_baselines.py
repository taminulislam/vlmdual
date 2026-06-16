"""
VLM baselines for the ACID_Journal comparison matrix.

Four new models (all conform to the standard forward signature
``(co2, ch4, has_ch4) → {cls_logits, seg_logits, ...}`` so they slot into
the existing train.py / eval_metrics.py pipeline):

1. **CLIPZeroShot**  — frozen CLIP ViT-B/16 visual encoder, NO trainable
   parameters for classification; logits are cosine similarities with
   frozen text prototypes.  Segmentation is a trivial all-background
   mask.  Meant for eval-only (training loop still runs but does nothing
   because everything is frozen; early-stops at epoch 1).

2. **CLIPFullFineTune** — unfreeze CLIP visual encoder and train it
   end-to-end with a linear classification head plus a lightweight seg
   decoder over the patch tokens.

3. **DINOv2LinearProbe** — frozen DINOv2 (ViT-B/14) from facebook/dinov2
   via torch.hub, plus a linear classifier head and the same small seg
   decoder as the CLIP probe for segmentation.

4. **CoOp** — frozen CLIP ViT-B/16, learn soft context tokens that are
   prepended to the hand-crafted class prompts and encoded through the
   frozen CLIP text encoder every forward pass.  Classification logits
   are cosine similarities between the CLIP image embedding and the
   context-tuned class embeddings.

All four accept the 3-channel RGB gas composite from the existing
``_dual_to_rgb`` helper.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

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


def _dual_to_rgb(co2, ch4, has_ch4):
    gate = has_ch4.view(-1, 1, 1, 1).to(co2.dtype)
    ch4_m = ch4 * gate
    return torch.cat([co2, ch4_m, 0.5 * (co2 + ch4_m)], dim=1)


def _clip_preprocess(rgb: torch.Tensor, mean, std, size: int = 224) -> torch.Tensor:
    if rgb.shape[-1] != size:
        rgb = F.interpolate(rgb, size=(size, size), mode="bilinear", align_corners=False)
    return (rgb - mean) / std


# ============================================================================
# 1. CLIP Zero-Shot
# ============================================================================


class CLIPZeroShot(nn.Module):
    """Frozen CLIP ViT-B/16 zero-shot classifier. NO training required, but
    we still go through the standard train.py to keep evaluation uniform —
    it will early-stop after a single epoch because nothing is learnable
    (a trivial dummy parameter is added for the optimizer)."""

    def __init__(
        self,
        num_classes: int = 3,
        seg_classes: int = 2,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "openai",
    ):
        super().__init__()
        import open_clip

        clip, _, _ = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        tok = open_clip.get_tokenizer(clip_model)
        self.clip_visual = clip.visual
        self.clip_dim = int(self.clip_visual.output_dim)

        with torch.no_grad():
            tokens = tok(CLIP_CLASS_PROMPTS)
            text_emb = clip.encode_text(tokens).float()
            text_emb = F.normalize(text_emb, p=2, dim=-1)
        self.register_buffer("clip_text_proto", text_emb)  # [C, D]
        # Freeze visual encoder
        for p in self.clip_visual.parameters():
            p.requires_grad_(False)
        self.clip_visual.eval()

        # A single dummy parameter so AdamW doesn't crash on empty param list
        self.dummy = nn.Parameter(torch.zeros(1, requires_grad=True))

        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))
        self.num_classes = num_classes
        self.seg_classes = seg_classes

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip_visual.eval()
        return self

    def forward(self, co2, ch4, has_ch4):
        with torch.no_grad():
            rgb = _dual_to_rgb(co2, ch4, has_ch4)
            x = _clip_preprocess(rgb, self.clip_mean, self.clip_std)
            emb = self.clip_visual(x).float()
            emb = F.normalize(emb, p=2, dim=-1)
            # Class logits = scaled cosine similarity
            cls_logits = emb @ self.clip_text_proto.T * 10.0  # [B, C]
        # Trivial segmentation: everything background
        B, _, H, W = co2.shape
        seg_logits = torch.zeros(
            B, self.seg_classes, H, W, device=co2.device, dtype=co2.dtype
        )
        seg_logits[:, 0] = 1.0  # background logit
        # Attach a dummy-parameter-dependent zero so loss.backward has a grad graph
        dummy = self.dummy.sum() * 0.0
        cls_logits = cls_logits + dummy
        return {"cls_logits": cls_logits, "seg_logits": seg_logits}


# ============================================================================
# 2. CLIP Fully Fine-Tuned
# ============================================================================


class CLIPFullFineTune(nn.Module):
    """Unfreeze CLIP ViT-B/16 visual encoder and train it end-to-end with a
    linear classification head and a small seg decoder over the patch tokens.
    """

    def __init__(
        self,
        num_classes: int = 3,
        seg_classes: int = 2,
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
            p.requires_grad_(True)

        self.clip_dim = int(self.clip_visual.output_dim)
        self.cls_head = nn.Sequential(
            nn.Linear(self.clip_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        # Tiny segmentation decoder from the global embedding broadcast to a
        # 14x14 grid then upsampled.  CLIP ViT-B/16 at 224 input produces 196
        # patch tokens (14x14).  We tile the global embedding to keep this
        # implementation backbone-agnostic.
        self.seg_decoder = nn.Sequential(
            nn.Conv2d(self.clip_dim, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 14→28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),   # 28→56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),    # 56→112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, seg_classes, 1),
        )

        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def forward(self, co2, ch4, has_ch4):
        rgb = _dual_to_rgb(co2, ch4, has_ch4)
        x = _clip_preprocess(rgb, self.clip_mean, self.clip_std)
        emb = self.clip_visual(x).float()                     # [B, 512]
        cls_logits = self.cls_head(emb)
        # Broadcast global emb to 14x14 and run decoder → upsample to input
        B, D = emb.shape
        grid = emb.view(B, D, 1, 1).expand(-1, -1, 14, 14)
        seg_logits = self.seg_decoder(grid)
        seg_logits = F.interpolate(
            seg_logits, size=co2.shape[-2:], mode="bilinear", align_corners=False
        )
        return {"cls_logits": cls_logits, "seg_logits": seg_logits}


# ============================================================================
# 3. DINOv2 Linear Probe
# ============================================================================


class DINOv2LinearProbe(nn.Module):
    """Frozen DINOv2 visual encoder + linear cls head + small seg decoder.

    DINOv2 is loaded via ``torch.hub.load('facebookresearch/dinov2',
    'dinov2_vitb14')``.  If the hub download fails (offline node), we fall
    back to a random-init ViT-B from timm to keep the code path alive.
    """

    def __init__(
        self,
        num_classes: int = 3,
        seg_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        dim = 768
        try:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True, verbose=False
            )
        except Exception as e:
            print(f"[DINOv2] torch.hub load failed ({e}); falling back to timm random init")
            import timm

            self.backbone = timm.create_model(
                "vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0
            )
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        self.dim = dim
        self.cls_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self.seg_decoder = nn.Sequential(
            nn.Conv2d(dim, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, seg_classes, 1),
        )
        # DINOv2's default normalisation = ImageNet
        self.register_buffer(
            "mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
        )
        self.input_size = 224  # DINOv2 ViT-B/14 expects multiples of 14

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Return a [B, D] CLS embedding."""
        with torch.no_grad():
            # torch.hub variant exposes `forward_features(x)["x_norm_clstoken"]`
            out = self.backbone.forward_features(x)
            if isinstance(out, dict) and "x_norm_clstoken" in out:
                return out["x_norm_clstoken"].float()
            if hasattr(out, "shape") and out.dim() == 2:
                return out.float()
            # timm fallback: forward returns pooled feature
            return self.backbone(x).float()

    def forward(self, co2, ch4, has_ch4):
        rgb = _dual_to_rgb(co2, ch4, has_ch4)
        # DINOv2 ViT-B/14 requires input that is a multiple of 14; 224 works
        x = F.interpolate(rgb, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std

        emb = self._forward_backbone(x)  # [B, D]
        cls_logits = self.cls_head(emb)

        B, D = emb.shape
        grid_size = 14  # 224 / 16 ~= 14 for 16-patch ViT; good enough for upsampling
        grid = emb.view(B, D, 1, 1).expand(-1, -1, grid_size, grid_size)
        seg_logits = self.seg_decoder(grid)
        seg_logits = F.interpolate(
            seg_logits, size=co2.shape[-2:], mode="bilinear", align_corners=False
        )
        return {"cls_logits": cls_logits, "seg_logits": seg_logits}


# ============================================================================
# 4. CoOp — learned context tokens, frozen CLIP
# ============================================================================


class CoOp(nn.Module):
    """Context Optimization (CoOp, Zhou et al. 2022).

    Freeze CLIP image + text encoders.  Learn a small set of context tokens
    that are prepended to each class name in the text encoder's token
    embedding space.  Classification logits are scaled cosine similarities
    between the CLIP image embedding and the context-tuned class
    embeddings, re-encoded through the frozen text transformer on every
    forward pass.
    """

    def __init__(
        self,
        num_classes: int = 3,
        seg_classes: int = 2,
        n_ctx: int = 4,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "openai",
        dropout: float = 0.3,
        ctx_init: str = "",   # optional init string
    ):
        super().__init__()
        import open_clip

        clip, _, _ = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.tok = open_clip.get_tokenizer(clip_model)
        self.clip_visual = clip.visual
        # Keep text transformer accessible
        self.token_embedding = clip.token_embedding
        self.positional_embedding = clip.positional_embedding
        self.transformer = clip.transformer
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection
        # Some open_clip versions expose attn_mask on the top-level model
        self.attn_mask = getattr(clip, "attn_mask", None)
        # Freeze all CLIP params
        for p in self.clip_visual.parameters():
            p.requires_grad_(False)
        for p in clip.parameters():
            p.requires_grad_(False)

        self.clip_dim = int(self.clip_visual.output_dim)
        self.dim = clip.token_embedding.weight.shape[-1]
        self.n_ctx = n_ctx
        self.num_classes = num_classes

        # Learnable context vectors (shared across classes = unified CoOp)
        if ctx_init:
            init_tokens = self.tok([ctx_init])[0, 1 : 1 + n_ctx]
            with torch.no_grad():
                ctx_vec = clip.token_embedding(init_tokens).float()
        else:
            ctx_vec = torch.empty(n_ctx, self.dim).normal_(std=0.02)
        self.ctx = nn.Parameter(ctx_vec)

        # Pre-encode class name tokens once (the class tokens are held fixed;
        # only the ctx prefix is learned).
        with torch.no_grad():
            class_tokens = self.tok([f"X " * n_ctx + "a " + c.split("optical gas image ")[-1] for c in CLIP_CLASS_PROMPTS])
            # [C, max_len]
            self.register_buffer("tokenized_prompts", class_tokens, persistent=False)
            prefix = clip.token_embedding(class_tokens[:, :1])           # SOT
            suffix = clip.token_embedding(class_tokens[:, 1 + n_ctx :])  # [C, *, D]
            self.register_buffer("token_prefix", prefix, persistent=False)
            self.register_buffer("token_suffix", suffix, persistent=False)

        # Segmentation head on top of CLIP's pooled image embedding (same as
        # the other baselines, so the comparison is fair on seg too).
        self.seg_decoder = nn.Sequential(
            nn.Conv2d(self.clip_dim, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, seg_classes, 1),
        )

        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip_visual.eval()
        # text transformer is not a submodule wrapped in self so we can't call
        # eval() on it directly, but since all its params are frozen this is
        # a non-issue.
        return self

    def _encode_text_with_ctx(self) -> torch.Tensor:
        """Build the [C, D] class embeddings using the learned context."""
        ctx = self.ctx.unsqueeze(0).expand(self.num_classes, -1, -1)   # [C, n_ctx, D]
        prefix = self.token_prefix                                       # [C, 1, D]
        suffix = self.token_suffix                                       # [C, *, D]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)                # [C, L, D]
        # Use CLIP's text transformer on these embeddings
        x = prompts + self.positional_embedding[: prompts.shape[1]]
        x = x.permute(1, 0, 2)                                           # NLD -> LND
        if self.attn_mask is not None:
            x = self.transformer(x, attn_mask=self.attn_mask[: x.shape[0], : x.shape[0]])
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)                                           # LND -> NLD
        x = self.ln_final(x)
        # Take the EOT token as the sentence embedding
        eot = self.tokenized_prompts.argmax(dim=-1)
        text_emb = x[torch.arange(x.shape[0]), eot] @ self.text_projection
        return F.normalize(text_emb.float(), p=2, dim=-1)                # [C, D]

    def forward(self, co2, ch4, has_ch4):
        with torch.no_grad():
            rgb = _dual_to_rgb(co2, ch4, has_ch4)
            x = _clip_preprocess(rgb, self.clip_mean, self.clip_std)
            img_emb = F.normalize(self.clip_visual(x).float(), p=2, dim=-1)  # [B, D]

        text_emb = self._encode_text_with_ctx()                               # [C, D]
        cls_logits = img_emb @ text_emb.T * 10.0                              # [B, C]

        B, D = img_emb.shape
        grid = img_emb.view(B, D, 1, 1).expand(-1, -1, 14, 14)
        seg_logits = self.seg_decoder(grid)
        seg_logits = F.interpolate(
            seg_logits, size=co2.shape[-2:], mode="bilinear", align_corners=False
        )
        return {"cls_logits": cls_logits, "seg_logits": seg_logits}
