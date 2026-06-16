"""
Generate LLaVA-1.5-7B natural-language explanations for every test-set
prediction from ours_vlm. For each sample:

  - Run the trained VLM model to get (pred, confidence, seg_mask_stats).
  - Build a side-by-side PIL image (CO2 | CH4).
  - Prompt LLaVA with a templated clinical question.
  - Save {sample_id, pred, label, confidence, explanation, gas_stats}.

Designed to run on a single A100 via sbatch (jobs/run_llava.sh). ~3 s/sample
on LLaVA-1.5-7B fp16 — ~3 h for 3,366 test samples.

Usage:
    python scripts/explain_test_set.py \
        --config configs/vlm.yaml \
        --checkpoint checkpoints/vlm/best_acc.pth \
        --out results/explanations.json \
        --template diagnostic \
        --limit 0     # 0 = all samples; set to small int for smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dual_gas_dataset import DualGasDataset
from train_utils import load_yaml, set_seed

CLASS_NAMES = ["Healthy", "Transitional", "Acidotic"]

# ---------- prompt templates ----------

_PH_RANGES = {
    "Healthy": "pH 6.2–6.5",
    "Transitional": "pH 5.9 (borderline sub-acute)",
    "Acidotic": "pH 5.0–5.6 (sub-acute rumen acidosis, SARA)",
}

TEMPLATES = {
    "diagnostic": (
        "You are assisting a veterinary team that uses optical gas imaging "
        "(CO₂ and CH₄) to screen dairy cattle for sub-acute rumen acidosis. "
        "The image on the LEFT is the CO₂ gas plume from the rumen, and the "
        "image on the RIGHT is the CH₄ plume from the same animal. A trained "
        "model predicted the animal is **{pred}** ({ph_range}) with "
        "confidence {conf:.1%}. Visible CO₂ gas intensity: {co2_stat}. "
        "Visible CH₄ gas intensity: {ch4_stat}. "
        "In 4–6 sentences, explain the visual evidence that supports this "
        "diagnosis. Describe plume density, dispersion, and any asymmetry "
        "between CO₂ and CH₄. Do NOT invent features that are not visible."
    ),
    "comparative": (
        "Compare the visible CO₂ and CH₄ gas plumes in this dual-gas optical "
        "image (CO₂ on the LEFT, CH₄ on the RIGHT) against the expected "
        "pattern for a {pred} animal ({ph_range}). The model is "
        "{conf:.1%} confident. Describe in 4–6 sentences what is typical vs "
        "atypical for this class."
    ),
    "counterfactual": (
        "The model predicts **{pred}** ({ph_range}) at {conf:.1%} confidence "
        "from this dual-gas image. CO₂ plume is on the LEFT, CH₄ on the "
        "RIGHT. In 4–6 sentences, describe what visual changes in the plume "
        "patterns would shift the prediction to a different class, citing "
        "only features visible in the images."
    ),
    "clinical": (
        "From this dual-gas optical image (CO₂ LEFT, CH₄ RIGHT), the model "
        "predicts **{pred}** ({ph_range}) at {conf:.1%} confidence. Provide "
        "a 4–6 sentence actionable recommendation for the farm veterinarian: "
        "any urgent checks, feeding adjustments, or follow-up imaging you "
        "would suggest. Base your reasoning only on visible gas patterns and "
        "the model's output — do not speculate beyond that."
    ),
}


def build_prompt(
    template: str, pred: str, conf: float, co2_stat: str, ch4_stat: str
) -> str:
    tmpl = TEMPLATES[template]
    return tmpl.format(
        pred=pred,
        ph_range=_PH_RANGES[pred],
        conf=conf,
        co2_stat=co2_stat,
        ch4_stat=ch4_stat,
    )


# ---------- model inference & image building ----------


def _side_by_side(co2: np.ndarray, ch4: np.ndarray) -> Image.Image:
    """Build a PIL RGB image with CO2 left, CH4 right, 1px black gutter."""
    h = max(co2.shape[0], ch4.shape[0])
    w = co2.shape[1] + ch4.shape[1] + 4
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    c0 = np.stack([co2] * 3, axis=-1)
    c1 = np.stack([ch4] * 3, axis=-1)
    canvas[: c0.shape[0], : c0.shape[1]] = c0
    canvas[: c1.shape[0], c0.shape[1] + 4 :] = c1
    return Image.fromarray(canvas)


def _stat_word(intensity: float) -> str:
    """Map a 0-1 normalized intensity to a qualitative word."""
    if intensity < 0.15:
        return "very low"
    if intensity < 0.3:
        return "low"
    if intensity < 0.5:
        return "moderate"
    if intensity < 0.75:
        return "high"
    return "very high"


# ---------- main pipeline ----------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--template", default="diagnostic", choices=list(TEMPLATES))
    parser.add_argument("--limit", type=int, default=0, help="0 = all; small int = smoke")
    parser.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=1, help="LLaVA batch size")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Our VLM model ---
    print("Loading VLM classifier...")
    from vlm_model import VLMGuidedDualGasNet

    model = VLMGuidedDualGasNet(
        num_classes=cfg["model"]["num_classes"],
        seg_classes=cfg["model"]["seg_classes"],
        pretrained_resnet=False,
        dropout=cfg["model"]["dropout"],
        attn_heads=cfg["model"]["attn_heads"],
        clip_model=cfg["model"]["clip_model"],
        clip_pretrained=cfg["model"]["clip_pretrained"],
        clip_dim=cfg["model"]["clip_dim"],
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    model.to(device).eval()
    print("  loaded")

    # --- Dataset ---
    data = cfg["data"]
    ds = DualGasDataset(
        csv_path=os.path.join(data["ext_dir"], data["test_csv"]),
        dataset_root=data["dataset_root"],
        img_size=data["img_size"],
        seed=cfg["seed"] + 100,
    )
    n_total = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    print(f"Test samples: {n_total}")

    # --- LLaVA ---
    print(f"Loading LLaVA: {args.llava_model} (this downloads ~14GB on first run)")
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.llava_model)
    llava = LlavaForConditionalGeneration.from_pretrained(
        args.llava_model,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
    ).to(device)
    llava.eval()
    print("  loaded")

    # --- Explanation loop ---
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results: List[Dict] = []

    softmax = torch.nn.Softmax(dim=1)

    for i in range(n_total):
        item = ds[i]
        co2_t = item["co2"].unsqueeze(0).to(device)
        ch4_t = item["ch4"].unsqueeze(0).to(device)
        has_ch4_t = item["has_ch4"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(co2_t, ch4_t, has_ch4_t)
            probs = softmax(out["cls_logits"].float())[0].cpu().numpy()
        pred_id = int(probs.argmax())
        pred = CLASS_NAMES[pred_id]
        confidence = float(probs[pred_id])
        label_id = int(item["label"].item())

        # Gas stats from the input frames (mean intensity in [0,1])
        co2_np = (item["co2"][0].numpy() * 255).astype(np.uint8)
        ch4_np = (item["ch4"][0].numpy() * 255).astype(np.uint8)
        co2_mean_norm = float(co2_np.mean() / 255)
        ch4_mean_norm = float(ch4_np.mean() / 255)
        co2_stat = _stat_word(co2_mean_norm)
        ch4_stat = _stat_word(ch4_mean_norm)

        # Build side-by-side image and prompt
        sxs = _side_by_side(co2_np, ch4_np)
        prompt_text = build_prompt(args.template, pred, confidence, co2_stat, ch4_stat)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        chat_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = processor(images=sxs, text=chat_prompt, return_tensors="pt").to(
            device, torch.float16
        )

        with torch.no_grad():
            out_ids = llava.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        text = processor.batch_decode(
            out_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0].strip()

        results.append(
            {
                "sample_id": item["sample_id"],
                "label": label_id,
                "label_name": CLASS_NAMES[label_id],
                "pred": pred_id,
                "pred_name": pred,
                "correct": pred_id == label_id,
                "confidence": confidence,
                "co2_mean_norm": co2_mean_norm,
                "ch4_mean_norm": ch4_mean_norm,
                "co2_stat": co2_stat,
                "ch4_stat": ch4_stat,
                "template": args.template,
                "explanation": text,
            }
        )

        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            print(f"  {i+1}/{n_total}")
            # Flush incremental save
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}  ({len(results)} explanations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
