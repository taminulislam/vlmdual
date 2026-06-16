"""Render a paper-ready VLM explanation panel.

Uses the already-rendered diagnostic PNG (which contains the *correct*
predicted mask from ours_vlm_seed42 inference) as the source for the
four image panels, and pulls the LLaVA-1.5 explanation text from
explanations_diagnostic.json.  Layout: CO2 | CH4 | GT | Pred row,
followed by an explanation text box.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

ROOT = Path("/work/nvme/bgte/tislam6/ACID_Journal")
SAMPLE_ID = "sample_00163_aug_009"   # correct Acidotic case w/ visible CO2 & CH4

SRC_PNG = (
    ROOT / "results" / "figures" / "vlm_explanations" / "correct" /
    f"{SAMPLE_ID}.png"
)
EXPL_JSON = ROOT / "results" / "explanations_diagnostic.json"
OUT       = ROOT / "paper" / "overleaf" / "figures" / "vlm_explanations.png"


def crop_panels(src_png: Path):
    """Crop the four image panels from the rendered diagnostic PNG.

    The rendered PNG layout (W=1435, H=1027) has 4 equal panels on the top
    row, each in a green border roughly at the y-band 130..490.  We trim
    a few pixels off the green border so the new panels look clean.
    """
    img = np.array(Image.open(src_png).convert("RGB"))
    H, W, _ = img.shape
    # Empirical bounding boxes for the 4 panels in the rendered PNG.
    # (left, top, right, bottom) — tight inside the green frame and
    # skipping the inner panel title row so we don't duplicate it.
    boxes = [
        ( 105, 175,  402, 488),  # CO2
        ( 425, 175,  722, 488),  # CH4
        ( 745, 175, 1042, 488),  # GT mask
        (1066, 175, 1363, 488),  # Pred mask
    ]
    return [img[t:b, l:r] for (l, t, r, b) in boxes]


def load_explanation(sample_id: str) -> dict:
    data = json.load(open(EXPL_JSON))
    for e in data:
        if e["sample_id"] == sample_id:
            return e
    raise SystemExit(f"sample_id {sample_id} not found in explanations JSON")


def main():
    panels = crop_panels(SRC_PNG)
    expl   = load_explanation(SAMPLE_ID)

    titles = ["CO$_2$ Frame", "CH$_4$ Frame", "Ground Truth", "Predicted Mask"]

    fig = plt.figure(figsize=(13.5, 6.0))
    gs = gridspec.GridSpec(
        2, 4,
        height_ratios=[1.0, 0.55],
        hspace=0.20, wspace=0.04,
        left=0.015, right=0.985, top=0.96, bottom=0.025,
    )

    for i, (panel, title) in enumerate(zip(panels, titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(panel)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=4)

    # Explanation text spanning all 4 columns.
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")

    header = (
        f"Sample: {expl['sample_id']}    "
        f"GT: {expl['label_name']}    "
        f"Pred: {expl['pred_name']}  "
        f"(conf {expl['confidence']:.3f})\n"
        f"CO$_2$ intensity: {expl['co2_stat']} "
        f"({expl['co2_mean_norm']:.3f})    "
        f"CH$_4$ intensity: {expl['ch4_stat']} "
        f"({expl['ch4_mean_norm']:.3f})"
    )
    body = "LLaVA-1.5 Explanation: " + expl["explanation"].strip()

    ax_text.text(
        0.0, 1.0, header,
        ha="left", va="top",
        fontsize=11, fontfamily="DejaVu Sans",
        fontweight="bold",
        transform=ax_text.transAxes,
    )
    ax_text.text(
        0.0, 0.62, body,
        ha="left", va="top",
        fontsize=11, fontfamily="DejaVu Sans",
        wrap=True,
        transform=ax_text.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.6",
            edgecolor="#888888", facecolor="#f6f6f6",
            linewidth=0.7,
        ),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
