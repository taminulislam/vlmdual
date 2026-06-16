"""Render a paper-ready dual-gas explanation figure for the test_pred sample
pair (MOV CO2 + FLIR CH4).

Layout
------
[CO2 frame]  [CH4 frame]  [CO2 Predicted]  [CH4 Predicted]
Header line: GT / Pred / confidence
LLaVA-1.5 narrative (long form)        + VLMDual one-line verdict

Output: test_pred/acidosis_explain/test_pred_explanation.png
"""
from pathlib import Path
import textwrap
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

ROOT = Path("/work/nvme/bgte/tislam6/ACID_Journal")
TP   = ROOT / "test_pred" / "acidosis_explain"
OUT  = ROOT / "test_pred" / "acidosis_explain" / "test_pred_explanation.png"

CO2_NAME = "MOV0429_frame_0331"
CH4_NAME = "FLIR0286_frame_0719"

# ── Load images ──────────────────────────────────────────────────────────────
co2_frame = cv2.imread(str(TP / f"{CO2_NAME}_original.png"), cv2.IMREAD_GRAYSCALE)
ch4_frame = cv2.imread(str(TP / f"{CH4_NAME}_original.png"), cv2.IMREAD_GRAYSCALE)
co2_pred  = cv2.imread(str(TP / f"{CO2_NAME}_prediction.png"))   # BGR
ch4_pred  = cv2.imread(str(TP / f"{CH4_NAME}_prediction.png"))   # BGR
co2_pred  = cv2.cvtColor(co2_pred, cv2.COLOR_BGR2RGB)
ch4_pred  = cv2.cvtColor(ch4_pred, cv2.COLOR_BGR2RGB)

# ── Texts ─────────────────────────────────────────────────────────────────────
LABEL    = "Acidotic"
PRED     = "Acidotic"
CONF     = 0.9996
CORRECT  = (LABEL == PRED)
MARK     = "✓" if CORRECT else "✗"
EDGE     = "#27ae60" if CORRECT else "#e74c3c"

LLAVA_TEXT = (
    "The image on the left shows a high density of CO₂ gas plume coming "
    "from the rumen of the animal. This indicates that the animal is experiencing "
    "sub-acute rumen acidosis, which is a condition characterized by an imbalance "
    "in the pH levels of the rumen. The high CO₂ concentration in the plume "
    "suggests that the animal's rumen is producing excessive amounts of CO₂ "
    "due to the breakdown of organic matter.\n\n"
    "On the right, the CH₄ plume is very low, indicating that the animal's "
    "rumen is not producing significant amounts of methane gas. This is consistent "
    "with the diagnosis of sub-acute rumen acidosis, as methane production is "
    "typically low in animals with this condition.\n\n"
    "The asymmetry between the CO₂ and CH₄ plumes further supports the "
    "diagnosis of sub-acute rumen acidosis, as it is a characteristic feature of "
    "this condition. The high CO₂ concentration and low CH₄ "
    "concentration in the plumes provide clear evidence of the imbalance in the "
    "animal's rumen."
)

VLM_VERDICT = (
    f"The model classifies this animal as Acidotic with very high confidence "
    f"({CONF:.4f}).  The dense CO₂ plume on the left and the strongly "
    "suppressed CH₄ signal on the right exhibit the cross-modal asymmetry "
    "that is characteristic of sub-acute rumen acidosis, supporting the "
    "predicted diagnosis."
)

# Hard-wrap each text to fit its box width (boxes are now 2 cols each).
LLAVA_WRAPPED = "\n".join(
    textwrap.fill(p, width=78, replace_whitespace=False)
    for p in LLAVA_TEXT.split("\n")
)
VLM_WRAPPED = textwrap.fill(VLM_VERDICT, width=78, replace_whitespace=False)

# ── Build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9.5))
gs  = gridspec.GridSpec(
    3, 4,
    figure=fig,
    height_ratios=[1.0, 0.06, 0.75],
    hspace=0.20, wspace=0.06,
    left=0.012, right=0.988, top=0.94, bottom=0.05,
)

panels = [
    (co2_frame, "CO$_2$ Frame",                "gray"),
    (ch4_frame, "CH$_4$ Frame",                "gray"),
    (co2_pred,  "Predicted Mask (CO$_2$)",     None),
    (ch4_pred,  "Predicted Mask (CH$_4$)",     None),
]

for c, (img, title, cmap) in enumerate(panels):
    ax = fig.add_subplot(gs[0, c])
    if cmap == "gray":
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE)
        sp.set_linewidth(2.0)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=4)

# Header bar
ax_hdr = fig.add_subplot(gs[1, :])
ax_hdr.axis("off")
ax_hdr.text(
    0.5, 0.5,
    f"GT: {LABEL}    |    Pred: {PRED} {MARK}    |    Confidence: {CONF:.4f}",
    ha="center", va="center",
    fontsize=14, fontweight="bold", color=EDGE,
    transform=ax_hdr.transAxes,
)

# Two text boxes side-by-side, each spanning two columns.
ax_llava = fig.add_subplot(gs[2, :2])
ax_llava.axis("off")
ax_llava.text(
    0.0, 1.0,
    "LLaVA-1.5 Explanation:\n" + LLAVA_WRAPPED,
    transform=ax_llava.transAxes,
    fontsize=10, va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.6",
              facecolor="#f6f6f6", edgecolor=EDGE, linewidth=1.0),
)

ax_vlm = fig.add_subplot(gs[2, 2:])
ax_vlm.axis("off")
ax_vlm.text(
    0.0, 1.0,
    "VLMDual Verdict:\n" + VLM_WRAPPED,
    transform=ax_vlm.transAxes,
    fontsize=11, va="top", ha="left", fontweight="normal",
    bbox=dict(boxstyle="round,pad=0.6",
              facecolor="#eef6ff", edgecolor="#1f6fb2", linewidth=1.0),
)

# Legend
legend_handles = [
    patches.Patch(color="#87CEEB", label="VLMDual predicted plume"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=1,
           fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.005))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.05)
print(f"Saved: {OUT}")
