"""Compact dual-gas explanation figure: visual evidence only.
The long LLaVA-vanilla and VLMDual+LLaVA narratives are written as
tcolorbox blocks directly in main.tex, so this figure carries only the
four thumbnails and the GT / Pred / Confidence header.

Layout (16 x 4.6 in)
--------------------
Row 0: [CO2 Frame]  [CH4 Frame]  [Pred CO2 Mask]  [Pred CH4 Mask]
Row 1: header strip with GT / Pred / Confidence

Output: test_pred/acidosis_explain/test_pred_explanation_v2.png
"""
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path("/work/nvme/bgte/tislam6/ACID_Journal")
TP   = ROOT / "test_pred" / "acidosis_explain"
OUT  = TP / "test_pred_explanation_v2.png"

CO2_NAME = "MOV0429_frame_0331"
CH4_NAME = "FLIR0286_frame_0719"

co2_frame = cv2.imread(str(TP / f"{CO2_NAME}_original.png"),   cv2.IMREAD_GRAYSCALE)
ch4_frame = cv2.imread(str(TP / f"{CH4_NAME}_original.png"),   cv2.IMREAD_GRAYSCALE)
co2_pred  = cv2.cvtColor(cv2.imread(str(TP / f"{CO2_NAME}_prediction.png")),
                         cv2.COLOR_BGR2RGB)
ch4_pred  = cv2.cvtColor(cv2.imread(str(TP / f"{CH4_NAME}_prediction.png")),
                         cv2.COLOR_BGR2RGB)

LABEL    = "Acidotic"
PRED     = "Acidotic"
CONF     = 0.9996
CORRECT  = (LABEL == PRED)
MARK     = "✓" if CORRECT else "✗"
# Use a darker, higher-contrast green for the header text and panel
# borders — the previous #27ae60 was too light to read cleanly in print.
EDGE     = "#0d6b30" if CORRECT else "#b21f2d"

fig = plt.figure(figsize=(16, 4.6))
gs = gridspec.GridSpec(
    2, 4,
    figure=fig,
    height_ratios=[1.0, 0.10],
    hspace=0.15, wspace=0.06,
    left=0.020, right=0.980, top=0.94, bottom=0.04,
)

panels = [
    (co2_frame, "CO$_2$ Frame",            "gray"),
    (ch4_frame, "CH$_4$ Frame",            "gray"),
    (co2_pred,  "Predicted Mask (CO$_2$)", None),
    (ch4_pred,  "Predicted Mask (CH$_4$)", None),
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

ax_hdr = fig.add_subplot(gs[1, :])
ax_hdr.axis("off")
ax_hdr.text(
    0.5, 0.5,
    f"GT: {LABEL}    |    Pred: {PRED} {MARK}    |    Confidence: {CONF:.4f}",
    ha="center", va="center",
    fontsize=14, fontweight="bold", color=EDGE,
    transform=ax_hdr.transAxes,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.05)
print(f"saved {OUT}")
