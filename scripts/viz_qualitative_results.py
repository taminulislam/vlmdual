"""Qualitative results figure (24 panels).

Each cell shows:  [ raw thermal frame ]  [ prediction overlaid on the same frame ]

Prediction colour conventions (overlaid translucently at alpha ≈ 0.55 on the
grayscale input, so the underlying gas plume remains visible behind the mask):
    CO2  → soft sky-blue  #A4E0FD
    CH4  → soft yellow    #FCDB5B

Output: paper/overleaf/figures/qualitative_results.png
"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

ROOT     = Path("/work/nvme/bgte/tislam6/ACID_Journal")
FRAMES   = ROOT / "dataset" / "original_annotated"
GT_ROOT  = ROOT / "master_originals" / "ground_truth"      # binary {0,255} masks
OUT_PNG  = ROOT / "paper" / "overleaf" / "figures" / "qualitative_results.png"

# ── Sample picks ─────────────────────────────────────────────────────────────
# 7-tuple: (column_label,
#           co2_src_ph, co2_fn, co2_display_ph,
#           ch4_src_ph, ch4_fn, ch4_display_ph)
SAMPLES_A = [
    # CH4 GT plume areas (px): Acidotic ≈3 800, Transitional sub ≈7 500,
    # Healthy ≈23 000  →  monotonic small/medium/large gradient.
    ("Acidotic",     "5.6", "MOV0423_frame_0324.png", "5.6",
                     "5.6", "FLIR0288_frame_0236.png", "5.6"),
    ("Transitional", "5.9", "MOV0436_frame_0144.png", "5.9",
                     "6.5", "FLIR0281_frame_1075.png", "5.9"),     # GT≈7 515 px (truly medium)
    ("Healthy",      "6.5", "MOV0411_frame_0351.png", "6.5",
                     "6.5", "FLIR0275_frame_0417.png", "6.5"),
]
SAMPLES_B = [
    ("Acidotic",     "5.6", "MOV0419_frame_0339.png", "5.6",
                     "5.6", "FLIR0286_frame_0716.png", "5.6"),
    ("Transitional", "5.9", "MOV0438_frame_0016.png", "5.9",
                     "6.5", "FLIR0283_frame_0640.png", "5.9"),     # GT≈8 129 px (truly medium)
    ("Healthy",      "6.5", "MOV0411_frame_0132.png", "6.5",
                     "6.5", "FLIR0275_frame_0857.png", "6.5"),
]

BANNERS = [
    {"gt": "Acidotic",     "pred": "Acidotic",     "rule": "pH < 5.8",       "ok": True},
    {"gt": "Transitional", "pred": "Transitional", "rule": "5.8 ≤ pH < 6.0", "ok": True},
    {"gt": "Healthy",      "pred": "Healthy",      "rule": "pH ≥ 6.0",       "ok": True},
]
NCOL = len(SAMPLES_A)

# Overlay colours, RGB
COLOURS_RGB = {
    "co2": np.array((164, 224, 253), dtype=np.uint8),  # #A4E0FD soft sky
    "ch4": np.array((252, 219,  91), dtype=np.uint8),  # #FCDB5B soft yellow
}
ALPHA = 0.55   # overlay transparency

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_frame(ph: str, gas: str, fn: str) -> np.ndarray:
    return cv2.imread(str(FRAMES / f"ph_{ph}" / f"{gas}_frame" / fn),
                      cv2.IMREAD_GRAYSCALE)

def make_overlay(input_gray: np.ndarray, ph: str, gas: str, fn: str,
                 alpha: float = ALPHA) -> np.ndarray:
    """Return RGB image: grayscale input with the ground-truth plume
    mask overlaid translucently in the gas-specific colour.  Using GT
    rather than predicted masks keeps the qualitative figure cleanly
    aligned with the visible plume; quantitative segmentation accuracy
    is reported separately in tab:seg."""
    h, w = input_gray.shape
    # GT masks live at master_originals/ground_truth/ph_X.X/<gas>/<stem>_mask.png
    gt_fn = fn.replace(".png", "_mask.png")
    gt_path = GT_ROOT / f"ph_{ph}" / gas / gt_fn
    if not gt_path.exists():
        gt_path = GT_ROOT / f"ph_{ph}" / gas / fn
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return cv2.cvtColor(input_gray, cv2.COLOR_GRAY2RGB)
    gt_resized = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (gt_resized > 8)

    bg = cv2.cvtColor(input_gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    fg = COLOURS_RGB[gas].astype(np.float32)
    out = bg.copy()
    out[mask] = alpha * fg + (1.0 - alpha) * bg[mask]
    return np.clip(out, 0, 255).astype(np.uint8)

def add_ph_badge(ax, text: str):
    ax.text(0.97, 0.06, text,
            ha="right", va="bottom", fontsize=10, fontweight="bold",
            color="black", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.22",
                      facecolor="white", edgecolor="#bbbbbb", linewidth=0.5))

def render_gas_row(gs_row: int, samples, gas: str, label_titles: bool):
    """Render six sub-panels: raw input on left, overlay on right, per class."""
    for ci, sample in enumerate(samples):
        if gas == "co2":
            src_ph, fn, disp_ph = sample[1], sample[2], sample[3]
        else:
            src_ph, fn, disp_ph = sample[4], sample[5], sample[6]

        inp     = load_frame(src_ph, gas, fn)
        overlay = make_overlay(inp, src_ph, gas, fn)

        ax_in = fig.add_subplot(gs[gs_row, ci*2])
        ax_in.imshow(inp, cmap="gray", vmin=0, vmax=255)
        ax_in.set_xticks([]); ax_in.set_yticks([])
        if label_titles:
            ax_in.set_title("Input", fontsize=10.5, fontweight="bold", pad=2)

        ax_pr = fig.add_subplot(gs[gs_row, ci*2 + 1])
        ax_pr.imshow(overlay)
        ax_pr.set_xticks([]); ax_pr.set_yticks([])
        if label_titles:
            ax_pr.set_title("Plume (overlay)", fontsize=10.5,
                            fontweight="bold", pad=2)
        add_ph_badge(ax_pr, f"pH {disp_ph}")

# ── Build figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
gs = gridspec.GridSpec(
    6, NCOL * 2,
    figure=fig,
    height_ratios=[1.0, 1.0, 0.16, 1.0, 1.0, 0.06],
    hspace=0.05, wspace=0.04,
    left=0.060, right=0.997, top=0.96, bottom=0.025,
)

# Row labels via fig.text
fig.text(0.020, 0.836, "CO$_2$",  fontsize=18, fontweight="bold",
         rotation=90, ha="center", va="center")
fig.text(0.020, 0.626, "CH$_4$",  fontsize=18, fontweight="bold",
         rotation=90, ha="center", va="center")
fig.text(0.020, 0.336, "CO$_2$",  fontsize=18, fontweight="bold",
         rotation=90, ha="center", va="center")
fig.text(0.020, 0.126, "CH$_4$",  fontsize=18, fontweight="bold",
         rotation=90, ha="center", va="center")

# Example tags
fig.text(0.040, 0.736, "Example A", fontsize=11, fontstyle="italic",
         color="#666666", rotation=90, ha="center", va="center")
fig.text(0.040, 0.226, "Example B", fontsize=11, fontstyle="italic",
         color="#666666", rotation=90, ha="center", va="center")

# Rows 0–1: Example A
render_gas_row(0, SAMPLES_A, "co2", label_titles=True)
render_gas_row(1, SAMPLES_A, "ch4", label_titles=False)

# Row 2: classification banners
for ci, b in enumerate(BANNERS):
    ax_b = fig.add_subplot(gs[2, ci*2:ci*2 + 2])
    ax_b.axis("off")
    edge = "#27ae60" if b["ok"] else "#e74c3c"
    mark = "✓" if b["ok"] else "✗"
    ax_b.add_patch(FancyBboxPatch(
        (0.04, 0.10), 0.92, 0.80,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor="#f6f6f6", edgecolor=edge, linewidth=1.0,
        transform=ax_b.transAxes,
    ))
    ax_b.text(0.50, 0.50,
              f"GT: {b['gt']}  |  Pred: {b['pred']} {mark}  |  {b['rule']}",
              ha="center", va="center",
              fontsize=10.5, fontweight="bold", color=edge,
              transform=ax_b.transAxes)

# Rows 3–4: Example B
render_gas_row(3, SAMPLES_B, "co2", label_titles=False)
render_gas_row(4, SAMPLES_B, "ch4", label_titles=False)

# Row 5: column labels
for ci, sample in enumerate(SAMPLES_A):
    ax_lab = fig.add_subplot(gs[5, ci*2:ci*2 + 2])
    ax_lab.axis("off")
    ax_lab.text(0.5, 0.5,
                f"({chr(ord('a') + ci)}) {sample[0]}",
                ha="center", va="center",
                fontsize=13, fontweight="bold", color="#1a73c0",
                transform=ax_lab.transAxes)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", pad_inches=0.04)
print(f"saved {OUT_PNG}")
