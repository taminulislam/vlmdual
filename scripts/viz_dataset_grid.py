"""Generate a 2x6 dataset preview grid.

Row 1: CO2 originals.  Row 2: CH4 originals.
Columns: pH labels.  Tight column gap.

Constraint: original CH4 frames exist only at pH 5.6 (Acidotic) and pH 6.5
(Healthy).  We therefore use 3 columns of pH 5.6 + 3 columns of pH 6.5.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

ROOT = Path("/work/nvme/bgte/tislam6/ACID_Journal")
FRAMES = ROOT / "dataset" / "augmented_dataset" / "frames"
OUT = ROOT / "paper" / "overleaf" / "figures" / "dataset_grid.png"

# Hand-picked originals spaced across each pH cohort (all _orig).
columns = [
    ("pH 5.6\nAcidotic", "sample_00110_orig", "sample_00187_orig"),
    ("pH 5.6\nAcidotic", "sample_00125_orig", "sample_00200_orig"),
    ("pH 5.6\nAcidotic", "sample_00150_orig", "sample_00220_orig"),
    ("pH 6.5\nHealthy",  "sample_00242_orig", "sample_00333_orig"),
    ("pH 6.5\nHealthy",  "sample_00280_orig", "sample_00370_orig"),
    ("pH 6.5\nHealthy",  "sample_00320_orig", "sample_00420_orig"),
]

n_cols = len(columns)
fig = plt.figure(figsize=(2.0 * n_cols, 4.5))
gs = gridspec.GridSpec(
    2, n_cols + 1,
    width_ratios=[0.18] + [1.0] * n_cols,
    wspace=0.04, hspace=0.06,
    left=0.005, right=0.998, top=0.90, bottom=0.02,
)

# Left row labels
for r, label in enumerate(["CO$_2$", "CH$_4$"]):
    ax = fig.add_subplot(gs[r, 0])
    ax.axis("off")
    ax.text(
        0.5, 0.5, label,
        rotation=90, ha="center", va="center",
        fontsize=18, fontweight="bold",
    )

for c, (ph_label, co2_id, ch4_id) in enumerate(columns):
    for r, sid in enumerate([co2_id, ch4_id]):
        ax = fig.add_subplot(gs[r, c + 1])
        path = FRAMES / f"{sid}.png"
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if r == 0:
            ax.set_title(ph_label, fontsize=13, fontweight="bold", pad=4)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.05)
print(f"Saved: {OUT}")
