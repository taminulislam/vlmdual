"""Recolor all `ours_vlm` predicted masks: keep CO2 sky-blue, recolor CH4 to
soft magenta.  Writes to a sibling tree `predictions_colored/` so the
original sky-blue versions are preserved.

Layout
------
master_originals/
  predictions/ours_vlm/
    ph_X.X/co2/<F>.png    → predictions_colored/ours_vlm/ph_X.X/co2/<F>.png  (sky blue)
    ph_X.X/ch4/<F>.png    → predictions_colored/ours_vlm/ph_X.X/ch4/<F>.png  (magenta)

Colour spec (OpenCV BGR uint8)
------------------------------
CO2  → (235, 206, 135)   ≡ #87CEEB sky blue
CH4  → (199, 125, 186)   ≡ #BA7DC7 soft magenta
"""
from pathlib import Path
import cv2
import numpy as np

SRC_ROOT = Path("/work/nvme/bgte/tislam6/ACID_Journal/master_originals/predictions/ours_vlm")
DST_ROOT = Path("/work/nvme/bgte/tislam6/ACID_Journal/master_originals/predictions_colored/ours_vlm")

CO2_COLOUR_BGR = np.array((253, 224, 164), dtype=np.uint8)   # #A4E0FD soft sky (light sky 300 ⊕ 30% white)
CH4_COLOUR_BGR = np.array(( 91, 219, 252), dtype=np.uint8)   # #FCDB5B soft yellow (#FACC15 amber ⊕ 30% white)

def recolor_one(src: Path, dst: Path, colour_bgr: np.ndarray):
    img = cv2.imread(str(src))
    if img is None:
        print(f"  WARN: cannot read {src}")
        return
    # mask = any non-zero pixel (works whether the input is binary 0/255
    # or already colourised in some other tone)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask  = gray > 8                                         # tolerate JPG noise
    out   = np.zeros_like(img)
    out[mask] = colour_bgr
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out)

co2_count = ch4_count = 0
for ph_dir in sorted(SRC_ROOT.iterdir()):
    if not ph_dir.is_dir():
        continue
    for gas in ("co2", "ch4"):
        src_gas = ph_dir / gas
        if not src_gas.exists():
            continue
        dst_gas = DST_ROOT / ph_dir.name / gas
        colour  = CO2_COLOUR_BGR if gas == "co2" else CH4_COLOUR_BGR
        for src in sorted(src_gas.iterdir()):
            if src.suffix.lower() != ".png":
                continue
            dst = dst_gas / src.name
            recolor_one(src, dst, colour)
            if gas == "co2": co2_count += 1
            else:            ch4_count += 1
        print(f"  {ph_dir.name}/{gas}: {len(list(dst_gas.iterdir()))} files written")

print(f"\nDone.  CO2 (sky-blue) recoloured: {co2_count}   "
      f"CH4 (magenta) recoloured: {ch4_count}")
print(f"Output: {DST_ROOT}")
