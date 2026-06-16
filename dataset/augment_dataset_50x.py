"""
Extended 50x augmentation for the VLM-guided rumen acidosis project.

Reads the 427 original samples from augmented_dataset/augmented_annotations.csv
(rows where is_augmented == False), generates 1 original copy + 50 augmented
versions each (=21,777 samples total), and writes to extended_dataset_50x/.

Pipeline matches the Phase 1 spec of the journal plan: geometric transforms
applied to both frame and mask, photometric / noise / blur / cutout transforms
applied to the frame only. Seeded for reproducibility.

Run from the ACID_Journal project root:
    conda activate acidvlm
    cd /work/nvme/bgte/tislam6/ACID_Journal
    python dataset/augment_dataset_50x.py
"""

import os
import random
import shutil
import sys
import warnings

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# ---------- Configuration ----------

PROJECT_ROOT = "/work/nvme/bgte/tislam6/ACID_Journal"
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
INPUT_CSV = os.path.join(DATASET_ROOT, "augmented_dataset/augmented_annotations.csv")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "extended_dataset_50x")
AUGMENTATIONS_PER_IMAGE = 50  # 50 augmented + 1 original = 51 per source
SEED = 42


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_pipeline() -> A.Compose:
    """Enhanced Albumentations pipeline per journal plan Phase 1."""
    return A.Compose(
        [
            # --- Geometric (applied to frame AND mask) ---
            A.HorizontalFlip(p=0.5),
            A.Rotate(
                limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=1,
                        sigma=50,
                        alpha_affine=50,
                        border_mode=cv2.BORDER_REFLECT_101,
                    ),
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=0.3,
                        border_mode=cv2.BORDER_REFLECT_101,
                    ),
                    A.OpticalDistortion(
                        distort_limit=0.3,
                        shift_limit=0.1,
                        border_mode=cv2.BORDER_REFLECT_101,
                    ),
                ],
                p=0.4,
            ),
            # --- Photometric (frame only) ---
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.6,
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2), p=0.2),
            A.OneOf(
                [
                    A.Posterize(num_bits=(4, 6)),
                    A.RandomToneCurve(scale=0.1),
                ],
                p=0.2,
            ),
            # --- Noise / blur (frame only) ---
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=5),
                ],
                p=0.35,
            ),
            # --- Cutout (frame only; mask left intact so GT is preserved) ---
            A.CoarseDropout(
                max_holes=6,
                max_height=24,
                max_width=24,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                mask_fill_value=None,
                p=0.3,
            ),
        ],
        additional_targets={"mask": "mask"},
    )


def resolve_input_path(p: str) -> str:
    """CSV paths are stored relative to the dataset directory."""
    if os.path.isabs(p):
        return p
    return os.path.join(DATASET_ROOT, p)


def relpath_from_root(p: str) -> str:
    return os.path.relpath(p, DATASET_ROOT)


def main() -> int:
    set_seeds(SEED)

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: input CSV not found: {INPUT_CSV}", file=sys.stderr)
        return 2

    os.makedirs(os.path.join(OUTPUT_DIR, "frames"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)

    print("=" * 70)
    print("EXTENDED 50x AUGMENTATION")
    print("=" * 70)
    print(f"Input CSV:  {INPUT_CSV}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Aug/image:  {AUGMENTATIONS_PER_IMAGE}  (+ 1 original copy)")
    print(f"Seed:       {SEED}")
    print()

    df = pd.read_csv(INPUT_CSV)
    originals = df[df["is_augmented"] == False].reset_index(drop=True)
    n_orig = len(originals)
    expected_total = n_orig * (AUGMENTATIONS_PER_IMAGE + 1)
    print(f"Found {n_orig} originals → expected total output: {expected_total:,}")
    print()

    transform = build_pipeline()
    rows_out = []
    failed_copies = 0
    failed_augs = 0

    # --- Step 1: copy originals ---
    # Use the `original_sample_id` column (e.g. "sample_00001") as the identity
    # — NOT `sample_id` (which in the source CSV is already "sample_00001_orig").
    # This matches the keying used by augmented_dataset/split_config.json.
    print("Step 1/2: copying original frames/masks")
    for _, row in tqdm(originals.iterrows(), total=n_orig, desc="originals"):
        orig_id = row["original_sample_id"]
        src_frame = resolve_input_path(row["frame_path"])
        src_mask = resolve_input_path(row["mask_path"])
        if not (os.path.exists(src_frame) and os.path.exists(src_mask)):
            print(f"  missing source for {orig_id}: {src_frame}")
            failed_copies += 1
            continue

        dst_frame_abs = os.path.join(OUTPUT_DIR, "frames", f"{orig_id}_orig.png")
        dst_mask_abs = os.path.join(OUTPUT_DIR, "masks", f"{orig_id}_orig.png")
        shutil.copy(src_frame, dst_frame_abs)
        shutil.copy(src_mask, dst_mask_abs)

        rows_out.append(
            {
                "sample_id": f"{orig_id}_orig",
                "original_sample_id": orig_id,
                "ph_value": row["ph_value"],
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "gas_type": row["gas_type"],
                "frame_path": relpath_from_root(dst_frame_abs),
                "mask_path": relpath_from_root(dst_mask_abs),
                "is_augmented": False,
                "augmentation_id": -1,
            }
        )

    print(f"Copied {n_orig - failed_copies}/{n_orig} originals")
    print()

    # --- Step 2: generate augmentations ---
    print(f"Step 2/2: generating {AUGMENTATIONS_PER_IMAGE} augmentations per original")
    for _, row in tqdm(originals.iterrows(), total=n_orig, desc="augmenting"):
        orig_id = row["original_sample_id"]
        src_frame = resolve_input_path(row["frame_path"])
        src_mask = resolve_input_path(row["mask_path"])

        image = cv2.imread(src_frame, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(f"  could not load {orig_id}")
            failed_augs += AUGMENTATIONS_PER_IMAGE
            continue

        for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
            try:
                out = transform(image=image, mask=mask)
            except Exception as exc:
                print(f"  aug failed {orig_id}[{aug_idx}]: {exc}")
                failed_augs += 1
                continue

            aug_sid = f"{orig_id}_aug_{aug_idx:03d}"
            dst_frame_abs = os.path.join(OUTPUT_DIR, "frames", f"{aug_sid}.png")
            dst_mask_abs = os.path.join(OUTPUT_DIR, "masks", f"{aug_sid}.png")
            cv2.imwrite(dst_frame_abs, out["image"])
            cv2.imwrite(dst_mask_abs, out["mask"])

            rows_out.append(
                {
                    "sample_id": aug_sid,
                    "original_sample_id": orig_id,
                    "ph_value": row["ph_value"],
                    "class_id": row["class_id"],
                    "class_name": row["class_name"],
                    "gas_type": row["gas_type"],
                    "frame_path": relpath_from_root(dst_frame_abs),
                    "mask_path": relpath_from_root(dst_mask_abs),
                    "is_augmented": True,
                    "augmentation_id": aug_idx,
                }
            )

    print()
    print(f"Rows written: {len(rows_out):,} (expected {expected_total:,})")
    if failed_copies or failed_augs:
        print(f"  failures: {failed_copies} copies, {failed_augs} augs")

    out_csv = os.path.join(OUTPUT_DIR, "extended_annotations.csv")
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    if len(rows_out) != expected_total:
        print("WARNING: output count does not match expected count.", file=sys.stderr)
        return 1

    # --- Summary ---
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    out_df = pd.DataFrame(rows_out)
    print(f"Total:     {len(out_df):,}")
    print(f"Originals: {(~out_df['is_augmented']).sum():,}")
    print(f"Augmented: {out_df['is_augmented'].sum():,}")
    print()
    print("By class:")
    print(out_df.groupby(["class_name", "gas_type"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
