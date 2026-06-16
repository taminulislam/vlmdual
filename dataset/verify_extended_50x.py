"""
Post-generation validation for extended_dataset_50x/.

Checks:
  1. Total row count == 21,777  (427 originals x 51)
  2. Per-split counts match split_config.json
  3. Every frame_path / mask_path exists on disk
  4. Mask label values are all a subset of {0, 1, 2}
  5. Frame and mask HxW shapes match for every row
  6. No original_sample_id leakage across splits
  7. Visual spot-check grid saved to extended_dataset_50x/verification.png
  8. Per-split class/gas composition report

Exits 0 on success, non-zero on failure.

Run:
    conda activate acidvlm
    cd /work/nvme/bgte/tislam6/ACID_Journal
    python dataset/verify_extended_50x.py
"""

import json
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = "/work/nvme/bgte/tislam6/ACID_Journal"
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
EXT_DIR = os.path.join(DATASET_ROOT, "extended_dataset_50x")
EXT_CSV = os.path.join(EXT_DIR, "extended_annotations.csv")
SPLIT_CFG = os.path.join(EXT_DIR, "split_config.json")
VERIFY_PNG = os.path.join(EXT_DIR, "verification.png")

EXPECTED_ORIGINALS = 427
AUG_PER_ORIG = 50
EXPECTED_TOTAL = EXPECTED_ORIGINALS * (AUG_PER_ORIG + 1)  # 21,777
MASK_SUBSET_SAMPLE = 200
SPOT_CHECK_SAMPLES = 10


def resolve(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(DATASET_ROOT, p)


def check(flag: bool, msg: str, errors: list) -> None:
    mark = "PASS" if flag else "FAIL"
    print(f"  [{mark}] {msg}")
    if not flag:
        errors.append(msg)


def main() -> int:
    errors: list = []

    if not os.path.exists(EXT_CSV):
        print(f"ERROR: {EXT_CSV} not found", file=sys.stderr)
        return 2
    if not os.path.exists(SPLIT_CFG):
        print(f"ERROR: {SPLIT_CFG} not found", file=sys.stderr)
        return 2

    df = pd.read_csv(EXT_CSV)
    with open(SPLIT_CFG) as f:
        cfg = json.load(f)

    print("=" * 70)
    print("EXTENDED 50x DATASET VERIFICATION")
    print("=" * 70)
    print(f"Loaded {len(df):,} rows from {EXT_CSV}")
    print()

    # --- 1. Total count ---
    print("Check 1 — counts")
    check(
        len(df) == EXPECTED_TOTAL,
        f"total rows = {len(df)} (expected {EXPECTED_TOTAL})",
        errors,
    )
    n_orig = int((~df["is_augmented"]).sum())
    n_aug = int(df["is_augmented"].sum())
    check(
        n_orig == EXPECTED_ORIGINALS,
        f"originals = {n_orig} (expected {EXPECTED_ORIGINALS})",
        errors,
    )
    check(
        n_aug == EXPECTED_ORIGINALS * AUG_PER_ORIG,
        f"augmented = {n_aug} (expected {EXPECTED_ORIGINALS * AUG_PER_ORIG})",
        errors,
    )
    print()

    # --- 2. Per-split counts ---
    print("Check 2 — per-split counts match split_config")
    split_rows = {}
    for split in ("train", "val", "test"):
        p = os.path.join(EXT_DIR, f"{split}_annotations.csv")
        if not os.path.exists(p):
            check(False, f"{split}_annotations.csv missing", errors)
            split_rows[split] = pd.DataFrame()
            continue
        split_rows[split] = pd.read_csv(p)
        expected = cfg["counts"][split]
        check(
            len(split_rows[split]) == expected,
            f"{split}: rows={len(split_rows[split])} (expected {expected})",
            errors,
        )
    total_split = sum(len(s) for s in split_rows.values())
    check(
        total_split == len(df),
        f"sum(train+val+test) = {total_split} matches total {len(df)}",
        errors,
    )
    print()

    # --- 3. File existence (all rows) ---
    print("Check 3 — frame/mask files exist on disk (every row)")
    missing = 0
    for _, row in df.iterrows():
        if not os.path.exists(resolve(row["frame_path"])):
            missing += 1
        if not os.path.exists(resolve(row["mask_path"])):
            missing += 1
    check(missing == 0, f"missing files: {missing}", errors)
    print()

    # --- 4. Mask label subset {0,1,2} on random sample ---
    print(
        f"Check 4 — mask values ⊆ {{0,1,2}} on random sample of "
        f"{MASK_SUBSET_SAMPLE}"
    )
    rng = random.Random(42)
    sample_idx = rng.sample(range(len(df)), k=min(MASK_SUBSET_SAMPLE, len(df)))
    bad = 0
    pixel_hist: dict = defaultdict(int)
    for i in sample_idx:
        row = df.iloc[i]
        m = cv2.imread(resolve(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if m is None:
            bad += 1
            continue
        u = set(np.unique(m).tolist())
        if not u.issubset({0, 1, 2}):
            bad += 1
        for v in (0, 1, 2):
            pixel_hist[v] += int((m == v).sum())
    check(bad == 0, f"masks with out-of-range values: {bad}", errors)
    total_px = sum(pixel_hist.values()) or 1
    print("    pixel distribution over sampled masks:")
    for v, label in [(0, "background"), (1, "tube"), (2, "gas")]:
        pct = 100.0 * pixel_hist[v] / total_px
        print(f"      {v} {label:10s}: {pct:5.2f}%")
    print()

    # --- 5. Frame / mask shape match ---
    print("Check 5 — frame and mask HxW match on random sample")
    mismatches = 0
    for i in sample_idx:
        row = df.iloc[i]
        f = cv2.imread(resolve(row["frame_path"]), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(resolve(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if f is None or m is None or f.shape != m.shape:
            mismatches += 1
    check(mismatches == 0, f"shape mismatches in sample: {mismatches}", errors)
    print()

    # --- 6. No-leakage (original_sample_id disjoint across splits) ---
    print("Check 6 — original_sample_id disjoint across splits")
    sets = {
        split: set(split_rows[split]["original_sample_id"].unique())
        for split in ("train", "val", "test")
    }
    check(
        sets["train"].isdisjoint(sets["val"]),
        "train ∩ val (originals) == ∅",
        errors,
    )
    check(
        sets["train"].isdisjoint(sets["test"]),
        "train ∩ test (originals) == ∅",
        errors,
    )
    check(
        sets["val"].isdisjoint(sets["test"]),
        "val ∩ test (originals) == ∅",
        errors,
    )
    print()

    # --- 7. Spot-check visual grid ---
    print(f"Check 7 — spot-check grid ({SPOT_CHECK_SAMPLES} samples)")
    aug_df = df[df["is_augmented"]].sample(n=SPOT_CHECK_SAMPLES, random_state=42)
    fig, axes = plt.subplots(SPOT_CHECK_SAMPLES, 4, figsize=(14, 3 * SPOT_CHECK_SAMPLES))
    for i, (_, row) in enumerate(aug_df.iterrows()):
        orig_id = row["original_sample_id"]
        orig_row = df[
            (df["original_sample_id"] == orig_id) & (~df["is_augmented"])
        ].iloc[0]
        orig_frame = cv2.imread(resolve(orig_row["frame_path"]), cv2.IMREAD_GRAYSCALE)
        orig_mask = cv2.imread(resolve(orig_row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        aug_frame = cv2.imread(resolve(row["frame_path"]), cv2.IMREAD_GRAYSCALE)
        aug_mask = cv2.imread(resolve(row["mask_path"]), cv2.IMREAD_GRAYSCALE)

        axes[i, 0].imshow(orig_frame, cmap="gray")
        axes[i, 0].set_title(
            f"orig {orig_id}\n{row['class_name']} / {row['gas_type']}"
        )
        axes[i, 1].imshow(orig_mask, cmap="gray", vmin=0, vmax=2)
        axes[i, 1].set_title("orig mask")
        axes[i, 2].imshow(aug_frame, cmap="gray")
        axes[i, 2].set_title(f"aug {row['augmentation_id']:03d}")
        axes[i, 3].imshow(aug_mask, cmap="gray", vmin=0, vmax=2)
        axes[i, 3].set_title("aug mask")
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(VERIFY_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    check(os.path.exists(VERIFY_PNG), f"verification.png written → {VERIFY_PNG}", errors)
    print()

    # --- 8. Composition report ---
    print("Check 8 — per-split composition report")
    for split in ("train", "val", "test"):
        sub = split_rows[split]
        if sub.empty:
            continue
        print(f"  [{split}] total={len(sub):,}  originals={int((~sub['is_augmented']).sum())}")
        pivot = (
            sub.groupby(["class_name", "gas_type"]).size().unstack(fill_value=0).to_string()
        )
        for line in pivot.splitlines():
            print(f"    {line}")
    print()

    # --- Verdict ---
    print("=" * 70)
    if errors:
        print(f"VERIFICATION FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
