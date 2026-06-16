"""
Re-split the extended_50x dataset at the original-sample level with per-class
minimum-count guarantees, so every class (including the severely underrepresented
Transitional class) is represented in train/val/test.

Motivation:
    The split inherited from augmented_dataset/split_config.json put only 11
    Transitional originals as 10/0/1 across train/val/test. This made
    `val_balanced_accuracy` effectively a 2-class metric and trained models
    that got 99% val accuracy while producing F1=0 on Transitional. The re-
    split guarantees Transitional is present in every split.

Strategy:
    - Stratify within each (class_name, gas_type) bucket.
    - For each bucket, target 70/15/15 BUT enforce
      `min_val = min_test = 3` physical samples per class_name total.
    - Transitional (11 CO2 originals) → 5 train / 3 val / 3 test.
    - Acidotic / Healthy: standard 70/15/15 stratified by gas.
    - Deterministic: uses a separate random seed so we do not collide with
      the inherited split.
    - Does NOT touch the augmented PNG frames; only rewrites the per-split
      CSVs and split_config.json.

Output:
    extended_dataset_50x/
        train_annotations.csv       (new)
        val_annotations.csv         (new)
        test_annotations.csv        (new)
        split_config.json           (new)
        _split_backup_<timestamp>/  (old CSVs + old split_config, kept)

Run:
    conda activate acidvlm
    cd /work/nvme/bgte/tislam6/ACID_Journal
    python dataset/resplit_extended_50x.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = "/work/nvme/bgte/tislam6/ACID_Journal"
EXT_DIR = os.path.join(PROJECT_ROOT, "dataset/extended_dataset_50x")
EXT_CSV = os.path.join(EXT_DIR, "extended_annotations.csv")

SEED = 13  # intentionally different from the old seed (42) so we don't replay it
TARGET_RATIOS = (0.70, 0.15, 0.15)  # train, val, test

# Per-class minimums (applied across all gases of the class).
MIN_VAL_PER_CLASS = 3
MIN_TEST_PER_CLASS = 3


def _split_counts(n: int, ratios, min_val: int, min_test: int) -> Tuple[int, int, int]:
    """Return (n_train, n_val, n_test) summing to n, obeying minimums."""
    if n <= 0:
        return (0, 0, 0)
    # Start from round-to-nearest on the target ratios
    n_val = round(n * ratios[1])
    n_test = round(n * ratios[2])
    # Apply minimums, capped by n
    n_val = min(max(n_val, min_val), n)
    n_test = min(max(n_test, min_test), n - n_val)
    n_train = n - n_val - n_test
    if n_train < 0:
        # extreme case: bucket too small for both minimums
        # give priority to test, then val
        n_test = min(min_test, n)
        n_val = min(min_val, n - n_test)
        n_train = n - n_val - n_test
    return (n_train, n_val, n_test)


def _allocate_originals_for_class(
    class_name: str,
    class_df: pd.DataFrame,
    rng: np.random.Generator,
) -> Dict[str, List[str]]:
    """Stratify within (class, gas) but respect per-class minimums.

    To honor the per-class minimums we first decide the total (val, test)
    count for the class, then distribute those counts across gas buckets in
    proportion to the bucket sizes.
    """
    n_class = len(class_df)
    n_train_c, n_val_c, n_test_c = _split_counts(
        n_class, TARGET_RATIOS, MIN_VAL_PER_CLASS, MIN_TEST_PER_CLASS
    )

    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    gases = class_df["gas_type"].unique().tolist()
    # Determine per-gas counts proportional to gas-bucket size, but obey
    # per-class totals summing correctly.
    gas_sizes = {g: int((class_df["gas_type"] == g).sum()) for g in gases}
    total = sum(gas_sizes.values())

    def _apportion(target_total: int) -> Dict[str, int]:
        raw = {g: target_total * gas_sizes[g] / total for g in gases}
        floor = {g: int(np.floor(raw[g])) for g in gases}
        leftover = target_total - sum(floor.values())
        if leftover > 0:
            remainder = sorted(
                gases, key=lambda g: raw[g] - floor[g], reverse=True
            )
            for g in remainder[:leftover]:
                floor[g] += 1
        return floor

    val_per_gas = _apportion(n_val_c)
    test_per_gas = _apportion(n_test_c)

    # For each gas bucket: shuffle originals, split into train/val/test counts
    for g in gases:
        ids = (
            class_df[class_df["gas_type"] == g]["original_sample_id"]
            .drop_duplicates()
            .tolist()
        )
        ids_arr = np.array(ids, dtype=object)
        rng.shuffle(ids_arr)

        nv = val_per_gas[g]
        nt = test_per_gas[g]
        ntr = max(0, len(ids_arr) - nv - nt)

        out["val"].extend(ids_arr[:nv].tolist())
        out["test"].extend(ids_arr[nv:nv + nt].tolist())
        out["train"].extend(ids_arr[nv + nt:].tolist())

    # Sanity: sums match the per-class totals (small drift possible due to
    # apportion; accept what we got as long as every split is non-empty for
    # any class that had >=3 originals to begin with).
    return out


def main() -> int:
    if not os.path.exists(EXT_CSV):
        print(f"ERROR: {EXT_CSV} not found", file=sys.stderr)
        return 2

    df = pd.read_csv(EXT_CSV)
    originals = (
        df[~df["is_augmented"]][
            ["original_sample_id", "class_name", "gas_type", "ph_value"]
        ]
        .drop_duplicates("original_sample_id")
        .reset_index(drop=True)
    )
    print(f"Total rows: {len(df):,}  originals: {len(originals)}")

    # --- Allocate originals per class ---
    rng = np.random.default_rng(SEED)
    alloc: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for class_name in sorted(originals["class_name"].unique()):
        class_df = originals[originals["class_name"] == class_name]
        class_alloc = _allocate_originals_for_class(class_name, class_df, rng)
        for k, v in class_alloc.items():
            alloc[k].extend(v)
        print(
            f"  {class_name:13s}: n={len(class_df):4d} → "
            f"train={len(class_alloc['train']):3d}  "
            f"val={len(class_alloc['val']):3d}  "
            f"test={len(class_alloc['test']):3d}"
        )

    # Sanity: disjoint, union == all originals
    s_train = set(alloc["train"])
    s_val = set(alloc["val"])
    s_test = set(alloc["test"])
    assert s_train.isdisjoint(s_val), "leak: train/val"
    assert s_train.isdisjoint(s_test), "leak: train/test"
    assert s_val.isdisjoint(s_test), "leak: val/test"
    present = set(originals["original_sample_id"])
    assert (s_train | s_val | s_test) == present, "missing/extra originals"

    # --- Back up the old split files before overwriting ---
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(EXT_DIR, f"_split_backup_{stamp}")
    os.makedirs(backup_dir, exist_ok=True)
    for fname in (
        "train_annotations.csv",
        "val_annotations.csv",
        "test_annotations.csv",
        "split_config.json",
    ):
        src = os.path.join(EXT_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(backup_dir, fname))
    print(f"Old split backed up to: {backup_dir}")

    # --- Rewrite per-split CSVs ---
    assign_map = {}
    for split in ("train", "val", "test"):
        for oid in alloc[split]:
            assign_map[oid] = split
    df["split"] = df["original_sample_id"].map(assign_map)
    assert df["split"].isna().sum() == 0, "unassigned rows"

    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].drop(columns=["split"]).reset_index(drop=True)
        path = os.path.join(EXT_DIR, f"{split}_annotations.csv")
        sub.to_csv(path, index=False)
        print(f"Wrote {path}  ({len(sub):,} rows)")

    # --- Composition report ---
    def _composition(sub: pd.DataFrame) -> Dict:
        return {
            "total": int(len(sub)),
            "originals": int(sub["original_sample_id"].nunique()),
            "by_class": {
                k: int(v) for k, v in sub["class_name"].value_counts().to_dict().items()
            },
            "by_gas": {
                k: int(v) for k, v in sub["gas_type"].value_counts().to_dict().items()
            },
            "by_class_gas": {
                f"{c}|{g}": int(n)
                for (c, g), n in sub.groupby(["class_name", "gas_type"]).size().items()
            },
        }

    split_cfg = {
        "pipeline": "extended_50x_v2_balanced",
        "augmentations_per_original": 50,
        "total_per_original": 51,
        "random_seed": SEED,
        "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "min_val_per_class": MIN_VAL_PER_CLASS,
        "min_test_per_class": MIN_TEST_PER_CLASS,
        "source": "manual re-split by resplit_extended_50x.py",
        "counts": {
            split: int(len(df[df["split"] == split])) for split in ("train", "val", "test")
        },
        "total": int(len(df)),
        "composition": {
            split: _composition(df[df["split"] == split])
            for split in ("train", "val", "test")
        },
        "original_samples": {
            "train": sorted(alloc["train"]),
            "val": sorted(alloc["val"]),
            "test": sorted(alloc["test"]),
        },
        "original_counts": {k: len(v) for k, v in alloc.items()},
    }

    cfg_path = os.path.join(EXT_DIR, "split_config.json")
    with open(cfg_path, "w") as f:
        json.dump(split_cfg, f, indent=2)
    print(f"Wrote {cfg_path}")

    print()
    print("Per-split composition (class × gas):")
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        print(f"  [{split}] total={len(sub):,}  originals={sub.original_sample_id.nunique()}")
        pivot = (
            sub.groupby(["class_name", "gas_type"]).size().unstack(fill_value=0).to_string()
        )
        for line in pivot.splitlines():
            print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
