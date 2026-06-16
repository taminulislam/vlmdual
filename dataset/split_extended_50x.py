"""
Create train/val/test splits for the extended 50x dataset by reusing the
per-original assignments already recorded in augmented_dataset/split_config.json.

Splitting at the original-sample level guarantees zero leakage: all 51 versions
of a given original stay in the same split. Reusing the prior assignments lets
us compare new results head-to-head against prior baselines trained on the
20x dataset.

Run:
    conda activate acidvlm
    cd /work/nvme/bgte/tislam6/ACID_Journal
    python dataset/split_extended_50x.py
"""

import json
import os
import sys
from collections import Counter

import pandas as pd

PROJECT_ROOT = "/work/nvme/bgte/tislam6/ACID_Journal"
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
EXT_DIR = os.path.join(DATASET_ROOT, "extended_dataset_50x")
EXT_CSV = os.path.join(EXT_DIR, "extended_annotations.csv")
SOURCE_SPLIT_CFG = os.path.join(
    DATASET_ROOT, "augmented_dataset/split_config.json"
)
PIPELINE_TAG = "extended_50x_v1"


def load_source_splits() -> dict:
    with open(SOURCE_SPLIT_CFG) as f:
        cfg = json.load(f)
    if "original_samples" not in cfg:
        raise RuntimeError(
            f"source split_config missing 'original_samples': {SOURCE_SPLIT_CFG}"
        )
    return cfg


def composition(df: pd.DataFrame) -> dict:
    return {
        "total": int(len(df)),
        "originals": int((~df["is_augmented"]).sum()),
        "augmented": int(df["is_augmented"].sum()),
        "by_class": {
            k: int(v) for k, v in df["class_name"].value_counts().to_dict().items()
        },
        "by_gas": {
            k: int(v) for k, v in df["gas_type"].value_counts().to_dict().items()
        },
        "by_class_gas": {
            f"{c}|{g}": int(n)
            for (c, g), n in df.groupby(["class_name", "gas_type"]).size().items()
        },
    }


def main() -> int:
    if not os.path.exists(EXT_CSV):
        print(f"ERROR: extended CSV not found: {EXT_CSV}", file=sys.stderr)
        print("Run augment_dataset_50x.py first.", file=sys.stderr)
        return 2
    if not os.path.exists(SOURCE_SPLIT_CFG):
        print(f"ERROR: source split config not found: {SOURCE_SPLIT_CFG}", file=sys.stderr)
        return 2

    src_cfg = load_source_splits()
    src_splits = src_cfg["original_samples"]
    train_ids = set(src_splits["train"])
    val_ids = set(src_splits["val"])
    test_ids = set(src_splits["test"])

    # Leakage sanity at the source level
    assert train_ids.isdisjoint(val_ids), "leak: train/val at source"
    assert train_ids.isdisjoint(test_ids), "leak: train/test at source"
    assert val_ids.isdisjoint(test_ids), "leak: val/test at source"

    df = pd.read_csv(EXT_CSV)
    present = set(df["original_sample_id"].unique())
    union = train_ids | val_ids | test_ids
    missing = present - union
    extra = union - present
    if missing or extra:
        print(
            f"ERROR: original_sample_id mismatch — missing {len(missing)}, "
            f"extra {len(extra)}",
            file=sys.stderr,
        )
        if missing:
            print("  sample missing IDs:", list(sorted(missing))[:5], file=sys.stderr)
        if extra:
            print("  sample extra IDs:", list(sorted(extra))[:5], file=sys.stderr)
        return 2

    def assign(orig_id: str) -> str:
        if orig_id in train_ids:
            return "train"
        if orig_id in val_ids:
            return "val"
        return "test"

    df["split"] = df["original_sample_id"].map(assign)

    counts = Counter(df["split"])
    print("Split counts:")
    for k in ("train", "val", "test"):
        print(f"  {k:5s}: {counts[k]:>6,}")
    print(f"  TOTAL: {len(df):>6,}")
    print()

    # --- Write per-split CSVs ---
    out_paths = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].drop(columns=["split"]).reset_index(drop=True)
        path = os.path.join(EXT_DIR, f"{split}_annotations.csv")
        sub.to_csv(path, index=False)
        out_paths[split] = path
        print(f"Wrote {path}  ({len(sub):,} rows)")
    print()

    # --- Compose split_config.json ---
    split_cfg = {
        "pipeline": PIPELINE_TAG,
        "augmentations_per_original": 50,
        "total_per_original": 51,
        "random_seed": src_cfg.get("random_seed", 42),
        "ratios": src_cfg.get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15}),
        "source_split_config": os.path.relpath(SOURCE_SPLIT_CFG, DATASET_ROOT),
        "counts": {k: int(counts[k]) for k in ("train", "val", "test")},
        "total": int(len(df)),
        "composition": {
            split: composition(df[df["split"] == split]) for split in ("train", "val", "test")
        },
        "original_samples": {
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
        },
        "original_counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
    }
    cfg_path = os.path.join(EXT_DIR, "split_config.json")
    with open(cfg_path, "w") as f:
        json.dump(split_cfg, f, indent=2)
    print(f"Wrote {cfg_path}")

    # --- Report per-split composition ---
    print()
    print("Per-split composition (class × gas):")
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        print(f"  [{split}]")
        pivot = (
            sub.groupby(["class_name", "gas_type"]).size().unstack(fill_value=0).to_string()
        )
        for line in pivot.splitlines():
            print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
