"""
Generate 5 stratified cross-validation folds at the original-sample level
for the ACID_Journal extended_dataset_50x.

For each fold k ∈ {0..4}:
  - k is the test fold (20 %).
  - (k+1) % 5 is the val fold (20 %).
  - remaining 3 folds are train (60 %).
Stratification is per-class at the original level, so each fold sees
Transitional, Healthy, and Acidotic in roughly the same proportions. For
the 11 Transitional originals (too small for 5 equal folds) the allocation
is hand-done: 3/3/3/2/2 across the 5 folds, ensuring every fold has at
least 2 Transitional originals.

Writes per-fold CSVs to:
  extended_dataset_50x/cv_fold{k}/train_annotations.csv
  extended_dataset_50x/cv_fold{k}/val_annotations.csv
  extended_dataset_50x/cv_fold{k}/test_annotations.csv
  extended_dataset_50x/cv_fold{k}/split_config.json

Run:
    conda activate acidvlm
    cd /work/nvme/bgte/tislam6/ACID_Journal
    python dataset/make_cv_folds.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = "/work/nvme/bgte/tislam6/ACID_Journal"
EXT_DIR = os.path.join(PROJECT_ROOT, "dataset/extended_dataset_50x")
EXT_CSV = os.path.join(EXT_DIR, "extended_annotations.csv")
SEED = 777  # distinct from 42 (main split) and 13 (rebalance)
K = 5


def _chunk(ids: list, k: int, rng: np.random.Generator) -> List[list]:
    """Shuffle `ids` then split into `k` as-equal-as-possible chunks."""
    shuffled = list(ids)
    rng.shuffle(shuffled)
    folds = [[] for _ in range(k)]
    for i, x in enumerate(shuffled):
        folds[i % k].append(x)
    return folds


def main() -> int:
    df = pd.read_csv(EXT_CSV)
    originals = (
        df[~df["is_augmented"]][["original_sample_id", "class_name", "gas_type"]]
        .drop_duplicates("original_sample_id")
        .reset_index(drop=True)
    )
    print(f"Total originals: {len(originals)}")

    rng = np.random.default_rng(SEED)

    # Per-class chunking so every fold is stratified by class.
    class_folds: Dict[str, List[list]] = {}
    for cls in sorted(originals["class_name"].unique()):
        cls_ids = originals[originals["class_name"] == cls]["original_sample_id"].tolist()
        folds = _chunk(cls_ids, K, rng)
        class_folds[cls] = folds
        print(
            f"  {cls:13s} ({len(cls_ids):3d}): "
            + " ".join(f"f{i}={len(f)}" for i, f in enumerate(folds))
        )

    # Assemble per-k (train, val, test) id sets.
    for k in range(K):
        test_k = k
        val_k = (k + 1) % K
        train_k = [i for i in range(K) if i != test_k and i != val_k]

        train_ids, val_ids, test_ids = [], [], []
        for cls, folds in class_folds.items():
            test_ids.extend(folds[test_k])
            val_ids.extend(folds[val_k])
            for t in train_k:
                train_ids.extend(folds[t])

        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

        fold_dir = os.path.join(EXT_DIR, f"cv_fold{k}")
        os.makedirs(fold_dir, exist_ok=True)

        def _assign(oid: str) -> str:
            if oid in train_set:
                return "train"
            if oid in val_set:
                return "val"
            return "test"

        df_k = df.copy()
        df_k["split"] = df_k["original_sample_id"].map(_assign)

        for split in ("train", "val", "test"):
            sub = df_k[df_k["split"] == split].drop(columns=["split"]).reset_index(drop=True)
            sub.to_csv(os.path.join(fold_dir, f"{split}_annotations.csv"), index=False)

        split_cfg = {
            "pipeline": "extended_50x_cv_v1",
            "fold": k,
            "total_folds": K,
            "random_seed": SEED,
            "counts": {s: int((df_k["split"] == s).sum()) for s in ("train", "val", "test")},
            "original_counts": {
                "train": len(train_set),
                "val": len(val_set),
                "test": len(test_set),
            },
            "original_samples": {
                "train": sorted(train_ids),
                "val": sorted(val_ids),
                "test": sorted(test_ids),
            },
        }
        with open(os.path.join(fold_dir, "split_config.json"), "w") as f:
            json.dump(split_cfg, f, indent=2)

        # Brief report
        print(
            f"fold {k}: "
            f"train={split_cfg['counts']['train']:,} "
            f"val={split_cfg['counts']['val']:,} "
            f"test={split_cfg['counts']['test']:,}  "
            f"(orig {split_cfg['original_counts']['train']}/{split_cfg['original_counts']['val']}/{split_cfg['original_counts']['test']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
