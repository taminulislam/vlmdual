"""Build leave-one-session-out (LOSO) splits for R1-Q4.

A "session" is one continuous-recording capture: MOVNNNN for CO2 (GF343 camera)
or FLIRNNNN for CH4 (GF320 camera). Each session corresponds to a single
fermenter at a single time, so holding sessions out at test time is the
strongest within-experiment cross-session generalization probe we can build
without an external animal cohort.

Two folds (mirror images):
  fold_AtoB: train on session group A, test on session group B
  fold_BtoA: train on session group B, test on session group A

Group assignment is per (class_name, gas_type) stratum and deterministic
(sorted session id, first half -> group A, second half -> group B). Within
each fold's training group, one session per (class, gas) is held out as val
so train/val/test are session-level disjoint.

Outputs:
  dataset/extended_dataset_50x/loso/fold_AtoB/{train,val,test}_annotations.csv
  dataset/extended_dataset_50x/loso/fold_AtoB/split_config.json
  dataset/extended_dataset_50x/loso/fold_BtoA/{train,val,test}_annotations.csv
  dataset/extended_dataset_50x/loso/fold_BtoA/split_config.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_ROOT = Path("/work/nvme/bgte/tislam6/ACID_Journal/dataset")
EXT = DATA_ROOT / "extended_dataset_50x"
OUT_BASE = EXT / "loso"


def sessions_per_class_gas(root: pd.DataFrame) -> dict:
    out: dict = {}
    for (cls, gas), sub in root.groupby(["class_name", "gas_type"]):
        out[(cls, gas)] = sorted(sub["session_id"].unique())
    return out


def build_groups(root: pd.DataFrame):
    group_a: set = set()
    group_b: set = set()
    val_a: set = set()
    val_b: set = set()
    for (cls, gas), sess in sessions_per_class_gas(root).items():
        half = len(sess) // 2
        a, b = sess[:half], sess[half:]
        group_a.update(a)
        group_b.update(b)
        # Hold out the last session of each side for val (only if there is room)
        if len(a) >= 2:
            val_a.add(a[-1])
        if len(b) >= 2:
            val_b.add(b[-1])
    return group_a, group_b, val_a, val_b


def write_fold(
    name: str,
    train_sessions: Iterable[str],
    val_sessions: Iterable[str],
    test_sessions: Iterable[str],
    ext_all: pd.DataFrame,
) -> None:
    out_dir = OUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_set, val_set, test_set = set(train_sessions), set(val_sessions), set(test_sessions)

    train_df = ext_all[ext_all["session_id"].isin(train_set)].drop(columns=["session_id"])
    val_df = ext_all[ext_all["session_id"].isin(val_set)].drop(columns=["session_id"])
    test_df = ext_all[ext_all["session_id"].isin(test_set)].drop(columns=["session_id"])

    train_df.to_csv(out_dir / "train_annotations.csv", index=False)
    val_df.to_csv(out_dir / "val_annotations.csv", index=False)
    test_df.to_csv(out_dir / "test_annotations.csv", index=False)

    cfg = {
        "fold": name,
        "pipeline": "loso_session_level_v1",
        "train_sessions": sorted(train_set),
        "val_sessions": sorted(val_set),
        "test_sessions": sorted(test_set),
        "counts": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        "by_class_train": {k: int(v) for k, v in train_df["class_name"].value_counts().to_dict().items()},
        "by_class_val": {k: int(v) for k, v in val_df["class_name"].value_counts().to_dict().items()},
        "by_class_test": {k: int(v) for k, v in test_df["class_name"].value_counts().to_dict().items()},
    }
    with open(out_dir / "split_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(
        f"  {name}: train={len(train_df):5d}  val={len(val_df):5d}  test={len(test_df):5d}"
        f"   train_classes={cfg['by_class_train']}"
    )


def main() -> None:
    # 1. Load original-frame annotations and extract session_id (MOVNNNN or FLIRNNNN)
    root = pd.read_csv(DATA_ROOT / "annotations.csv")
    root["session_id"] = root["frame_path"].str.extract(r"(MOV\d{4}|FLIR\d{4})")
    if root["session_id"].isna().any():
        raise RuntimeError(
            f"{int(root['session_id'].isna().sum())} original frames had no recognisable "
            f"session id (expected MOVNNNN or FLIRNNNN in frame_path)."
        )

    sample_to_session = dict(zip(root["sample_id"], root["session_id"]))

    # 2. Build deterministic session groups stratified by (class, gas)
    group_a, group_b, val_a, val_b = build_groups(root)
    print(f"|group_A|={len(group_a)}  |group_B|={len(group_b)}")
    print(f"val_holdout_from_A = {sorted(val_a)}")
    print(f"val_holdout_from_B = {sorted(val_b)}")

    # 3. Load the existing extended dataset rows (all augmented + original samples)
    ext_parts = []
    for sub in ("train", "val", "test"):
        ext_parts.append(pd.read_csv(EXT / f"{sub}_annotations.csv"))
    ext_all = pd.concat(ext_parts, ignore_index=True)
    ext_all["session_id"] = ext_all["original_sample_id"].map(sample_to_session)
    missing = int(ext_all["session_id"].isna().sum())
    if missing:
        raise RuntimeError(
            f"{missing} extended-dataset rows did not join to a session id. Check that "
            f"original_sample_id values exist in dataset/annotations.csv."
        )
    print(f"extended dataset rows: {len(ext_all)}")

    # 4. Write both fold directions
    write_fold(
        name="fold_AtoB",
        train_sessions=group_a - val_a,
        val_sessions=val_a,
        test_sessions=group_b,
        ext_all=ext_all,
    )
    write_fold(
        name="fold_BtoA",
        train_sessions=group_b - val_b,
        val_sessions=val_b,
        test_sessions=group_a,
        ext_all=ext_all,
    )


if __name__ == "__main__":
    main()
