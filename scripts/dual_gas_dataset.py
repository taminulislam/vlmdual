"""
DualGasDataset — pairs CO2 and CH4 frames from extended_dataset_50x/ into a
single sample. When a partner gas does not exist (e.g. Transitional has no
CH4 data at all), the missing slot is zero-imputed and a ``has_ch4`` flag is
set to 0 so the model can condition its fusion on presence.

Each CSV row is treated as one primary sample — so every row contributes a
seg/cls gradient on real data. The partner comes from the same
``(ph_value, class_name, augmentation_id)`` bucket if available.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CLASS_TO_ID = {"Healthy": 0, "Transitional": 1, "Acidotic": 2}
GAS_TO_ID = {"co2": 0, "ch4": 1}


def _load_gray(path: str, img_size: int, is_mask: bool) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    if img.shape != (img_size, img_size):
        img = cv2.resize(img, (img_size, img_size), interpolation=interp)
    return img


class DualGasDataset(Dataset):
    """Dual-gas dataset with zero-imputation when partner gas is missing."""

    def __init__(
        self,
        csv_path: str,
        dataset_root: str,
        img_size: int = 256,
        seed: int = 42,
        merge_tube: bool = False,
        label_fraction: float = 1.0,
    ) -> None:
        super().__init__()
        df = pd.read_csv(csv_path).reset_index(drop=True)
        self.dataset_root = dataset_root
        self.img_size = img_size
        self.merge_tube = merge_tube
        self.label_fraction = float(label_fraction)
        self._rng = np.random.default_rng(seed)

        # Label-fraction subsampling — at the original-sample level, stratified
        # by class_name, so the VLM label-efficiency experiments compare fairly
        # against supervised baselines without data leakage. Val/test CSVs
        # should use label_fraction=1.0 so evaluation is unchanged.
        if self.label_fraction < 1.0:
            sub_rng = np.random.default_rng(seed + 99)
            keep_orig: List[str] = []
            # Originals per class, sorted for determinism
            orig_df = (
                df[["original_sample_id", "class_name"]]
                .drop_duplicates("original_sample_id")
                .sort_values("original_sample_id")
                .reset_index(drop=True)
            )
            for cls_name, grp in orig_df.groupby("class_name"):
                ids = grp["original_sample_id"].tolist()
                n_keep = max(1, int(round(len(ids) * self.label_fraction)))
                idx = sub_rng.permutation(len(ids))[:n_keep]
                keep_orig.extend([ids[i] for i in idx])
            keep_set = set(keep_orig)
            df = df[df["original_sample_id"].isin(keep_set)].reset_index(drop=True)
            print(
                f"[DualGasDataset] label_fraction={self.label_fraction:.2f} → "
                f"{len(keep_orig)} originals / {len(df)} samples"
            )
        self.df = df

        # Pre-compute per-bucket index lists for O(1) partner lookup.
        # Bucket key: (ph_value, class_name, augmentation_id, gas_type)
        self._buckets: Dict[Tuple, List[int]] = {}
        for idx, row in self.df.iterrows():
            key = (
                float(row["ph_value"]),
                row["class_name"],
                int(row["augmentation_id"]),
                row["gas_type"],
            )
            self._buckets.setdefault(key, []).append(idx)

    def __len__(self) -> int:
        return len(self.df)

    def _partner_index(
        self, ph: float, cls: str, aug_id: int, partner_gas: str
    ) -> int | None:
        key = (ph, cls, aug_id, partner_gas)
        bucket = self._buckets.get(key)
        if not bucket:
            return None
        return int(self._rng.choice(bucket))

    def _load_frame_mask(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        frame_path = os.path.join(self.dataset_root, row["frame_path"])
        mask_path = os.path.join(self.dataset_root, row["mask_path"])
        frame = _load_gray(frame_path, self.img_size, is_mask=False)
        mask = _load_gray(mask_path, self.img_size, is_mask=True)
        return frame, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        primary_gas = row["gas_type"]
        partner_gas = "ch4" if primary_gas == "co2" else "co2"

        primary_frame, primary_mask = self._load_frame_mask(row)

        partner_idx = self._partner_index(
            ph=float(row["ph_value"]),
            cls=row["class_name"],
            aug_id=int(row["augmentation_id"]),
            partner_gas=partner_gas,
        )
        if partner_idx is None:
            partner_frame = np.zeros_like(primary_frame)
            has_partner = 0.0
        else:
            partner_row = self.df.iloc[partner_idx]
            partner_frame, _ = self._load_frame_mask(partner_row)
            has_partner = 1.0

        # Slot-assign by gas: co2 always in co2 slot, ch4 always in ch4 slot.
        if primary_gas == "co2":
            co2_np = primary_frame
            ch4_np = partner_frame
            has_ch4 = has_partner
        else:
            co2_np = partner_frame
            ch4_np = primary_frame
            has_ch4 = 1.0  # primary is real CH4
            if partner_idx is None:
                # No CO2 partner found for this CH4 — shouldn't happen for our
                # dataset since pH 5.6 and 6.5 both have CO2, but be safe.
                co2_np = np.zeros_like(ch4_np)

        co2_t = torch.from_numpy(co2_np).float().unsqueeze(0) / 255.0  # [1,H,W]
        ch4_t = torch.from_numpy(ch4_np).float().unsqueeze(0) / 255.0  # [1,H,W]
        if self.merge_tube:
            # Remap: tube(1)→background(0), gas(2)→gas(1)
            primary_mask = np.where(primary_mask == 1, 0, primary_mask)
            primary_mask = np.where(primary_mask == 2, 1, primary_mask)
        mask_t = torch.from_numpy(primary_mask).long()                 # [H,W]
        label_t = torch.tensor(CLASS_TO_ID[row["class_name"]], dtype=torch.long)
        has_ch4_t = torch.tensor([has_ch4], dtype=torch.float32)
        gas_id_t = torch.tensor(GAS_TO_ID[primary_gas], dtype=torch.long)

        return {
            "co2": co2_t,
            "ch4": ch4_t,
            "mask": mask_t,
            "label": label_t,
            "has_ch4": has_ch4_t,
            "gas_id": gas_id_t,
            "sample_id": str(row["sample_id"]),
        }


def _build_class_balanced_sampler(ds: "DualGasDataset") -> "torch.utils.data.WeightedRandomSampler":
    """One weight per sample = 1 / (n_classes * count_of_its_class).

    Effective mini-batches are class-balanced regardless of split skew. Used
    because Transitional has ~14x fewer samples than the other classes.
    """
    from torch.utils.data import WeightedRandomSampler

    class_ids = ds.df["class_name"].map(CLASS_TO_ID).to_numpy()
    counts = np.bincount(class_ids, minlength=len(CLASS_TO_ID)).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    weights = inv[class_ids] / len(CLASS_TO_ID)  # [N]
    weights_t = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights_t, num_samples=len(ds), replacement=True
    )


def build_dataloaders(cfg):
    from torch.utils.data import DataLoader

    data = cfg["data"]
    merge_tube = bool(data.get("merge_tube", False))
    label_fraction = float(data.get("label_fraction", 1.0))
    train_ds = DualGasDataset(
        csv_path=os.path.join(data["ext_dir"], data["train_csv"]),
        dataset_root=data["dataset_root"],
        img_size=data["img_size"],
        seed=cfg["seed"],
        merge_tube=merge_tube,
        label_fraction=label_fraction,
    )
    val_ds = DualGasDataset(
        csv_path=os.path.join(data["ext_dir"], data["val_csv"]),
        dataset_root=data["dataset_root"],
        img_size=data["img_size"],
        seed=cfg["seed"] + 1,
        merge_tube=merge_tube,
        # val/test always use full labels for fair comparison across fractions
        label_fraction=1.0,
    )

    use_weighted = bool(data.get("use_weighted_sampler", False))
    sampler = None
    if use_weighted:
        sampler = _build_class_balanced_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=data["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=data["num_workers"],
        pin_memory=data["pin_memory"],
        persistent_workers=data["num_workers"] > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data["batch_size"],
        shuffle=False,
        num_workers=data["num_workers"],
        pin_memory=data["pin_memory"],
        persistent_workers=data["num_workers"] > 0,
    )
    return train_loader, val_loader
