# VLMDual: Cross-Modal Vision-Language Distillation for Rumen Acidosis Detection

VLMDual is a dual-stream vision-language distillation framework that jointly
classifies subacute rumen acidosis (SARA) and segments gas plumes from paired
CO<sub>2</sub> / CH<sub>4</sub> optical-gas-imaging (OGI) thermal video.
A frozen CLIP ViT-B/16 is exploited at **both** encoders: the visual encoder
provides multi-scale distillation targets that regularise a dual-stream
ResNet-50 backbone, and the text encoder provides class prototypes from
clinical prompts that are aligned with the CNN's classification features via a
cross-modal InfoNCE loss. A post-hoc LLaVA-1.5 module turns each prediction
into an auditable, clinician-readable narrative grounded in the model's own
segmentation mask and class posterior.

This repository accompanies our manuscript:

> **VLMDual: A Vision-Language Distillation Framework for Joint
> Classification and Plume Segmentation in Rumen Acidosis Detection**
> *Smart Agricultural Technology* (Elsevier), manuscript ID **ATECH-D-26-01700**.

## Highlights

* **Single model, two heads.** Joint classification (Healthy / Transitional /
  Acidotic) and binary gas-plume segmentation.
* **Dual-encoder CLIP distillation.** Multi-scale visual distillation at the
  c4 and c5 stages plus cross-modal text-prototype InfoNCE; no extra
  annotation required.
* **Strong empirical results.** Balanced accuracy **0.985 ± 0.007**,
  segmentation mIoU **0.836 ± 0.004** on the held-out test set; best
  discriminator and best segmenter in an 18-baseline benchmark.
* **Robustness audits.** Includes a leave-one-session-out (LOSO)
  cross-recording-shift audit and a CH<sub>4</sub>-availability shortcut
  audit.
* **Post-hoc explanations.** LLaVA-1.5 narratives grounded in the model's own
  segmentation + class posterior.

## Repository layout

```
vlmdual/
├── scripts/                Python source
│   ├── train_vlm.py        VLMDual training entry-point
│   ├── train_baseline.py   Dual-stream baseline training
│   ├── train.py            Generic supervised baseline training
│   ├── vlm_model_v2.py     VLMDual architecture (headline)
│   ├── vlm_model.py        VLMDual v1 architecture
│   ├── baseline_model.py   Dual-stream baseline architecture
│   ├── dual_gas_dataset.py PyTorch dataset for paired CO2/CH4 frames
│   ├── losses_metrics.py   Loss functions, metrics, calibration
│   ├── train_utils.py      LR schedules, early stopping, AMP helpers
│   ├── eval_metrics.py     Standalone test-set evaluation
│   ├── explain_test_set.py LLaVA-1.5 post-hoc narration
│   ├── aggregate.py        Mean ± std aggregation across seeds
│   ├── journal_metrics.py  Per-class F1, BalAcc, ECE, mIoU, etc.
│   ├── stats.py            Bootstrap CIs and McNemar tests
│   ├── build_loso_splits.py Builds LOSO fold partitions
│   ├── visualize_predictions.py     Qualitative overlay plots
│   ├── viz_*.py            Per-figure plotting scripts (paper figures)
│   ├── smoke_test_*.py     Lightweight sanity-checks
│   └── models/             Baseline encoders, CLIP-LP, SegFormer, SMP wrappers, VLM baselines
├── configs/                YAML training configs
│   ├── vlm_v2.yaml         Headline VLMDual configuration (alpha_text_1p0)
│   ├── vlm.yaml            VLMDual v1 configuration
│   ├── baseline.yaml       Dual-stream baseline
│   └── baselines/          Baseline-zoo configs
├── dataset/                Dataset preprocessing utilities (no data shipped)
│   ├── dataset.py          Dataset definition
│   ├── augment_dataset_50x.py Augmentation pipeline
│   ├── split_dataset.py    Source-sample-level train/val/test splitting
│   ├── make_cv_folds.py    5-fold cross-validation partitioning
│   └── ...
├── jobs/                   SLURM job scripts (NCSA Delta H200/A100 partitions)
│   ├── train_vlm.sh        Single-seed VLMDual training
│   ├── run_vlm_v2_ablation.sh   8-cell × 3-seed ablation grid
│   ├── run_loso_audit.sh   Leave-one-session-out audit
│   ├── run_r2q2_shortcut_audit.sh CH4-gate shortcut audit
│   ├── run_baseline_zoo.sh 18-baseline benchmark
│   ├── run_cv_folds.sh     5-fold cross-validation
│   ├── run_label_efficiency.sh   Label-budget sweep
│   ├── run_llava_h200.sh   Post-hoc LLaVA narrative generation
│   └── ...
├── requirements.txt        Python dependencies
├── LICENSE                 MIT
└── README.md               This file
```

## Installation

We recommend a fresh conda environment with CUDA 12.x and PyTorch 2.1+:

```bash
conda create -n vlmdual python=3.10 -y
conda activate vlmdual
pip install -r requirements.txt
```

The headline VLMDual fits in a single 80 GB H200 / A100 GPU. Training peaks
at ≈58 GB GPU memory; steady-state is ≈22 GB after the frozen CLIP teachers
load. End-to-end inference (dual-gas forward pass + post-hoc LLaVA-1.5
narrative on a 7B-parameter VLM) runs at ≈0.6 s/frame on a single H200; the
segmentation+classification pass alone runs at ≈16 ms/frame (≈60 fps), so
deployments that emit narratives only on low-confidence frames can sustain
near-real-time throughput.

## Dataset

The paired CO<sub>2</sub> / CH<sub>4</sub> optical-gas-imaging dataset is an
extension of the corpus released in *FUME* (Islam et al., 2026): 21,885
paired frames from two non-lactating Angus donor cows across six pH
conditions, with pixel-level plume masks produced by a trained veterinary
annotator. Of these, **8,754 paired frames are raw captures** (raw count =
annotated count) and **13,131 paired frames are augmented copies**
(40 % / 60 % split, uniform across classes and splits). Three-class
labelling follows standard SARA thresholds:

* **Healthy** (pH ≥ 6.0) — jars at 6.5 and 6.2
* **Transitional** (5.8 ≤ pH < 6.0) — jar at 5.9
* **Acidotic** (pH < 5.8) — jars at 5.6, 5.3, 5.0

The dataset is **not shipped in this repository.** Access can be requested
from the corresponding author; once provided, place the extracted folders
under `dataset/extended_dataset_50x/` and run `dataset/split_dataset.py` to
produce the train / val / test partitions used in the paper.

## Quick start

### 1. Train the headline VLMDual model

```bash
python scripts/train_vlm.py \
    --config configs/vlm_v2.yaml \
    --output_dir results/vlmdual_alpha_text_1p0 \
    --seed 42
```

This reproduces the headline `alpha_text_1p0` configuration
(α<sub>dist</sub> = 0.10, α<sub>text</sub> = 1.00) reported in the paper.

### 2. Evaluate on the held-out test set

```bash
python scripts/eval_metrics.py \
    --ckpt results/vlmdual_alpha_text_1p0/best.pt \
    --config configs/vlm_v2.yaml \
    --out results/vlmdual_alpha_text_1p0/test_metrics.json
```

### 3. Generate post-hoc LLaVA-1.5 explanations

```bash
python scripts/explain_test_set.py \
    --ckpt results/vlmdual_alpha_text_1p0/best.pt \
    --config configs/vlm_v2.yaml \
    --vlm liuhaotian/llava-v1.5-7b \
    --out results/vlmdual_alpha_text_1p0/explanations.jsonl
```

### 4. Reproduce paper experiments (SLURM)

The `jobs/` folder contains SLURM scripts targeting the NCSA Delta cluster
(gpuH200x8 / gpuA100x4 partitions). Edit the `--account` / partition lines
before submitting on a different system.

| Script | Reproduces |
|---|---|
| `jobs/train_vlm.sh` | Single-seed VLMDual training |
| `jobs/run_vlm_v2_ablation.sh` | 8-cell × 3-seed VLM component ablation (Table 9) |
| `jobs/run_vlm_v2_cv.sh` | 5-fold cross-validation of VLMDual (Table 13a) |
| `jobs/run_baseline_zoo.sh` | 18-baseline benchmark (Tables 4–6) |
| `jobs/run_label_efficiency.sh` | Label-budget sweep at 10 / 25 / 50 / 100 % (Table 11) |
| `jobs/run_loso_audit.sh` | Leave-one-session-out audit (Table 14) |
| `jobs/run_r2q2_shortcut_audit.sh` | CH₄-gate shortcut audit (Table 10) |
| `jobs/run_llava_h200.sh` | Post-hoc LLaVA narrative generation |

## Headline results

Test-set performance of `VLMDual (alpha_text_1p0)` (mean ± std over three
seeds {42, 1337, 2024}):

| Metric | Value |
|---|---|
| Accuracy | 0.983 ± 0.011 |
| Balanced accuracy | 0.985 ± 0.007 |
| Macro F<sub>1</sub> | 0.970 ± 0.024 |
| MCC | 0.969 ± 0.020 |
| AUROC | 0.997 ± 0.003 |
| ECE ↓ | 0.015 ± 0.014 |
| mIoU (background + foreground) | 0.836 ± 0.004 |
| Dice (background + foreground) | 0.911 ± 0.003 |
| HD95 (px) ↓ | 35.8 ± 0.8 |
| Latency (ms / image, H200) | 16.0 |

## Robustness audits

**Leave-one-session-out (LOSO) audit.** Session-disjoint mirror folds
(fold A→B and fold B→A) trained on one half of the recording sessions and
evaluated on the other, three seeds each. Pooled LOSO drops accuracy by
20.5 pp and balanced accuracy by 19.7 pp relative to the within-experiment
headline split, while AUROC degrades by only 0.121 and mIoU by only 0.051 —
a substantive but graceful cross-recording distribution shift.

**CH<sub>4</sub>-gate shortcut audit.** Three controlled ablations of the
`has_ch4` availability flag (`Gate-random` randomises the flag; `Gate-zero`
forces it to 0; `CO2-only` additionally zeros the CH<sub>4</sub> input
plane). The worst ablation costs only 4.7 pp balanced accuracy versus the
headline, with foreground segmentation statistically indistinguishable
across all four settings — inconsistent with the gate-as-shortcut
hypothesis.

## Citation

If you use this code or build on this work, please cite our manuscript:

```bibtex
@article{islam2026vlmdual,
  title   = {{VLMDual}: A Vision-Language Distillation Framework for Joint
             Classification and Plume Segmentation in Rumen Acidosis Detection},
  author  = {Islam, Taminul and Sarker, Toqi Tahamid and Embaby, Mohamed and
             Ahmed, Khaled R. and AbuGhazaleh, Amer},
  journal = {Smart Agricultural Technology},
  year    = {2026},
  note    = {Manuscript ID ATECH-D-26-01700}
}
```

## Acknowledgements

This work is supported by the United States Department of Agriculture,
National Institute of Food and Agriculture (USDA-NIFA), through the Capacity
Building Grants for Non-Land-Grant Colleges of Agriculture
(Grant No. **2023-70001-40997**). Donor animals were maintained at the
Southern Illinois University Beef Research Center (Carbondale, IL) under
IACUC protocol **#21-012**.

## License

Released under the MIT License — see [LICENSE](LICENSE).
