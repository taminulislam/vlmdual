#!/bin/bash
#SBATCH --job-name=acid_resub
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:30:00
#SBATCH --output=jobs/logs/acid_resub_%A_%a.out
#SBATCH --error=jobs/logs/acid_resub_%A_%a.err
#SBATCH --array=0-34

# Resubmission batch: all failed runs + new models + ours multi-seed + LLaVA
#
# Idx  0-2  : vit_b16_dual × 3 seeds          (was zoo 15-17, code fixed)
# Idx  3-5  : swin_base_dual × 3 seeds         (was zoo 18-20, code fixed)
# Idx  6-8  : unet_hrnet_w48 × 3 seeds         (NEW seg baseline)
# Idx  9-11 : unet_swinv2_b × 3 seeds          (NEW seg baseline)
# Idx 12-14 : unet_convnextv2_b × 3 seeds      (NEW seg baseline)
# Idx 15-16 : ours_baseline seed 1337, 2024     (need 3 seeds for main table)
# Idx 17-18 : ours_vlm seed 1337, 2024          (need 3 seeds for main table)
# Idx 19-23 : VLM grid CUDA failures (alpha_0.10/2024, alpha_0.20/2024,
#             alpha_0.50/42, alpha_0.50/2024, clip_vitb16/42)
# Idx 24-26 : VLM grid ViT-B-32 × 3 seeds      (CUDA failures)
# Idx 27    : VLM grid ViT-L-14 seed 42         (code fixed)
# Idx 28-33 : CV folds CUDA failures (baseline fold0-2, vlm fold0-2)
# Idx 34    : (reserved — can add more if needed)

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

SEEDS=(42 1337 2024)
I=$SLURM_ARRAY_TASK_ID

run_zoo() {
    local MODEL=$1 SEED=$2
    local CKPT=/work/nvme/bgte/tislam6/ACID_Journal/checkpoints/zoo/${MODEL}_seed${SEED}
    echo "=== Zoo: $MODEL seed=$SEED ==="
    python -u scripts/train.py \
        --config configs/baselines/_base.yaml \
        --model-name "$MODEL" --seed "$SEED" --ckpt-dir "$CKPT"
    python -u scripts/eval_metrics.py \
        --config configs/baselines/_base.yaml \
        --checkpoint "$CKPT/best_acc.pth" \
        --model-type zoo --model-name "$MODEL" \
        --exp-name "${MODEL}_seed${SEED}" || echo "Eval failed"
}

run_ours() {
    local MODEL=$1 SEED=$2 CONFIG=$3
    local CKPT=/work/nvme/bgte/tislam6/ACID_Journal/checkpoints/${MODEL}_seed${SEED}
    echo "=== Ours: $MODEL seed=$SEED ==="
    python -u scripts/train.py \
        --config "$CONFIG" --model-name "$MODEL" --seed "$SEED" --ckpt-dir "$CKPT"
    python -u scripts/eval_metrics.py \
        --config "$CONFIG" --checkpoint "$CKPT/best_acc.pth" \
        --model-type zoo --model-name "$MODEL" \
        --exp-name "${MODEL}_seed${SEED}" || echo "Eval failed"
}

run_vlm_grid() {
    local ALPHA=$1 SEED=$2 CLIP=$3 NAME=$4
    local CKPT=/work/nvme/bgte/tislam6/ACID_Journal/checkpoints/vlm_grid/${NAME}
    echo "=== VLM grid: $NAME (alpha=$ALPHA clip=$CLIP seed=$SEED) ==="
    python -u scripts/train.py \
        --config configs/vlm.yaml --model-name ours_vlm \
        --seed "$SEED" --align-w "$ALPHA" --clip-model "$CLIP" --ckpt-dir "$CKPT"
    python -u scripts/eval_metrics.py \
        --config configs/vlm.yaml --checkpoint "$CKPT/best_acc.pth" \
        --model-type zoo --model-name ours_vlm --exp-name "$NAME" || echo "Eval failed"
}

run_cv() {
    local MODEL=$1 FOLD=$2 CONFIG=$3
    local CKPT=/work/nvme/bgte/tislam6/ACID_Journal/checkpoints/cv/${MODEL}_fold${FOLD}
    local EXP=${MODEL}_cv_fold${FOLD}
    echo "=== CV: $MODEL fold=$FOLD ==="
    python -u scripts/train.py \
        --config "$CONFIG" --model-name "$MODEL" --seed 42 --ckpt-dir "$CKPT" \
        --train-csv "cv_fold${FOLD}/train_annotations.csv" \
        --val-csv "cv_fold${FOLD}/val_annotations.csv" \
        --test-csv "cv_fold${FOLD}/test_annotations.csv"
    python -u scripts/eval_metrics.py \
        --config "$CONFIG" --checkpoint "$CKPT/best_acc.pth" \
        --model-type zoo --model-name "$MODEL" --exp-name "$EXP" \
        --test-csv-override "/work/nvme/bgte/tislam6/ACID_Journal/dataset/extended_dataset_50x/cv_fold${FOLD}/test_annotations.csv" \
        || echo "Eval failed"
}

nvidia-smi --query-gpu=name,memory.total --format=csv || true

case $I in
    0)  run_zoo vit_b16_dual 42 ;;
    1)  run_zoo vit_b16_dual 1337 ;;
    2)  run_zoo vit_b16_dual 2024 ;;
    3)  run_zoo swin_base_dual 42 ;;
    4)  run_zoo swin_base_dual 1337 ;;
    5)  run_zoo swin_base_dual 2024 ;;
    6)  run_zoo unet_hrnet_w48 42 ;;
    7)  run_zoo unet_hrnet_w48 1337 ;;
    8)  run_zoo unet_hrnet_w48 2024 ;;
    9)  run_zoo unet_swinv2_b 42 ;;
    10) run_zoo unet_swinv2_b 1337 ;;
    11) run_zoo unet_swinv2_b 2024 ;;
    12) run_zoo unet_convnextv2_b 42 ;;
    13) run_zoo unet_convnextv2_b 1337 ;;
    14) run_zoo unet_convnextv2_b 2024 ;;
    15) run_ours ours_baseline 1337 configs/baseline.yaml ;;
    16) run_ours ours_baseline 2024 configs/baseline.yaml ;;
    17) run_ours ours_vlm 1337 configs/vlm.yaml ;;
    18) run_ours ours_vlm 2024 configs/vlm.yaml ;;
    19) run_vlm_grid 0.10 2024 ViT-B-16 "alpha_0.10_seed2024" ;;
    20) run_vlm_grid 0.20 2024 ViT-B-16 "alpha_0.20_seed2024" ;;
    21) run_vlm_grid 0.50 42   ViT-B-16 "alpha_0.50_seed42" ;;
    22) run_vlm_grid 0.50 2024 ViT-B-16 "alpha_0.50_seed2024" ;;
    23) run_vlm_grid 0.10 42   ViT-B-16 "clip_vitb16_seed42" ;;
    24) run_vlm_grid 0.10 42   ViT-B-32 "clip_vitb32_seed42" ;;
    25) run_vlm_grid 0.10 1337 ViT-B-32 "clip_vitb32_seed1337" ;;
    26) run_vlm_grid 0.10 2024 ViT-B-32 "clip_vitb32_seed2024" ;;
    27) run_vlm_grid 0.10 42   ViT-L-14 "clip_vitl14_seed42" ;;
    28) run_cv ours_baseline 0 configs/baseline.yaml ;;
    29) run_cv ours_baseline 1 configs/baseline.yaml ;;
    30) run_cv ours_baseline 2 configs/baseline.yaml ;;
    31) run_cv ours_vlm 0 configs/vlm.yaml ;;
    32) run_cv ours_vlm 1 configs/vlm.yaml ;;
    33) run_cv ours_vlm 2 configs/vlm.yaml ;;
    34) echo "Reserved slot — nothing to run" ;;
esac
