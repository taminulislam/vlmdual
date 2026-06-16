#!/bin/bash
#SBATCH --job-name=acid_2cls
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:30:00
#SBATCH --output=jobs/logs/acid_2cls_%A_%a.out
#SBATCH --error=jobs/logs/acid_2cls_%A_%a.err
#SBATCH --array=0-56

# FULL 2-class retraining (tube merged into background).
#
# Idx  0-44 : Baseline zoo (15 models × 3 seeds)
# Idx 45-46 : Ours baseline seed 42, 1337
# Idx 47-48 : Ours VLM seed 42, 1337
# Idx 49-53 : VLM alpha sweep (0.00, 0.05, 0.10, 0.20, 0.50) × 1 seed each
# Idx 54-56 : CV folds (ours_vlm fold 0, 1, 2)

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

MODELS=(
    resnet50_single
    resnet50_dual
    resnet101_dual
    effnet_b3_dual
    convnext_base_dual
    vit_b16_dual
    swin_base_dual
    unet_r50
    deeplabv3p_r50
    pspnet_r50
    segformer_b2
    clip_linear_probe
    unet_hrnet_w48
    unet_swinv2_b
    unet_convnextv2_b
)
SEEDS=(42 1337 2024)

run_zoo() {
    local MODEL=$1 SEED=$2
    local CKPT=checkpoints/zoo/${MODEL}_seed${SEED}
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
    local CKPT=checkpoints/${MODEL}_seed${SEED}
    echo "=== Ours: $MODEL seed=$SEED ==="
    python -u scripts/train.py \
        --config "$CONFIG" --model-name "$MODEL" --seed "$SEED" --ckpt-dir "$CKPT"
    python -u scripts/eval_metrics.py \
        --config "$CONFIG" --checkpoint "$CKPT/best_acc.pth" \
        --model-type zoo --model-name "$MODEL" \
        --exp-name "${MODEL}_seed${SEED}" || echo "Eval failed"
}

run_vlm_alpha() {
    local ALPHA=$1 SEED=$2
    local NAME="alpha_${ALPHA}_seed${SEED}"
    local CKPT=checkpoints/vlm_grid/${NAME}
    echo "=== VLM alpha=$ALPHA seed=$SEED ==="
    python -u scripts/train.py \
        --config configs/vlm.yaml --model-name ours_vlm \
        --seed "$SEED" --align-w "$ALPHA" --ckpt-dir "$CKPT"
    python -u scripts/eval_metrics.py \
        --config configs/vlm.yaml --checkpoint "$CKPT/best_acc.pth" \
        --model-type zoo --model-name ours_vlm --exp-name "$NAME" || echo "Eval failed"
}

run_cv() {
    local MODEL=$1 FOLD=$2 CONFIG=$3
    local CKPT=checkpoints/cv/${MODEL}_fold${FOLD}
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
        --test-csv-override "$(pwd)/dataset/extended_dataset_50x/cv_fold${FOLD}/test_annotations.csv" \
        || echo "Eval failed"
}

nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

I=$SLURM_ARRAY_TASK_ID

if (( I < 45 )); then
    # Zoo: 15 models × 3 seeds
    MODEL_IDX=$(( I / 3 ))
    SEED_IDX=$(( I % 3 ))
    run_zoo "${MODELS[$MODEL_IDX]}" "${SEEDS[$SEED_IDX]}"
elif (( I == 45 )); then run_ours ours_baseline 42 configs/baseline.yaml
elif (( I == 46 )); then run_ours ours_baseline 1337 configs/baseline.yaml
elif (( I == 47 )); then run_ours ours_vlm 42 configs/vlm.yaml
elif (( I == 48 )); then run_ours ours_vlm 1337 configs/vlm.yaml
elif (( I == 49 )); then run_vlm_alpha 0.00 42
elif (( I == 50 )); then run_vlm_alpha 0.05 42
elif (( I == 51 )); then run_vlm_alpha 0.10 42
elif (( I == 52 )); then run_vlm_alpha 0.20 42
elif (( I == 53 )); then run_vlm_alpha 0.50 42
elif (( I == 54 )); then run_cv ours_vlm 0 configs/vlm.yaml
elif (( I == 55 )); then run_cv ours_vlm 1 configs/vlm.yaml
elif (( I == 56 )); then run_cv ours_vlm 2 configs/vlm.yaml
fi
