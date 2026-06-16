#!/bin/bash
#SBATCH --job-name=loso_audit
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/loso_audit_%A_%a.out
#SBATCH --error=jobs/logs/loso_audit_%A_%a.err
#SBATCH --array=0-5

# Reviewer R1 Q4 — leave-one-session-out (LOSO) generalization audit.
# Two folds (mirror images) at the headline VLMDual config × three seeds.
# Sessions = MOVNNNN (CO2) or FLIRNNNN (CH4) — one continuous recording each.
# Train/val/test are session-level disjoint within each fold (built by
# scripts/build_loso_splits.py).
#
# Index = fold_idx * 3 + seed_idx
#   fold_idx 0 = fold_AtoB (train on session group A, test on group B)
#   fold_idx 1 = fold_BtoA (train on session group B, test on group A)
# Seeds: {42, 1337, 2024}.

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

FOLDS=(fold_AtoB fold_BtoA)
SEEDS=(42 1337 2024)

I=$SLURM_ARRAY_TASK_ID
FOLD_IDX=$(( I / 3 ))
SEED_IDX=$(( I % 3 ))

FOLD=${FOLDS[$FOLD_IDX]}
SEED=${SEEDS[$SEED_IDX]}

CKPT=checkpoints/vlm_v2_loso/${FOLD}_seed${SEED}
EXP=vlm_v2_loso_${FOLD}_seed${SEED}

echo "=== ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $SLURM_NODELIST ==="
echo "Fold: $FOLD  seed=$SEED"
echo "Ckpt: $CKPT"

python -u scripts/train.py \
    --config configs/vlm_v2.yaml \
    --model-name ours_vlm_v2 \
    --seed "$SEED" \
    --align-dist-w 0.10 \
    --align-text-w 1.00 \
    --train-csv "loso/${FOLD}/train_annotations.csv" \
    --val-csv   "loso/${FOLD}/val_annotations.csv" \
    --test-csv  "loso/${FOLD}/test_annotations.csv" \
    --ckpt-dir "$CKPT"

python -u scripts/eval_metrics.py \
    --config configs/vlm_v2.yaml \
    --checkpoint "$CKPT/best_acc.pth" \
    --model-type zoo \
    --model-name ours_vlm_v2 \
    --exp-name "$EXP" \
    --test-csv-override "/work/nvme/bgte/tislam6/ACID_Journal/dataset/extended_dataset_50x/loso/${FOLD}/test_annotations.csv" \
    || echo "Eval failed — continuing"
