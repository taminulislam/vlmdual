#!/bin/bash
#SBATCH --job-name=vlm_v2_cv
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=jobs/logs/vlm_v2_cv_%A_%a.out
#SBATCH --error=jobs/logs/vlm_v2_cv_%A_%a.err
#SBATCH --array=0-4

# 5-fold CV for ours_vlm_v2 (full configuration). Array indices 0..4 map 1:1
# to cv_fold0 .. cv_fold4. Each task trains on fold k's train split, evaluates
# on fold k's test split using the existing CV CSVs under
# dataset/extended_dataset_50x/cv_fold<k>/.

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

FOLD=$SLURM_ARRAY_TASK_ID
CKPT=checkpoints/vlm_v2_cv/fold${FOLD}
EXP=vlm_v2_cv_fold${FOLD}

echo "=== ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $SLURM_NODELIST ==="
echo "Fold $FOLD"

python -u scripts/train.py \
    --config configs/vlm_v2.yaml \
    --model-name ours_vlm_v2 \
    --seed 42 \
    --ckpt-dir "$CKPT" \
    --train-csv "cv_fold${FOLD}/train_annotations.csv" \
    --val-csv "cv_fold${FOLD}/val_annotations.csv" \
    --test-csv "cv_fold${FOLD}/test_annotations.csv"

python -u scripts/eval_metrics.py \
    --config configs/vlm_v2.yaml \
    --checkpoint "$CKPT/best_acc.pth" \
    --model-type zoo \
    --model-name ours_vlm_v2 \
    --exp-name "$EXP" \
    --test-csv-override "$(pwd)/dataset/extended_dataset_50x/cv_fold${FOLD}/test_annotations.csv" \
    || echo "Eval failed — continuing"
