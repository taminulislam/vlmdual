#!/bin/bash
#SBATCH --job-name=acid_cv
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/acid_cv_%A_%a.out
#SBATCH --error=jobs/logs/acid_cv_%A_%a.err
#SBATCH --array=0-9

# 5-fold CV for ours_baseline and ours_vlm (headline robustness claim).
# Indices 0-4 : ours_baseline on folds 0-4
# Indices 5-9 : ours_vlm on folds 0-4

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

I=$SLURM_ARRAY_TASK_ID
if (( I < 5 )); then
    MODEL=ours_baseline
    FOLD=$I
    CONFIG=configs/baseline.yaml
else
    MODEL=ours_vlm
    FOLD=$(( I - 5 ))
    CONFIG=configs/vlm.yaml
fi

CKPT_DIR=/work/nvme/bgte/tislam6/ACID_Journal/checkpoints/cv/${MODEL}_fold${FOLD}
EXP_NAME=${MODEL}_cv_fold${FOLD}

echo "=== ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $SLURM_NODELIST ==="
echo "Model: $MODEL  Fold: $FOLD"
echo "Ckpt:  $CKPT_DIR"

python -u scripts/train.py \
    --config "$CONFIG" \
    --model-name "$MODEL" \
    --seed 42 \
    --ckpt-dir "$CKPT_DIR" \
    --train-csv "cv_fold${FOLD}/train_annotations.csv" \
    --val-csv   "cv_fold${FOLD}/val_annotations.csv" \
    --test-csv  "cv_fold${FOLD}/test_annotations.csv"

python -u scripts/eval_metrics.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT_DIR/best_acc.pth" \
    --model-type zoo \
    --model-name "$MODEL" \
    --exp-name "$EXP_NAME" \
    --test-csv-override "/work/nvme/bgte/tislam6/ACID_Journal/dataset/extended_dataset_50x/cv_fold${FOLD}/test_annotations.csv" \
    || echo "Eval failed"
