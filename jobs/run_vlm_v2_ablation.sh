#!/bin/bash
#SBATCH --job-name=vlm_v2_abl
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/vlm_v2_abl_%A_%a.out
#SBATCH --error=jobs/logs/vlm_v2_abl_%A_%a.err
#SBATCH --array=0-23

# VLM v2 component ablation × 3 seeds = 24 runs.
#
# Cells (index = cell_idx * 3 + seed_idx):
#   0  visual_c5           (α_dist=0.1 c5 only, α_text=0)
#   1  visual_c4_c5        (α_dist=0.1 multi-scale, α_text=0)
#   2  text_only           (α_dist=0,   α_text=0.5)
#   3  visual_c5_text      (α_dist=0.1 c5 only, α_text=0.5)
#   4  full                (α_dist=0.1 multi-scale, α_text=0.5)
#   5  alpha_text_0p1      (full, α_text=0.1)
#   6  alpha_text_1p0      (full, α_text=1.0)
#   7  alpha_dist_0p5      (full, α_dist=0.5)
#
# Seeds: {42, 1337, 2024}

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

CELLS=(
    "visual_c5        0.10 0.00 true"
    "visual_c4_c5    0.10 0.00 true"
    "text_only       0.00 0.50 true"
    "visual_c5_text  0.10 0.50 false"
    "full            0.10 0.50 true"
    "alpha_text_0p1  0.10 0.10 true"
    "alpha_text_1p0  0.10 1.00 true"
    "alpha_dist_0p5  0.50 0.50 true"
)
SEEDS=(42 1337 2024)

I=$SLURM_ARRAY_TASK_ID
CELL_IDX=$(( I / 3 ))
SEED_IDX=$(( I % 3 ))

read -r NAME A_DIST A_TEXT MULTI <<< "${CELLS[$CELL_IDX]}"
SEED=${SEEDS[$SEED_IDX]}

CKPT=checkpoints/vlm_v2/${NAME}_seed${SEED}
EXP=vlm_v2_${NAME}_seed${SEED}

echo "=== ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $SLURM_NODELIST ==="
echo "Cell: $NAME  α_dist=$A_DIST  α_text=$A_TEXT  multiscale=$MULTI  seed=$SEED"

# use_multiscale_dist is read from config; we override via env var translated to YAML key
# simpler: patch via CLI for config knobs that support it, and inline for the rest.
python -u scripts/train.py \
    --config configs/vlm_v2.yaml \
    --model-name ours_vlm_v2 \
    --seed "$SEED" \
    --align-dist-w "$A_DIST" \
    --align-text-w "$A_TEXT" \
    --ckpt-dir "$CKPT"

python -u scripts/eval_metrics.py \
    --config configs/vlm_v2.yaml \
    --checkpoint "$CKPT/best_acc.pth" \
    --model-type zoo \
    --model-name ours_vlm_v2 \
    --exp-name "$EXP" \
    || echo "Eval failed — continuing"
