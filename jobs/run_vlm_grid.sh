#!/bin/bash
#SBATCH --job-name=acid_vlm_grid
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:30:00
#SBATCH --output=jobs/logs/acid_vlm_grid_%A_%a.out
#SBATCH --error=jobs/logs/acid_vlm_grid_%A_%a.err
#SBATCH --array=0-21

# VLM ablation grid (CLIP family only)
#
# Cells 0-14  : α sweep
#     α ∈ {0.00, 0.05, 0.10, 0.20, 0.50} × 3 seeds {42, 1337, 2024}
#     (α=0.00 duplicates ours_baseline but trains via the VLM path so the
#      hyperparams and checkpoint layout are consistent.)
# Cells 15-20 : CLIP encoder architecture sweep
#     ViT-B-32 × 3 seeds, ViT-B-16 × 3 seeds (ViT-L-14 too heavy, skip)
# Cell  21    : ViT-L-14 × 1 seed (single spot-check)
#
# Every run uses model_name=ours_vlm (= VLMGuidedDualGasNet).

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

ALPHAS=(0.00 0.05 0.10 0.20 0.50)
SEEDS=(42 1337 2024)

I=$SLURM_ARRAY_TASK_ID

if (( I < 15 )); then
    # α sweep (5 × 3 = 15)
    A_IDX=$(( I / 3 ))
    S_IDX=$(( I % 3 ))
    ALPHA=${ALPHAS[$A_IDX]}
    SEED=${SEEDS[$S_IDX]}
    CLIP=ViT-B-16
    NAME="alpha_${ALPHA}_seed${SEED}"
elif (( I < 18 )); then
    # ViT-B-32 × 3 seeds
    S_IDX=$(( I - 15 ))
    ALPHA=0.10
    SEED=${SEEDS[$S_IDX]}
    CLIP=ViT-B-32
    NAME="clip_vitb32_seed${SEED}"
elif (( I < 21 )); then
    # ViT-B-16 × 3 seeds (this is a re-run of α=0.10 grid, but saved under
    # the clip_vitb16 prefix so the aggregator can group by CLIP arch cleanly)
    S_IDX=$(( I - 18 ))
    ALPHA=0.10
    SEED=${SEEDS[$S_IDX]}
    CLIP=ViT-B-16
    NAME="clip_vitb16_seed${SEED}"
else
    # ViT-L-14 × 1 seed
    ALPHA=0.10
    SEED=42
    CLIP=ViT-L-14
    NAME="clip_vitl14_seed${SEED}"
fi

CKPT_DIR=/work/nvme/bgte/tislam6/ACID_Journal/checkpoints/vlm_grid/${NAME}

echo "=== ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $SLURM_NODELIST ==="
echo "Cell: $NAME (alpha=$ALPHA clip=$CLIP seed=$SEED)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -u scripts/train.py \
    --config configs/vlm.yaml \
    --model-name ours_vlm \
    --seed "$SEED" \
    --align-w "$ALPHA" \
    --clip-model "$CLIP" \
    --ckpt-dir "$CKPT_DIR"

python -u scripts/eval_metrics.py \
    --config configs/vlm.yaml \
    --checkpoint "$CKPT_DIR/best_acc.pth" \
    --model-type zoo \
    --model-name ours_vlm \
    --exp-name "$NAME" \
    || echo "Eval failed — check logs"
