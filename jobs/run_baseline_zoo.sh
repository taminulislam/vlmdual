#!/bin/bash
#SBATCH --job-name=acid_zoo
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/acid_zoo_%A_%a.out
#SBATCH --error=jobs/logs/acid_zoo_%A_%a.err
#SBATCH --array=0-35

# 12 baselines × 3 seeds = 36 array indices.
# Index i → (model i/3, seed [42,1337,2024][i%3])

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
)
SEEDS=(42 1337 2024)

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
MODEL_NAME=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

CKPT_DIR=/work/nvme/bgte/tislam6/ACID_Journal/checkpoints/zoo/${MODEL_NAME}_seed${SEED}

echo "=== $SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID on $SLURM_NODELIST ==="
echo "Model: $MODEL_NAME  Seed: $SEED"
echo "Ckpt : $CKPT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -u scripts/train.py \
    --config configs/baselines/_base.yaml \
    --model-name "$MODEL_NAME" \
    --seed "$SEED" \
    --ckpt-dir "$CKPT_DIR"

# Immediately evaluate on the test set
python -u scripts/eval_metrics.py \
    --config configs/baselines/_base.yaml \
    --checkpoint "$CKPT_DIR/best_acc.pth" \
    --model-type zoo \
    --model-name "$MODEL_NAME" \
    --exp-name "${MODEL_NAME}_seed${SEED}" \
    || echo "Eval failed — check logs"
