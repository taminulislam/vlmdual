#!/bin/bash
#SBATCH --job-name=vlm_base
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/vlm_base_%A_%a.out
#SBATCH --error=jobs/logs/vlm_base_%A_%a.err
#SBATCH --array=0-8%3

# Extra VLM baselines × 3 seeds = 9 runs:
#   idx 0-2  clip_zero_shot      (no training, but still runs pipeline)
#   idx 3-5  clip_fine_tuned
#   idx 6-8  dinov2_linear_probe
# (CoOp deferred — attn_mask shape bug in my context-splicing; fix later)

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

# HF cache: reuse existing pre-downloaded weights
export HF_HOME=/work/nvme/bgte/tislam6/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

MODELS=(clip_zero_shot clip_fine_tuned dinov2_linear_probe)
SEEDS=(42 1337 2024)

I=$SLURM_ARRAY_TASK_ID
MODEL_IDX=$(( I / 3 ))
SEED_IDX=$(( I % 3 ))
MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

CKPT=checkpoints/vlm_baselines/${MODEL}_seed${SEED}
EXP=${MODEL}_seed${SEED}

echo "=== ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $SLURM_NODELIST ==="
echo "Model: $MODEL  seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
sleep $(( RANDOM % 10 + 5 ))

python -u scripts/train.py \
    --config configs/baselines/_base.yaml \
    --model-name "$MODEL" \
    --seed "$SEED" \
    --ckpt-dir "$CKPT"

python -u scripts/eval_metrics.py \
    --config configs/baselines/_base.yaml \
    --checkpoint "$CKPT/best_acc.pth" \
    --model-type zoo \
    --model-name "$MODEL" \
    --exp-name "$EXP" \
    || echo "Eval failed — continuing"
