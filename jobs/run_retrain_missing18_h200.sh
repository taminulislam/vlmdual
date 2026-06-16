#!/bin/bash
#SBATCH --job-name=acid_fill
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-17
#SBATCH --output=jobs/logs/acid_fill_%A_%a.out
#SBATCH --error=jobs/logs/acid_fill_%A_%a.err

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

echo "=== $SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID on $SLURM_NODELIST ==="
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

export HF_HOME=/work/nvme/bgte/tislam6/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

# 18 missing entries (empty checkpoint dirs — CUDA-busy failures)
TASKS=(
  "clip_linear_probe 42"
  "clip_linear_probe 1337"
  "clip_linear_probe 2024"
  "convnext_base_dual 42"
  "deeplabv3p_r50 1337"
  "pspnet_r50 2024"
  "segformer_b2 1337"
  "segformer_b2 2024"
  "swin_base_dual 42"
  "swin_base_dual 1337"
  "swin_base_dual 2024"
  "unet_hrnet_w48 42"
  "unet_hrnet_w48 2024"
  "unet_r50 42"
  "unet_r50 1337"
  "unet_swinv2_b 1337"
  "vit_b16_dual 42"
  "vit_b16_dual 2024"
)

read MODEL SEED <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
CKPT=checkpoints/zoo/${MODEL}_seed${SEED}

echo "=== Retrain + eval: $MODEL seed=$SEED ==="

python -u scripts/train.py \
    --config configs/baselines/_base.yaml \
    --model-name "$MODEL" --seed "$SEED" --ckpt-dir "$CKPT"

python -u scripts/eval_metrics.py \
    --config configs/baselines/_base.yaml \
    --checkpoint "$CKPT/best_acc.pth" \
    --model-type zoo --model-name "$MODEL" \
    --exp-name "${MODEL}_seed${SEED}" || echo "Eval failed"

echo "=== Done: $MODEL seed=$SEED ==="
