#!/bin/bash
#SBATCH --job-name=acid_retrain
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/acid_retrain_%j.out
#SBATCH --error=jobs/logs/acid_retrain_%j.err

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

echo "=== $SLURM_JOB_ID on $SLURM_NODELIST ==="
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

export HF_HOME=/work/nvme/bgte/tislam6/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

MODEL=resnet101_dual
SEED=2024
CKPT=checkpoints/zoo/${MODEL}_seed${SEED}

echo "=== Retraining: $MODEL seed=$SEED ==="
python -u scripts/train.py \
    --config configs/baselines/_base.yaml \
    --model-name "$MODEL" --seed "$SEED" --ckpt-dir "$CKPT"

python -u scripts/eval_metrics.py \
    --config configs/baselines/_base.yaml \
    --checkpoint "$CKPT/best_acc.pth" \
    --model-type zoo --model-name "$MODEL" \
    --exp-name "${MODEL}_seed${SEED}" || echo "Eval failed"

echo "=== Done ==="
