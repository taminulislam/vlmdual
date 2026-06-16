#!/bin/bash
#SBATCH --job-name=acid_eval
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/acid_eval_%j.out
#SBATCH --error=jobs/logs/acid_eval_%j.err

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

echo "=== $SLURM_JOB_ID on $SLURM_NODELIST ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "--- ours_baseline (best_acc.pth) ---"
python -u scripts/eval_metrics.py \
    --config configs/baseline.yaml \
    --checkpoint checkpoints/baseline/best_acc.pth \
    --model-type baseline \
    --exp-name ours_baseline_seed42

echo "--- ours_vlm (best_acc.pth) ---"
python -u scripts/eval_metrics.py \
    --config configs/vlm.yaml \
    --checkpoint checkpoints/vlm/best_acc.pth \
    --model-type vlm \
    --exp-name ours_vlm_seed42

echo "Eval complete."
