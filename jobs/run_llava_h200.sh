#!/bin/bash
#SBATCH --job-name=acid_llava
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=jobs/logs/acid_llava_%j.out
#SBATCH --error=jobs/logs/acid_llava_%j.err

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

echo "=== $SLURM_JOB_ID on $SLURM_NODELIST ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# Use HF cache on scratch so we don't fill $HOME
export HF_HOME=/work/nvme/bgte/tislam6/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

# Diagnostic template over the full test set
python -u scripts/explain_test_set.py \
    --config configs/vlm.yaml \
    --checkpoint checkpoints/ours_vlm_seed42/best_acc.pth \
    --template diagnostic \
    --out results/explanations_diagnostic.json \
    --limit 0
