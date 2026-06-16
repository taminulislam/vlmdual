#!/bin/bash
#SBATCH --job-name=master_originals
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=jobs/logs/master_originals_%j.out
#SBATCH --error=jobs/logs/master_originals_%j.err

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

echo "=== $SLURM_JOB_ID on $SLURM_NODELIST ==="
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

export HF_HOME=/work/nvme/bgte/tislam6/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

# Run all 20 models.  Set MASTER_ONLY=name1,name2 to limit to a subset
# (e.g.  MASTER_ONLY=ours_vlm sbatch jobs/build_master_originals.sh).
python -u scripts/build_master_originals.py

echo "=== done ==="
ls -la master_originals/predictions/ | head -25
