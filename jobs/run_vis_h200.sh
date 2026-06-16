#!/bin/bash
#SBATCH --job-name=acid_vis
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/acid_vis_%j.out
#SBATCH --error=jobs/logs/acid_vis_%j.err

set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm
cd /work/nvme/bgte/tislam6/ACID_Journal
export HF_HOME=/work/nvme/bgte/tislam6/hf_cache

echo "=== $SLURM_JOB_ID on $SLURM_NODELIST ==="
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

# All 4 models: 50 correct + 10 incorrect each (per-model folders)
# VLM explanations folder: 100 correct + 30 incorrect (with LLaVA text)
python -u scripts/visualize_predictions.py \
    --all-models \
    --n-correct      50 \
    --n-incorrect    10 \
    --n-correct-vlm  100 \
    --n-incorrect-vlm 30

echo "=== Done ==="
