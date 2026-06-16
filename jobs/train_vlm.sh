#!/bin/bash
#SBATCH --job-name=acid_vlm
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=jobs/logs/acid_vlm_%j.out
#SBATCH --error=jobs/logs/acid_vlm_%j.err

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

echo "=== Job Info ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "================"

python -u scripts/train_vlm.py --config configs/vlm.yaml
