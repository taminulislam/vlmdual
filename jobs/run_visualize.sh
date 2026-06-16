#!/bin/bash
#SBATCH --job-name=viz_preds
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=00:15:00
#SBATCH --output=jobs/logs/viz_%j.out
#SBATCH --error=jobs/logs/viz_%j.err

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

# Only verify the fix on ours_vlm first; re-enable other models after sanity check.
MODEL="${MODEL:-ours_vlm}"

# Move the stale figures for this specific model aside so the new output is clear.
if [ -d "results/figures/${MODEL}" ]; then
    mv "results/figures/${MODEL}" "results/figures/${MODEL}.stale_$(date +%Y%m%d_%H%M%S)" || true
fi
# And the vlm_explanations folder if regenerating ours_vlm
if [ "$MODEL" = "ours_vlm" ] && [ -d "results/figures/vlm_explanations" ]; then
    mv "results/figures/vlm_explanations" "results/figures/vlm_explanations.stale_$(date +%Y%m%d_%H%M%S)" || true
fi

echo "=== viz on $SLURM_NODELIST (model=$MODEL) ==="
nvidia-smi | head -5

python -u scripts/visualize_predictions.py --model-name "$MODEL" --n-correct 20 --n-incorrect 10

echo "=== done ==="
ls results/figures/${MODEL}/correct | head -5
ls results/figures/${MODEL}/incorrect | head -5
