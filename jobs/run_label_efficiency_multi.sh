#!/bin/bash
#SBATCH --job-name=label_eff
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/label_eff_%A_%a.out
#SBATCH --error=jobs/logs/label_eff_%A_%a.err
#SBATCH --array=0-35

# Label-efficiency sweep: 4 fractions × 3 models × 3 seeds = 36 runs.
#
# Index i:
#   frac_idx  = i / 9
#   model_idx = (i % 9) / 3
#   seed_idx  =  i % 3
#
# Val/test always use 100% labels so cross-fraction comparison is fair.

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

FRACS=(0.10 0.25 0.50 1.00)
MODELS=(ours_vlm_v2 unet_r50 unet_convnextv2_b)
SEEDS=(42 1337 2024)

I=$SLURM_ARRAY_TASK_ID
FRAC_IDX=$(( I / 9 ))
MODEL_IDX=$(( (I % 9) / 3 ))
SEED_IDX=$(( I % 3 ))

FRAC=${FRACS[$FRAC_IDX]}
MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# ours_vlm_v2 uses configs/vlm_v2.yaml; baselines use configs/baselines/_base.yaml
if [[ "$MODEL" == "ours_vlm_v2" ]]; then
    CONFIG=configs/vlm_v2.yaml
else
    CONFIG=configs/baselines/_base.yaml
fi

FRAC_TAG=$(echo "$FRAC" | sed 's/\./p/')
CKPT=checkpoints/label_eff/${MODEL}_frac${FRAC_TAG}_seed${SEED}
EXP=label_eff_${MODEL}_frac${FRAC_TAG}_seed${SEED}

echo "=== ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $SLURM_NODELIST ==="
echo "Model: $MODEL  frac=$FRAC  seed=$SEED"

python -u scripts/train.py \
    --config "$CONFIG" \
    --model-name "$MODEL" \
    --seed "$SEED" \
    --label-fraction "$FRAC" \
    --ckpt-dir "$CKPT"

python -u scripts/eval_metrics.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT/best_acc.pth" \
    --model-type zoo \
    --model-name "$MODEL" \
    --exp-name "$EXP" \
    || echo "Eval failed — continuing"
