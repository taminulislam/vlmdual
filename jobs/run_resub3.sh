#!/bin/bash
#SBATCH --job-name=resub3
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8,gpuA100x4
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/resub3_%A_%a.out
#SBATCH --error=jobs/logs/resub3_%A_%a.err
#SBATCH --array=0-7%2

# Final resub: 8 specific tasks that failed with CUDA-busy in resub2.
# Cap to 2 concurrent + ~30s random sleep to fully de-collide CUDA init.

set -e
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm
cd /work/nvme/bgte/tislam6/ACID_Journal
export HF_HOME=/work/nvme/bgte/tislam6/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

# Stagger: 30-60s random sleep
sleep $(( RANDOM % 30 + 30 ))

I=$SLURM_ARRAY_TASK_ID

# Manual dispatch table
case $I in
  0) MODEL=ours_vlm_v2;        FRAC=0.50; SEED=42;   CONFIG=configs/vlm_v2.yaml ;;
  1) MODEL=ours_vlm_v2;        FRAC=0.50; SEED=1337; CONFIG=configs/vlm_v2.yaml ;;
  2) MODEL=unet_r50;            FRAC=1.00; SEED=2024; CONFIG=configs/baselines/_base.yaml ;;
  3) MODEL=unet_convnextv2_b;   FRAC=1.00; SEED=1337; CONFIG=configs/baselines/_base.yaml ;;
  4) MODEL=unet_convnextv2_b;   FRAC=1.00; SEED=2024; CONFIG=configs/baselines/_base.yaml ;;
  5) MODEL=clip_fine_tuned;     FRAC=1.00; SEED=42;   CONFIG=configs/baselines/_base.yaml ;;
  6) MODEL=dinov2_linear_probe; FRAC=1.00; SEED=1337; CONFIG=configs/baselines/_base.yaml ;;
  7) MODEL=dinov2_linear_probe; FRAC=1.00; SEED=2024; CONFIG=configs/baselines/_base.yaml ;;
esac

if [[ "$FRAC" == "1.00" ]]; then
    # Plain training (no label fraction)
    if [[ "$MODEL" == clip* ]] || [[ "$MODEL" == dinov2* ]]; then
        CKPT=checkpoints/vlm_baselines/${MODEL}_seed${SEED}
        EXP=${MODEL}_seed${SEED}
    else
        FRAC_TAG=$(echo "$FRAC" | sed 's/\./p/')
        CKPT=checkpoints/label_eff/${MODEL}_frac${FRAC_TAG}_seed${SEED}
        EXP=label_eff_${MODEL}_frac${FRAC_TAG}_seed${SEED}
    fi
else
    FRAC_TAG=$(echo "$FRAC" | sed 's/\./p/')
    CKPT=checkpoints/label_eff/${MODEL}_frac${FRAC_TAG}_seed${SEED}
    EXP=label_eff_${MODEL}_frac${FRAC_TAG}_seed${SEED}
fi

echo "=== ${SLURM_JOB_ID}_${I}  $MODEL frac=$FRAC seed=$SEED ==="

# Build train args conditionally for label fraction
if [[ "$FRAC" == "1.00" ]] && ([[ "$MODEL" == clip* ]] || [[ "$MODEL" == dinov2* ]]); then
    python -u scripts/train.py \
        --config "$CONFIG" --model-name "$MODEL" --seed "$SEED" --ckpt-dir "$CKPT"
else
    python -u scripts/train.py \
        --config "$CONFIG" --model-name "$MODEL" --seed "$SEED" \
        --label-fraction "$FRAC" --ckpt-dir "$CKPT"
fi

python -u scripts/eval_metrics.py \
    --config "$CONFIG" --checkpoint "$CKPT/best_acc.pth" \
    --model-type zoo --model-name "$MODEL" --exp-name "$EXP" \
    || echo "Eval failed"
