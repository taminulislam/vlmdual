#!/bin/bash
#SBATCH --job-name=r2q2_audit
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/r2q2_audit_%A_%a.out
#SBATCH --error=jobs/logs/r2q2_audit_%A_%a.err
#SBATCH --array=0-8

# Reviewer R2 Q2 — CH4-gate shortcut audit. Three ablations × three seeds.
# All runs use the headline alpha_text_1p0 configuration (alpha_dist=0.10,
# alpha_text=1.00), the same one reported as ours_vlm_v2 in tab:vlm_v2_ablation,
# so any drop is attributable strictly to the gate/CH4 manipulation.
#
# Index = mode_idx * 3 + seed_idx
#   mode_idx 0 = gate_random  per-sample Bernoulli(0.5) gate
#   mode_idx 1 = gate_zero    gate forced to 0; CH4 frame still real
#   mode_idx 2 = co2_only     gate=0 AND CH4 frame zeroed (true single-gas)
# Seeds: {42, 1337, 2024}.

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

MODES=(gate_random gate_zero co2_only)
SEEDS=(42 1337 2024)

I=$SLURM_ARRAY_TASK_ID
MODE_IDX=$(( I / 3 ))
SEED_IDX=$(( I % 3 ))

MODE=${MODES[$MODE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

CKPT=checkpoints/vlm_v2_r2q2/${MODE}_seed${SEED}
EXP=vlm_v2_r2q2_${MODE}_seed${SEED}

echo "=== ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $SLURM_NODELIST ==="
echo "Mode: $MODE  seed=$SEED"
echo "Ckpt: $CKPT"

python -u scripts/train.py \
    --config configs/vlm_v2.yaml \
    --model-name ours_vlm_v2 \
    --seed "$SEED" \
    --align-dist-w 0.10 \
    --align-text-w 1.00 \
    --ch4-mode "$MODE" \
    --ckpt-dir "$CKPT"

python -u scripts/eval_metrics.py \
    --config configs/vlm_v2.yaml \
    --checkpoint "$CKPT/best_acc.pth" \
    --model-type zoo \
    --model-name ours_vlm_v2 \
    --exp-name "$EXP" \
    --ch4-mode "$MODE" \
    || echo "Eval failed — continuing"
