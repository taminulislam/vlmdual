#!/bin/bash
#SBATCH --job-name=acid_eval2
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/acid_eval2_%j.out
#SBATCH --error=jobs/logs/acid_eval2_%j.err

set -e

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate acidvlm

cd /work/nvme/bgte/tislam6/ACID_Journal

echo "=== $SLURM_JOB_ID on $SLURM_NODELIST ==="
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

export HF_HOME=/work/nvme/bgte/tislam6/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

eval_zoo() {
    local MODEL=$1 SEED=$2
    local CKPT=checkpoints/zoo/${MODEL}_seed${SEED}
    echo "=== Zoo eval: $MODEL seed=$SEED ==="
    python -u scripts/eval_metrics.py \
        --config configs/baselines/_base.yaml \
        --checkpoint "$CKPT/best_acc.pth" \
        --model-type zoo --model-name "$MODEL" \
        --exp-name "${MODEL}_seed${SEED}" || echo "Eval failed: ${MODEL}_seed${SEED}"
}

# 18 missing results — all checkpoints confirmed present
eval_zoo convnext_base_dual  42
eval_zoo vit_b16_dual        42
eval_zoo swin_base_dual      42
eval_zoo unet_r50            42
eval_zoo clip_linear_probe   42
eval_zoo unet_hrnet_w48      42

eval_zoo swin_base_dual      1337
eval_zoo unet_r50            1337
eval_zoo deeplabv3p_r50      1337
eval_zoo segformer_b2        1337
eval_zoo clip_linear_probe   1337
eval_zoo unet_swinv2_b       1337

eval_zoo vit_b16_dual        2024
eval_zoo swin_base_dual      2024
eval_zoo pspnet_r50          2024
eval_zoo segformer_b2        2024
eval_zoo clip_linear_probe   2024
eval_zoo unet_hrnet_w48      2024

echo "=== All missing evals done ==="
