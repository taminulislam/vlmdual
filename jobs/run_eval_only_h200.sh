#!/bin/bash
#SBATCH --job-name=acid_eval
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=jobs/logs/acid_eval_%j.out
#SBATCH --error=jobs/logs/acid_eval_%j.err

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
    local EXP=${MODEL}_seed${SEED}
    echo "=== Zoo eval: $MODEL seed=$SEED ==="
    python -u scripts/eval_metrics.py \
        --config configs/baselines/_base.yaml \
        --checkpoint "$CKPT/best_acc.pth" \
        --model-type zoo --model-name "$MODEL" \
        --exp-name "$EXP" || echo "Eval failed: $EXP"
}

eval_ours() {
    local MODEL=$1 SEED=$2 CONFIG=$3
    local CKPT=checkpoints/${MODEL}_seed${SEED}
    echo "=== Ours eval: $MODEL seed=$SEED ==="
    python -u scripts/eval_metrics.py \
        --config "$CONFIG" --checkpoint "$CKPT/best_acc.pth" \
        --model-type zoo --model-name "$MODEL" \
        --exp-name "${MODEL}_seed${SEED}" || echo "Eval failed: ${MODEL}_seed${SEED}"
}

eval_vlm_alpha() {
    local ALPHA=$1 SEED=$2
    local NAME="alpha_${ALPHA}_seed${SEED}"
    local CKPT=checkpoints/vlm_grid/${NAME}
    echo "=== VLM alpha eval: alpha=$ALPHA seed=$SEED ==="
    python -u scripts/eval_metrics.py \
        --config configs/vlm.yaml --checkpoint "$CKPT/best_acc.pth" \
        --model-type zoo --model-name ours_vlm --exp-name "$NAME" || echo "Eval failed: $NAME"
}

eval_cv() {
    local MODEL=$1 FOLD=$2 CONFIG=$3
    local CKPT=checkpoints/cv/${MODEL}_fold${FOLD}
    local EXP=${MODEL}_cv_fold${FOLD}
    echo "=== CV eval: $MODEL fold=$FOLD ==="
    python -u scripts/eval_metrics.py \
        --config "$CONFIG" --checkpoint "$CKPT/best_acc.pth" \
        --model-type zoo --model-name "$MODEL" --exp-name "$EXP" \
        --test-csv-override "$(pwd)/dataset/extended_dataset_50x/cv_fold${FOLD}/test_annotations.csv" \
        || echo "Eval failed: $EXP"
}

# === Zoo: 15 models × 3 seeds ===
MODELS=(resnet50_single resnet50_dual resnet101_dual effnet_b3_dual
        convnext_base_dual vit_b16_dual swin_base_dual unet_r50
        deeplabv3p_r50 pspnet_r50 segformer_b2 clip_linear_probe
        unet_hrnet_w48 unet_swinv2_b unet_convnextv2_b)
SEEDS=(42 1337 2024)

for SEED in "${SEEDS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        CKPT=checkpoints/zoo/${MODEL}_seed${SEED}
        if [ -f "$CKPT/best_acc.pth" ]; then
            eval_zoo "$MODEL" "$SEED"
        else
            echo "SKIP (no checkpoint): ${MODEL}_seed${SEED}"
        fi
    done
done

# === Ours baseline + VLM ===
eval_ours ours_baseline 42  configs/baseline.yaml
eval_ours ours_baseline 1337 configs/baseline.yaml
eval_ours ours_vlm      42  configs/vlm.yaml
eval_ours ours_vlm      1337 configs/vlm.yaml

# === VLM alpha sweep ===
eval_vlm_alpha 0.00 42
eval_vlm_alpha 0.05 42
eval_vlm_alpha 0.10 42
eval_vlm_alpha 0.20 42
eval_vlm_alpha 0.50 42

# === CV folds ===
eval_cv ours_vlm 0 configs/vlm.yaml
eval_cv ours_vlm 1 configs/vlm.yaml
eval_cv ours_vlm 2 configs/vlm.yaml

echo "=== All evals done ==="
