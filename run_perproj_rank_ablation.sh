#!/bin/bash
# =============================================================================
# Per-projection rank sweep on SSv2 (Phase 2 distill + linear probe).
#
# Sweeps --lora_rank for the per-projection hypernetwork baseline
# (--lora_type per_projection). Mirrors run_fhlora_rank_ablation.sh; together they
# produce the two curves in Figure 3a / Table 4 of the paper.
#
# Usage:
#   ./run_perproj_rank_ablation.sh
#   RANKS="8 16 32 48 64 96" ./run_perproj_rank_ablation.sh
#
# Adjust TEACHER / STUDENT / PATH_PREFIX below for your machine.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEACHER="${TEACHER:-checkpoints/TimeSformer_divST_8_224_SSv2.pyth}"
STUDENT="${STUDENT:-output_ssv2/phase1/checkpoint_best.pth}"
PATH_PREFIX="${PATH_PREFIX:-/path/to/ssv2_frames}"
NGPU="${NGPU:-1}"

RANKS="${RANKS:-8 16 32 48 64 96}"

DISTILL_COMMON="--teacher_weights $TEACHER \
    --student_weights $STUDENT \
    --data_path data/downstream/ssv2 \
    --path_prefix $PATH_PREFIX \
    --dataset ssv2 \
    --epochs 15 --batch_size_per_gpu 32 --lr 5e-4 \
    --num_workers 16 --saveckp_freq 5 \
    --student_arch small \
    --lora_type per_projection \
    --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
    --opts DATA.PATH_PREFIX $PATH_PREFIX"

EVAL_COMMON="--data_path data/downstream/ssv2 \
    --path_prefix $PATH_PREFIX \
    --dataset ssv2 \
    --num_labels 174 \
    --epochs 20 --batch_size_per_gpu 64 --lr 1e-3 \
    --num_workers 16 \
    --freeze_backbone True \
    --student_arch small \
    --use_tdlora True \
    --lora_type per_projection \
    --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
    --opts DATA.PATH_PREFIX $PATH_PREFIX"

for RANK in $RANKS; do
    TAG="perproj_r${RANK}"
    PHASE2="output_ssv2/phase2_${TAG}"
    EVAL="output_ssv2/eval_${TAG}"

    echo "============================================"
    echo " Per-projection rank=$RANK — Phase 2 distillation"
    echo "============================================"

    torchrun --nproc_per_node=$NGPU distill_temporal_lora.py \
        --lora_rank "$RANK" \
        --output_dir "$PHASE2" \
        $DISTILL_COMMON

    echo "============================================"
    echo " Per-projection rank=$RANK — Linear probe"
    echo "============================================"

    torchrun --nproc_per_node=$NGPU eval_student.py \
        --student_weights "${PHASE2}/checkpoint_best.pth" \
        --lora_rank "$RANK" \
        --output_dir "$EVAL" \
        $EVAL_COMMON

    echo "=> probe log: ${EVAL}/log.txt"
done

echo ""
echo "Done. Compare best test_acc1 in each log under output_ssv2/eval_perproj_r*/"
