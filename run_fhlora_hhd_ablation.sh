#!/bin/bash
# =============================================================================
# FH-LoRA — hyper-trunk hidden-dim sweep on SSv2 (Phase 2 distill + probe).
#
# Fixes --lora_rank and varies --hyper_hidden_dim, the width of the shared
# trunk in FH-LoRA (--lora_type fh_lora).
# Complements run_fhlora_rank_ablation.sh.
#
# Usage:
#   ./run_fhlora_hhd_ablation.sh
#   HHD_LIST="8 16 32 64" LORA_RANK=8 ./run_fhlora_hhd_ablation.sh
#   EVAL_ONLY=1 ./run_fhlora_hhd_ablation.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEACHER="${TEACHER:-checkpoints/TimeSformer_divST_8_224_SSv2.pyth}"
STUDENT="${STUDENT:-output_ssv2/phase1/checkpoint_best.pth}"
PATH_PREFIX="${PATH_PREFIX:-/path/to/ssv2_frames}"
LORA_RANK="${LORA_RANK:-8}"
NGPU="${NGPU:-1}"

HHD_LIST="${HHD_LIST:-8 16 32}"

for HYPER_HHD in $HHD_LIST; do
    TAG="fhlora_r${LORA_RANK}_hhd${HYPER_HHD}"
    PHASE2="output_ssv2/phase2_${TAG}"
    EVAL="output_ssv2/eval_${TAG}"

    DISTILL_COMMON="--teacher_weights $TEACHER \
        --student_weights $STUDENT \
        --data_path data/downstream/ssv2 \
        --path_prefix $PATH_PREFIX \
        --dataset ssv2 \
        --epochs 15 --batch_size_per_gpu 32 --lr 5e-4 \
        --num_workers 16 --saveckp_freq 5 \
        --student_arch small \
        --lora_type fh_lora \
        --hyper_hidden_dim $HYPER_HHD \
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
        --lora_type fh_lora \
        --hyper_hidden_dim $HYPER_HHD \
        --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
        --opts DATA.PATH_PREFIX $PATH_PREFIX"

    echo "============================================"
    echo " FH-LoRA  rank=$LORA_RANK  hyper_hidden_dim=$HYPER_HHD"
    echo "============================================"

    if [ "${EVAL_ONLY:-0}" != "1" ]; then
        torchrun --nproc_per_node=$NGPU distill_temporal_lora.py \
            --lora_rank "$LORA_RANK" \
            --output_dir "$PHASE2" \
            $DISTILL_COMMON
    else
        echo "[skip distill] EVAL_ONLY=1"
    fi

    torchrun --nproc_per_node=$NGPU eval_student.py \
        --student_weights "${PHASE2}/checkpoint_best.pth" \
        --lora_rank "$LORA_RANK" \
        --output_dir "$EVAL" \
        $EVAL_COMMON

    echo "=> probe log: ${EVAL}/log.txt"
done

echo ""
echo "Done. Compare logs under output_ssv2/eval_fhlora_r${LORA_RANK}_hhd*/"
