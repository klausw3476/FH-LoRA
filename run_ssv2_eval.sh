#!/bin/bash
set -e

cd "$(dirname "$0")"

COMMON_ARGS="--data_path data/downstream/ssv2 \
    --path_prefix /mnt/data/ssv2_frames \
    --dataset ssv2 \
    --num_labels 174 \
    --epochs 20 \
    --batch_size_per_gpu 64 \
    --lr 1e-3 \
    --num_workers 16 \
    --freeze_backbone True \
    --student_arch small \
    --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
    --opts DATA.PATH_PREFIX /mnt/data/ssv2_frames"

echo "============================================"
echo "Eval 1: With TD-LoRA"
echo "============================================"

torchrun --nproc_per_node=1 eval_student.py \
    --student_weights output_ssv2/phase2/checkpoint_best.pth \
    --use_tdlora True \
    --lora_type fh_lora \
    --lora_rank 8 \
    --hyper_hidden_dim 32 \
    --output_dir output_ssv2/eval_linear_probe \
    $COMMON_ARGS

echo "============================================"
echo "Eval 1 done. Starting Eval 2: Without TD-LoRA"
echo "============================================"

torchrun --nproc_per_node=1 eval_student.py \
    --student_weights output_ssv2/phase2/checkpoint_best.pth \
    --use_tdlora False \
    --output_dir output_ssv2/eval_linear_probe_no_tdlora \
    $COMMON_ARGS

echo "============================================"
echo "Both evaluations complete!"
echo "With TD-LoRA:    output_ssv2/eval_linear_probe/log.txt"
echo "Without TD-LoRA: output_ssv2/eval_linear_probe_no_tdlora/log.txt"
echo "============================================"
