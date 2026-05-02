#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "============================================"
echo "Phase 1: Spatial Distillation on SSv2"
echo "============================================"

torchrun --nproc_per_node=1 distill_spatial.py \
    --teacher_weights checkpoints/TimeSformer_divST_8_224_SSv2.pyth \
    --data_path data/downstream/ssv2 \
    --path_prefix /mnt/data/ssv2_frames \
    --dataset ssv2 \
    --output_dir output_ssv2/phase1 \
    --epochs 15 --batch_size_per_gpu 32 --lr 2e-4 \
    --num_workers 16 --saveckp_freq 5 \
    --student_arch small \
    --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
    --opts DATA.PATH_PREFIX /mnt/data/ssv2_frames

echo "============================================"
echo "Phase 1 complete. Starting Phase 2..."
echo "============================================"

torchrun --nproc_per_node=1 distill_temporal_lora.py \
    --teacher_weights checkpoints/TimeSformer_divST_8_224_SSv2.pyth \
    --student_weights output_ssv2/phase1/checkpoint_best.pth \
    --data_path data/downstream/ssv2 \
    --path_prefix /mnt/data/ssv2_frames \
    --dataset ssv2 \
    --output_dir output_ssv2/phase2 \
    --epochs 15 --batch_size_per_gpu 32 --lr 5e-4 \
    --num_workers 16 --saveckp_freq 5 \
    --student_arch small \
    --lora_type fh_lora \
    --lora_rank 8 \
    --hyper_hidden_dim 32 \
    --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
    --opts DATA.PATH_PREFIX /mnt/data/ssv2_frames

echo "============================================"
echo "Both phases complete!"
echo "Phase 1 output: output_ssv2/phase1/"
echo "Phase 2 output: output_ssv2/phase2/"
echo "============================================"
