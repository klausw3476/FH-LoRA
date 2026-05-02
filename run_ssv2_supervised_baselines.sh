#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

DATA_PATH="${DATA_PATH:-data/downstream/ssv2}"
PATH_PREFIX="${PATH_PREFIX:-/mnt/data/ssv2_frames}"
NGPU="${NGPU:-1}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-16}"

# Baseline A: ViT-Small full fine-tuning without distillation/adapters.
PLAIN_INIT="${PLAIN_INIT:-output_ssv2/init_small_scratch.pth}"
PLAIN_OUT="${PLAIN_OUT:-output_ssv2/eval_full_finetune_small_scratch}"
PLAIN_LR="${PLAIN_LR:-1e-4}"
PLAIN_BS="${PLAIN_BS:-32}"

# Baseline B: ViT-Small + FH-LoRA without distillation.
ST_RANK="${ST_RANK:-48}"
ST_HHD="${ST_HHD:-32}"
ST_INIT="${ST_INIT:-output_ssv2/init_fhlora_r${ST_RANK}_hhd${ST_HHD}_scratch.pth}"
ST_OUT="${ST_OUT:-output_ssv2/eval_fhlora_supervised_r${ST_RANK}_hhd${ST_HHD}}"
ST_LR="${ST_LR:-1e-4}"
ST_BS="${ST_BS:-32}"

python scripts/make_student_init_ckpt.py \
  --output "${PLAIN_INIT}" \
  --student_arch small \
  --use_tdlora false

python scripts/make_student_init_ckpt.py \
  --output "${ST_INIT}" \
  --student_arch small \
  --use_tdlora true \
  --lora_type fh_lora \
  --lora_rank "${ST_RANK}" \
  --hyper_hidden_dim "${ST_HHD}"

echo "============================================"
echo "Baseline A: ViT-Small full fine-tuning (no distillation, no adapters)"
echo "============================================"
torchrun --nproc_per_node="${NGPU}" eval_student.py \
  --student_weights "${PLAIN_INIT}" \
  --data_path "${DATA_PATH}" \
  --path_prefix "${PATH_PREFIX}" \
  --dataset ssv2 \
  --num_labels 174 \
  --epochs "${EPOCHS}" \
  --batch_size_per_gpu "${PLAIN_BS}" \
  --lr "${PLAIN_LR}" \
  --num_workers "${NUM_WORKERS}" \
  --freeze_backbone False \
  --student_arch small \
  --use_tdlora False \
  --output_dir "${PLAIN_OUT}" \
  --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
  --opts DATA.PATH_PREFIX "${PATH_PREFIX}"

echo "============================================"
echo "Baseline B: ViT-Small + FH-LoRA supervised only (no distillation)"
echo "============================================"
torchrun --nproc_per_node="${NGPU}" eval_student.py \
  --student_weights "${ST_INIT}" \
  --data_path "${DATA_PATH}" \
  --path_prefix "${PATH_PREFIX}" \
  --dataset ssv2 \
  --num_labels 174 \
  --epochs "${EPOCHS}" \
  --batch_size_per_gpu "${ST_BS}" \
  --lr "${ST_LR}" \
  --num_workers "${NUM_WORKERS}" \
  --freeze_backbone False \
  --student_arch small \
  --use_tdlora True \
  --lora_type fh_lora \
  --lora_rank "${ST_RANK}" \
  --hyper_hidden_dim "${ST_HHD}" \
  --output_dir "${ST_OUT}" \
  --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
  --opts DATA.PATH_PREFIX "${PATH_PREFIX}"

echo ""
echo "Done."
echo "  full fine-tune: ${PLAIN_OUT}/log.txt"
echo "  FH-LoRA no distill:  ${ST_OUT}/log.txt"
