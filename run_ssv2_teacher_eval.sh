#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

TEACHER="${TEACHER:-checkpoints/TimeSformer_divST_8_224_SSv2.pyth}"
DATA_PATH="${DATA_PATH:-data/downstream/ssv2}"
PATH_PREFIX="${PATH_PREFIX:-/mnt/data/ssv2_frames}"
OUTDIR="${OUTDIR:-output_ssv2/eval_teacher_ssv2}"

python eval_teacher_ssv2.py \
  --teacher_weights "${TEACHER}" \
  --data_path "${DATA_PATH}" \
  --path_prefix "${PATH_PREFIX}" \
  --output_dir "${OUTDIR}" \
  --batch_size_per_gpu 8 \
  --num_workers 8 \
  --cfg models/configs/SSv2/TimeSformer_divST_8_224.yaml \
  --opts DATA.PATH_PREFIX "${PATH_PREFIX}"

echo "Done. Teacher results written to ${OUTDIR}/teacher_eval.json"
