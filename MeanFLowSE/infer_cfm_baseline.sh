#!/usr/bin/env bash
set -euo pipefail

cd /cmy/cmy/MeanFlowSE123/MeanFLowSE

# ==============================================================================
# Pure CFM baseline 推理：多步 Euler ODE 求解
# ==============================================================================
INPUT_WAV="${INPUT_WAV:-/cmy/cmy/enhance/valid_set_10/noisy/book_11346_chp_0030_reader_00812_8_zQP6GRQPSIA_snr12_tl-25_fileid_89653.wav}"

FOLDER_DESTINATION="${FOLDER_DESTINATION:-/cmy/cmy/MeanFlowSE123/MeanFLowSE}"
CKPT="${CKPT:-/cmy/cmy/MeanFlowSE123/MeanFLowSE/lightning_logs/cfm_baseline/version_0/checkpoints/last.ckpt}"

N="${N:-30}"

python evaluate.py \
  --input_wav "$INPUT_WAV" \
  --folder_destination "$FOLDER_DESTINATION" \
  --ckpt "$CKPT" \
  --odesolver euler \
  --N "$N" \
  --reverse_starting_point 1.0 \
  --last_eval_point 0.03

echo "Done. Results in: $FOLDER_DESTINATION/files/"
