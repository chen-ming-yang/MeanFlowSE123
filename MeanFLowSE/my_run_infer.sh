#!/usr/bin/env bash
set -euo pipefail

cd /cmy/cmy/MeanFlowSE123/MeanFLowSE

TEST_DATA_DIR=/cmy/cmy/enhance \
OUTPUT_ROOT=/tmp/meanflow_infer \
CKPT_INPUT=/cmy/cmy/MeanFlowSE123/MeanFLowSE/lightning_logs/dataset_enhance_20260425021023/version_0/checkpoints \
MODE=multistep_mf \
STEPS=5 \
bash run_inference.sh
