#!/usr/bin/env bash
set -euo pipefail

cd /cmy/cmy/MeanFlowSE123/MeanFLowSE

python train.py \
  --base_dir /cmy/cmy/enhance \
  --use_direct_denoising \
  --spec_abs_exponent 1.0 \
  --loss_lowfreq_weight 0.3 \
  --val_metrics_every_n_epochs 5 \
  --num_eval_files 50 \
  --max_epochs 100 \
  "$@"
