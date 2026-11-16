#!/usr/bin/env bash
set -euo pipefail

cd /cmy/cmy/MeanFlowSE123/MeanFLowSE

# Pure CFM baseline — no MF branch
python train.py \
  --backbone ncsnpp_base \
  --ode flowmatching \
  --base_dir /cmy/cmy/enhance \
  --batch_size 8 \
  --num_workers 4 \
  --max_epochs 50 \
  --precision 32 \
  --gradient_clip_val 1.0 \
  --t_eps 0.03 \
  --T_rev 1.0 \
  --sigma_min 0.0 \
  --sigma_max 0.487 \
  --val_metrics_every_n_epochs 1 \
  --num_eval_files 50 \
  "$@"
