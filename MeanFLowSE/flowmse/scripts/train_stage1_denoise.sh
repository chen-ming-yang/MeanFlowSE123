#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  Stage 1: Direct-Denoising Pretraining
#  ----------------------------------------------------------
#  Trains the network as a *plain one-step denoiser* (MSE
#  between one-step output and clean speech).  This avoids the
#  early-training NaNs commonly observed when training the
#  flow-matching / MFSE objective from scratch, and gives a
#  stable initialization for Stage 2.
#
#  After this stage finishes, pass the resulting checkpoint to
#  Stage 2 via:
#      bash train_stage2_generative.sh /path/to/stage1_last.ckpt
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/../../train.py"

DATA_DIR="${DATA_DIR:-/cmy/cmy/enhance}"   # Dataset root; accepts train/valid or training_set_10/valid_set_10 layouts.
NPROC=1
BATCH_PER_GPU=8
LOG_DIR="lightning_logs"

export CUDA_VISIBLE_DEVICES=0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

torchrun --standalone --nproc_per_node="${NPROC}" \
  "${TRAIN_PY}" \
  --backbone ncsnpp_small \
  --ode flowmatching \
  --base_dir "${DATA_DIR}" \
  --batch_size "${BATCH_PER_GPU}" \
  --num_workers 4 \
  --max_epochs 50 \
  --precision 32 \
  --gradient_clip_val 1.0 \
  --t_eps 0.03 \
  --T_rev 1.0 \
  --sigma_min 0.0 \
  --sigma_max 0.487 \
  --use_direct_denoising \
  --val_metrics_every_n_epochs 1 \
  --log_every_n_steps 10 \
  --default_root_dir "${LOG_DIR}"

# Notes:
# - --use_direct_denoising disables flow-matching / MFSE losses; the network is
#   trained purely as: x_hat = x_T - T*model(x_T, T, y, r=None);  loss = MSE(x_hat, x_clean).
# - Validation also runs the same one-step path (see flowmse/util/inference.py),
#   so PESQ/SI-SDR/ESTOI here reflect Stage-1 denoiser quality.
# - Suggested duration: 20-40 epochs is usually enough as a warm start.
