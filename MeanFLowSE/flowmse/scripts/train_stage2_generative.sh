#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  Stage 2: Generative Fine-Tuning (Flow Matching + MFSE)
#  ----------------------------------------------------------
#  Loads the *weights only* from a Stage-1 (direct-denoising)
#  checkpoint and continues training with the flow-matching
#  CFM loss plus the optional MeanFlow-SE branch.
#
#  Usage:
#      bash train_stage2_generative.sh /path/to/stage1_last.ckpt
#  Or via env var:
#      INIT_CKPT=/path/to/stage1_last.ckpt bash train_stage2_generative.sh
#
#  IMPORTANT: We use --init_ckpt (weights-only), NOT --ckpt_path
#  (full resume).  This way the optimizer/epoch counters start
#  fresh for Stage 2 while the network parameters and EMA shadow
#  are inherited from Stage 1.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/../../train.py"

INIT_CKPT="${1:-${INIT_CKPT:-}}"
if [[ -z "${INIT_CKPT}" ]]; then
  echo "ERROR: please pass the Stage-1 checkpoint path as the first argument," >&2
  echo "       or set the INIT_CKPT environment variable." >&2
  exit 1
fi
echo "[stage2] Initializing from Stage-1 checkpoint: ${INIT_CKPT}"

DATA_DIR="/home/cmy/cmy/DNS-Challenge/datasets/mfse_dataset"
NPROC=1
BATCH_PER_GPU=8
LOG_DIR="lightning_logs"

export CUDA_VISIBLE_DEVICES=0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

torchrun --standalone --nproc_per_node="${NPROC}" \
  "${TRAIN_PY}" \
  --backbone ncsnpp \
  --ode flowmatching \
  --base_dir "${DATA_DIR}" \
  --batch_size "${BATCH_PER_GPU}" \
  --num_workers 4 \
  --max_epochs 150 \
  --precision 32 \
  --gradient_clip_val 1.0 \
  --t_eps 0.03 \
  --T_rev 1.0 \
  --sigma_min 0.0 \
  --sigma_max 0.487 \
  --use_mfse \
  --mf_weight_final 0.25 \
  --mf_warmup_frac 0.5 \
  --mf_delta_gamma_start 8.0 \
  --mf_delta_gamma_end 1.0 \
  --mf_delta_warmup_frac 0.7 \
  --mf_r_equals_t_prob 0.1 \
  --mf_jvp_clip 5.0 \
  --mf_jvp_eps 1e-3 \
  --mf_jvp_impl fd \
  --mf_jvp_chunk 1 \
  --mf_skip_weight_thresh 0.05 \
  --val_metrics_every_n_epochs 1 \
  --log_every_n_steps 10 \
  --default_root_dir "${LOG_DIR}" \
  --init_ckpt "${INIT_CKPT}"

# Notes:
# - DO NOT pass --use_direct_denoising here; that flag is exclusive to Stage 1.
# - Because --init_ckpt only restores weights (and EMA), the new run starts at
#   epoch 0 with a fresh optimizer.  All MFSE warm-up schedules therefore
#   apply with respect to the Stage-2 max_epochs, NOT cumulative epochs.
# - If you want to fully resume an interrupted Stage-2 run instead, swap
#   --init_ckpt for --ckpt_path.
