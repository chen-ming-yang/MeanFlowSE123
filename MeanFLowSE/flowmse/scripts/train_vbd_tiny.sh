#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Tiny-model training script  (NCSNpp-Tiny: nf=32, 4 levels)
# Roughly 30-50x fewer parameters than the full NCSNpp model,
# suitable for fast iteration / debugging on a single GPU.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/../../train.py"

DATA_DIR="/home/cmy/cmy/DNS-Challenge/datasets/mfse_dataset"   # VB-DMD root (must contain train/valid/test with clean/noisy)
NPROC=1                                           
BATCH_PER_GPU=8                                   
LOG_DIR="lightning_logs"                           

# Optional: resume from a Lightning .ckpt.
# Pass as 1st arg:  bash train_vbd_tiny.sh /path/to/last.ckpt
# Or via env var:   CKPT_PATH=/path/to/last.ckpt bash train_vbd_tiny.sh
CKPT_PATH="${1:-${CKPT_PATH:-}}"
RESUME_ARGS=()
if [[ -n "${CKPT_PATH}" ]]; then
  echo "[train_vbd_tiny] Resuming from checkpoint: ${CKPT_PATH}"
  RESUME_ARGS=(--ckpt_path "${CKPT_PATH}")
fi

export CUDA_VISIBLE_DEVICES=0

torchrun --standalone --nproc_per_node="${NPROC}" \
  "${TRAIN_PY}" \
  --backbone ncsnpp_small \
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
  "${RESUME_ARGS[@]}"
