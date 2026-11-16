#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  Stage 2 from CFM baseline (ncsnpp_base + MFSE, 1-NFE)
#  ----------------------------------------------------------
#  Usage:
#    Start new Stage-2 from baseline:
#      bash train_stage2_from_baseline.sh /path/to/baseline.ckpt
#
#    Resume interrupted Stage-2:
#      bash train_stage2_from_baseline.sh --resume /path/to/stage2_last.ckpt
#
#    Environment variable form:
#      INIT_CKPT=/path/to/baseline.ckpt bash train_stage2_from_baseline.sh
# ============================================================

cd /cmy/cmy/MeanFlowSE123/MeanFLowSE

RESUME=false
CKPT=""

if [[ "${1:-}" == "--resume" ]]; then
  RESUME=true
  CKPT="${2:-}"
  shift 2 || true
else
  CKPT="${1:-${INIT_CKPT:-}}"
  if [[ $# -gt 0 ]]; then shift; fi
fi

if [[ -z "${CKPT}" ]]; then
  echo "ERROR: pass a checkpoint path. Examples:" >&2
  echo "  bash train_stage2_from_baseline.sh baseline.ckpt" >&2
  echo "  bash train_stage2_from_baseline.sh --resume stage2_last.ckpt" >&2
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi

if [[ "$RESUME" == true ]]; then
  echo "[stage2] Resuming from: ${CKPT}"
  CKPT_FLAG="--ckpt_path"
else
  echo "[stage2] Loading weights only from: ${CKPT}"
  CKPT_FLAG="--init_ckpt"
fi

python train.py \
  --backbone ncsnpp_base \
  --ode flowmatching \
  --base_dir /cmy/cmy/enhance \
  --batch_size 8 \
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
  --mf_jvp_chunk 0 \
  --mf_skip_weight_thresh 0.05 \
  --use_mf_sampler \
  --val_metrics_every_n_epochs 1 \
  --num_eval_files 50 \
  ${CKPT_FLAG} "${CKPT}" \
  "$@"
