# MeanFlow-SE Training Pipeline

## Overview

MeanFlow-SE is a speech enhancement system based on **Flow Matching** with an optional **Mean Flow (MFSE)** training branch. The model learns a velocity field that transforms noisy speech spectrograms into clean ones via an ODE-based generative process, and can optionally learn an *average* velocity field enabling single-step (1-NFE) inference.

---

## Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION                             │
│                                                                     │
│  Waveforms (clean/*.wav, noisy/*.wav)                               │
│       │                                                             │
│       ▼                                                             │
│  STFT  ──►  Spectral Transform (abs^e · exp(j·angle) · factor)     │
│       │         e = spec_abs_exponent (0.5)                         │
│       │         factor = spec_factor (0.15)                         │
│       ▼                                                             │
│  Complex Spectrogram Pairs  (X_clean, Y_noisy)                     │
│       │     shape: [B, 2, F, T]  (real + imag as channels)         │
│       ▼                                                             │
│  DataLoader (SpecsDataModule)                                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FLOW MATCHING ODE                               │
│                                                                     │
│  Forward process (t ∈ [t_eps, T_rev]):                              │
│    μ_t = (1 - t) · x_clean + t · y_noisy                           │
│    σ_t = (1 - t) · σ_min   + t · σ_max                             │
│    x_t = μ_t + σ_t · z,     z ~ N(0, I)                            │
│                                                                     │
│  Conditional velocity field target:                                 │
│    v_target = dμ/dt + (dσ/dt) · z                                  │
│             = (y - x_clean) + (σ_max - σ_min) · z                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     BACKBONE NETWORK (NCSNpp)                       │
│                                                                     │
│  Input:  concat(x_t, y_noisy)  →  4 channels (re/im × 2)          │
│  Conditions:                                                        │
│    • t  → Gaussian Fourier Projection → MLP → time embedding       │
│    • d = t - r → separate Fourier Projection → MLP → Δ embedding   │
│    • Two embeddings are summed and injected into every ResNet block │
│                                                                     │
│  Architecture: U-Net (NCSN++) with                                  │
│    - BigGAN ResNet blocks                                           │
│    - Channel multipliers: (1,1,2,2,2,2,2) × nf=128                 │
│    - Attention at resolution 16                                     │
│    - Progressive input skip + output skip connections               │
│  Output:  2 channels (re, im) → negated as velocity prediction     │
│           v_pred = -DNN(concat(x_t, y), t, d)                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        LOSS COMPUTATION                             │
│                                                                     │
│  ┌──── CFM Loss (always active) ────────────────────────────────┐   │
│  │  Sample t ~ U[t_eps, T_rev], construct x_t                  │   │
│  │  v_pred = forward(x_t, t, y, r=t)   (r=t → instantaneous)  │   │
│  │  L_cfm = MSE(v_pred, v_target)                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──── MFSE Loss (when --use_mfse) ────────────────────────────┐   │
│  │  Sample r ∈ [t_eps, t] via curriculum:                       │   │
│  │    r = t - U^γ · (t - t_eps)                                 │   │
│  │    γ anneals: γ_start(8) → γ_end(1) over mf_delta_warmup    │   │
│  │    With prob mf_r_equals_t_prob(0.1), force r = t            │   │
│  │                                                              │   │
│  │  Compute JVP ≈ ∂v/∂t + v·∂v/∂x  (autograd or finite diff)  │   │
│  │  u_target = v_target - c · (t-r) · JVP   (c=0.5 default)   │   │
│  │  u_pred = forward(x_t, t, y, r)                             │   │
│  │  L_mf = MSE(u_pred, u_target)                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Total loss = (1 - w_mf) · L_cfm  +  w_mf · L_mf                  │
│    w_mf warms up: 0 → mf_weight_final(0.25) over mf_warmup_frac   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OPTIMIZATION                                 │
│                                                                     │
│  Optimizer: Adam (lr = 1e-4)                                        │
│  EMA: ExponentialMovingAverage (decay = 0.999)                      │
│    → Updated after every optimizer step                             │
│    → Used for validation and inference                              │
│  Gradient clipping: max norm = 1.0                                  │
│  Distributed: DDP (find_unused_parameters=False)                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        VALIDATION                                   │
│                                                                     │
│  Every val_metrics_every_n_epochs:                                  │
│    1. Switch to EMA weights                                         │
│    2. Run ODE solver on num_eval_files samples:                     │
│       - euler    (multi-step, instantaneous field)                  │
│       - euler_mf (multi-step or 1-step, average field)              │
│    3. Convert enhanced spectrogram → waveform via iSTFT             │
│    4. Compute metrics: PESQ, SI-SDR, ESTOI                         │
│    5. Checkpoint callbacks save top-K by PESQ and SI-SDR            │
│  Logging: TensorBoard                                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE                                    │
│                                                                     │
│  Load checkpoint → EMA weights                                      │
│  For each noisy waveform:                                           │
│    1. STFT + spectral transform → Y                                 │
│    2. Prior sample: x_T = y + σ_T · z                               │
│    3. Reverse ODE from t=T_rev to t=t_eps:                          │
│       ┌─ Euler (multi-step):  x ← x + v(x,t,y) · dt               │
│       └─ Euler-MF (1-step):   x ← x - Δ · u(x,t,y,r)             │
│    4. Inverse spectral transform + iSTFT → enhanced waveform       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Walkthrough

### 1. Data Loading (`SpecsDataModule` / `Specs`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_fft` | 510 | FFT size → 256 frequency bins |
| `hop_length` | 128 | STFT hop length |
| `num_frames` | 256 | Number of time frames per sample |
| `window` | hann | Window function |
| `spec_abs_exponent` | 0.5 | Power-law compression exponent |
| `spec_factor` | 0.15 | Scaling factor after compression |
| `normalize` | noisy | Normalize waveform by noisy signal max |
| `batch_size` | 8 | Batch size per GPU |

**Process:**
1. Load paired clean/noisy `.wav` files from `{base_dir}/{train,valid,test}/{clean,noisy}/`.
2. Crop or pad waveforms to `target_len = (num_frames - 1) × hop_length`.
3. Normalize by the noisy signal's peak amplitude.
4. Apply STFT → complex spectrogram.
5. Apply spectral transform: `abs(Z)^0.5 · exp(j·angle(Z)) · 0.15`.
6. Return `(X_clean, Y_noisy)` as complex tensors.

### 2. Flow Matching ODE (`FLOWMATCHING`)

The forward interpolation path from clean to noisy is defined as:

$$\mu_t = (1 - t) \cdot x_{\text{clean}} + t \cdot y_{\text{noisy}}$$

$$\sigma_t = (1 - t) \cdot \sigma_{\min} + t \cdot \sigma_{\max}$$

$$x_t = \mu_t + \sigma_t \cdot z, \quad z \sim \mathcal{N}(0, I)$$

The conditional velocity field target (what the network learns to predict):

$$v_{\text{target}} = \frac{d\mu_t}{dt} + \frac{d\sigma_t}{dt} \cdot z = (y - x_{\text{clean}}) + (\sigma_{\max} - \sigma_{\min}) \cdot z$$

| Parameter | Default |
|-----------|---------|
| `sigma_min` | 0.0 |
| `sigma_max` | 0.487 |
| `t_eps` | 0.03 |
| `T_rev` | 1.0 |

### 3. Backbone Network (`NCSNpp`)

A U-Net architecture adapted from NCSN++ (Score-Based Generative Modeling):

- **Input**: `concat(x_t, y_noisy)` → 4 channels (real/imag of both)
- **Time conditioning**: Gaussian Fourier Projection of `t` → 2-layer MLP → embedding
- **Delta conditioning** (MFSE): Separate Fourier Projection of `d = t - r` → 2-layer MLP → embedding (summed with time embedding)
- **Encoder**: 7 resolution levels, channel mults `(1,1,2,2,2,2,2) × 128`, 2 ResNet blocks per level, BigGAN-style with FIR up/downsampling
- **Attention**: At resolution 16
- **Decoder**: Progressive output-skip connections
- **Output**: 2-channel (real, imag), negated: `v = -DNN(input, t, d)`

### 4. Training Step (`_step`)

```
For each batch (x_clean, y_noisy):
    1. Sample t ~ U[t_eps, T_rev]
    2. Compute x_t from flow ODE marginal: x_t = μ_t + σ_t · z
    3. Compute conditional VF target: v_target = der_mean + der_std · z
    4. Predict: v_pred = model(x_t, t, y, r=t)
    5. L_cfm = MSE(v_pred, v_target)

    If use_mfse:
        6.  Compute w_mf (warm-up schedule: 0 → 0.25)
        7.  Sample r given t (curriculum annealing)
        8.  Compute JVP via autograd or finite differences
        9.  u_target = v_target - 0.5 · (t-r) · JVP       [detached]
        10. u_pred = model(x_t, t, y, r)
        11. L_mf = MSE(u_pred, u_target)
        12. L_total = (1 - w_mf) · L_cfm + w_mf · L_mf
    Else:
        L_total = L_cfm
```

### 5. MFSE Curriculum Schedules

The Mean Flow branch uses several annealing schedules over training epochs:

| Schedule | Start → End | Warmup fraction | Purpose |
|----------|-------------|----------------|---------|
| `w_mf` (MF loss weight) | 0 → 0.25 | 50% of max epochs | Gradually introduce MF loss |
| `γ` (delta sampling exponent) | 8.0 → 1.0 | 70% of max epochs | Start with r≈t (small intervals), anneal to uniform |
| `r = t` probability | 10% (constant) | — | Stabilize by forcing degenerate r=t samples |

### 6. JVP Computation

The Jacobian-Vector Product `JVP ≈ ∂v/∂t + v · ∂v/∂x` is computed via:

- **`autograd`**: `torch.autograd.functional.jvp` (exact, higher memory)
- **`fd`**: Central finite differences `(f(x+ε, t+ε) − f(x−ε, t−ε)) / 2ε` (approximate, lower memory, supports batch chunking)
- **`auto`**: Try autograd first, fallback to fd on failure

JVP output is L2-clipped per sample (default clip = 5.0) and detached (no second-order gradients).

### 7. Optimization

- **Optimizer**: Adam, lr = 1e-4
- **EMA**: Decay 0.999, updated after every optimizer step
- **Gradient clipping**: Global norm ≤ 1.0
- **Distributed training**: PyTorch DDP via `torchrun`

### 8. Validation & Checkpointing

- Every `val_metrics_every_n_epochs` epochs, run `evaluate_model()` on rank 0:
  - Use EMA weights
  - Run reverse ODE (Euler or Euler-MF, N=5 steps) on validation samples
  - Compute PESQ (wideband), SI-SDR, ESTOI
  - Broadcast results to all ranks
- **Checkpoints saved by**: Last epoch, top-20 by PESQ, top-20 by SI-SDR
- **Logger**: TensorBoard

### 9. Inference (`evaluate.py`)

```
Load checkpoint → EMA weights
For each noisy file:
    1. y_wav → normalize → STFT → spectral transform → Y (padded)
    2. Prior sample: x_T, z = ode.prior_sampling(Y)
       x_T = y + σ_T · z
    3. Reverse ODE solve from T_rev → t_eps:
       - Euler:    x ← x - dt · v(x, t, y)           [multi-step]
       - Euler-MF: x ← x - Δ · u(x, t, y, r=t-Δ)    [1-step or multi-step]
    4. Inverse spectral transform → iSTFT → enhanced waveform
    5. Save .wav
```

---

## Training Command Example

```bash
torchrun --standalone --nproc_per_node=4 \
  train.py \
  --backbone ncsnpp \
  --ode flowmatching \
  --base_dir /path/to/dataset \
  --batch_size 2 \
  --num_workers 8 \
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
  --mf_jvp_impl fd \
  --mf_jvp_chunk 1 \
  --val_metrics_every_n_epochs 1 \
  --default_root_dir lightning_logs
```

---

## Key File Map

| File | Role |
|------|------|
| `train.py` | Entry point: argument parsing, model/trainer construction, `trainer.fit()` |
| `flowmse/model.py` | `VFModel` (LightningModule): forward pass, loss, training/validation steps, MFSE logic |
| `flowmse/odes.py` | `FLOWMATCHING` ODE: marginal distribution, velocity field targets |
| `flowmse/data_module.py` | `SpecsDataModule` / `Specs`: data loading, STFT, spectral transforms |
| `flowmse/backbones/ncsnpp.py` | `NCSNpp`: U-Net backbone with dual time conditioning (t, d) |
| `flowmse/backbones/shared.py` | `BackboneRegistry`, shared embedding layers |
| `flowmse/sampling/__init__.py` | `get_white_box_solver`: constructs reverse ODE solver |
| `flowmse/sampling/odesolvers.py` | `EulerODEsolver`, `EulerMFODESolver` |
| `flowmse/drift_diffusion.py` | Drift/diffusion terms for SDE counterpart |
| `flowmse/util/inference.py` | `evaluate_model()`: validation-time metric computation |
| `evaluate.py` | Standalone inference script |
| `flowmse/scripts/train_vbd.sh` | Example multi-GPU training shell script |
| `run_inference.sh` | Example inference shell script (multi-step / 1-step modes) |
