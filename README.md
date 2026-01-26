<div align="center"> <h1>MeanFlowSE · One-Step Generative Speech Enhancement</h1> <p>   <a href="https://arxiv.org/abs/2509.14858">     <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white">   </a>   <a href="https://huggingface.co/liduojia/MeanFlowSE">     <img alt="HF Model" src="https://img.shields.io/badge/Model-HuggingFace-yellow?logo=huggingface">   </a> </p>


</div>



**MeanFlowSE** is a conditional generative approach to speech enhancement. It learns **average velocities over short time spans** and performs enhancement with a **single backward-in-time displacement** (1-NFE), avoiding long ODE rollouts. The training objective is local (JVP-based) and **matches conditional flow matching on the diagonal (r = t)**—no teacher models, schedulers, or distillation required. In practice, 1-NFE inference makes real-time or near-real-time deployment straightforward on standard hardware.

![MeanFlowSE](MeanFlowSE.png)

------

## 🎧 Demos

- Online demo: **coming  soon**
- See **🧰 Pretrained Models** below for ready-to-use weights

------

## 🗂️ Table of Contents

- [✨ Highlights](#-highlights)
- [🔎 What’s Inside](#-whats-inside)
- [⚡ Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Inference](#inference)
- [🛠️ Configuration](#️-configuration)
- [🏗️ Repository Structure](#️-repository-structure)
- [🧰 Pretrained Models](#-pretrained-models)
- [📚 Built Upon & Related Work](#-built-upon--related-work)
- [🙏 Acknowledgments](#-acknowledgments)
- [📝 Citation](#-citation)

------

## ✨ Highlights

- **One-step enhancement (1-NFE):** A single **displacement** replaces long ODE trajectories—suitable for real-time scenarios on CPUs/GPUs.
- **No teachers, no distillation:** Local JVP-based training; exactly matches conditional flow matching when r=t.
- **Two samplers, one model:**
  - `euler_mf` → **average-field displacement** (one-step/few-step; recommended)
  - `euler` → **instantaneous-field Euler** (multi-step fallback for ablations)
- **End-to-end front-end:** Complex STFT pipeline; metrics include **PESQ / ESTOI / SI-SDR / DNSMOS / RTF**.

------

## 🔎 What’s Inside

- **Training:** Supervision from the **average velocity field** (1-step displacement sampler), with JVP for stability; when r=t the objective reduces to standard conditional flow matching.
- **Inference:** `euler_mf` for one-step displacement; `euler` for multi-step Euler along the instantaneous field.
- **Audio front-end:** Complex STFT with configurable transforms and normalization.
- **Metrics:** PESQ, ESTOI, SI-SDR, DNSMOS, and end-to-end **RTF** measurement.

------

## ⚡ Quick Start

### Installation

```
# Python 3.10 recommended
pip install -r requirements.txt
# Install a recent PyTorch + CUDA build compatible with your GPUs if you train multi-GPU
```

### Data Preparation

Expected layout (defaults assume 16 kHz, centered frames, Hann windows, complex STFT):

```
<BASE_DIR>/
  train/clean/*.wav
  train/noisy/*.wav
  valid/clean/*.wav
  valid/noisy/*.wav
  test/clean/*.wav
  test/noisy/*.wav
```


### Training

**Single machine, multi-GPU (DDP)**

```
# Edit DATA_DIR and GPU count inside the script if needed
bash scripts/train_vbd.sh
```

**Or run directly**

```
torchrun --standalone --nproc_per_node=4 train.py \
  --backbone ncsnpp \
  --ode flowmatching \
  --base_dir <BASE_DIR> \
  --batch_size 2 --num_workers 8 \
  --max_epochs 150 --precision 32 --gradient_clip_val 1.0 \
  --t_eps 0.03 --T_rev 1.0 \
  --sigma_min 0.0 --sigma_max 0.487 \
  --use_mfse \
  --mf_weight_final 0.25 --mf_warmup_frac 0.5 \
  --mf_delta_gamma_start 8.0 --mf_delta_gamma_end 1.0 \
  --mf_delta_warmup_frac 0.7 \
  --mf_r_equals_t_prob 0.1 \
  --mf_jvp_clip 5.0 --mf_jvp_eps 1e-3 \
  --mf_jvp_impl fd --mf_jvp_chunk 1 \
  --mf_skip_weight_thresh 0.05 \
  --val_metrics_every_n_epochs 1 \
  --default_root_dir lightning_logs
```

- Logs & checkpoints under `lightning_logs/<exp_name>/version_x/`.
- Heavy validation (PESQ/ESTOI/SI-SDR) runs **periodically on rank-0**; other ranks log placeholders so checkpoint monitors remain consistent.

### Inference

**Convenience script**

```
# MODE = multistep | multistep_mf | onestep
MODE=onestep STEPS=1 \
TEST_DATA_DIR=<BASE_DIR> \
CKPT_INPUT=path/to/best.ckpt \
bash run_inference.sh
```

**Or call the evaluator**

```
python evaluate.py \
  --test_dir <BASE_DIR> \
  --folder_destination /path/to/output \
  --ckpt path/to/best.ckpt \
  --odesolver euler_mf \
  --reverse_starting_point 1.0 \
  --last_eval_point 0.0 \
  --one_step
```

> `evaluate.py` writes **enhanced WAVs**.
>  If `--odesolver` is omitted, it **auto-selects** (`euler_mf` when MF-SE was used; otherwise `euler`).

------

## 🛠️ Configuration

Common flags to tweak:

- **Time & schedule** — `--T_rev` (reverse start, default 1.0), `--t_eps` (terminal time), `--sigma_min`, `--sigma_max`
- **MF-SE stability** — `--mf_jvp_impl {auto,fd,autograd}`, `--mf_jvp_chunk`, `--mf_jvp_clip`, `--mf_jvp_eps`; curriculum: `--mf_weight_final`, `--mf_warmup_frac`, `--mf_delta_*`, `--mf_r_equals_t_prob`
- **Validation cost** — `--val_metrics_every_n_epochs`, `--num_eval_files`
- **Backbone & front-end** — see `flowmse/backbones/` and `SpecsDataModule`

------

## 🏗️ Repository Structure

```
MeanFlowSE/
├── train.py                  # Lightning entry point
├── evaluate.py               # Enhancement script (saves WAV)
├── run_inference.sh          # One-step / few-step convenience runner
├── flowmse/
│   ├── model.py              # Losses, JVP, curriculum, logging
│   ├── odes.py               # Path definition & registry
│   ├── sampling/
│   │   ├── __init__.py
│   │   └── odesolvers.py     # Euler (instantaneous) & Euler-MF (displacement)
│   ├── backbones/
│   │   ├── ncsnpp.py         # U-Net with time/Δt embeddings
│   │   └── ...
│   ├── data_module.py        # STFT I/O pipeline
│   └── util/                 # metrics, registry, tensors, inference helpers
├── requirements.txt
└── scripts/
    └── train_vbd.sh
```

------

## 🧰 Pretrained Models

- **VoiceBank–DEMAND (16 kHz)** — weights on Google Drive:
   👉 [Download](https://drive.google.com/file/d/1QAxgd5BWrxiNi0q2qD3n1Xcv6bW0X86-/view?usp=sharing)

------

## 📚 Built Upon & Related Work

This repository builds upon and is inspired by the following excellent works (front-end design, training/evaluation infrastructure, etc.):

- **SGMSE** — https://github.com/sp-uhh/sgmse
- **SGMSE-CRP** — https://github.com/sp-uhh/sgmse_crp
- **SGMSE-BBED** — https://github.com/sp-uhh/sgmse-bbed
- **FLOWMSE (FlowSE)** — https://github.com/seongq/flowmse

------

## 🙏 Acknowledgments

We gratefully acknowledge **Prof. Xie Chen’s group (X-LANCE Lab, SJTU)** for valuable guidance and engineering tips during training.

------

## 📝 Citation

**Preprint**

```
@misc{li2025meanflowseonestepgenerativespeech,
  title         = {MeanFlowSE: one-step generative speech enhancement via conditional mean flow},
  author        = {Duojia Li and Shenghui Lu and Hongchen Pan and Zongyi Zhan and Qingyang Hong and Lin Li},
  year          = {2025},
  eprint        = {2509.14858},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2509.14858}
}
```

> **Status:** Our article has been accepted by ICASSP2026.
>
> **License:** This repository is released under the **MIT License**.

------

**Questions or issues?** Please open a GitHub issue or pull request. Contributions are welcome—from bug fixes to new backbones and front-ends.
