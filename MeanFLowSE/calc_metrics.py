#!/cmy/cmy/miniconda3/envs/meanse/bin/python
"""
calc_metrics.py — 计算语音增强评测指标，对标论文 Table 1

依赖环境: conda activate meanse
  已有: pesq, pystoi, torch, torchmetrics, librosa, soundfile

用法：
  conda run -n meanse python calc_metrics.py \
    --clean_dir    /cmy/cmy/enhance/VoiceBank-DEMAND-16k/test/clean \
    --enhanced_dir /path/to/enhanced/files

指标输出：
  PESQ (wb)   — 感知语音质量 ↑
  ESTOI       — 可懂度 ↑
  SI-SDR (dB) — 信号保真度 ↑
  NISQA MOS   — 非侵入式感知质量（对应论文 DNSMOS OVRL）↑
    Noisiness   — 对应论文 DNSMOS BAK（背景噪声抑制）↑
"""

import os
import glob
import argparse
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

from pesq import pesq as pesq_fn
from pystoi import stoi as stoi_fn
from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment as NISQA
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SISDR

SR = 16000


def load_wav(path: str) -> np.ndarray:
    data, sr = sf.read(path)
    if sr != SR:
        raise ValueError(f"Expected {SR} Hz, got {sr} Hz: {path}")
    if data.ndim > 1:
        data = data[:, 0]
    return data.astype(np.float32)


def trim_to_same(a: np.ndarray, b: np.ndarray):
    n = min(len(a), len(b))
    return a[:n], b[:n]


def si_sdr_np(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-8)
    proj = alpha * ref
    noise = est - proj
    return float(10 * np.log10((np.dot(proj, proj) + 1e-8) / (np.dot(noise, noise) + 1e-8)))


def evaluate(clean_dir: str, enhanced_dir: str):
    enhanced_files = sorted(glob.glob(os.path.join(enhanced_dir, "*.wav")))
    if not enhanced_files:
        raise FileNotFoundError(f"No .wav files found in: {enhanced_dir}")

    print(f"\n评测文件数: {len(enhanced_files)}")
    print(f"Clean dir:    {clean_dir}")
    print(f"Enhanced dir: {enhanced_dir}\n")

    nisqa_metric = NISQA(fs=SR)

    pesq_scores, estoi_scores, sisdr_scores = [], [], []
    nisqa_mos, nisqa_noisiness, nisqa_coloration = [], [], []
    skipped = 0

    for enh_path in tqdm(enhanced_files, desc="Computing metrics"):
        fname = os.path.basename(enh_path)
        clean_path = os.path.join(clean_dir, fname)

        if not os.path.exists(clean_path):
            skipped += 1
            continue

        try:
            enh = load_wav(enh_path)
            ref = load_wav(clean_path)
            ref, enh = trim_to_same(ref, enh)
        except Exception as e:
            print(f"[skip] {fname}: {e}")
            skipped += 1
            continue

        # PESQ
        try:
            pesq_scores.append(pesq_fn(SR, ref, enh, "wb"))
        except Exception as e:
            pass

        # ESTOI
        try:
            estoi_scores.append(stoi_fn(ref, enh, SR, extended=True))
        except Exception:
            pass

        # SI-SDR
        sisdr_scores.append(si_sdr_np(ref, enh))

        # NISQA (non-intrusive, no reference needed)
        try:
            scores = nisqa_metric(torch.tensor(enh))
            # scores: [MOS, noisiness, discontinuity, coloration, loudness]
            nisqa_mos.append(scores[0].item())
            nisqa_noisiness.append(scores[1].item())
            nisqa_coloration.append(scores[3].item())
        except Exception:
            pass

    if skipped:
        print(f"\n[warn] {skipped} files skipped")

    # ── 打印结果 ──────────────────────────────────────────────────────────────
    def avg(lst):
        return np.mean(lst) if lst else float("nan")

    print("\n" + "=" * 55)
    print(f"{'Metric':<22} {'Ours':>8}   {'Paper (1-NFE)':>13}")
    print("-" * 55)
    print(f"{'PESQ (wb) ↑':<22} {avg(pesq_scores):>8.4f}   {'2.942':>13}")
    print(f"{'ESTOI ↑':<22} {avg(estoi_scores):>8.4f}   {'0.881':>13}")
    print(f"{'SI-SDR (dB) ↑':<22} {avg(sisdr_scores):>8.3f}   {'19.975':>13}")
    if nisqa_mos:
        print("-" * 55)
        print(f"{'NISQA MOS ↑':<22} {avg(nisqa_mos):>8.4f}   {'(DNSMOS OVRL=3.207)':>13}")
        print(f"{'NISQA Noisiness ↑':<22} {avg(nisqa_noisiness):>8.4f}   {'(DNSMOS BAK=4.073)':>13}")
        print(f"{'NISQA Coloration ↑':<22} {avg(nisqa_coloration):>8.4f}   {'(DNSMOS SIG=3.471)':>13}")
    print("=" * 55)
    print(f"\n注: NISQA 与 DNSMOS P.835 是不同的非侵入式 MOS 模型，")
    print(f"    数值范围相近但不直接可比。SpkSim 需要 speechbrain（未安装）。")

    return {
        "pesq": avg(pesq_scores),
        "estoi": avg(estoi_scores),
        "si_sdr": avg(sisdr_scores),
        "nisqa_mos": avg(nisqa_mos),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir",    required=True, help="Clean reference wav directory")
    parser.add_argument("--enhanced_dir", required=True, help="Enhanced wav directory")
    args = parser.parse_args()
    evaluate(args.clean_dir, args.enhanced_dir)
