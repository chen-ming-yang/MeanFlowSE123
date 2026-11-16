#!/usr/bin/env python3
"""
Batch denoise the 'clean' training audio using DeepFilterNet3.

Usage (in screen/tmux):
    conda activate meanse
    python denoise_clean.py

Features:
    - Resume support: skips files that already exist in output dir
    - Progress logging every 100 files
    - Processes files sequentially (GPU-bound, no benefit from multiprocessing)
"""

import os
import sys
import time
import glob
import argparse

import torch
import torchaudio
from df.enhance import init_df, enhance


def parse_args():
    parser = argparse.ArgumentParser(description="Batch denoise clean audio with DeepFilterNet3")
    parser.add_argument("--input_dir", type=str,
                        default="/cmy/cmy/enhance/training_set_10/clean",
                        help="Directory containing clean wav files")
    parser.add_argument("--output_dir", type=str,
                        default="/cmy/cmy/enhance/training_set_10/clean_denoised",
                        help="Output directory for denoised files")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log progress every N files")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all wav files
    wav_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
    total = len(wav_files)
    print(f"[denoise] Found {total} wav files in {args.input_dir}")
    print(f"[denoise] Output dir: {args.output_dir}")

    # Filter out already processed (resume support)
    to_process = []
    for f in wav_files:
        out_path = os.path.join(args.output_dir, os.path.basename(f))
        if not os.path.exists(out_path):
            to_process.append(f)

    skipped = total - len(to_process)
    if skipped > 0:
        print(f"[denoise] Skipping {skipped} already-processed files, {len(to_process)} remaining")

    if len(to_process) == 0:
        print("[denoise] All files already processed. Done.")
        return

    # Initialize DeepFilterNet3
    print("[denoise] Initializing DeepFilterNet3...")
    model, df_state, _ = init_df()
    print(f"[denoise] Model loaded. Sample rate: {df_state.sr()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[denoise] Using device: {device}")

    start_time = time.time()
    processed = 0
    errors = 0

    for i, filepath in enumerate(to_process):
        try:
            audio, sr = torchaudio.load(filepath)
            # DeepFilterNet expects [C, T] mono
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            # Resample to model's expected sr if needed
            model_sr = df_state.sr()
            if sr != model_sr:
                audio = torchaudio.functional.resample(audio, sr, model_sr)

            # Enhance: input/output shape [C, T]
            enhanced_tensor = enhance(model, df_state, audio)

            # Resample back to original sr if needed
            if sr != model_sr:
                enhanced_tensor = torchaudio.functional.resample(enhanced_tensor, model_sr, sr)

            # Save
            out_path = os.path.join(args.output_dir, os.path.basename(filepath))
            torchaudio.save(out_path, enhanced_tensor, sr)
            processed += 1

        except Exception as e:
            errors += 1
            print(f"[denoise] ERROR processing {os.path.basename(filepath)}: {e}", file=sys.stderr)

        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(to_process) - i - 1) / rate if rate > 0 else 0
            print(f"[denoise] Progress: {i+1}/{len(to_process)} "
                  f"({100*(i+1)/len(to_process):.1f}%) | "
                  f"{rate:.1f} files/s | "
                  f"ETA: {eta/3600:.1f}h | "
                  f"Errors: {errors}")

    elapsed = time.time() - start_time
    print(f"\n[denoise] Done! Processed: {processed}, Errors: {errors}, "
          f"Time: {elapsed/3600:.2f}h ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
