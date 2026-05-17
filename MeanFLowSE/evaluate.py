# evaluate.py
import glob
from os.path import join
from argparse import ArgumentParser

import torch
from torchaudio import load
from tqdm import tqdm
from soundfile import write

from flowmse.model import VFModel
from flowmse.data_module import SpecsDataModule
from flowmse.util.other import pad_spec
from flowmse.sampling import get_white_box_solver
from utils import ensure_dir


def _auto_pick_solver(model, user_choice):
    if user_choice is not None:
        return user_choice
    use_mf = getattr(model, "use_mfse", False)
    return "euler_mf" if use_mf else "euler"


class _DiscriminativeWrapper:
    """
    Wraps a VFModel trained with --use_direct_denoising (discriminative=True, no time conditioning).
    
    At training time, forward() is always called with t=None.  During inference the ODE solver
    passes a non-None t, which would crash the NCSNpp backbone (no GaussianFourierProjection was built).

    This wrapper intercepts every VF_fn(x, t, y, r=None) call and replaces t with None so that the
    forward pass is identical to the training path.
    """
    def __init__(self, model):
        self._model = model
        self.use_direct_denoising = getattr(model, "use_direct_denoising", False)
        self.T_rev = model.T_rev

    def __call__(self, x, t, y, r=None):
        # Always pass t=None to match the discriminative training path.
        return self._model(x, None, y, r=None)

    def eval(self):
        self._model.eval()
        return self

    def to(self, device):
        self._model.to(device)
        return self


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--test_dir", type=str, default=None,
                        help="Directory containing the test data (must have test/{clean,noisy}).")
    parser.add_argument("--input_wav", type=str, default=None,
                        help="Path to a single noisy .wav file for inference.")
    parser.add_argument("--folder_destination", type=str, required=True,
                        help="Destination path of inference results. Absolute path is required.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.ckpt).")

    parser.add_argument("--odesolver_type", type=str, choices=("white",), default="white",
                        help="Sampler family. We use 'white' (Euler family).")

    parser.add_argument("--odesolver", type=str, choices=("euler", "euler_mf"), default=None,
                        help="Numerical integrator: 'euler' (FlowSE) or 'euler_mf' (MeanFlow displacement). "
                             "Default: auto pick based on checkpoint config.")
    parser.add_argument("--reverse_starting_point", type=float, default=1.0,
                        help="Starting point t_N in the reverse ODE (default: 1.0).")
    parser.add_argument("--last_eval_point", type=float, default=0.03,
                        help="Terminal time t_eps for numerical stability (default: 0.03).")

    parser.add_argument("--N", type=int, default=5, help="Number of time steps (multi-step).")
    parser.add_argument("--one_step", action="store_true",
                        help="Use 1-NFE displacement (MeanFlow). If set, N is ignored and N=1.")

    parser.add_argument("--N_mid", type=int, default=0, help="It is not related to FlowSE")

    args = parser.parse_args()

    if args.input_wav is None and args.test_dir is None:
        parser.error("one of --input_wav or --test_dir is required")

    checkpoint_file = args.ckpt
    single_file_mode = args.input_wav is not None

    if not single_file_mode:
        clean_dir = join(args.test_dir, "test", "clean")
        noisy_dir = join(args.test_dir, "test", "noisy")

    sr = 16000
    odesolver_type = args.odesolver_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VFModel.load_from_checkpoint(
        checkpoint_file,
        data_module_cls=SpecsDataModule,
        map_location=device,
        weights_only=False
    )
    model = model.eval(no_ema=False).to(device)

    # --- Discriminative (stage1): direct spectrogram regression -------------
    # The patched model outputs clean spectrogram directly (no ODE, no velocity).
    # Just do one forward pass: clean_pred = model(noisy_spec, None, noisy_spec)
    use_direct = getattr(model, "use_direct_denoising", False)
    if use_direct:
        print("Stage-1 direct regression model — using one-shot forward pass (no ODE)")
        model_direct = model
    # -------------------------------------------------------------------------

    if single_file_mode:
        noisy_files = [args.input_wav]
    else:
        noisy_files = sorted(glob.glob(f"{noisy_dir}/*.wav"))

    target_dir = f"{args.folder_destination}/"
    ensure_dir(target_dir + "files/")

    solver_name = _auto_pick_solver(model, args.odesolver)

    N_eff = 1 if args.one_step else int(args.N)
    reverse_starting_point = float(args.reverse_starting_point)
    reverse_end_point = float(args.last_eval_point)

    model.T_rev = reverse_starting_point
    if hasattr(model, "ode"):
        model.ode.T_rev = reverse_starting_point

    if args.one_step and solver_name == "euler_mf" and reverse_end_point > 0:
        print(f"[Warn] 1‑NFE + euler_mf，当前 last_eval_point={reverse_end_point}，"
              f"若需严格一步从 1→0，可把 --last_eval_point 设为 0.0。")

    for _, noisy_file in tqdm(list(enumerate(noisy_files)), total=len(noisy_files)):
        filename = noisy_file.split('/')[-1]

        y, _ = load(noisy_file)
        if not single_file_mode:
            x, _ = load(join(clean_dir, filename))

        T_orig = y.size(1)
        norm_factor = y.abs().max().item()
        y = y / norm_factor

        Y = torch.unsqueeze(model._forward_transform(model._stft(y.to(device))), 0)
        Y = pad_spec(Y)

        if use_direct:
            # Direct regression: forward pass with t=None → clean spectrogram
            # Feed noisy spectrogram directly (no prior_sampling noise).
            with torch.no_grad():
                clean_spec = model_direct(Y, None, Y, r=None)  # [1,1,F,T_padded]
                sample = clean_spec.squeeze()  # [F, T_padded]
        else:
            if odesolver_type == "white":
                sampler = get_white_box_solver(
                    solver_name, model.ode, model, Y=Y, Y_prior=Y,
                    T_rev=reverse_starting_point, t_eps=reverse_end_point, N=N_eff
                )
            else:
                raise ValueError(f"{odesolver_type} is not a valid sampler type!")

            with torch.no_grad():
                sample, _ = sampler()
                sample = sample.squeeze()

        x_hat = model.to_audio(sample.unsqueeze(0), T_orig)  # [F,T] → [1,1,F,T]
        x_hat = x_hat * norm_factor

        write(target_dir + "files/" + filename, x_hat.squeeze().cpu().numpy(), sr)
