import torch
from torchaudio import load
import torch.nn.functional as F

from .other import si_sdr, pad_spec
from ..sampling import get_white_box_solver
# Settings
sr = 16000

N = 5

def evaluate_model(model, num_eval_files, inference_N=N):
    from pesq import pesq
    from pystoi import stoi

    device = next(model.parameters()).device

    T_rev = model.T_rev
    model.ode.T_rev = T_rev
    t_eps = model.t_eps
    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files

    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.long)
    clean_files = [clean_files[i.item()] for i in indices]
    noisy_files = [noisy_files[i.item()] for i in indices]

    inference_N = inference_N
    _pesq = 0
    _si_sdr = 0
    _estoi = 0
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        x, _ = load(clean_file)
        y, _ = load(noisy_file)
        T_orig = x.size(1)

        norm_factor = y.abs().max()
        y = y / norm_factor

        Y = torch.unsqueeze(model._forward_transform(model._stft(y.to(device))), 0)
        Y = pad_spec(Y)

        y = y * norm_factor

        if getattr(model, "use_direct_denoising", False):
            # Stage-1 / direct-denoising mode: the network was trained ONLY at t = T_rev
            # to predict v* = (x_T - x_clean)/T_rev (one-step Euler from T -> 0).
            # Running a multi-step ODE solver here would query the net at unseen t values.
            # So we evaluate the same one-step path used during training.
            with torch.no_grad():
                x_T, _ = model.ode.prior_sampling(Y.shape, Y.to(device))
                x_T = x_T.to(Y.device)
                # Match FlowAVSE: stage1 discriminative path does not pass t
                v_pred = model(x_T, None, Y.to(device), r=None)
                sample = x_T - T_rev * v_pred
        else:
            use_mf = getattr(model, "use_mf_sampler", False) or (inference_N == 1)
            solver_name = "euler_mf" if use_mf else "euler"

            sampler = get_white_box_solver(
                solver_name, model.ode, model, Y.to(device),
                T_rev=T_rev, t_eps=t_eps, N=inference_N
            )
            sample, _ = sampler()
        sample = sample.squeeze()

        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        x_hat = x_hat.squeeze().cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()

        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(sr, x, x_hat, 'wb')
        _estoi += stoi(x, x_hat, sr, extended=True)

    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files
