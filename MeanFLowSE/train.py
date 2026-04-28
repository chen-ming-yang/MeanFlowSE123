# train.py
import argparse
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from flowmse.backbones.shared import BackboneRegistry
from flowmse.data_module import SpecsDataModule
from flowmse.odes import ODERegistry
from flowmse.model import VFModel

from datetime import datetime
import pytz


def get_argparse_groups(parser, args):
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


if __name__ == '__main__':
    base_parser = ArgumentParser(add_help=False)
    base_parser.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
    base_parser.add_argument("--ode", type=str, choices=ODERegistry.get_all_names(), default="flowmatching")
    base_parser.add_argument("--no_wandb", action='store_true',
                             help="(保留选项供兼容) 不使用 W&B；我们统一使用 TensorBoard")
    temp_args, _ = base_parser.parse_known_args()

    parser = ArgumentParser()
    parser.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(),
                        default=temp_args.backbone)
    parser.add_argument("--ode", type=str, choices=ODERegistry.get_all_names(), default=temp_args.ode)
    parser.add_argument("--no_wandb", action='store_true', default=temp_args.no_wandb)

    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    ode_class = ODERegistry.get_by_name(temp_args.ode)

    # Trainer args (add_argparse_args removed in PL 2.x)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--default_root_dir", type=str, default="lightning_logs")
    parser.add_argument("--num_sanity_val_steps", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to a Lightning .ckpt file to resume training from (optimizer/epoch/step restored).")
    parser.add_argument("--init_ckpt", type=str, default=None,
                        help="Path to a .ckpt file from which only model weights (and EMA) are loaded "
                             "before training starts. Optimizer, epoch and step are NOT restored. "
                             "Use this to start stage-2 (generative fine-tuning) from a stage-1 "
                             "(direct-denoising) pretrained checkpoint. Mutually exclusive with --ckpt_path.")
    VFModel.add_argparse_args(parser.add_argument_group("VFModel", description=VFModel.__name__))
    ode_class.add_argparse_args(parser.add_argument_group("ODE", description=ode_class.__name__))
    backbone_cls.add_argparse_args(parser.add_argument_group("Backbone", description=backbone_cls.__name__))
    SpecsDataModule.add_argparse_args(parser.add_argument_group("DataModule", description=SpecsDataModule.__name__))

    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser, args)

    dataset = os.path.basename(os.path.normpath(arg_groups['DataModule'].base_dir))
    kst = pytz.timezone('Asia/Seoul')
    now_kst = datetime.now(kst)
    formatted_time_kst = now_kst.strftime("%Y%m%d%H%M%S")
    exp_name = f"dataset_{dataset}_{formatted_time_kst}"

    root_dir = getattr(args, "default_root_dir", None) or "lightning_logs"

    model = VFModel(
        backbone=args.backbone,
        ode=args.ode,
        data_module_cls=SpecsDataModule,
        **{
            **vars(arg_groups['VFModel']),
            **vars(arg_groups['ODE']),
            **vars(arg_groups['Backbone']),
            **vars(arg_groups['DataModule'])
        }
    )

    logger = TensorBoardLogger(save_dir=root_dir, name=exp_name)

    ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_last = ModelCheckpoint(
        dirpath=ckpt_dir, save_last=True, filename='{epoch}_last',
        save_on_train_epoch_end=False
    )
    checkpoint_pesq = ModelCheckpoint(
        dirpath=ckpt_dir, save_top_k=20, monitor="pesq", mode="max",
        filename='{epoch}_{pesq:.2f}', save_on_train_epoch_end=False
    )
    checkpoint_si_sdr = ModelCheckpoint(
        dirpath=ckpt_dir, save_top_k=20, monitor="si_sdr", mode="max",
        filename='{epoch}_{si_sdr:.2f}', save_on_train_epoch_end=False
    )
    callbacks = [checkpoint_last, checkpoint_pesq, checkpoint_si_sdr]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        accelerator='gpu',
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=logger,
        default_root_dir=root_dir,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
    )

    # 训练
    # weights_only=False: our own checkpoints contain pickled objects (e.g. SpecsDataModule);
    # PyTorch 2.6 made weights_only=True the default, which rejects them.
    if args.init_ckpt and args.ckpt_path:
        raise ValueError("--init_ckpt and --ckpt_path are mutually exclusive: "
                         "use --ckpt_path to fully resume, or --init_ckpt to load weights only.")

    if args.init_ckpt:
        import torch as _torch
        print(f"[train] Loading weights only from --init_ckpt: {args.init_ckpt}")
        _ckpt = _torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        _state = _ckpt.get("state_dict", _ckpt)
        missing, unexpected = model.load_state_dict(_state, strict=False)
        if missing:
            print(f"[train]   missing keys ({len(missing)}): "
                  f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[train]   unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
        # Also restore EMA shadow if present, so warm start is consistent.
        _ema = _ckpt.get("ema", None)
        if _ema is not None:
            try:
                model.ema.load_state_dict(_ema)
                print("[train]   EMA shadow weights restored from init_ckpt.")
            except Exception as _e:
                print(f"[train]   WARNING: failed to restore EMA from init_ckpt ({_e}); "
                      f"keeping EMA initialized from current model parameters.")
        del _ckpt, _state

    trainer.fit(model, ckpt_path=args.ckpt_path, weights_only=False)
