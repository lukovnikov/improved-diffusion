"""
Train a diffusion model on images.
"""

import argparse
import json

import torch

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser, create_model_and_distilled_diffusion, model_and_distilled_diffusion_defaults,
)
from improved_diffusion.train_util import TrainLoop, DistillTrainLoop
from image_train import save_args


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values (we can retain multiple EMAs)
        log_interval=20,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu=-2,
        trainiters=int(1e6),
        model_path="",
        savedir="",
    )
    defaults.update(model_and_distilled_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    print(json.dumps(args.__dict__, indent=4))
    save_args(args, "config_distill.json")

    dist_util.setup_dist()
    logger.configure()

    device = (torch.device(f"cuda:{args.gpu}") if args.gpu > -1 else torch.device("cpu")) if args.gpu > -2 else dist_util.dev()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_distilled_diffusion(
        **args_to_dict(args, model_and_distilled_diffusion_defaults().keys())
    )

    diffusion.initialize_jump_schedule(model, args.jumpsched)

    if len(args.model_path) > 0:
        logger.log(f"Loading model from {args.model_path}")
        statedict = dist_util.load_state_dict(args.model_path, map_location="cpu")
        if "jumpsched" not in statedict:
            statedict["jumpsched"] = model.jumpsched
            statedict["distillphase"] = model.distillphase
        model.load_state_dict(statedict)

    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    DistillTrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        device=device,
        trainiters=args.trainiters,
        savedir=args.savedir,
    ).run_loop()


if __name__ == "__main__":
    main()
