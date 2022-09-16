"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from PIL import Image

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict, create_model_and_distilled_diffusion, model_and_distilled_diffusion_defaults,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_distilled_diffusion(
        **args_to_dict(args, model_and_distilled_diffusion_defaults().keys())
    )

    diffusion.initialize_jump_schedule(model, args.jumpsched)

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    os.makedirs(args.save_path, exist_ok=True)

    device = dist_util.dev() if args.device == -2 else th.device(f"cuda:{args.device}")

    model.to(device)
    model.eval()

    print(model)

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"# Params: {count_parameters(model)}")

    logger.log("sampling...")
    # all_images = []
    # all_labels = []
    number_generated = 0
    batsize = args.batch_size
    while number_generated < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device
            )
            model_kwargs["y"] = classes

        # sample_fn = diffusion.distilled_ddim_sample_loop

        sample = diffusion.distilled_ddim_sample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        images = [sample.cpu().numpy() for sample in gathered_samples]
        cnt = 0
        for imagebatch in images:
            for i in range(len(imagebatch)):
                img = Image.fromarray(imagebatch[i])
                img.save(f"{args.save_path}/{cnt+number_generated}.tiff")
                cnt += 1
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            assert False
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        number_generated += cnt
        logger.log(f"created {number_generated} samples")

    """
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
    """

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        save_path="",
        device="-1",
    )
    defaults.update(model_and_distilled_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
