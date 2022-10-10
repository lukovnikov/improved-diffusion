"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import json
import os

import fire
import numpy as np
import torch as th
import torch.distributed as dist

from PIL import Image

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES, create_model, create_gaussian_diffusion, modelargs, diffusionargs, sampleargs,
)


def save_args(args, filename="config.json"):
    # save args in args.savedir
    savedir = args["savedir"]
    if savedir is not None and savedir != "":
        os.makedirs(savedir, exist_ok=True)
        with open(os.path.join(savedir, filename), "w") as f:
            json.dump(args, f, indent=4)
        logger.log(f"config saved")


# To use DDIM, must use --use_ddim, and ALSO use --timestep_respacing ddim1024 if number of steps is 1024
def main(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        clip_denoised=False,
        num_samples=50000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        save_path="",
        device="-1",
    ):
    args = locals().copy()
    argnames = set(args.keys())
    assert len(argnames & set(modelargs)) == len(modelargs)
    assert len(argnames & set(diffusionargs)) == len(diffusionargs)
    assert len(argnames & set(sampleargs)) == len(sampleargs)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )

    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )

    os.makedirs(save_path, exist_ok=True)


    device = dist_util.dev() if device == -2 else th.device(f"cuda:{device}")

    model.to(device)
    model.eval()

    # print(model)

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"# Params: {count_parameters(model)//1e6:.2f}M")

    logger.log("sampling...")
    # all_images = []
    # all_labels = []
    number_generated = 0
    batsize = batch_size
    while number_generated < num_samples:
        model_kwargs = {}
        if class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (batch_size, 3, image_size, image_size),
            clip_denoised=clip_denoised,
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
                img.save(f"{save_path}/{cnt+number_generated}.tiff")
                cnt += 1
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if class_cond:
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


def testfiremain(
        a=1,
        b="str",
        c=True,
        d=False
):
    print(f"Defaults: 1, 'str', True, False")
    print(f"{a}, {b}, {c}, {d}")


if __name__ == "__main__":
    # fire.Fire(testfiremain)
    fire.Fire(main)
