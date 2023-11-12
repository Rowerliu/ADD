"""
Generating the intermediate images between two domains
"""

import argparse
import os
import pathlib
import datetime
import torch as th
import numpy as np
import torch.distributed as dist

from scripts.common import read_model_and_diffusion
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import add_dict_to_argparser, model_and_diffusion_defaults
from guided_diffusion.image_datasets import load_data
from guided_diffusion.fourier_trans import get_adaptive_list


def main():
    args_source = create_argparser(type='source').parse_args()
    args_target = create_argparser(type='target').parse_args()
    args = args_source

    dist_util.setup_dist()

    folder = os.path.join(rf'\results')  # Assign the results path
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    time_start = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    logger.configure(dir=folder, log_suffix=time_start)
    logger.log("\nargs_source: ", args_source)
    logger.log("\nargs_target: ", args_target)
    logger.log("\nstarting to synthesis data.")

    time_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.log("time_start:", time_start)

    i = args.source
    j = args.target
    logger.log(f"reading models for synthetic data...")

    source_dir = r''
    source_model, diffusion = read_model_and_diffusion(args_source, source_dir, synthetic=False)

    target_dir = r''
    target_model, _ = read_model_and_diffusion(args_target, target_dir, synthetic=False)

    image_subfolder = os.path.join(folder, f"translation_{i}_{j}")
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    sources = []
    latents = []
    targets = []
    diffway_list_all = []

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True
    )
    lens = len(os.listdir(args.data_dir))
    amount = args.amount

    for k, (source, extra) in enumerate(data):

        if k < lens//args.batch_size:

            source = source.to(dist_util.dev())
            source_path_k = os.path.join(image_subfolder, 'source_np')
            pathlib.Path(source_path_k).mkdir(parents=True, exist_ok=True)
            source_path_k = os.path.join(source_path_k, f'source_{k:04d}.npy')
            np.save(source_path_k, source.cpu().numpy())
            sources.append(source.cpu().numpy())

            out_fourier_magnitude_list = diffusion.fourier_frequency_list_loop(
                source_model,
                source,
                clip_denoised=False,
                device=dist_util.dev(),
                progress=True,
            )

            diffway_list = get_adaptive_list(out_fourier_magnitude_list, amount)
            diffway_list_all.extend(diffway_list)
            diffway_list.insert(0, 0)  # initial timestep 0 for diffway_start

            logger.log(f"\ndevice: {dist_util.dev()}")
            logger.log(f"translating: {i}->{j}, batch: {k+1}, shape: {source.shape}, diffway_list: ", diffway_list)

            m = 0
            while m < amount:

                time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.log("\ntime_now:", time_now)
                logger.log(f"[batch / amount / total]: [{k+1} / {m+1} / {amount}]")

                diffway_start = diffway_list[m]
                diffway_end = diffway_list[m+1]

                noise = diffusion.ddim_reverse_sample_loop(
                    source_model,
                    source,
                    clip_denoised=False,
                    device=dist_util.dev(),
                    progress=True,
                    amount=amount,
                    diffway_start=diffway_start,
                    diffway_end=diffway_end,
                )
                source = noise

                logger.log(f"obtained latent representation for {source.shape[0]} samples...")
                logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

                target = diffusion.ddim_sample_loop(
                    target_model, (args.batch_size, 3, args.image_size, args.image_size),
                    noise=noise,
                    clip_denoised=False,
                    device=dist_util.dev(),
                    progress=True,
                    amount=amount,
                    diffway_end=diffway_end,
                )

                logger.log(f"finished translation {target.shape}")

                noise = ((noise + 1) * 127.5).clamp(0, 255).to(th.uint8)
                latent_path_k = os.path.join(image_subfolder, 'latent_np')
                pathlib.Path(latent_path_k).mkdir(parents=True, exist_ok=True)
                latent_path_k = os.path.join(latent_path_k, f'latent_{k:04d}_{m:04d}.npy')
                np.save(latent_path_k, noise.cpu().numpy())


                target = ((target + 1) * 127.5).clamp(0, 255).to(th.uint8)
                target_path_k = os.path.join(image_subfolder, 'target_np')
                pathlib.Path(target_path_k).mkdir(parents=True, exist_ok=True)
                target_path_k = os.path.join(target_path_k, f'target_{k:04d}_{m:04d}.npy')
                np.save(target_path_k, target.cpu().numpy())

                latents.append(noise.cpu().numpy())
                targets.append(target.cpu().numpy())

                m = m + 1

        else:
            break


    sources = np.concatenate(sources, axis=0)
    sources_path = os.path.join(image_subfolder, 'source.npy')
    np.save(sources_path, sources)

    latents = np.concatenate(latents, axis=0)
    grouped_latents = np.reshape(latents, (lens, amount, latents.shape[1], latents.shape[2], latents.shape[3]))
    latents_path = os.path.join(image_subfolder, 'latent.npy')
    np.save(latents_path, grouped_latents)

    targets = np.concatenate(targets, axis=0)
    grouped_targets = np.reshape(targets, (lens, amount, targets.shape[1], targets.shape[2], targets.shape[3]))
    targets_path = os.path.join(image_subfolder, 'target.npy')
    np.save(targets_path, grouped_targets)

    diffway_np = np.array(diffway_list_all).reshape(-1, amount)
    diffway_path = os.path.join(image_subfolder, 'diffway.npy')
    np.save(diffway_path, diffway_np)

    dist.barrier()
    logger.log(f"synthetic data translation complete: {i}->{j}\n\n")
    time_complete = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.log("time_complete:", time_complete)


def create_argparser(type=None):
    defaults = dict(
        data_dir=r"",  # Assign the source data path
        image_size=256,
        batch_size=1,
        amount=10,
    )
    defaults.update(model_and_diffusion_defaults())

    # Use the "Hybrid Attention" strategy (True: Global priority; False: Local priority)
    if type == 'source':
        new_defaults = dict(
            use_new_attention_order=True,
        )
    elif type == 'target':
        new_defaults = dict(
            use_new_attention_order=False,
        )
    else:
        new_defaults = defaults
    defaults.update(new_defaults)

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--source",
            type=int,
            default=0,
            help="Source data."
        )
    parser.add_argument(
            "--target",
            type=int,
            default=1,
            help="Target data."
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
