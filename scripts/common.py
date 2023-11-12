import os
import numpy as np
import torch as th
import torch.distributed as dist
from skimage import io
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
)
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    args_to_dict,
)


def get_latest_model_path_in_directory(directory):
    """Returns the path to the latest model in the given directory."""
    model_files = [file for file in os.listdir(directory) if file.startswith("model")]
    model_numbers = sorted([int(file[5:-3]) for file in model_files])
    if len(model_numbers) == 0:
        return ""
    model_number = str(f"{model_numbers[-1]}").zfill(6)
    model_file = f"model{model_number}.pt"
    model_path = os.path.join(directory, model_file)
    return model_path, model_number


def read_model_and_diffusion(args, log_dir, synthetic=True):
    """Reads the latest model from the given directory."""
    model_path, _ = get_latest_model_path_in_directory(log_dir)
    logger.log(f"Model path: {model_path}")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"), strict=False)
    # model.load_state_dict(dist_util.load_state_dict(model_path))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    return model, diffusion

