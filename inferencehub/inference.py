import torch
from torch import nn

import dnnlib
import legacy


def preprocess_function(input_pre: dict,  model, input_parameters: dict) -> dict:
    # preprocessing
    z_dim = 512

    z = torch.randn([1, z_dim]).cpu()  # latent codes
    c = None  # class labels (not used in this example)

    return {"z": z, "c": c}


def postprocess_function(image_torch: torch.tensor) -> torch.tensor:
    # postprocessing
    out = (image_torch.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    return out
