import torch
from torch import nn

import dnnlib
import legacy





def preprocess_function(input_pre: dict) -> dict:
    # preprocessing
    z_dim = 512

    z = torch.randn([1, z_dim]).cpu()  # latent codes
    c = None  # class labels (not used in this example)

    return {"z": z, "c": c}


def postprocess_function(image_torch: torch.tensor) -> torch.tensor:
    # postprocessing
    out = (image_torch.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    return out


def predict(inp: dict):
    # Preprocess
    input_dict = preprocess_function(inp)

    # Inference
    with torch.no_grad():
        model.eval()
        output = model.forward(input_dict)

    # Postprocess
    output_tensor = postprocess_function(output)


if __name__ == '__main__':
    network = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl"

    model = ModelWrapper(network_pkl=network)
    predict(inp={})
