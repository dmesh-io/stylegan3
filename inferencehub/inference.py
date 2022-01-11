import torch
from torch import nn

import dnnlib
import legacy


class ModelWrapper(nn.Module):
    def __init__(self, network_pkl):
        super().__init__()
        # network parameters
        self.network_pkl = network_pkl

        # model loading
        with dnnlib.util.open_url(self.network_pkl) as f:
            self.model = legacy.load_network_pkl(f)['G_ema'].to("cpu")

    def forward(self, x: dict) -> torch.tensor:
        z = x.get("z")
        c = x.get("c")

        out = self.model(z, c)

        return out


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
