from typing import Dict
import legacy
import dnnlib
import torch 
import torch.nn as nn

class ModelWrapper(nn.Module):
    def __init__(self, network_pkl):
        super().__init__()
        # network parameters
        self.network_pkl = network_pkl

        # model loading
        with dnnlib.util.open_url(self.network_pkl) as f:
            self.model = legacy.load_network_pkl(f)['G_ema'].to("cpu")
        self.add_module("model", self.model)

    def forward(self, x: dict, input_parameters: dict) -> torch.tensor:
        z = x.get("z")
        c = x.get("c")

        out = self.model(z, c)

        return out

def get_model(weights_path: str = None, map_location="cpu",
              model_initialization_parameters: Dict = None) -> torch.nn.Module:
              return ModelWrapper(network_pkl=weights_path)
