import torch

from models.modulators import FiLM
from typing import Dict, Any

from pipelines.diffuser import generate_diffuser


def generate_speedometer(parameters: Dict[str, Any]):
    activation_layer = torch.nn.LeakyReLU(negative_slope=0.005)
    dropout_layer = torch.nn.Dropout(p=0.0)

    temporal_conditioner = torch.nn.Sequential(
        torch.nn.Linear(2 * parameters["num_sinusoids"], 32),
        activation_layer,
        torch.nn.Linear(32, 32),
        activation_layer,
        dropout_layer,
        torch.nn.Linear(32, 18),
        activation_layer,
    ).to(parameters["device"])

    spatial_head = torch.nn.Sequential(
        torch.nn.Linear(9, 80),
        activation_layer,
        dropout_layer,
        torch.nn.Linear(80, 80),
        activation_layer,
        torch.nn.Linear(80, 3),
    ).to(parameters["device"])

    speedometer = FiLM(modulator=temporal_conditioner, head=spatial_head).to(
        parameters["device"]
    )

    checkpoint = torch.load(parameters["velocity_checkpoint"], weights_only=True)

    state_dict = checkpoint["model_state_dict"]
    module_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    speedometer.load_state_dict(module_state_dict)
    for param in speedometer.parameters():
        param.requires_grad = False
    speedometer.eval()
    speedometer.to(parameters["device"])

    return speedometer
