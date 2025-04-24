from typing import Any, Dict

import torch
from diffusionmodels import dataprocessing


def create_encoder(parameters: Dict[str, Any]) -> dataprocessing.sequential.Transform:
    """
    Create a sinusoidal time encoder based on training parameters.
    """
    waves = torch.arange(
        1,
        parameters["num_frequencies"] + 1,
        device=parameters["device"],
    ).unsqueeze(0)

    return dataprocessing.sequential.Pipeline(
        transforms=[
            lambda time: torch.cat(
                [
                    torch.sin(
                        2.0
                        * torch.pi
                        * time.unsqueeze(-1)
                        / parameters["period"]
                        * waves,
                    ),
                    torch.cos(
                        2.0
                        * torch.pi
                        * time.unsqueeze(-1)
                        / parameters["period"]
                        * waves,
                    ),
                ],
                dim=-1,
            ),
            lambda time: time.unsqueeze(1),
            lambda time: time.repeat(1, parameters["batch_size"], 1),
        ],
    )
